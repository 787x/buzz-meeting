import os
import sys
import threading
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from PyQt6.QtCore import QThread

from buzz.locale import _
from buzz.assets import APP_BASE_DIR
from buzz.model_loader import TranscriptionModel, ModelType, WhisperModelSize
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import TranscriptionOptions, Task
from buzz.settings.recording_transcriber_mode import RecordingTranscriberMode
from buzz.audio_capture.sounddevice_source import SoundDeviceAudioSource
from tests.audio_capture.fake_audio_source import FakeAudioSource
from tests.mock_sounddevice import MockSoundDevice
from tests.model_loader import get_model_path


class TestAmplitude:
    def test_symmetric_array(self):
        arr = np.array([1.0, -1.0, 2.0, -2.0])
        amplitude = RecordingTranscriber.amplitude(arr)
        # RMS: sqrt(mean([1, 1, 4, 4])) = sqrt(2.5) ≈ 1.5811
        assert abs(amplitude - np.sqrt(2.5)) < 1e-6

    def test_asymmetric_array(self):
        arr = np.array([1.0, 2.0, 3.0, -1.0])
        amplitude = RecordingTranscriber.amplitude(arr)
        # RMS: sqrt(mean([1, 4, 9, 1])) = sqrt(3.75) ≈ 1.9365
        assert abs(amplitude - np.sqrt(3.75)) < 1e-6

    def test_all_zeros(self):
        arr = np.array([0.0, 0.0, 0.0])
        amplitude = RecordingTranscriber.amplitude(arr)
        assert amplitude == 0.0

    def test_all_positive(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        amplitude = RecordingTranscriber.amplitude(arr)
        # RMS: sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.7386
        assert abs(amplitude - np.sqrt(7.5)) < 1e-6

    def test_all_negative(self):
        arr = np.array([-1.0, -2.0, -3.0, -4.0])
        amplitude = RecordingTranscriber.amplitude(arr)
        # RMS is symmetric: same as all_positive
        assert abs(amplitude - np.sqrt(7.5)) < 1e-6

    def test_returns_float(self):
        arr = np.array([0.5], dtype=np.float32)
        amplitude = RecordingTranscriber.amplitude(arr)
        assert isinstance(amplitude, float)


class TestGetDeviceSampleRate:
    def test_returns_default_16khz_when_supported(self):
        with patch("sounddevice.check_input_settings"):
            rate = RecordingTranscriber.get_device_sample_rate(None)
            assert rate == 16000

    def test_falls_back_to_device_default(self):
        from sounddevice import PortAudioError

        def raise_error(*args, **kwargs):
            raise PortAudioError("Device doesn't support 16000")

        device_info = {"default_samplerate": 44100}
        with patch("sounddevice.check_input_settings", side_effect=raise_error), \
             patch("sounddevice.query_devices", return_value=device_info):
            rate = RecordingTranscriber.get_device_sample_rate(0)
            assert rate == 44100

    def test_returns_default_when_query_fails(self):
        from sounddevice import PortAudioError

        def raise_error(*args, **kwargs):
            raise PortAudioError("Device doesn't support 16000")

        with patch("sounddevice.check_input_settings", side_effect=raise_error), \
             patch("sounddevice.query_devices", return_value=None):
            rate = RecordingTranscriber.get_device_sample_rate(0)
            assert rate == 16000


class TestRecordingTranscriber:

    def test_should_transcribe(self, qtbot):
        with (patch("sounddevice.check_input_settings")):
            thread = QThread()

            transcription_model = TranscriptionModel(
                model_type=ModelType.WHISPER_CPP, whisper_model_size=WhisperModelSize.TINY
            )

            model_path = get_model_path(transcription_model)

            model_exe_path = os.path.join(APP_BASE_DIR, "whisper_cpp", "whisper-server.exe")
            if sys.platform.startswith("win"):
                assert os.path.exists(model_exe_path), f"{model_exe_path} does not exist"

            transcriber = RecordingTranscriber(
                transcription_options=TranscriptionOptions(
                    model=transcription_model, language="fr", task=Task.TRANSCRIBE
                ),
                input_device_index=0,
                sample_rate=16_000,
                model_path=model_path,
                sounddevice=MockSoundDevice(),
            )
            transcriber.moveToThread(thread)

            thread.started.connect(transcriber.start)

            transcriptions = []

            def on_transcription(text):
                transcriptions.append(text)

            transcriber.transcription.connect(on_transcription)

            thread.start()
            try:
                qtbot.waitUntil(lambda: len(transcriptions) == 3, timeout=120_000)

                # any string in any transcription
                strings_to_check = [_("Starting Whisper.cpp..."), "Bienvenue dans Passe"]
                assert any(s in t for s in strings_to_check for t in transcriptions)
            finally:
                # Ensure cleanup runs even if waitUntil times out
                transcriber.stop_recording()
                time.sleep(10)

                thread.quit()
                thread.wait()

                # Ensure process is cleaned up
                if transcriber.process and transcriber.process.poll() is None:
                    transcriber.process.terminate()
                    try:
                        transcriber.process.wait(timeout=2)
                    except:
                        pass

                # Process pending events to ensure cleanup
                from PyQt6.QtCore import QCoreApplication
                QCoreApplication.processEvents()
                time.sleep(0.1)


class TestRecordingTranscriberInit:
    def test_init_default_mode(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            assert transcriber.transcription_options == transcription_options
            assert transcriber.input_device_index == 0
            assert transcriber.sample_rate == 16000
            assert isinstance(transcriber.audio_source, SoundDeviceAudioSource)
            assert transcriber.model_path == "/fake/path"
            assert transcriber.segmenter.max_utterance_seconds == 12.0
            assert transcriber.keep_sample_seconds == 0.15
            assert transcriber.max_pending_samples == 15 * 16000
            assert transcriber.is_running is False
            assert transcriber.openai_client is None

    def test_init_append_and_correct_mode(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"), \
             patch("buzz.transcriber.recording_transcriber.Settings") as mock_settings_class:
            # Mock settings to return APPEND_AND_CORRECT mode (index 2 in the enum)
            mock_settings_instance = MagicMock()
            mock_settings_class.return_value = mock_settings_instance
            # Return 2 for APPEND_AND_CORRECT mode (it's the third item in the enum)
            mock_settings_instance.value.return_value = 2

            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # APPEND_AND_CORRECT keeps its configured maximum update interval.
            assert (
                transcriber.segmenter.max_utterance_seconds
                == transcription_options.transcription_step
            )
            assert transcriber.keep_sample_seconds == 1.5

    def test_init_stores_silence_threshold(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
            silence_threshold=0.01,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            assert transcriber.transcription_options.silence_threshold == 0.01

    def test_init_uses_default_sample_rate_when_none(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=None,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # Should use default whisper sample rate
            assert transcriber.sample_rate == 16000


class TestAudioSourceCallback:
    def test_on_audio_adds_to_segmenter_buffer(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # Create test audio data
            in_data = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)

            initial_size = transcriber.segmenter.buffered_sample_count
            transcriber.on_audio(in_data.reshape(-1))

            assert transcriber.segmenter.buffered_sample_count == initial_size + 4

    def test_on_audio_emits_amplitude_changed(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # Mock the amplitude_changed signal
            amplitude_values = []
            transcriber.amplitude_changed.connect(lambda amp: amplitude_values.append(amp))

            # Create test audio data
            in_data = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
            transcriber.on_audio(in_data.reshape(-1))

            # Should have emitted one amplitude value
            assert len(amplitude_values) == 1
            assert amplitude_values[0] > 0

    def test_on_audio_drops_completed_utterance_when_pending_queue_full(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            transcriber.pending_sample_count = transcriber.max_pending_samples
            transcriber.segmenter = MagicMock()
            transcriber.segmenter.push.return_value = [
                np.ones(100, dtype=np.float32),
            ]
            transcriber.segmenter.buffered_sample_count = 0

            transcriber.on_audio(np.ones(100, dtype=np.float32))

            assert list(transcriber.pending_utterances) == []
            assert transcriber.pending_sample_count == transcriber.max_pending_samples

    def test_queue_owns_samples_after_borrowed_callback_returns(self):
        audio_source = FakeAudioSource(sample_rate=16_000)
        sounddevice_module = MagicMock()
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )
        transcriber = RecordingTranscriber(
            transcription_options=transcription_options,
            input_device_index=0,
            sample_rate=16_000,
            model_path="/fake/path",
            sounddevice=sounddevice_module,
            audio_source=audio_source,
        )
        audio_source.start(transcriber.on_audio)
        samples = np.full(3_200, 0.1, dtype=np.float32)

        audio_source.deliver(samples)
        samples[:] = 0.9
        utterances = transcriber.segmenter.flush()

        assert len(utterances) == 1
        np.testing.assert_array_equal(utterances[0], np.full(3_200, 0.1, dtype=np.float32))
        sounddevice_module.InputStream.assert_not_called()
        audio_source.stop()


class TestInjectedAudioSource:
    def _make_transcriber(self, audio_source: FakeAudioSource) -> RecordingTranscriber:
        return RecordingTranscriber(
            transcription_options=TranscriptionOptions(
                model=TranscriptionModel(model_type=ModelType.WHISPER),
                language="en",
                task=Task.TRANSCRIBE,
            ),
            input_device_index=0,
            sample_rate=audio_source.sample_rate,
            model_path="/fake/path",
            sounddevice=MagicMock(),
            audio_source=audio_source,
        )

    def test_rejects_sample_rate_mismatch(self):
        audio_source = FakeAudioSource(sample_rate=48_000)

        with pytest.raises(ValueError, match="sample rate"):
            RecordingTranscriber(
                transcription_options=TranscriptionOptions(
                    model=TranscriptionModel(model_type=ModelType.WHISPER),
                ),
                input_device_index=0,
                sample_rate=16_000,
                model_path="/fake/path",
                sounddevice=MagicMock(),
                audio_source=audio_source,
            )

    def test_uses_injected_source_sample_rate_for_segmentation_when_rate_is_none(self):
        audio_source = FakeAudioSource(sample_rate=48_000)
        transcriber = RecordingTranscriber(
            transcription_options=TranscriptionOptions(
                model=TranscriptionModel(model_type=ModelType.WHISPER),
            ),
            input_device_index=None,
            sample_rate=None,
            model_path="/fake/path",
            sounddevice=MagicMock(),
            audio_source=audio_source,
        )

        assert transcriber.sample_rate == 48_000
        assert transcriber.segmenter.sample_rate == 48_000
        assert transcriber.segmenter.max_utterance_seconds == 12.0

    def test_normal_stop_closes_source_before_finished(self):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)
        cleanup_observations = []

        def cleanup(model):
            cleanup_observations.append(audio_source.stopped)
            transcriber.finished.emit()

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(transcriber, "_cleanup_model", side_effect=cleanup):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)

            transcriber.stop_recording()
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert audio_source.start_count == 1
        assert audio_source.stop_count == 1
        assert cleanup_observations == [True]

    def test_transcribe_none_stops_source(self):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(transcriber, "_transcribe", return_value=None):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)
            audio_source.deliver(
                np.ones(12 * audio_source.sample_rate, dtype=np.float32)
            )
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert audio_source.start_count == 1
        assert audio_source.stop_count == 1

    def test_transcribe_exception_stops_source_before_emitting_error(self, qtbot):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)
        errors = []
        cleanup_observations = []

        def record_error(error):
            cleanup_observations.append(audio_source.stopped)
            errors.append(error)

        transcriber.error.connect(record_error)

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(
                 transcriber,
                 "_transcribe",
                 side_effect=RuntimeError("test transcription failure"),
             ):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)
            audio_source.deliver(
                np.ones(12 * audio_source.sample_rate, dtype=np.float32)
            )
            worker.join(timeout=2)
            qtbot.waitUntil(lambda: len(errors) == 1, timeout=2_000)

        assert not worker.is_alive()
        assert audio_source.start_count == 1
        assert audio_source.stop_count == 1
        assert errors == ["test transcription failure"]
        assert cleanup_observations == [True]

    def test_stop_during_transcription_waits_for_backend_before_source_stop(self):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)
        transcription_started = threading.Event()
        release_transcription = threading.Event()

        def transcribe(samples, model, initial_prompt):
            transcription_started.set()
            assert release_transcription.wait(timeout=2)
            return {"text": "finished batch"}

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(transcriber, "_transcribe", side_effect=transcribe), \
             patch.object(transcriber, "_cleanup_model"):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)
            audio_source.deliver(
                np.ones(12 * audio_source.sample_rate, dtype=np.float32)
            )
            assert transcription_started.wait(timeout=2)

            transcriber.stop_recording()
            assert audio_source.started
            assert audio_source.stop_count == 0

            release_transcription.set()
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert audio_source.stop_count == 1

    def test_stop_does_not_transcribe_unfinalized_segmenter_tail(self):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(transcriber, "_transcribe") as transcribe, \
             patch.object(transcriber, "_cleanup_model"):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)
            samples = np.ones(100, dtype=np.float32)
            audio_source.deliver(samples)

            transcriber.stop_recording()
            worker.join(timeout=2)

        assert not worker.is_alive()
        transcribe.assert_not_called()


def _live_pcm(seconds: float, amplitude: float, sample_rate: int = 1_000):
    samples = np.full(int(seconds * sample_rate), amplitude, dtype=np.float32)
    samples[1::2] *= -1
    return samples


def _make_live_transcriber(
    mode: RecordingTranscriberMode = RecordingTranscriberMode.APPEND_BELOW,
    transcription_step: float = 3.5,
):
    source = FakeAudioSource(sample_rate=1_000)
    options = TranscriptionOptions(
        model=TranscriptionModel(model_type=ModelType.WHISPER),
        language="en",
        task=Task.TRANSCRIBE,
        silence_threshold=0.01,
        transcription_step=transcription_step,
        initial_prompt="seed prompt",
    )
    mode_index = list(RecordingTranscriberMode).index(mode)
    with patch("buzz.transcriber.recording_transcriber.Settings") as settings_class:
        settings_class.return_value.value.side_effect = [mode_index, "whisper-1"]
        transcriber = RecordingTranscriber(
            transcription_options=options,
            input_device_index=None,
            sample_rate=source.sample_rate,
            model_path="/fake/path",
            sounddevice=MagicMock(),
            audio_source=source,
        )
    return transcriber, source


def _start_live_worker(transcriber, transcribe):
    cleanup = patch.object(transcriber, "_cleanup_model")
    load_model = patch.object(transcriber, "_load_model", return_value=object())
    cleanup.start()
    load_model.start()
    transcribe_patch = patch.object(transcriber, "_transcribe", side_effect=transcribe)
    transcribe_mock = transcribe_patch.start()
    worker = threading.Thread(target=transcriber.start)
    worker.start()
    assert transcriber.audio_source.started_event.wait(timeout=2)
    return worker, transcribe_mock, (transcribe_patch, load_model, cleanup)


def _stop_live_worker(transcriber, worker, patches):
    transcriber.stop_recording()
    worker.join(timeout=2)
    for active_patch in patches:
        active_patch.stop()
    assert not worker.is_alive()


class _PausedWaitEvent:
    """Pause immediately before Event.wait to exercise set-before-wait ordering."""

    def __init__(self):
        self._event = threading.Event()
        self.wait_entered = threading.Event()
        self.allow_wait = threading.Event()

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def wait(self, timeout=None):
        self.wait_entered.set()
        assert self.allow_wait.wait(timeout=2)
        return self._event.wait(timeout)


class TestAdaptiveLiveIntegration:
    def test_backend_waits_for_endpoint(self):
        transcriber, _ = _make_live_transcriber()
        worker, transcribe, patches = _start_live_worker(
            transcriber,
            lambda samples, model, prompt: {"text": "text"},
        )

        transcriber.audio_source.deliver(_live_pcm(1.0, 0.1))

        transcribe.assert_not_called()
        _stop_live_worker(transcriber, worker, patches)

    def test_natural_pause_wakes_worker_and_transcribes(self):
        transcriber, source = _make_live_transcriber()
        called = threading.Event()

        def transcribe(samples, model, prompt):
            called.set()
            return {"text": "natural"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(np.concatenate((_live_pcm(1.2, 0.1), _live_pcm(0.6, 0))))

        assert called.wait(timeout=2)
        assert transcribe_mock.call_count == 1
        _stop_live_worker(transcriber, worker, patches)

    def test_short_pause_does_not_wake_backend(self):
        transcriber, source = _make_live_transcriber()
        worker, transcribe, patches = _start_live_worker(
            transcriber,
            lambda samples, model, prompt: {"text": "text"},
        )
        source.deliver(np.concatenate((
            _live_pcm(1.2, 0.1),
            _live_pcm(0.15, 0),
            _live_pcm(1.0, 0.1),
        )))

        transcribe.assert_not_called()
        _stop_live_worker(transcriber, worker, patches)

    def test_normal_mode_forces_update_at_12_seconds(self):
        transcriber, source = _make_live_transcriber()
        called = threading.Event()

        def transcribe(samples, model, prompt):
            called.set()
            return {"text": "normal"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(_live_pcm(12.1, 0.1))

        assert called.wait(timeout=2)
        assert transcribe_mock.call_args.args[0].size == 12_000
        _stop_live_worker(transcriber, worker, patches)

    def test_append_and_correct_transcription_step_is_forced_deadline(self):
        transcriber, source = _make_live_transcriber(
            RecordingTranscriberMode.APPEND_AND_CORRECT,
            transcription_step=3.5,
        )
        called = threading.Event()

        def transcribe(samples, model, prompt):
            called.set()
            return {"text": "checkpoint"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(_live_pcm(3.6, 0.1))

        assert called.wait(timeout=2)
        assert transcribe_mock.call_args.args[0].size == 3_500
        _stop_live_worker(transcriber, worker, patches)

    def test_append_and_correct_adds_exactly_one_context_tail(self):
        transcriber, source = _make_live_transcriber(
            RecordingTranscriberMode.APPEND_AND_CORRECT,
        )
        first_call = threading.Event()
        second_call = threading.Event()

        def transcribe(samples, model, prompt):
            if transcribe_mock.call_count == 1:
                first_call.set()
            else:
                second_call.set()
            return {"text": "result"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(_live_pcm(3.5, 0.1))
        assert first_call.wait(timeout=2)
        source.deliver(_live_pcm(3.5, 0.2))

        assert second_call.wait(timeout=2)
        first_samples = transcribe_mock.call_args_list[0].args[0]
        second_samples = transcribe_mock.call_args_list[1].args[0]
        assert first_samples.size == 3_500
        assert second_samples.size == 5_000
        np.testing.assert_array_equal(second_samples[:1_500], first_samples[-1_500:])
        assert np.allclose(np.abs(second_samples[1_500:]), 0.2)
        _stop_live_worker(transcriber, worker, patches)

    def test_append_and_correct_context_uses_only_previous_unique_segment(self):
        transcriber, _ = _make_live_transcriber(
            RecordingTranscriberMode.APPEND_AND_CORRECT,
        )
        unique_segments = [
            np.full(2_000, value, dtype=np.float32)
            for value in (0.1, 0.2, 0.3, 0.4)
        ]

        backend_inputs = [
            transcriber._prepare_backend_samples(segment)
            for segment in unique_segments
        ]

        np.testing.assert_array_equal(backend_inputs[0], unique_segments[0])
        for index in range(1, 4):
            expected = np.concatenate((
                unique_segments[index - 1][-1_500:],
                unique_segments[index],
            ))
            np.testing.assert_array_equal(backend_inputs[index], expected)

    def test_normal_mode_does_not_add_context(self):
        transcriber, _ = _make_live_transcriber()
        first = np.full(100, 0.1, dtype=np.float32)
        second = np.full(100, 0.2, dtype=np.float32)

        assert transcriber._prepare_backend_samples(first) is first
        assert transcriber._prepare_backend_samples(second) is second

    def test_one_callback_preserves_multiple_utterance_order(self):
        transcriber, source = _make_live_transcriber()
        first = np.full(100, 0.1, dtype=np.float32)
        second = np.full(100, 0.2, dtype=np.float32)
        transcriber.segmenter = MagicMock()
        transcriber.segmenter.push.return_value = [first, second]
        transcriber.segmenter.buffered_sample_count = 0
        finished = threading.Event()

        def transcribe(samples, model, prompt):
            if transcribe_mock.call_count == 2:
                finished.set()
            return {"text": "result"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(np.ones(10, dtype=np.float32))

        assert finished.wait(timeout=2)
        np.testing.assert_array_equal(transcribe_mock.call_args_list[0].args[0], first)
        np.testing.assert_array_equal(transcribe_mock.call_args_list[1].args[0], second)
        _stop_live_worker(transcriber, worker, patches)

    def test_pending_overflow_keeps_oldest_complete_utterance(self):
        transcriber, _ = _make_live_transcriber()
        transcriber.max_pending_samples = 150
        first = np.full(100, 0.1, dtype=np.float32)
        second = np.full(100, 0.2, dtype=np.float32)
        transcriber.segmenter = MagicMock()
        transcriber.segmenter.push.return_value = [first, second]
        transcriber.segmenter.buffered_sample_count = 0

        transcriber.on_audio(np.ones(10, dtype=np.float32))

        assert len(transcriber.pending_utterances) == 1
        assert transcriber.pending_utterances[0] is first
        assert transcriber.pending_sample_count == 100

    def test_queue_size_signal_remains_a_sample_count(self):
        transcriber, _ = _make_live_transcriber()
        emitted = []
        transcriber.queue_size_changed.connect(emitted.append)

        transcriber.on_audio(np.concatenate((_live_pcm(1.2, 0.1), _live_pcm(0.6, 0))))

        assert emitted[-1] == (
            transcriber.segmenter.buffered_sample_count
            + transcriber.pending_sample_count
        )

    def test_set_before_wait_cannot_miss_completed_utterance(self):
        transcriber, source = _make_live_transcriber()
        controlled_event = _PausedWaitEvent()
        transcriber._utterance_available = controlled_event
        called = threading.Event()

        def transcribe(samples, model, prompt):
            called.set()
            return {"text": "ready"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        assert controlled_event.wait_entered.wait(timeout=2)

        source.deliver(np.concatenate((_live_pcm(0.1, 0.1), _live_pcm(0.6, 0))))
        assert len(transcriber.pending_utterances) == 1
        assert controlled_event._event.is_set()

        controlled_event.allow_wait.set()
        assert called.wait(timeout=2)
        assert transcribe_mock.call_count == 1
        _stop_live_worker(transcriber, worker, patches)

    def test_stop_discards_already_finalized_pending_without_transcribing(self):
        transcriber, source = _make_live_transcriber()
        controlled_event = _PausedWaitEvent()
        transcriber._utterance_available = controlled_event
        worker, transcribe, patches = _start_live_worker(
            transcriber,
            lambda samples, model, prompt: {"text": "unexpected"},
        )
        assert controlled_event.wait_entered.wait(timeout=2)

        source.deliver(np.concatenate((_live_pcm(0.1, 0.1), _live_pcm(0.6, 0))))
        assert len(transcriber.pending_utterances) == 1

        transcriber.stop_recording()
        controlled_event.allow_wait.set()
        worker.join(timeout=2)
        for active_patch in patches:
            active_patch.stop()

        assert not worker.is_alive()
        assert source.stop_count == 1
        assert list(transcriber.pending_utterances) == []
        assert transcriber.pending_sample_count == 0
        transcribe.assert_not_called()

    def test_rolling_prompt_uses_only_last_1000_characters(self):
        transcriber, source = _make_live_transcriber()
        transcriber.segmenter = MagicMock()
        transcriber.segmenter.push.return_value = [
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
        transcriber.segmenter.buffered_sample_count = 0
        second_call = threading.Event()
        long_result = "x" * 1_500

        def transcribe(samples, model, prompt):
            if transcribe_mock.call_count == 1:
                return {"text": long_result}
            second_call.set()
            return {"text": "done"}

        worker, transcribe_mock, patches = _start_live_worker(transcriber, transcribe)
        source.deliver(np.ones(10, dtype=np.float32))

        assert second_call.wait(timeout=2)
        assert transcribe_mock.call_args_list[0].args[2] == "seed prompt"
        assert transcribe_mock.call_args_list[1].args[2] == "x" * 1_000
        _stop_live_worker(transcriber, worker, patches)


class TestStopRecording:
    def test_stop_recording_sets_is_running_false(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            transcriber.is_running = True
            transcriber.stop_recording()

            assert transcriber.is_running is False

    def test_stop_recording_terminates_process(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # Mock a running process
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            transcriber.process = mock_process

            transcriber.stop_recording()

            # Process should have been terminated and waited
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=5)

    def test_stop_recording_skips_terminated_process(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"):
            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            # Mock an already terminated process
            mock_process = MagicMock()
            mock_process.poll.return_value = 0  # Process already terminated
            transcriber.process = mock_process

            transcriber.stop_recording()

            # terminate and wait should not be called
            mock_process.terminate.assert_not_called()
            mock_process.wait.assert_not_called()


class TestStartLocalWhisperServer:
    def test_start_local_whisper_server_creates_openai_client(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):

            # Mock a successful process
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process

            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            try:
                transcriber.is_running = True
                transcriber.start_local_whisper_server()

                # Should have created an OpenAI client
                assert transcriber.openai_client is not None
                assert transcriber.process is not None
            finally:
                # Clean up to prevent QThread warnings
                transcriber.is_running = False
                transcriber.process = None

    def test_start_local_whisper_server_with_language(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="fr",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):

            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            try:
                transcriber.is_running = True
                transcriber.start_local_whisper_server()

                # Check that the language was passed to the command
                call_args = mock_popen.call_args
                cmd = call_args[0][0]
                assert "--language" in cmd
                assert "fr" in cmd
            finally:
                transcriber.is_running = False
                transcriber.process = None

    def test_start_local_whisper_server_auto_language(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language=None,
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):

            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            try:
                transcriber.is_running = True
                transcriber.start_local_whisper_server()

                # Check that auto language was used
                call_args = mock_popen.call_args
                cmd = call_args[0][0]
                assert "--language" in cmd
                assert "auto" in cmd
            finally:
                transcriber.is_running = False
                transcriber.process = None

    def test_start_local_whisper_server_handles_failure(self):
        transcription_options = TranscriptionOptions(
            model=TranscriptionModel(model_type=ModelType.WHISPER_CPP),
            language="en",
            task=Task.TRANSCRIBE,
        )

        with patch("sounddevice.check_input_settings"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):

            # Mock a failed process
            mock_process = MagicMock()
            mock_process.poll.return_value = 1  # Process terminated with error
            mock_process.stderr.read.return_value = b"Error loading model"
            mock_popen.return_value = mock_process

            transcriber = RecordingTranscriber(
                transcription_options=transcription_options,
                input_device_index=0,
                sample_rate=16000,
                model_path="/fake/path",
                sounddevice=MockSoundDevice(),
            )

            transcriptions = []
            transcriber.transcription.connect(lambda text: transcriptions.append(text))

            try:
                transcriber.is_running = True
                transcriber.start_local_whisper_server()

                # Should not have created a client when server failed
                assert transcriber.openai_client is None
                # Should have emitted starting and error messages
                assert len(transcriptions) >= 1
                # First message should be about starting Whisper.cpp
                assert "Whisper" in transcriptions[0]
            finally:
                transcriber.is_running = False
                transcriber.process = None
