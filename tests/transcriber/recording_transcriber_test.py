import os
import sys
import threading
import time
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

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
        import sounddevice
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
            assert transcriber.n_batch_samples == 5 * 16000
            assert transcriber.keep_sample_seconds == 0.15
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

            # APPEND_AND_CORRECT mode should use smaller batch size and longer keep duration
            assert transcriber.n_batch_samples == int(transcription_options.transcription_step * 16000)
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
    def test_on_audio_adds_to_queue(self):
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

            initial_size = transcriber.queue.size
            transcriber.on_audio(in_data.reshape(-1))

            # Queue should have grown by 4 samples
            assert transcriber.queue.size == initial_size + 4

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

    def test_on_audio_drops_data_when_queue_full(self):
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

            # Fill the queue beyond max_queue_size
            transcriber.queue = np.ones(transcriber.max_queue_size, dtype=np.float32)
            initial_size = transcriber.queue.size

            # Try to add more data
            in_data = np.array([[0.1], [0.2]], dtype=np.float32)
            transcriber.on_audio(in_data.reshape(-1))

            # Queue should not have grown (data was dropped)
            assert transcriber.queue.size == initial_size

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
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        audio_source.deliver(samples)
        queued_samples = transcriber.queue[-samples.size:].copy()
        samples[:] = 0.9

        np.testing.assert_array_equal(
            transcriber.queue[-queued_samples.size:], queued_samples
        )
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

    def test_uses_injected_source_sample_rate_for_batching_when_rate_is_none(self):
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
        assert transcriber.n_batch_samples == 5 * 48_000

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
                np.ones(transcriber.n_batch_samples, dtype=np.float32)
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
                np.ones(transcriber.n_batch_samples, dtype=np.float32)
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
                np.ones(transcriber.n_batch_samples, dtype=np.float32)
            )
            assert transcription_started.wait(timeout=2)

            transcriber.stop_recording()
            assert audio_source.started
            assert audio_source.stop_count == 0

            release_transcription.set()
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert audio_source.stop_count == 1

    def test_stop_does_not_drain_partial_queue(self):
        audio_source = FakeAudioSource()
        transcriber = self._make_transcriber(audio_source)

        with patch.object(transcriber, "_load_model", return_value=object()), \
             patch.object(transcriber, "_cleanup_model"):
            worker = threading.Thread(target=transcriber.start)
            worker.start()
            assert audio_source.started_event.wait(timeout=2)
            samples = np.ones(100, dtype=np.float32)
            audio_source.deliver(samples)
            queue_size = transcriber.queue.size

            transcriber.stop_recording()
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert transcriber.queue.size == queue_size


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
