from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sounddevice import PortAudioError

from buzz.audio_capture.source import AudioSourceError
from buzz.model_loader import TranscriptionModel, ModelType, WhisperModelSize
from buzz.settings.recording_transcriber_mode import RecordingTranscriberMode
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import TranscriptionOptions, Task


def make_transcriber(
    model_type=ModelType.WHISPER,
    mode_index=0,
    silence_threshold=0.0,
    language=None,
) -> RecordingTranscriber:
    options = TranscriptionOptions(
        language=language,
        task=Task.TRANSCRIBE,
        model=TranscriptionModel(model_type=model_type, whisper_model_size=WhisperModelSize.TINY),
        silence_threshold=silence_threshold,
    )
    mock_sounddevice = MagicMock()

    with patch("buzz.transcriber.recording_transcriber.Settings") as MockSettings:
        instance = MockSettings.return_value
        instance.value.return_value = mode_index
        transcriber = RecordingTranscriber(
            transcription_options=options,
            input_device_index=None,
            sample_rate=16000,
            model_path="tiny",
            sounddevice=mock_sounddevice,
        )
    return transcriber


class TestRecordingTranscriberInit:
    def test_default_max_utterance_is_12_seconds(self):
        t = make_transcriber(mode_index=0)
        assert t.segmenter.max_utterance_seconds == 12.0

    def test_append_and_correct_deadline_uses_transcription_step(self):
        mode_index = list(RecordingTranscriberMode).index(RecordingTranscriberMode.APPEND_AND_CORRECT)
        t = make_transcriber(mode_index=mode_index)
        assert t.segmenter.max_utterance_seconds == t.transcription_options.transcription_step

    def test_append_and_correct_mode_keep_sample_seconds(self):
        mode_index = list(RecordingTranscriberMode).index(RecordingTranscriberMode.APPEND_AND_CORRECT)
        t = make_transcriber(mode_index=mode_index)
        assert t.keep_sample_seconds == 1.5

    def test_default_keep_sample_seconds(self):
        t = make_transcriber(mode_index=0)
        assert t.keep_sample_seconds == 0.15

    def test_pending_utterance_queue_starts_empty(self):
        t = make_transcriber()
        assert list(t.pending_utterances) == []
        assert t.pending_sample_count == 0

    def test_pending_pcm_is_bounded_to_15_seconds(self):
        t = make_transcriber()
        assert t.max_pending_samples == 15 * t.sample_rate


class TestAmplitude:
    def test_silence_returns_zero(self):
        arr = np.zeros(100, dtype=np.float32)
        assert RecordingTranscriber.amplitude(arr) == 0.0

    def test_unit_signal_returns_one(self):
        arr = np.ones(100, dtype=np.float32)
        assert abs(RecordingTranscriber.amplitude(arr) - 1.0) < 1e-6

    def test_rms_calculation(self):
        arr = np.array([0.6, 0.8], dtype=np.float32)
        expected = float(np.sqrt(np.mean(arr ** 2)))
        assert abs(RecordingTranscriber.amplitude(arr) - expected) < 1e-6


class TestStreamCallback:
    def test_emits_amplitude_changed(self):
        t = make_transcriber()
        emitted = []
        t.amplitude_changed.connect(lambda v: emitted.append(v))

        chunk = np.array([[0.5], [0.5]], dtype=np.float32)
        t.on_audio(chunk.reshape(-1))

        assert len(emitted) == 1

    def test_appends_to_segmenter_buffer_before_endpoint(self):
        t = make_transcriber()
        initial_size = t.segmenter.buffered_sample_count
        chunk = np.ones((100,), dtype=np.float32)
        t.on_audio(chunk)
        assert t.segmenter.buffered_sample_count == initial_size + 100

    def test_drops_completed_utterance_when_pending_queue_full(self):
        t = make_transcriber()
        t.pending_sample_count = t.max_pending_samples
        t.segmenter = MagicMock()
        t.segmenter.push.return_value = [np.ones(100, dtype=np.float32)]
        t.segmenter.buffered_sample_count = 0

        t.on_audio(np.ones(100, dtype=np.float32))

        assert list(t.pending_utterances) == []
        assert t.pending_sample_count == t.max_pending_samples

    def test_multiple_callback_blocks_preserve_samples_for_flush(self):
        t = make_transcriber()
        first = np.full(1_600, 0.1, dtype=np.float32)
        second = np.full(1_600, 0.2, dtype=np.float32)

        t.on_audio(first)
        t.on_audio(second)
        utterances = t.segmenter.flush()

        assert len(utterances) == 1
        np.testing.assert_array_equal(utterances[0], np.concatenate((first, second)))


class TestGetDeviceSampleRate:
    def test_returns_whisper_sample_rate_when_supported(self):
        with patch("sounddevice.check_input_settings"):
            rate = RecordingTranscriber.get_device_sample_rate(None)
        assert rate == 16000

    def test_falls_back_to_device_default_sample_rate(self):
        with patch("sounddevice.check_input_settings", side_effect=PortAudioError()), \
             patch("sounddevice.query_devices", return_value={"default_samplerate": 44100.0}):
            rate = RecordingTranscriber.get_device_sample_rate(None)
        assert rate == 44100

    def test_falls_back_to_whisper_rate_when_query_returns_non_dict(self):
        with patch("sounddevice.check_input_settings", side_effect=PortAudioError()), \
             patch("sounddevice.query_devices", return_value=None):
            rate = RecordingTranscriber.get_device_sample_rate(None)
        assert rate == 16000


class TestStopRecording:
    def test_sets_is_running_false(self):
        t = make_transcriber()
        t.is_running = True
        t.stop_recording()
        assert t.is_running is False

    def test_terminates_running_process(self):
        t = make_transcriber()
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is running
        t.process = mock_process

        t.stop_recording()

        mock_process.terminate.assert_called_once()

    def test_kills_process_on_timeout(self):
        import subprocess
        t = make_transcriber()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        t.process = mock_process

        t.stop_recording()

        mock_process.kill.assert_called_once()

    def test_skips_terminate_when_process_already_stopped(self):
        t = make_transcriber()
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # already exited
        t.process = mock_process

        t.stop_recording()

        mock_process.terminate.assert_not_called()


class TestStartWithSilence:
    """Tests for the main transcription loop with silence threshold."""

    def test_silent_audio_skips_transcription(self):
        t = make_transcriber(silence_threshold=1.0)  # very high threshold
        t.on_audio(np.zeros(30 * t.sample_rate, dtype=np.float32))

        assert list(t.pending_utterances) == []


class TestStartPortAudioError:
    def test_emits_error_on_portaudio_failure(self):
        t = make_transcriber()
        errors = []
        t.error.connect(lambda e: errors.append(e))

        with patch("buzz.transcriber.recording_transcriber.whisper") as mock_whisper, \
             patch("buzz.transcriber.recording_transcriber.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_whisper.load_model.return_value = MagicMock()

            with patch.object(t, "audio_source") as audio_source:
                audio_source.start.side_effect = AudioSourceError("open failed")
                t.start()

        assert len(errors) == 1


class TestAdaptiveSegmentation:
    def test_continuous_speech_reaches_normal_deadline(self):
        t = make_transcriber()

        t.on_audio(np.ones(12 * t.sample_rate, dtype=np.float32))

        assert len(t.pending_utterances) == 1
        assert t.pending_utterances[0].size == 12 * t.sample_rate


def _drive_one_cycle(transcriber, samples):
    """Run the transcription loop for exactly one batch then stop."""
    received = []
    transcriber.transcription.connect(received.append)
    transcriber.pending_utterances.append(samples.copy())
    transcriber.pending_sample_count = samples.size
    transcriber._utterance_available.set()
    transcriber.is_running = True

    def stop_after_first(_text):
        transcriber.is_running = False

    transcriber.transcription.connect(stop_after_first)

    transcriber.audio_source = MagicMock()

    transcriber.start()
    return received


class TestModelBackends:
    def test_faster_whisper_joins_segment_text(self):
        t = make_transcriber(model_type=ModelType.FASTER_WHISPER)
        samples = np.ones(12 * t.sample_rate, dtype=np.float32)

        class FakeWhisperModel:
            def __init__(self, **kwargs):
                pass

            def transcribe(self, **kwargs):
                seg1 = MagicMock()
                seg1.text = "hello"
                seg2 = MagicMock()
                seg2.text = "world"
                return [seg1, seg2], MagicMock()

        with patch("buzz.transcriber.recording_transcriber.torch") as mock_torch, \
             patch("buzz.transcriber.recording_transcriber.faster_whisper") as mock_fw:
            mock_torch.cuda.is_available.return_value = False
            mock_fw.WhisperModel = FakeWhisperModel

            received = _drive_one_cycle(t, samples)

        assert received == ["hello world"]

    def test_openai_api_returns_text(self):
        t = make_transcriber(model_type=ModelType.OPEN_AI_WHISPER_API)
        samples = np.ones(12 * t.sample_rate, dtype=np.float32)

        transcript = MagicMock()
        transcript.model_extra = {}
        transcript.text = "api text"

        with patch("buzz.transcriber.recording_transcriber.torch") as mock_torch, \
             patch("buzz.transcriber.recording_transcriber.OpenAI") as MockOpenAI:
            mock_torch.cuda.is_available.return_value = False
            client = MockOpenAI.return_value
            client.audio.transcriptions.create.return_value = transcript

            received = _drive_one_cycle(t, samples)

        assert received == ["api text"]

    def test_hugging_face_returns_text(self):
        t = make_transcriber(model_type=ModelType.HUGGING_FACE)
        samples = np.ones(12 * t.sample_rate, dtype=np.float32)

        class FakeTransformers:
            is_mms_model = False

            def __init__(self, path):
                pass

            def transcribe(self, audio, language, task):
                return {"text": "hf text"}

        with patch("buzz.transcriber.recording_transcriber.torch") as mock_torch, \
             patch(
                 "buzz.transcriber.recording_transcriber.TransformersTranscriber",
                 FakeTransformers,
             ):
            mock_torch.cuda.is_available.return_value = False
            received = _drive_one_cycle(t, samples)

        assert received == ["hf text"]


class TestEffectivePrompt:
    def test_whisper_uses_effective_prompt(self):
        t = make_transcriber(model_type=ModelType.WHISPER)

        class FakeWhisper:
            def __init__(self):
                self.transcribe = MagicMock(return_value={"text": "text"})

        model = FakeWhisper()
        with patch("buzz.transcriber.recording_transcriber.whisper.Whisper", FakeWhisper):
            t._transcribe_whisper(np.ones(100, dtype=np.float32), model, "dynamic")

        assert model.transcribe.call_args.kwargs["initial_prompt"] == "dynamic"

    def test_faster_whisper_uses_effective_prompt(self):
        t = make_transcriber(model_type=ModelType.FASTER_WHISPER)

        class FakeFasterWhisper:
            def __init__(self):
                self.transcribe = MagicMock(return_value=([], MagicMock()))

        model = FakeFasterWhisper()
        with patch(
            "buzz.transcriber.recording_transcriber.faster_whisper.WhisperModel",
            FakeFasterWhisper,
        ):
            t._transcribe_faster_whisper(
                np.ones(100, dtype=np.float32),
                model,
                "dynamic",
            )

        assert model.transcribe.call_args.kwargs["initial_prompt"] == "dynamic"

    @pytest.mark.parametrize(
        "model_type",
        [ModelType.OPEN_AI_WHISPER_API, ModelType.WHISPER_CPP],
    )
    def test_api_paths_use_effective_prompt(self, model_type):
        t = make_transcriber(model_type=model_type)
        t.openai_client = MagicMock()
        transcript = MagicMock()
        transcript.model_extra = {}
        transcript.text = "text"
        t.openai_client.audio.transcriptions.create.return_value = transcript

        t._transcribe_via_api(np.ones(100, dtype=np.float32), "dynamic")

        assert (
            t.openai_client.audio.transcriptions.create.call_args.kwargs["prompt"]
            == "dynamic"
        )

    def test_hugging_face_call_signature_is_unchanged(self):
        t = make_transcriber(model_type=ModelType.HUGGING_FACE)

        class FakeTransformers:
            is_mms_model = False

            def __init__(self):
                self.transcribe = MagicMock(return_value={"text": "text"})

        model = FakeTransformers()
        with patch(
            "buzz.transcriber.recording_transcriber.TransformersTranscriber",
            FakeTransformers,
        ):
            t._transcribe_hugging_face(np.ones(100, dtype=np.float32), model)

        assert "initial_prompt" not in model.transcribe.call_args.kwargs


class TestStartLocalWhisperServer:
    def test_success_creates_openai_client(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)

        process = MagicMock()
        process.poll.return_value = None  # still running
        process.stderr = []

        with patch("buzz.transcriber.recording_transcriber.subprocess.Popen", return_value=process) as popen, \
             patch("buzz.transcriber.recording_transcriber.time.sleep"), \
             patch("buzz.transcriber.recording_transcriber._", lambda s: s), \
             patch("buzz.transcriber.recording_transcriber.OpenAI") as MockOpenAI:
            t.is_running = True
            t.start_local_whisper_server()

        assert t.openai_client is MockOpenAI.return_value
        cmd = popen.call_args[0][0]
        assert "--language" in cmd and "auto" in cmd
        assert any("transcription" in m.lower() for m in emitted)

    def test_failure_emits_error_message(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)

        process = MagicMock()
        process.poll.return_value = 1  # exited immediately
        process.stderr = [b"some failure\n"]

        with patch("buzz.transcriber.recording_transcriber.subprocess.Popen", return_value=process), \
             patch("buzz.transcriber.recording_transcriber.time.sleep"), \
             patch("buzz.transcriber.recording_transcriber._", lambda s: s):
            t.is_running = True
            t.start_local_whisper_server()

        assert t.openai_client is None
        assert any("failed to start" in m.lower() for m in emitted)

    def test_out_of_memory_emits_specific_message(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)

        process = MagicMock()
        process.poll.return_value = 1
        process.stderr = [b"ErrorOutOfDeviceMemory\n"]

        with patch("buzz.transcriber.recording_transcriber.subprocess.Popen", return_value=process), \
             patch("buzz.transcriber.recording_transcriber.time.sleep"), \
             patch("buzz.transcriber.recording_transcriber._", lambda s: s):
            t.is_running = True
            t.start_local_whisper_server()

        assert any("memory" in m.lower() for m in emitted)

    def test_popen_failure_is_handled(self):
        t = make_transcriber()

        with patch(
            "buzz.transcriber.recording_transcriber.subprocess.Popen",
            side_effect=OSError("cannot exec"),
        ):
            t.is_running = True
            t.start_local_whisper_server()

        assert t.openai_client is None
        assert t.process is None
