import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sounddevice import PortAudioError

from buzz.audio_capture.source import AudioSourceError
from buzz.model_loader import TranscriptionModel, ModelType, WhisperModelSize
from buzz.settings.recording_transcriber_mode import RecordingTranscriberMode
from buzz.transcriber.recording_transcriber import (
    RecordingTranscriber,
    _WHISPER_CPP_STDERR_TAIL_MAX_BYTES,
)
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


def diagnostic_bytes(start: int, length: int) -> bytes:
    return bytes(
        0x0A if (index + 1) % 128 == 0 else index % 251
        for index in range(start, start + length)
    )


class TrackingStderr:
    line_bytes = 128

    def __init__(self, total_bytes: int, transcriber: RecordingTranscriber):
        self.total_bytes = total_bytes
        self.transcriber = transcriber
        self.yielded_count = 0
        self.tail_lengths = []

    def __iter__(self):
        offset = 0
        while offset < self.total_bytes:
            length = min(self.line_bytes, self.total_bytes - offset)
            line = diagnostic_bytes(offset, length)
            self.yielded_count += 1
            yield line
            self.tail_lengths.append(len(self.transcriber._stderr_tail))
            offset += length


class TrackingChunks:
    def __init__(self, chunks):
        self.chunks = chunks
        self.yielded_count = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded_count += 1
            yield chunk


class BlockingStderr:
    def __init__(self, chunks):
        self.chunks = chunks
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.reached_eof = threading.Event()
        self.yielded_count = 0

    def __iter__(self):
        self.blocked.set()
        if not self.release.wait(timeout=5):
            raise AssertionError("blocked stderr source was not released")

        try:
            for chunk in self.chunks:
                self.yielded_count += 1
                yield chunk
        finally:
            self.reached_eof.set()


class _ObservableLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_attempted = threading.Event()

    def acquire(self):
        self.acquire_attempted.set()
        return self._lock.acquire()

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def run_failed_whisper_server(
    transcriber: RecordingTranscriber,
    stderr,
) -> None:
    process = MagicMock()
    process.poll.return_value = 1
    process.stderr = stderr

    with (
        patch(
            "buzz.transcriber.recording_transcriber.subprocess.Popen",
            return_value=process,
        ),
        patch("buzz.transcriber.recording_transcriber.time.sleep"),
        patch("buzz.transcriber.recording_transcriber._", lambda s: s),
    ):
        transcriber.is_running = True
        transcriber.start_local_whisper_server()


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

    def test_stderr_diagnostics_start_empty(self):
        t = make_transcriber()

        assert t._stderr_tail == bytearray()
        assert t._stderr_out_of_device_memory is False
        assert t._stderr_lock is not None
        assert t._stderr_generation == 0


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


class TestWhisperServerStderr:
    @pytest.mark.parametrize(
        "total_bytes",
        [
            _WHISPER_CPP_STDERR_TAIL_MAX_BYTES // 2,
            _WHISPER_CPP_STDERR_TAIL_MAX_BYTES,
            _WHISPER_CPP_STDERR_TAIL_MAX_BYTES + 1,
            2 * _WHISPER_CPP_STDERR_TAIL_MAX_BYTES,
            10 * _WHISPER_CPP_STDERR_TAIL_MAX_BYTES,
        ],
        ids=["half-cap", "exact-cap", "cap-plus-one", "two-caps", "ten-caps"],
    )
    def test_drain_retains_exact_bounded_tail_and_consumes_all_lines(
        self,
        total_bytes,
    ):
        t = make_transcriber()
        stderr = TrackingStderr(total_bytes, t)
        process = MagicMock()
        process.stderr = stderr
        t.process = process

        t._drain_stderr(process, t._stderr_generation)

        retained_bytes = min(total_bytes, _WHISPER_CPP_STDERR_TAIL_MAX_BYTES)
        expected_tail = diagnostic_bytes(total_bytes - retained_bytes, retained_bytes)
        expected_line_count = (total_bytes + stderr.line_bytes - 1) // stderr.line_bytes
        assert stderr.yielded_count == expected_line_count
        assert max(stderr.tail_lengths) <= _WHISPER_CPP_STDERR_TAIL_MAX_BYTES
        assert len(t._stderr_tail) == retained_bytes
        assert bytes(t._stderr_tail) == expected_tail

    def test_single_huge_line_is_bounded_without_losing_its_tail(self):
        t = make_transcriber()
        line = b"discarded-prefix-" + (
            b"0123456789" * (_WHISPER_CPP_STDERR_TAIL_MAX_BYTES // 2)
        )
        stderr = TrackingChunks([line])
        process = MagicMock()
        process.stderr = stderr
        t.process = process

        t._drain_stderr(process, t._stderr_generation)

        assert stderr.yielded_count == 1
        assert len(t._stderr_tail) == _WHISPER_CPP_STDERR_TAIL_MAX_BYTES
        assert bytes(t._stderr_tail) == line[-_WHISPER_CPP_STDERR_TAIL_MAX_BYTES:]

    def test_blocked_stale_drainer_does_not_contaminate_current_generation(self):
        t = make_transcriber()
        with t._stderr_lock:
            t._stderr_generation += 1
            old_generation = t._stderr_generation
            t._stderr_tail.clear()
            t._stderr_out_of_device_memory = False

        old_stderr = BlockingStderr(
            [b"old ErrorOutOfDeviceMemory\n", b"old trailing diagnostics\n"]
        )
        old_process = MagicMock()
        old_process.stderr = old_stderr
        old_thread = threading.Thread(
            target=t._drain_stderr,
            args=(old_process, old_generation),
        )
        old_thread.start()

        try:
            assert old_stderr.blocked.wait(timeout=5)
            assert old_thread.is_alive()

            with t._stderr_lock:
                t._stderr_generation += 1
                current_generation = t._stderr_generation
                t._stderr_tail.clear()
                t._stderr_out_of_device_memory = False
            t._append_stderr_chunk(b"current diagnostics\n", current_generation)
        finally:
            old_stderr.release.set()
            old_thread.join(timeout=5)

        assert not old_thread.is_alive()
        assert old_stderr.reached_eof.is_set()
        assert old_stderr.yielded_count == 2
        assert bytes(t._stderr_tail) == b"current diagnostics\n"
        assert t._stderr_out_of_device_memory is False

    def test_stale_stderr_append_rechecks_generation_after_acquiring_lock(self):
        t = make_transcriber()
        observable_lock = _ObservableLock()
        t._stderr_lock = observable_lock
        old_generation = t._stderr_generation
        stale_chunk = b"STALE-GENERATION ErrorOutOfDeviceMemory\n"
        worker_errors = []

        def append_stale_chunk():
            try:
                t._append_stderr_chunk(stale_chunk, old_generation)
            except BaseException as error:
                worker_errors.append(error)

        observable_lock.acquire()
        observable_lock.acquire_attempted.clear()
        worker = threading.Thread(target=append_stale_chunk)
        worker.start()
        acquisition_attempted = False
        try:
            acquisition_attempted = observable_lock.acquire_attempted.wait(timeout=5)
            if acquisition_attempted:
                t._stderr_generation = old_generation + 1
                t._stderr_tail[:] = b"CURRENT-GENERATION\n"
                t._stderr_out_of_device_memory = False
        finally:
            observable_lock.release()

        worker.join(timeout=5)

        assert acquisition_attempted
        assert not worker.is_alive()
        if worker_errors:
            raise worker_errors[0]
        assert bytes(t._stderr_tail) == b"CURRENT-GENERATION\n"
        assert b"STALE-GENERATION" not in t._stderr_tail
        assert t._stderr_out_of_device_memory is False

    def test_drainer_uses_bound_process_instead_of_current_process(self):
        t = make_transcriber()
        old_stderr = TrackingChunks([b"old process diagnostics\n"])
        old_process = MagicMock()
        old_process.stderr = old_stderr
        new_stderr = TrackingChunks([b"new process diagnostics\n"])
        new_process = MagicMock()
        new_process.stderr = new_stderr
        t.process = new_process

        t._drain_stderr(old_process, t._stderr_generation)

        assert old_stderr.yielded_count == 1
        assert new_stderr.yielded_count == 0
        assert bytes(t._stderr_tail) == b"old process diagnostics\n"

    def test_current_generation_records_tail_and_oom(self):
        t = make_transcriber()
        stderr = TrackingChunks([b"current ErrorOutOfDeviceMemory diagnostics\n"])
        process = MagicMock()
        process.stderr = stderr

        t._drain_stderr(process, t._stderr_generation)

        assert stderr.yielded_count == 1
        assert bytes(t._stderr_tail) == b"current ErrorOutOfDeviceMemory diagnostics\n"
        assert t._stderr_out_of_device_memory is True

    def test_oom_guidance_survives_marker_eviction_from_tail(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)
        stderr = TrackingChunks(
            [b"ErrorOutOfDeviceMemory\n"] + [b"x" * 1023 + b"\n"] * 65
        )

        run_failed_whisper_server(t, stderr)

        assert stderr.yielded_count == 66
        assert b"ErrorOutOfDeviceMemory" not in t._stderr_tail
        assert t._stderr_out_of_device_memory is True
        assert any("insufficient memory" in message.lower() for message in emitted)

    def test_oom_guidance_survives_huge_line_prefix_truncation(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)
        line = b"ErrorOutOfDeviceMemory:" + (
            b"x" * (3 * _WHISPER_CPP_STDERR_TAIL_MAX_BYTES)
        )
        stderr = TrackingChunks([line])

        run_failed_whisper_server(t, stderr)

        assert stderr.yielded_count == 1
        assert len(t._stderr_tail) == _WHISPER_CPP_STDERR_TAIL_MAX_BYTES
        assert b"ErrorOutOfDeviceMemory" not in t._stderr_tail
        assert t._stderr_out_of_device_memory is True
        assert any("insufficient memory" in message.lower() for message in emitted)

    def test_large_non_oom_diagnostics_do_not_emit_memory_guidance(self):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)
        line = b"erroroutofdevicememory:" + (
            b"x" * (2 * _WHISPER_CPP_STDERR_TAIL_MAX_BYTES)
        )
        stderr = TrackingChunks([line])

        run_failed_whisper_server(t, stderr)

        assert stderr.yielded_count == 1
        assert t._stderr_out_of_device_memory is False
        assert not any("insufficient memory" in message.lower() for message in emitted)

    def test_oom_state_is_reset_between_start_attempts(self):
        t = make_transcriber()
        initial_generation = t._stderr_generation
        emitted = []
        t.transcription.connect(emitted.append)
        first_process = MagicMock()
        first_process.poll.return_value = 1
        first_process.stderr = TrackingChunks([b"ErrorOutOfDeviceMemory\n"])
        second_process = MagicMock()
        second_process.poll.return_value = 1
        second_process.stderr = TrackingChunks([b"ordinary failure\n"])

        with (
            patch(
                "buzz.transcriber.recording_transcriber.subprocess.Popen",
                side_effect=[first_process, second_process],
            ),
            patch("buzz.transcriber.recording_transcriber.time.sleep"),
            patch("buzz.transcriber.recording_transcriber._", lambda s: s),
        ):
            t.is_running = True
            t.start_local_whisper_server()
            assert t._stderr_generation == initial_generation + 1
            assert t._stderr_out_of_device_memory is True
            t.start_local_whisper_server()

        memory_messages = [
            message for message in emitted if "insufficient memory" in message.lower()
        ]
        assert len(memory_messages) == 1
        assert t._stderr_generation == initial_generation + 2
        assert t._stderr_out_of_device_memory is False
        assert bytes(t._stderr_tail) == b"ordinary failure\n"

    def test_diagnostics_are_reset_before_popen_failure(self):
        t = make_transcriber()
        initial_generation = t._stderr_generation
        t._append_stderr_chunk(
            b"old ErrorOutOfDeviceMemory diagnostics\n",
            initial_generation,
        )

        with patch(
            "buzz.transcriber.recording_transcriber.subprocess.Popen",
            side_effect=OSError("cannot exec"),
        ):
            t.is_running = True
            t.start_local_whisper_server()

        assert t._stderr_tail == bytearray()
        assert t._stderr_out_of_device_memory is False
        assert t._stderr_generation == initial_generation + 1

    def test_invalid_utf8_diagnostics_do_not_break_failure_handling(self, caplog):
        t = make_transcriber()
        emitted = []
        t.transcription.connect(emitted.append)
        stderr = TrackingChunks([b"invalid: \xff\xfe\n"])

        run_failed_whisper_server(t, stderr)

        assert stderr.yielded_count == 1
        assert any("failed to start" in message.lower() for message in emitted)
        assert "\ufffd\ufffd" in caplog.text


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
