from __future__ import annotations

import math
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import buzz.transcriber.incremental_transcript as incremental_transcript_module
from buzz.audio_capture.meeting_audio_fanout import MeetingAudioFanout
from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
)
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracks,
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
    _TrackTimingTracker,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderState,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSession,
    MeetingSessionState,
)
from buzz.model_loader import ModelType, TranscriptionModel
from buzz.transcriber.incremental_transcript import IncrementalTranscript
from buzz.transcriber.live_segmenter import LiveSegmenter
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import Task, TranscriptionOptions


_DEADLOCK_TIMEOUT_SECONDS = 10
_MARKER_OFFSET = 3_000
_RECORDER_SAMPLE_RATE = 100
_RECORDER_BLOCK_SAMPLES = 100
_RECORDER_BUFFER_SECONDS = 64
_LIVE_SAMPLE_RATE = 1_000
_MAX_TEST_ARRAY_SAMPLES = 20_000


@pytest.fixture(autouse=True)
def _reject_whole_meeting_array_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail if this stress harness starts materializing whole-meeting PCM."""

    for constructor_name in ("empty", "full", "ones", "zeros"):
        original = getattr(np, constructor_name)

        def guarded_constructor(
            shape,
            *args,
            _original=original,
            _name=constructor_name,
            **kwargs,
        ):
            sample_count = int(shape) if np.isscalar(shape) else math.prod(shape)
            assert sample_count <= _MAX_TEST_ARRAY_SAMPLES, (
                f"whole-meeting array allocation rejected: constructor=np.{_name}, "
                f"samples={sample_count}, cap={_MAX_TEST_ARRAY_SAMPLES}"
            )
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, constructor_name, guarded_constructor)

    original_arange = np.arange

    def guarded_arange(*args, **kwargs):
        if not 1 <= len(args) <= 3:
            return original_arange(*args, **kwargs)
        if set(kwargs) - {"dtype", "device", "like"}:
            return original_arange(*args, **kwargs)
        if kwargs.get("device") is not None or kwargs.get("like") is not None:
            return original_arange(*args, **kwargs)
        try:
            if kwargs.get("dtype") is not None:
                np.dtype(kwargs["dtype"])
            if len(args) == 1:
                start, stop, step = 0, args[0], 1
            elif len(args) == 2:
                start, stop = args
                step = 1
            else:
                start, stop, step = args
            if step == 0:
                return original_arange(*args, **kwargs)
            sample_count = max(0, math.ceil((stop - start) / step))
        except (ArithmeticError, TypeError, ValueError):
            return original_arange(*args, **kwargs)

        assert sample_count <= _MAX_TEST_ARRAY_SAMPLES, (
            "whole-meeting array allocation rejected: constructor=np.arange, "
            f"samples={sample_count}, cap={_MAX_TEST_ARRAY_SAMPLES}"
        )
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(np, "arange", guarded_arange)


def test_allocation_guard_rejects_large_arange_and_preserves_small_arange() -> None:
    with pytest.raises(AssertionError, match="np.arange.*samples=20001"):
        np.arange(_MAX_TEST_ARRAY_SAMPLES + 1, dtype=np.float32)

    np.testing.assert_array_equal(
        np.arange(4, dtype=np.int16),
        np.array([0, 1, 2, 3], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        np.arange(2, 10, 2, dtype=np.int16),
        np.array([2, 4, 6, 8], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        np.arange(10, 2, -2, dtype=np.int16),
        np.array([10, 8, 6, 4], dtype=np.int16),
    )
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        np.arange(0, 5, 0)
    with pytest.raises(TypeError, match="not understood"):
        np.arange(3, dtype="not-a-dtype")


def _marker(ordinal: int) -> np.float32:
    return np.float32((_MARKER_OFFSET + ordinal) / 32_767.0)


def _expected_pcm16_marker(ordinal: int) -> np.int16:
    return np.int16(_MARKER_OFFSET + ordinal)


class _CountingArchiveWriter:
    """Validate and discard each archive block without retaining historical PCM."""

    def __init__(self, block_samples: int) -> None:
        self.block_samples = block_samples
        self.written_samples = 0
        self.write_calls = 0
        self.expected_ordinal = 0
        self.order_signature = 0
        self.flush_calls = 0
        self.finalized = False
        self.published = False
        self.discarded = False
        self.closed_after_failure = False
        self.validation_error: Optional[AssertionError] = None
        self._condition = threading.Condition()

    def write(self, pcm16: np.ndarray) -> None:
        try:
            assert pcm16.dtype == np.int16
            assert pcm16.ndim == 1
            assert pcm16.size == self.block_samples, (
                f"ordinal={self.expected_ordinal}, size={pcm16.size}, "
                f"expected_size={self.block_samples}"
            )
            expected = _expected_pcm16_marker(self.expected_ordinal)
            assert bool(np.all(pcm16 == expected)), (
                f"ordinal={self.expected_ordinal}, expected_marker={int(expected)}, "
                f"actual_first={int(pcm16[0]) if pcm16.size else None}"
            )
        except AssertionError as exc:
            with self._condition:
                self.validation_error = exc
                self._condition.notify_all()
            raise

        with self._condition:
            self.written_samples += int(pcm16.size)
            self.write_calls += 1
            self.order_signature = (
                (self.order_signature * 1_000_003) + self.expected_ordinal + 1
            ) % 2_147_483_647
            self.expected_ordinal += 1
            self._condition.notify_all()

    def flush(self) -> None:
        self.flush_calls += 1

    def finalize(self) -> None:
        self.finalized = True

    def publish(self) -> None:
        self.published = True

    def discard(self) -> None:
        self.discarded = True

    def close_after_failure(self) -> None:
        self.closed_after_failure = True

    def wait_for_writes(self, target: int) -> None:
        with self._condition:
            reached = self._condition.wait_for(
                lambda: self.write_calls >= target or self.validation_error is not None,
                timeout=_DEADLOCK_TIMEOUT_SECONDS,
            )
            assert (
                reached
            ), f"writer catch-up timeout: target={target}, actual={self.write_calls}"
            if self.validation_error is not None:
                raise self.validation_error


class _GatedArchiveWriter(_CountingArchiveWriter):
    def __init__(self, block_samples: int) -> None:
        super().__init__(block_samples)
        self.first_write_entered = threading.Event()
        self.release_first_write = threading.Event()

    def write(self, pcm16: np.ndarray) -> None:
        if self.write_calls == 0:
            self.first_write_entered.set()
            assert self.release_first_write.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        super().write(pcm16)


class _SyntheticAudioSource(AudioSource):
    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._on_audio: Optional[AudioFrameCallback] = None
        self._on_error: Optional[AudioErrorCallback] = None
        self.started = False
        self.start_count = 0
        self.stop_count = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        with self._lock:
            self._on_audio = on_audio
            self._on_error = on_error
            self.started = True
            self.start_count += 1

    def stop(self) -> None:
        with self._lock:
            if self.started:
                self.started = False
                self.stop_count += 1

    def deliver(self, samples: np.ndarray) -> None:
        with self._lock:
            callback = self._on_audio
            started = self.started
        assert started and callback is not None
        callback(samples)


class _StartObservedAudioSource(AudioSource):
    """Expose deterministic subscription readiness around a real source adapter."""

    def __init__(self, source: AudioSource) -> None:
        self.source = source
        self.started = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self.source.sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        self.source.start(on_audio, on_error)
        self.started.set()

    def stop(self) -> None:
        self.source.stop()


class _AdvancingClock:
    def __init__(self, value: int = 0, step: int = 1_000_000) -> None:
        self.value = value
        self.step = step
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            value = self.value
            self.value += self.step
            return value


class _LogicalSessionClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.monotonic_ns = 0

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> int:
        return self.monotonic_ns

    def advance(self, seconds: int) -> None:
        self.wall += timedelta(seconds=seconds)
        self.monotonic_ns += seconds * 1_000_000_000


class _RealRecorderFactory:
    def __init__(self) -> None:
        self.recorders: dict[str, MeetingRecorder] = {}
        self.writers: dict[str, _CountingArchiveWriter] = {}

    def __call__(
        self,
        output_path: str | Path,
        sample_rate: int,
        *,
        max_buffer_seconds: float,
        on_error: Callable[[Exception], None],
    ) -> MeetingRecorder:
        name = Path(output_path).name
        writer = _CountingArchiveWriter(sample_rate)
        recorder = MeetingRecorder(
            output_path,
            sample_rate,
            max_buffer_seconds=max_buffer_seconds,
            on_error=on_error,
            _writer_factory=lambda _path, _rate, writer=writer: writer,
        )
        self.recorders[name] = recorder
        self.writers[name] = writer
        return recorder


class _RecordingThreadFactory:
    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []

    def __call__(
        self,
        *,
        target: Callable[[], None],
        name: str,
        daemon: bool,
    ) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=daemon)
        self.threads.append(thread)
        return thread


def _make_recorder(
    tmp_path: Path,
    writer: _CountingArchiveWriter,
    *,
    sample_rate: int = _RECORDER_SAMPLE_RATE,
    max_buffer_seconds: float = _RECORDER_BUFFER_SECONDS,
) -> MeetingRecorder:
    return MeetingRecorder(
        tmp_path / "archive.wav",
        sample_rate,
        max_buffer_seconds=max_buffer_seconds,
        _writer_factory=lambda _path, _rate: writer,
    )


def _wait_for_recorder_drain(recorder: MeetingRecorder) -> None:
    with recorder._condition:
        drained = recorder._condition.wait_for(
            lambda: recorder._buffered_sample_count == 0
            or recorder._state == MeetingRecorderState.FAILED,
            timeout=_DEADLOCK_TIMEOUT_SECONDS,
        )
        assert drained, (
            f"recorder drain timeout: buffered={recorder._buffered_sample_count}, "
            f"capacity={recorder.max_buffered_samples}"
        )
        assert recorder._state != MeetingRecorderState.FAILED, recorder.error


def _enqueue_owned_marker(
    recorder: MeetingRecorder,
    reusable_block: np.ndarray,
    ordinal: int,
) -> bool:
    # Holding the recorder's re-entrant Condition until after producer mutation
    # makes the ownership-copy check deterministic: the writer cannot dequeue
    # the block before the borrowed source storage changes.
    with recorder._condition:
        reusable_block.fill(_marker(ordinal))
        accepted = recorder.enqueue(reusable_block)
        reusable_block.fill(-1.0)
        return accepted


def _deliver_owned_marker(
    source: _SyntheticAudioSource,
    recorder: MeetingRecorder,
    reusable_block: np.ndarray,
    ordinal: int,
) -> None:
    with recorder._condition:
        reusable_block.fill(_marker(ordinal))
        source.deliver(reusable_block)
        reusable_block.fill(-1.0)


def _feed_recorder(
    recorder: MeetingRecorder,
    writer: _CountingArchiveWriter,
    *,
    block_count: int,
    block_samples: int,
    batch_blocks: int = 16,
) -> int:
    assert block_samples * batch_blocks < recorder.max_buffered_samples
    reusable_block = np.empty(block_samples, dtype=np.float32)
    peak_buffered = 0
    for ordinal in range(block_count):
        assert _enqueue_owned_marker(recorder, reusable_block, ordinal)
        peak_buffered = max(peak_buffered, recorder.buffered_sample_count)
        if (ordinal + 1) % batch_blocks == 0:
            writer.wait_for_writes(ordinal + 1)
            _wait_for_recorder_drain(recorder)
    writer.wait_for_writes(block_count)
    _wait_for_recorder_drain(recorder)
    return peak_buffered


@pytest.mark.parametrize("logical_seconds", [7_200, 14_400, 28_800])
def test_recorder_2h_4h_8h_is_lossless_ordered_and_duration_bounded(
    tmp_path: Path,
    logical_seconds: int,
) -> None:
    writer = _CountingArchiveWriter(_RECORDER_BLOCK_SAMPLES)
    recorder = _make_recorder(tmp_path, writer)
    recorder.start()

    block_count = logical_seconds
    expected_samples = _RECORDER_SAMPLE_RATE * logical_seconds
    peak_buffered = _feed_recorder(
        recorder,
        writer,
        block_count=block_count,
        block_samples=_RECORDER_BLOCK_SAMPLES,
    )
    result = recorder.stop()

    metrics = (
        f"logical_seconds={logical_seconds}, expected={expected_samples}, "
        f"accepted={recorder.accepted_sample_count}, "
        f"written={writer.written_samples}, result={result.sample_count}, "
        f"peak_buffered={peak_buffered}, capacity={recorder.max_buffered_samples}"
    )
    assert block_count == logical_seconds, metrics
    assert recorder.accepted_sample_count == expected_samples, metrics
    assert writer.written_samples == expected_samples, metrics
    assert result.sample_count == expected_samples, metrics
    assert writer.write_calls == block_count, metrics
    assert writer.expected_ordinal == block_count, metrics
    assert peak_buffered <= recorder.max_buffered_samples, metrics
    assert recorder.buffered_sample_count == 0, metrics
    assert recorder.max_buffered_samples == 6_400
    assert 28_800 == 4 * 7_200
    assert result.state == recorder.state == MeetingRecorderState.STOPPED
    assert result.published and writer.published and writer.finalized
    assert not writer.discarded and not writer.closed_after_failure
    assert recorder._writer_thread is not None
    assert not recorder._writer_thread.is_alive()


def test_recorder_below_capacity_backlog_drains_exactly_once(tmp_path: Path) -> None:
    writer = _GatedArchiveWriter(_RECORDER_BLOCK_SAMPLES)
    recorder = _make_recorder(
        tmp_path,
        writer,
        max_buffer_seconds=20,
    )
    recorder.start()
    reusable_block = np.empty(_RECORDER_BLOCK_SAMPLES, dtype=np.float32)

    assert _enqueue_owned_marker(recorder, reusable_block, 0)
    assert writer.first_write_entered.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
    for ordinal in range(1, 11):
        assert _enqueue_owned_marker(recorder, reusable_block, ordinal)

    peak_buffered = recorder.buffered_sample_count
    assert 0 < peak_buffered < recorder.max_buffered_samples
    writer.release_first_write.set()
    result = recorder.stop()

    assert recorder.accepted_sample_count == 1_100
    assert writer.written_samples == result.sample_count == 1_100
    assert writer.write_calls == writer.expected_ordinal == 11
    assert recorder.buffered_sample_count == 0
    assert result.published and writer.finalized and writer.published


def test_two_track_session_archives_interleaved_logical_8h(tmp_path: Path) -> None:
    logical_seconds = 28_800
    microphone = _SyntheticAudioSource(100)
    remote = _SyntheticAudioSource(200)
    recorder_factory = _RealRecorderFactory()
    thread_factory = _RecordingThreadFactory()
    track_clock = _AdvancingClock()
    tracks = MeetingAudioTracks(
        microphone,
        remote,
        tmp_path / "microphone.wav",
        tmp_path / "remote.wav",
        max_buffer_seconds=_RECORDER_BUFFER_SECONDS,
        _clock_ns=track_clock,
        _recorder_factory=recorder_factory,
        _thread_factory=thread_factory,
    )
    session_clock = _LogicalSessionClock()
    session = MeetingSession(
        tracks,
        MeetingRemoteSourceKind.SYSTEM,
        _wall_clock=session_clock.wall_now,
        _monotonic_ns=session_clock.monotonic_now,
    )
    session.start()

    mic_recorder = recorder_factory.recorders["microphone.wav"]
    remote_recorder = recorder_factory.recorders["remote.wav"]
    mic_writer = recorder_factory.writers["microphone.wav"]
    remote_writer = recorder_factory.writers["remote.wav"]
    mic_block = np.empty(100, dtype=np.float32)
    remote_block = np.empty(200, dtype=np.float32)
    mic_peak = remote_peak = 0
    for ordinal in range(logical_seconds):
        _deliver_owned_marker(microphone, mic_recorder, mic_block, ordinal)
        _deliver_owned_marker(remote, remote_recorder, remote_block, ordinal)
        mic_peak = max(mic_peak, mic_recorder.buffered_sample_count)
        remote_peak = max(remote_peak, remote_recorder.buffered_sample_count)
        if (ordinal + 1) % 16 == 0:
            mic_writer.wait_for_writes(ordinal + 1)
            remote_writer.wait_for_writes(ordinal + 1)
            _wait_for_recorder_drain(mic_recorder)
            _wait_for_recorder_drain(remote_recorder)

    mic_writer.wait_for_writes(logical_seconds)
    remote_writer.wait_for_writes(logical_seconds)
    _wait_for_recorder_drain(mic_recorder)
    _wait_for_recorder_drain(remote_recorder)
    session_clock.advance(logical_seconds)
    result = session.stop()

    assert microphone.start_count == remote.start_count == 1
    assert microphone.stop_count == remote.stop_count == 1
    assert result.outcome == MeetingAudioTracksOutcome.COMPLETE
    assert result.microphone.complete and result.remote.complete
    assert result.microphone.recording.sample_count == 2_880_000
    assert result.remote.recording.sample_count == 5_760_000
    assert mic_recorder.accepted_sample_count == mic_writer.written_samples == 2_880_000
    assert (
        remote_recorder.accepted_sample_count
        == remote_writer.written_samples
        == 5_760_000
    )
    assert mic_writer.write_calls == remote_writer.write_calls == logical_seconds
    assert mic_writer is not remote_writer
    assert mic_recorder is not remote_recorder
    assert mic_peak <= mic_recorder.max_buffered_samples
    assert remote_peak <= remote_recorder.max_buffered_samples
    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert session.state == MeetingSessionState.COMPLETED
    assert session.duration_ns == 28_800 * 1_000_000_000
    assert len(thread_factory.threads) == 4
    assert {thread.name for thread in thread_factory.threads} == {
        "meeting-track-start-microphone",
        "meeting-track-start-remote",
        "meeting-track-stop-microphone",
        "meeting-track-stop-remote",
    }
    assert all(not thread.is_alive() for thread in thread_factory.threads)

    for track_result in (result.microphone, result.remote):
        anchors = track_result.timing.anchors
        positions = [anchor.sample_end for anchor in anchors]
        assert positions == sorted(positions)
        assert len(positions) == len(set(positions))
        assert positions[-1] <= track_result.recording.sample_count
        assert 2_800 <= len(positions) <= 3_000
        assert len(positions) <= 4_096


def test_timing_tracker_retains_first_latest_and_exact_cap() -> None:
    tracker = _TrackTimingTracker(sample_rate=100)
    tracker.set_coordinator_start(0)
    for ordinal in range(5_000):
        sample_end = (ordinal + 1) * 1_000
        tracker.observe(sample_end, sample_end * 10)
    tracker.observe(5_000_100, 50_001_000)

    anchors = tracker.snapshot().anchors
    positions = [anchor.sample_end for anchor in anchors]
    assert len(anchors) == tracker._MAX_ANCHORS == 4_096
    assert positions[0] == 1_000
    assert positions[-1] == 5_000_100
    assert 2_000 not in positions
    assert positions == sorted(positions)


def _make_live_transcriber(
    audio_source: AudioSource,
    sample_rate: int,
) -> RecordingTranscriber:
    options = TranscriptionOptions(
        model=TranscriptionModel(model_type=ModelType.WHISPER),
        language="en",
        task=Task.TRANSCRIBE,
        silence_threshold=0.01,
    )
    with patch("buzz.transcriber.recording_transcriber.Settings") as settings:
        settings.return_value.value.side_effect = [0, "whisper-1"]
        return RecordingTranscriber(
            transcription_options=options,
            input_device_index=None,
            sample_rate=sample_rate,
            model_path="/synthetic/no-model",
            sounddevice=MagicMock(),
            audio_source=audio_source,
        )


def _start_live_harness(
    tmp_path: Path,
) -> tuple[
    _SyntheticAudioSource,
    MeetingRecorder,
    _CountingArchiveWriter,
    MeetingAudioFanout,
    RecordingTranscriber,
]:
    source = _SyntheticAudioSource(_LIVE_SAMPLE_RATE)
    writer = _CountingArchiveWriter(_LIVE_SAMPLE_RATE)
    recorder = _make_recorder(
        tmp_path,
        writer,
        sample_rate=_LIVE_SAMPLE_RATE,
        max_buffer_seconds=64,
    )
    fanout = MeetingAudioFanout(source, recorder)
    fanout.start()
    observed_live_source = _StartObservedAudioSource(fanout.live_source)
    transcriber = _make_live_transcriber(observed_live_source, _LIVE_SAMPLE_RATE)
    return source, recorder, writer, fanout, transcriber


def _deliver_live_seconds(
    source: _SyntheticAudioSource,
    recorder: MeetingRecorder,
    writer: _CountingArchiveWriter,
    reusable_block: np.ndarray,
    start_ordinal: int,
    seconds: int,
) -> None:
    for ordinal in range(start_ordinal, start_ordinal + seconds):
        _deliver_owned_marker(source, recorder, reusable_block, ordinal)
        if (ordinal + 1) % 8 == 0:
            writer.wait_for_writes(ordinal + 1)
            _wait_for_recorder_drain(recorder)


def test_slow_live_asr_backlog_never_breaks_lossless_archive(tmp_path: Path) -> None:
    source, recorder, writer, fanout, transcriber = _start_live_harness(tmp_path)
    backend_entered = threading.Event()
    release_backend = threading.Event()

    def gated_backend(
        samples: np.ndarray, model: object, prompt: str
    ) -> dict[str, str]:
        del samples, model, prompt
        backend_entered.set()
        assert release_backend.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        return {"text": "synthetic"}

    reusable_block = np.empty(_LIVE_SAMPLE_RATE, dtype=np.float32)
    with (
        patch.object(transcriber, "_load_model", return_value=object()),
        patch.object(transcriber, "_transcribe", side_effect=gated_backend),
        patch.object(transcriber, "_release_model"),
    ):
        worker = threading.Thread(target=transcriber.start, name="test-slow-live-asr")
        worker.start()
        observed_source = transcriber.audio_source
        assert isinstance(observed_source, _StartObservedAudioSource)
        assert observed_source.started.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)

        _deliver_live_seconds(source, recorder, writer, reusable_block, 0, 12)
        assert backend_entered.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)

        peak_pending = 0
        for ordinal in range(12, 120):
            _deliver_owned_marker(source, recorder, reusable_block, ordinal)
            with transcriber.mutex:
                pending = transcriber.pending_sample_count
            peak_pending = max(peak_pending, pending)
            assert pending <= transcriber.max_pending_samples
            if (ordinal + 1) % 8 == 0:
                writer.wait_for_writes(ordinal + 1)
                _wait_for_recorder_drain(recorder)

        writer.wait_for_writes(120)
        _wait_for_recorder_drain(recorder)
        stop_results = []
        archive_stopped = threading.Event()

        def stop_archive() -> None:
            stop_results.append(fanout.stop())
            archive_stopped.set()

        stop_worker = threading.Thread(
            target=stop_archive,
            name="test-stop-archive-with-blocked-live-asr",
        )
        stop_worker.start()
        assert archive_stopped.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        stop_worker.join(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        assert not stop_worker.is_alive()
        assert len(stop_results) == 1
        result = stop_results[0]

        assert backend_entered.is_set() and not release_backend.is_set()
        assert peak_pending > 0
        assert peak_pending <= transcriber.max_pending_samples == 15_000
        assert recorder.accepted_sample_count == 120_000
        assert writer.written_samples == result.sample_count == 120_000
        assert writer.write_calls == writer.expected_ordinal == 120
        assert result.state == MeetingRecorderState.STOPPED and result.published
        assert writer.finalized and writer.published

        transcriber.stop_recording()
        release_backend.set()
        worker.join(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        assert not worker.is_alive()
        assert transcriber.pending_sample_count == 0


def test_live_backend_failure_does_not_stop_or_truncate_archive(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source, recorder, writer, fanout, transcriber = _start_live_harness(tmp_path)
    backend_entered = threading.Event()

    def fail_backend(samples: np.ndarray, model: object, prompt: str) -> None:
        del samples, model, prompt
        backend_entered.set()
        raise RuntimeError("deterministic backend failure")

    reusable_block = np.empty(_LIVE_SAMPLE_RATE, dtype=np.float32)
    with (
        patch.object(transcriber, "_load_model", return_value=object()),
        patch.object(transcriber, "_transcribe", side_effect=fail_backend),
        patch.object(transcriber, "_release_model"),
    ):
        worker = threading.Thread(target=transcriber.start, name="test-failed-live-asr")
        worker.start()
        observed_source = transcriber.audio_source
        assert isinstance(observed_source, _StartObservedAudioSource)
        assert observed_source.started.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        _deliver_live_seconds(source, recorder, writer, reusable_block, 0, 12)
        assert backend_entered.wait(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        worker.join(timeout=_DEADLOCK_TIMEOUT_SECONDS)
        assert not worker.is_alive()
        assert "Unexpected error during recording: deterministic backend failure" in (
            caplog.text
        )

        _deliver_live_seconds(source, recorder, writer, reusable_block, 12, 48)
        writer.wait_for_writes(60)
        _wait_for_recorder_drain(recorder)
        result = fanout.stop()

    assert recorder.accepted_sample_count == 60_000
    assert writer.written_samples == result.sample_count == 60_000
    assert writer.write_calls == writer.expected_ordinal == 60
    assert result.state == MeetingRecorderState.STOPPED and result.published
    assert writer.finalized and writer.published


def test_live_segmenter_logical_2h_transient_state_is_bounded() -> None:
    segmenter = LiveSegmenter(
        sample_rate=_LIVE_SAMPLE_RATE,
        speech_threshold=0.01,
        max_utterance_seconds=12,
    )
    block = np.empty(500, dtype=np.float32)
    block[::2] = 0.1
    block[1::2] = -0.1
    peak_buffered = peak_energy_frames = peak_chunks = 0

    for _ in range(14_400):
        segmenter.push(block)
        peak_buffered = max(peak_buffered, segmenter.buffered_sample_count)
        peak_energy_frames = max(peak_energy_frames, len(segmenter._energy_frames))
        peak_chunks = max(peak_chunks, len(segmenter._chunks))
        assert segmenter.buffered_sample_count <= segmenter.max_buffered_sample_count
        assert segmenter._analysis_tail.size < segmenter._frame_samples
        assert len(segmenter._energy_frames) <= (
            math.ceil(segmenter.max_buffered_sample_count / segmenter._frame_samples)
            + 1
        )
        assert sum(chunk.size for _, chunk in segmenter._chunks) == (
            segmenter.buffered_sample_count
        )

    segmenter.flush()
    assert peak_buffered <= segmenter.max_buffered_sample_count
    assert peak_energy_frames <= 602
    assert peak_chunks <= 25
    assert segmenter.buffered_sample_count == 0
    assert not segmenter._chunks
    assert not segmenter._energy_frames
    assert segmenter._analysis_tail.size == 0


def test_live_segmenter_large_push_never_crosses_internal_sample_cap() -> None:
    segmenter = LiveSegmenter(
        sample_rate=_LIVE_SAMPLE_RATE,
        speech_threshold=0.01,
        max_utterance_seconds=12,
    )
    speech_500ms = np.empty(500, dtype=np.float32)
    speech_500ms[::2] = 0.1
    speech_500ms[1::2] = -0.1
    for _ in range(23):
        segmenter.push(speech_500ms)
    assert segmenter.buffered_sample_count == 11_500

    original_append = segmenter._append_chunk
    instantaneous_peaks: list[int] = []

    def observed_append(samples: np.ndarray) -> None:
        original_append(samples)
        instantaneous_peaks.append(segmenter._buffered_samples)

    segmenter._append_chunk = observed_append  # type: ignore[method-assign]
    speech_1s = np.empty(1_000, dtype=np.float32)
    speech_1s[::2] = 0.1
    speech_1s[1::2] = -0.1
    segmenter.push(speech_1s)

    assert instantaneous_peaks
    assert max(instantaneous_peaks) <= segmenter.max_buffered_sample_count
    segmenter.flush()
    assert segmenter.buffered_sample_count == 0


def test_incremental_transcript_8h_equivalent_keeps_revision_and_match_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical_seconds = 8 * 60 * 60
    update_interval_seconds = 12
    assert logical_seconds % update_interval_seconds == 0
    expected_updates = logical_seconds // update_interval_seconds
    max_revision_chars = 256
    match_window_units = 128
    previous_lengths: list[int] = []
    bounded_unit_counts: list[int] = []
    unbounded_input_lengths: list[int] = []
    original_find_anchor = IncrementalTranscript._find_anchor
    original_comparison_units = incremental_transcript_module._comparison_units

    def observed_find_anchor(
        self: IncrementalTranscript,
        previous: str,
        current: str,
    ):
        previous_lengths.append(len(previous))
        return original_find_anchor(self, previous, current)

    def observed_comparison_units(text: str, *, max_units: Optional[int] = None):
        units = original_comparison_units(text, max_units=max_units)
        if max_units is not None:
            assert max_units == match_window_units
            bounded_unit_counts.append(len(units))
        else:
            unbounded_input_lengths.append(len(text))
        return units

    monkeypatch.setattr(IncrementalTranscript, "_find_anchor", observed_find_anchor)
    monkeypatch.setattr(
        incremental_transcript_module,
        "_comparison_units",
        observed_comparison_units,
    )
    transcript = IncrementalTranscript(
        max_revision_chars=max_revision_chars,
        match_window_units=match_window_units,
    )

    executed_updates = 0
    for ordinal in range(expected_updates):
        if ordinal % 200 == 0:
            hypothesis = "x" * 400
        else:
            hypothesis = "bravo" if ordinal % 2 else "cider"
        update = transcript.update(hypothesis)
        executed_updates += 1
        assert len(update.provisional_text) <= max_revision_chars
        assert len(transcript._committed_tail) <= max_revision_chars

    final_snapshot = transcript.snapshot(include_provisional=True)
    assert len(final_snapshot) > 10_000
    assert len(transcript._committed_chunks) > 2_000
    assert previous_lengths and max(previous_lengths) <= max_revision_chars
    assert bounded_unit_counts and max(bounded_unit_counts) <= match_window_units
    assert unbounded_input_lengths
    assert max(unbounded_input_lengths) <= max_revision_chars
    assert executed_updates == expected_updates
    assert executed_updates * update_interval_seconds == logical_seconds


def test_three_sequential_logical_10m_recorders_leave_no_writer_threads(
    tmp_path: Path,
) -> None:
    created_threads: list[threading.Thread] = []
    for session_ordinal in range(3):
        writer = _CountingArchiveWriter(_RECORDER_BLOCK_SAMPLES)
        recorder = MeetingRecorder(
            tmp_path / f"session-{session_ordinal}.wav",
            _RECORDER_SAMPLE_RATE,
            max_buffer_seconds=32,
            _writer_factory=lambda _path, _rate, writer=writer: writer,
        )
        assert recorder.accepted_sample_count == recorder.buffered_sample_count == 0
        recorder.start()
        assert recorder._writer_thread is not None
        created_threads.append(recorder._writer_thread)

        peak = _feed_recorder(
            recorder,
            writer,
            block_count=600,
            block_samples=_RECORDER_BLOCK_SAMPLES,
        )
        result = recorder.stop()
        assert peak <= recorder.max_buffered_samples
        assert recorder.buffered_sample_count == 0
        assert recorder.accepted_sample_count == 60_000
        assert writer.written_samples == result.sample_count == 60_000
        assert result.published and writer.finalized
        assert not recorder._writer_thread.is_alive()

    live_thread_ids = {thread.ident for thread in threading.enumerate()}
    assert all(not thread.is_alive() for thread in created_threads)
    assert all(thread.ident not in live_thread_ids for thread in created_threads)
