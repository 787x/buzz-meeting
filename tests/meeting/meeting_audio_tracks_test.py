from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile

from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracks,
    MeetingAudioTracksOutcome,
    MeetingAudioTracksStartError,
    MeetingAudioTracksState,
    MeetingAudioTracksStateError,
    MeetingAudioTracksStopError,
    MeetingTrackErrorStage,
    MeetingTrackRecordingResult,
    MeetingTrackRole,
    _TimingAudioSource,
    _TrackTimingTracker,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderOperationalError,
    MeetingRecorderState,
    MeetingRecordingResult,
)


class ControlledAudioSource(AudioSource):
    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self.on_audio: Optional[AudioFrameCallback] = None
        self.on_error: Optional[AudioErrorCallback] = None
        self.start_error: Optional[Exception] = None
        self.start_action: Optional[Callable[[AudioFrameCallback], None]] = None
        self.stop_errors: list[Exception] = []
        self.pause_start = False
        self.pause_stop = False
        self.start_entered = threading.Event()
        self.stop_entered = threading.Event()
        self.allow_start = threading.Event()
        self.allow_stop = threading.Event()
        self.active = False
        self.start_count = 0
        self.stop_attempt_count = 0
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
            self.on_audio = on_audio
            self.on_error = on_error
            self.active = True
            self.start_count += 1
        self.start_entered.set()
        if self.start_action is not None:
            self.start_action(on_audio)
        if self.pause_start:
            assert self.allow_start.wait(timeout=5)
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stop_entered.set()
        if self.pause_stop:
            assert self.allow_stop.wait(timeout=5)
        with self._lock:
            self.stop_attempt_count += 1
            if self.stop_errors:
                raise self.stop_errors.pop(0)
            if self.active:
                self.active = False
                self.stop_count += 1

    def deliver(self, samples: np.ndarray) -> None:
        with self._lock:
            callback = self.on_audio
            active = self.active
        assert active and callback is not None
        callback(samples)

    def fail(self, error: Exception) -> None:
        with self._lock:
            callback = self.on_error
        assert callback is not None
        callback(error)


class FakeRecorder:
    def __init__(
        self,
        output_path: Path,
        sample_rate: int,
        on_error: Callable[[Exception], None],
    ) -> None:
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.on_error = on_error
        self.blocks: list[np.ndarray] = []
        self.start_error: Optional[Exception] = None
        self.fail_enqueue = False
        self.pause_enqueue = False
        self.pause_stop = False
        self.enqueue_entered = threading.Event()
        self.allow_enqueue = threading.Event()
        self.stop_entered = threading.Event()
        self.allow_stop = threading.Event()
        self.request_stop_count = 0
        self.stop_count = 0
        self.cancel_count = 0
        self.pause_accepted_read_number: Optional[int] = None
        self.accepted_read_count = 0
        self.accepted_read_entered = threading.Event()
        self.allow_accepted_read = threading.Event()
        self._state = MeetingRecorderState.CREATED
        self._result: Optional[MeetingRecordingResult] = None
        self._lock = threading.Lock()
        self._accepted_read_lock = threading.Lock()

    @property
    def state(self) -> MeetingRecorderState:
        with self._lock:
            return self._state

    @property
    def accepted_sample_count(self) -> int:
        with self._accepted_read_lock:
            self.accepted_read_count += 1
            should_pause = self.pause_accepted_read_number == self.accepted_read_count
        if should_pause:
            self.accepted_read_entered.set()
            assert self.allow_accepted_read.wait(timeout=5)
        with self._lock:
            return sum(block.size for block in self.blocks)

    def start(self) -> None:
        if self.start_error is not None:
            with self._lock:
                self._state = MeetingRecorderState.FAILED
            self.on_error(self.start_error)
            raise self.start_error
        with self._lock:
            self._state = MeetingRecorderState.RUNNING

    def enqueue(self, samples: np.ndarray) -> bool:
        self.enqueue_entered.set()
        if self.pause_enqueue:
            assert self.allow_enqueue.wait(timeout=5)
        if self.fail_enqueue:
            error = MeetingRecorderOperationalError("recorder failed")
            with self._lock:
                self._state = MeetingRecorderState.FAILED
            self.on_error(error)
            return False
        with self._lock:
            if self._state != MeetingRecorderState.RUNNING:
                return False
            self.blocks.append(samples.copy())
        return True

    def request_stop(self) -> None:
        self.request_stop_count += 1

    def stop(self) -> MeetingRecordingResult:
        self.stop_entered.set()
        if self.pause_stop:
            assert self.allow_stop.wait(timeout=5)
        with self._lock:
            if self._result is not None:
                return self._result
            self.stop_count += 1
            sample_count = sum(block.size for block in self.blocks)
            if self._state == MeetingRecorderState.CREATED:
                result_state = MeetingRecorderState.CREATED
                published = False
            elif self._state == MeetingRecorderState.FAILED:
                result_state = MeetingRecorderState.FAILED
                published = False
            else:
                result_state = MeetingRecorderState.STOPPED
                published = True
            self._state = result_state
            self._result = MeetingRecordingResult(
                output_path=self.output_path,
                sample_rate=self.sample_rate,
                sample_count=sample_count,
                duration_seconds=sample_count / self.sample_rate,
                state=result_state,
                error=(
                    MeetingRecorderOperationalError("recorder failed")
                    if result_state == MeetingRecorderState.FAILED
                    else None
                ),
                published=published,
            )
            return self._result

    def cancel_empty_start(self) -> MeetingRecordingResult:
        self.cancel_count += 1
        with self._lock:
            self._state = MeetingRecorderState.STOPPED
            self._result = MeetingRecordingResult(
                output_path=self.output_path,
                sample_rate=self.sample_rate,
                sample_count=0,
                duration_seconds=0.0,
                state=MeetingRecorderState.STOPPED,
                error=None,
                published=False,
            )
            return self._result


class FakeRecorderFactory:
    def __init__(self) -> None:
        self.recorders: dict[str, FakeRecorder] = {}
        self.max_buffer_seconds: list[float] = []

    def __call__(
        self,
        output_path: str | Path,
        sample_rate: int,
        *,
        max_buffer_seconds: float,
        on_error: Callable[[Exception], None],
    ) -> FakeRecorder:
        assert max_buffer_seconds > 0
        self.max_buffer_seconds.append(max_buffer_seconds)
        recorder = FakeRecorder(Path(output_path), sample_rate, on_error)
        self.recorders[Path(output_path).name] = recorder
        return recorder


class EventGatedArchiveWriter:
    def __init__(self, *, fail_on_write: Optional[int] = None) -> None:
        self.fail_on_write = fail_on_write
        self.write_count = 0
        self.blocks: list[np.ndarray] = []
        self.failure_entered = threading.Event()
        self.allow_failure = threading.Event()
        self._condition = threading.Condition()

    def write(self, pcm16: np.ndarray) -> None:
        with self._condition:
            self.write_count += 1
            write_number = self.write_count
        if write_number == self.fail_on_write:
            self.failure_entered.set()
            assert self.allow_failure.wait(timeout=5)
            raise OSError(f"write {write_number} failed")
        with self._condition:
            self.blocks.append(pcm16.copy())
            self._condition.notify_all()

    def flush(self) -> None:
        pass

    def finalize(self) -> None:
        pass

    def publish(self) -> None:
        pass

    def discard(self) -> None:
        pass

    def close_after_failure(self) -> None:
        pass

    def wait_for_written_blocks(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: len(self.blocks) >= count,
                timeout=5,
            )


class RealRecorderFactory:
    def __init__(self, writers: dict[str, EventGatedArchiveWriter]) -> None:
        self.writers = writers
        self.recorders: dict[str, MeetingRecorder] = {}

    def __call__(
        self,
        output_path: str | Path,
        sample_rate: int,
        *,
        max_buffer_seconds: float,
        on_error: Callable[[Exception], None],
    ) -> MeetingRecorder:
        output_path = Path(output_path)
        writer = self.writers[output_path.name]
        recorder = MeetingRecorder(
            output_path,
            sample_rate,
            max_buffer_seconds=max_buffer_seconds,
            on_error=on_error,
            _writer_factory=lambda _path, _rate: writer,
        )
        self.recorders[output_path.name] = recorder
        return recorder


class FailingSecondThreadFactory:
    def __init__(self) -> None:
        self.thread_count = 0

    def __call__(self, **kwargs) -> threading.Thread:
        self.thread_count += 1
        thread = threading.Thread(**kwargs)
        if self.thread_count != 2:
            return thread

        def fail_start() -> None:
            raise RuntimeError("temporary thread start failed")

        thread.start = fail_start  # type: ignore[method-assign]
        return thread


class FailingStopThreadFactory:
    def __init__(self, failing_roles: set[MeetingTrackRole]) -> None:
        self.failing_roles = failing_roles

    def __call__(self, **kwargs) -> threading.Thread:
        thread = threading.Thread(**kwargs)
        name = kwargs["name"]
        role = (
            MeetingTrackRole.MICROPHONE
            if name.endswith("microphone")
            else MeetingTrackRole.REMOTE
        )
        if name.startswith("meeting-track-stop") and role in self.failing_roles:

            def fail_start() -> None:
                raise RuntimeError(f"{role.name.lower()} stop worker failed")

            thread.start = fail_start  # type: ignore[method-assign]
        return thread


class SequenceClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return next(self._values)


def _make_tracks(
    tmp_path: Path,
    microphone: Optional[ControlledAudioSource] = None,
    remote: Optional[ControlledAudioSource] = None,
    *,
    recorder_factory: Optional[FakeRecorderFactory] = None,
    **kwargs,
) -> tuple[
    MeetingAudioTracks,
    ControlledAudioSource,
    ControlledAudioSource,
    Optional[FakeRecorderFactory],
]:
    microphone = microphone or ControlledAudioSource(48_000)
    remote = remote or ControlledAudioSource(16_000)
    constructor_kwargs = dict(kwargs)
    if recorder_factory is not None:
        constructor_kwargs["_recorder_factory"] = recorder_factory
    tracks = MeetingAudioTracks(
        microphone,
        remote,
        tmp_path / "microphone.wav",
        tmp_path / "remote.wav",
        **constructor_kwargs,
    )
    return tracks, microphone, remote, recorder_factory


def _run_in_thread(operation: Callable[[], object]):
    result: list[object] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            result.append(operation())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result, errors


def test_rejects_same_source_invalid_rates_and_equivalent_paths(tmp_path: Path):
    source = ControlledAudioSource(48_000)
    with pytest.raises(ValueError, match="sources must be distinct"):
        MeetingAudioTracks(
            source,
            source,
            tmp_path / "mic.wav",
            tmp_path / "remote.wav",
        )

    invalid = ControlledAudioSource(0)
    with pytest.raises(ValueError, match="sample_rate"):
        MeetingAudioTracks(
            invalid,
            ControlledAudioSource(16_000),
            tmp_path / "mic.wav",
            tmp_path / "remote.wav",
        )

    with pytest.raises(ValueError, match="output paths must be distinct"):
        MeetingAudioTracks(
            ControlledAudioSource(48_000),
            ControlledAudioSource(16_000),
            tmp_path / "same.wav",
            tmp_path / "missing" / ".." / "same.wav",
        )


def test_preserves_exact_independent_pcm_and_source_sample_rates(tmp_path: Path):
    tracks, microphone, remote, _ = _make_tracks(tmp_path)
    microphone_pcm = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float32)
    remote_pcm = np.array([0.6, 0.2, -0.2], dtype=np.float32)

    tracks.start()
    microphone.deliver(microphone_pcm)
    remote.deliver(remote_pcm)
    result = tracks.stop()

    microphone_file, microphone_rate = soundfile.read(
        tmp_path / "microphone.wav", dtype="float32"
    )
    remote_file, remote_rate = soundfile.read(tmp_path / "remote.wav", dtype="float32")
    assert result.outcome == MeetingAudioTracksOutcome.COMPLETE
    assert result.microphone.complete
    assert result.remote.complete
    assert microphone_rate == 48_000
    assert remote_rate == 16_000
    with soundfile.SoundFile(tmp_path / "microphone.wav") as microphone_archive:
        assert microphone_archive.format == "RF64"
        assert microphone_archive.subtype == "PCM_16"
        assert microphone_archive.channels == 1
    with soundfile.SoundFile(tmp_path / "remote.wav") as remote_archive:
        assert remote_archive.format == "RF64"
        assert remote_archive.subtype == "PCM_16"
        assert remote_archive.channels == 1
    np.testing.assert_allclose(microphone_file, microphone_pcm, atol=1 / 32_768)
    np.testing.assert_allclose(remote_file, remote_pcm, atol=1 / 32_768)
    assert not np.array_equal(microphone_file[:3], remote_file)


def test_both_sources_cross_the_common_start_gate_concurrently(tmp_path: Path):
    microphone = ControlledAudioSource(48_000)
    remote = ControlledAudioSource(16_000)
    microphone.pause_start = True
    remote.pause_start = True
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(
        tmp_path, microphone, remote, recorder_factory=factory
    )

    start_thread, _, errors = _run_in_thread(tracks.start)
    assert microphone.start_entered.wait(timeout=5)
    assert remote.start_entered.wait(timeout=5)
    assert tracks.state == MeetingAudioTracksState.STARTING
    microphone.allow_start.set()
    remote.allow_start.set()
    start_thread.join(timeout=5)

    assert not start_thread.is_alive()
    assert not errors
    assert tracks.state == MeetingAudioTracksState.RUNNING
    tracks.stop()


@pytest.mark.parametrize("failing_role", list(MeetingTrackRole))
def test_one_source_start_failure_fails_both_and_cleans_both(
    tmp_path: Path, failing_role: MeetingTrackRole
):
    microphone = ControlledAudioSource(48_000)
    remote = ControlledAudioSource(16_000)
    failing = microphone if failing_role == MeetingTrackRole.MICROPHONE else remote
    failing.start_error = AudioSourceError("start failed")
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(
        tmp_path, microphone, remote, recorder_factory=factory
    )

    with pytest.raises(MeetingAudioTracksStartError) as caught:
        tracks.start()

    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert microphone.stop_attempt_count == 1
    assert remote.stop_attempt_count == 1
    assert not microphone.active
    assert not remote.active
    assert caught.value.result.outcome == MeetingAudioTracksOutcome.FAILED
    assert any(
        error.role == failing_role and error.stage == MeetingTrackErrorStage.START
        for error in caught.value.errors
    )


def test_start_failure_with_unresolved_cleanup_remains_failed(tmp_path: Path):
    microphone = ControlledAudioSource(48_000)
    remote = ControlledAudioSource(16_000)
    microphone.start_error = AudioSourceError("start failed")
    microphone.stop_errors.extend(
        [OSError("first cleanup failed"), OSError("second cleanup failed")]
    )
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(
        tmp_path, microphone, remote, recorder_factory=factory
    )

    with pytest.raises(MeetingAudioTracksStartError):
        tracks.start()

    assert tracks.state == MeetingAudioTracksState.FAILED
    assert microphone.active
    tracks.stop()
    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert not microphone.active


def test_recorder_start_failure_is_aggregated_and_other_track_is_cleaned(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    factory.recorders["microphone.wav"].start_error = MeetingRecorderOperationalError(
        "writer open failed"
    )

    with pytest.raises(MeetingAudioTracksStartError) as caught:
        tracks.start()

    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert microphone.stop_attempt_count == 0
    assert remote.stop_attempt_count == 1
    assert {error.stage for error in caught.value.errors} >= {
        MeetingTrackErrorStage.START,
        MeetingTrackErrorStage.RECORDER,
    }


def test_source_callback_then_start_failure_preserves_accepted_prefix(tmp_path: Path):
    microphone = ControlledAudioSource(48_000)
    remote = ControlledAudioSource(16_000)
    prefix = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    microphone.start_action = lambda callback: callback(prefix)
    microphone.start_error = AudioSourceError("failed after callback")
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(
        tmp_path, microphone, remote, recorder_factory=factory
    )

    with pytest.raises(MeetingAudioTracksStartError) as caught:
        tracks.start()

    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert caught.value.result.outcome == MeetingAudioTracksOutcome.PARTIAL
    assert caught.value.result.microphone.recording.published
    assert caught.value.result.microphone.recording.sample_count == prefix.size
    assert caught.value.result.microphone.timing.anchors[0].sample_end == prefix.size


def test_second_temporary_start_thread_failure_releases_first_worker(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,
        _thread_factory=FailingSecondThreadFactory(),
    )

    with pytest.raises(MeetingAudioTracksStartError) as caught:
        tracks.start()

    assert microphone.start_count == 0
    assert remote.start_count == 0
    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert any(
        error.role == MeetingTrackRole.REMOTE
        and error.stage == MeetingTrackErrorStage.START
        for error in caught.value.errors
    )


def test_stop_during_start_cancels_gate_and_prevents_late_source_start(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    original_factory = tracks._thread_factory
    first_worker_created = threading.Event()
    release_second_creation = threading.Event()
    calls = 0

    def pausing_thread_factory(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            first_worker_created.set()
            assert release_second_creation.wait(timeout=5)
        return original_factory(**kwargs)

    tracks._thread_factory = pausing_thread_factory
    start_thread, _, start_errors = _run_in_thread(tracks.start)
    assert first_worker_created.wait(timeout=5)
    stop_thread, stop_results, stop_errors = _run_in_thread(tracks.stop)
    release_second_creation.set()
    start_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert len(start_errors) == 1
    assert isinstance(start_errors[0], MeetingAudioTracksStartError)
    assert not stop_errors
    assert len(stop_results) == 1
    assert microphone.start_count == 0
    assert remote.start_count == 0
    assert tracks.state == MeetingAudioTracksState.STOPPED
    cancellation_roles = {
        error.role
        for error in start_errors[0].result.errors
        if error.stage == MeetingTrackErrorStage.START
    }
    assert cancellation_roles == set(MeetingTrackRole)


@pytest.mark.parametrize("failing_role", list(MeetingTrackRole))
def test_runtime_source_failure_degrades_but_other_track_continues(
    tmp_path: Path, failing_role: MeetingTrackRole
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone.deliver(np.ones(3, dtype=np.float32))
    remote.deliver(np.ones(4, dtype=np.float32))
    failing = microphone if failing_role == MeetingTrackRole.MICROPHONE else remote
    surviving = remote if failing is microphone else microphone

    failing.fail(AudioSourceError("runtime failed"))
    assert tracks.state == MeetingAudioTracksState.DEGRADED
    surviving.deliver(np.ones(2, dtype=np.float32))
    result = tracks.stop()

    failed_result = (
        result.microphone
        if failing_role == MeetingTrackRole.MICROPHONE
        else result.remote
    )
    surviving_result = (
        result.remote
        if failing_role == MeetingTrackRole.MICROPHONE
        else result.microphone
    )
    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert result.outcome == MeetingAudioTracksOutcome.PARTIAL
    assert not failed_result.complete
    assert failed_result.recording.published
    assert surviving_result.complete
    assert any(
        error.role == failing_role
        and error.stage == MeetingTrackErrorStage.SOURCE_RUNTIME
        for error in result.errors
    )


def test_recorder_failure_rejects_timing_anchor_and_other_track_continues(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    clock = SequenceClock([1_000, 1_100, 1_200, 1_300])
    tracks, microphone, remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,
        _clock_ns=clock,
    )
    tracks.start()
    microphone.deliver(np.ones(3, dtype=np.float32))
    factory.recorders["microphone.wav"].fail_enqueue = True
    microphone.deliver(np.ones(5, dtype=np.float32))
    remote.deliver(np.ones(4, dtype=np.float32))

    assert tracks.state == MeetingAudioTracksState.DEGRADED
    result = tracks.stop()

    assert result.outcome == MeetingAudioTracksOutcome.PARTIAL
    assert result.microphone.timing.anchors == (result.microphone.timing.anchors[0],)
    assert result.microphone.timing.anchors[0].sample_end == 3
    assert result.remote.timing.anchors[0].sample_end == 4
    assert any(
        error.role == MeetingTrackRole.MICROPHONE
        and error.stage == MeetingTrackErrorStage.RECORDER
        for error in result.errors
    )


def test_async_writer_failure_clips_final_timing_to_zero_durable_samples(
    tmp_path: Path,
):
    microphone_writer = EventGatedArchiveWriter(fail_on_write=1)
    remote_writer = EventGatedArchiveWriter()
    factory = RealRecorderFactory(
        {
            "microphone.wav": microphone_writer,
            "remote.wav": remote_writer,
        }
    )
    tracks, microphone, _remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,  # type: ignore[arg-type]
    )
    block = np.ones(32, dtype=np.float32)

    tracks.start()
    microphone.deliver(block)
    assert factory.recorders["microphone.wav"].accepted_sample_count == block.size
    assert microphone_writer.failure_entered.wait(timeout=5)
    runtime_timing = tracks._tracks[MeetingTrackRole.MICROPHONE].timing.snapshot()
    assert runtime_timing.anchors[-1].sample_end == block.size
    microphone_writer.allow_failure.set()
    result = tracks.stop()

    recording = result.microphone.recording
    assert recording.state == MeetingRecorderState.FAILED
    assert not recording.published
    assert recording.sample_count == 0
    assert result.microphone.timing.anchors == ()


def test_async_writer_failure_clips_timing_to_durable_prefix(tmp_path: Path):
    microphone_writer = EventGatedArchiveWriter(fail_on_write=2)
    remote_writer = EventGatedArchiveWriter()
    factory = RealRecorderFactory(
        {
            "microphone.wav": microphone_writer,
            "remote.wav": remote_writer,
        }
    )
    tracks, microphone, _remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,  # type: ignore[arg-type]
    )
    first = np.ones(12, dtype=np.float32)
    rejected_tail = np.ones(20, dtype=np.float32)

    tracks.start()
    microphone.deliver(first)
    microphone_writer.wait_for_written_blocks(1)
    microphone.deliver(rejected_tail)
    assert microphone_writer.failure_entered.wait(timeout=5)
    runtime_timing = tracks._tracks[MeetingTrackRole.MICROPHONE].timing.snapshot()
    assert runtime_timing.anchors[-1].sample_end == first.size + rejected_tail.size
    microphone_writer.allow_failure.set()
    result = tracks.stop()

    recording = result.microphone.recording
    assert recording.state == MeetingRecorderState.FAILED
    assert not recording.published
    assert recording.sample_count == first.size
    assert tuple(anchor.sample_end for anchor in result.microphone.timing.anchors) == (
        first.size,
    )
    assert all(
        0 < anchor.sample_end <= recording.sample_count
        for anchor in result.microphone.timing.anchors
    )


def test_timing_offsets_are_relative_to_common_epoch_and_not_clipped(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    clock = SequenceClock([1_000, 900, 1_250])
    tracks, microphone, remote, _ = _make_tracks(
        tmp_path, recorder_factory=factory, _clock_ns=clock
    )

    tracks.start()
    microphone.deliver(np.ones(2, dtype=np.float32))
    remote.deliver(np.ones(3, dtype=np.float32))
    result = tracks.stop()

    assert result.coordinator_start_monotonic_ns == 1_000
    assert result.microphone.timing.timing_basis == "host_callback_arrival"
    assert result.microphone.timing.anchors[0].callback_arrival_offset_ns == -100
    assert result.remote.timing.anchors[0].callback_arrival_offset_ns == 250


def test_anchor_storage_is_bounded_and_retains_first_and_latest():
    tracker = _TrackTimingTracker(sample_rate=1)
    tracker.set_coordinator_start(0)
    observations = 4_200
    for index in range(observations):
        sample_end = 1 + index * 10
        tracker.observe(sample_end, sample_end * 1_000)

    timing = tracker.snapshot()

    assert len(timing.anchors) <= 4_096
    assert timing.anchors[0].sample_end == 1
    assert timing.anchors[-1].sample_end == 1 + (observations - 1) * 10


def test_synthetic_anchors_represent_80_ppm_relative_drift_over_one_hour():
    hour_seconds = 3_600
    microphone_rate = 48_000
    remote_rate = 16_000
    microphone = _TrackTimingTracker(microphone_rate)
    remote = _TrackTimingTracker(remote_rate)
    for tracker in (microphone, remote):
        tracker.set_coordinator_start(0)
    first_offset_ns = 1_000_000_000
    final_offset_ns = (hour_seconds + 1) * 1_000_000_000
    microphone.observe(microphone_rate, first_offset_ns)
    remote.observe(remote_rate, first_offset_ns)
    microphone.observe(
        microphone_rate + round(microphone_rate * hour_seconds * (1 + 50e-6)),
        final_offset_ns,
    )
    remote.observe(
        remote_rate + round(remote_rate * hour_seconds * (1 - 30e-6)),
        final_offset_ns,
    )

    microphone_timing = microphone.snapshot()
    remote_timing = remote.snapshot()

    def track_metadata(
        role: MeetingTrackRole,
        timing,
        sample_rate: int,
    ) -> MeetingTrackRecordingResult:
        sample_count = timing.anchors[-1].sample_end
        return MeetingTrackRecordingResult(
            role=role,
            recording=MeetingRecordingResult(
                output_path=Path(f"{role.name.lower()}.wav"),
                sample_rate=sample_rate,
                sample_count=sample_count,
                duration_seconds=sample_count / sample_rate,
                state=MeetingRecorderState.STOPPED,
                error=None,
                published=True,
            ),
            timing=timing,
            errors=(),
            complete=True,
        )

    microphone_metadata = track_metadata(
        MeetingTrackRole.MICROPHONE, microphone_timing, microphone_rate
    )
    remote_metadata = track_metadata(
        MeetingTrackRole.REMOTE, remote_timing, remote_rate
    )

    def effective_rate_ratio(track: MeetingTrackRecordingResult) -> float:
        first, latest = track.timing.anchors[0], track.timing.anchors[-1]
        elapsed_seconds = (
            latest.callback_arrival_offset_ns - first.callback_arrival_offset_ns
        ) / 1_000_000_000
        observed_rate = (latest.sample_end - first.sample_end) / elapsed_seconds
        return observed_rate / track.recording.sample_rate

    microphone_ratio = effective_rate_ratio(microphone_metadata)
    remote_ratio = effective_rate_ratio(remote_metadata)
    relative_ppm = (microphone_ratio - remote_ratio) * 1_000_000
    observed_seconds = (
        microphone_metadata.timing.anchors[-1].callback_arrival_offset_ns
        - microphone_metadata.timing.anchors[0].callback_arrival_offset_ns
    ) / 1_000_000_000
    relative_offset_seconds = (microphone_ratio - remote_ratio) * observed_seconds

    assert relative_ppm == pytest.approx(80)
    assert relative_offset_seconds == pytest.approx(0.288)
    assert all(anchor.sample_end > 0 for anchor in microphone_timing.anchors)
    assert all(anchor.sample_end > 0 for anchor in remote_timing.anchors)


def test_stop_attempts_both_tracks_in_parallel(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone.pause_stop = True
    remote.pause_stop = True

    stop_thread, results, errors = _run_in_thread(tracks.stop)
    assert microphone.stop_entered.wait(timeout=5)
    assert remote.stop_entered.wait(timeout=5)
    microphone.allow_stop.set()
    remote.allow_stop.set()
    stop_thread.join(timeout=5)

    assert not stop_thread.is_alive()
    assert not errors
    assert results[0].outcome == MeetingAudioTracksOutcome.COMPLETE


@pytest.mark.parametrize("fallback_role", list(MeetingTrackRole))
def test_stop_worker_start_failure_dispatches_other_cleanup_before_fallback(
    tmp_path: Path, fallback_role: MeetingTrackRole
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,
        _thread_factory=FailingStopThreadFactory({fallback_role}),
    )
    tracks.start()
    fallback_source = (
        microphone if fallback_role == MeetingTrackRole.MICROPHONE else remote
    )
    healthy_role = (
        MeetingTrackRole.REMOTE
        if fallback_role == MeetingTrackRole.MICROPHONE
        else MeetingTrackRole.MICROPHONE
    )
    healthy_recorder = factory.recorders[
        "remote.wav" if healthy_role == MeetingTrackRole.REMOTE else "microphone.wav"
    ]
    fallback_source.pause_stop = True

    stop_thread, results, errors = _run_in_thread(tracks.stop)
    assert fallback_source.stop_entered.wait(timeout=5)
    assert healthy_recorder.stop_entered.wait(timeout=5)
    assert healthy_recorder.stop_count == 1
    assert stop_thread.is_alive()
    fallback_source.allow_stop.set()
    stop_thread.join(timeout=5)

    assert not stop_thread.is_alive()
    assert not errors
    assert len(results) == 1


def test_both_stop_worker_start_failures_fall_back_and_clean_both(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(
        tmp_path,
        recorder_factory=factory,
        _thread_factory=FailingStopThreadFactory(set(MeetingTrackRole)),
    )
    tracks.start()

    result = tracks.stop()

    assert result.outcome == MeetingAudioTracksOutcome.COMPLETE
    assert microphone.stop_attempt_count == 1
    assert remote.stop_attempt_count == 1
    assert tracks.state == MeetingAudioTracksState.STOPPED


def test_one_blocked_writer_does_not_delay_other_track_cleanup(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone_recorder = factory.recorders["microphone.wav"]
    remote_recorder = factory.recorders["remote.wav"]
    microphone_recorder.pause_stop = True

    stop_thread, results, errors = _run_in_thread(tracks.stop)
    assert microphone_recorder.stop_entered.wait(timeout=5)
    assert remote_recorder.stop_entered.wait(timeout=5)
    assert remote_recorder.stop_count == 1
    assert stop_thread.is_alive()
    microphone_recorder.allow_stop.set()
    stop_thread.join(timeout=5)

    assert not errors
    assert len(results) == 1


def test_stop_waits_for_in_flight_callback_but_cleans_other_track(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, _remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone_recorder = factory.recorders["microphone.wav"]
    remote_recorder = factory.recorders["remote.wav"]
    microphone_recorder.pause_enqueue = True

    callback_thread, _, callback_errors = _run_in_thread(
        lambda: microphone.deliver(np.ones(8, dtype=np.float32))
    )
    assert microphone_recorder.enqueue_entered.wait(timeout=5)
    stop_thread, stop_results, stop_errors = _run_in_thread(tracks.stop)
    assert remote_recorder.stop_entered.wait(timeout=5)
    assert remote_recorder.stop_count == 1
    assert stop_thread.is_alive()
    microphone_recorder.allow_enqueue.set()
    callback_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not callback_errors and not stop_errors
    assert stop_results[0].microphone.recording.sample_count == 8


def test_stop_waits_for_timing_observation_after_fanout_callback_returns(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    tracks, microphone, _remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone_recorder = factory.recorders["microphone.wav"]
    microphone_recorder.pause_accepted_read_number = 2

    callback_thread, _, callback_errors = _run_in_thread(
        lambda: microphone.deliver(np.ones(9, dtype=np.float32))
    )
    assert microphone_recorder.accepted_read_entered.wait(timeout=5)
    stop_thread, stop_results, stop_errors = _run_in_thread(tracks.stop)
    assert microphone.stop_entered.wait(timeout=5)
    assert stop_thread.is_alive()
    microphone_recorder.allow_accepted_read.set()
    callback_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not callback_errors and not stop_errors
    assert not callback_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_results[0].microphone.timing.anchors[0].sample_end == 9


def test_timing_source_stop_from_own_callback_does_not_self_wait(tmp_path: Path):
    source = ControlledAudioSource(1_000)
    recorder = FakeRecorder(tmp_path / "self-stop.wav", 1_000, lambda _error: None)
    tracker = _TrackTimingTracker(1_000)
    tracker.set_coordinator_start(0)
    timed_source = _TimingAudioSource(source, recorder, tracker, lambda: 1)
    source.start_action = lambda callback: callback(np.ones(1, dtype=np.float32))

    start_thread, _, errors = _run_in_thread(
        lambda: timed_source.start(lambda _samples: timed_source.stop())
    )
    start_thread.join(timeout=5)

    assert not start_thread.is_alive()
    assert not errors
    assert not source.active


def test_timing_source_waits_for_observation_when_underlying_stop_raises(
    tmp_path: Path,
):
    source = ControlledAudioSource(1_000)
    recorder = FakeRecorder(tmp_path / "stop-error.wav", 1_000, lambda _error: None)
    recorder.start()
    recorder.pause_accepted_read_number = 2
    tracker = _TrackTimingTracker(1_000)
    tracker.set_coordinator_start(0)
    timed_source = _TimingAudioSource(source, recorder, tracker, lambda: 1)
    timed_source.start(recorder.enqueue)
    source.stop_errors.append(OSError("underlying stop failed"))

    callback_thread, _, callback_errors = _run_in_thread(
        lambda: source.deliver(np.ones(4, dtype=np.float32))
    )
    assert recorder.accepted_read_entered.wait(timeout=5)
    stop_thread, _, stop_errors = _run_in_thread(timed_source.stop)
    assert source.stop_entered.wait(timeout=5)
    assert stop_thread.is_alive()
    recorder.allow_accepted_read.set()
    callback_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not callback_errors
    assert len(stop_errors) == 1
    assert isinstance(stop_errors[0], OSError)
    source.stop()


def test_both_stop_failures_are_aggregated_and_retry_cleans_both(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone.stop_errors.append(OSError("microphone stop failed"))
    remote.stop_errors.append(OSError("remote stop failed"))

    with pytest.raises(MeetingAudioTracksStopError) as caught:
        tracks.stop()

    assert tracks.state == MeetingAudioTracksState.FAILED
    assert {error.role for error in caught.value.errors} == set(MeetingTrackRole)
    result = tracks.stop()
    assert tracks.state == MeetingAudioTracksState.STOPPED
    assert result.outcome == MeetingAudioTracksOutcome.COMPLETE
    assert microphone.stop_attempt_count == 2
    assert remote.stop_attempt_count == 2


def test_repeated_cleanup_errors_are_bounded_per_role_and_stage(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, _remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone.stop_errors.extend(
        OSError(f"microphone cleanup failure {index}") for index in range(100)
    )

    for _ in range(100):
        with pytest.raises(MeetingAudioTracksStopError):
            tracks.stop()

    result = tracks.stop()
    microphone_stop_errors = tuple(
        error
        for error in result.microphone.errors
        if error.stage == MeetingTrackErrorStage.STOP
    )

    assert len(microphone_stop_errors) == 2
    assert "failure 0" in str(microphone_stop_errors[0].exception)
    assert "failure 99" in str(microphone_stop_errors[-1].exception)
    assert result.remote.errors == ()
    assert microphone.stop_attempt_count == 101


def test_duplicate_concurrent_stop_shares_one_attempt(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    microphone.pause_stop = True
    remote.pause_stop = True

    first, first_results, first_errors = _run_in_thread(tracks.stop)
    assert microphone.stop_entered.wait(timeout=5)
    assert remote.stop_entered.wait(timeout=5)
    second, second_results, second_errors = _run_in_thread(tracks.stop)
    microphone.allow_stop.set()
    remote.allow_stop.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first_errors and not second_errors
    assert first_results[0] is second_results[0]
    assert tracks.stop() is first_results[0]
    assert microphone.stop_attempt_count == 1
    assert remote.stop_attempt_count == 1


def test_runtime_error_during_concurrent_callbacks_keeps_roles_separate(
    tmp_path: Path,
):
    factory = FakeRecorderFactory()
    tracks, microphone, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    barrier = threading.Barrier(3)

    def deliver(source: ControlledAudioSource, value: float) -> None:
        barrier.wait()
        source.deliver(np.full(10, value, dtype=np.float32))

    mic_thread = threading.Thread(target=deliver, args=(microphone, 0.25))
    remote_thread = threading.Thread(target=deliver, args=(remote, -0.25))
    mic_thread.start()
    remote_thread.start()
    barrier.wait()
    mic_thread.join(timeout=5)
    remote_thread.join(timeout=5)
    microphone.fail(AudioSourceError("microphone disconnected"))
    result = tracks.stop()

    assert result.microphone.recording.sample_count == 10
    assert result.remote.recording.sample_count == 10
    assert all(
        error.role == MeetingTrackRole.MICROPHONE
        for error in result.errors
        if error.stage == MeetingTrackErrorStage.SOURCE_RUNTIME
    )


def test_duplicate_start_is_rejected(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(tmp_path, recorder_factory=factory)
    tracks.start()
    with pytest.raises(MeetingAudioTracksStateError):
        tracks.start()
    tracks.stop()


def test_buffer_bound_is_applied_independently_to_both_recorders(tmp_path: Path):
    factory = FakeRecorderFactory()
    tracks, _, _, _ = _make_tracks(
        tmp_path, recorder_factory=factory, max_buffer_seconds=7.5
    )

    assert factory.max_buffer_seconds == [7.5, 7.5]
    tracks.stop()


def test_ten_lifecycles_leave_no_coordinator_workers(tmp_path: Path):
    for index in range(10):
        factory = FakeRecorderFactory()
        tracks = MeetingAudioTracks(
            ControlledAudioSource(48_000),
            ControlledAudioSource(16_000),
            tmp_path / f"microphone-{index}.wav",
            tmp_path / f"remote-{index}.wav",
            _recorder_factory=factory,
        )
        tracks.start()
        tracks.stop()

    worker_names = {
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(("meeting-track-start", "meeting-track-stop"))
    }
    assert worker_names == set()
