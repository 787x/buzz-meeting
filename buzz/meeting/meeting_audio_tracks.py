from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from buzz.audio_capture.meeting_audio_fanout import (
    MeetingAudioFanout,
    MeetingAudioFanoutState,
)
from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderState,
    MeetingRecordingResult,
)


class MeetingTrackRole(Enum):
    MICROPHONE = auto()
    REMOTE = auto()


class MeetingAudioTracksState(Enum):
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    DEGRADED = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class MeetingAudioTracksOutcome(Enum):
    COMPLETE = auto()
    PARTIAL = auto()
    FAILED = auto()


class MeetingTrackErrorStage(Enum):
    START = auto()
    SOURCE_RUNTIME = auto()
    RECORDER = auto()
    STOP = auto()


@dataclass(frozen=True, slots=True)
class MeetingTrackTimingAnchor:
    sample_end: int
    callback_arrival_offset_ns: int


@dataclass(frozen=True, slots=True)
class MeetingTrackTiming:
    anchors: tuple[MeetingTrackTimingAnchor, ...]
    timing_basis: str = "host_callback_arrival"


@dataclass(frozen=True, slots=True)
class MeetingTrackError:
    role: MeetingTrackRole
    stage: MeetingTrackErrorStage
    exception: Exception


@dataclass(frozen=True, slots=True)
class MeetingTrackRecordingResult:
    role: MeetingTrackRole
    recording: MeetingRecordingResult
    timing: MeetingTrackTiming
    errors: tuple[MeetingTrackError, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class MeetingAudioTracksResult:
    coordinator_start_monotonic_ns: Optional[int]
    microphone: MeetingTrackRecordingResult
    remote: MeetingTrackRecordingResult
    outcome: MeetingAudioTracksOutcome
    errors: tuple[MeetingTrackError, ...]


class MeetingAudioTracksError(RuntimeError):
    """Base error for a two-track meeting capture lifecycle failure."""


class MeetingAudioTracksStateError(MeetingAudioTracksError):
    """Raised when an operation is invalid for the coordinator state."""


class MeetingAudioTracksStartError(MeetingAudioTracksError):
    def __init__(
        self,
        errors: tuple[MeetingTrackError, ...],
        result: MeetingAudioTracksResult,
    ) -> None:
        super().__init__("Could not start both requested meeting audio tracks")
        self.errors = errors
        self.result = result


class MeetingAudioTracksStopError(MeetingAudioTracksError):
    def __init__(
        self,
        errors: tuple[MeetingTrackError, ...],
        result: MeetingAudioTracksResult,
    ) -> None:
        super().__init__("Could not cleanly stop both meeting audio tracks")
        self.errors = errors
        self.result = result


class _RecorderFactory(Protocol):
    def __call__(
        self,
        output_path: os.PathLike[str] | str,
        sample_rate: int,
        *,
        max_buffer_seconds: float,
        on_error: Callable[[Exception], None],
    ) -> MeetingRecorder:
        ...


class _FanoutFactory(Protocol):
    def __call__(
        self,
        source: AudioSource,
        recorder: MeetingRecorder,
        *,
        on_source_error: AudioErrorCallback,
    ) -> MeetingAudioFanout:
        ...


class _ThreadFactory(Protocol):
    def __call__(
        self,
        *,
        target: Callable[[], None],
        name: str,
        daemon: bool,
    ) -> threading.Thread:
        ...


class _TrackTimingTracker:
    _ANCHOR_INTERVAL_SECONDS = 10
    _MAX_ANCHORS = 4096

    def __init__(self, sample_rate: int) -> None:
        self._anchor_interval_samples = sample_rate * self._ANCHOR_INTERVAL_SECONDS
        self._lock = threading.Lock()
        self._coordinator_start_ns: Optional[int] = None
        self._first: Optional[MeetingTrackTimingAnchor] = None
        self._periodic: deque[MeetingTrackTimingAnchor] = deque(
            maxlen=self._MAX_ANCHORS - 2
        )
        self._latest: Optional[MeetingTrackTimingAnchor] = None
        self._last_periodic_sample_end = 0

    def set_coordinator_start(self, coordinator_start_ns: int) -> None:
        with self._lock:
            if self._coordinator_start_ns is not None:
                raise RuntimeError("Track timing epoch was already set")
            self._coordinator_start_ns = coordinator_start_ns

    def observe(self, sample_end: int, callback_arrival_ns: int) -> None:
        with self._lock:
            if self._coordinator_start_ns is None:
                raise RuntimeError("Track timing epoch was not set before capture")
            anchor = MeetingTrackTimingAnchor(
                sample_end=sample_end,
                callback_arrival_offset_ns=(
                    callback_arrival_ns - self._coordinator_start_ns
                ),
            )
            if self._first is None:
                self._first = anchor
                self._latest = anchor
                self._last_periodic_sample_end = sample_end
                return

            self._latest = anchor
            if (
                sample_end - self._last_periodic_sample_end
                >= self._anchor_interval_samples
            ):
                self._periodic.append(anchor)
                self._last_periodic_sample_end = sample_end

    def snapshot(
        self, *, durable_sample_count: Optional[int] = None
    ) -> MeetingTrackTiming:
        with self._lock:
            candidates = [self._first, *self._periodic, self._latest]
            anchors: list[MeetingTrackTimingAnchor] = []
            for anchor in candidates:
                if anchor is None or anchor.sample_end <= 0:
                    continue
                if (
                    durable_sample_count is not None
                    and anchor.sample_end > durable_sample_count
                ):
                    continue
                if not anchors or anchor != anchors[-1]:
                    anchors.append(anchor)
            return MeetingTrackTiming(anchors=tuple(anchors))


class _TimingAudioSource(AudioSource):
    """Observe timing only when the downstream recorder accepted the block."""

    def __init__(
        self,
        source: AudioSource,
        recorder: MeetingRecorder,
        tracker: _TrackTimingTracker,
        clock_ns: Callable[[], int],
    ) -> None:
        self._source = source
        self._recorder = recorder
        self._tracker = tracker
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._in_flight = 0
        self._callback_threads: Counter[int] = Counter()

    @property
    def sample_rate(self) -> int:
        return self._source.sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        def on_timed_audio(samples: np.ndarray) -> None:
            thread_id = threading.get_ident()
            with self._condition:
                self._in_flight += 1
                self._callback_threads[thread_id] += 1
            try:
                arrival_ns = self._clock_ns()
                sample_count = int(samples.size)
                accepted_before = self._recorder.accepted_sample_count
                on_audio(samples)
                accepted_after = self._recorder.accepted_sample_count
                if (
                    sample_count > 0
                    and accepted_after == accepted_before + sample_count
                ):
                    self._tracker.observe(accepted_after, arrival_ns)
            finally:
                with self._condition:
                    self._in_flight -= 1
                    self._callback_threads[thread_id] -= 1
                    if self._callback_threads[thread_id] == 0:
                        del self._callback_threads[thread_id]
                    self._condition.notify_all()

        self._source.start(on_timed_audio, on_error)

    def stop(self) -> None:
        try:
            self._source.stop()
        finally:
            thread_id = threading.get_ident()
            with self._condition:
                own_callbacks = self._callback_threads.get(thread_id, 0)
                self._condition.wait_for(lambda: self._in_flight <= own_callbacks)


@dataclass(slots=True)
class _OwnedTrack:
    role: MeetingTrackRole
    recorder: MeetingRecorder
    fanout: MeetingAudioFanout
    timing: _TrackTimingTracker
    errors: list[MeetingTrackError] = field(default_factory=list)
    start_succeeded: bool = False
    cleanup_complete: bool = False
    recording_result: Optional[MeetingRecordingResult] = None


class MeetingAudioTracks:
    """Own exactly one microphone and one remote durable recording track.

    The coordinator deliberately does not mix, resample, transcribe, or identify
    the platform subtype of either source. Timing anchors use host callback
    arrival time and are approximate; they are not hardware capture timestamps.
    """

    _START_THREAD_PREFIX = "meeting-track-start"
    _STOP_THREAD_PREFIX = "meeting-track-stop"
    _ERRORS_PER_ROLE_STAGE_LIMIT = 2

    def __init__(
        self,
        microphone_source: AudioSource,
        remote_source: AudioSource,
        microphone_output_path: os.PathLike[str] | str,
        remote_output_path: os.PathLike[str] | str,
        *,
        max_buffer_seconds: float = 60.0,
        _clock_ns: Callable[[], int] = time.perf_counter_ns,
        _recorder_factory: _RecorderFactory = MeetingRecorder,
        _fanout_factory: _FanoutFactory = MeetingAudioFanout,
        _thread_factory: _ThreadFactory = threading.Thread,
    ) -> None:
        if microphone_source is remote_source:
            raise ValueError("Microphone and remote audio sources must be distinct")
        self._validate_sample_rate("microphone", microphone_source.sample_rate)
        self._validate_sample_rate("remote", remote_source.sample_rate)
        if self._normalized_path(microphone_output_path) == self._normalized_path(
            remote_output_path
        ):
            raise ValueError("Microphone and remote output paths must be distinct")

        self._condition = threading.Condition()
        self._state = MeetingAudioTracksState.CREATED
        self._clock_ns = _clock_ns
        self._thread_factory = _thread_factory
        self._coordinator_start_ns: Optional[int] = None
        self._stop_requested = False
        self._start_in_progress = False
        self._stop_in_progress = False
        self._cleanup_in_progress = False
        self._startup_gate: Optional[threading.Event] = None
        self._startup_cancel: Optional[threading.Event] = None
        self._startup_ready_count = 0
        self._result: Optional[MeetingAudioTracksResult] = None

        self._tracks: dict[MeetingTrackRole, _OwnedTrack] = {}
        for role, source, output_path in (
            (
                MeetingTrackRole.MICROPHONE,
                microphone_source,
                microphone_output_path,
            ),
            (MeetingTrackRole.REMOTE, remote_source, remote_output_path),
        ):
            recorder = _recorder_factory(
                output_path,
                source.sample_rate,
                max_buffer_seconds=max_buffer_seconds,
                on_error=lambda error, role=role: self._on_recorder_error(role, error),
            )
            timing = _TrackTimingTracker(source.sample_rate)
            timed_source = _TimingAudioSource(source, recorder, timing, _clock_ns)
            fanout = _fanout_factory(
                timed_source,
                recorder,
                on_source_error=lambda error, role=role: self._on_source_error(
                    role, error
                ),
            )
            self._tracks[role] = _OwnedTrack(role, recorder, fanout, timing)

    @property
    def state(self) -> MeetingAudioTracksState:
        with self._condition:
            return self._state

    def start(self) -> None:
        with self._condition:
            if self._state != MeetingAudioTracksState.CREATED:
                raise MeetingAudioTracksStateError(
                    f"Cannot start meeting audio tracks in state {self._state.name}"
                )
            self._state = MeetingAudioTracksState.STARTING
            self._start_in_progress = True
            self._stop_requested = False
            self._startup_gate = threading.Event()
            self._startup_cancel = threading.Event()
            self._startup_ready_count = 0
            gate = self._startup_gate
            cancel = self._startup_cancel

        startup_errors: dict[MeetingTrackRole, Exception] = {}
        startup_errors_lock = threading.Lock()
        started_threads: list[threading.Thread] = []

        def start_track(track: _OwnedTrack) -> None:
            with self._condition:
                self._startup_ready_count += 1
                self._condition.notify_all()
            gate.wait()
            if cancel.is_set():
                return
            try:
                track.fanout.start()
            except Exception as exc:
                with startup_errors_lock:
                    startup_errors[track.role] = exc
            else:
                track.start_succeeded = True

        for track in self._tracks_in_role_order():
            try:
                thread = self._thread_factory(
                    target=lambda track=track: start_track(track),
                    name=f"{self._START_THREAD_PREFIX}-{track.role.name.lower()}",
                    daemon=False,
                )
                thread.start()
            except Exception as exc:
                with startup_errors_lock:
                    startup_errors[track.role] = exc
                cancel.set()
                gate.set()
                with self._condition:
                    self._condition.notify_all()
                break
            started_threads.append(thread)

        if len(started_threads) == len(self._tracks):
            with self._condition:
                self._condition.wait_for(
                    lambda: self._startup_ready_count == len(self._tracks)
                    or self._stop_requested
                )
                if self._stop_requested:
                    cancel.set()
                else:
                    self._coordinator_start_ns = self._clock_ns()
                    for track in self._tracks_in_role_order():
                        track.timing.set_coordinator_start(self._coordinator_start_ns)
                gate.set()

        for thread in started_threads:
            thread.join()

        with self._condition:
            stop_requested = self._stop_requested

        for role, error in startup_errors.items():
            self._record_error(role, MeetingTrackErrorStage.START, error)

        if startup_errors or stop_requested:
            if stop_requested and not startup_errors:
                self._record_start_cancellation()
            with self._condition:
                self._cleanup_in_progress = True
            stop_errors = self._run_parallel_stop(self._tracks_in_role_order())
            with self._condition:
                self._cleanup_in_progress = False
            for role, error in stop_errors.items():
                self._record_error(role, MeetingTrackErrorStage.STOP, error)
            self._collect_recording_results()
            result = self._make_result()
            cleanup_complete = self._all_cleanup_complete()
            with self._condition:
                self._result = result
                self._state = (
                    MeetingAudioTracksState.STOPPED
                    if cleanup_complete
                    else MeetingAudioTracksState.FAILED
                )
                self._start_in_progress = False
                self._clear_startup_locked()
                self._condition.notify_all()
            errors = self._all_errors()
            raise MeetingAudioTracksStartError(errors, result)

        with self._condition:
            if self._stop_requested:
                # A stop request can arrive after workers finish but before the
                # startup outcome is committed.
                stop_requested = True
            else:
                degraded = any(
                    track.fanout.state == MeetingAudioFanoutState.FAILED
                    or any(
                        error.stage
                        in (
                            MeetingTrackErrorStage.SOURCE_RUNTIME,
                            MeetingTrackErrorStage.RECORDER,
                        )
                        for error in track.errors
                    )
                    for track in self._tracks_in_role_order()
                )
                self._state = (
                    MeetingAudioTracksState.DEGRADED
                    if degraded
                    else MeetingAudioTracksState.RUNNING
                )
                self._start_in_progress = False
                self._clear_startup_locked()
                self._condition.notify_all()
                return

        if stop_requested:
            self._record_start_cancellation()
            with self._condition:
                self._cleanup_in_progress = True
            stop_errors = self._run_parallel_stop(self._tracks_in_role_order())
            with self._condition:
                self._cleanup_in_progress = False
            for role, error in stop_errors.items():
                self._record_error(role, MeetingTrackErrorStage.STOP, error)
            self._collect_recording_results()
            result = self._make_result()
            with self._condition:
                self._result = result
                self._state = (
                    MeetingAudioTracksState.STOPPED
                    if self._all_cleanup_complete()
                    else MeetingAudioTracksState.FAILED
                )
                self._start_in_progress = False
                self._clear_startup_locked()
                self._condition.notify_all()
            raise MeetingAudioTracksStartError(self._all_errors(), result)

    def stop(self) -> MeetingAudioTracksResult:
        with self._condition:
            if self._state == MeetingAudioTracksState.STARTING:
                self._stop_requested = True
                if self._startup_cancel is not None:
                    self._startup_cancel.set()
                if self._startup_gate is not None:
                    self._startup_gate.set()
                self._condition.notify_all()
                self._condition.wait_for(lambda: not self._start_in_progress)

            if self._stop_in_progress:
                self._condition.wait_for(lambda: not self._stop_in_progress)
                if self._all_cleanup_complete() and self._result is not None:
                    return self._result
                result = self._result or self._make_result()
                errors = tuple(
                    error
                    for error in self._all_errors()
                    if error.stage == MeetingTrackErrorStage.STOP
                )
                raise MeetingAudioTracksStopError(errors, result)

            if self._all_cleanup_complete() and self._result is not None:
                return self._result
            if self._state == MeetingAudioTracksState.FAILED and not any(
                not track.cleanup_complete for track in self._tracks.values()
            ):
                assert self._result is not None
                return self._result

            self._stop_in_progress = True
            self._stop_requested = True
            if self._state != MeetingAudioTracksState.FAILED:
                self._state = MeetingAudioTracksState.STOPPING
            self._condition.notify_all()

        pending = tuple(
            track
            for track in self._tracks_in_role_order()
            if not track.cleanup_complete
        )
        with self._condition:
            self._cleanup_in_progress = True
        try:
            stop_errors = self._run_parallel_stop(pending)
        finally:
            with self._condition:
                self._cleanup_in_progress = False
        for role, error in stop_errors.items():
            self._record_error(role, MeetingTrackErrorStage.STOP, error)
        self._collect_recording_results()
        result = self._make_result()
        cleanup_complete = self._all_cleanup_complete()

        with self._condition:
            self._result = result
            self._state = (
                MeetingAudioTracksState.STOPPED
                if cleanup_complete
                else MeetingAudioTracksState.FAILED
            )
            self._stop_in_progress = False
            self._condition.notify_all()

        if stop_errors or not cleanup_complete:
            attempt_errors = tuple(
                error
                for error in self._all_errors()
                if error.stage == MeetingTrackErrorStage.STOP
            )
            raise MeetingAudioTracksStopError(attempt_errors, result)
        return result

    def _run_parallel_stop(
        self, tracks: tuple[_OwnedTrack, ...]
    ) -> dict[MeetingTrackRole, Exception]:
        errors: dict[MeetingTrackRole, Exception] = {}
        errors_lock = threading.Lock()
        started_threads: list[threading.Thread] = []
        fallback_tracks: list[_OwnedTrack] = []

        def stop_track(track: _OwnedTrack) -> None:
            try:
                track.recording_result = track.fanout.stop()
            except Exception as exc:
                with errors_lock:
                    errors[track.role] = exc
                track.cleanup_complete = False
            else:
                track.cleanup_complete = True

        for track in tracks:
            try:
                thread = self._thread_factory(
                    target=lambda track=track: stop_track(track),
                    name=f"{self._STOP_THREAD_PREFIX}-{track.role.name.lower()}",
                    daemon=False,
                )
                thread.start()
            except Exception:
                fallback_tracks.append(track)
            else:
                started_threads.append(thread)

        # All launchable cleanup workers have been dispatched before a
        # synchronous fallback is allowed to block this coordinator thread.
        for track in fallback_tracks:
            stop_track(track)

        for thread in started_threads:
            thread.join()
        return errors

    def _collect_recording_results(self) -> None:
        for track in self._tracks_in_role_order():
            if track.recording_result is not None:
                continue
            try:
                track.recording_result = track.recorder.stop()
            except Exception as exc:
                self._record_error(track.role, MeetingTrackErrorStage.STOP, exc)

    def _make_result(self) -> MeetingAudioTracksResult:
        track_results: dict[MeetingTrackRole, MeetingTrackRecordingResult] = {}
        for track in self._tracks_in_role_order():
            if track.recording_result is None:
                raise MeetingAudioTracksError(
                    f"No recording result is available for {track.role.name.lower()}"
                )
            disqualifying_error = any(
                error.stage
                in (
                    MeetingTrackErrorStage.START,
                    MeetingTrackErrorStage.SOURCE_RUNTIME,
                    MeetingTrackErrorStage.RECORDER,
                )
                for error in track.errors
            )
            complete = (
                track.start_succeeded
                and track.cleanup_complete
                and track.fanout.state == MeetingAudioFanoutState.STOPPED
                and track.recording_result.state == MeetingRecorderState.STOPPED
                and track.recording_result.published
                and not disqualifying_error
            )
            track_results[track.role] = MeetingTrackRecordingResult(
                role=track.role,
                recording=track.recording_result,
                timing=track.timing.snapshot(
                    durable_sample_count=track.recording_result.sample_count
                ),
                errors=tuple(track.errors),
                complete=complete,
            )

        microphone = track_results[MeetingTrackRole.MICROPHONE]
        remote = track_results[MeetingTrackRole.REMOTE]
        if microphone.complete and remote.complete:
            outcome = MeetingAudioTracksOutcome.COMPLETE
        elif self._is_usable(microphone) or self._is_usable(remote):
            outcome = MeetingAudioTracksOutcome.PARTIAL
        else:
            outcome = MeetingAudioTracksOutcome.FAILED
        return MeetingAudioTracksResult(
            coordinator_start_monotonic_ns=self._coordinator_start_ns,
            microphone=microphone,
            remote=remote,
            outcome=outcome,
            errors=self._all_errors(),
        )

    @staticmethod
    def _is_usable(track: MeetingTrackRecordingResult) -> bool:
        return track.recording.published and track.recording.sample_count > 0

    def _on_source_error(self, role: MeetingTrackRole, error: Exception) -> None:
        with self._condition:
            stage = (
                MeetingTrackErrorStage.STOP
                if self._cleanup_in_progress
                or self._state
                in (MeetingAudioTracksState.STOPPING, MeetingAudioTracksState.STOPPED)
                else MeetingTrackErrorStage.SOURCE_RUNTIME
            )
            self._record_error_locked(role, stage, error)
            if self._state == MeetingAudioTracksState.RUNNING:
                self._state = MeetingAudioTracksState.DEGRADED
            self._condition.notify_all()

    def _on_recorder_error(self, role: MeetingTrackRole, error: Exception) -> None:
        with self._condition:
            self._record_error_locked(role, MeetingTrackErrorStage.RECORDER, error)
            if self._state == MeetingAudioTracksState.RUNNING:
                self._state = MeetingAudioTracksState.DEGRADED
            self._condition.notify_all()

    def _record_error(
        self,
        role: MeetingTrackRole,
        stage: MeetingTrackErrorStage,
        error: Exception,
    ) -> None:
        with self._condition:
            self._record_error_locked(role, stage, error)
            self._condition.notify_all()

    def _record_error_locked(
        self,
        role: MeetingTrackRole,
        stage: MeetingTrackErrorStage,
        error: Exception,
    ) -> None:
        track = self._tracks[role]
        stage_indices = [
            index for index, item in enumerate(track.errors) if item.stage == stage
        ]
        if any(
            self._same_error_incident(track.errors[index].exception, error)
            for index in stage_indices
        ):
            return
        track_error = MeetingTrackError(role, stage, error)
        if len(stage_indices) < self._ERRORS_PER_ROLE_STAGE_LIMIT:
            track.errors.append(track_error)
        else:
            track.errors[stage_indices[-1]] = track_error

    def _record_start_cancellation(self) -> None:
        for role in (MeetingTrackRole.MICROPHONE, MeetingTrackRole.REMOTE):
            self._record_error(
                role,
                MeetingTrackErrorStage.START,
                RuntimeError("Meeting audio track startup was cancelled by stop()"),
            )

    @staticmethod
    def _same_error_incident(first: Exception, second: Exception) -> bool:
        return first is second or first.__cause__ is second or second.__cause__ is first

    def _all_errors(self) -> tuple[MeetingTrackError, ...]:
        with self._condition:
            return tuple(
                error
                for track in self._tracks_in_role_order()
                for error in track.errors
            )

    def _all_cleanup_complete(self) -> bool:
        return all(track.cleanup_complete for track in self._tracks.values())

    def _tracks_in_role_order(self) -> tuple[_OwnedTrack, _OwnedTrack]:
        return (
            self._tracks[MeetingTrackRole.MICROPHONE],
            self._tracks[MeetingTrackRole.REMOTE],
        )

    def _clear_startup_locked(self) -> None:
        self._startup_gate = None
        self._startup_cancel = None

    @staticmethod
    def _normalized_path(path: os.PathLike[str] | str) -> str:
        return os.path.normcase(str(Path(path).resolve(strict=False)))

    @staticmethod
    def _validate_sample_rate(label: str, sample_rate: int) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise ValueError(f"{label} sample_rate must be a positive integer")
        if sample_rate <= 0:
            raise ValueError(f"{label} sample_rate must be positive")


__all__ = [
    "MeetingAudioTracks",
    "MeetingAudioTracksError",
    "MeetingAudioTracksOutcome",
    "MeetingAudioTracksResult",
    "MeetingAudioTracksStartError",
    "MeetingAudioTracksState",
    "MeetingAudioTracksStateError",
    "MeetingAudioTracksStopError",
    "MeetingTrackError",
    "MeetingTrackErrorStage",
    "MeetingTrackRecordingResult",
    "MeetingTrackRole",
    "MeetingTrackTiming",
    "MeetingTrackTimingAnchor",
]
