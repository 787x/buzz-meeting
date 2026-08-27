"""Deterministic tests for ``MeetingSession``."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pytest

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksResult,
    MeetingAudioTracksStartError,
    MeetingAudioTracksState,
    MeetingAudioTracksStopError,
    MeetingAudioTracksOutcome,
    MeetingTrackRecordingResult,
    MeetingTrackRole,
    MeetingTrackTiming,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorderState,
    MeetingRecordingResult,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSession,
    MeetingSessionState,
    MeetingSessionStateError,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SequenceClock:
    """Deterministic wall-clock that returns pre-configured datetimes."""

    def __init__(self, values: list[datetime]) -> None:
        self._values = iter(values)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return next(self._values)


class _SequenceMonotonic:
    """Deterministic monotonic clock that returns pre-configured ints."""

    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return next(self._values)


def _utc(y: int, m: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, s, tzinfo=timezone.utc)


def _make_recording_result(
    *,
    published: bool = True,
    sample_count: int = 0,
    state: MeetingRecorderState = MeetingRecorderState.STOPPED,
) -> MeetingRecordingResult:
    return MeetingRecordingResult(
        output_path="/dev/null",
        sample_rate=16_000,
        sample_count=sample_count,
        duration_seconds=sample_count / 16_000,
        state=state,
        error=None,
        published=published,
    )


def _make_track_result(
    role: MeetingTrackRole,
    *,
    complete: bool = True,
    published: bool = True,
    sample_count: int = 0,
) -> MeetingTrackRecordingResult:
    return MeetingTrackRecordingResult(
        role=role,
        recording=_make_recording_result(
            published=published, sample_count=sample_count
        ),
        timing=MeetingTrackTiming(anchors=()),
        errors=(),
        complete=complete,
    )


def _make_tracks_result(
    *,
    outcome: MeetingAudioTracksOutcome = MeetingAudioTracksOutcome.COMPLETE,
    coordinator_start_monotonic_ns: Optional[int] = 0,
) -> MeetingAudioTracksResult:
    return MeetingAudioTracksResult(
        coordinator_start_monotonic_ns=coordinator_start_monotonic_ns,
        microphone=_make_track_result(MeetingTrackRole.MICROPHONE),
        remote=_make_track_result(MeetingTrackRole.REMOTE),
        outcome=outcome,
        errors=(),
    )


class FakeTracks:
    """Controllable fake ``MeetingAudioTracks`` for deterministic testing."""

    def __init__(
        self,
        *,
        start_result: Optional[MeetingAudioTracksResult] = None,
        start_error: Optional[MeetingAudioTracksStartError] = None,
        stop_result: Optional[MeetingAudioTracksResult] = None,
        stop_error: Optional[MeetingAudioTracksStopError] = None,
    ) -> None:
        self._state = MeetingAudioTracksState.CREATED
        self._start_result = start_result or _make_tracks_result()
        self._start_error = start_error
        self._stop_result = stop_result or _make_tracks_result()
        self._stop_error = stop_error
        self.start_count = 0
        self.stop_count = 0
        self.pause_start = False
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()
        self.pause_stop = False
        self.stop_entered = threading.Event()
        self.allow_stop = threading.Event()
        self._stop_results: list[MeetingAudioTracksResult] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> MeetingAudioTracksState:
        with self._lock:
            return self._state

    def start(self) -> None:
        with self._lock:
            self.start_count += 1
        self.start_entered.set()
        if self.pause_start:
            assert self.allow_start.wait(timeout=5)
        if self._start_error is not None:
            with self._lock:
                self._state = MeetingAudioTracksState.FAILED
            raise self._start_error
        with self._lock:
            self._state = MeetingAudioTracksState.RUNNING

    def stop(self) -> MeetingAudioTracksResult:
        with self._lock:
            self.stop_count += 1
        self.stop_entered.set()
        if self.pause_stop:
            assert self.allow_stop.wait(timeout=5)
        if self._stop_error is not None:
            raise self._stop_error
        with self._lock:
            self._state = MeetingAudioTracksState.STOPPED
            self._stop_results.append(self._stop_result)
        return self._stop_result


class FailingThenSucceedingTracks(FakeTracks):
    """Tracks whose stop fails N times then succeeds."""

    def __init__(
        self,
        fail_stop_count: int,
        *,
        stop_error_factory=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._fail_stop_count = fail_stop_count
        self._stop_error_factory = stop_error_factory or (
            lambda result: MeetingAudioTracksStopError((), result)
        )

    def stop(self) -> MeetingAudioTracksResult:
        with self._lock:
            self.stop_count += 1
            current = self.stop_count
        self.stop_entered.set()
        if self.pause_stop:
            assert self.allow_stop.wait(timeout=5)
        if current <= self._fail_stop_count:
            error = self._stop_error_factory(self._stop_result)
            raise error
        with self._lock:
            self._state = MeetingAudioTracksState.STOPPED
            self._stop_results.append(self._stop_result)
        return self._stop_result


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------


class TestBasicAPI:
    def test_default_uuid_is_uuid4_and_stable(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        sid = session.session_id
        assert isinstance(sid, uuid.UUID)
        assert sid.version == 4
        assert session.session_id == sid

    def test_public_injected_uuid_preserved(self):
        injected = uuid.UUID("12345678-1234-5678-1234-567812345678")
        tracks = FakeTracks()
        session = MeetingSession(
            tracks, MeetingRemoteSourceKind.SYSTEM, session_id=injected
        )
        assert session.session_id == injected

    def test_invalid_session_id_rejected(self):
        tracks = FakeTracks()
        with pytest.raises(TypeError, match="session_id"):
            MeetingSession(
                tracks,
                MeetingRemoteSourceKind.SYSTEM,
                session_id="not-a-uuid",  # type: ignore[arg-type]
            )

    def test_invalid_remote_source_kind_rejected(self):
        tracks = FakeTracks()
        with pytest.raises(TypeError, match="remote_source_kind"):
            MeetingSession(tracks, "SYSTEM")  # type: ignore[arg-type]

    def test_system_provenance_roundtrip(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        assert session.remote_source_kind == MeetingRemoteSourceKind.SYSTEM
        snap = session.snapshot()
        assert snap.remote_source_kind == MeetingRemoteSourceKind.SYSTEM

    def test_application_provenance_roundtrip(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.APPLICATION)
        assert session.remote_source_kind == MeetingRemoteSourceKind.APPLICATION
        snap = session.snapshot()
        assert snap.remote_source_kind == MeetingRemoteSourceKind.APPLICATION

    def test_snapshot_immutable(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        snap = session.snapshot()
        with pytest.raises(AttributeError):
            snap.state = MeetingSessionState.ACTIVE  # type: ignore[misc]

    def test_audio_state_non_optional_from_construction(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        snap = session.snapshot()
        assert snap.audio_state == MeetingAudioTracksState.CREATED
        assert snap.audio_state is not None

    def test_stop_before_start_raises_and_does_not_call_tracks(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        with pytest.raises(MeetingSessionStateError, match="never started"):
            session.stop()
        assert tracks.stop_count == 0

    def test_duplicate_start_raises(self):
        tracks = FakeTracks()
        session = MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM)
        session.start()
        with pytest.raises(MeetingSessionStateError, match="ACTIVE"):
            session.start()
        session.stop()


# ---------------------------------------------------------------------------
# UTC wall-clock tests
# ---------------------------------------------------------------------------


class TestUTCClock:
    def test_utc_aware_clock_accepted(self):
        clock = _SequenceClock([_utc(2026, 1, 1)])
        tracks = FakeTracks()
        session = MeetingSession(
            tracks, MeetingRemoteSourceKind.SYSTEM, _wall_clock=clock
        )
        assert session.created_at == _utc(2026, 1, 1)
        assert session.created_at.tzinfo == timezone.utc

    def test_aware_plus0800_normalized_to_utc(self):
        tz_plus8 = ZoneInfo("Asia/Shanghai")
        local = datetime(2026, 6, 15, 10, 0, 0, tzinfo=tz_plus8)
        clock = _SequenceClock([local])
        tracks = FakeTracks()
        session = MeetingSession(
            tracks, MeetingRemoteSourceKind.SYSTEM, _wall_clock=clock
        )
        assert session.created_at.tzinfo == timezone.utc
        expected_utc = local.astimezone(timezone.utc)
        assert session.created_at == expected_utc

    def test_naive_datetime_rejected(self):
        naive = datetime(2026, 1, 1)
        clock = _SequenceClock([naive])
        tracks = FakeTracks()
        with pytest.raises(ValueError, match="timezone-aware"):
            MeetingSession(tracks, MeetingRemoteSourceKind.SYSTEM, _wall_clock=clock)

    def test_all_timestamps_utc_aware(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200])
        tracks = FakeTracks()
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )
        session.start()
        session.stop()
        assert session.created_at.tzinfo == timezone.utc
        assert session.started_at is not None
        assert session.started_at.tzinfo == timezone.utc
        assert session.ended_at is not None
        assert session.ended_at.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Test A — startup capture timestamp
# ---------------------------------------------------------------------------


class TestStartupCaptureTimestamp:
    def test_started_at_set_before_tracks_start_returns(self):
        """Tracks.start() blocks; session.started_at must already exist."""
        t_created = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t_created, t_start, t_stop])
        mono = _SequenceMonotonic([1000, 2000, 3000])
        tracks = FakeTracks()
        tracks.pause_start = True
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        observed_start: list[Optional[datetime]] = []

        def start_session():
            session.start()

        start_thread = threading.Thread(target=start_session)
        start_thread.start()

        # tracks.start() is blocked — session should already have started_at.
        assert tracks.start_entered.wait(timeout=5)
        observed_start.append(session.started_at)
        assert session.state == MeetingSessionState.STARTING

        tracks.allow_start.set()
        start_thread.join(timeout=5)

        # Verify started_at was set before tracks.start returned.
        assert observed_start[0] == t_start
        assert session.state == MeetingSessionState.ACTIVE

        session.stop()


# ---------------------------------------------------------------------------
# Test B — spontaneous start failure time
# ---------------------------------------------------------------------------


class TestSpontaneousStartFailure:
    def test_ended_at_is_not_created_at(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)  # created_at
        t1 = _utc(2026, 1, 1, 0, 0, 1)  # started_at (before tracks.start)
        t2 = _utc(2026, 1, 1, 0, 0, 2)  # ended_at (at failure)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        tracks_result = _make_tracks_result(outcome=MeetingAudioTracksOutcome.FAILED)
        start_error = MeetingAudioTracksStartError((), tracks_result)
        tracks = FakeTracks(start_error=start_error)
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        with pytest.raises(MeetingAudioTracksStartError):
            session.start()

        assert session.state == MeetingSessionState.FAILED
        assert session.created_at == t0
        assert session.started_at == t1
        assert session.ended_at == t2
        assert session.ended_at != session.created_at
        assert session.duration_ns == 300 - 200
        assert session.audio is tracks_result


# ---------------------------------------------------------------------------
# Test C — FAILED cleanup retry
# ---------------------------------------------------------------------------


class TestFailedCleanupRetry:
    def test_failed_session_stop_retries_underlying_cleanup(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        tracks_result = _make_tracks_result(outcome=MeetingAudioTracksOutcome.FAILED)
        start_error = MeetingAudioTracksStartError((), tracks_result)
        tracks = FailingThenSucceedingTracks(
            fail_stop_count=1,
            start_error=start_error,
            stop_result=tracks_result,
        )
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        with pytest.raises(MeetingAudioTracksStartError):
            session.start()
        assert session.state == MeetingSessionState.FAILED

        # First stop — underlying cleanup fails.
        with pytest.raises(MeetingAudioTracksStopError):
            session.stop()
        assert session.state == MeetingSessionState.FAILED
        assert tracks.stop_count == 1

        # Second stop — cleanup succeeds.
        result = session.stop()
        assert session.state == MeetingSessionState.FAILED  # stays FAILED
        assert session.audio is tracks_result
        assert result is tracks_result
        assert tracks.stop_count == 2

        # Timestamps not overwritten by retry.
        assert session.ended_at == t2
        assert session.duration_ns == 300 - 200

    def test_failed_session_audio_updates_on_successful_retry(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        initial_result = _make_tracks_result(outcome=MeetingAudioTracksOutcome.FAILED)
        retry_result = _make_tracks_result(outcome=MeetingAudioTracksOutcome.COMPLETE)
        start_error = MeetingAudioTracksStartError((), initial_result)
        tracks = FailingThenSucceedingTracks(
            fail_stop_count=1,
            start_error=start_error,
            stop_result=retry_result,
        )
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        with pytest.raises(MeetingAudioTracksStartError):
            session.start()
        assert session.audio is initial_result

        with pytest.raises(MeetingAudioTracksStopError):
            session.stop()

        session.stop()
        assert session.audio is retry_result


# ---------------------------------------------------------------------------
# Test D — stop during STARTING / cancellation exception
# ---------------------------------------------------------------------------


class TestStopDuringStartingCancellation:
    def test_late_start_exception_does_not_overwrite_completed(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_stop = _utc(2026, 1, 1, 0, 0, 5)
        # Enough clock values for both threads.
        clock = _SequenceClock([t0, t_stop, t_stop])
        mono = _SequenceMonotonic([100, 500, 501])

        stop_result = _make_tracks_result()
        tracks = FakeTracks(stop_result=stop_result)

        # We'll replace start to simulate: blocks, then raises after stop.
        start_error_result = _make_tracks_result(
            outcome=MeetingAudioTracksOutcome.FAILED
        )
        start_error = MeetingAudioTracksStartError((), start_error_result)

        def controlled_start():
            tracks.start_entered.set()
            tracks.allow_start.wait(timeout=5)
            raise start_error

        tracks.start = controlled_start  # type: ignore[method-assign]

        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        start_errors: list[BaseException] = []

        def start_thread():
            try:
                session.start()
            except BaseException as exc:
                start_errors.append(exc)

        t = threading.Thread(target=start_thread)
        t.start()

        # Wait for start to enter.
        assert tracks.start_entered.wait(timeout=5)
        assert session.state == MeetingSessionState.STARTING

        # Stop while starting.
        result = session.stop()
        assert session.state == MeetingSessionState.COMPLETED

        # Now let start thread continue (it will raise).
        tracks.allow_start.set()
        t.join(timeout=5)

        # State must still be COMPLETED.
        assert session.state == MeetingSessionState.COMPLETED
        assert len(start_errors) == 1
        assert isinstance(start_errors[0], MeetingAudioTracksStartError)
        assert result is stop_result


# ---------------------------------------------------------------------------
# Test E — stop during STARTING / late start success
# ---------------------------------------------------------------------------


class TestStopDuringStartingLateSuccess:
    def test_late_start_success_does_not_set_active(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_stop = _utc(2026, 1, 1, 0, 0, 5)
        clock = _SequenceClock([t0, t_stop, t_stop])
        mono = _SequenceMonotonic([100, 500, 501])

        stop_result = _make_tracks_result()
        tracks = FakeTracks(stop_result=stop_result)
        tracks.pause_start = True

        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        start_succeeded = threading.Event()

        def start_thread():
            session.start()
            start_succeeded.set()

        t = threading.Thread(target=start_thread)
        t.start()

        assert tracks.start_entered.wait(timeout=5)
        assert session.state == MeetingSessionState.STARTING

        # Stop while starting.
        result = session.stop()
        assert session.state == MeetingSessionState.COMPLETED

        # Let start complete (success path).
        tracks.allow_start.set()
        t.join(timeout=5)

        # Start succeeded internally, but session must NOT be ACTIVE.
        assert session.state == MeetingSessionState.COMPLETED
        assert result is stop_result


# ---------------------------------------------------------------------------
# Test F — stop failure retry
# ---------------------------------------------------------------------------


class TestStopFailureRetry:
    def test_retry_after_stop_failure(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop1 = _utc(2026, 1, 1, 0, 0, 2)
        t_stop2 = _utc(2026, 1, 1, 0, 0, 3)  # should NOT be used
        clock = _SequenceClock([t0, t_start, t_stop1, t_stop2])
        mono = _SequenceMonotonic([100, 200, 300, 400])

        final_result = _make_tracks_result()
        tracks = FailingThenSucceedingTracks(
            fail_stop_count=1, stop_result=final_result
        )
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        assert session.state == MeetingSessionState.ACTIVE

        # First stop fails.
        with pytest.raises(MeetingAudioTracksStopError):
            session.stop()
        assert session.state == MeetingSessionState.STOPPING
        assert session.ended_at == t_stop1
        first_end = session.ended_at

        # Second stop succeeds.
        result = session.stop()
        assert session.state == MeetingSessionState.COMPLETED
        assert result is final_result

        # Timestamps unchanged.
        assert session.ended_at == first_end
        assert session.duration_ns == 300 - 200


# ---------------------------------------------------------------------------
# Test G — concurrent stop
# ---------------------------------------------------------------------------


class TestConcurrentStop:
    def test_two_threads_stop_safely(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t_start, t_stop, t_stop])
        mono = _SequenceMonotonic([100, 200, 300, 301])

        stop_result = _make_tracks_result()
        tracks = FakeTracks(stop_result=stop_result)
        tracks.pause_stop = True

        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()

        results: list[MeetingAudioTracksResult] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def stop_worker():
            try:
                barrier.wait()
                results.append(session.stop())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=stop_worker) for _ in range(2)]
        for t in threads:
            t.start()

        # Both threads enter stop; release the fake tracks stop.
        assert tracks.stop_entered.wait(timeout=5)
        tracks.allow_stop.set()

        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert session.state == MeetingSessionState.COMPLETED
        assert len(results) == 2
        assert results[0] is results[1]
        assert tracks.stop_count >= 1


# ---------------------------------------------------------------------------
# Test H — PARTIAL audio
# ---------------------------------------------------------------------------


class TestPartialAudio:
    def test_partial_audio_session_completed(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        partial_result = _make_tracks_result(outcome=MeetingAudioTracksOutcome.PARTIAL)
        tracks = FakeTracks(stop_result=partial_result)
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        result = session.stop()

        assert session.state == MeetingSessionState.COMPLETED
        assert result is partial_result
        snap = session.snapshot()
        assert snap.audio is not None
        assert snap.audio.outcome == MeetingAudioTracksOutcome.PARTIAL


# ---------------------------------------------------------------------------
# Test I — DEGRADED audio
# ---------------------------------------------------------------------------


class TestDegradedAudio:
    def test_degraded_tracks_session_still_active(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        tracks = FakeTracks()
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        assert session.state == MeetingSessionState.ACTIVE

        # Simulate runtime degradation at tracks level.
        with tracks._lock:
            tracks._state = MeetingAudioTracksState.DEGRADED

        assert session.state == MeetingSessionState.ACTIVE
        snap = session.snapshot()
        assert snap.audio_state == MeetingAudioTracksState.DEGRADED

        session.stop()


# ---------------------------------------------------------------------------
# Test J — wall-clock backjump
# ---------------------------------------------------------------------------


class TestWallClockBackjump:
    def test_duration_uses_monotonic_not_wall_clock(self):
        # Wall clock goes BACKWARD.
        t0 = _utc(2026, 1, 1, 12, 0, 0)
        t_start = _utc(2026, 1, 1, 12, 0, 0)  # same
        t_end = _utc(2026, 1, 1, 11, 59, 0)  # BACKWARD
        clock = _SequenceClock([t0, t_start, t_end])
        # Monotonic goes forward normally (start=100, stop=500).
        mono = _SequenceMonotonic([100, 500])

        tracks = FakeTracks()
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        session.stop()

        # Duration from monotonic (stop=500 - start=100).
        assert session.duration_ns == 500 - 100
        # Wall timestamps preserved as-is (backward is fine).
        assert session.ended_at == t_end


# ---------------------------------------------------------------------------
# Test K — no late state overwrite
# ---------------------------------------------------------------------------


class TestNoLateStateOverwrite:
    def test_completed_not_overridden_by_late_start(self):
        """Test D covered the exception path; this covers the success path."""

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_stop = _utc(2026, 1, 1, 0, 0, 5)
        clock = _SequenceClock([t0, t_stop, t_stop])
        mono = _SequenceMonotonic([100, 500, 501])

        stop_result = _make_tracks_result()
        tracks = FakeTracks(stop_result=stop_result)
        tracks.pause_start = True

        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        def start_thread():
            session.start()

        t = threading.Thread(target=start_thread)
        t.start()
        assert tracks.start_entered.wait(timeout=5)

        session.stop()
        assert session.state == MeetingSessionState.COMPLETED

        tracks.allow_start.set()
        t.join(timeout=5)

        assert session.state == MeetingSessionState.COMPLETED

    def test_failed_stays_failed_after_cleanup_retry_succeeds(self):
        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([100, 200, 300])

        tracks_result = _make_tracks_result()
        start_error = MeetingAudioTracksStartError((), tracks_result)
        tracks = FakeTracks(start_error=start_error, stop_result=tracks_result)
        session = MeetingSession(
            tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        with pytest.raises(MeetingAudioTracksStartError):
            session.start()
        assert session.state == MeetingSessionState.FAILED

        session.stop()
        assert session.state == MeetingSessionState.FAILED


# ---------------------------------------------------------------------------
# Real PR8 integration (lightweight, no hardware)
# ---------------------------------------------------------------------------


class TestRealTracksIntegration:
    """Build a real ``MeetingAudioTracks`` with deterministic fakes to verify
    full-stack session delegation without hardware."""

    def test_successful_session_lifecycle(self, tmp_path):
        from tests.meeting.meeting_audio_tracks_test import (
            FakeRecorderFactory,
            _make_tracks,
        )

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([1000, 2000, 3000])

        factory = FakeRecorderFactory()
        real_tracks, mic, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)

        session = MeetingSession(
            real_tracks,
            MeetingRemoteSourceKind.APPLICATION,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        assert session.state == MeetingSessionState.ACTIVE

        mic.deliver(np.ones(10, dtype=np.float32))
        remote.deliver(np.ones(5, dtype=np.float32))

        result = session.stop()

        assert session.state == MeetingSessionState.COMPLETED
        assert result is not None
        assert result.outcome == MeetingAudioTracksOutcome.COMPLETE
        assert session.audio is result
        assert session.duration_ns == 3000 - 2000

    def test_partial_outcome_session_still_completed(self, tmp_path):
        from tests.meeting.meeting_audio_tracks_test import (
            FakeRecorderFactory,
            _make_tracks,
        )

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t1 = _utc(2026, 1, 1, 0, 0, 1)
        t2 = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t1, t2])
        mono = _SequenceMonotonic([1000, 2000, 3000])

        factory = FakeRecorderFactory()
        real_tracks, mic, remote, _ = _make_tracks(tmp_path, recorder_factory=factory)

        session = MeetingSession(
            real_tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()

        mic.deliver(np.ones(5, dtype=np.float32))

        from buzz.audio_capture.source import AudioSourceError

        mic.fail(AudioSourceError("microphone disconnected"))
        remote.deliver(np.ones(3, dtype=np.float32))

        result = session.stop()

        assert session.state == MeetingSessionState.COMPLETED
        assert result.outcome in (
            MeetingAudioTracksOutcome.PARTIAL,
            MeetingAudioTracksOutcome.COMPLETE,
        )


# ---------------------------------------------------------------------------
# M1 — Pre-tracks.start race (real PR8 integration)
# ---------------------------------------------------------------------------


class _PreStartGateTracks:
    """Proxy that intercepts ``start`` attribute lookup on real tracks.

    Python evaluates the property getter each time ``tracks.start`` is
    accessed, so the ``start_attribute_entered`` event fires *after* the
    session has released its own lock but *before* the underlying
    ``MeetingAudioTracks.start()`` is actually invoked.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.start_attribute_entered = threading.Event()
        self.allow_start_attribute = threading.Event()

    @property
    def state(self) -> MeetingAudioTracksState:
        return self.inner.state

    @property
    def start(self):
        self.start_attribute_entered.set()
        self.allow_start_attribute.wait(timeout=5)
        return self.inner.start

    def stop(self, *args, **kwargs):
        return self.inner.stop(*args, **kwargs)


class TestPreStartGateRace:
    """Deterministic coverage of the M1 race window:

    Session has ``state == STARTING``, timestamps set, lock released — but
    ``tracks.start()`` has not been called.  Another thread calls
    ``session.stop()``.
    """

    def test_stop_before_tracks_start_called(self, tmp_path):
        from tests.meeting.meeting_audio_tracks_test import (
            FakeRecorderFactory,
            _make_tracks,
        )

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t_start, t_stop])
        mono = _SequenceMonotonic([1000, 2000, 3000])

        factory = FakeRecorderFactory()
        real_tracks, _mic, _remote, _ = _make_tracks(tmp_path, recorder_factory=factory)
        proxy = _PreStartGateTracks(real_tracks)

        session = MeetingSession(
            proxy,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        late_start_error: list[BaseException] = []

        def start_session():
            try:
                session.start()
            except BaseException as exc:
                late_start_error.append(exc)

        start_thread = threading.Thread(target=start_session)
        start_thread.start()

        # Wait until the session has released its lock and is about to call
        # tracks.start() — the proxy's property getter blocks here.
        assert proxy.start_attribute_entered.wait(timeout=5)

        # The session is in STARTING with timestamps set, but tracks.start()
        # has NOT been called.  Prove it:
        assert session.state == MeetingSessionState.STARTING
        assert session.started_at == t_start
        assert session.duration_ns is None
        assert real_tracks.state == MeetingAudioTracksState.CREATED

        # Stop from another thread — must converge without deadlock.
        stop_result = session.stop()

        assert session.state == MeetingSessionState.COMPLETED
        assert session.audio is stop_result

        # Now let the late start thread call the real tracks.start().
        proxy.allow_start_attribute.set()
        start_thread.join(timeout=5)

        # Hard invariants — no late overwrite.
        assert session.state == MeetingSessionState.COMPLETED
        assert session.audio is stop_result

        # The real underlying tracks must not be active — stop already
        # completed cleanup.
        assert real_tracks.state not in (
            MeetingAudioTracksState.RUNNING,
            MeetingAudioTracksState.DEGRADED,
        )


# ---------------------------------------------------------------------------
# Concurrent stop result precedence (real PR8 integration)
# ---------------------------------------------------------------------------


class TestConcurrentStopResultPrecedence:
    """Prove that concurrent ``session.stop()`` callers are safe under the
    real ``MeetingAudioTracks`` concurrency contract.

    PR8 guarantee: when a source.stop() error occurs during cleanup, the
    first caller raises ``MeetingAudioTracksStopError`` and subsequent
    concurrent callers also see the error (because cleanup never completed).
    This is all-or-nothing: either ALL concurrent callers get the cached
    success result, or ALL see the error.

    This test proves:
    1. Concurrent stop callers on a failing source both get errors.
    2. Session remains in STOPPING (not COMPLETED) after both fail.
    3. A subsequent single stop() retry succeeds and commits COMPLETED.
    4. The retry result is the final audio, not a stale error result.
    """

    def test_concurrent_stop_failure_then_retry_succeeds(self, tmp_path):
        from tests.meeting.meeting_audio_tracks_test import (
            ControlledAudioSource,
            FakeRecorderFactory,
            _make_tracks,
        )

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop = _utc(2026, 1, 1, 0, 0, 2)
        t_retry = _utc(2026, 1, 1, 0, 0, 3)
        clock = _SequenceClock([t0, t_start, t_stop, t_stop, t_retry])
        mono = _SequenceMonotonic([1000, 2000, 3000, 3001, 4000])

        mic = ControlledAudioSource(48_000)
        remote = ControlledAudioSource(16_000)

        # First source.stop() raises → first stop-caller in
        # MeetingAudioTracks gets the error; second caller also gets error
        # (PR8 all-or-nothing guarantee for incomplete cleanup).
        mic.stop_errors.append(OSError("microphone hardware stop failed"))

        factory = FakeRecorderFactory()
        real_tracks, _mic_src, _remote_src, _ = _make_tracks(
            tmp_path, microphone=mic, remote=remote, recorder_factory=factory
        )

        session = MeetingSession(
            real_tracks,
            MeetingRemoteSourceKind.APPLICATION,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        mic.deliver(np.ones(10, dtype=np.float32))
        remote.deliver(np.ones(5, dtype=np.float32))

        barrier = threading.Barrier(2)
        stop_errors: list[BaseException] = []

        def stop_worker():
            try:
                barrier.wait()
                session.stop()
            except BaseException as exc:
                stop_errors.append(exc)

        threads = [threading.Thread(target=stop_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # PR8 all-or-nothing: both callers must have received the error.
        assert len(stop_errors) == 2
        assert all(isinstance(e, MeetingAudioTracksStopError) for e in stop_errors)

        # Session is STOPPING — not COMPLETED, not FAILED.
        assert session.state == MeetingSessionState.STOPPING
        assert session.audio is None

        # Timestamps from the first stop invocation are stable.
        assert session.ended_at is not None
        first_end = session.ended_at

        # Retry — this time source.stop() succeeds (error was one-shot).
        retry_result = session.stop()

        assert session.state == MeetingSessionState.COMPLETED
        assert session.audio is retry_result
        assert session.ended_at == first_end  # timestamps unchanged

    def test_concurrent_stop_both_succeed(self, tmp_path):
        """When no source error occurs, concurrent stop callers both get
        the same success result."""

        from tests.meeting.meeting_audio_tracks_test import (
            ControlledAudioSource,
            FakeRecorderFactory,
            _make_tracks,
        )

        t0 = _utc(2026, 1, 1, 0, 0, 0)
        t_start = _utc(2026, 1, 1, 0, 0, 1)
        t_stop = _utc(2026, 1, 1, 0, 0, 2)
        clock = _SequenceClock([t0, t_start, t_stop, t_stop])
        mono = _SequenceMonotonic([1000, 2000, 3000, 3001])

        mic = ControlledAudioSource(48_000)
        remote = ControlledAudioSource(16_000)

        factory = FakeRecorderFactory()
        real_tracks, _mic_src, _remote_src, _ = _make_tracks(
            tmp_path, microphone=mic, remote=remote, recorder_factory=factory
        )

        session = MeetingSession(
            real_tracks,
            MeetingRemoteSourceKind.SYSTEM,
            _wall_clock=clock,
            _monotonic_ns=mono,
        )

        session.start()
        mic.deliver(np.ones(10, dtype=np.float32))
        remote.deliver(np.ones(5, dtype=np.float32))

        barrier = threading.Barrier(2)
        results: list[MeetingAudioTracksResult] = []
        errors: list[BaseException] = []

        def stop_worker():
            try:
                barrier.wait()
                results.append(session.stop())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=stop_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Both callers must succeed.
        assert not errors
        assert len(results) == 2

        # Same result object (cached by PR8).
        assert results[0] is results[1]

        # Session committed to COMPLETED.
        assert session.state == MeetingSessionState.COMPLETED
        assert session.audio is results[0]


# Need numpy for integration tests.
import numpy as np  # noqa: E402
