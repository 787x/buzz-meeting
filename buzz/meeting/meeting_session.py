"""Meeting session lifecycle and identity.

``MeetingSession`` coordinates a meeting's lifecycle around a pre-constructed
``MeetingAudioTracks`` instance.  It adds stable identity, UTC wall-clock
timestamps, monotonic duration, and an immutable snapshot — but deliberately
delegates audio capture, recording, and cleanup to the tracks layer.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracks,
    MeetingAudioTracksResult,
    MeetingAudioTracksStartError,
    MeetingAudioTracksState,
    MeetingAudioTracksStopError,
)


class MeetingRemoteSourceKind(Enum):
    """Provenance of the remote/system audio source."""

    SYSTEM = auto()
    APPLICATION = auto()


class MeetingSessionState(Enum):
    """Meeting-level lifecycle states.

    Session lifecycle is intentionally separate from audio outcome.
    A ``PARTIAL`` audio result does not prevent ``COMPLETED``.
    Audio degradation is visible through ``MeetingAudioTracks.state``.
    """

    CREATED = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()
    COMPLETED = auto()
    FAILED = auto()


class MeetingSessionStateError(RuntimeError):
    """Raised when a ``MeetingSession`` API is used in an invalid state."""


@dataclass(frozen=True, slots=True)
class MeetingSessionSnapshot:
    """Immutable point-in-time view of a ``MeetingSession``."""

    session_id: uuid.UUID
    remote_source_kind: MeetingRemoteSourceKind
    state: MeetingSessionState
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration_ns: Optional[int]
    audio_state: MeetingAudioTracksState
    audio: Optional[MeetingAudioTracksResult]


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize *dt* to a UTC-aware ``datetime``.

    Raises ``ValueError`` if *dt* is a naive (timezone-unaware) datetime.
    Aware datetimes in non-UTC zones are converted.
    """

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware (got naive datetime)")
    return dt.astimezone(timezone.utc)


class MeetingSession:
    """Coordinate a single meeting's lifecycle around pre-built audio tracks.

    The session owns identity, timestamps, and lifecycle state.  All audio
    capture, recording, and cleanup is delegated to the injected
    ``MeetingAudioTracks`` instance.
    """

    def __init__(
        self,
        tracks: MeetingAudioTracks,
        remote_source_kind: MeetingRemoteSourceKind,
        *,
        session_id: Optional[uuid.UUID] = None,
        _wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        _monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(remote_source_kind, MeetingRemoteSourceKind):
            raise TypeError(
                "remote_source_kind must be a MeetingRemoteSourceKind member"
            )
        if session_id is not None and not isinstance(session_id, uuid.UUID):
            raise TypeError("session_id must be a uuid.UUID or None")

        self._tracks = tracks
        self._remote_source_kind = remote_source_kind
        self._session_id = session_id or uuid.uuid4()
        self._wall_clock = _wall_clock
        self._monotonic_ns = _monotonic_ns

        self._condition = threading.Condition()
        self._state = MeetingSessionState.CREATED
        self._created_at = _ensure_utc(_wall_clock())
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None
        self._start_monotonic_ns: Optional[int] = None
        self._stop_monotonic_ns: Optional[int] = None
        self._stop_requested = False
        self._audio: Optional[MeetingAudioTracksResult] = None

    # -- properties ----------------------------------------------------------

    @property
    def session_id(self) -> uuid.UUID:
        return self._session_id

    @property
    def remote_source_kind(self) -> MeetingRemoteSourceKind:
        return self._remote_source_kind

    @property
    def state(self) -> MeetingSessionState:
        with self._condition:
            return self._state

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def started_at(self) -> Optional[datetime]:
        with self._condition:
            return self._started_at

    @property
    def ended_at(self) -> Optional[datetime]:
        with self._condition:
            return self._ended_at

    @property
    def duration_ns(self) -> Optional[int]:
        with self._condition:
            if self._start_monotonic_ns is None or self._stop_monotonic_ns is None:
                return None
            return self._stop_monotonic_ns - self._start_monotonic_ns

    @property
    def audio(self) -> Optional[MeetingAudioTracksResult]:
        with self._condition:
            return self._audio

    @property
    def audio_state(self) -> MeetingAudioTracksState:
        return self._tracks.state

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the meeting session.

        Raises ``MeetingSessionStateError`` if the session is not ``CREATED``.
        Propagates ``MeetingAudioTracksStartError`` on track start failure.
        """

        with self._condition:
            if self._state != MeetingSessionState.CREATED:
                raise MeetingSessionStateError(
                    f"Cannot start meeting session in state {self._state.name}"
                )
            self._state = MeetingSessionState.STARTING
            # Timestamp BEFORE tracks.start — covers startup capture
            self._started_at = _ensure_utc(self._wall_clock())
            self._start_monotonic_ns = self._monotonic_ns()

        try:
            self._tracks.start()
        except MeetingAudioTracksStartError as exc:
            self._handle_start_failure(exc)
            raise

        # Commit success — unless a concurrent stop already intervened.
        with self._condition:
            if self._stop_requested:
                # Stop thread already owns the lifecycle state.
                return
            self._state = MeetingSessionState.ACTIVE

    def stop(self) -> MeetingAudioTracksResult:
        """Stop the meeting session and return the audio tracks result.

        On ``CREATED`` raises ``MeetingSessionStateError``.
        On ``FAILED`` performs cleanup-retry via ``tracks.stop()``.
        On ``COMPLETED`` returns cached result (idempotent).
        Propagates ``MeetingAudioTracksStopError`` on track stop failure.
        """

        with self._condition:
            if self._state == MeetingSessionState.COMPLETED:
                assert self._audio is not None
                return self._audio

            if self._state == MeetingSessionState.CREATED:
                raise MeetingSessionStateError(
                    "Cannot stop a meeting session that was never started"
                )

            if self._state == MeetingSessionState.FAILED:
                # Cleanup retry — delegate to tracks, don't change session state.
                pass
            else:
                # Normal / stop-during-start path.
                self._record_end_timestamps_locked()
                self._stop_requested = True
                if self._state != MeetingSessionState.STOPPING:
                    self._state = MeetingSessionState.STOPPING

        # Delegate to tracks outside session lock.
        if self._state == MeetingSessionState.FAILED:
            result = self._tracks.stop()
            with self._condition:
                self._audio = result
            return result

        try:
            result = self._tracks.stop()
        except MeetingAudioTracksStopError:
            raise

        with self._condition:
            self._audio = result
            if self._state == MeetingSessionState.FAILED:
                # Stop-during-start race: start thread already committed FAILED.
                # Don't overwrite.
                pass
            else:
                self._state = MeetingSessionState.COMPLETED
            self._condition.notify_all()
        return result

    def snapshot(self) -> MeetingSessionSnapshot:
        """Return an immutable snapshot of the session's current state."""

        with self._condition:
            return MeetingSessionSnapshot(
                session_id=self._session_id,
                remote_source_kind=self._remote_source_kind,
                state=self._state,
                created_at=self._created_at,
                started_at=self._started_at,
                ended_at=self._ended_at,
                duration_ns=(
                    None
                    if self._start_monotonic_ns is None
                    or self._stop_monotonic_ns is None
                    else self._stop_monotonic_ns - self._start_monotonic_ns
                ),
                audio_state=self._tracks.state,
                audio=self._audio,
            )

    # -- internal helpers ----------------------------------------------------

    def _record_end_timestamps_locked(self) -> None:
        """Record end timestamps exactly once (first stop invocation)."""

        if self._ended_at is not None:
            return
        self._ended_at = _ensure_utc(self._wall_clock())
        self._stop_monotonic_ns = self._monotonic_ns()

    def _handle_start_failure(self, exc: MeetingAudioTracksStartError) -> None:
        """Process a start failure from the tracks layer.

        If a concurrent stop request already owns the lifecycle, the session
        state is NOT changed to FAILED.
        """

        with self._condition:
            self._audio = exc.result
            if self._stop_requested:
                # A stop thread already owns the lifecycle.  Don't overwrite.
                return
            self._record_end_timestamps_locked()
            self._state = MeetingSessionState.FAILED
            self._condition.notify_all()


__all__ = [
    "MeetingRemoteSourceKind",
    "MeetingSession",
    "MeetingSessionSnapshot",
    "MeetingSessionState",
    "MeetingSessionStateError",
]
