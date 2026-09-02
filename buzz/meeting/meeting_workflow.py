"""Pure-Python application-level meeting workflow orchestrator.

Coordinates ``MeetingStorage``, ``MeetingAudioTracks``, and
``MeetingSession`` into a single lifecycle owner.  No Qt, no
QSql, no UI, no threading primitives beyond a reentrant lock
for mutable ownership bookkeeping.

The caller owns the execution context (thread, event loop, etc.).
Blocking operations like ``start_capture`` and ``stop_capture``
may block for a significant time; PR40 will invoke them from a
Qt worker thread.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracks,
    MeetingAudioTracksResult,
    MeetingAudioTracksStartError,
    MeetingAudioTracksState,
    MeetingAudioTracksStopError,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSession,
    MeetingSessionSnapshot,
)
from buzz.meeting.meeting_storage import (
    MeetingStorage,
    MeetingStorageError,
    MeetingStoragePaths,
    StoredMeeting,
)
from buzz.audio_capture.source import AudioSource


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


class MeetingWorkflowState(Enum):
    """Application-oriented workflow lifecycle states.

    These are deliberately coarser than domain states.  The workflow
    does not mirror every ``MeetingSessionState`` or
    ``MeetingAudioTracksState`` value; it only tracks enough to
    enforce the one-active-meeting invariant and the
    capture/persistence separation.
    """

    IDLE = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()
    AWAITING_PERSISTENCE = auto()
    CLEANUP_REQUIRED = auto()


# ---------------------------------------------------------------------------
# Immutable result DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeetingWorkflowStartResult:
    """Returned on successful capture start."""

    session_id: uuid.UUID
    snapshot: MeetingSessionSnapshot


@dataclass(frozen=True, slots=True)
class MeetingWorkflowStopResult:
    """Returned on stop (whether or not cleanup succeeded)."""

    snapshot: MeetingSessionSnapshot
    audio: Optional[MeetingAudioTracksResult]
    cleanup_succeeded: bool


@dataclass(frozen=True, slots=True)
class MeetingWorkflowPersistResult:
    """Returned after a successful persistence."""

    session_id: uuid.UUID
    stored_meeting: StoredMeeting


# ---------------------------------------------------------------------------
# Workflow errors
# ---------------------------------------------------------------------------


class MeetingWorkflowError(Exception):
    """Base error for workflow failures."""


class MeetingWorkflowStateError(MeetingWorkflowError):
    """Raised when an operation is invalid for the current workflow state."""


class MeetingWorkflowPersistenceError(MeetingWorkflowError):
    """Raised when persistence fails.

    The workflow retains ownership so callers can retry.
    """

    def __init__(self, message: str, *, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class MeetingWorkflowPersistenceInProgressError(MeetingWorkflowError):
    """Raised when a concurrent persist() call is already in progress."""


class MeetingWorkflowStartError(MeetingWorkflowError):
    """Raised when capture start fails.

    Exposes the real domain snapshot so callers can persist the
    failure truthfully if desired.  The workflow retains ownership
    of the failed session until it is properly persisted.
    """

    def __init__(
        self,
        message: str,
        *,
        snapshot: MeetingSessionSnapshot,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.__cause__ = cause
        self.snapshot = snapshot


class MeetingWorkflowStopError(MeetingWorkflowError):
    """Raised when stop cleanup fails.

    Exposes the real domain snapshot and partial audio result so
    callers can persist the failure truthfully.
    """

    def __init__(
        self,
        message: str,
        *,
        snapshot: MeetingSessionSnapshot,
        audio: Optional[MeetingAudioTracksResult],
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.__cause__ = cause
        self.snapshot = snapshot
        self.audio = audio


# ---------------------------------------------------------------------------
# MeetingWorkflow
# ---------------------------------------------------------------------------


class MeetingWorkflow:
    """Manage one meeting's full lifecycle: prepare → start → stop → persist.

    Thread-safety: mutable ownership fields are guarded by ``_lock``.
    Blocking domain calls (``start``, ``stop``, storage I/O) execute
    **outside** the lock to avoid deadlock.

    The caller is responsible for execution context (main thread,
    worker thread, etc.).  PR40 will invoke blocking operations from
    a Qt worker thread and call persistence on the database-owner
    thread.
    """

    def __init__(self, meeting_storage: MeetingStorage) -> None:
        self._storage = meeting_storage
        self._lock = threading.Lock()
        # Mutable ownership state (guarded by _lock):
        self._state = MeetingWorkflowState.IDLE
        self._session: Optional[MeetingSession] = None
        self._session_id: Optional[uuid.UUID] = None
        self._pending_snapshot: Optional[MeetingSessionSnapshot] = None
        self._pending_audio: Optional[MeetingAudioTracksResult] = None
        self._paths: Optional[MeetingStoragePaths] = None
        # Persistence guard (guarded by _lock):
        self._persisting = False
        self._persisted_snapshot: Optional[MeetingSessionSnapshot] = None
        self._last_stored_meeting: Optional[StoredMeeting] = None

    # -- properties ----------------------------------------------------------

    @property
    def state(self) -> MeetingWorkflowState:
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state in (
                MeetingWorkflowState.STARTING,
                MeetingWorkflowState.ACTIVE,
                MeetingWorkflowState.STOPPING,
                MeetingWorkflowState.AWAITING_PERSISTENCE,
                MeetingWorkflowState.CLEANUP_REQUIRED,
            )

    @property
    def session_id(self) -> Optional[uuid.UUID]:
        with self._lock:
            return self._session_id

    def snapshot(self) -> Optional[MeetingSessionSnapshot]:
        """Return the current domain snapshot, if a session exists."""
        with self._lock:
            session = self._session
        if session is not None:
            return session.snapshot()
        with self._lock:
            return self._pending_snapshot

    # -- capture lifecycle ---------------------------------------------------

    def start_capture(
        self,
        microphone_source: AudioSource,
        remote_source: AudioSource,
        remote_source_kind: MeetingRemoteSourceKind,
    ) -> MeetingWorkflowStartResult:
        """Prepare storage, build tracks and session, start capture.

        Blocks until ``MeetingSession.start()`` completes.  On
        success the workflow transitions to ``ACTIVE``.  On failure
        the workflow transitions to ``AWAITING_PERSISTENCE`` or
        ``CLEANUP_REQUIRED`` depending on domain cleanup status,
        and raises ``MeetingWorkflowStartError`` exposing the real
        domain snapshot.  Ownership is retained so the failed
        meeting can be persisted.

        Raises ``MeetingWorkflowStateError`` if a meeting is
        already active.
        Raises ``MeetingWorkflowStartError`` if capture start fails.
        Raises ``MeetingWorkflowError`` if storage preparation fails.
        """

        session_id = uuid.uuid4()

        # --- enforce one-active-meeting invariant ---
        with self._lock:
            if self._state != MeetingWorkflowState.IDLE:
                raise MeetingWorkflowStateError(
                    f"Cannot start a new meeting while workflow is "
                    f"{self._state.name}"
                )
            self._state = MeetingWorkflowState.STARTING
            self._session_id = session_id

        # --- prepare storage (outside lock) ---
        try:
            paths = self._storage.prepare(session_id)
        except MeetingStorageError as exc:
            with self._lock:
                self._state = MeetingWorkflowState.IDLE
                self._session_id = None
            raise MeetingWorkflowError(f"Storage preparation failed: {exc}") from exc

        with self._lock:
            self._paths = paths

        # --- construct tracks and session (outside lock) ---
        tracks = MeetingAudioTracks(
            microphone_source=microphone_source,
            remote_source=remote_source,
            microphone_output_path=paths.microphone,
            remote_output_path=paths.remote,
        )
        session = MeetingSession(
            tracks=tracks,
            remote_source_kind=remote_source_kind,
            session_id=session_id,
        )

        with self._lock:
            self._session = session

        # --- start capture (outside lock, blocks) ---
        try:
            session.start()
        except MeetingAudioTracksStartError as exc:
            snapshot = session.snapshot()
            # The tracks layer always cleans up on start failure.
            # Classify cleanup status from domain state.
            cleanup_complete = snapshot.audio_state == MeetingAudioTracksState.STOPPED
            with self._lock:
                self._pending_snapshot = snapshot
                self._pending_audio = exc.result
                self._persisted_snapshot = None
                if cleanup_complete:
                    self._state = MeetingWorkflowState.AWAITING_PERSISTENCE
                else:
                    self._state = MeetingWorkflowState.CLEANUP_REQUIRED
            raise MeetingWorkflowStartError(
                f"Meeting capture start failed: {exc}",
                snapshot=snapshot,
                cause=exc,
            )

        snapshot = session.snapshot()
        with self._lock:
            self._state = MeetingWorkflowState.ACTIVE

        return MeetingWorkflowStartResult(
            session_id=session_id,
            snapshot=snapshot,
        )

    # -- stop ----------------------------------------------------------------

    def stop_capture(self) -> MeetingWorkflowStopResult:
        """Stop capture and snapshot the result.

        Blocks until ``MeetingSession.stop()`` completes.  Does
        **not** call ``MeetingStorage.save()`` — persistence is a
        separate explicit phase.

        On success the workflow transitions to
        ``AWAITING_PERSISTENCE``.  On cleanup failure the workflow
        transitions to ``CLEANUP_REQUIRED``.

        Raises ``MeetingWorkflowStateError`` if no meeting is
        active.
        """

        session = self._get_active_session_or_raise()

        with self._lock:
            self._state = MeetingWorkflowState.STOPPING

        # --- stop capture (outside lock, blocks) ---
        try:
            audio = session.stop()
            cleanup_succeeded = True
        except MeetingAudioTracksStopError as exc:
            audio = exc.result
            cleanup_succeeded = False

        snapshot = session.snapshot()

        with self._lock:
            self._pending_snapshot = snapshot
            self._pending_audio = audio
            self._persisted_snapshot = None
            if cleanup_succeeded:
                self._state = MeetingWorkflowState.AWAITING_PERSISTENCE
            else:
                self._state = MeetingWorkflowState.CLEANUP_REQUIRED

        if not cleanup_succeeded:
            raise MeetingWorkflowStopError(
                "Meeting stop cleanup failed",
                snapshot=snapshot,
                audio=audio,
            )

        return MeetingWorkflowStopResult(
            snapshot=snapshot,
            audio=audio,
            cleanup_succeeded=True,
        )

    # -- cleanup retry -------------------------------------------------------

    def retry_cleanup(self) -> MeetingWorkflowStopResult:
        """Retry stop cleanup for a failed/stopping session.

        Delegates to ``MeetingSession.stop()`` again.  If the
        domain reaches ``COMPLETED``, transitions to
        ``AWAITING_PERSISTENCE``.  Otherwise remains in
        ``CLEANUP_REQUIRED``.

        Raises ``MeetingWorkflowStateError`` if the workflow is
        not in ``CLEANUP_REQUIRED`` state.
        """

        session = self._get_session_or_raise_for_cleanup()

        try:
            audio = session.stop()
            cleanup_succeeded = True
        except MeetingAudioTracksStopError as exc:
            audio = exc.result
            cleanup_succeeded = False

        snapshot = session.snapshot()

        with self._lock:
            self._pending_snapshot = snapshot
            self._pending_audio = audio
            self._persisted_snapshot = None
            if cleanup_succeeded:
                self._state = MeetingWorkflowState.AWAITING_PERSISTENCE
            # else: remain CLEANUP_REQUIRED

        if not cleanup_succeeded:
            raise MeetingWorkflowStopError(
                "Meeting cleanup retry still failed",
                snapshot=snapshot,
                audio=audio,
            )

        return MeetingWorkflowStopResult(
            snapshot=snapshot,
            audio=audio,
            cleanup_succeeded=True,
        )

    # -- persistence ---------------------------------------------------------

    def persist(self) -> MeetingWorkflowPersistResult:
        """Persist the pending snapshot via ``MeetingStorage.save()``.

        Must be called from the database-owner thread.  Does not
        create or switch threads.

        In ``AWAITING_PERSISTENCE`` state, on success the workflow
        transitions to ``IDLE`` and becomes reusable for a new
        meeting.

        In ``CLEANUP_REQUIRED`` state, on success the snapshot is
        persisted but ownership is retained so ``retry_cleanup()``
        remains available.  A second ``persist()`` with the same
        pending snapshot is a no-op (returns the stored meeting
        without re-saving).

        Raises ``MeetingWorkflowStateError`` if there is no
        pending snapshot.
        Raises ``MeetingWorkflowPersistenceError`` if save fails
        (workflow retains ownership for retry).
        Raises ``MeetingWorkflowPersistenceInProgressError`` if
        another persist() call is already in progress.
        """

        with self._lock:
            if self._state not in (
                MeetingWorkflowState.AWAITING_PERSISTENCE,
                MeetingWorkflowState.CLEANUP_REQUIRED,
            ):
                raise MeetingWorkflowStateError(
                    f"Cannot persist while workflow is {self._state.name}"
                )
            if self._persisting:
                raise MeetingWorkflowPersistenceInProgressError(
                    "A persistence operation is already in progress"
                )
            snapshot = self._pending_snapshot
            session_id = self._session_id
            state = self._state

        if snapshot is None:
            raise MeetingWorkflowStateError("No pending snapshot to persist")
        if session_id is None:
            raise MeetingWorkflowStateError("No session ID for persistence")
        if snapshot.session_id != session_id:
            raise MeetingWorkflowStateError(
                "Pending snapshot belongs to a different session"
            )

        with self._lock:
            if snapshot is self._persisted_snapshot:
                # Same snapshot already persisted — no-op.
                return MeetingWorkflowPersistResult(
                    session_id=session_id,
                    stored_meeting=self._last_stored_meeting,  # type: ignore[attr-defined]
                )
            self._persisting = True

        try:
            stored = self._storage.save(snapshot)
        except MeetingStorageError as exc:
            with self._lock:
                self._persisting = False
            raise MeetingWorkflowPersistenceError(
                f"Persistence failed: {exc}",
                cause=exc,
            ) from exc

        with self._lock:
            self._persisting = False
            if state == MeetingWorkflowState.CLEANUP_REQUIRED:
                self._persisted_snapshot = snapshot
                self._last_stored_meeting = stored  # type: ignore[attr-defined]
            else:
                self._clear_ownership_locked()

        return MeetingWorkflowPersistResult(
            session_id=session_id,
            stored_meeting=stored,
        )

    # -- internal helpers ----------------------------------------------------

    def _get_active_session_or_raise(self) -> MeetingSession:
        with self._lock:
            if self._state not in (
                MeetingWorkflowState.ACTIVE,
                MeetingWorkflowState.STARTING,
            ):
                raise MeetingWorkflowStateError(
                    f"Cannot stop: workflow is {self._state.name}"
                )
            session = self._session
        if session is None:
            raise MeetingWorkflowStateError("Workflow is active but has no session")
        return session

    def _get_session_or_raise_for_cleanup(self) -> MeetingSession:
        with self._lock:
            if self._state != MeetingWorkflowState.CLEANUP_REQUIRED:
                raise MeetingWorkflowStateError(
                    f"Cannot retry cleanup: workflow is {self._state.name}"
                )
            session = self._session
        if session is None:
            raise MeetingWorkflowStateError("Workflow needs cleanup but has no session")
        return session

    def _clear_ownership_locked(self) -> None:
        """Release all owned meeting state.  Caller must hold ``_lock``."""
        self._state = MeetingWorkflowState.IDLE
        self._session = None
        self._session_id = None
        self._pending_snapshot = None
        self._pending_audio = None
        self._paths = None
        self._persisted_snapshot = None


__all__ = [
    "MeetingWorkflow",
    "MeetingWorkflowError",
    "MeetingWorkflowPersistResult",
    "MeetingWorkflowPersistenceError",
    "MeetingWorkflowPersistenceInProgressError",
    "MeetingWorkflowStartError",
    "MeetingWorkflowState",
    "MeetingWorkflowStateError",
    "MeetingWorkflowStopError",
    "MeetingWorkflowStopResult",
    "MeetingWorkflowStartResult",
]
