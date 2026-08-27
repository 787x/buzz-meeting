"""Pure final-transcription domain, service, and timeline mapper.

No Qt, QSql, Settings, or concrete FileTranscriber imports.
All types are frozen dataclasses or pure Python protocols.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Protocol

from buzz.meeting.meeting_audio_tracks import MeetingTrackRole
from buzz.meeting.meeting_storage import (
    StoredMeeting,
    StoredMeetingAudioTrack,
    StoredMeetingTimingAnchor,
)
from buzz.meeting.meeting_session import MeetingSessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FinalTranscriptionError(Exception):
    """Base error for final transcription failures."""


class FinalTranscriptionConfigError(FinalTranscriptionError):
    """Raised for invalid transcription configuration."""


class FinalTranscriptionEligibilityError(FinalTranscriptionError):
    """Raised when a meeting or track is ineligible for transcription."""


class FinalTranscriptionConflictError(FinalTranscriptionError):
    """Raised when a request conflicts with an existing generation."""


class FinalTranscriptionStateError(FinalTranscriptionError):
    """Raised for invalid generation lifecycle operations."""


class FinalTranscriptionDecodeError(FinalTranscriptionError):
    """Raised when persisted data is corrupt or contains unknown values."""


class TimelineMappingError(FinalTranscriptionError):
    """Raised when track-local time cannot be mapped to meeting time."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FinalTranscriptionStatus(Enum):
    QUEUED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    PARTIAL = auto()
    FAILED = auto()


class FinalTranscriptionTrackStatus(Enum):
    QUEUED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    INELIGIBLE = auto()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROFILE_VERSION_1 = 1
_KNOWN_PROFILE_VERSIONS = {_PROFILE_VERSION_1}
_WHISPER_MODEL_TYPES = frozenset({"WHISPER", "WHISPER_CPP", "FASTER_WHISPER"})
_SUPPORTED_MODEL_TYPES = _WHISPER_MODEL_TYPES | {"HUGGING_FACE"}
_KNOWN_WHISPER_MODEL_SIZES = frozenset(
    {
        "TINY",
        "TINYEN",
        "BASE",
        "BASEEN",
        "SMALL",
        "SMALLEN",
        "MEDIUM",
        "MEDIUMEN",
        "LARGE",
        "LARGEV2",
        "LARGEV3",
        "LARGEV3TURBO",
        "CUSTOM",
        "LUMII",
    }
)


@dataclass(frozen=True, slots=True)
class FinalTranscriptionConfig:
    """Immutable, persistable transcription configuration snapshot.

    Created at request time and stored durably. Recovery uses persisted
    config, never re-derives from current Settings.
    """

    profile_version: int = _PROFILE_VERSION_1
    model_type: str = "FASTER_WHISPER"
    whisper_model_size: Optional[str] = "TINY"
    hugging_face_model_id: str = ""
    language: Optional[str] = None

    def __post_init__(self) -> None:
        if self.profile_version not in _KNOWN_PROFILE_VERSIONS:
            raise FinalTranscriptionConfigError(
                f"Unknown profile_version: {self.profile_version}"
            )
        if self.model_type not in _SUPPORTED_MODEL_TYPES:
            raise FinalTranscriptionConfigError(
                f"Unsupported model_type: {self.model_type!r}"
            )
        if self.model_type in _WHISPER_MODEL_TYPES:
            if not self.whisper_model_size:
                raise FinalTranscriptionConfigError(
                    f"model_type {self.model_type} requires whisper_model_size"
                )
            if self.whisper_model_size not in _KNOWN_WHISPER_MODEL_SIZES:
                raise FinalTranscriptionConfigError(
                    f"Unknown whisper_model_size: {self.whisper_model_size!r}"
                )
            if self.hugging_face_model_id:
                raise FinalTranscriptionConfigError(
                    f"model_type {self.model_type} must not have "
                    "hugging_face_model_id"
                )
        elif self.model_type == "HUGGING_FACE":
            if not self.hugging_face_model_id:
                raise FinalTranscriptionConfigError(
                    "HUGGING_FACE requires hugging_face_model_id"
                )
            if self.whisper_model_size is not None:
                raise FinalTranscriptionConfigError(
                    "HUGGING_FACE must have whisper_model_size=None"
                )
        if self.language is not None and not self.language:
            raise FinalTranscriptionConfigError(
                "language must be None or a nonempty string"
            )


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinalTranscriptionGeneration:
    """Immutable view of a final-transcription generation."""

    generation_id: uuid.UUID
    meeting_id: uuid.UUID
    profile_version: int
    status: FinalTranscriptionStatus
    config: FinalTranscriptionConfig
    tracks: tuple[FinalTranscriptionTrack, ...]
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass(frozen=True, slots=True)
class FinalTranscriptionTrack:
    """Immutable view of a track within a generation."""

    role: MeetingTrackRole
    status: FinalTranscriptionTrackStatus
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    segment_count: int = 0


@dataclass(frozen=True, slots=True)
class MeetingTranscriptSegment:
    """One segment in the merged meeting transcript projection."""

    merged_ordinal: int
    source_role: MeetingTrackRole
    source_track_ordinal: int
    local_start_ms: int
    local_end_ms: int
    start_ns: int
    end_ns: int
    text: str


@dataclass(frozen=True, slots=True)
class MeetingTranscript:
    """The merged meeting transcript from a completed/partial generation."""

    generation_id: uuid.UUID
    meeting_id: uuid.UUID
    status: FinalTranscriptionStatus
    segments: tuple[MeetingTranscriptSegment, ...]


@dataclass(frozen=True, slots=True)
class TrackTranscriptionInputSegment:
    """Pure adapter result from FileTranscriber — no Qt types."""

    start_ms: int
    end_ms: int
    text: str


# ---------------------------------------------------------------------------
# Timeline mapper
# ---------------------------------------------------------------------------

_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000


def map_track_time_to_meeting_ns(
    local_ms: int,
    sample_rate: int,
    anchors: tuple[StoredMeetingTimingAnchor, ...],
) -> int:
    """Map track-local milliseconds to meeting-relative nanoseconds.

    Uses integer piecewise-linear interpolation/extrapolation between
    stored timing anchors.  The meeting epoch is the audio coordinator
    start (PR8 ``coordinator_start_ns``), NOT the MeetingSession start.

    Raises ``TimelineMappingError`` on:
      - zero anchors
      - non-monotonic meeting-time anchors
    """
    if sample_rate <= 0:
        raise TimelineMappingError(f"sample_rate must be positive: {sample_rate}")
    if not isinstance(local_ms, int) or isinstance(local_ms, bool):
        raise TimelineMappingError(f"local_ms must be int, got {type(local_ms)}")
    if local_ms < 0:
        raise TimelineMappingError(f"local_ms must be >= 0: {local_ms}")

    local_ns = local_ms * _NANOSECONDS_PER_MILLISECOND

    if len(anchors) == 0:
        raise TimelineMappingError("Zero timing anchors: track is timeline-ineligible")

    # Pre-compute anchor local ns and meeting ns
    anchor_local_ns: list[int] = []
    anchor_meeting_ns: list[int] = []
    for anchor in anchors:
        anc_local = anchor.sample_end * _NANOSECONDS_PER_SECOND // sample_rate
        anchor_local_ns.append(anc_local)
        anchor_meeting_ns.append(anchor.callback_arrival_offset_ns)

    # Single anchor: constant offset
    if len(anchors) == 1:
        offset = anchor_meeting_ns[0] - anchor_local_ns[0]
        return local_ns + offset

    # Validate meeting-time monotonicity (sample_end monotonicity is
    # enforced by PR10 storage)
    for i in range(1, len(anchor_meeting_ns)):
        if anchor_meeting_ns[i] <= anchor_meeting_ns[i - 1]:
            raise TimelineMappingError(
                f"Non-monotonic meeting-time anchors at index {i}: "
                f"{anchor_meeting_ns[i - 1]} >= {anchor_meeting_ns[i]}"
            )

    # Before first anchor: extrapolate using first two
    if local_ns <= anchor_local_ns[0]:
        return _interpolate(
            local_ns,
            anchor_local_ns[0],
            anchor_local_ns[1],
            anchor_meeting_ns[0],
            anchor_meeting_ns[1],
        )

    # After last anchor: extrapolate using last two
    if local_ns >= anchor_local_ns[-1]:
        return _interpolate(
            local_ns,
            anchor_local_ns[-2],
            anchor_local_ns[-1],
            anchor_meeting_ns[-2],
            anchor_meeting_ns[-1],
        )

    # Between: find bracketing pair
    for i in range(len(anchor_local_ns) - 1):
        if anchor_local_ns[i] <= local_ns < anchor_local_ns[i + 1]:
            return _interpolate(
                local_ns,
                anchor_local_ns[i],
                anchor_local_ns[i + 1],
                anchor_meeting_ns[i],
                anchor_meeting_ns[i + 1],
            )

    raise TimelineMappingError("Internal error: bracket not found")


def _interpolate(
    x: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> int:
    """Integer piecewise-linear interpolation/extrapolation.

    Computes: y0 + (x - x0) * (y1 - y0) // (x1 - x0)

    Uses Python floor-division semantics (deterministic for negative
    operands — rounds toward negative infinity).
    """
    return y0 + (x - x0) * (y1 - y0) // (x1 - x0)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def check_track_eligibility(
    track: StoredMeetingAudioTrack,
) -> Optional[str]:
    """Return None if track is eligible, or a reason string if not.

    Eligible means: published, nonempty, asset exists, and has at
    least one timing anchor with monotonic meeting-time offsets.
    Does NOT require complete=True.
    """
    if not track.published:
        return "track not published"
    if track.sample_count <= 0:
        return "track has zero samples"
    if not track.asset_exists_at_load:
        return "audio asset missing"

    # Timeline preflight
    if len(track.timing_anchors) == 0:
        return "zero timing anchors"

    # Validate monotonicity of meeting-time offsets
    for i in range(1, len(track.timing_anchors)):
        if (
            track.timing_anchors[i].callback_arrival_offset_ns
            <= track.timing_anchors[i - 1].callback_arrival_offset_ns
        ):
            return "non-monotonic meeting-time anchors"

    return None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class MeetingTranscriptionRepository(Protocol):
    """Pure persistence boundary for final-transcription data.

    All methods must run on the DB-owner thread.
    """

    def create_generation(
        self,
        generation_id: str,
        meeting_id: str,
        config: FinalTranscriptionConfig,
        initial_status: str,
        time_created: str,
        time_completed: Optional[str],
        tracks: tuple[TrackPersistenceRecord, ...],
    ) -> None:
        """Atomically create generation + track rows.

        ``initial_status`` is the correct generation status derived
        from track states (QUEUED or FAILED).  ``time_completed`` is
        non-None only for terminal initial states.
        """
        ...

    def find_generation_by_key(
        self,
        meeting_id: str,
        profile_version: int,
    ) -> Optional[GenerationPersistenceRecord]:
        """Return existing generation for idempotency check."""
        ...

    def load_generation(
        self,
        generation_id: str,
    ) -> Optional[GenerationPersistenceRecord]:
        """Load generation with tracks."""
        ...

    def load_tracks(
        self,
        generation_id: str,
    ) -> tuple[TrackPersistenceRecord, ...]:
        """Load all tracks for a generation."""
        ...

    def load_segments(
        self,
        generation_id: str,
        role: str,
    ) -> tuple[SegmentPersistenceRecord, ...]:
        """Load segments for a specific track."""
        ...

    def begin_track(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        """Atomically mark track IN_PROGRESS and generation IN_PROGRESS.

        Sets time_started on track and generation (if NULL).
        """
        ...

    def complete_track(
        self,
        generation_id: str,
        role: str,
        segments: tuple[SegmentPersistenceRecord, ...],
        now: str,
    ) -> None:
        """Atomically replace segments, mark track COMPLETED, derive
        generation status."""
        ...

    def fail_track(
        self,
        generation_id: str,
        role: str,
        error_message: str,
        now: str,
    ) -> None:
        """Atomically mark track FAILED, derive generation status."""
        ...

    def mark_track_ineligible(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        """Mark track INELIGIBLE, derive generation status."""
        ...

    def update_generation_status(
        self,
        generation_id: str,
        now: str,
    ) -> None:
        """Derive and update generation status from current track statuses."""
        ...

    def reset_for_retry(
        self,
        generation_id: str,
        desired_track_statuses: dict[str, str],
        now: str,
    ) -> None:
        """Atomically reset non-COMPLETED tracks to desired states.

        ``desired_track_statuses`` maps role strings to the exact
        desired status (QUEUED or INELIGIBLE) for each non-COMPLETED
        role.  COMPLETED roles are preserved untouched.

        Transaction: reset segments/statuses for non-COMPLETED,
        recompute generation status, clear generation error.
        All-or-nothing.
        """
        ...

    def load_recoverable_generations(
        self,
    ) -> tuple[GenerationPersistenceRecord, ...]:
        """Load QUEUED and IN_PROGRESS generations for recovery."""
        ...

    def reset_in_progress_tracks(
        self,
        generation_id: str,
    ) -> None:
        """Reset IN_PROGRESS tracks to QUEUED for recovery."""
        ...


class TranscriptionRunner(Protocol):
    """Runs ASR on a single audio file, returning pure segment DTOs."""

    def transcribe_track(
        self,
        audio_path: str,
        sample_rate: int,
        config: FinalTranscriptionConfig,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> tuple[TrackTranscriptionInputSegment, ...]:
        """Transcribe one audio file. Returns segments. Raises on error."""
        ...

    def shutdown(self) -> None:
        """Stop any in-flight work for graceful shutdown."""
        ...


# ---------------------------------------------------------------------------
# Persistence records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationPersistenceRecord:
    """SQLite-compatible generation record."""

    id: str
    meeting_id: str
    profile_version: int
    status: str
    config_model_type: str
    config_whisper_model_size: Optional[str]
    config_hugging_face_model_id: str
    config_language: Optional[str]
    error_message: Optional[str]
    time_created: str
    time_started: Optional[str]
    time_completed: Optional[str]


@dataclass(frozen=True, slots=True)
class TrackPersistenceRecord:
    """SQLite-compatible track record."""

    generation_id: str
    role: str
    status: str
    error_message: Optional[str]
    time_started: Optional[str]
    time_completed: Optional[str]
    segment_count: int


@dataclass(frozen=True, slots=True)
class SegmentPersistenceRecord:
    """SQLite-compatible segment record."""

    generation_id: str
    role: str
    ordinal: int
    local_start_ms: int
    local_end_ms: int
    start_ns: int
    end_ns: int
    text: str


# ---------------------------------------------------------------------------
# Encoding/decoding helpers
# ---------------------------------------------------------------------------


def encode_generation_status(status: FinalTranscriptionStatus) -> str:
    return status.name


def decode_generation_status(raw: str) -> FinalTranscriptionStatus:
    try:
        return FinalTranscriptionStatus[raw]
    except KeyError:
        raise FinalTranscriptionDecodeError(f"Unknown generation status: {raw!r}")


def encode_track_status(status: FinalTranscriptionTrackStatus) -> str:
    return status.name


def decode_track_status(raw: str) -> FinalTranscriptionTrackStatus:
    try:
        return FinalTranscriptionTrackStatus[raw]
    except KeyError:
        raise FinalTranscriptionDecodeError(f"Unknown track status: {raw!r}")


def encode_role(role: MeetingTrackRole) -> str:
    return role.name


def decode_role(raw: str) -> MeetingTrackRole:
    try:
        return MeetingTrackRole[raw]
    except KeyError:
        raise FinalTranscriptionDecodeError(f"Unknown role: {raw!r}")


def encode_config(config: FinalTranscriptionConfig) -> dict[str, Optional[str]]:
    return {
        "config_model_type": config.model_type,
        "config_whisper_model_size": config.whisper_model_size,
        "config_hugging_face_model_id": config.hugging_face_model_id,
        "config_language": config.language,
    }


def decode_config(
    profile_version: int,
    model_type: str,
    whisper_model_size: Optional[str],
    hugging_face_model_id: str,
    language: Optional[str],
) -> FinalTranscriptionConfig:
    return FinalTranscriptionConfig(
        profile_version=profile_version,
        model_type=model_type,
        whisper_model_size=whisper_model_size
        if whisper_model_size is not None
        else None,
        hugging_face_model_id=hugging_face_model_id,
        language=language if language is not None else None,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_datetime(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise FinalTranscriptionDecodeError(f"Malformed timestamp: {raw!r}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise FinalTranscriptionDecodeError(
            f"Timestamp must be timezone-aware: {raw!r}"
        )
    return value.astimezone(timezone.utc)


def encode_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise FinalTranscriptionError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _safe_error_message(error: object) -> str:
    """Extract bounded error message string, never raising."""
    try:
        raw = str(error)
    except Exception:
        raw = f"<{type(error).__name__}: str() failed>"
    if not raw:
        raw = f"<{type(error).__name__}: empty message>"
    return raw[:4096]


# ---------------------------------------------------------------------------
# Generation status derivation
# ---------------------------------------------------------------------------

_TRACK_STATUS_RANK = {
    FinalTranscriptionTrackStatus.IN_PROGRESS: 0,
    FinalTranscriptionTrackStatus.QUEUED: 1,
    FinalTranscriptionTrackStatus.COMPLETED: 2,
    FinalTranscriptionTrackStatus.FAILED: 3,
    FinalTranscriptionTrackStatus.INELIGIBLE: 4,
}


def derive_generation_status(
    track_statuses: tuple[FinalTranscriptionTrackStatus, ...],
) -> FinalTranscriptionStatus:
    """Derive generation status from current track statuses.

    Rules:
      - any IN_PROGRESS → IN_PROGRESS
      - any QUEUED → QUEUED
      - all COMPLETED → COMPLETED
      - any COMPLETED → PARTIAL (others terminal non-success)
      - else → FAILED
    """
    if any(s is FinalTranscriptionTrackStatus.IN_PROGRESS for s in track_statuses):
        return FinalTranscriptionStatus.IN_PROGRESS
    if any(s is FinalTranscriptionTrackStatus.QUEUED for s in track_statuses):
        return FinalTranscriptionStatus.QUEUED
    completed_count = sum(
        1 for s in track_statuses if s is FinalTranscriptionTrackStatus.COMPLETED
    )
    if completed_count == len(track_statuses):
        return FinalTranscriptionStatus.COMPLETED
    if completed_count > 0:
        return FinalTranscriptionStatus.PARTIAL
    return FinalTranscriptionStatus.FAILED


def is_terminal(status: FinalTranscriptionStatus) -> bool:
    return status in (
        FinalTranscriptionStatus.COMPLETED,
        FinalTranscriptionStatus.PARTIAL,
        FinalTranscriptionStatus.FAILED,
    )


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def assemble_generation(
    rec: GenerationPersistenceRecord,
    track_recs: tuple[TrackPersistenceRecord, ...],
) -> FinalTranscriptionGeneration:
    """Build a generation DTO from persistence records."""
    config = decode_config(
        rec.profile_version,
        rec.config_model_type,
        rec.config_whisper_model_size,
        rec.config_hugging_face_model_id,
        rec.config_language,
    )
    tracks = tuple(
        FinalTranscriptionTrack(
            role=decode_role(tr.role),
            status=decode_track_status(tr.status),
            error_message=tr.error_message,
            started_at=decode_datetime(tr.time_started),
            completed_at=decode_datetime(tr.time_completed),
            segment_count=tr.segment_count,
        )
        for tr in track_recs
    )
    return FinalTranscriptionGeneration(
        generation_id=uuid.UUID(rec.id),
        meeting_id=uuid.UUID(rec.meeting_id),
        profile_version=rec.profile_version,
        status=decode_generation_status(rec.status),
        config=config,
        tracks=tracks,
        error_message=rec.error_message,
        created_at=decode_datetime(rec.time_created),
        started_at=decode_datetime(rec.time_started),
        completed_at=decode_datetime(rec.time_completed),
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

_ROLE_ORDER: tuple[MeetingTrackRole, MeetingTrackRole] = (
    MeetingTrackRole.MICROPHONE,
    MeetingTrackRole.REMOTE,
)


class FinalTranscriptionService:
    """Orchestrate final meeting transcription.

    Pure Python — no Qt, no QSql, no Settings.  Uses injected
    ``MeetingTranscriptionRepository`` for persistence and
    ``TranscriptionRunner`` for ASR execution.

    At most one active track ASR globally per service instance.
    Within a generation, tracks are processed in stable role order:
    MICROPHONE then REMOTE.
    """

    def __init__(
        self,
        meeting_storage: object,
        repository: MeetingTranscriptionRepository,
        runner: TranscriptionRunner,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._meeting_storage = meeting_storage
        self._repository = repository
        self._runner = runner
        self._clock = clock
        self._queue: deque[tuple[str, MeetingTrackRole]] = deque()
        self._active = False
        self._shutdown_requested = False

    # -- public API ---------------------------------------------------------

    def request(
        self,
        meeting_id: uuid.UUID,
        config: FinalTranscriptionConfig,
    ) -> FinalTranscriptionGeneration:
        """Request final transcription for a completed meeting.

        Idempotent: returns existing generation for same key if config
        matches. Raises ``FinalTranscriptionConflictError`` if config
        differs.
        """
        meeting_id_str = str(meeting_id)

        # Validate meeting exists and is COMPLETED
        stored = self._load_meeting(meeting_id)
        if stored is None:
            raise FinalTranscriptionEligibilityError(f"Meeting {meeting_id} not found")
        if stored.state is not MeetingSessionState.COMPLETED:
            raise FinalTranscriptionEligibilityError(
                f"Meeting {meeting_id} is {stored.state.name}, " "not COMPLETED"
            )

        # Idempotency check
        existing = self._repository.find_generation_by_key(
            meeting_id_str, config.profile_version
        )
        if existing is not None:
            # Verify config matches
            existing_config = decode_config(
                existing.profile_version,
                existing.config_model_type,
                existing.config_whisper_model_size,
                existing.config_hugging_face_model_id,
                existing.config_language,
            )
            if existing_config != config:
                raise FinalTranscriptionConflictError(
                    f"Generation already exists for meeting {meeting_id} "
                    f"profile v{config.profile_version} with different config"
                )
            track_recs = self._repository.load_tracks(existing.id)
            return assemble_generation(existing, track_recs)

        # Create generation
        return self._create_generation(meeting_id, stored, config)

    def retry(
        self,
        generation_id: uuid.UUID,
    ) -> FinalTranscriptionGeneration:
        """Explicit retry for FAILED or PARTIAL generations.

        Preserves COMPLETED tracks and their segments. Re-evaluates
        eligibility for non-COMPLETED roles. All desired track states
        are computed before the single atomic repository transaction.
        """
        gen_rec = self._repository.load_generation(str(generation_id))
        if gen_rec is None:
            raise FinalTranscriptionStateError(f"Generation {generation_id} not found")

        status = decode_generation_status(gen_rec.status)
        if status is FinalTranscriptionStatus.COMPLETED:
            raise FinalTranscriptionStateError("Cannot retry a COMPLETED generation")
        if status in (
            FinalTranscriptionStatus.QUEUED,
            FinalTranscriptionStatus.IN_PROGRESS,
        ):
            raise FinalTranscriptionStateError(
                f"Cannot retry generation in {status.name} state"
            )

        now = self._clock()
        now_str = encode_datetime(now)
        assert now_str is not None

        # Load current tracks to know which are COMPLETED
        current_tracks = self._repository.load_tracks(str(generation_id))

        # Compute desired statuses for all non-COMPLETED roles
        stored = self._load_meeting(uuid.UUID(gen_rec.meeting_id))
        desired_track_statuses: dict[str, str] = {}

        for tr in current_tracks:
            track_status = decode_track_status(tr.status)
            if track_status is FinalTranscriptionTrackStatus.COMPLETED:
                continue  # preserved, not in desired map

            role = decode_role(tr.role)
            if stored is not None:
                audio_track = self._get_track(stored, role)
                if audio_track is not None:
                    reason = check_track_eligibility(audio_track)
                    if reason is None:
                        desired_track_statuses[tr.role] = encode_track_status(
                            FinalTranscriptionTrackStatus.QUEUED
                        )
                        continue

            desired_track_statuses[tr.role] = encode_track_status(
                FinalTranscriptionTrackStatus.INELIGIBLE
            )

        # Single atomic transaction
        self._repository.reset_for_retry(
            str(generation_id), desired_track_statuses, now_str
        )

        # Reload final state
        gen_rec = self._repository.load_generation(str(generation_id))
        assert gen_rec is not None
        track_recs = self._repository.load_tracks(str(generation_id))

        # Schedule any QUEUED tracks AFTER commit
        self._schedule_generation_tracks(str(generation_id), track_recs)

        return assemble_generation(gen_rec, track_recs)

    def recover_pending(self) -> tuple[uuid.UUID, ...]:
        """Recover interrupted generations on startup.

        Returns IDs of generations that were recovered and scheduled.
        """
        if self._shutdown_requested:
            return ()

        recoverable = self._repository.load_recoverable_generations()
        recovered_ids: list[uuid.UUID] = []

        for gen_rec in recoverable:
            gid = gen_rec.id
            status = decode_generation_status(gen_rec.status)

            if status is FinalTranscriptionStatus.IN_PROGRESS:
                self._repository.reset_in_progress_tracks(gid)

            track_recs = self._repository.load_tracks(gid)
            self._schedule_generation_tracks(gid, track_recs)
            recovered_ids.append(uuid.UUID(gid))

        return tuple(recovered_ids)

    def load_generation(
        self,
        generation_id: uuid.UUID,
    ) -> Optional[FinalTranscriptionGeneration]:
        """Load a generation by ID, or None if not found."""
        gen_rec = self._repository.load_generation(str(generation_id))
        if gen_rec is None:
            return None
        track_recs = self._repository.load_tracks(gen_rec.id)
        return assemble_generation(gen_rec, track_recs)

    def load_transcript(
        self,
        generation_id: uuid.UUID,
    ) -> Optional[MeetingTranscript]:
        """Load the merged transcript projection for a generation.

        Returns None for FAILED/QUEUED/IN_PROGRESS generations, or for
        generations with zero completed tracks.
        """
        gen_rec = self._repository.load_generation(str(generation_id))
        if gen_rec is None:
            return None

        status = decode_generation_status(gen_rec.status)
        if status not in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
        ):
            return None

        track_recs = self._repository.load_tracks(gen_rec.id)

        # Collect segments from COMPLETED tracks only
        all_segments: list[MeetingTranscriptSegment] = []
        for tr in track_recs:
            track_status = decode_track_status(tr.status)
            if track_status is not FinalTranscriptionTrackStatus.COMPLETED:
                continue
            role = decode_role(tr.role)
            seg_recs = self._repository.load_segments(gen_rec.id, tr.role)
            for sr in seg_recs:
                all_segments.append(
                    MeetingTranscriptSegment(
                        merged_ordinal=0,  # assigned below
                        source_role=role,
                        source_track_ordinal=sr.ordinal,
                        local_start_ms=sr.local_start_ms,
                        local_end_ms=sr.local_end_ms,
                        start_ns=sr.start_ns,
                        end_ns=sr.end_ns,
                        text=sr.text,
                    )
                )

        # Stable sort: start_ns, role order, local ordinal
        role_order = {
            MeetingTrackRole.MICROPHONE: 0,
            MeetingTrackRole.REMOTE: 1,
        }
        all_segments.sort(
            key=lambda s: (
                s.start_ns,
                role_order.get(s.source_role, 99),
                s.source_track_ordinal,
            )
        )

        # Assign merged ordinals
        merged = tuple(
            MeetingTranscriptSegment(
                merged_ordinal=i,
                source_role=seg.source_role,
                source_track_ordinal=seg.source_track_ordinal,
                local_start_ms=seg.local_start_ms,
                local_end_ms=seg.local_end_ms,
                start_ns=seg.start_ns,
                end_ns=seg.end_ns,
                text=seg.text,
            )
            for i, seg in enumerate(all_segments)
        )

        return MeetingTranscript(
            generation_id=uuid.UUID(gen_rec.id),
            meeting_id=uuid.UUID(gen_rec.meeting_id),
            status=status,
            segments=merged,
        )

    def shutdown(self) -> None:
        """Request graceful shutdown. Does not cancel persisted state."""
        self._shutdown_requested = True
        self._runner.shutdown()

    # -- internal -----------------------------------------------------------

    def _load_meeting(self, meeting_id: uuid.UUID) -> Optional[StoredMeeting]:
        """Load meeting via storage. Returns None if not found."""
        load = getattr(self._meeting_storage, "load", None)
        if load is None:
            return None
        return load(meeting_id)

    def _get_track(
        self,
        stored: StoredMeeting,
        role: MeetingTrackRole,
    ) -> Optional[StoredMeetingAudioTrack]:
        if role is MeetingTrackRole.MICROPHONE:
            return stored.microphone
        if role is MeetingTrackRole.REMOTE:
            return stored.remote
        return None

    def _create_generation(
        self,
        meeting_id: uuid.UUID,
        stored: StoredMeeting,
        config: FinalTranscriptionConfig,
    ) -> FinalTranscriptionGeneration:
        """Create generation with track rows. One atomic transaction."""
        now = self._clock()
        now_str = encode_datetime(now)
        assert now_str is not None

        generation_id = uuid.uuid4()

        # Evaluate per-track eligibility
        track_records: list[TrackPersistenceRecord] = []
        any_queued = False

        for role in _ROLE_ORDER:
            audio_track = self._get_track(stored, role)
            if audio_track is None:
                reason = "track data missing"
            else:
                reason = check_track_eligibility(audio_track)

            if reason is None:
                status = FinalTranscriptionTrackStatus.QUEUED
                any_queued = True
            else:
                status = FinalTranscriptionTrackStatus.INELIGIBLE
                logger.info(
                    "Track %s/%s ineligible: %s",
                    meeting_id,
                    role.name,
                    reason,
                )

            track_records.append(
                TrackPersistenceRecord(
                    generation_id=str(generation_id),
                    role=encode_role(role),
                    status=encode_track_status(status),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                )
            )

        if not any_queued:
            gen_status = FinalTranscriptionStatus.FAILED
        else:
            gen_status = FinalTranscriptionStatus.QUEUED

        gen_rec = GenerationPersistenceRecord(
            id=str(generation_id),
            meeting_id=str(meeting_id),
            profile_version=config.profile_version,
            status=encode_generation_status(gen_status),
            **encode_config(config),
            error_message=None,
            time_created=now_str,
            time_started=None,
            time_completed=now_str if is_terminal(gen_status) else None,
        )

        self._repository.create_generation(
            str(generation_id),
            str(meeting_id),
            config,
            encode_generation_status(gen_status),
            now_str,
            now_str if is_terminal(gen_status) else None,
            tuple(track_records),
        )

        # Schedule if any tracks are queued
        if any_queued:
            self._schedule_generation_tracks(str(generation_id), tuple(track_records))

        return assemble_generation(gen_rec, tuple(track_records))

    def _schedule_generation_tracks(
        self,
        generation_id: str,
        track_recs: tuple[TrackPersistenceRecord, ...],
    ) -> None:
        """Enqueue unfinished QUEUED tracks in role order."""
        for role in _ROLE_ORDER:
            role_str = encode_role(role)
            for tr in track_recs:
                if tr.role == role_str:
                    track_status = decode_track_status(tr.status)
                    if track_status is FinalTranscriptionTrackStatus.QUEUED:
                        self._queue.append((generation_id, role))
                    break

        self._process_queue()

    def _process_queue(self) -> None:
        """Process next track in queue if none active."""
        if self._active or self._shutdown_requested:
            return
        if not self._queue:
            return

        self._active = True
        try:
            while self._queue and not self._shutdown_requested:
                generation_id, role = self._queue.popleft()
                self._execute_track(generation_id, role)
        finally:
            self._active = False

    def _execute_track(
        self,
        generation_id: str,
        role: MeetingTrackRole,
    ) -> None:
        """Execute ASR for one track: mark in-progress, run, persist."""
        role_str = encode_role(role)
        now_str = encode_datetime(self._clock())
        assert now_str is not None

        # Mark IN_PROGRESS
        try:
            self._repository.begin_track(generation_id, role_str, now_str)
        except Exception as exc:
            logger.error(
                "Failed to mark track %s/%s IN_PROGRESS: %s",
                generation_id,
                role_str,
                exc,
            )
            return

        if self._shutdown_requested:
            return  # Leave as IN_PROGRESS for recovery

        # Load meeting to get source audio path
        gen_rec = self._repository.load_generation(generation_id)
        if gen_rec is None:
            logger.error("Generation %s disappeared", generation_id)
            return

        stored = self._load_meeting(uuid.UUID(gen_rec.meeting_id))
        if stored is None:
            self._fail_track(
                generation_id,
                role_str,
                "Meeting data not found at execution time",
            )
            return

        audio_track = self._get_track(stored, role)
        if audio_track is None:
            self._fail_track(
                generation_id,
                role_str,
                "Track data missing at execution time",
            )
            return

        # Re-check asset existence (TOCTOU)
        if not audio_track.asset_exists_at_load:
            self._fail_track(
                generation_id,
                role_str,
                "Audio asset missing at execution time",
            )
            return

        config = decode_config(
            gen_rec.profile_version,
            gen_rec.config_model_type,
            gen_rec.config_whisper_model_size,
            gen_rec.config_hugging_face_model_id,
            gen_rec.config_language,
        )

        # Run ASR
        try:
            input_segments = self._runner.transcribe_track(
                str(audio_track.path),
                audio_track.sample_rate,
                config,
            )
        except Exception as exc:
            self._fail_track(
                generation_id,
                role_str,
                _safe_error_message(exc),
            )
            return

        # Map segments to meeting timeline
        try:
            mapped_segments = self._map_segments(
                input_segments,
                audio_track.sample_rate,
                audio_track.timing_anchors,
            )
        except TimelineMappingError as exc:
            self._fail_track(
                generation_id,
                role_str,
                f"Timeline mapping failed: {_safe_error_message(exc)}",
            )
            return

        # Persist segments atomically
        completion_time = encode_datetime(self._clock())
        assert completion_time is not None
        try:
            self._repository.complete_track(
                generation_id,
                role_str,
                mapped_segments,
                completion_time,
            )
        except Exception as exc:
            logger.error(
                "Failed to persist track %s/%s: %s",
                generation_id,
                role_str,
                exc,
            )
            # Track stays IN_PROGRESS — recovery will retry

    def _map_segments(
        self,
        input_segments: tuple[TrackTranscriptionInputSegment, ...],
        sample_rate: int,
        anchors: tuple[StoredMeetingTimingAnchor, ...],
    ) -> tuple[SegmentPersistenceRecord, ...]:
        """Map adapter segments to meeting timeline and build records."""
        result: list[SegmentPersistenceRecord] = []
        for ordinal, seg in enumerate(input_segments):
            if not isinstance(seg.start_ms, int) or isinstance(seg.start_ms, bool):
                raise TimelineMappingError(f"Segment {ordinal} start_ms must be int")
            if not isinstance(seg.end_ms, int) or isinstance(seg.end_ms, bool):
                raise TimelineMappingError(f"Segment {ordinal} end_ms must be int")
            if seg.start_ms < 0:
                raise TimelineMappingError(
                    f"Segment {ordinal} start_ms < 0: {seg.start_ms}"
                )
            if seg.end_ms < seg.start_ms:
                raise TimelineMappingError(
                    f"Segment {ordinal} end_ms < start_ms: "
                    f"{seg.end_ms} < {seg.start_ms}"
                )

            start_ns = map_track_time_to_meeting_ns(seg.start_ms, sample_rate, anchors)
            end_ns = map_track_time_to_meeting_ns(seg.end_ms, sample_rate, anchors)

            if end_ns < start_ns:
                raise TimelineMappingError(
                    f"Segment {ordinal} mapped end_ns < start_ns: "
                    f"{end_ns} < {start_ns}"
                )

            result.append(
                SegmentPersistenceRecord(
                    generation_id="",  # set by repository
                    role="",  # set by repository
                    ordinal=ordinal,
                    local_start_ms=seg.start_ms,
                    local_end_ms=seg.end_ms,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    text=seg.text,
                )
            )

        return tuple(result)

    def _fail_track(
        self,
        generation_id: str,
        role_str: str,
        error_message: str,
    ) -> None:
        """Persist track failure."""
        now_str = encode_datetime(self._clock())
        assert now_str is not None
        try:
            self._repository.fail_track(
                generation_id,
                role_str,
                error_message[:4096],
                now_str,
            )
        except Exception as exc:
            logger.error(
                "Failed to persist track failure %s/%s: %s",
                generation_id,
                role_str,
                exc,
            )


__all__ = [
    "FinalTranscriptionConfig",
    "FinalTranscriptionConfigError",
    "FinalTranscriptionConflictError",
    "FinalTranscriptionDecodeError",
    "FinalTranscriptionEligibilityError",
    "FinalTranscriptionError",
    "FinalTranscriptionGeneration",
    "FinalTranscriptionService",
    "FinalTranscriptionStateError",
    "FinalTranscriptionStatus",
    "FinalTranscriptionTrack",
    "FinalTranscriptionTrackStatus",
    "GenerationPersistenceRecord",
    "MeetingTranscript",
    "MeetingTranscriptSegment",
    "MeetingTranscriptionRepository",
    "SegmentPersistenceRecord",
    "TimelineMappingError",
    "TrackPersistenceRecord",
    "TrackTranscriptionInputSegment",
    "TranscriptionRunner",
    "check_track_eligibility",
    "decode_generation_status",
    "decode_role",
    "decode_track_status",
    "derive_generation_status",
    "encode_config",
    "encode_datetime",
    "encode_generation_status",
    "encode_role",
    "encode_track_status",
    "is_terminal",
    "map_track_time_to_meeting_ns",
]
