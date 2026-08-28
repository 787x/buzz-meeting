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
_PROFILE_VERSION_2 = 2
_KNOWN_PROFILE_VERSIONS = {_PROFILE_VERSION_1, _PROFILE_VERSION_2}
_WHISPER_MODEL_TYPES = frozenset({"WHISPER", "WHISPER_CPP", "FASTER_WHISPER"})
_SUPPORTED_MODEL_TYPES = _WHISPER_MODEL_TYPES | {"HUGGING_FACE"}
_PROFILE_VERSION_2_MODEL_TYPES = frozenset({"WHISPER", "FASTER_WHISPER"})
_PROFILE_VERSION_2_MODEL_SIZES = frozenset(
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
    }
)
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


_OMITTED = object()
"""Sentinel for fields whose default is profile-version-dependent.

Unlike ``None`` (which is a valid explicit value for ``whisper_model_size``
in HUGGING_FACE configs), ``_OMITTED`` means "caller did not pass this
argument".
"""


class _OMITTEDType:
    """Runtime type of the ``_OMITTED`` singleton."""

    pass


@dataclass(frozen=True, slots=True)
class FinalTranscriptionConfig:
    """Immutable, persistable transcription configuration snapshot.

    Created at request time and stored durably. Recovery uses persisted
    config, never re-derives from current Settings.
    """

    profile_version: int = _PROFILE_VERSION_1
    model_type: str = "FASTER_WHISPER"
    whisper_model_size: Optional[str] | _OMITTEDType = _OMITTED
    hugging_face_model_id: str = ""
    language: Optional[str] = None

    def __post_init__(self) -> None:
        # Resolve version-dependent defaults for omitted whisper_model_size.
        if self.whisper_model_size is _OMITTED:
            if self.profile_version == _PROFILE_VERSION_1:
                # v1 default unchanged: TINY for whisper types, None for HF
                # (HF is validated below to require an explicit None).
                if self.model_type in _WHISPER_MODEL_TYPES:
                    object.__setattr__(self, "whisper_model_size", "TINY")
                # else: stays _OMITTED; HF/HF-like paths below will reject.
            # v2 will be rejected below.

        if self.profile_version not in _KNOWN_PROFILE_VERSIONS:
            raise FinalTranscriptionConfigError(
                f"Unknown profile_version: {self.profile_version}"
            )
        if self.model_type not in _SUPPORTED_MODEL_TYPES:
            raise FinalTranscriptionConfigError(
                f"Unsupported model_type: {self.model_type!r}"
            )
        if self.profile_version == _PROFILE_VERSION_2:
            if self.model_type not in _PROFILE_VERSION_2_MODEL_TYPES:
                raise FinalTranscriptionConfigError(
                    "profile_version 2 supports only WHISPER and FASTER_WHISPER"
                )
            if self.whisper_model_size is _OMITTED or self.whisper_model_size is None:
                raise FinalTranscriptionConfigError(
                    "profile_version 2 requires an explicit whisper_model_size"
                )
            if self.whisper_model_size not in _PROFILE_VERSION_2_MODEL_SIZES:
                raise FinalTranscriptionConfigError(
                    "profile_version 2 requires an explicit standard Whisper "
                    f"model size, got {self.whisper_model_size!r}"
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
class MeetingTranscriptWord:
    """One durable word in the meeting timeline."""

    source_role: MeetingTrackRole
    source_segment_ordinal: int
    source_word_ordinal: int
    local_start_ms: int
    local_end_ms: int
    start_ns: int
    end_ns: int
    text: str


@dataclass(frozen=True, slots=True)
class TrackTranscriptionInputSegment:
    """Pure adapter result from FileTranscriber — no Qt types."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class TrackTranscriptionInputWord:
    """One backend-native word associated with a phrase segment."""

    source_segment_ordinal: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class TrackTranscriptionResult:
    """Pure same-inference result returned by a track runner."""

    segments: tuple[TrackTranscriptionInputSegment, ...]
    words: tuple[TrackTranscriptionInputWord, ...]


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

    def load_words(
        self,
        generation_id: str,
    ) -> tuple[WordPersistenceRecord, ...]:
        """Load all word rows without joining away corrupt provenance."""
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
        words: tuple[WordPersistenceRecord, ...] = (),
    ) -> None:
        """Atomically replace results, mark COMPLETED, derive generation."""
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
    """Runs ASR on one audio file, returning a pure same-inference result.

    v1 runners may return a plain tuple of ``TrackTranscriptionInputSegment``
    (legacy PR11 contract).  v2 runners must return ``TrackTranscriptionResult``.
    """

    def transcribe_track(
        self,
        audio_path: str,
        sample_rate: int,
        config: FinalTranscriptionConfig,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TrackTranscriptionResult | tuple[TrackTranscriptionInputSegment, ...]:
        """Transcribe one audio file. Raises on error."""
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
    word_count: int = 0


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


@dataclass(frozen=True, slots=True)
class WordPersistenceRecord:
    """SQLite-compatible word record."""

    generation_id: str
    role: str
    ordinal: int
    segment_ordinal: int
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


class FinalTranscriptionReadService:
    """Fresh, fully validated reads of durable final-transcription data."""

    def __init__(self, repository: MeetingTranscriptionRepository) -> None:
        self._repository = repository

    def load_generation(
        self,
        generation_id: uuid.UUID,
    ) -> Optional[FinalTranscriptionGeneration]:
        result = self._load_validated_aggregate(str(generation_id))
        if result is None:
            return None
        gen_rec, track_recs, _, _ = result
        return assemble_generation(gen_rec, track_recs)

    def load_generation_for_meeting(
        self,
        meeting_id: uuid.UUID,
        profile_version: int,
    ) -> Optional[FinalTranscriptionGeneration]:
        discovered = self._repository.find_generation_by_key(
            str(meeting_id), profile_version
        )
        if discovered is None:
            return None
        result = self._load_validated_aggregate(discovered.id)
        if result is None:
            raise FinalTranscriptionDecodeError(
                "Discovered final-transcription generation disappeared"
            )
        generation_record, track_records, _, _ = result
        generation = assemble_generation(generation_record, track_records)
        if generation.meeting_id != meeting_id:
            raise FinalTranscriptionDecodeError(
                "Repository returned generation for a different meeting"
            )
        if generation.profile_version != profile_version:
            raise FinalTranscriptionDecodeError(
                "Repository returned generation for a different profile"
            )
        return generation

    def load_transcript(
        self,
        generation_id: uuid.UUID,
    ) -> Optional[MeetingTranscript]:
        result = self._load_validated_aggregate(str(generation_id))
        if result is None:
            return None

        gen_rec, track_recs, segments_by_role, _ = result
        status = decode_generation_status(gen_rec.status)
        if status not in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
        ):
            return None

        all_segments: list[MeetingTranscriptSegment] = []
        for track_rec in track_recs:
            if (
                decode_track_status(track_rec.status)
                is not FinalTranscriptionTrackStatus.COMPLETED
            ):
                continue
            role = decode_role(track_rec.role)
            for segment in segments_by_role.get(track_rec.role, ()):
                all_segments.append(
                    MeetingTranscriptSegment(
                        merged_ordinal=0,
                        source_role=role,
                        source_track_ordinal=segment.ordinal,
                        local_start_ms=segment.local_start_ms,
                        local_end_ms=segment.local_end_ms,
                        start_ns=segment.start_ns,
                        end_ns=segment.end_ns,
                        text=segment.text,
                    )
                )

        role_order = {
            MeetingTrackRole.MICROPHONE: 0,
            MeetingTrackRole.REMOTE: 1,
        }
        all_segments.sort(
            key=lambda segment: (
                segment.start_ns,
                role_order.get(segment.source_role, 99),
                segment.source_track_ordinal,
            )
        )
        merged = tuple(
            MeetingTranscriptSegment(
                merged_ordinal=ordinal,
                source_role=segment.source_role,
                source_track_ordinal=segment.source_track_ordinal,
                local_start_ms=segment.local_start_ms,
                local_end_ms=segment.local_end_ms,
                start_ns=segment.start_ns,
                end_ns=segment.end_ns,
                text=segment.text,
            )
            for ordinal, segment in enumerate(all_segments)
        )
        return MeetingTranscript(
            generation_id=uuid.UUID(gen_rec.id),
            meeting_id=uuid.UUID(gen_rec.meeting_id),
            status=status,
            segments=merged,
        )

    def load_words(
        self,
        generation_id: uuid.UUID,
    ) -> tuple[MeetingTranscriptWord, ...]:
        generation_id_str = str(generation_id)
        word_recs_check = self._repository.load_words(generation_id_str)
        gen_rec_check = self._repository.load_generation(generation_id_str)
        if gen_rec_check is None and word_recs_check:
            raise FinalTranscriptionDecodeError(
                f"Word rows reference missing generation {generation_id}"
            )

        result = self._load_validated_aggregate(generation_id_str)
        if result is None:
            return ()
        _, track_recs, segments_by_role, all_words = result
        tracks_by_role = {track.role: track for track in track_recs}
        words_by_role: dict[str, list[WordPersistenceRecord]] = {}
        for word in all_words:
            words_by_role.setdefault(word.role, []).append(word)

        projected: list[MeetingTranscriptWord] = []
        for role_raw, track_rec in tracks_by_role.items():
            if (
                decode_track_status(track_rec.status)
                is not FinalTranscriptionTrackStatus.COMPLETED
            ):
                continue
            role = decode_role(role_raw)
            segment_ordinals = {
                segment.ordinal for segment in segments_by_role.get(role_raw, ())
            }
            for word in words_by_role.get(role_raw, []):
                self._validate_persisted_word(word, segment_ordinals)
                projected.append(
                    MeetingTranscriptWord(
                        source_role=role,
                        source_segment_ordinal=word.segment_ordinal,
                        source_word_ordinal=word.ordinal,
                        local_start_ms=word.local_start_ms,
                        local_end_ms=word.local_end_ms,
                        start_ns=word.start_ns,
                        end_ns=word.end_ns,
                        text=word.text,
                    )
                )

        role_order = {
            MeetingTrackRole.MICROPHONE: 0,
            MeetingTrackRole.REMOTE: 1,
        }
        projected.sort(
            key=lambda word: (
                word.start_ns,
                role_order.get(word.source_role, 99),
                word.source_word_ordinal,
            )
        )
        return tuple(projected)

    @staticmethod
    def _validate_persisted_word(
        word: WordPersistenceRecord,
        segment_ordinals: set[int],
    ) -> None:
        int_fields = {
            "ordinal": word.ordinal,
            "segment_ordinal": word.segment_ordinal,
            "local_start_ms": word.local_start_ms,
            "local_end_ms": word.local_end_ms,
            "start_ns": word.start_ns,
            "end_ns": word.end_ns,
        }
        for name, value in int_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise FinalTranscriptionDecodeError(
                    f"Word {name} must be int, got {type(value)}"
                )
        if word.ordinal < 0 or word.segment_ordinal < 0:
            raise FinalTranscriptionDecodeError("Word ordinals must be nonnegative")
        if word.segment_ordinal not in segment_ordinals:
            raise FinalTranscriptionDecodeError(
                f"Word {word.ordinal} references missing phrase segment "
                f"{word.segment_ordinal}"
            )
        if word.local_start_ms < 0 or word.local_end_ms < word.local_start_ms:
            raise FinalTranscriptionDecodeError(
                f"Invalid local word interval at ordinal {word.ordinal}"
            )
        if word.end_ns < word.start_ns:
            raise FinalTranscriptionDecodeError(
                f"Invalid mapped word interval at ordinal {word.ordinal}"
            )
        if not isinstance(word.text, str):
            raise FinalTranscriptionDecodeError(f"Word {word.ordinal} text must be str")

    @staticmethod
    def _validate_aggregate_result(
        config: FinalTranscriptionConfig,
        tracks_by_role: dict[str, TrackPersistenceRecord],
        segments_by_role: dict[str, tuple[SegmentPersistenceRecord, ...]],
        words: tuple[WordPersistenceRecord, ...],
    ) -> None:
        words_by_role: dict[str, list[WordPersistenceRecord]] = {}
        for word in words:
            if word.role not in tracks_by_role:
                raise FinalTranscriptionDecodeError(
                    f"Word row references missing track role {word.role!r}"
                )
            words_by_role.setdefault(word.role, []).append(word)

        for role_raw, track_rec in tracks_by_role.items():
            track_status = decode_track_status(track_rec.status)
            role_words = words_by_role.get(role_raw, [])
            if track_status is not FinalTranscriptionTrackStatus.COMPLETED:
                if role_words:
                    raise FinalTranscriptionDecodeError(
                        f"Non-COMPLETED track {role_raw} contains word rows"
                    )
                continue

            role_segments = segments_by_role.get(role_raw, ())
            if config.profile_version == _PROFILE_VERSION_1:
                if role_words:
                    raise FinalTranscriptionDecodeError(
                        "profile_version 1 generation must not contain word rows"
                    )
                if track_rec.word_count != 0:
                    raise FinalTranscriptionDecodeError(
                        f"profile_version 1 track {role_raw} has "
                        f"word_count={track_rec.word_count}"
                    )
            if (
                config.profile_version == _PROFILE_VERSION_2
                and len(role_words) != track_rec.word_count
            ):
                raise FinalTranscriptionDecodeError(
                    f"Word count mismatch for role {role_raw}: "
                    f"expected {track_rec.word_count}, actual {len(role_words)}"
                )
            expected_word_ordinals = list(range(track_rec.word_count))
            actual_word_ordinals = [word.ordinal for word in role_words]
            if actual_word_ordinals != expected_word_ordinals:
                raise FinalTranscriptionDecodeError(
                    f"Word ordinal gap for role {role_raw}: "
                    f"{actual_word_ordinals!r}"
                )
            if track_rec.segment_count != len(role_segments):
                raise FinalTranscriptionDecodeError(
                    f"Phrase count mismatch for role {role_raw}: "
                    f"segment_count={track_rec.segment_count}, "
                    f"actual={len(role_segments)}"
                )
            expected_seg_ordinals = list(range(track_rec.segment_count))
            actual_seg_ordinals = [segment.ordinal for segment in role_segments]
            if actual_seg_ordinals != expected_seg_ordinals:
                raise FinalTranscriptionDecodeError(
                    f"Phrase ordinal gap for role {role_raw}: "
                    f"{actual_seg_ordinals!r}"
                )
            if config.profile_version == _PROFILE_VERSION_2:
                covered = {word.segment_ordinal for word in role_words}
                for segment in role_segments:
                    if segment.text.strip() and segment.ordinal not in covered:
                        raise FinalTranscriptionDecodeError(
                            f"v2 nonempty phrase {segment.ordinal} in role "
                            f"{role_raw} has no covering word rows"
                        )

    def _load_validated_aggregate(
        self, generation_id: str
    ) -> Optional[
        tuple[
            GenerationPersistenceRecord,
            tuple[TrackPersistenceRecord, ...],
            dict[str, tuple[SegmentPersistenceRecord, ...]],
            tuple[WordPersistenceRecord, ...],
        ]
    ]:
        gen_rec = self._repository.load_generation(generation_id)
        if gen_rec is None:
            return None
        track_recs = self._repository.load_tracks(generation_id)
        status = decode_generation_status(gen_rec.status)
        if status in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
            FinalTranscriptionStatus.FAILED,
        ):
            config = decode_config(
                gen_rec.profile_version,
                gen_rec.config_model_type,
                gen_rec.config_whisper_model_size,
                gen_rec.config_hugging_face_model_id,
                gen_rec.config_language,
            )
            tracks_by_role = {track.role: track for track in track_recs}
            segments_by_role = {
                track.role: self._repository.load_segments(generation_id, track.role)
                for track in track_recs
            }
            all_words = self._repository.load_words(generation_id)
            self._validate_aggregate_result(
                config, tracks_by_role, segments_by_role, all_words
            )
            return gen_rec, track_recs, segments_by_role, all_words
        return gen_rec, track_recs, {}, ()


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
        self._reader = FinalTranscriptionReadService(repository)
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
        return self._reader.load_generation(generation_id)

    def load_transcript(
        self,
        generation_id: uuid.UUID,
    ) -> Optional[MeetingTranscript]:
        """Load the merged transcript projection for a generation.

        Returns None for FAILED/QUEUED/IN_PROGRESS generations, or for
        generations with zero completed tracks.
        """
        return self._reader.load_transcript(generation_id)

    def load_words(
        self,
        generation_id: uuid.UUID,
    ) -> tuple[MeetingTranscriptWord, ...]:
        """Load validated durable words from completed tracks.

        Word rows are queried independently from generations, tracks, and
        phrase segments so corrupt provenance cannot disappear through a JOIN.
        """
        return self._reader.load_words(generation_id)

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

        # Run ASR — normalize legacy tuple result for v1 runners
        try:
            raw_result = self._runner.transcribe_track(
                str(audio_track.path),
                audio_track.sample_rate,
                config,
            )
            if isinstance(raw_result, tuple):
                transcription_result = TrackTranscriptionResult(
                    segments=raw_result, words=()
                )
            elif isinstance(raw_result, TrackTranscriptionResult):
                transcription_result = raw_result
            else:
                self._fail_track(
                    generation_id,
                    role_str,
                    "Transcription runner returned an unsupported result type",
                )
                return
        except Exception as exc:
            self._fail_track(
                generation_id,
                role_str,
                _safe_error_message(exc),
            )
            return

        # Map segments to meeting timeline
        try:
            mapped_segments, mapped_words = self._map_transcription_result(
                transcription_result,
                config,
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
                generation_id=generation_id,
                role=role_str,
                segments=mapped_segments,
                now=completion_time,
                words=mapped_words,
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

    def _map_transcription_result(
        self,
        transcription_result: TrackTranscriptionResult,
        config: FinalTranscriptionConfig,
        sample_rate: int,
        anchors: tuple[StoredMeetingTimingAnchor, ...],
    ) -> tuple[
        tuple[SegmentPersistenceRecord, ...],
        tuple[WordPersistenceRecord, ...],
    ]:
        """Validate and map a same-inference track result."""
        if not isinstance(transcription_result, TrackTranscriptionResult):
            raise TimelineMappingError(
                "Transcription runner returned an invalid result type"
            )
        if config.profile_version == _PROFILE_VERSION_1:
            if transcription_result.words:
                raise TimelineMappingError(
                    "profile_version 1 result must not contain words"
                )
        elif config.profile_version == _PROFILE_VERSION_2:
            if transcription_result.segments and not transcription_result.words:
                raise TimelineMappingError(
                    "profile_version 2 result contains phrase segments but no words"
                )

        segments = self._map_segments(
            transcription_result.segments,
            sample_rate,
            anchors,
        )
        words = self._map_words(
            transcription_result.words,
            len(transcription_result.segments),
            sample_rate,
            anchors,
        )

        # v2 pre-commit coverage check: every nonempty segment must have
        # at least one persisted word referencing it.
        if config.profile_version == _PROFILE_VERSION_2:
            covered_ordinals = {word.segment_ordinal for word in words}
            for seg in segments:
                if seg.text.strip() and seg.ordinal not in covered_ordinals:
                    raise TimelineMappingError(
                        f"v2 nonempty phrase {seg.ordinal} has no " "covering word rows"
                    )

        return segments, words

    def _map_words(
        self,
        input_words: tuple[TrackTranscriptionInputWord, ...],
        segment_count: int,
        sample_rate: int,
        anchors: tuple[StoredMeetingTimingAnchor, ...],
    ) -> tuple[WordPersistenceRecord, ...]:
        """Map words with the unchanged PR11 timeline mapper."""
        result: list[WordPersistenceRecord] = []
        for ordinal, word in enumerate(input_words):
            if not isinstance(word.source_segment_ordinal, int) or isinstance(
                word.source_segment_ordinal, bool
            ):
                raise TimelineMappingError(
                    f"Word {ordinal} source_segment_ordinal must be int"
                )
            if not 0 <= word.source_segment_ordinal < segment_count:
                raise TimelineMappingError(
                    f"Word {ordinal} references missing phrase segment "
                    f"{word.source_segment_ordinal}"
                )
            if not isinstance(word.start_ms, int) or isinstance(word.start_ms, bool):
                raise TimelineMappingError(f"Word {ordinal} start_ms must be int")
            if not isinstance(word.end_ms, int) or isinstance(word.end_ms, bool):
                raise TimelineMappingError(f"Word {ordinal} end_ms must be int")
            if word.start_ms < 0:
                raise TimelineMappingError(
                    f"Word {ordinal} start_ms < 0: {word.start_ms}"
                )
            if word.end_ms < word.start_ms:
                raise TimelineMappingError(
                    f"Word {ordinal} end_ms < start_ms: "
                    f"{word.end_ms} < {word.start_ms}"
                )
            if not isinstance(word.text, str):
                raise TimelineMappingError(f"Word {ordinal} text must be str")

            start_ns = map_track_time_to_meeting_ns(word.start_ms, sample_rate, anchors)
            end_ns = map_track_time_to_meeting_ns(word.end_ms, sample_rate, anchors)
            if end_ns < start_ns:
                raise TimelineMappingError(
                    f"Word {ordinal} mapped end_ns < start_ns: "
                    f"{end_ns} < {start_ns}"
                )

            result.append(
                WordPersistenceRecord(
                    generation_id="",
                    role="",
                    ordinal=ordinal,
                    segment_ordinal=word.source_segment_ordinal,
                    local_start_ms=word.start_ms,
                    local_end_ms=word.end_ms,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    text=word.text,
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
    "FinalTranscriptionReadService",
    "FinalTranscriptionService",
    "FinalTranscriptionStateError",
    "FinalTranscriptionStatus",
    "FinalTranscriptionTrack",
    "FinalTranscriptionTrackStatus",
    "GenerationPersistenceRecord",
    "MeetingTranscript",
    "MeetingTranscriptSegment",
    "MeetingTranscriptWord",
    "MeetingTranscriptionRepository",
    "SegmentPersistenceRecord",
    "TimelineMappingError",
    "TrackPersistenceRecord",
    "TrackTranscriptionInputSegment",
    "TrackTranscriptionInputWord",
    "TrackTranscriptionResult",
    "TranscriptionRunner",
    "WordPersistenceRecord",
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
