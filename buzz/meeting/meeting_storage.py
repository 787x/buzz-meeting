"""Pure meeting storage facade, codecs, and filesystem policy.

This module deliberately has no Qt dependency.  SQL execution is delegated to
an injected :class:`MeetingRepository`; the facade owns all meeting-domain
validation, serialization, conflict policy, and asset-path safety checks.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional, Protocol

from platformdirs import user_data_dir

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksResult,
    MeetingAudioTracksState,
    MeetingTrackError,
    MeetingTrackErrorStage,
    MeetingTrackRecordingResult,
    MeetingTrackRole,
    MeetingTrackTiming,
    MeetingTrackTimingAnchor,
)
from buzz.meeting.meeting_recorder import MeetingRecorderState, MeetingRecordingResult
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionSnapshot,
    MeetingSessionState,
)


class MeetingStorageError(Exception):
    """Base error for meeting storage failures."""


class MeetingStorageValidationError(MeetingStorageError):
    """Raised for an invalid save-time snapshot or persistence input."""


class MeetingStorageCollisionError(MeetingStorageValidationError):
    """Raised when a meeting directory cannot be allocated without clobbering."""


class MeetingStorageFilesystemError(MeetingStorageError):
    """Raised for non-collision filesystem allocation or inspection failures."""


class MeetingStorageConflictError(MeetingStorageError):
    """Raised when an incoming snapshot would regress durable meeting data."""


class MeetingStorageDecodeError(MeetingStorageError):
    """Raised when persisted meeting metadata or asset paths are corrupt."""


class MeetingStorageDatabaseError(MeetingStorageError):
    """Raised for repository/database failures.

    ``commit_outcome_unknown`` is true only when the database reported a
    commit failure, at which point callers must reload to determine whether the
    transaction became durable.
    """

    def __init__(
        self,
        message: str,
        *,
        commit_outcome_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.commit_outcome_unknown = commit_outcome_unknown


@dataclass(frozen=True, slots=True)
class MeetingStoragePaths:
    session_id: uuid.UUID
    directory: Path
    microphone: Path
    remote: Path


@dataclass(frozen=True, slots=True)
class StoredMeetingTimingAnchor:
    sample_end: int
    callback_arrival_offset_ns: int


@dataclass(frozen=True, slots=True)
class StoredMeetingError:
    role: MeetingTrackRole
    stage: MeetingTrackErrorStage
    exception_module: str
    exception_name: str
    message: str


@dataclass(frozen=True, slots=True)
class StoredMeetingAudioTrack:
    role: MeetingTrackRole
    relative_path: PurePosixPath
    path: Path
    sample_rate: int
    sample_count: int
    recording_state: MeetingRecorderState
    published: bool
    complete: bool
    timing_basis: str
    timing_anchors: tuple[StoredMeetingTimingAnchor, ...]
    errors: tuple[StoredMeetingError, ...]
    asset_exists_at_load: bool

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


@dataclass(frozen=True, slots=True)
class StoredMeeting:
    session_id: uuid.UUID
    remote_source_kind: MeetingRemoteSourceKind
    state: MeetingSessionState
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration_ns: Optional[int]
    audio_state: MeetingAudioTracksState
    audio_outcome: Optional[MeetingAudioTracksOutcome]
    microphone: Optional[StoredMeetingAudioTrack]
    remote: Optional[StoredMeetingAudioTrack]

    def __post_init__(self) -> None:
        absent = (
            self.microphone is None,
            self.remote is None,
            self.audio_outcome is None,
        )
        if len(set(absent)) != 1:
            raise ValueError(
                "microphone, remote, and audio_outcome must be present together"
            )


# Persistence records contain only SQLite-compatible primitives.  They are
# module-visible so the QSql adapter can consume them without importing any
# runtime meeting objects.
@dataclass(frozen=True, slots=True)
class MeetingTrackPersistenceRecord:
    role: str
    relative_path: str
    sample_rate: int
    sample_count: int
    recording_state: str
    published: int
    complete: int
    timing_basis: str


@dataclass(frozen=True, slots=True)
class MeetingTimingPersistenceRecord:
    role: str
    ordinal: int
    sample_end: int
    callback_arrival_offset_ns: int


@dataclass(frozen=True, slots=True)
class MeetingErrorPersistenceRecord:
    role: str
    ordinal: int
    stage: str
    exception_module: str
    exception_name: str
    message: str


@dataclass(frozen=True, slots=True)
class MeetingPersistenceBundle:
    session_id: str
    remote_source_kind: str
    session_state: str
    created_at: str
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_ns: Optional[int]
    audio_state: str
    audio_outcome: Optional[str]
    tracks: tuple[MeetingTrackPersistenceRecord, ...]
    timings: tuple[MeetingTimingPersistenceRecord, ...]
    errors: tuple[MeetingErrorPersistenceRecord, ...]


@dataclass(frozen=True, slots=True)
class MeetingPersistenceReadBundle:
    """Internal raw read shape for child rows whose parent is absent."""

    meeting: None
    tracks: tuple[MeetingTrackPersistenceRecord, ...]
    timings: tuple[MeetingTimingPersistenceRecord, ...]
    errors: tuple[MeetingErrorPersistenceRecord, ...]


MeetingPersistenceLoadResult = MeetingPersistenceBundle | MeetingPersistenceReadBundle
ExistingBundleValidator = Callable[[Optional[MeetingPersistenceLoadResult]], None]


class MeetingRepository(Protocol):
    """Primitive persistence boundary implemented by the QSql adapter."""

    def atomic_replace(
        self,
        bundle: MeetingPersistenceBundle,
        *,
        validate_existing: ExistingBundleValidator,
    ) -> None:
        ...

    def load_bundle(
        self,
        session_id: str,
    ) -> Optional[MeetingPersistenceLoadResult]:
        ...


_ROLE_FILENAMES = {
    MeetingTrackRole.MICROPHONE: "microphone.wav",
    MeetingTrackRole.REMOTE: "remote.wav",
}
_KNOWN_TIMING_BASIS = "host_callback_arrival"
_MAX_TIMING_ANCHORS = 4096
_MAX_ERRORS_PER_ROLE_STAGE = 2
_MAX_ERRORS_PER_TRACK = 8
_MAX_ERROR_MESSAGE_LENGTH = 4096
_NONTERMINAL_STATE_RANK = {
    MeetingSessionState.CREATED: 0,
    MeetingSessionState.STARTING: 1,
    MeetingSessionState.ACTIVE: 2,
    MeetingSessionState.STOPPING: 3,
}
_TERMINAL_STATES = {
    MeetingSessionState.COMPLETED,
    MeetingSessionState.FAILED,
}


class MeetingStorage:
    """Persist meeting snapshots without rehydrating runtime sessions.

    Repository operations inherit the supplied database connection's thread
    affinity and are not thread-safe.  ``prepare`` has its own lock solely for
    in-process filesystem reservation; save correctness never depends on that
    reservation or on this instance having called ``prepare``.
    """

    def __init__(
        self,
        repository: MeetingRepository,
        *,
        root: Optional[Path] = None,
    ) -> None:
        self._repository = repository
        configured_root = (
            Path(root) if root is not None else Path(user_data_dir("Buzz")) / "meetings"
        )
        try:
            self._root = configured_root.resolve(strict=False)
        except OSError as exc:
            raise MeetingStorageFilesystemError(
                f"Could not resolve meeting storage root: {configured_root}"
            ) from exc
        self._prepare_lock = threading.Lock()
        self._reservations: dict[uuid.UUID, MeetingStoragePaths] = {}

    def prepare(self, session_id: uuid.UUID) -> MeetingStoragePaths:
        session_id = self._require_session_id(session_id)
        with self._prepare_lock:
            reserved = self._reservations.get(session_id)
            if reserved is not None:
                self._validate_empty_reserved_directory(reserved.directory)
                return reserved

            self._ensure_root_directory()
            paths = self._paths_for(session_id)
            try:
                paths.directory.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise MeetingStorageCollisionError(
                    f"Meeting storage target already exists: {paths.directory}"
                ) from exc
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise MeetingStorageCollisionError(
                        f"Meeting storage target already exists: {paths.directory}"
                    ) from exc
                raise MeetingStorageFilesystemError(
                    f"Could not create meeting storage directory: {paths.directory}"
                ) from exc

            self._reservations[session_id] = paths
            return paths

    def save(self, snapshot: MeetingSessionSnapshot) -> StoredMeeting:
        bundle = self._encode_snapshot(snapshot)

        def validate_existing(
            existing: Optional[MeetingPersistenceLoadResult],
        ) -> None:
            if existing is None:
                return
            existing_bundle = self._require_parent_bundle(existing)
            existing_meeting = self._decode_bundle(
                existing_bundle,
                expected_session_id=snapshot.session_id,
                inspect_assets=False,
            )
            incoming_meeting = self._decode_bundle(
                bundle,
                expected_session_id=snapshot.session_id,
                inspect_assets=False,
            )
            self._validate_update(
                existing_bundle=existing_bundle,
                incoming_bundle=bundle,
                existing=existing_meeting,
                incoming=incoming_meeting,
            )

        self._repository.atomic_replace(
            bundle,
            validate_existing=validate_existing,
        )
        return self._decode_bundle(
            bundle,
            expected_session_id=snapshot.session_id,
            inspect_assets=True,
        )

    def load(self, session_id: uuid.UUID) -> Optional[StoredMeeting]:
        session_id = self._require_session_id(session_id)
        bundle = self._repository.load_bundle(str(session_id))
        if bundle is None:
            return None
        return self._decode_bundle(
            bundle,
            expected_session_id=session_id,
            inspect_assets=True,
        )

    def _encode_snapshot(
        self,
        snapshot: MeetingSessionSnapshot,
    ) -> MeetingPersistenceBundle:
        if not isinstance(snapshot, MeetingSessionSnapshot):
            raise MeetingStorageValidationError(
                "snapshot must be a MeetingSessionSnapshot"
            )
        session_id = self._require_session_id(snapshot.session_id)
        remote_source_kind = self._encode_enum(
            snapshot.remote_source_kind,
            MeetingRemoteSourceKind,
            "remote_source_kind",
        )
        session_state = self._encode_enum(
            snapshot.state,
            MeetingSessionState,
            "state",
        )
        audio_state = self._encode_enum(
            snapshot.audio_state,
            MeetingAudioTracksState,
            "audio_state",
        )
        created_at = self._encode_datetime(snapshot.created_at, "created_at")
        started_at = self._encode_optional_datetime(
            snapshot.started_at,
            "started_at",
        )
        ended_at = self._encode_optional_datetime(snapshot.ended_at, "ended_at")
        duration_ns = self._validate_optional_nonnegative_int(
            snapshot.duration_ns,
            "duration_ns",
            MeetingStorageValidationError,
        )

        if snapshot.audio is None:
            return MeetingPersistenceBundle(
                session_id=str(session_id),
                remote_source_kind=remote_source_kind,
                session_state=session_state,
                created_at=created_at,
                started_at=started_at,
                ended_at=ended_at,
                duration_ns=duration_ns,
                audio_state=audio_state,
                audio_outcome=None,
                tracks=(),
                timings=(),
                errors=(),
            )
        if not isinstance(snapshot.audio, MeetingAudioTracksResult):
            raise MeetingStorageValidationError(
                "snapshot.audio must be a MeetingAudioTracksResult or None"
            )

        audio_outcome = self._encode_enum(
            snapshot.audio.outcome,
            MeetingAudioTracksOutcome,
            "audio.outcome",
        )
        tracks: list[MeetingTrackPersistenceRecord] = []
        timings: list[MeetingTimingPersistenceRecord] = []
        errors: list[MeetingErrorPersistenceRecord] = []
        for expected_role, track in (
            (MeetingTrackRole.MICROPHONE, snapshot.audio.microphone),
            (MeetingTrackRole.REMOTE, snapshot.audio.remote),
        ):
            track_record, track_timings, track_errors = self._encode_track(
                session_id,
                expected_role,
                track,
            )
            tracks.append(track_record)
            timings.extend(track_timings)
            errors.extend(track_errors)

        return MeetingPersistenceBundle(
            session_id=str(session_id),
            remote_source_kind=remote_source_kind,
            session_state=session_state,
            created_at=created_at,
            started_at=started_at,
            ended_at=ended_at,
            duration_ns=duration_ns,
            audio_state=audio_state,
            audio_outcome=audio_outcome,
            tracks=tuple(tracks),
            timings=tuple(timings),
            errors=tuple(errors),
        )

    def _encode_track(
        self,
        session_id: uuid.UUID,
        expected_role: MeetingTrackRole,
        track: MeetingTrackRecordingResult,
    ) -> tuple[
        MeetingTrackPersistenceRecord,
        tuple[MeetingTimingPersistenceRecord, ...],
        tuple[MeetingErrorPersistenceRecord, ...],
    ]:
        if not isinstance(track, MeetingTrackRecordingResult):
            raise MeetingStorageValidationError(
                f"{expected_role.name.lower()} must be a MeetingTrackRecordingResult"
            )
        if track.role is not expected_role:
            raise MeetingStorageValidationError(
                f"Expected {expected_role.name} track, got {track.role!r}"
            )
        recording = track.recording
        if not isinstance(recording, MeetingRecordingResult):
            raise MeetingStorageValidationError(
                f"{expected_role.name}.recording must be a MeetingRecordingResult"
            )
        sample_rate = self._validate_positive_int(
            recording.sample_rate,
            f"{expected_role.name}.sample_rate",
            MeetingStorageValidationError,
        )
        sample_count = self._validate_nonnegative_int(
            recording.sample_count,
            f"{expected_role.name}.sample_count",
            MeetingStorageValidationError,
        )
        if not isinstance(recording.published, bool):
            raise MeetingStorageValidationError("published must be a bool")
        if not isinstance(track.complete, bool):
            raise MeetingStorageValidationError("complete must be a bool")
        recording_state = self._encode_enum(
            recording.state,
            MeetingRecorderState,
            f"{expected_role.name}.recording_state",
        )
        expected_path = self._expected_asset_path(session_id, expected_role)
        self._validate_declared_output_path(recording.output_path, expected_path)
        self._inspect_asset(expected_path, MeetingStorageValidationError)
        relative_path = self._relative_path_string(session_id, expected_role)

        timing = track.timing
        if not isinstance(timing, MeetingTrackTiming):
            raise MeetingStorageValidationError(
                f"{expected_role.name}.timing must be a MeetingTrackTiming"
            )
        if not isinstance(timing.anchors, tuple):
            raise MeetingStorageValidationError("timing anchors must be a tuple")
        if timing.timing_basis != _KNOWN_TIMING_BASIS:
            raise MeetingStorageValidationError(
                f"Unknown timing basis: {timing.timing_basis!r}"
            )
        if len(timing.anchors) > _MAX_TIMING_ANCHORS:
            raise MeetingStorageValidationError(
                f"Too many timing anchors for {expected_role.name}"
            )
        timing_records: list[MeetingTimingPersistenceRecord] = []
        previous_sample_end = 0
        for ordinal, anchor in enumerate(timing.anchors):
            if not isinstance(anchor, MeetingTrackTimingAnchor):
                raise MeetingStorageValidationError(
                    "timing anchors must be MeetingTrackTimingAnchor values"
                )
            sample_end = self._validate_positive_int(
                anchor.sample_end,
                "timing sample_end",
                MeetingStorageValidationError,
            )
            if sample_end > sample_count:
                raise MeetingStorageValidationError(
                    "timing sample_end exceeds durable sample_count"
                )
            if sample_end <= previous_sample_end:
                raise MeetingStorageValidationError(
                    "timing sample_end values must be strictly increasing"
                )
            offset = self._validate_int(
                anchor.callback_arrival_offset_ns,
                "callback_arrival_offset_ns",
                MeetingStorageValidationError,
            )
            timing_records.append(
                MeetingTimingPersistenceRecord(
                    role=expected_role.name,
                    ordinal=ordinal,
                    sample_end=sample_end,
                    callback_arrival_offset_ns=offset,
                )
            )
            previous_sample_end = sample_end

        if not isinstance(track.errors, tuple):
            raise MeetingStorageValidationError("track errors must be a tuple")
        if len(track.errors) > _MAX_ERRORS_PER_TRACK:
            raise MeetingStorageValidationError(
                f"Too many stored errors for {expected_role.name}"
            )
        stage_counts: Counter[MeetingTrackErrorStage] = Counter()
        error_records: list[MeetingErrorPersistenceRecord] = []
        for ordinal, error in enumerate(track.errors):
            if not isinstance(error, MeetingTrackError):
                raise MeetingStorageValidationError(
                    "track errors must be MeetingTrackError values"
                )
            if error.role is not expected_role:
                raise MeetingStorageValidationError(
                    "Track error role does not match its parent track"
                )
            stage_name = self._encode_enum(
                error.stage,
                MeetingTrackErrorStage,
                "error.stage",
            )
            stage_counts[error.stage] += 1
            if stage_counts[error.stage] > _MAX_ERRORS_PER_ROLE_STAGE:
                raise MeetingStorageValidationError(
                    f"Too many {stage_name} errors for {expected_role.name}"
                )
            if not isinstance(error.exception, Exception):
                raise MeetingStorageValidationError(
                    "Track error.exception must be an Exception"
                )
            try:
                message = str(error.exception)[:_MAX_ERROR_MESSAGE_LENGTH]
            except Exception as exc:
                raise MeetingStorageValidationError(
                    "Could not stringify meeting track error"
                ) from exc
            error_type = type(error.exception)
            error_records.append(
                MeetingErrorPersistenceRecord(
                    role=expected_role.name,
                    ordinal=ordinal,
                    stage=stage_name,
                    exception_module=error_type.__module__,
                    exception_name=error_type.__name__,
                    message=message,
                )
            )

        return (
            MeetingTrackPersistenceRecord(
                role=expected_role.name,
                relative_path=relative_path,
                sample_rate=sample_rate,
                sample_count=sample_count,
                recording_state=recording_state,
                published=int(recording.published),
                complete=int(track.complete),
                timing_basis=timing.timing_basis,
            ),
            tuple(timing_records),
            tuple(error_records),
        )

    def _decode_bundle(
        self,
        bundle: MeetingPersistenceLoadResult,
        *,
        expected_session_id: uuid.UUID,
        inspect_assets: bool,
    ) -> StoredMeeting:
        bundle = self._require_parent_bundle(bundle)
        return self._decode_parent_bundle(
            bundle,
            expected_session_id=expected_session_id,
            inspect_assets=inspect_assets,
        )

    def _require_parent_bundle(
        self,
        bundle: MeetingPersistenceLoadResult,
    ) -> MeetingPersistenceBundle:
        if isinstance(bundle, MeetingPersistenceReadBundle):
            if bundle.meeting is None and (
                bundle.tracks or bundle.timings or bundle.errors
            ):
                raise MeetingStorageDecodeError(
                    "Persisted meeting children exist without a meeting parent"
                )
            raise MeetingStorageDecodeError(
                "Repository returned an empty meeting read bundle"
            )
        if not isinstance(bundle, MeetingPersistenceBundle):
            raise MeetingStorageDecodeError(
                "Repository returned an invalid meeting persistence bundle"
            )
        return bundle

    def _decode_parent_bundle(
        self,
        bundle: MeetingPersistenceBundle,
        *,
        expected_session_id: uuid.UUID,
        inspect_assets: bool,
    ) -> StoredMeeting:
        session_id = self._decode_canonical_uuid(bundle.session_id)
        if session_id != expected_session_id:
            raise MeetingStorageDecodeError(
                "Persisted meeting ID does not match request"
            )
        remote_source_kind = self._decode_enum(
            bundle.remote_source_kind,
            MeetingRemoteSourceKind,
            "remote_source_kind",
        )
        state = self._decode_enum(
            bundle.session_state,
            MeetingSessionState,
            "session_state",
        )
        created_at = self._decode_datetime(bundle.created_at, "created_at")
        started_at = self._decode_optional_datetime(bundle.started_at, "started_at")
        ended_at = self._decode_optional_datetime(bundle.ended_at, "ended_at")
        duration_ns = self._validate_optional_nonnegative_int(
            bundle.duration_ns,
            "duration_ns",
            MeetingStorageDecodeError,
        )
        audio_state = self._decode_enum(
            bundle.audio_state,
            MeetingAudioTracksState,
            "audio_state",
        )
        if not isinstance(bundle.tracks, tuple):
            raise MeetingStorageDecodeError("tracks must be a tuple")
        if not isinstance(bundle.timings, tuple):
            raise MeetingStorageDecodeError("timings must be a tuple")
        if not isinstance(bundle.errors, tuple):
            raise MeetingStorageDecodeError("errors must be a tuple")

        if bundle.audio_outcome is None:
            if bundle.tracks or bundle.timings or bundle.errors:
                raise MeetingStorageDecodeError(
                    "Meeting without audio outcome has persisted audio children"
                )
            return StoredMeeting(
                session_id=session_id,
                remote_source_kind=remote_source_kind,
                state=state,
                created_at=created_at,
                started_at=started_at,
                ended_at=ended_at,
                duration_ns=duration_ns,
                audio_state=audio_state,
                audio_outcome=None,
                microphone=None,
                remote=None,
            )

        audio_outcome = self._decode_enum(
            bundle.audio_outcome,
            MeetingAudioTracksOutcome,
            "audio_outcome",
        )
        if len(bundle.tracks) != 2:
            raise MeetingStorageDecodeError(
                "Persisted audio aggregate must contain exactly two tracks"
            )
        track_records: dict[MeetingTrackRole, MeetingTrackPersistenceRecord] = {}
        for record in bundle.tracks:
            if not isinstance(record, MeetingTrackPersistenceRecord):
                raise MeetingStorageDecodeError("Invalid track persistence record")
            role = self._decode_enum(record.role, MeetingTrackRole, "track.role")
            if role in track_records:
                raise MeetingStorageDecodeError("Duplicate persisted track role")
            track_records[role] = record
        if set(track_records) != set(MeetingTrackRole):
            raise MeetingStorageDecodeError(
                "Persisted audio aggregate must contain microphone and remote"
            )

        timings_by_role: dict[
            MeetingTrackRole,
            list[MeetingTimingPersistenceRecord],
        ] = {role: [] for role in MeetingTrackRole}
        for timing in bundle.timings:
            if not isinstance(timing, MeetingTimingPersistenceRecord):
                raise MeetingStorageDecodeError("Invalid timing persistence record")
            role = self._decode_enum(timing.role, MeetingTrackRole, "timing.role")
            if role not in track_records:
                raise MeetingStorageDecodeError("Orphan timing persistence record")
            timings_by_role[role].append(timing)

        errors_by_role: dict[
            MeetingTrackRole,
            list[MeetingErrorPersistenceRecord],
        ] = {role: [] for role in MeetingTrackRole}
        for error in bundle.errors:
            if not isinstance(error, MeetingErrorPersistenceRecord):
                raise MeetingStorageDecodeError("Invalid error persistence record")
            role = self._decode_enum(error.role, MeetingTrackRole, "error.role")
            if role not in track_records:
                raise MeetingStorageDecodeError("Orphan error persistence record")
            errors_by_role[role].append(error)

        decoded_tracks = {
            role: self._decode_track(
                session_id,
                role,
                track_records[role],
                tuple(timings_by_role[role]),
                tuple(errors_by_role[role]),
                inspect_assets=inspect_assets,
            )
            for role in MeetingTrackRole
        }
        return StoredMeeting(
            session_id=session_id,
            remote_source_kind=remote_source_kind,
            state=state,
            created_at=created_at,
            started_at=started_at,
            ended_at=ended_at,
            duration_ns=duration_ns,
            audio_state=audio_state,
            audio_outcome=audio_outcome,
            microphone=decoded_tracks[MeetingTrackRole.MICROPHONE],
            remote=decoded_tracks[MeetingTrackRole.REMOTE],
        )

    def _decode_track(
        self,
        session_id: uuid.UUID,
        role: MeetingTrackRole,
        record: MeetingTrackPersistenceRecord,
        timings: tuple[MeetingTimingPersistenceRecord, ...],
        errors: tuple[MeetingErrorPersistenceRecord, ...],
        *,
        inspect_assets: bool,
    ) -> StoredMeetingAudioTrack:
        relative_path = self._decode_relative_path(
            record.relative_path,
            session_id,
            role,
        )
        sample_rate = self._validate_positive_int(
            record.sample_rate,
            "sample_rate",
            MeetingStorageDecodeError,
        )
        sample_count = self._validate_nonnegative_int(
            record.sample_count,
            "sample_count",
            MeetingStorageDecodeError,
        )
        recording_state = self._decode_enum(
            record.recording_state,
            MeetingRecorderState,
            "recording_state",
        )
        published = self._decode_bool(record.published, "published")
        complete = self._decode_bool(record.complete, "complete")
        if record.timing_basis != _KNOWN_TIMING_BASIS:
            raise MeetingStorageDecodeError(
                f"Unknown timing basis: {record.timing_basis!r}"
            )
        if len(timings) > _MAX_TIMING_ANCHORS:
            raise MeetingStorageDecodeError("Too many persisted timing anchors")
        decoded_timings: list[StoredMeetingTimingAnchor] = []
        previous_sample_end = 0
        for expected_ordinal, timing in enumerate(timings):
            if timing.ordinal != expected_ordinal:
                raise MeetingStorageDecodeError(
                    "Timing ordinals must be contiguous from zero"
                )
            sample_end = self._validate_positive_int(
                timing.sample_end,
                "timing sample_end",
                MeetingStorageDecodeError,
            )
            if sample_end > sample_count:
                raise MeetingStorageDecodeError(
                    "Timing sample_end exceeds durable sample_count"
                )
            if sample_end <= previous_sample_end:
                raise MeetingStorageDecodeError(
                    "Timing sample_end values must be strictly increasing"
                )
            offset = self._validate_int(
                timing.callback_arrival_offset_ns,
                "callback_arrival_offset_ns",
                MeetingStorageDecodeError,
            )
            decoded_timings.append(
                StoredMeetingTimingAnchor(
                    sample_end=sample_end,
                    callback_arrival_offset_ns=offset,
                )
            )
            previous_sample_end = sample_end

        if len(errors) > _MAX_ERRORS_PER_TRACK:
            raise MeetingStorageDecodeError("Too many persisted track errors")
        stage_counts: Counter[MeetingTrackErrorStage] = Counter()
        decoded_errors: list[StoredMeetingError] = []
        for expected_ordinal, error in enumerate(errors):
            if error.ordinal != expected_ordinal:
                raise MeetingStorageDecodeError(
                    "Error ordinals must be contiguous from zero"
                )
            stage = self._decode_enum(
                error.stage, MeetingTrackErrorStage, "error.stage"
            )
            stage_counts[stage] += 1
            if stage_counts[stage] > _MAX_ERRORS_PER_ROLE_STAGE:
                raise MeetingStorageDecodeError(
                    f"Too many persisted {stage.name} errors"
                )
            for field_name, value in (
                ("exception_module", error.exception_module),
                ("exception_name", error.exception_name),
                ("message", error.message),
            ):
                if not isinstance(value, str):
                    raise MeetingStorageDecodeError(f"{field_name} must be text")
            if len(error.message) > _MAX_ERROR_MESSAGE_LENGTH:
                raise MeetingStorageDecodeError("Persisted error message is too long")
            decoded_errors.append(
                StoredMeetingError(
                    role=role,
                    stage=stage,
                    exception_module=error.exception_module,
                    exception_name=error.exception_name,
                    message=error.message,
                )
            )

        path = self._expected_asset_path(session_id, role)
        asset_exists = (
            self._inspect_asset(path, MeetingStorageDecodeError)
            if inspect_assets
            else False
        )
        return StoredMeetingAudioTrack(
            role=role,
            relative_path=relative_path,
            path=path,
            sample_rate=sample_rate,
            sample_count=sample_count,
            recording_state=recording_state,
            published=published,
            complete=complete,
            timing_basis=record.timing_basis,
            timing_anchors=tuple(decoded_timings),
            errors=tuple(decoded_errors),
            asset_exists_at_load=asset_exists,
        )

    def _validate_update(
        self,
        *,
        existing_bundle: MeetingPersistenceBundle,
        incoming_bundle: MeetingPersistenceBundle,
        existing: StoredMeeting,
        incoming: StoredMeeting,
    ) -> None:
        if existing.created_at != incoming.created_at:
            raise MeetingStorageConflictError("created_at cannot change")
        if existing.remote_source_kind is not incoming.remote_source_kind:
            raise MeetingStorageConflictError("remote_source_kind cannot change")
        self._validate_optional_progression(
            "started_at",
            existing.started_at,
            incoming.started_at,
        )
        self._validate_optional_progression(
            "ended_at",
            existing.ended_at,
            incoming.ended_at,
        )
        self._validate_optional_progression(
            "duration_ns",
            existing.duration_ns,
            incoming.duration_ns,
        )
        if existing.audio_outcome is not None and incoming.audio_outcome is None:
            raise MeetingStorageConflictError(
                "Persisted audio metadata cannot disappear"
            )

        if existing.state is MeetingSessionState.COMPLETED:
            if incoming.state is not MeetingSessionState.COMPLETED:
                raise MeetingStorageConflictError(
                    "A completed meeting cannot change state"
                )
            if existing_bundle != incoming_bundle:
                raise MeetingStorageConflictError(
                    "A completed meeting aggregate is immutable"
                )
            return
        if existing.state is MeetingSessionState.FAILED:
            if incoming.state is not MeetingSessionState.FAILED:
                raise MeetingStorageConflictError(
                    "A failed meeting cannot change terminal state"
                )
            return
        if incoming.state in _TERMINAL_STATES:
            return
        existing_rank = _NONTERMINAL_STATE_RANK[existing.state]
        incoming_rank = _NONTERMINAL_STATE_RANK[incoming.state]
        if incoming_rank < existing_rank:
            raise MeetingStorageConflictError("Meeting session state cannot regress")

    @staticmethod
    def _validate_optional_progression(
        field_name: str,
        existing: object,
        incoming: object,
    ) -> None:
        if existing is not None and incoming != existing:
            raise MeetingStorageConflictError(f"{field_name} cannot regress or change")

    def _paths_for(self, session_id: uuid.UUID) -> MeetingStoragePaths:
        directory = self._root / str(session_id)
        return MeetingStoragePaths(
            session_id=session_id,
            directory=directory,
            microphone=directory / _ROLE_FILENAMES[MeetingTrackRole.MICROPHONE],
            remote=directory / _ROLE_FILENAMES[MeetingTrackRole.REMOTE],
        )

    def _expected_asset_path(
        self,
        session_id: uuid.UUID,
        role: MeetingTrackRole,
    ) -> Path:
        return self._root / str(session_id) / _ROLE_FILENAMES[role]

    @staticmethod
    def _relative_path_string(
        session_id: uuid.UUID,
        role: MeetingTrackRole,
    ) -> str:
        return f"{session_id}/{_ROLE_FILENAMES[role]}"

    def _decode_relative_path(
        self,
        raw_path: str,
        session_id: uuid.UUID,
        role: MeetingTrackRole,
    ) -> PurePosixPath:
        if not isinstance(raw_path, str):
            raise MeetingStorageDecodeError("relative_path must be text")
        expected = self._relative_path_string(session_id, role)
        if raw_path != expected:
            raise MeetingStorageDecodeError(
                f"Invalid canonical meeting audio path: {raw_path!r}"
            )
        parts = raw_path.split("/")
        if len(parts) != 2:
            raise MeetingStorageDecodeError(
                "Meeting audio relative path must have exactly two segments"
            )
        return PurePosixPath(raw_path)

    def _validate_declared_output_path(
        self,
        declared_path: os.PathLike[str] | str,
        expected_path: Path,
    ) -> None:
        try:
            raw_path = os.fspath(declared_path)
            if not isinstance(raw_path, str):
                raise TypeError("meeting recording output path must be text")
            expected_raw_path = os.fspath(expected_path)
            if raw_path != expected_raw_path:
                raise MeetingStorageValidationError(
                    f"Meeting recording output path must be exactly {expected_path}"
                )
            declared = Path(raw_path)
            actual = declared.resolve(strict=False)
            expected = expected_path.resolve(strict=False)
        except MeetingStorageValidationError:
            raise
        except (OSError, TypeError) as exc:
            raise MeetingStorageValidationError(
                f"Could not resolve meeting recording output path: {declared_path!r}"
            ) from exc
        if actual != expected:
            raise MeetingStorageValidationError(
                f"Meeting recording output path must be exactly {expected_path}"
            )

    def _inspect_asset(
        self,
        path: Path,
        unsafe_error: type[MeetingStorageError],
    ) -> bool:
        directory = path.parent
        directory_stat = self._lstat_optional(directory)
        if directory_stat is None:
            return False
        if self._is_reparse(directory, directory_stat) or not stat.S_ISDIR(
            directory_stat.st_mode
        ):
            raise unsafe_error(
                f"Meeting session directory is not a safe real directory: {directory}"
            )
        asset_stat = self._lstat_optional(path)
        if asset_stat is None:
            return False
        if self._is_reparse(path, asset_stat) or not stat.S_ISREG(asset_stat.st_mode):
            raise unsafe_error(
                f"Meeting audio asset is not a safe regular file: {path}"
            )
        return True

    @staticmethod
    def _lstat_optional(path: Path) -> Optional[os.stat_result]:
        try:
            return path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MeetingStorageFilesystemError(
                f"Could not inspect meeting storage path: {path}"
            ) from exc

    @staticmethod
    def _is_reparse(path: Path, path_stat: os.stat_result) -> bool:
        if stat.S_ISLNK(path_stat.st_mode):
            return True
        file_attributes = getattr(path_stat, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if file_attributes & reparse_flag:
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())

    def _validate_empty_reserved_directory(self, directory: Path) -> None:
        directory_stat = self._lstat_optional(directory)
        if directory_stat is None:
            raise MeetingStorageCollisionError(
                f"Reserved meeting directory no longer exists: {directory}"
            )
        if self._is_reparse(directory, directory_stat) or not stat.S_ISDIR(
            directory_stat.st_mode
        ):
            raise MeetingStorageCollisionError(
                f"Reserved meeting target is unsafe: {directory}"
            )
        try:
            has_entries = next(directory.iterdir(), None) is not None
        except OSError as exc:
            raise MeetingStorageFilesystemError(
                f"Could not inspect reserved meeting directory: {directory}"
            ) from exc
        if has_entries:
            raise MeetingStorageCollisionError(
                f"Reserved meeting directory is no longer empty: {directory}"
            )

    def _ensure_root_directory(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            if not self._root.is_dir():
                raise NotADirectoryError(str(self._root))
        except OSError as exc:
            raise MeetingStorageFilesystemError(
                f"Could not create meeting storage root: {self._root}"
            ) from exc

    @staticmethod
    def _require_session_id(session_id: uuid.UUID) -> uuid.UUID:
        if not isinstance(session_id, uuid.UUID):
            raise MeetingStorageValidationError("session_id must be a uuid.UUID")
        return session_id

    @staticmethod
    def _decode_canonical_uuid(raw: object) -> uuid.UUID:
        if not isinstance(raw, str):
            raise MeetingStorageDecodeError("Persisted meeting ID must be text")
        try:
            parsed = uuid.UUID(raw)
        except (ValueError, AttributeError) as exc:
            raise MeetingStorageDecodeError("Invalid persisted meeting UUID") from exc
        if str(parsed) != raw:
            raise MeetingStorageDecodeError("Persisted meeting UUID is not canonical")
        return parsed

    @staticmethod
    def _encode_enum(value: object, enum_type: type, field_name: str) -> str:
        if not isinstance(value, enum_type):
            raise MeetingStorageValidationError(
                f"{field_name} must be a {enum_type.__name__}"
            )
        return value.name

    @staticmethod
    def _decode_enum(raw: object, enum_type: type, field_name: str):
        if not isinstance(raw, str):
            raise MeetingStorageDecodeError(f"{field_name} must be text")
        try:
            return enum_type[raw]
        except KeyError as exc:
            raise MeetingStorageDecodeError(f"Unknown {field_name}: {raw!r}") from exc

    @staticmethod
    def _encode_datetime(value: object, field_name: str) -> str:
        if not isinstance(value, datetime):
            raise MeetingStorageValidationError(f"{field_name} must be a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise MeetingStorageValidationError(f"{field_name} must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()

    @classmethod
    def _encode_optional_datetime(
        cls,
        value: object,
        field_name: str,
    ) -> Optional[str]:
        if value is None:
            return None
        return cls._encode_datetime(value, field_name)

    @staticmethod
    def _decode_datetime(raw: object, field_name: str) -> datetime:
        if not isinstance(raw, str):
            raise MeetingStorageDecodeError(f"{field_name} must be text")
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise MeetingStorageDecodeError(
                f"Malformed persisted {field_name}"
            ) from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise MeetingStorageDecodeError(
                f"Persisted {field_name} must be timezone-aware"
            )
        return value.astimezone(timezone.utc)

    @classmethod
    def _decode_optional_datetime(
        cls,
        raw: object,
        field_name: str,
    ) -> Optional[datetime]:
        if raw is None:
            return None
        return cls._decode_datetime(raw, field_name)

    @staticmethod
    def _validate_int(
        value: object,
        field_name: str,
        error_type: type[MeetingStorageError],
    ) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise error_type(f"{field_name} must be an integer")
        return value

    @classmethod
    def _validate_nonnegative_int(
        cls,
        value: object,
        field_name: str,
        error_type: type[MeetingStorageError],
    ) -> int:
        result = cls._validate_int(value, field_name, error_type)
        if result < 0:
            raise error_type(f"{field_name} must be nonnegative")
        return result

    @classmethod
    def _validate_positive_int(
        cls,
        value: object,
        field_name: str,
        error_type: type[MeetingStorageError],
    ) -> int:
        result = cls._validate_int(value, field_name, error_type)
        if result <= 0:
            raise error_type(f"{field_name} must be positive")
        return result

    @classmethod
    def _validate_optional_nonnegative_int(
        cls,
        value: object,
        field_name: str,
        error_type: type[MeetingStorageError],
    ) -> Optional[int]:
        if value is None:
            return None
        return cls._validate_nonnegative_int(value, field_name, error_type)

    @staticmethod
    def _decode_bool(value: object, field_name: str) -> bool:
        if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
            raise MeetingStorageDecodeError(f"{field_name} must be 0 or 1")
        return bool(value)


__all__ = [
    "ExistingBundleValidator",
    "MeetingErrorPersistenceRecord",
    "MeetingPersistenceBundle",
    "MeetingRepository",
    "MeetingStorage",
    "MeetingStorageCollisionError",
    "MeetingStorageConflictError",
    "MeetingStorageDatabaseError",
    "MeetingStorageDecodeError",
    "MeetingStorageError",
    "MeetingStorageFilesystemError",
    "MeetingStoragePaths",
    "MeetingStorageValidationError",
    "MeetingTimingPersistenceRecord",
    "MeetingTrackPersistenceRecord",
    "StoredMeeting",
    "StoredMeetingAudioTrack",
    "StoredMeetingError",
    "StoredMeetingTimingAnchor",
]
