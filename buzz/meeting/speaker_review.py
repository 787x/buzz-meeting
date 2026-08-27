"""Pure meeting-speaker review domain and service.

PR15 consumes durable final-transcription words plus caller-provided
diarization turns and speaker attributions.  It never invokes ASR,
diarization, or the PR14 mapper, and it has no Qt or filesystem dependency.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from buzz.meeting.final_transcription import (
    FinalTranscriptionGeneration,
    FinalTranscriptionStatus,
    FinalTranscriptionTrack,
    FinalTranscriptionTrackStatus,
    MeetingTranscriptWord,
)
from buzz.meeting.meeting_audio_tracks import MeetingTrackRole
from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationBackend,
    SpeakerDiarizationTurn,
)
from buzz.meeting.speaker_mapping import (
    MeetingSpeakerKey,
    SpeakerAttributionStatus,
    SpeakerAttributedWord,
)


class SpeakerReviewError(Exception):
    """Base error for speaker-review failures."""


class SpeakerReviewConfigError(SpeakerReviewError):
    """Invalid caller input."""


class SpeakerReviewConflictError(SpeakerReviewError):
    """A canonical review already exists."""


class SpeakerReviewNotFoundError(SpeakerReviewError):
    """A requested review, speaker, or word does not exist."""


class SpeakerReviewStateError(SpeakerReviewError):
    """The requested lifecycle operation is invalid."""


class SpeakerReviewDecodeError(SpeakerReviewError):
    """Persisted speaker-review state is corrupt."""


class SpeakerReviewStaleError(SpeakerReviewError):
    """The durable source generation changed after the review snapshot."""


class SpeakerReviewStatus(Enum):
    UNREVIEWED = "UNREVIEWED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class SpeakerReviewAnalysisState(Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    COMPLETED = "COMPLETED"


_ROLE_ORDER = {
    MeetingTrackRole.MICROPHONE: 0,
    MeetingTrackRole.REMOTE: 1,
}
_MAPPING_ALGORITHM_VERSION = 1
_SUPPORTED_SOURCE_PROFILE_VERSION = 2
_MAX_DISPLAY_NAME_LENGTH = 256


def _validate_int(value: object, name: str, *, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpeakerReviewConfigError(f"{name} must be an int, not bool")
    if minimum is not None and value < minimum:
        raise SpeakerReviewConfigError(f"{name} must be >= {minimum}")
    return value


def _validate_turn(turn: object, name: str) -> SpeakerDiarizationTurn:
    if not isinstance(turn, SpeakerDiarizationTurn):
        raise SpeakerReviewConfigError(
            f"{name} must be SpeakerDiarizationTurn, got {type(turn)}"
        )
    _validate_int(turn.speaker_index, f"{name}.speaker_index")
    start_ms = _validate_int(turn.start_ms, f"{name}.start_ms")
    end_ms = _validate_int(turn.end_ms, f"{name}.end_ms")
    if end_ms < start_ms:
        raise SpeakerReviewConfigError(f"{name}.end_ms must be >= start_ms")
    return turn


def _normalize_display_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SpeakerReviewConfigError("display_name must be str or None")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_DISPLAY_NAME_LENGTH:
        raise SpeakerReviewConfigError(
            f"display_name must be <= {_MAX_DISPLAY_NAME_LENGTH} code points"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SpeakerReviewTrackAnalysis:
    source_role: MeetingTrackRole
    turns: tuple[SpeakerDiarizationTurn, ...]
    backend: SpeakerDiarizationBackend
    profile_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_role, MeetingTrackRole):
            raise SpeakerReviewConfigError("source_role must be MeetingTrackRole")
        if not isinstance(self.turns, tuple):
            raise SpeakerReviewConfigError("turns must be a tuple")
        if not isinstance(self.backend, SpeakerDiarizationBackend):
            raise SpeakerReviewConfigError("backend must be SpeakerDiarizationBackend")
        _validate_int(self.profile_version, "profile_version", minimum=1)
        for ordinal, turn in enumerate(self.turns):
            _validate_turn(turn, f"turns[{ordinal}]")


@dataclass(frozen=True, slots=True)
class ReviewedSpeaker:
    id: uuid.UUID
    ordinal: int
    display_name: str | None


@dataclass(frozen=True, slots=True)
class SpeakerReviewTrack:
    source_role: MeetingTrackRole
    source_track_status: FinalTranscriptionTrackStatus
    source_word_count: int
    analysis_state: SpeakerReviewAnalysisState
    turn_count: int
    diarization_backend: SpeakerDiarizationBackend | None
    diarization_profile_version: int | None


@dataclass(frozen=True, slots=True)
class SpeakerReviewTurn:
    source_role: MeetingTrackRole
    ordinal: int
    speaker_index: int
    local_start_ms: int
    local_end_ms: int


@dataclass(frozen=True, slots=True)
class SpeakerReviewCluster:
    machine_speaker: MeetingSpeakerKey
    reviewed_speaker_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class ReviewedSpeakerWord:
    word: MeetingTranscriptWord
    machine_status: SpeakerAttributionStatus
    machine_speaker: MeetingSpeakerKey | None
    effective_speaker_id: uuid.UUID | None
    overridden: bool


@dataclass(frozen=True, slots=True)
class MeetingSpeakerReview:
    id: uuid.UUID
    source_generation_id: uuid.UUID
    source_profile_version: int
    source_track_count: int
    mapping_algorithm_version: int
    status: SpeakerReviewStatus
    revision: int
    next_speaker_ordinal: int
    time_created: datetime
    time_updated: datetime
    time_completed: datetime | None
    tracks: tuple[SpeakerReviewTrack, ...]
    turns: tuple[SpeakerReviewTurn, ...]
    clusters: tuple[SpeakerReviewCluster, ...]
    speakers: tuple[ReviewedSpeaker, ...]
    words: tuple[ReviewedSpeakerWord, ...]


# Persistence records intentionally contain only SQLite-compatible values.


@dataclass(frozen=True, slots=True)
class SpeakerReviewHeaderRecord:
    id: str
    source_generation_id: str
    source_profile_version: int
    source_track_count: int
    mapping_algorithm_version: int
    status: str
    revision: int
    next_speaker_ordinal: int
    time_created: str
    time_updated: str
    time_completed: str | None


@dataclass(frozen=True, slots=True)
class SpeakerReviewTrackRecord:
    review_id: str
    source_generation_id: str
    role: str
    source_track_status: str
    source_word_count: int
    analysis_state: str
    turn_count: int
    diarization_backend: str | None
    diarization_profile_version: int | None


@dataclass(frozen=True, slots=True)
class SpeakerReviewTurnRecord:
    review_id: str
    role: str
    ordinal: int
    speaker_index: int
    local_start_ms: int
    local_end_ms: int


@dataclass(frozen=True, slots=True)
class SpeakerReviewClusterRecord:
    review_id: str
    role: str
    speaker_index: int


@dataclass(frozen=True, slots=True)
class ReviewedSpeakerRecord:
    review_id: str
    id: str
    ordinal: int
    display_name: str | None


@dataclass(frozen=True, slots=True)
class SpeakerClusterAssignmentRecord:
    review_id: str
    role: str
    speaker_index: int
    reviewed_speaker_id: str


@dataclass(frozen=True, slots=True)
class SpeakerWordAttributionRecord:
    review_id: str
    source_generation_id: str
    role: str
    word_ordinal: int
    attribution_status: str
    machine_speaker_index: int | None


@dataclass(frozen=True, slots=True)
class SpeakerWordOverrideRecord:
    review_id: str
    role: str
    word_ordinal: int
    reviewed_speaker_id: str | None


@dataclass(frozen=True, slots=True)
class SpeakerReviewPersistenceBundle:
    header: SpeakerReviewHeaderRecord
    tracks: tuple[SpeakerReviewTrackRecord, ...]
    turns: tuple[SpeakerReviewTurnRecord, ...]
    clusters: tuple[SpeakerReviewClusterRecord, ...]
    speakers: tuple[ReviewedSpeakerRecord, ...]
    assignments: tuple[SpeakerClusterAssignmentRecord, ...]
    attributions: tuple[SpeakerWordAttributionRecord, ...]
    overrides: tuple[SpeakerWordOverrideRecord, ...]


@dataclass(frozen=True, slots=True)
class SpeakerReviewMutationRecord:
    status: str
    revision: int
    next_speaker_ordinal: int
    time_updated: str
    time_completed: str | None


class FinalTranscriptSource(Protocol):
    def load_generation(
        self, generation_id: uuid.UUID
    ) -> FinalTranscriptionGeneration | None:
        ...

    def load_words(self, generation_id: uuid.UUID) -> tuple[MeetingTranscriptWord, ...]:
        ...


class MeetingSpeakerRepository(Protocol):
    def create_review(self, bundle: SpeakerReviewPersistenceBundle) -> None:
        ...

    def load_review(self, review_id: str) -> SpeakerReviewPersistenceBundle | None:
        ...

    def load_review_for_generation(
        self, generation_id: str
    ) -> SpeakerReviewPersistenceBundle | None:
        ...

    def rename_speaker(
        self,
        review_id: str,
        speaker_id: str,
        display_name: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        ...

    def create_speaker(
        self,
        speaker: ReviewedSpeakerRecord,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        ...

    def merge_speakers(
        self,
        review_id: str,
        source_speaker_id: str,
        target_speaker_id: str,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        ...

    def set_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        reviewed_speaker_id: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        ...

    def clear_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        ...

    def mark_completed(
        self, review_id: str, mutation: SpeakerReviewMutationRecord
    ) -> None:
        ...

    def delete_review_for_generation(self, generation_id: str) -> None:
        ...


class MeetingSpeakerReviewService:
    """Create, validate, and mutate durable meeting speaker reviews."""

    def __init__(
        self,
        repository: MeetingSpeakerRepository,
        final_transcript_source: FinalTranscriptSource,
        *,
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._source = final_transcript_source
        self._id_factory = id_factory
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def create_review(
        self,
        source_generation_id: uuid.UUID,
        analyses: Sequence[SpeakerReviewTrackAnalysis],
        attributed_words: Sequence[SpeakerAttributedWord],
    ) -> MeetingSpeakerReview:
        generation_id = _require_uuid(source_generation_id, "source_generation_id")
        if self._repository.load_review_for_generation(str(generation_id)) is not None:
            raise SpeakerReviewConflictError(
                f"A review already exists for generation {generation_id}"
            )

        generation = self._source.load_generation(generation_id)
        if generation is None:
            raise SpeakerReviewNotFoundError(
                f"Final transcription generation {generation_id} not found"
            )
        self._validate_creation_generation(generation_id, generation)
        words = self._load_source_words(generation_id, creation=True)
        tracks_by_role = self._creation_tracks(generation, words)
        analyses_by_role = self._validate_analyses(analyses, tracks_by_role)

        turn_records_by_role: dict[
            MeetingTrackRole, tuple[SpeakerDiarizationTurn, ...]
        ] = {}
        cluster_keys: set[MeetingSpeakerKey] = set()
        for role in tracks_by_role:
            analysis = analyses_by_role.get(role)
            turns = () if analysis is None else tuple(analysis.turns)
            turn_records_by_role[role] = turns
            for turn in turns:
                cluster_keys.add(MeetingSpeakerKey(role, turn.speaker_index))

        attributed_by_key = self._validate_attributed_words(
            words,
            attributed_words,
            analyses_by_role,
            turn_records_by_role,
            cluster_keys,
        )

        review_id = self._new_uuid("review id")
        review_id_raw = str(review_id)
        generation_id_raw = str(generation_id)
        ordered_cluster_keys = tuple(sorted(cluster_keys, key=_speaker_key_order))
        speaker_records: list[ReviewedSpeakerRecord] = []
        assignment_records: list[SpeakerClusterAssignmentRecord] = []
        for ordinal, cluster_key in enumerate(ordered_cluster_keys):
            speaker_id = self._new_uuid("reviewed speaker id")
            speaker_records.append(
                ReviewedSpeakerRecord(
                    review_id=review_id_raw,
                    id=str(speaker_id),
                    ordinal=ordinal,
                    display_name=None,
                )
            )
            assignment_records.append(
                SpeakerClusterAssignmentRecord(
                    review_id=review_id_raw,
                    role=cluster_key.source_role.name,
                    speaker_index=cluster_key.speaker_index,
                    reviewed_speaker_id=str(speaker_id),
                )
            )

        now = self._now()
        now_raw = _encode_datetime(now)
        track_records: list[SpeakerReviewTrackRecord] = []
        turn_records: list[SpeakerReviewTurnRecord] = []
        cluster_records: list[SpeakerReviewClusterRecord] = []
        for role, source_track in sorted(
            tracks_by_role.items(), key=lambda item: _role_order(item[0])
        ):
            analysis = analyses_by_role.get(role)
            role_turns = turn_records_by_role[role]
            track_records.append(
                SpeakerReviewTrackRecord(
                    review_id=review_id_raw,
                    source_generation_id=generation_id_raw,
                    role=role.name,
                    source_track_status=source_track.status.name,
                    source_word_count=sum(
                        1 for word in words if word.source_role is role
                    ),
                    analysis_state=(
                        SpeakerReviewAnalysisState.NOT_PROVIDED.value
                        if analysis is None
                        else SpeakerReviewAnalysisState.COMPLETED.value
                    ),
                    turn_count=len(role_turns),
                    diarization_backend=(
                        None if analysis is None else analysis.backend.value
                    ),
                    diarization_profile_version=(
                        None if analysis is None else analysis.profile_version
                    ),
                )
            )
            for ordinal, turn in enumerate(role_turns):
                turn_records.append(
                    SpeakerReviewTurnRecord(
                        review_id=review_id_raw,
                        role=role.name,
                        ordinal=ordinal,
                        speaker_index=turn.speaker_index,
                        local_start_ms=turn.start_ms,
                        local_end_ms=turn.end_ms,
                    )
                )

        for key in ordered_cluster_keys:
            cluster_records.append(
                SpeakerReviewClusterRecord(
                    review_id=review_id_raw,
                    role=key.source_role.name,
                    speaker_index=key.speaker_index,
                )
            )

        attribution_records = tuple(
            SpeakerWordAttributionRecord(
                review_id=review_id_raw,
                source_generation_id=generation_id_raw,
                role=word.source_role.name,
                word_ordinal=word.source_word_ordinal,
                attribution_status=attributed_by_key[
                    (word.source_role, word.source_word_ordinal)
                ].status.name,
                machine_speaker_index=(
                    attributed_by_key[
                        (word.source_role, word.source_word_ordinal)
                    ].speaker.speaker_index
                    if attributed_by_key[
                        (word.source_role, word.source_word_ordinal)
                    ].speaker
                    is not None
                    else None
                ),
            )
            for word in words
        )
        bundle = SpeakerReviewPersistenceBundle(
            header=SpeakerReviewHeaderRecord(
                id=review_id_raw,
                source_generation_id=generation_id_raw,
                source_profile_version=generation.profile_version,
                source_track_count=len(generation.tracks),
                mapping_algorithm_version=_MAPPING_ALGORITHM_VERSION,
                status=SpeakerReviewStatus.UNREVIEWED.value,
                revision=0,
                next_speaker_ordinal=len(speaker_records),
                time_created=now_raw,
                time_updated=now_raw,
                time_completed=None,
            ),
            tracks=tuple(track_records),
            turns=tuple(turn_records),
            clusters=tuple(cluster_records),
            speakers=tuple(speaker_records),
            assignments=tuple(assignment_records),
            attributions=attribution_records,
            overrides=(),
        )
        self._repository.create_review(bundle)
        created = self._load_review_id(review_id)
        if created is None:
            raise SpeakerReviewStateError("Repository did not return created review")
        return created

    def load_review(self, review_id: uuid.UUID) -> MeetingSpeakerReview | None:
        return self._load_review_id(_require_uuid(review_id, "review_id"))

    def load_review_for_generation(
        self, source_generation_id: uuid.UUID
    ) -> MeetingSpeakerReview | None:
        generation_id = _require_uuid(source_generation_id, "source_generation_id")
        bundle = self._repository.load_review_for_generation(str(generation_id))
        if bundle is None:
            return None
        review = self._decode_and_validate(bundle)
        if review.source_generation_id != generation_id:
            raise SpeakerReviewDecodeError(
                "Repository returned review for a different source generation"
            )
        return review

    def rename_speaker(
        self,
        review_id: uuid.UUID,
        speaker_id: uuid.UUID,
        display_name: str | None,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        sid = _require_uuid(speaker_id, "speaker_id")
        speaker = _find_speaker(review, sid)
        normalized = _normalize_display_name(display_name)
        if speaker.display_name == normalized:
            return review
        mutation = self._edit_mutation(review)
        self._repository.rename_speaker(str(review.id), str(sid), normalized, mutation)
        return self._require_reloaded(review.id)

    def create_speaker(
        self,
        review_id: uuid.UUID,
        display_name: str | None = None,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        normalized = _normalize_display_name(display_name)
        speaker_id = self._new_uuid("reviewed speaker id")
        mutation = self._edit_mutation(
            review, next_speaker_ordinal=review.next_speaker_ordinal + 1
        )
        self._repository.create_speaker(
            ReviewedSpeakerRecord(
                review_id=str(review.id),
                id=str(speaker_id),
                ordinal=review.next_speaker_ordinal,
                display_name=normalized,
            ),
            mutation,
        )
        return self._require_reloaded(review.id)

    def merge_speakers(
        self,
        review_id: uuid.UUID,
        source_speaker_id: uuid.UUID,
        target_speaker_id: uuid.UUID,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        source_id = _require_uuid(source_speaker_id, "source_speaker_id")
        target_id = _require_uuid(target_speaker_id, "target_speaker_id")
        _find_speaker(review, source_id)
        _find_speaker(review, target_id)
        if source_id == target_id:
            return review
        self._repository.merge_speakers(
            str(review.id),
            str(source_id),
            str(target_id),
            self._edit_mutation(review),
        )
        return self._require_reloaded(review.id)

    def assign_word(
        self,
        review_id: uuid.UUID,
        role: MeetingTrackRole,
        word_ordinal: int,
        reviewed_speaker_id: uuid.UUID,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        role_value, ordinal, word = self._mutation_word(review, role, word_ordinal)
        speaker_id = _require_uuid(reviewed_speaker_id, "reviewed_speaker_id")
        _find_speaker(review, speaker_id)
        if word.overridden and word.effective_speaker_id == speaker_id:
            return review
        self._repository.set_word_override(
            str(review.id),
            role_value.name,
            ordinal,
            str(speaker_id),
            self._edit_mutation(review),
        )
        return self._require_reloaded(review.id)

    def unassign_word(
        self,
        review_id: uuid.UUID,
        role: MeetingTrackRole,
        word_ordinal: int,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        role_value, ordinal, word = self._mutation_word(review, role, word_ordinal)
        if word.overridden and word.effective_speaker_id is None:
            return review
        self._repository.set_word_override(
            str(review.id),
            role_value.name,
            ordinal,
            None,
            self._edit_mutation(review),
        )
        return self._require_reloaded(review.id)

    def clear_word_override(
        self,
        review_id: uuid.UUID,
        role: MeetingTrackRole,
        word_ordinal: int,
    ) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        role_value, ordinal, word = self._mutation_word(review, role, word_ordinal)
        if not word.overridden:
            return review
        self._repository.clear_word_override(
            str(review.id),
            role_value.name,
            ordinal,
            self._edit_mutation(review),
        )
        return self._require_reloaded(review.id)

    def mark_completed(self, review_id: uuid.UUID) -> MeetingSpeakerReview:
        review = self._require_review(review_id)
        if review.status is SpeakerReviewStatus.COMPLETED:
            return review
        now = self._now()
        self._repository.mark_completed(
            str(review.id),
            SpeakerReviewMutationRecord(
                status=SpeakerReviewStatus.COMPLETED.value,
                revision=review.revision + 1,
                next_speaker_ordinal=review.next_speaker_ordinal,
                time_updated=_encode_datetime(now),
                time_completed=_encode_datetime(now),
            ),
        )
        return self._require_reloaded(review.id)

    def reset_review(self, source_generation_id: uuid.UUID) -> None:
        generation_id = _require_uuid(source_generation_id, "source_generation_id")
        self._repository.delete_review_for_generation(str(generation_id))

    def _load_review_id(self, review_id: uuid.UUID) -> MeetingSpeakerReview | None:
        bundle = self._repository.load_review(str(review_id))
        if bundle is None:
            return None
        review = self._decode_and_validate(bundle)
        if review.id != review_id:
            raise SpeakerReviewDecodeError("Repository returned a different review")
        return review

    def _require_review(self, review_id: uuid.UUID) -> MeetingSpeakerReview:
        rid = _require_uuid(review_id, "review_id")
        review = self._load_review_id(rid)
        if review is None:
            raise SpeakerReviewNotFoundError(f"Speaker review {rid} not found")
        return review

    def _require_reloaded(self, review_id: uuid.UUID) -> MeetingSpeakerReview:
        review = self._load_review_id(review_id)
        if review is None:
            raise SpeakerReviewStateError("Review disappeared during mutation")
        return review

    def _edit_mutation(
        self,
        review: MeetingSpeakerReview,
        *,
        next_speaker_ordinal: int | None = None,
    ) -> SpeakerReviewMutationRecord:
        return SpeakerReviewMutationRecord(
            status=SpeakerReviewStatus.IN_PROGRESS.value,
            revision=review.revision + 1,
            next_speaker_ordinal=(
                review.next_speaker_ordinal
                if next_speaker_ordinal is None
                else next_speaker_ordinal
            ),
            time_updated=_encode_datetime(self._now()),
            time_completed=None,
        )

    @staticmethod
    def _mutation_word(
        review: MeetingSpeakerReview,
        role: MeetingTrackRole,
        word_ordinal: int,
    ) -> tuple[MeetingTrackRole, int, ReviewedSpeakerWord]:
        if not isinstance(role, MeetingTrackRole):
            raise SpeakerReviewConfigError("role must be MeetingTrackRole")
        ordinal = _validate_int(word_ordinal, "word_ordinal")
        for word in review.words:
            if (
                word.word.source_role is role
                and word.word.source_word_ordinal == ordinal
            ):
                return role, ordinal, word
        raise SpeakerReviewNotFoundError(
            f"Word {role.name}/{ordinal} not found in review {review.id}"
        )

    def _decode_and_validate(
        self, bundle: SpeakerReviewPersistenceBundle
    ) -> MeetingSpeakerReview:
        if not isinstance(bundle, SpeakerReviewPersistenceBundle):
            raise SpeakerReviewDecodeError("Repository returned an invalid bundle")
        header = bundle.header
        review_id = _decode_uuid(header.id, "review id")
        generation_id = _decode_uuid(
            header.source_generation_id, "source generation id"
        )
        source_profile_version = _decode_int(
            header.source_profile_version, "source_profile_version", minimum=1
        )
        if source_profile_version != _SUPPORTED_SOURCE_PROFILE_VERSION:
            raise SpeakerReviewDecodeError(
                f"Unsupported source_profile_version: {source_profile_version}"
            )
        source_track_count = _decode_int(
            header.source_track_count, "source_track_count"
        )
        mapping_version = _decode_int(
            header.mapping_algorithm_version,
            "mapping_algorithm_version",
            minimum=1,
        )
        if mapping_version != _MAPPING_ALGORITHM_VERSION:
            raise SpeakerReviewDecodeError(
                f"Unsupported mapping_algorithm_version: {mapping_version}"
            )
        status = _decode_enum(SpeakerReviewStatus, header.status, "review status")
        revision = _decode_int(header.revision, "revision")
        next_ordinal = _decode_int(header.next_speaker_ordinal, "next_speaker_ordinal")
        time_created = _decode_datetime(header.time_created, "time_created")
        time_updated = _decode_datetime(header.time_updated, "time_updated")
        time_completed = (
            None
            if header.time_completed is None
            else _decode_datetime(header.time_completed, "time_completed")
        )
        if time_updated < time_created:
            raise SpeakerReviewDecodeError(
                "Review update timestamp precedes creation timestamp"
            )
        if status is SpeakerReviewStatus.UNREVIEWED:
            lifecycle_coherent = (
                revision == 0
                and time_updated == time_created
                and time_completed is None
            )
        elif status is SpeakerReviewStatus.IN_PROGRESS:
            lifecycle_coherent = revision >= 1 and time_completed is None
        else:
            lifecycle_coherent = (
                revision >= 1
                and time_completed is not None
                and time_completed == time_updated
            )
        if not lifecycle_coherent:
            raise SpeakerReviewDecodeError(
                "Review status, revision, and timestamps are incoherent"
            )
        if len(bundle.tracks) != source_track_count:
            raise SpeakerReviewDecodeError(
                "Persisted review track count does not match frozen source_track_count"
            )

        tracks: dict[MeetingTrackRole, SpeakerReviewTrack] = {}
        track_records_by_role: dict[MeetingTrackRole, SpeakerReviewTrackRecord] = {}
        for record in bundle.tracks:
            _require_record_review(record.review_id, header.id, "review track")
            if record.source_generation_id != header.source_generation_id:
                raise SpeakerReviewDecodeError(
                    "Review track source generation does not match header"
                )
            role = _decode_role(record.role)
            if role in tracks:
                raise SpeakerReviewDecodeError(f"Duplicate review track {role.name}")
            track_status = _decode_enum(
                FinalTranscriptionTrackStatus,
                record.source_track_status,
                "source track status",
                by_name=True,
            )
            word_count = _decode_int(record.source_word_count, "source_word_count")
            analysis_state = _decode_enum(
                SpeakerReviewAnalysisState, record.analysis_state, "analysis_state"
            )
            turn_count = _decode_int(record.turn_count, "turn_count")
            backend: SpeakerDiarizationBackend | None
            profile_version: int | None
            if analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED:
                if (
                    turn_count != 0
                    or record.diarization_backend is not None
                    or record.diarization_profile_version is not None
                ):
                    raise SpeakerReviewDecodeError(
                        "NOT_PROVIDED track has diarization provenance or turns"
                    )
                backend = None
                profile_version = None
            else:
                if record.diarization_backend is None:
                    raise SpeakerReviewDecodeError(
                        "COMPLETED track is missing diarization backend"
                    )
                backend = _decode_enum(
                    SpeakerDiarizationBackend,
                    record.diarization_backend,
                    "diarization backend",
                )
                if record.diarization_profile_version is None:
                    raise SpeakerReviewDecodeError(
                        "COMPLETED track is missing diarization profile version"
                    )
                profile_version = _decode_int(
                    record.diarization_profile_version,
                    "diarization_profile_version",
                    minimum=1,
                )
            tracks[role] = SpeakerReviewTrack(
                source_role=role,
                source_track_status=track_status,
                source_word_count=word_count,
                analysis_state=analysis_state,
                turn_count=turn_count,
                diarization_backend=backend,
                diarization_profile_version=profile_version,
            )
            track_records_by_role[role] = record

        turns_by_role: dict[MeetingTrackRole, list[SpeakerReviewTurn]] = {
            role: [] for role in tracks
        }
        for record in bundle.turns:
            _require_record_review(record.review_id, header.id, "speaker turn")
            role = _decode_role(record.role)
            if role not in tracks:
                raise SpeakerReviewDecodeError("Speaker turn references missing track")
            ordinal = _decode_int(record.ordinal, "turn ordinal")
            speaker_index = _decode_int(record.speaker_index, "speaker_index")
            start_ms = _decode_int(record.local_start_ms, "local_start_ms")
            end_ms = _decode_int(record.local_end_ms, "local_end_ms")
            if end_ms < start_ms:
                raise SpeakerReviewDecodeError("Turn end precedes turn start")
            turns_by_role[role].append(
                SpeakerReviewTurn(
                    source_role=role,
                    ordinal=ordinal,
                    speaker_index=speaker_index,
                    local_start_ms=start_ms,
                    local_end_ms=end_ms,
                )
            )
        for role, track in tracks.items():
            role_turns = turns_by_role[role]
            ordinals = sorted(turn.ordinal for turn in role_turns)
            if len(role_turns) != track.turn_count or ordinals != list(
                range(track.turn_count)
            ):
                raise SpeakerReviewDecodeError(
                    f"Turn rows for {role.name} do not match frozen turn_count"
                )

        cluster_keys: set[MeetingSpeakerKey] = set()
        for record in bundle.clusters:
            _require_record_review(record.review_id, header.id, "speaker cluster")
            role = _decode_role(record.role)
            if role not in tracks:
                raise SpeakerReviewDecodeError(
                    "Speaker cluster references missing track"
                )
            key = MeetingSpeakerKey(
                role, _decode_int(record.speaker_index, "cluster speaker_index")
            )
            if key in cluster_keys:
                raise SpeakerReviewDecodeError("Duplicate machine speaker cluster")
            cluster_keys.add(key)
        expected_clusters = {
            MeetingSpeakerKey(turn.source_role, turn.speaker_index)
            for role_turns in turns_by_role.values()
            for turn in role_turns
        }
        if cluster_keys != expected_clusters:
            raise SpeakerReviewDecodeError(
                "Machine cluster rows do not exactly match persisted turns"
            )
        for role, track in tracks.items():
            if track.analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED and any(
                key.source_role is role for key in cluster_keys
            ):
                raise SpeakerReviewDecodeError(
                    "NOT_PROVIDED track contains machine clusters"
                )

        speakers_by_id: dict[uuid.UUID, ReviewedSpeaker] = {}
        speaker_ordinals: set[int] = set()
        for record in bundle.speakers:
            _require_record_review(record.review_id, header.id, "reviewed speaker")
            speaker_id = _decode_uuid(record.id, "reviewed speaker id")
            if speaker_id in speakers_by_id:
                raise SpeakerReviewDecodeError("Duplicate reviewed speaker UUID")
            ordinal = _decode_int(record.ordinal, "reviewed speaker ordinal")
            if ordinal in speaker_ordinals:
                raise SpeakerReviewDecodeError("Duplicate reviewed speaker ordinal")
            if ordinal >= next_ordinal:
                raise SpeakerReviewDecodeError(
                    "Reviewed speaker ordinal reaches allocation watermark"
                )
            display_name = _decode_display_name(record.display_name)
            speaker_ordinals.add(ordinal)
            speakers_by_id[speaker_id] = ReviewedSpeaker(
                id=speaker_id, ordinal=ordinal, display_name=display_name
            )

        assignments: dict[MeetingSpeakerKey, uuid.UUID] = {}
        for record in bundle.assignments:
            _require_record_review(record.review_id, header.id, "cluster assignment")
            key = MeetingSpeakerKey(
                _decode_role(record.role),
                _decode_int(record.speaker_index, "assignment speaker_index"),
            )
            if key in assignments:
                raise SpeakerReviewDecodeError("Duplicate cluster assignment")
            speaker_id = _decode_uuid(
                record.reviewed_speaker_id, "assigned reviewed speaker id"
            )
            if speaker_id not in speakers_by_id:
                raise SpeakerReviewDecodeError(
                    "Cluster assignment references another review or missing speaker"
                )
            assignments[key] = speaker_id
        if set(assignments) != cluster_keys:
            raise SpeakerReviewDecodeError(
                "Cluster assignments do not exactly match machine clusters"
            )

        attributions: dict[
            tuple[MeetingTrackRole, int],
            tuple[SpeakerAttributionStatus, MeetingSpeakerKey | None],
        ] = {}
        for record in bundle.attributions:
            _require_record_review(record.review_id, header.id, "word attribution")
            if record.source_generation_id != header.source_generation_id:
                raise SpeakerReviewDecodeError(
                    "Word attribution source generation does not match header"
                )
            role = _decode_role(record.role)
            if role not in tracks:
                raise SpeakerReviewDecodeError(
                    "Word attribution references missing review track"
                )
            ordinal = _decode_int(record.word_ordinal, "word ordinal")
            key = (role, ordinal)
            if key in attributions:
                raise SpeakerReviewDecodeError("Duplicate machine word attribution")
            machine_status = _decode_enum(
                SpeakerAttributionStatus,
                record.attribution_status,
                "attribution status",
                by_name=True,
            )
            machine_speaker: MeetingSpeakerKey | None
            if machine_status is SpeakerAttributionStatus.ASSIGNED:
                if record.machine_speaker_index is None:
                    raise SpeakerReviewDecodeError(
                        "ASSIGNED attribution is missing machine speaker"
                    )
                machine_speaker = MeetingSpeakerKey(
                    role,
                    _decode_int(
                        record.machine_speaker_index,
                        "attribution machine_speaker_index",
                    ),
                )
                if machine_speaker not in cluster_keys:
                    raise SpeakerReviewDecodeError(
                        "ASSIGNED attribution references missing machine cluster"
                    )
            else:
                if record.machine_speaker_index is not None:
                    raise SpeakerReviewDecodeError(
                        f"{machine_status.name} attribution has machine speaker"
                    )
                machine_speaker = None
            if (
                tracks[role].analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED
                or tracks[role].turn_count == 0
            ) and machine_status is not SpeakerAttributionStatus.NO_OVERLAP:
                raise SpeakerReviewDecodeError(
                    "Track without turns has non-NO_OVERLAP attribution"
                )
            attributions[key] = (machine_status, machine_speaker)

        overrides: dict[tuple[MeetingTrackRole, int], uuid.UUID | None] = {}
        for record in bundle.overrides:
            _require_record_review(record.review_id, header.id, "word override")
            role = _decode_role(record.role)
            key = (
                role,
                _decode_int(record.word_ordinal, "override word ordinal"),
            )
            if key in overrides:
                raise SpeakerReviewDecodeError("Duplicate word override")
            if key not in attributions:
                raise SpeakerReviewDecodeError(
                    "Word override references another review or missing word"
                )
            speaker_id = (
                None
                if record.reviewed_speaker_id is None
                else _decode_uuid(
                    record.reviewed_speaker_id, "override reviewed speaker id"
                )
            )
            if speaker_id is not None and speaker_id not in speakers_by_id:
                raise SpeakerReviewDecodeError(
                    "Word override references another review or missing speaker"
                )
            overrides[key] = speaker_id

        generation = self._source.load_generation(generation_id)
        if generation is None:
            raise SpeakerReviewDecodeError(
                f"Review source generation {generation_id} is missing"
            )
        if generation.generation_id != generation_id:
            raise SpeakerReviewDecodeError(
                "Source returned a generation with a mismatched UUID"
            )
        current_tracks = self._decode_source_tracks(generation)
        words = self._load_source_words(generation_id, creation=False)
        current_counts = {role: 0 for role in current_tracks}
        source_words_by_key: dict[
            tuple[MeetingTrackRole, int], MeetingTranscriptWord
        ] = {}
        for word in words:
            if word.source_role not in current_tracks:
                raise SpeakerReviewDecodeError(
                    "Source word references a missing source track"
                )
            key = (word.source_role, word.source_word_ordinal)
            if key in source_words_by_key:
                raise SpeakerReviewDecodeError("Source words contain duplicate keys")
            source_words_by_key[key] = word
            current_counts[word.source_role] += 1

        if generation.profile_version != source_profile_version:
            raise SpeakerReviewStaleError("Source profile version changed")
        if len(current_tracks) != source_track_count:
            raise SpeakerReviewStaleError("Source track count changed")
        if set(current_tracks) != set(tracks):
            raise SpeakerReviewStaleError("Source track role set changed")
        for role, track in tracks.items():
            if (
                current_tracks[role].status is not track.source_track_status
                or current_counts[role] != track.source_word_count
            ):
                raise SpeakerReviewStaleError(
                    f"Source track snapshot changed for {role.name}"
                )

        if set(attributions) != set(source_words_by_key):
            raise SpeakerReviewDecodeError(
                "Machine attributions do not exactly match source words"
            )
        for role, track in tracks.items():
            if (
                sum(1 for key in attributions if key[0] is role)
                != track.source_word_count
            ):
                raise SpeakerReviewDecodeError(
                    f"Attribution count does not match frozen count for {role.name}"
                )

        reviewed_words: list[ReviewedSpeakerWord] = []
        for word in words:
            key = (word.source_role, word.source_word_ordinal)
            machine_status, machine_speaker = attributions[key]
            if key in overrides:
                effective = overrides[key]
                overridden = True
            elif machine_speaker is not None:
                effective = assignments[machine_speaker]
                overridden = False
            else:
                effective = None
                overridden = False
            reviewed_words.append(
                ReviewedSpeakerWord(
                    word=word,
                    machine_status=machine_status,
                    machine_speaker=machine_speaker,
                    effective_speaker_id=effective,
                    overridden=overridden,
                )
            )

        ordered_turns = tuple(
            sorted(
                (turn for values in turns_by_role.values() for turn in values),
                key=lambda turn: (_role_order(turn.source_role), turn.ordinal),
            )
        )
        return MeetingSpeakerReview(
            id=review_id,
            source_generation_id=generation_id,
            source_profile_version=source_profile_version,
            source_track_count=source_track_count,
            mapping_algorithm_version=mapping_version,
            status=status,
            revision=revision,
            next_speaker_ordinal=next_ordinal,
            time_created=time_created,
            time_updated=time_updated,
            time_completed=time_completed,
            tracks=tuple(
                sorted(tracks.values(), key=lambda item: _role_order(item.source_role))
            ),
            turns=ordered_turns,
            clusters=tuple(
                SpeakerReviewCluster(key, assignments[key])
                for key in sorted(cluster_keys, key=_speaker_key_order)
            ),
            speakers=tuple(
                sorted(speakers_by_id.values(), key=lambda item: item.ordinal)
            ),
            words=tuple(reviewed_words),
        )

    @staticmethod
    def _validate_creation_generation(
        generation_id: uuid.UUID, generation: FinalTranscriptionGeneration
    ) -> None:
        if not isinstance(generation, FinalTranscriptionGeneration):
            raise SpeakerReviewConfigError(
                "Final transcript source returned an invalid generation"
            )
        if generation.generation_id != generation_id:
            raise SpeakerReviewConfigError(
                "Final transcript source returned a mismatched generation"
            )
        if generation.profile_version != _SUPPORTED_SOURCE_PROFILE_VERSION:
            raise SpeakerReviewConfigError(
                "Speaker review supports exactly final-transcription profile version 2"
            )
        if generation.status not in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
        ):
            raise SpeakerReviewStateError(
                f"Cannot review generation in {generation.status.name} state"
            )

    def _creation_tracks(
        self,
        generation: FinalTranscriptionGeneration,
        words: tuple[MeetingTranscriptWord, ...],
    ) -> dict[MeetingTrackRole, FinalTranscriptionTrack]:
        tracks = self._decode_source_tracks(generation, creation=True)
        for word in words:
            if word.source_role not in tracks:
                raise SpeakerReviewConfigError(
                    "Source word references role absent from generation tracks"
                )
        return tracks

    @staticmethod
    def _decode_source_tracks(
        generation: FinalTranscriptionGeneration, *, creation: bool = False
    ) -> dict[MeetingTrackRole, FinalTranscriptionTrack]:
        error = SpeakerReviewConfigError if creation else SpeakerReviewDecodeError
        if not isinstance(generation.tracks, tuple):
            raise error("Source generation tracks must be a tuple")
        result: dict[MeetingTrackRole, FinalTranscriptionTrack] = {}
        for track in generation.tracks:
            if not isinstance(track, FinalTranscriptionTrack):
                raise error("Source generation has an invalid track")
            role = getattr(track, "role", None)
            status = getattr(track, "status", None)
            if not isinstance(role, MeetingTrackRole):
                raise error("Source track has invalid role")
            if not isinstance(status, FinalTranscriptionTrackStatus):
                raise error("Source track has invalid status")
            if role in result:
                raise error(f"Source generation has duplicate role {role.name}")
            result[role] = track
        return result

    def _load_source_words(
        self, generation_id: uuid.UUID, *, creation: bool
    ) -> tuple[MeetingTranscriptWord, ...]:
        words = self._source.load_words(generation_id)
        error = SpeakerReviewConfigError if creation else SpeakerReviewDecodeError
        if not isinstance(words, tuple):
            raise error("Final transcript source words must be a tuple")
        for index, word in enumerate(words):
            try:
                _validate_source_word(word, f"words[{index}]")
            except SpeakerReviewConfigError as exc:
                raise error(str(exc)) from exc
        return words

    @staticmethod
    def _validate_analyses(
        analyses: Sequence[SpeakerReviewTrackAnalysis],
        tracks_by_role: dict[MeetingTrackRole, object],
    ) -> dict[MeetingTrackRole, SpeakerReviewTrackAnalysis]:
        if isinstance(analyses, (str, bytes)) or not isinstance(analyses, Sequence):
            raise SpeakerReviewConfigError("analyses must be a sequence")
        result: dict[MeetingTrackRole, SpeakerReviewTrackAnalysis] = {}
        for analysis in analyses:
            if not isinstance(analysis, SpeakerReviewTrackAnalysis):
                raise SpeakerReviewConfigError(
                    "analyses must contain SpeakerReviewTrackAnalysis values"
                )
            if analysis.source_role in result:
                raise SpeakerReviewConfigError(
                    f"Duplicate analysis for {analysis.source_role.name}"
                )
            if analysis.source_role not in tracks_by_role:
                raise SpeakerReviewConfigError(
                    f"Analysis role {analysis.source_role.name} is not a source track"
                )
            result[analysis.source_role] = analysis
        return result

    @staticmethod
    def _validate_attributed_words(
        words: tuple[MeetingTranscriptWord, ...],
        attributed_words: Sequence[SpeakerAttributedWord],
        analyses_by_role: dict[MeetingTrackRole, SpeakerReviewTrackAnalysis],
        turns_by_role: dict[MeetingTrackRole, tuple[SpeakerDiarizationTurn, ...]],
        cluster_keys: set[MeetingSpeakerKey],
    ) -> dict[tuple[MeetingTrackRole, int], SpeakerAttributedWord]:
        if isinstance(attributed_words, (str, bytes)) or not isinstance(
            attributed_words, Sequence
        ):
            raise SpeakerReviewConfigError("attributed_words must be a sequence")
        source_by_key: dict[tuple[MeetingTrackRole, int], MeetingTranscriptWord] = {}
        for word in words:
            key = (word.source_role, word.source_word_ordinal)
            if key in source_by_key:
                raise SpeakerReviewConfigError("Source words contain duplicate keys")
            source_by_key[key] = word
        result: dict[tuple[MeetingTrackRole, int], SpeakerAttributedWord] = {}
        for attributed in attributed_words:
            if not isinstance(attributed, SpeakerAttributedWord):
                raise SpeakerReviewConfigError(
                    "attributed_words must contain SpeakerAttributedWord values"
                )
            word = _validate_source_word(attributed.word, "attributed.word")
            key = (word.source_role, word.source_word_ordinal)
            if key in result:
                raise SpeakerReviewConfigError(
                    f"Duplicate attributed word key {word.source_role.name}/{word.source_word_ordinal}"
                )
            if key not in source_by_key:
                raise SpeakerReviewConfigError(
                    "Attributed word does not exist in source"
                )
            if word != source_by_key[key]:
                raise SpeakerReviewConfigError(
                    "Attributed word does not equal its current source word"
                )
            if not isinstance(attributed.status, SpeakerAttributionStatus):
                raise SpeakerReviewConfigError("Invalid attribution status")
            if attributed.status is SpeakerAttributionStatus.ASSIGNED:
                if attributed.speaker is None:
                    raise SpeakerReviewConfigError(
                        "ASSIGNED attribution requires a machine speaker"
                    )
                if attributed.speaker.source_role is not word.source_role:
                    raise SpeakerReviewConfigError(
                        "Attributed machine speaker has a different source role"
                    )
                if attributed.speaker not in cluster_keys:
                    raise SpeakerReviewConfigError(
                        "Attributed machine speaker is absent from raw turns"
                    )
            elif attributed.speaker is not None:
                raise SpeakerReviewConfigError(
                    f"{attributed.status.name} attribution requires speaker=None"
                )
            analysis = analyses_by_role.get(word.source_role)
            if (analysis is None or not turns_by_role[word.source_role]) and (
                attributed.status is not SpeakerAttributionStatus.NO_OVERLAP
            ):
                raise SpeakerReviewConfigError(
                    "Role without provided turns requires NO_OVERLAP attribution"
                )
            result[key] = attributed
        if set(result) != set(source_by_key):
            missing = set(source_by_key) - set(result)
            extra = set(result) - set(source_by_key)
            raise SpeakerReviewConfigError(
                f"Attributed word keys must exactly match source (missing={missing}, extra={extra})"
            )
        return result

    def _new_uuid(self, name: str) -> uuid.UUID:
        value = self._id_factory()
        if not isinstance(value, uuid.UUID):
            raise SpeakerReviewConfigError(f"id_factory returned invalid {name}")
        return value

    def _now(self) -> datetime:
        value = self._clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() != timedelta(0)
        ):
            raise SpeakerReviewConfigError("clock must return a UTC-aware datetime")
        return value.astimezone(timezone.utc)


def _find_speaker(
    review: MeetingSpeakerReview, speaker_id: uuid.UUID
) -> ReviewedSpeaker:
    for speaker in review.speakers:
        if speaker.id == speaker_id:
            return speaker
    raise SpeakerReviewNotFoundError(
        f"Reviewed speaker {speaker_id} not found in review {review.id}"
    )


def _validate_source_word(word: object, name: str) -> MeetingTranscriptWord:
    if not isinstance(word, MeetingTranscriptWord):
        raise SpeakerReviewConfigError(f"{name} must be MeetingTranscriptWord")
    if not isinstance(word.source_role, MeetingTrackRole):
        raise SpeakerReviewConfigError(f"{name}.source_role is invalid")
    segment_ordinal = _validate_int(
        word.source_segment_ordinal, f"{name}.source_segment_ordinal"
    )
    del segment_ordinal
    _validate_int(word.source_word_ordinal, f"{name}.source_word_ordinal")
    local_start = _validate_int(word.local_start_ms, f"{name}.local_start_ms")
    local_end = _validate_int(word.local_end_ms, f"{name}.local_end_ms")
    if local_end < local_start:
        raise SpeakerReviewConfigError(f"{name} local interval is invalid")
    start_ns = _validate_int(word.start_ns, f"{name}.start_ns", minimum=None)
    end_ns = _validate_int(word.end_ns, f"{name}.end_ns", minimum=None)
    if end_ns < start_ns:
        raise SpeakerReviewConfigError(f"{name} meeting interval is invalid")
    if not isinstance(word.text, str):
        raise SpeakerReviewConfigError(f"{name}.text must be str")
    return word


def _require_uuid(value: object, name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise SpeakerReviewConfigError(f"{name} must be uuid.UUID")
    return value


def _decode_uuid(raw: object, name: str) -> uuid.UUID:
    if not isinstance(raw, str):
        raise SpeakerReviewDecodeError(f"{name} must be canonical UUID text")
    try:
        value = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise SpeakerReviewDecodeError(f"Invalid {name}: {raw!r}") from exc
    if str(value) != raw:
        raise SpeakerReviewDecodeError(f"Non-canonical {name}: {raw!r}")
    return value


def _decode_int(raw: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise SpeakerReviewDecodeError(f"Invalid {name}: {raw!r}")
    return raw


def _decode_enum(
    enum_type: type[Enum],
    raw: object,
    name: str,
    *,
    by_name: bool = False,
):
    if not isinstance(raw, str):
        raise SpeakerReviewDecodeError(f"Invalid {name}: {raw!r}")
    try:
        return enum_type[raw] if by_name else enum_type(raw)
    except (KeyError, ValueError) as exc:
        raise SpeakerReviewDecodeError(f"Unknown {name}: {raw!r}") from exc


def _decode_role(raw: object) -> MeetingTrackRole:
    if not isinstance(raw, str):
        raise SpeakerReviewDecodeError(f"Invalid source role: {raw!r}")
    try:
        return MeetingTrackRole[raw]
    except KeyError as exc:
        raise SpeakerReviewDecodeError(f"Unknown source role: {raw!r}") from exc


def _decode_display_name(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SpeakerReviewDecodeError("Persisted display_name must be str or NULL")
    try:
        normalized = _normalize_display_name(raw)
    except SpeakerReviewConfigError as exc:
        raise SpeakerReviewDecodeError(str(exc)) from exc
    if normalized is None or normalized != raw:
        raise SpeakerReviewDecodeError("Persisted display_name is not canonical")
    return normalized


def _decode_datetime(raw: object, name: str) -> datetime:
    if not isinstance(raw, str):
        raise SpeakerReviewDecodeError(f"{name} must be UTC timestamp text")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SpeakerReviewDecodeError(f"Malformed {name}: {raw!r}") from exc
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SpeakerReviewDecodeError(f"{name} must be UTC-aware")
    return value.astimezone(timezone.utc)


def _encode_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SpeakerReviewConfigError("datetime must be UTC-aware")
    return value.astimezone(timezone.utc).isoformat()


def _require_record_review(raw: object, expected: str, name: str) -> None:
    if raw != expected:
        raise SpeakerReviewDecodeError(
            f"{name} references another review: {raw!r} != {expected!r}"
        )


def _role_order(role: MeetingTrackRole) -> int:
    try:
        return _ROLE_ORDER[role]
    except KeyError as exc:
        raise SpeakerReviewDecodeError(f"Unknown source role: {role!r}") from exc


def _speaker_key_order(key: MeetingSpeakerKey) -> tuple[int, int]:
    return (_role_order(key.source_role), key.speaker_index)


__all__ = [
    "FinalTranscriptSource",
    "MeetingSpeakerRepository",
    "MeetingSpeakerReview",
    "MeetingSpeakerReviewService",
    "ReviewedSpeaker",
    "ReviewedSpeakerRecord",
    "ReviewedSpeakerWord",
    "SpeakerClusterAssignmentRecord",
    "SpeakerReviewAnalysisState",
    "SpeakerReviewCluster",
    "SpeakerReviewClusterRecord",
    "SpeakerReviewConfigError",
    "SpeakerReviewConflictError",
    "SpeakerReviewDecodeError",
    "SpeakerReviewError",
    "SpeakerReviewHeaderRecord",
    "SpeakerReviewMutationRecord",
    "SpeakerReviewNotFoundError",
    "SpeakerReviewPersistenceBundle",
    "SpeakerReviewStaleError",
    "SpeakerReviewStateError",
    "SpeakerReviewStatus",
    "SpeakerReviewTrack",
    "SpeakerReviewTrackAnalysis",
    "SpeakerReviewTrackRecord",
    "SpeakerReviewTurn",
    "SpeakerReviewTurnRecord",
    "SpeakerWordAttributionRecord",
    "SpeakerWordOverrideRecord",
]
