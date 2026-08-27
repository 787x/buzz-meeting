from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
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
from buzz.meeting.speaker_review import (
    MeetingSpeakerReviewService,
    ReviewedSpeakerRecord,
    SpeakerReviewAnalysisState,
    SpeakerReviewConfigError,
    SpeakerReviewConflictError,
    SpeakerReviewDecodeError,
    SpeakerReviewMutationRecord,
    SpeakerReviewNotFoundError,
    SpeakerReviewPersistenceBundle,
    SpeakerReviewStaleError,
    SpeakerReviewStateError,
    SpeakerReviewStatus,
    SpeakerReviewTrackAnalysis,
    SpeakerWordOverrideRecord,
)

MIC = MeetingTrackRole.MICROPHONE
REMOTE = MeetingTrackRole.REMOTE
UTC = timezone.utc


class FakeFinalTranscriptSource:
    def __init__(
        self,
        generation: FinalTranscriptionGeneration,
        words: tuple[MeetingTranscriptWord, ...],
    ) -> None:
        self.generation = generation
        self.words = words

    def load_generation(
        self, generation_id: uuid.UUID
    ) -> FinalTranscriptionGeneration | None:
        if generation_id != self.generation.generation_id:
            return None
        return self.generation

    def load_words(self, generation_id: uuid.UUID) -> tuple[MeetingTranscriptWord, ...]:
        if generation_id != self.generation.generation_id:
            return ()
        return self.words


class FakeMeetingSpeakerRepository:
    def __init__(self) -> None:
        self.bundle: SpeakerReviewPersistenceBundle | None = None

    def create_review(self, bundle: SpeakerReviewPersistenceBundle) -> None:
        if self.bundle is not None:
            raise SpeakerReviewConflictError("exists")
        self.bundle = bundle

    def load_review(self, review_id: str) -> SpeakerReviewPersistenceBundle | None:
        if self.bundle is None or self.bundle.header.id != review_id:
            return None
        return self.bundle

    def load_review_for_generation(
        self, generation_id: str
    ) -> SpeakerReviewPersistenceBundle | None:
        if (
            self.bundle is None
            or self.bundle.header.source_generation_id != generation_id
        ):
            return None
        return self.bundle

    def rename_speaker(
        self,
        review_id: str,
        speaker_id: str,
        display_name: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        bundle = self._require(review_id)
        self.bundle = replace(
            bundle,
            header=_mutated_header(bundle, mutation),
            speakers=tuple(
                replace(speaker, display_name=display_name)
                if speaker.id == speaker_id
                else speaker
                for speaker in bundle.speakers
            ),
        )

    def create_speaker(
        self,
        speaker: ReviewedSpeakerRecord,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        bundle = self._require(speaker.review_id)
        self.bundle = replace(
            bundle,
            header=_mutated_header(bundle, mutation),
            speakers=(*bundle.speakers, speaker),
        )

    def merge_speakers(
        self,
        review_id: str,
        source_speaker_id: str,
        target_speaker_id: str,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        bundle = self._require(review_id)
        self.bundle = replace(
            bundle,
            header=_mutated_header(bundle, mutation),
            speakers=tuple(
                speaker
                for speaker in bundle.speakers
                if speaker.id != source_speaker_id
            ),
            assignments=tuple(
                replace(assignment, reviewed_speaker_id=target_speaker_id)
                if assignment.reviewed_speaker_id == source_speaker_id
                else assignment
                for assignment in bundle.assignments
            ),
            overrides=tuple(
                replace(override, reviewed_speaker_id=target_speaker_id)
                if override.reviewed_speaker_id == source_speaker_id
                else override
                for override in bundle.overrides
            ),
        )

    def set_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        reviewed_speaker_id: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        bundle = self._require(review_id)
        replacement = SpeakerWordOverrideRecord(
            review_id, role, word_ordinal, reviewed_speaker_id
        )
        retained = tuple(
            override
            for override in bundle.overrides
            if (override.role, override.word_ordinal) != (role, word_ordinal)
        )
        self.bundle = replace(
            bundle,
            header=_mutated_header(bundle, mutation),
            overrides=(*retained, replacement),
        )

    def clear_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        bundle = self._require(review_id)
        self.bundle = replace(
            bundle,
            header=_mutated_header(bundle, mutation),
            overrides=tuple(
                override
                for override in bundle.overrides
                if (override.role, override.word_ordinal) != (role, word_ordinal)
            ),
        )

    def mark_completed(
        self, review_id: str, mutation: SpeakerReviewMutationRecord
    ) -> None:
        bundle = self._require(review_id)
        self.bundle = replace(bundle, header=_mutated_header(bundle, mutation))

    def delete_review_for_generation(self, generation_id: str) -> None:
        if (
            self.bundle is not None
            and self.bundle.header.source_generation_id == generation_id
        ):
            self.bundle = None

    def _require(self, review_id: str) -> SpeakerReviewPersistenceBundle:
        assert self.bundle is not None
        assert self.bundle.header.id == review_id
        return self.bundle


class FixedIds:
    def __init__(self, start: int = 1) -> None:
        self.next = start

    def __call__(self) -> uuid.UUID:
        value = uuid.UUID(int=self.next)
        self.next += 1
        return value


class TickClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)
        self.calls = 0

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(minutes=1)
        self.calls += 1
        return result


def _mutated_header(bundle, mutation):
    return replace(
        bundle.header,
        status=mutation.status,
        revision=mutation.revision,
        next_speaker_ordinal=mutation.next_speaker_ordinal,
        time_updated=mutation.time_updated,
        time_completed=mutation.time_completed,
    )


def _word(
    role: MeetingTrackRole,
    ordinal: int,
    *,
    start_ms: int | None = None,
    text: str | None = None,
) -> MeetingTranscriptWord:
    local_start = ordinal * 100 if start_ms is None else start_ms
    return MeetingTranscriptWord(
        source_role=role,
        source_segment_ordinal=0,
        source_word_ordinal=ordinal,
        local_start_ms=local_start,
        local_end_ms=local_start + 80,
        start_ns=local_start * 1_000_000,
        end_ns=(local_start + 80) * 1_000_000,
        text=text or f"{role.name.lower()}-{ordinal}",
    )


def _generation(
    *,
    status: FinalTranscriptionStatus = FinalTranscriptionStatus.COMPLETED,
    profile_version: int = 2,
    mic_status: FinalTranscriptionTrackStatus = FinalTranscriptionTrackStatus.COMPLETED,
    remote_status: FinalTranscriptionTrackStatus = FinalTranscriptionTrackStatus.COMPLETED,
) -> FinalTranscriptionGeneration:
    config = (
        FinalTranscriptionConfig(profile_version=2, whisper_model_size="LARGE")
        if profile_version == 2
        else FinalTranscriptionConfig()
    )
    return FinalTranscriptionGeneration(
        generation_id=uuid.UUID(int=1000),
        meeting_id=uuid.UUID(int=2000),
        profile_version=profile_version,
        status=status,
        config=config,
        tracks=(
            FinalTranscriptionTrack(MIC, mic_status),
            FinalTranscriptionTrack(REMOTE, remote_status),
        ),
    )


def _inputs():
    words = (
        _word(MIC, 0, start_ms=0),
        _word(REMOTE, 0, start_ms=50),
        _word(MIC, 1, start_ms=100),
    )
    analyses = (
        SpeakerReviewTrackAnalysis(
            MIC,
            (
                SpeakerDiarizationTurn(7, 90, 190),
                SpeakerDiarizationTurn(2, 0, 90),
            ),
            SpeakerDiarizationBackend.MSDD,
            1,
        ),
        SpeakerReviewTrackAnalysis(
            REMOTE,
            (SpeakerDiarizationTurn(2, 0, 150),),
            SpeakerDiarizationBackend.SORTFORMER,
            1,
        ),
    )
    attributed = (
        SpeakerAttributedWord(
            words[0], MeetingSpeakerKey(MIC, 2), SpeakerAttributionStatus.ASSIGNED
        ),
        SpeakerAttributedWord(
            words[1], MeetingSpeakerKey(REMOTE, 2), SpeakerAttributionStatus.ASSIGNED
        ),
        SpeakerAttributedWord(words[2], None, SpeakerAttributionStatus.AMBIGUOUS),
    )
    return words, analyses, attributed


def _service(
    *,
    generation: FinalTranscriptionGeneration | None = None,
    words: tuple[MeetingTranscriptWord, ...] | None = None,
):
    default_words, analyses, attributed = _inputs()
    source = FakeFinalTranscriptSource(
        generation or _generation(), words or default_words
    )
    repository = FakeMeetingSpeakerRepository()
    clock = TickClock()
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=clock
    )
    return service, repository, source, clock, analyses, attributed


def test_completed_create_snapshots_every_track_and_machine_layer():
    service, repository, source, clock, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )

    assert review.id == uuid.UUID(int=1)
    assert review.status is SpeakerReviewStatus.UNREVIEWED
    assert review.revision == 0
    assert (
        review.time_created == review.time_updated == datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert review.time_completed is None
    assert review.source_track_count == 2
    assert review.mapping_algorithm_version == 1
    assert [track.source_role for track in review.tracks] == [MIC, REMOTE]
    assert [track.source_word_count for track in review.tracks] == [2, 1]
    assert review.tracks[0].diarization_backend is SpeakerDiarizationBackend.MSDD
    assert review.tracks[1].diarization_backend is SpeakerDiarizationBackend.SORTFORMER
    assert all(
        track.analysis_state is SpeakerReviewAnalysisState.COMPLETED
        for track in review.tracks
    )
    assert [
        (turn.source_role, turn.ordinal, turn.speaker_index) for turn in review.turns
    ] == [
        (MIC, 0, 7),
        (MIC, 1, 2),
        (REMOTE, 0, 2),
    ]
    assert [cluster.machine_speaker for cluster in review.clusters] == [
        MeetingSpeakerKey(MIC, 2),
        MeetingSpeakerKey(MIC, 7),
        MeetingSpeakerKey(REMOTE, 2),
    ]
    assert [speaker.id for speaker in review.speakers] == [
        uuid.UUID(int=2),
        uuid.UUID(int=3),
        uuid.UUID(int=4),
    ]
    assert [speaker.ordinal for speaker in review.speakers] == [0, 1, 2]
    assert review.next_speaker_ordinal == 3
    assert [word.word for word in review.words] == list(source.words)
    assert repository.bundle is not None
    assert repository.bundle.overrides == ()
    assert clock.calls == 1


def test_partial_create_and_not_provided_track_snapshot():
    words = (_word(MIC, 0),)
    generation = _generation(
        status=FinalTranscriptionStatus.PARTIAL,
        remote_status=FinalTranscriptionTrackStatus.FAILED,
    )
    service, _, source, _, _, _ = _service(generation=generation, words=words)
    analysis = SpeakerReviewTrackAnalysis(
        MIC,
        (SpeakerDiarizationTurn(0, 0, 100),),
        SpeakerDiarizationBackend.MSDD,
        1,
    )
    attributed = (
        SpeakerAttributedWord(
            words[0], MeetingSpeakerKey(MIC, 0), SpeakerAttributionStatus.ASSIGNED
        ),
    )

    review = service.create_review(
        source.generation.generation_id, (analysis,), attributed
    )

    remote = review.tracks[1]
    assert remote.source_track_status is FinalTranscriptionTrackStatus.FAILED
    assert remote.source_word_count == 0
    assert remote.analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED
    assert remote.turn_count == 0
    assert remote.diarization_backend is None
    assert remote.diarization_profile_version is None


@pytest.mark.parametrize(
    ("generation", "error"),
    [
        (_generation(profile_version=1), SpeakerReviewConfigError),
        (
            _generation(status=FinalTranscriptionStatus.QUEUED),
            SpeakerReviewStateError,
        ),
        (
            _generation(status=FinalTranscriptionStatus.IN_PROGRESS),
            SpeakerReviewStateError,
        ),
        (
            _generation(status=FinalTranscriptionStatus.FAILED),
            SpeakerReviewStateError,
        ),
    ],
)
def test_rejects_v1_and_nonterminal_sources(generation, error):
    service, _, source, _, analyses, attributed = _service(generation=generation)
    with pytest.raises(error):
        service.create_review(source.generation.generation_id, analyses, attributed)


def test_create_conflict_requires_explicit_reset():
    service, _, source, _, analyses, attributed = _service()
    service.create_review(source.generation.generation_id, analyses, attributed)
    with pytest.raises(SpeakerReviewConflictError):
        service.create_review(source.generation.generation_id, analyses, attributed)


def test_attributed_words_require_exact_unique_field_equal_source():
    service, _, source, _, analyses, attributed = _service()
    cases = [
        attributed[:-1],
        (*attributed, attributed[0]),
        (
            *attributed,
            SpeakerAttributedWord(
                _word(MIC, 9), None, SpeakerAttributionStatus.NO_OVERLAP
            ),
        ),
        (
            replace(attributed[0], word=replace(attributed[0].word, text="changed")),
            *attributed[1:],
        ),
    ]
    for candidate in cases:
        candidate_service = MeetingSpeakerReviewService(
            FakeMeetingSpeakerRepository(),
            source,
            id_factory=FixedIds(),
            clock=TickClock(),
        )
        with pytest.raises(SpeakerReviewConfigError):
            candidate_service.create_review(
                source.generation.generation_id, analyses, candidate
            )


def test_attribution_coherence_rejects_missing_cluster_and_cross_role():
    service, _, source, _, analyses, attributed = _service()
    missing_cluster = (
        replace(attributed[0], speaker=MeetingSpeakerKey(MIC, 99)),
        *attributed[1:],
    )
    cross_role = (
        replace(attributed[0], speaker=MeetingSpeakerKey(REMOTE, 2)),
        *attributed[1:],
    )
    for candidate in (missing_cluster, cross_role):
        candidate_service = MeetingSpeakerReviewService(
            FakeMeetingSpeakerRepository(),
            source,
            id_factory=FixedIds(),
            clock=TickClock(),
        )
        with pytest.raises(SpeakerReviewConfigError):
            candidate_service.create_review(
                source.generation.generation_id, analyses, candidate
            )


def test_not_provided_and_completed_empty_turns_require_no_overlap():
    word = _word(MIC, 0)
    generation = _generation()
    for analyses in (
        (),
        (SpeakerReviewTrackAnalysis(MIC, (), SpeakerDiarizationBackend.MSDD, 1),),
    ):
        source = FakeFinalTranscriptSource(generation, (word,))
        invalid = (
            SpeakerAttributedWord(word, None, SpeakerAttributionStatus.AMBIGUOUS),
        )
        service = MeetingSpeakerReviewService(
            FakeMeetingSpeakerRepository(),
            source,
            id_factory=FixedIds(),
            clock=TickClock(),
        )
        with pytest.raises(SpeakerReviewConfigError):
            service.create_review(generation.generation_id, analyses, invalid)

        service = MeetingSpeakerReviewService(
            FakeMeetingSpeakerRepository(),
            source,
            id_factory=FixedIds(),
            clock=TickClock(),
        )
        review = service.create_review(
            generation.generation_id,
            analyses,
            (SpeakerAttributedWord(word, None, SpeakerAttributionStatus.NO_OVERLAP),),
        )
        mic = review.tracks[0]
        assert mic.turn_count == 0
        assert mic.analysis_state is (
            SpeakerReviewAnalysisState.NOT_PROVIDED
            if not analyses
            else SpeakerReviewAnalysisState.COMPLETED
        )


def test_empty_source_words_and_zero_assigned_cluster_are_valid():
    source = FakeFinalTranscriptSource(_generation(), ())
    repository = FakeMeetingSpeakerRepository()
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    analysis = SpeakerReviewTrackAnalysis(
        MIC,
        (SpeakerDiarizationTurn(8, 0, 0),),
        SpeakerDiarizationBackend.MSDD,
        1,
    )
    review = service.create_review(source.generation.generation_id, (analysis,), ())
    assert review.words == ()
    assert review.clusters[0].machine_speaker == MeetingSpeakerKey(MIC, 8)
    assert len(review.speakers) == 1


def test_negative_meeting_timeline_word_offsets_are_preserved():
    word = replace(_word(MIC, 0), start_ns=-50_000_000, end_ns=30_000_000)
    source = FakeFinalTranscriptSource(_generation(), (word,))
    service = MeetingSpeakerReviewService(
        FakeMeetingSpeakerRepository(),
        source,
        id_factory=FixedIds(),
        clock=TickClock(),
    )
    review = service.create_review(
        source.generation.generation_id,
        (),
        (SpeakerAttributedWord(word, None, SpeakerAttributionStatus.NO_OVERLAP),),
    )
    assert review.words[0].word.start_ns == -50_000_000
    assert review.words[0].word.end_ns == 30_000_000


def test_manual_speaker_watermark_rename_normalization_duplicates_and_noops():
    service, _, source, clock, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    original_updated = review.time_updated
    speaker = review.speakers[0]

    unchanged = service.rename_speaker(review.id, speaker.id, "   ")
    assert unchanged.revision == 0
    assert unchanged.time_updated == original_updated
    assert clock.calls == 1

    named = service.rename_speaker(review.id, speaker.id, "  Alice  ")
    assert named.speakers[0].display_name == "Alice"
    assert named.status is SpeakerReviewStatus.IN_PROGRESS
    assert named.revision == 1
    same = service.rename_speaker(named.id, speaker.id, "Alice")
    assert same.revision == 1

    second_named = service.rename_speaker(named.id, named.speakers[1].id, " Alice ")
    assert [item.display_name for item in second_named.speakers[:2]] == [
        "Alice",
        "Alice",
    ]

    manual = service.create_speaker(second_named.id, "  Facilitator  ")
    created = manual.speakers[-1]
    assert created.id == uuid.UUID(int=5)
    assert created.ordinal == 3
    assert created.display_name == "Facilitator"
    assert manual.next_speaker_ordinal == 4
    assert all(cluster.reviewed_speaker_id != created.id for cluster in manual.clusters)

    cleared = service.rename_speaker(manual.id, created.id, " \t ")
    assert cleared.speakers[-1].display_name is None


def test_display_name_length_is_unicode_code_points():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    renamed = service.rename_speaker(review.id, review.speakers[0].id, "界" * 256)
    assert renamed.speakers[0].display_name == "界" * 256
    with pytest.raises(SpeakerReviewConfigError):
        service.rename_speaker(review.id, review.speakers[0].id, "界" * 257)


def test_merge_transfers_clusters_non_null_overrides_and_preserves_null_override():
    service, repository, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    source_speaker = review.speakers[0]
    target = review.speakers[2]
    review = service.assign_word(review.id, MIC, 1, source_speaker.id)
    review = service.unassign_word(review.id, REMOTE, 0)
    before_machine = repository.bundle.attributions
    before_turns = repository.bundle.turns

    merged = service.merge_speakers(review.id, source_speaker.id, target.id)

    assert source_speaker.id not in {speaker.id for speaker in merged.speakers}
    assert target.id in {speaker.id for speaker in merged.speakers}
    assert merged.next_speaker_ordinal == 3
    assert any(cluster.reviewed_speaker_id == target.id for cluster in merged.clusters)
    mic_override = next(
        word
        for word in merged.words
        if word.word.source_role is MIC and word.word.source_word_ordinal == 1
    )
    remote_override = next(
        word for word in merged.words if word.word.source_role is REMOTE
    )
    assert mic_override.overridden and mic_override.effective_speaker_id == target.id
    assert remote_override.overridden and remote_override.effective_speaker_id is None
    assert repository.bundle.attributions == before_machine
    assert repository.bundle.turns == before_turns

    no_op = service.merge_speakers(merged.id, target.id, target.id)
    assert no_op.revision == merged.revision
    with pytest.raises(SpeakerReviewNotFoundError):
        service.merge_speakers(merged.id, source_speaker.id, source_speaker.id)


def test_same_role_merge_keeps_target_uuid_and_never_reuses_source_ordinal():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    source_speaker, target = review.speakers[:2]
    merged = service.merge_speakers(review.id, source_speaker.id, target.id)
    assert target.id in {speaker.id for speaker in merged.speakers}
    assert source_speaker.id not in {speaker.id for speaker in merged.speakers}
    manual = service.create_speaker(merged.id)
    assert manual.speakers[-1].ordinal == 3
    assert manual.speakers[-1].ordinal != source_speaker.ordinal


@pytest.mark.parametrize(
    ("role", "ordinal"),
    [(MIC, 0), (MIC, 1), (REMOTE, 0)],
)
def test_assign_any_machine_status_then_explicit_unassign_and_clear(role, ordinal):
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    target = review.speakers[-1].id
    original = next(
        word
        for word in review.words
        if word.word.source_role is role and word.word.source_word_ordinal == ordinal
    )
    machine_status = original.machine_status

    assigned = service.assign_word(review.id, role, ordinal, target)
    changed = next(
        word
        for word in assigned.words
        if word.word.source_role is role and word.word.source_word_ordinal == ordinal
    )
    assert changed.overridden
    assert changed.effective_speaker_id == target
    assert changed.machine_status is machine_status
    identical = service.assign_word(assigned.id, role, ordinal, target)
    assert identical.revision == assigned.revision

    unassigned = service.unassign_word(assigned.id, role, ordinal)
    changed = next(
        word
        for word in unassigned.words
        if word.word.source_role is role and word.word.source_word_ordinal == ordinal
    )
    assert changed.overridden and changed.effective_speaker_id is None
    no_op = service.unassign_word(unassigned.id, role, ordinal)
    assert no_op.revision == unassigned.revision

    cleared = service.clear_word_override(unassigned.id, role, ordinal)
    restored = next(
        word
        for word in cleared.words
        if word.word.source_role is role and word.word.source_word_ordinal == ordinal
    )
    assert not restored.overridden
    assert restored.effective_speaker_id == original.effective_speaker_id
    no_op = service.clear_word_override(cleared.id, role, ordinal)
    assert no_op.revision == cleared.revision


def test_explicit_unassigned_is_distinct_from_no_override_even_if_effective_none():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    ambiguous = next(
        word
        for word in review.words
        if word.machine_status is SpeakerAttributionStatus.AMBIGUOUS
    )
    assert ambiguous.effective_speaker_id is None and not ambiguous.overridden
    changed = service.unassign_word(
        review.id, ambiguous.word.source_role, ambiguous.word.source_word_ordinal
    )
    explicit = next(word for word in changed.words if word.word == ambiguous.word)
    assert explicit.effective_speaker_id is None and explicit.overridden
    assert changed.revision == review.revision + 1


def test_assigning_current_machine_effective_speaker_still_records_override():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    machine_assigned = review.words[0]
    assert machine_assigned.effective_speaker_id is not None
    changed = service.assign_word(
        review.id,
        machine_assigned.word.source_role,
        machine_assigned.word.source_word_ordinal,
        machine_assigned.effective_speaker_id,
    )
    explicit = next(
        word for word in changed.words if word.word == machine_assigned.word
    )
    assert explicit.overridden
    assert explicit.effective_speaker_id == machine_assigned.effective_speaker_id
    assert changed.revision == review.revision + 1


def test_status_revision_and_timestamp_transitions_with_completed_edit():
    service, _, source, clock, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    completed = service.mark_completed(review.id)
    assert completed.status is SpeakerReviewStatus.COMPLETED
    assert completed.revision == 1
    assert completed.time_completed == completed.time_updated
    assert completed.time_updated == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)

    same = service.mark_completed(completed.id)
    assert same == completed
    assert clock.calls == 2

    edited = service.rename_speaker(completed.id, completed.speakers[0].id, "A")
    assert edited.status is SpeakerReviewStatus.IN_PROGRESS
    assert edited.revision == 2
    assert edited.time_completed is None
    assert edited.time_updated == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)


def test_constant_clock_semantic_rename_increments_revision_without_timestamp_bump():
    words, analyses, attributed = _inputs()
    source = FakeFinalTranscriptSource(_generation(), words)
    repository = FakeMeetingSpeakerRepository()
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    service = MeetingSpeakerReviewService(
        repository,
        source,
        id_factory=FixedIds(),
        clock=lambda: instant,
    )
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )

    renamed = service.rename_speaker(review.id, review.speakers[0].id, "Constant Clock")

    assert renamed.revision == 1
    assert renamed.status is SpeakerReviewStatus.IN_PROGRESS
    assert renamed.time_created == instant
    assert renamed.time_updated == instant
    assert service.load_review(review.id) == renamed


def test_mark_completed_allows_unresolved_words_and_unnamed_speakers():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    completed = service.mark_completed(review.id)
    assert completed.status is SpeakerReviewStatus.COMPLETED
    assert any(word.effective_speaker_id is None for word in completed.words)
    assert any(speaker.display_name is None for speaker in completed.speakers)


def test_partial_retry_makes_all_normal_entry_points_stale_but_reset_succeeds():
    words = (_word(MIC, 0),)
    generation = _generation(
        status=FinalTranscriptionStatus.PARTIAL,
        remote_status=FinalTranscriptionTrackStatus.FAILED,
    )
    source = FakeFinalTranscriptSource(generation, words)
    repository = FakeMeetingSpeakerRepository()
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    analysis = SpeakerReviewTrackAnalysis(
        MIC,
        (
            SpeakerDiarizationTurn(0, 0, 100),
            SpeakerDiarizationTurn(1, 100, 200),
        ),
        SpeakerDiarizationBackend.MSDD,
        1,
    )
    attributed = (
        SpeakerAttributedWord(
            words[0], MeetingSpeakerKey(MIC, 0), SpeakerAttributionStatus.ASSIGNED
        ),
    )
    review = service.create_review(generation.generation_id, (analysis,), attributed)
    source.generation = replace(
        generation,
        status=FinalTranscriptionStatus.COMPLETED,
        tracks=(
            generation.tracks[0],
            replace(
                generation.tracks[1], status=FinalTranscriptionTrackStatus.COMPLETED
            ),
        ),
    )
    source.words = (*words, _word(REMOTE, 0))

    calls = (
        lambda: service.load_review(review.id),
        lambda: service.load_review_for_generation(generation.generation_id),
        lambda: service.rename_speaker(review.id, review.speakers[0].id, "A"),
        lambda: service.create_speaker(review.id),
        lambda: service.merge_speakers(
            review.id, review.speakers[0].id, review.speakers[1].id
        ),
        lambda: service.assign_word(review.id, MIC, 0, review.speakers[1].id),
        lambda: service.unassign_word(review.id, MIC, 0),
        lambda: service.clear_word_override(review.id, MIC, 0),
        lambda: service.mark_completed(review.id),
    )
    for call in calls:
        with pytest.raises(SpeakerReviewStaleError):
            call()
    assert repository.bundle is not None

    service.reset_review(generation.generation_id)
    assert repository.bundle is None
    service.reset_review(generation.generation_id)


def test_real_source_track_count_change_remains_stale_not_decode():
    service, _, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    source.generation = replace(
        source.generation,
        tracks=(source.generation.tracks[0],),
    )
    source.words = tuple(word for word in source.words if word.source_role is MIC)

    with pytest.raises(SpeakerReviewStaleError):
        service.load_review(review.id)


def test_create_does_not_mutate_caller_sequences_or_values():
    service, _, source, _, analyses, attributed = _service()
    analyses_list = list(analyses)
    attributed_list = list(reversed(attributed))
    analyses_before = list(analyses_list)
    attributed_before = list(attributed_list)
    turns_before = tuple(analyses_list[0].turns)
    service.create_review(
        source.generation.generation_id, analyses_list, attributed_list
    )
    assert analyses_list == analyses_before
    assert attributed_list == attributed_before
    assert analyses_list[0].turns == turns_before


def test_fresh_load_detects_tail_corruption_and_bad_uuid_without_cache_clear():
    service, repository, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    assert repository.bundle is not None
    valid = repository.bundle
    repository.bundle = replace(valid, turns=valid.turns[:-1])
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)
    repository.bundle = replace(valid, attributions=valid.attributions[:-1])
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)
    repository.bundle = replace(valid, header=replace(valid.header, id="BAD"))
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review_for_generation(source.generation.generation_id)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"revision": 1}, id="unreviewed-revision-1"),
        pytest.param({"revision": 5}, id="unreviewed-revision-5"),
        pytest.param(
            {"status": "IN_PROGRESS", "revision": 0},
            id="in-progress-revision-0",
        ),
        pytest.param(
            {
                "status": "COMPLETED",
                "revision": 0,
                "time_completed": "2026-01-01T00:00:00+00:00",
            },
            id="completed-revision-0",
        ),
        pytest.param(
            {
                "status": "IN_PROGRESS",
                "revision": 1,
                "time_updated": "2025-12-31T23:59:59+00:00",
            },
            id="updated-before-created",
        ),
        pytest.param(
            {"time_updated": "2026-01-01T00:01:00+00:00"},
            id="unreviewed-updated-after-created",
        ),
        pytest.param(
            {"time_completed": "2026-01-01T00:00:00+00:00"},
            id="unreviewed-completed-non-null",
        ),
        pytest.param(
            {
                "status": "IN_PROGRESS",
                "revision": 1,
                "time_completed": "2026-01-01T00:00:00+00:00",
            },
            id="in-progress-completed-non-null",
        ),
        pytest.param(
            {"status": "COMPLETED", "revision": 1, "time_completed": None},
            id="completed-completed-null",
        ),
        pytest.param(
            {
                "status": "COMPLETED",
                "revision": 1,
                "time_completed": "2026-01-01T00:01:00+00:00",
            },
            id="completed-completed-not-updated",
        ),
    ],
)
def test_fresh_load_rejects_lifecycle_provenance_corruption(changes):
    service, repository, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    assert repository.bundle is not None
    valid = repository.bundle
    repository.bundle = replace(valid, header=replace(valid.header, **changes))

    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


@pytest.mark.parametrize("source_track_count", [1, 3])
def test_fresh_load_rejects_frozen_source_track_count_mismatch(source_track_count):
    service, repository, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    assert repository.bundle is not None
    valid = repository.bundle
    repository.bundle = replace(
        valid,
        header=replace(valid.header, source_track_count=source_track_count),
    )

    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


def test_fresh_load_detects_impossible_sql_corruption_and_missing_source():
    service, repository, source, _, analyses, attributed = _service()
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    assert repository.bundle is not None
    valid = repository.bundle

    duplicate_ordinal = replace(valid.speakers[1], ordinal=valid.speakers[0].ordinal)
    repository.bundle = replace(
        valid, speakers=(valid.speakers[0], duplicate_ordinal, *valid.speakers[2:])
    )
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)

    repository.bundle = replace(
        valid,
        tracks=(
            replace(valid.tracks[0], diarization_backend="unknown"),
            *valid.tracks[1:],
        ),
    )
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)

    repository.bundle = valid
    source.generation = replace(source.generation, generation_id=uuid.UUID(int=9999))
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


def test_missing_review_and_mutation_targets_use_not_found_errors():
    service, _, source, _, analyses, attributed = _service()
    assert service.load_review(uuid.UUID(int=999)) is None
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    with pytest.raises(SpeakerReviewNotFoundError):
        service.rename_speaker(review.id, uuid.UUID(int=999), "missing")
    with pytest.raises(SpeakerReviewNotFoundError):
        service.assign_word(review.id, MIC, 999, review.speakers[0].id)
