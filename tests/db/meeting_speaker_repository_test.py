"""QSql repository and persisted aggregate tests for PR15 speaker review."""

from __future__ import annotations

import gc
import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.db.meeting_speaker_repository import QSqlMeetingSpeakerRepository
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
    SpeakerReviewAnalysisState,
    SpeakerReviewConflictError,
    SpeakerReviewDecodeError,
    SpeakerReviewError,
    SpeakerReviewStatus,
    SpeakerReviewTrackAnalysis,
)

MIC = MeetingTrackRole.MICROPHONE
REMOTE = MeetingTrackRole.REMOTE
UTC = timezone.utc
SPEAKER_TABLES = (
    "meeting_speaker_review",
    "meeting_speaker_review_track",
    "meeting_speaker_turn",
    "meeting_speaker_cluster",
    "meeting_reviewed_speaker",
    "meeting_speaker_cluster_assignment",
    "meeting_speaker_word_attribution",
    "meeting_speaker_word_override",
)


class Source:
    def __init__(self, generation, words) -> None:
        self.generation = generation
        self.words = words

    def load_generation(self, generation_id):
        return (
            self.generation if generation_id == self.generation.generation_id else None
        )

    def load_words(self, generation_id):
        return self.words if generation_id == self.generation.generation_id else ()


class FixedIds:
    def __init__(self, start: int = 3000) -> None:
        self.next = start

    def __call__(self) -> uuid.UUID:
        value = uuid.UUID(int=self.next)
        self.next += 1
        return value


class TickClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 2, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(minutes=1)
        return result


@pytest.fixture(scope="module")
def qt_application():
    application = QCoreApplication.instance()
    owns_application = application is None
    if application is None:
        application = QCoreApplication([])
    yield application
    if owns_application:
        application.quit()


@pytest.fixture
def qsql_database(tmp_path: Path, qt_application):
    database_path = tmp_path / "meeting-speaker.sqlite"
    schema = Path("buzz/schema.sql").read_text()
    connection = sqlite3.connect(database_path)
    connection.executescript(schema)
    connection.close()
    name = f"meeting-speaker-{uuid.uuid4()}"
    database = QSqlDatabase.addDatabase("QSQLITE", name)
    database.setDatabaseName(str(database_path))
    assert database.open()
    _exec(database, "PRAGMA foreign_keys = ON")
    yield database, database_path
    database.close()
    del database
    gc.collect()
    QSqlDatabase.removeDatabase(name)


def _exec(database: QSqlDatabase, sql: str, values: tuple = ()) -> QSqlQuery:
    query = QSqlQuery(database)
    assert query.prepare(sql), query.lastError().text()
    for value in values:
        query.addBindValue(value)
    assert query.exec(), query.lastError().text()
    return query


def _scalar(database: QSqlDatabase, sql: str, values: tuple = ()):
    query = _exec(database, sql, values)
    assert query.next()
    return query.value(0)


def _word(role: MeetingTrackRole, ordinal: int, start_ms: int) -> MeetingTranscriptWord:
    return MeetingTranscriptWord(
        source_role=role,
        source_segment_ordinal=0,
        source_word_ordinal=ordinal,
        local_start_ms=start_ms,
        local_end_ms=start_ms + 50,
        start_ns=start_ms * 1_000_000,
        end_ns=(start_ms + 50) * 1_000_000,
        text=f"{role.name}-{ordinal}",
    )


def _seed_source(database: QSqlDatabase, *, generation_int: int = 2100) -> Source:
    meeting_id = str(uuid.UUID(int=generation_int + 1))
    generation_id = uuid.UUID(int=generation_int)
    _exec(
        database,
        """
        INSERT INTO meeting (
            id, remote_source_kind, session_state, created_at,
            started_at, ended_at, duration_ns, audio_state, audio_outcome
        ) VALUES (?, 'SYSTEM', 'COMPLETED', ?, ?, ?, 1000000000,
                  'STOPPED', 'COMPLETE')
        """,
        (
            meeting_id,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
        ),
    )
    _exec(
        database,
        """
        INSERT INTO meeting_final_transcription (
            id, meeting_id, profile_version, status, config_model_type,
            config_whisper_model_size, config_hugging_face_model_id,
            config_language, error_message, time_created, time_started,
            time_completed
        ) VALUES (?, ?, 2, 'COMPLETED', 'FASTER_WHISPER', 'LARGE', '',
                  NULL, NULL, ?, ?, ?)
        """,
        (
            str(generation_id),
            meeting_id,
            "2026-01-01T00:00:03+00:00",
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
        ),
    )
    words = (_word(MIC, 0, 0), _word(REMOTE, 0, 30), _word(MIC, 1, 100))
    for role, word_count in (("MICROPHONE", 2), ("REMOTE", 1)):
        _exec(
            database,
            """
            INSERT INTO meeting_final_transcription_track (
                generation_id, role, status, error_message, time_started,
                time_completed, segment_count, word_count
            ) VALUES (?, ?, 'COMPLETED', NULL, ?, ?, 1, ?)
            """,
            (
                str(generation_id),
                role,
                "2026-01-01T00:00:04+00:00",
                "2026-01-01T00:00:05+00:00",
                word_count,
            ),
        )
        _exec(
            database,
            """
            INSERT INTO meeting_final_transcription_segment (
                generation_id, role, ordinal, local_start_ms, local_end_ms,
                start_ns, end_ns, text
            ) VALUES (?, ?, 0, 0, 1000, 0, 1000000000, 'segment')
            """,
            (str(generation_id), role),
        )
    for word in words:
        _exec(
            database,
            """
            INSERT INTO meeting_final_transcription_word (
                generation_id, role, ordinal, segment_ordinal,
                local_start_ms, local_end_ms, start_ns, end_ns, text
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (
                str(generation_id),
                word.source_role.name,
                word.source_word_ordinal,
                word.local_start_ms,
                word.local_end_ms,
                word.start_ns,
                word.end_ns,
                word.text,
            ),
        )
    generation = FinalTranscriptionGeneration(
        generation_id=generation_id,
        meeting_id=uuid.UUID(meeting_id),
        profile_version=2,
        status=FinalTranscriptionStatus.COMPLETED,
        config=FinalTranscriptionConfig(profile_version=2, whisper_model_size="LARGE"),
        tracks=(
            FinalTranscriptionTrack(
                MIC, FinalTranscriptionTrackStatus.COMPLETED, segment_count=1
            ),
            FinalTranscriptionTrack(
                REMOTE, FinalTranscriptionTrackStatus.COMPLETED, segment_count=1
            ),
        ),
    )
    return Source(generation, words)


def _inputs(source: Source):
    analyses = (
        SpeakerReviewTrackAnalysis(
            MIC,
            (
                SpeakerDiarizationTurn(9, 100, 180),
                SpeakerDiarizationTurn(2, 0, 90),
            ),
            SpeakerDiarizationBackend.MSDD,
            1,
        ),
        SpeakerReviewTrackAnalysis(
            REMOTE,
            (SpeakerDiarizationTurn(4, 0, 100),),
            SpeakerDiarizationBackend.SORTFORMER,
            2,
        ),
    )
    attributed = (
        SpeakerAttributedWord(
            source.words[0],
            MeetingSpeakerKey(MIC, 2),
            SpeakerAttributionStatus.ASSIGNED,
        ),
        SpeakerAttributedWord(
            source.words[1],
            MeetingSpeakerKey(REMOTE, 4),
            SpeakerAttributionStatus.ASSIGNED,
        ),
        SpeakerAttributedWord(
            source.words[2], None, SpeakerAttributionStatus.AMBIGUOUS
        ),
    )
    return analyses, attributed


def _created(qsql_database):
    database, _ = qsql_database
    source = _seed_source(database)
    repository = QSqlMeetingSpeakerRepository(database)
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    analyses, attributed = _inputs(source)
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    return database, source, repository, service, review


def _foreign_keys_off(database: QSqlDatabase) -> None:
    _exec(database, "PRAGMA foreign_keys = OFF")
    assert _scalar(database, "PRAGMA foreign_keys") == 0


def _ignore_checks(database: QSqlDatabase) -> None:
    _exec(database, "PRAGMA ignore_check_constraints = ON")


def test_complete_round_trip_preserves_per_track_provenance_and_exact_machine_rows(
    qsql_database,
):
    database, source, repository, service, review = _created(qsql_database)
    loaded = service.load_review(review.id)
    assert loaded == review
    assert service.load_review_for_generation(source.generation.generation_id) == review
    assert [
        (
            track.source_role,
            track.diarization_backend,
            track.diarization_profile_version,
        )
        for track in review.tracks
    ] == [
        (MIC, SpeakerDiarizationBackend.MSDD, 1),
        (REMOTE, SpeakerDiarizationBackend.SORTFORMER, 2),
    ]
    assert [
        (turn.source_role, turn.ordinal, turn.speaker_index) for turn in review.turns
    ] == [
        (MIC, 0, 9),
        (MIC, 1, 2),
        (REMOTE, 0, 4),
    ]
    assert [cluster.machine_speaker for cluster in review.clusters] == [
        MeetingSpeakerKey(MIC, 2),
        MeetingSpeakerKey(MIC, 9),
        MeetingSpeakerKey(REMOTE, 4),
    ]
    assert [speaker.ordinal for speaker in review.speakers] == [0, 1, 2]
    assert review.source_track_count == 2
    assert repository.load_review(str(review.id)).header.source_track_count == 2
    assert len(repository.load_review(str(review.id)).attributions) == len(source.words)
    assert _scalar(database, "SELECT COUNT(*) FROM meeting_speaker_word_override") == 0
    assert _scalar(database, "SELECT COUNT(*) FROM meeting_speaker_review") == 1


def test_not_provided_and_completed_zero_turn_provenance(qsql_database):
    database, _ = qsql_database
    source = _seed_source(database)
    repository = QSqlMeetingSpeakerRepository(database)
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    analyses = (SpeakerReviewTrackAnalysis(MIC, (), SpeakerDiarizationBackend.MSDD, 1),)
    attributed = tuple(
        SpeakerAttributedWord(word, None, SpeakerAttributionStatus.NO_OVERLAP)
        for word in source.words
    )
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    assert review.tracks[0].analysis_state is SpeakerReviewAnalysisState.COMPLETED
    assert review.tracks[0].turn_count == 0
    assert review.tracks[0].diarization_backend is SpeakerDiarizationBackend.MSDD
    assert review.tracks[1].analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED
    assert review.tracks[1].diarization_backend is None
    assert review.tracks[1].diarization_profile_version is None


def test_repository_create_conflict(qsql_database):
    _, _, repository, _, review = _created(qsql_database)
    bundle = repository.load_review(str(review.id))
    assert bundle is not None
    with pytest.raises(SpeakerReviewConflictError):
        repository.create_review(bundle)


def test_atomic_mutations_and_nullable_override_round_trip(qsql_database):
    _, _, repository, service, review = _created(qsql_database)
    original_machine = repository.load_review(str(review.id))
    review = service.rename_speaker(review.id, review.speakers[0].id, "  Alice  ")
    assert review.speakers[0].display_name == "Alice"
    review = service.create_speaker(review.id, "Manual")
    manual = review.speakers[-1]
    assert manual.ordinal == 3
    review = service.assign_word(review.id, MIC, 1, manual.id)
    assigned = next(
        word
        for word in review.words
        if word.word.source_role is MIC and word.word.source_word_ordinal == 1
    )
    assert assigned.overridden and assigned.effective_speaker_id == manual.id
    review = service.unassign_word(review.id, REMOTE, 0)
    unassigned = next(word for word in review.words if word.word.source_role is REMOTE)
    assert unassigned.overridden and unassigned.effective_speaker_id is None
    raw = repository.load_review(str(review.id))
    null_override = next(
        row for row in raw.overrides if row.role == "REMOTE" and row.word_ordinal == 0
    )
    assert null_override.reviewed_speaker_id is None
    review = service.clear_word_override(review.id, REMOTE, 0)
    restored = next(word for word in review.words if word.word.source_role is REMOTE)
    assert not restored.overridden
    review = service.merge_speakers(
        review.id, review.speakers[0].id, review.speakers[1].id
    )
    review = service.mark_completed(review.id)
    assert review.status is SpeakerReviewStatus.COMPLETED
    assert review.revision == 7
    after_machine = repository.load_review(str(review.id))
    assert after_machine.turns == original_machine.turns
    assert after_machine.clusters == original_machine.clusters
    assert after_machine.attributions == original_machine.attributions


def test_create_failure_rolls_back_every_table(qsql_database):
    database, _ = qsql_database
    source = _seed_source(database)
    repository = QSqlMeetingSpeakerRepository(database)
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    analyses, attributed = _inputs(source)
    _exec(
        database,
        """
        CREATE TRIGGER fail_second_turn BEFORE INSERT ON meeting_speaker_turn
        WHEN NEW.ordinal = 1 BEGIN SELECT RAISE(FAIL, 'injected create failure'); END
        """,
    )
    with pytest.raises(SpeakerReviewError):
        service.create_review(source.generation.generation_id, analyses, attributed)
    for table in SPEAKER_TABLES:
        assert _scalar(database, f"SELECT COUNT(*) FROM {table}") == 0


def test_rename_failure_preserves_name_and_header(qsql_database):
    database, _, repository, service, review = _created(qsql_database)
    before = repository.load_review(str(review.id))
    _exec(
        database,
        """
        CREATE TRIGGER fail_header_update BEFORE UPDATE OF revision
        ON meeting_speaker_review
        WHEN NEW.revision > OLD.revision
        BEGIN SELECT RAISE(FAIL, 'injected header failure'); END
        """,
    )
    with pytest.raises(SpeakerReviewError):
        service.rename_speaker(review.id, review.speakers[0].id, "Changed")
    assert repository.load_review(str(review.id)) == before


def test_override_post_upsert_header_failure_rolls_back_new_row_and_header(
    qsql_database,
):
    database, _, repository, service, review = _created(qsql_database)
    before = repository.load_review(str(review.id))
    assert before is not None
    assert before.overrides == ()
    # Reaching this header trigger proves the preceding override upsert completed.
    _exec(
        database,
        f"""
        CREATE TRIGGER fail_override_header_update
        BEFORE UPDATE OF revision ON meeting_speaker_review
        WHEN OLD.id = '{review.id}' AND NEW.revision > OLD.revision
        BEGIN
            SELECT RAISE(FAIL, 'injected post-upsert header failure');
        END
        """,
    )
    with pytest.raises(SpeakerReviewError, match="injected post-upsert header failure"):
        service.assign_word(review.id, MIC, 1, review.speakers[0].id)

    after = repository.load_review(str(review.id))
    assert after is not None
    assert after.overrides == before.overrides
    assert (
        after.header.status,
        after.header.revision,
        after.header.time_updated,
        after.header.time_completed,
    ) == (
        before.header.status,
        before.header.revision,
        before.header.time_updated,
        before.header.time_completed,
    )
    assert after == before


def test_constant_clock_actual_rename_round_trips_with_unchanged_timestamp(
    qsql_database,
):
    database, _ = qsql_database
    source = _seed_source(database)
    repository = QSqlMeetingSpeakerRepository(database)
    instant = datetime(2026, 2, 1, tzinfo=UTC)
    service = MeetingSpeakerReviewService(
        repository,
        source,
        id_factory=FixedIds(),
        clock=lambda: instant,
    )
    analyses, attributed = _inputs(source)
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )

    renamed = service.rename_speaker(review.id, review.speakers[0].id, "Alice")

    assert renamed.revision == 1
    assert renamed.status is SpeakerReviewStatus.IN_PROGRESS
    assert renamed.time_created == renamed.time_updated == instant
    assert service.load_review(review.id) == renamed


def test_merge_failure_after_remap_rolls_back_exact_state(qsql_database):
    database, _, repository, service, review = _created(qsql_database)
    review = service.assign_word(review.id, MIC, 1, review.speakers[0].id)
    before = repository.load_review(str(review.id))
    _exec(
        database,
        """
        CREATE TRIGGER fail_speaker_delete BEFORE DELETE ON meeting_reviewed_speaker
        BEGIN SELECT RAISE(FAIL, 'injected merge failure'); END
        """,
    )
    with pytest.raises(SpeakerReviewError):
        service.merge_speakers(review.id, review.speakers[0].id, review.speakers[1].id)
    assert repository.load_review(str(review.id)) == before


def test_reset_and_source_generation_cascade_remove_entire_aggregate(qsql_database):
    database, source, _, service, review = _created(qsql_database)
    service.unassign_word(review.id, MIC, 0)
    service.reset_review(source.generation.generation_id)
    for table in SPEAKER_TABLES:
        assert _scalar(database, f"SELECT COUNT(*) FROM {table}") == 0
    service.reset_review(source.generation.generation_id)

    analyses, attributed = _inputs(source)
    review = service.create_review(
        source.generation.generation_id, analyses, attributed
    )
    _exec(
        database,
        "DELETE FROM meeting_final_transcription WHERE id = ?",
        (str(source.generation.generation_id),),
    )
    for table in SPEAKER_TABLES:
        assert _scalar(database, f"SELECT COUNT(*) FROM {table}") == 0
    assert service.load_review(review.id) is None


def test_meeting_delete_cascades_generation_and_review(qsql_database):
    database, source, _, service, _ = _created(qsql_database)
    service.unassign_word(
        service.load_review_for_generation(source.generation.generation_id).id, MIC, 0
    )
    _exec(
        database,
        "DELETE FROM meeting WHERE id = ?",
        (str(source.generation.meeting_id),),
    )
    assert _scalar(database, "SELECT COUNT(*) FROM meeting_final_transcription") == 0
    for table in SPEAKER_TABLES:
        assert _scalar(database, f"SELECT COUNT(*) FROM {table}") == 0


def test_foreign_key_check_clean_after_merge_and_nullable_override(qsql_database):
    database, _, _, service, review = _created(qsql_database)
    review = service.unassign_word(review.id, MIC, 0)
    service.merge_speakers(review.id, review.speakers[0].id, review.speakers[1].id)
    query = _exec(database, "PRAGMA foreign_key_check")
    assert not query.next()


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE meeting_speaker_review SET revision = 1",
        "UPDATE meeting_speaker_review SET revision = 5",
        "UPDATE meeting_speaker_review SET status = 'IN_PROGRESS', revision = 0",
        "UPDATE meeting_speaker_review SET status = 'COMPLETED', revision = 0, time_completed = time_updated",
        "UPDATE meeting_speaker_review SET status = 'IN_PROGRESS', revision = 1, time_updated = '2026-01-31T23:59:59+00:00'",
        "UPDATE meeting_speaker_review SET time_updated = '2026-02-01T00:01:00+00:00'",
        "UPDATE meeting_speaker_review SET time_completed = time_updated",
        "UPDATE meeting_speaker_review SET status = 'IN_PROGRESS', revision = 1, time_completed = time_updated",
        "UPDATE meeting_speaker_review SET status = 'COMPLETED', revision = 1, time_completed = NULL",
        "UPDATE meeting_speaker_review SET status = 'COMPLETED', revision = 1, time_completed = '2026-02-01T00:01:00+00:00'",
    ],
)
def test_schema_rejects_lifecycle_provenance_corruption(qsql_database, mutation):
    database, _, repository, _, review = _created(qsql_database)
    before = repository.load_review(str(review.id))
    query = QSqlQuery(database)
    assert query.prepare(mutation), query.lastError().text()

    assert not query.exec()
    assert repository.load_review(str(review.id)) == before


@pytest.mark.parametrize("source_track_count", [1, 3])
def test_public_load_rejects_frozen_source_track_count_mismatch(
    qsql_database, source_track_count
):
    database, _, _, service, review = _created(qsql_database)
    _exec(
        database,
        "UPDATE meeting_speaker_review SET source_track_count = ?",
        (source_track_count,),
    )

    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


def test_missing_zero_child_persisted_track_is_decode_error_before_stale_check(
    qsql_database,
):
    database, _ = qsql_database
    source = _seed_source(database)
    _exec(
        database,
        "DELETE FROM meeting_final_transcription_word WHERE generation_id = ? AND role = 'REMOTE'",
        (str(source.generation.generation_id),),
    )
    _exec(
        database,
        "DELETE FROM meeting_final_transcription_segment WHERE generation_id = ? AND role = 'REMOTE'",
        (str(source.generation.generation_id),),
    )
    _exec(
        database,
        """
        UPDATE meeting_final_transcription_track
        SET status = 'FAILED', segment_count = 0, word_count = 0
        WHERE generation_id = ? AND role = 'REMOTE'
        """,
        (str(source.generation.generation_id),),
    )
    _exec(
        database,
        "UPDATE meeting_final_transcription SET status = 'PARTIAL' WHERE id = ?",
        (str(source.generation.generation_id),),
    )
    source.words = tuple(word for word in source.words if word.source_role is MIC)
    source.generation = replace(
        source.generation,
        status=FinalTranscriptionStatus.PARTIAL,
        tracks=(
            source.generation.tracks[0],
            replace(
                source.generation.tracks[1],
                status=FinalTranscriptionTrackStatus.FAILED,
            ),
        ),
    )
    analysis = SpeakerReviewTrackAnalysis(
        MIC,
        (SpeakerDiarizationTurn(2, 0, 200),),
        SpeakerDiarizationBackend.MSDD,
        1,
    )
    attributed = tuple(
        SpeakerAttributedWord(
            word,
            MeetingSpeakerKey(MIC, 2),
            SpeakerAttributionStatus.ASSIGNED,
        )
        for word in source.words
    )
    repository = QSqlMeetingSpeakerRepository(database)
    service = MeetingSpeakerReviewService(
        repository, source, id_factory=FixedIds(), clock=TickClock()
    )
    review = service.create_review(
        source.generation.generation_id, (analysis,), attributed
    )
    assert service.load_review(review.id) == review
    remote = review.tracks[1]
    assert remote.source_role is REMOTE
    assert remote.source_word_count == 0
    assert remote.analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED
    assert remote.turn_count == 0

    _foreign_keys_off(database)
    _exec(
        database,
        "DELETE FROM meeting_speaker_review_track WHERE review_id = ? AND role = 'REMOTE'",
        (str(review.id),),
    )

    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE meeting_speaker_review SET id = 'BAD'",
        "UPDATE meeting_reviewed_speaker SET id = 'BAD' WHERE ordinal = 0",
        "UPDATE meeting_speaker_review SET status = 'UNKNOWN'",
        "UPDATE meeting_speaker_review_track SET analysis_state = 'UNKNOWN' WHERE role = 'REMOTE'",
        "UPDATE meeting_speaker_word_attribution SET attribution_status = 'UNKNOWN' WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "UPDATE meeting_speaker_review_track SET source_generation_id = '00000000-0000-0000-0000-000000009999' WHERE role = 'MICROPHONE'",
        "DELETE FROM meeting_speaker_review_track WHERE role = 'MICROPHONE'",
        "DELETE FROM meeting_speaker_turn WHERE role = 'MICROPHONE' AND ordinal = 0",
        "DELETE FROM meeting_speaker_turn WHERE role = 'REMOTE' AND ordinal = 0",
        "UPDATE meeting_speaker_turn SET ordinal = 99 WHERE role = 'MICROPHONE' AND ordinal = 0",
        "DELETE FROM meeting_speaker_word_attribution WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "DELETE FROM meeting_speaker_word_attribution WHERE role = 'MICROPHONE' AND word_ordinal = 1",
        "DELETE FROM meeting_speaker_cluster WHERE role = 'MICROPHONE' AND speaker_index = 2",
        "DELETE FROM meeting_speaker_cluster_assignment WHERE role = 'MICROPHONE' AND speaker_index = 2",
        "UPDATE meeting_speaker_word_attribution SET machine_speaker_index = NULL WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "UPDATE meeting_speaker_word_attribution SET machine_speaker_index = 999 WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "UPDATE meeting_speaker_word_attribution SET attribution_status = 'NO_OVERLAP', machine_speaker_index = 2 WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "UPDATE meeting_speaker_word_attribution SET attribution_status = 'AMBIGUOUS', machine_speaker_index = 2 WHERE role = 'MICROPHONE' AND word_ordinal = 0",
        "UPDATE meeting_reviewed_speaker SET display_name = ' bad ' WHERE ordinal = 0",
        "UPDATE meeting_speaker_review SET next_speaker_ordinal = 0",
        "UPDATE meeting_speaker_review SET status = 'COMPLETED', time_completed = NULL",
        "UPDATE meeting_speaker_review SET revision = 5",
        "UPDATE meeting_speaker_review SET status = 'IN_PROGRESS', revision = 0",
        "UPDATE meeting_speaker_review SET status = 'IN_PROGRESS', revision = 1, time_updated = '2026-01-31T23:59:59+00:00'",
        "UPDATE meeting_speaker_review SET status = 'COMPLETED', revision = 1, time_completed = '2026-02-01T00:01:00+00:00'",
    ],
)
def test_public_service_load_detects_raw_table_corruption(qsql_database, mutation):
    database, _, _, service, review = _created(qsql_database)
    _foreign_keys_off(database)
    _ignore_checks(database)
    _exec(database, mutation)
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review_for_generation(review.source_generation_id)


@pytest.mark.parametrize(
    "insert",
    [
        """
        INSERT INTO meeting_speaker_turn
        SELECT id, 'MICROPHONE', 99, 2, 0, 1
        FROM meeting_speaker_review LIMIT 1
        """,
        """
        INSERT INTO meeting_speaker_cluster
        SELECT id, 'MICROPHONE', 999 FROM meeting_speaker_review LIMIT 1
        """,
        """
        INSERT INTO meeting_speaker_word_attribution
        SELECT id, source_generation_id, 'MICROPHONE', 999, 'NO_OVERLAP', NULL
        FROM meeting_speaker_review LIMIT 1
        """,
        """
        INSERT INTO meeting_speaker_word_override
        SELECT id, 'MICROPHONE', 999, NULL FROM meeting_speaker_review LIMIT 1
        """,
    ],
)
def test_detects_extra_turn_attribution_cluster_and_override_word(
    qsql_database, insert
):
    database, _, _, service, review = _created(qsql_database)
    _foreign_keys_off(database)
    _exec(database, insert)
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)


@pytest.mark.parametrize("reference_kind", ["assignment", "override"])
def test_detects_cross_review_assignment_and_override(qsql_database, reference_kind):
    database, _, _, service, review = _created(qsql_database)
    other_source = _seed_source(database, generation_int=2200)
    other_repository = QSqlMeetingSpeakerRepository(database)
    other_service = MeetingSpeakerReviewService(
        other_repository, other_source, id_factory=FixedIds(4000), clock=TickClock()
    )
    analyses, attributed = _inputs(other_source)
    other = other_service.create_review(
        other_source.generation.generation_id, analyses, attributed
    )
    review = service.unassign_word(review.id, MIC, 1)

    _foreign_keys_off(database)
    if reference_kind == "assignment":
        _exec(
            database,
            """
            UPDATE meeting_speaker_cluster_assignment
            SET reviewed_speaker_id = ?
            WHERE review_id = ? AND role = 'MICROPHONE' AND speaker_index = 2
            """,
            (str(other.speakers[0].id), str(review.id)),
        )
    else:
        _exec(
            database,
            """
            UPDATE meeting_speaker_word_override
            SET reviewed_speaker_id = ?
            WHERE review_id = ? AND role = 'MICROPHONE' AND word_ordinal = 1
            """,
            (str(other.speakers[0].id), str(review.id)),
        )
    with pytest.raises(SpeakerReviewDecodeError):
        service.load_review(review.id)
