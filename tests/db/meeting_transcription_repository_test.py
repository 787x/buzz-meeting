"""Tests for the QSql meeting transcription repository."""

from __future__ import annotations

import gc
import sqlite3
import uuid
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.db.meeting_storage_repository import QSqlMeetingRepository
from buzz.db.meeting_transcription_repository import (
    QSqlMeetingTranscriptionRepository,
)
from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionDecodeError,
    FinalTranscriptionService,
    FinalTranscriptionStatus,
    FinalTranscriptionTrackStatus,
    SegmentPersistenceRecord,
    TrackPersistenceRecord,
    WordPersistenceRecord,
    encode_generation_status,
    encode_track_status,
)
from buzz.meeting.meeting_storage import MeetingPersistenceBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_meeting_bundle(
    meeting_id: str | None = None,
) -> MeetingPersistenceBundle:
    mid = meeting_id or str(uuid.uuid4())
    return MeetingPersistenceBundle(
        session_id=mid,
        remote_source_kind="SYSTEM",
        session_state="COMPLETED",
        created_at="2025-01-01T00:00:00+00:00",
        started_at="2025-01-01T00:01:00+00:00",
        ended_at="2025-01-01T00:02:00+00:00",
        duration_ns=60_000_000_000,
        audio_state="STOPPED",
        audio_outcome="COMPLETE",
        tracks=(
            __import__(
                "buzz.meeting.meeting_storage",
                fromlist=["MeetingTrackPersistenceRecord"],
            ).MeetingTrackPersistenceRecord(
                role="MICROPHONE",
                relative_path=f"{mid}/microphone.wav",
                sample_rate=16000,
                sample_count=160000,
                recording_state="STOPPED",
                published=1,
                complete=1,
                timing_basis="host_callback_arrival",
            ),
            __import__(
                "buzz.meeting.meeting_storage",
                fromlist=["MeetingTrackPersistenceRecord"],
            ).MeetingTrackPersistenceRecord(
                role="REMOTE",
                relative_path=f"{mid}/remote.wav",
                sample_rate=16000,
                sample_count=160000,
                recording_state="STOPPED",
                published=1,
                complete=1,
                timing_basis="host_callback_arrival",
            ),
        ),
        timings=(
            __import__(
                "buzz.meeting.meeting_storage",
                fromlist=["MeetingTimingPersistenceRecord"],
            ).MeetingTimingPersistenceRecord("MICROPHONE", 0, 16000, 0),
            __import__(
                "buzz.meeting.meeting_storage",
                fromlist=["MeetingTimingPersistenceRecord"],
            ).MeetingTimingPersistenceRecord("REMOTE", 0, 16000, 100_000_000),
        ),
        errors=(),
    )


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
    database_path = tmp_path / "meeting_transcription.sqlite"
    schema = Path("buzz/schema.sql").read_text()
    connection = sqlite3.connect(database_path)
    connection.executescript(schema)
    connection.close()
    name = f"meeting-transcription-{uuid.uuid4()}"
    database = QSqlDatabase.addDatabase("QSQLITE", name)
    database.setDatabaseName(str(database_path))
    assert database.open()
    pragma = QSqlQuery(database)
    assert pragma.exec("PRAGMA foreign_keys = ON")
    pragma.finish()
    del pragma
    yield database, database_path
    database.close()
    del database
    gc.collect()
    QSqlDatabase.removeDatabase(name)


@pytest.fixture
def repo(qsql_database):
    database, _ = qsql_database
    return QSqlMeetingTranscriptionRepository(database)


@pytest.fixture
def meeting_repo(qsql_database):
    database, _ = qsql_database
    return QSqlMeetingRepository(database)


def _insert_meeting(meeting_repo, meeting_id: str) -> None:
    bundle = _make_meeting_bundle(meeting_id)
    meeting_repo.atomic_replace(bundle, validate_existing=lambda e: None)


def _segment(gen_id: str, role: str, text: str = "phrase") -> SegmentPersistenceRecord:
    return SegmentPersistenceRecord(
        generation_id=gen_id,
        role=role,
        ordinal=0,
        local_start_ms=0,
        local_end_ms=1000,
        start_ns=-1_000_000_000,
        end_ns=0,
        text=text,
    )


def _word(gen_id: str, role: str, text: str = "word") -> WordPersistenceRecord:
    return WordPersistenceRecord(
        generation_id=gen_id,
        role=role,
        ordinal=0,
        segment_ordinal=0,
        local_start_ms=100,
        local_end_ms=500,
        start_ns=-900_000_000,
        end_ns=-500_000_000,
        text=text,
    )


# ---------------------------------------------------------------------------
# Generation CRUD
# ---------------------------------------------------------------------------


class TestCreateGeneration:
    def test_creates_generation_and_tracks(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)

        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )

        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )

        gen = repo.load_generation(gen_id)
        assert gen is not None
        assert gen.id == gen_id
        assert gen.meeting_id == meeting_id
        assert gen.profile_version == 1
        assert gen.status == encode_generation_status(FinalTranscriptionStatus.QUEUED)
        assert gen.config_model_type == "FASTER_WHISPER"
        assert gen.config_whisper_model_size == "TINY"

        loaded_tracks = repo.load_tracks(gen_id)
        assert len(loaded_tracks) == 2


class TestFindGenerationByKey:
    def test_finds_existing(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="REMOTE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )
        found = repo.find_generation_by_key(meeting_id, 1)
        assert found is not None
        assert found.id == gen_id

    def test_returns_none_for_missing(self, repo) -> None:
        assert repo.find_generation_by_key(str(uuid.uuid4()), 1) is None


# ---------------------------------------------------------------------------
# Track lifecycle
# ---------------------------------------------------------------------------


class TestTrackLifecycle:
    def _create_generation(self, repo, meeting_repo) -> tuple[str, str]:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )
        return gen_id, meeting_id

    def test_begin_track(self, repo, meeting_repo) -> None:
        gen_id, _ = self._create_generation(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")

        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.IN_PROGRESS
        )
        assert mic.time_started == "2025-01-01T00:03:00+00:00"

        gen = repo.load_generation(gen_id)
        assert gen is not None
        assert gen.status == encode_generation_status(
            FinalTranscriptionStatus.IN_PROGRESS
        )

    def test_complete_track_with_segments(self, repo, meeting_repo) -> None:
        gen_id, _ = self._create_generation(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")

        segments = (
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                local_start_ms=0,
                local_end_ms=1000,
                start_ns=0,
                end_ns=1_000_000_000,
                text="hello",
            ),
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=1,
                local_start_ms=1100,
                local_end_ms=2000,
                start_ns=1_100_000_000,
                end_ns=2_000_000_000,
                text="world",
            ),
        )
        repo.complete_track(gen_id, "MICROPHONE", segments, "2025-01-01T00:04:00+00:00")

        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.COMPLETED
        )
        assert mic.segment_count == 2
        assert mic.time_completed == "2025-01-01T00:04:00+00:00"

        loaded_segs = repo.load_segments(gen_id, "MICROPHONE")
        assert len(loaded_segs) == 2
        assert loaded_segs[0].text == "hello"
        assert loaded_segs[1].text == "world"

    def test_fail_track(self, repo, meeting_repo) -> None:
        gen_id, _ = self._create_generation(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")
        repo.fail_track(
            gen_id,
            "MICROPHONE",
            "ASR failed",
            "2025-01-01T00:04:00+00:00",
        )

        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        assert mic.status == encode_track_status(FinalTranscriptionTrackStatus.FAILED)
        assert mic.error_message == "ASR failed"
        assert mic.word_count == 0

    def test_complete_track_atomicity(self, repo, meeting_repo) -> None:
        """Segments + track status + generation status are atomic."""
        gen_id, _ = self._create_generation(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")
        repo.begin_track(gen_id, "REMOTE", "2025-01-01T00:03:00+00:00")

        segs = (
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                local_start_ms=0,
                local_end_ms=1000,
                start_ns=0,
                end_ns=1_000_000_000,
                text="hi",
            ),
        )
        repo.complete_track(gen_id, "MICROPHONE", segs, "2025-01-01T00:04:00+00:00")

        # Generation still IN_PROGRESS (REMOTE not done)
        gen = repo.load_generation(gen_id)
        assert gen is not None
        assert gen.status == encode_generation_status(
            FinalTranscriptionStatus.IN_PROGRESS
        )

        # Complete REMOTE
        repo.complete_track(gen_id, "REMOTE", (), "2025-01-01T00:05:00+00:00")
        gen = repo.load_generation(gen_id)
        assert gen is not None
        assert gen.status == encode_generation_status(
            FinalTranscriptionStatus.COMPLETED
        )
        assert gen.time_completed == "2025-01-01T00:05:00+00:00"


class TestV2AtomicPhraseWordCompletion:
    @staticmethod
    def _create_v2(repo, meeting_repo) -> str:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        tracks = tuple(
            TrackPersistenceRecord(
                generation_id=gen_id,
                role=role,
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            )
            for role in ("MICROPHONE", "REMOTE")
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            FinalTranscriptionConfig(
                profile_version=2,
                model_type="FASTER_WHISPER",
                whisper_model_size="SMALL",
            ),
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )
        return gen_id

    def test_success_persists_phrase_words_and_status_atomically(
        self, repo, meeting_repo
    ) -> None:
        gen_id = self._create_v2(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:01:00+00:00")

        repo.complete_track(
            gen_id,
            "MICROPHONE",
            (_segment(gen_id, "MICROPHONE"),),
            "2025-01-01T00:02:00+00:00",
            (_word(gen_id, "MICROPHONE"),),
        )

        assert len(repo.load_segments(gen_id, "MICROPHONE")) == 1
        assert repo.load_words(gen_id) == (_word(gen_id, "MICROPHONE"),)
        mic = next(
            track for track in repo.load_tracks(gen_id) if track.role == "MICROPHONE"
        )
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.COMPLETED
        )
        assert mic.segment_count == 1
        assert mic.word_count == 1

    @pytest.mark.parametrize(
        "table,trigger_name",
        [
            ("meeting_final_transcription_segment", "fail_phrase_insert"),
            ("meeting_final_transcription_word", "fail_word_insert"),
        ],
    )
    def test_insert_failure_rolls_back_everything(
        self,
        repo,
        meeting_repo,
        qsql_database,
        table: str,
        trigger_name: str,
    ) -> None:
        database, _ = qsql_database
        gen_id = self._create_v2(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:01:00+00:00")
        before_generation = repo.load_generation(gen_id)
        trigger = QSqlQuery(database)
        assert trigger.exec(
            f"CREATE TRIGGER {trigger_name} BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected insert failure'); END"
        )
        trigger.finish()

        with pytest.raises(Exception, match="complete_track failed"):
            repo.complete_track(
                gen_id,
                "MICROPHONE",
                (_segment(gen_id, "MICROPHONE"),),
                "2025-01-01T00:02:00+00:00",
                (_word(gen_id, "MICROPHONE"),),
            )

        assert repo.load_segments(gen_id, "MICROPHONE") == ()
        assert repo.load_words(gen_id) == ()
        mic = next(
            track for track in repo.load_tracks(gen_id) if track.role == "MICROPHONE"
        )
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.IN_PROGRESS
        )
        assert mic.segment_count == 0
        assert repo.load_generation(gen_id) == before_generation

    def test_retry_preserves_completed_words_and_clears_failed_role(
        self, repo, meeting_repo, qsql_database
    ) -> None:
        database, _ = qsql_database
        gen_id = self._create_v2(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:01:00+00:00")
        repo.complete_track(
            gen_id,
            "MICROPHONE",
            (_segment(gen_id, "MICROPHONE", "kept phrase"),),
            "2025-01-01T00:02:00+00:00",
            (_word(gen_id, "MICROPHONE", "kept word"),),
        )
        repo.begin_track(gen_id, "REMOTE", "2025-01-01T00:03:00+00:00")

        stale = QSqlQuery(database)
        stale.prepare(
            "INSERT INTO meeting_final_transcription_segment "
            "(generation_id, role, ordinal, local_start_ms, local_end_ms, "
            "start_ns, end_ns, text) VALUES (?, ?, 0, 0, 1000, 0, 1000, ?)"
        )
        stale.addBindValue(gen_id)
        stale.addBindValue("REMOTE")
        stale.addBindValue("stale phrase")
        assert stale.exec()
        stale.prepare(
            "INSERT INTO meeting_final_transcription_word "
            "(generation_id, role, ordinal, segment_ordinal, local_start_ms, "
            "local_end_ms, start_ns, end_ns, text) "
            "VALUES (?, ?, 0, 0, 0, 1000, 0, 1000, ?)"
        )
        stale.addBindValue(gen_id)
        stale.addBindValue("REMOTE")
        stale.addBindValue("stale word")
        assert stale.exec()
        stale.finish()

        repo.reset_for_retry(
            gen_id,
            {
                "MICROPHONE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                "REMOTE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
            },
            "2025-01-01T00:04:00+00:00",
        )

        assert [word.text for word in repo.load_words(gen_id)] == ["kept word"]
        assert [
            segment.text for segment in repo.load_segments(gen_id, "MICROPHONE")
        ] == ["kept phrase"]
        assert repo.load_segments(gen_id, "REMOTE") == ()

    def test_recovery_reset_clears_stale_phrase_and_words(
        self, repo, meeting_repo, qsql_database
    ) -> None:
        database, _ = qsql_database
        gen_id = self._create_v2(repo, meeting_repo)
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:01:00+00:00")
        stale = QSqlQuery(database)
        assert stale.exec(
            "INSERT INTO meeting_final_transcription_segment VALUES "
            f"('{gen_id}', 'MICROPHONE', 0, 0, 1000, 0, 1000, 'stale')"
        )
        assert stale.exec(
            "INSERT INTO meeting_final_transcription_word VALUES "
            f"('{gen_id}', 'MICROPHONE', 0, 0, 0, 1000, 0, 1000, 'stale')"
        )
        stale.finish()

        repo.reset_in_progress_tracks(gen_id)

        assert repo.load_segments(gen_id, "MICROPHONE") == ()
        assert repo.load_words(gen_id) == ()
        mic = next(
            track for track in repo.load_tracks(gen_id) if track.role == "MICROPHONE"
        )
        assert mic.status == encode_track_status(FinalTranscriptionTrackStatus.QUEUED)


class TestWordCorruptionProbes:
    @staticmethod
    def _create_generation(
        repo,
        meeting_repo,
        *,
        profile_version: int,
    ) -> str:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        repo.create_generation(
            gen_id,
            meeting_id,
            FinalTranscriptionConfig(
                profile_version=profile_version,
                model_type="FASTER_WHISPER",
                whisper_model_size="SMALL",
            ),
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:01:00+00:00")
        repo.complete_track(
            gen_id,
            "MICROPHONE",
            (_segment(gen_id, "MICROPHONE"),),
            "2025-01-01T00:02:00+00:00",
            (_word(gen_id, "MICROPHONE"),) if profile_version == 2 else (),
        )
        return gen_id

    @staticmethod
    def _service(repo) -> FinalTranscriptionService:
        return FinalTranscriptionService(object(), repo, object())  # type: ignore

    def test_v1_word_row_is_detected_with_foreign_keys_off(
        self, repo, meeting_repo, qsql_database
    ) -> None:
        _, database_path = qsql_database
        gen_id = self._create_generation(repo, meeting_repo, profile_version=1)
        raw = sqlite3.connect(database_path)
        raw.execute("PRAGMA foreign_keys = OFF")
        raw.execute(
            "INSERT INTO meeting_final_transcription_word VALUES "
            "(?, 'MICROPHONE', 0, 0, 0, 100, -1000000000, -900000000, 'bad')",
            (gen_id,),
        )
        raw.commit()
        raw.close()

        with pytest.raises(FinalTranscriptionDecodeError, match="version 1"):
            self._service(repo).load_words(uuid.UUID(gen_id))

    @pytest.mark.parametrize(
        "mutation",
        ["unknown_role", "orphan_track", "orphan_parent", "ordinal_gap", "bad_time"],
    )
    def test_v2_corruption_is_not_hidden_by_joins(
        self, repo, meeting_repo, qsql_database, mutation: str
    ) -> None:
        _, database_path = qsql_database
        gen_id = self._create_generation(repo, meeting_repo, profile_version=2)
        raw = sqlite3.connect(database_path)
        raw.execute("PRAGMA foreign_keys = OFF")
        if mutation == "unknown_role":
            raw.execute(
                "INSERT INTO meeting_final_transcription_word VALUES "
                "(?, 'ALIEN', 0, 0, 0, 100, 0, 100, 'bad')",
                (gen_id,),
            )
        elif mutation == "orphan_track":
            raw.execute(
                "INSERT INTO meeting_final_transcription_word VALUES "
                "(?, 'REMOTE', 0, 0, 0, 100, 0, 100, 'bad')",
                (gen_id,),
            )
        elif mutation == "orphan_parent":
            raw.execute(
                "UPDATE meeting_final_transcription_word "
                "SET segment_ordinal = 9 WHERE generation_id = ?",
                (gen_id,),
            )
        elif mutation == "ordinal_gap":
            raw.execute(
                "UPDATE meeting_final_transcription_word "
                "SET ordinal = 2 WHERE generation_id = ?",
                (gen_id,),
            )
        else:
            raw.execute("PRAGMA ignore_check_constraints = ON")
            raw.execute(
                "UPDATE meeting_final_transcription_word "
                "SET local_end_ms = -1, end_ns = start_ns - 1 "
                "WHERE generation_id = ?",
                (gen_id,),
            )
        raw.commit()
        raw.close()

        with pytest.raises(FinalTranscriptionDecodeError):
            self._service(repo).load_words(uuid.UUID(gen_id))

    def test_orphan_generation_word_is_detected(self, repo, qsql_database) -> None:
        _, database_path = qsql_database
        missing = uuid.uuid4()
        raw = sqlite3.connect(database_path)
        raw.execute("PRAGMA foreign_keys = OFF")
        raw.execute(
            "INSERT INTO meeting_final_transcription_word VALUES "
            "(?, 'MICROPHONE', 0, 0, 0, 100, 0, 100, 'bad')",
            (str(missing),),
        )
        raw.commit()
        raw.close()

        with pytest.raises(FinalTranscriptionDecodeError, match="missing generation"):
            self._service(repo).load_words(missing)


# ---------------------------------------------------------------------------
# Empty transcript
# ---------------------------------------------------------------------------


class TestEmptyTranscript:
    def test_completed_with_zero_segments(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.INELIGIBLE),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")
        repo.complete_track(gen_id, "MICROPHONE", (), "2025-01-01T00:04:00+00:00")
        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.COMPLETED
        )
        assert mic.segment_count == 0
        segs = repo.load_segments(gen_id, "MICROPHONE")
        assert len(segs) == 0


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def _create_partially_failed(self, repo, meeting_repo) -> str:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )

        # MIC succeeds
        repo.begin_track(gen_id, "MICROPHONE", "2025-01-01T00:03:00+00:00")
        segs = (
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                local_start_ms=0,
                local_end_ms=1000,
                start_ns=0,
                end_ns=1_000_000_000,
                text="hi",
            ),
        )
        repo.complete_track(gen_id, "MICROPHONE", segs, "2025-01-01T00:04:00+00:00")

        # REMOTE fails
        repo.begin_track(gen_id, "REMOTE", "2025-01-01T00:04:00+00:00")
        repo.fail_track(
            gen_id,
            "REMOTE",
            "ASR error",
            "2025-01-01T00:05:00+00:00",
        )
        return gen_id

    def test_reset_for_retry(self, repo, meeting_repo) -> None:
        gen_id = self._create_partially_failed(repo, meeting_repo)

        desired = {
            "MICROPHONE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
            "REMOTE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
        }
        repo.reset_for_retry(gen_id, desired, "2025-01-01T00:06:00+00:00")
        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        rem = next(t for t in tracks if t.role == "REMOTE")

        # MIC was COMPLETED — stays COMPLETED
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.COMPLETED
        )
        # REMOTE was FAILED — reset to QUEUED (desired)
        assert rem.status == encode_track_status(FinalTranscriptionTrackStatus.QUEUED)
        assert rem.error_message is None

    def test_retry_segments_cleaned(self, repo, meeting_repo) -> None:
        gen_id = self._create_partially_failed(repo, meeting_repo)
        desired = {
            "MICROPHONE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
            "REMOTE": encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
        }
        repo.reset_for_retry(gen_id, desired, "2025-01-01T00:06:00+00:00")
        # MIC segments should still exist
        mic_segs = repo.load_segments(gen_id, "MICROPHONE")
        assert len(mic_segs) == 1
        # REMOTE segments should be gone
        rem_segs = repo.load_segments(gen_id, "REMOTE")
        assert len(rem_segs) == 0


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_load_recoverable(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )
        recoverable = repo.load_recoverable_generations()
        assert len(recoverable) == 1
        assert recoverable[0].id == gen_id

    def test_reset_in_progress_tracks(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.IN_PROGRESS),
                error_message=None,
                time_started="2025-01-01T00:03:00+00:00",
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )
        # Set generation to IN_PROGRESS
        # (simulating a partial run)
        repo._update_generation_status(gen_id, "2025-01-01T00:03:00+00:00")

        repo.reset_in_progress_tracks(gen_id)

        tracks = repo.load_tracks(gen_id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        rem = next(t for t in tracks if t.role == "REMOTE")

        assert mic.status == encode_track_status(FinalTranscriptionTrackStatus.QUEUED)
        assert mic.time_started is None
        assert rem.status == encode_track_status(FinalTranscriptionTrackStatus.QUEUED)


# ---------------------------------------------------------------------------
# Mark track ineligible
# ---------------------------------------------------------------------------


class TestMarkIneligible:
    def test_mark_ineligible(self, repo, meeting_repo) -> None:
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )

        repo.mark_track_ineligible(gen_id, "REMOTE", "2025-01-01T00:03:00+00:00")

        tracks = repo.load_tracks(gen_id)
        rem = next(t for t in tracks if t.role == "REMOTE")
        assert rem.status == encode_track_status(
            FinalTranscriptionTrackStatus.INELIGIBLE
        )


# ---------------------------------------------------------------------------
# Cascading delete
# ---------------------------------------------------------------------------


class TestCascadeDelete:
    def test_delete_meeting_cascades(self, qsql_database, meeting_repo) -> None:
        database, _ = qsql_database
        meeting_id = str(uuid.uuid4())
        _insert_meeting(meeting_repo, meeting_id)

        repo = QSqlMeetingTranscriptionRepository(database)
        gen_id = str(uuid.uuid4())
        config = FinalTranscriptionConfig()
        tracks = (
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
            TrackPersistenceRecord(
                generation_id=gen_id,
                role="REMOTE",
                status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
            ),
        )
        repo.create_generation(
            gen_id,
            meeting_id,
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            tracks,
        )

        # Delete the meeting
        q = QSqlQuery(database)
        assert q.exec(f"DELETE FROM meeting WHERE id = '{meeting_id}'")
        q.finish()

        # Generation should be gone
        assert repo.load_generation(gen_id) is None
        assert repo.load_tracks(gen_id) == ()
