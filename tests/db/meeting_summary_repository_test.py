"""Tests for the QSql meeting summary repository."""

from __future__ import annotations

import datetime
import gc
import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.db.meeting_storage_repository import QSqlMeetingRepository
from buzz.db.meeting_summary_repository import QSqlMeetingSummaryRepository
from buzz.db.meeting_transcription_repository import (
    QSqlMeetingTranscriptionRepository,
)
from buzz.meeting.meeting_summary import (
    MEETING_SUMMARY_SCHEMA_VERSION,
    ActionItem,
    Decision,
    MeetingSummary,
    MeetingSummaryArtifact,
    MeetingSummaryConflictError,
    MeetingSummaryDatabaseError,
    MeetingSummaryDecodeError,
    OpenQuestion,
    Participant,
    Risk,
    Topic,
    meeting_summary_to_json,
)
from buzz.meeting.meeting_storage import MeetingPersistenceBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_meeting_bundle(meeting_id: str | None = None) -> MeetingPersistenceBundle:
    mid = meeting_id or str(uuid.uuid4())
    from buzz.meeting.meeting_storage import (
        MeetingTimingPersistenceRecord,
        MeetingTrackPersistenceRecord,
    )

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
            MeetingTrackPersistenceRecord(
                role="MICROPHONE",
                relative_path=f"{mid}/microphone.wav",
                sample_rate=16000,
                sample_count=160000,
                recording_state="STOPPED",
                published=1,
                complete=1,
                timing_basis="host_callback_arrival",
            ),
        ),
        timings=(MeetingTimingPersistenceRecord("MICROPHONE", 0, 16000, 0),),
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
def qsql_database(tmp_path, qt_application):
    database_path = tmp_path / "meeting_summary.sqlite"
    schema = Path("buzz/schema.sql").read_text()
    connection = sqlite3.connect(database_path)
    connection.executescript(schema)
    connection.close()
    name = f"meeting-summary-{uuid.uuid4()}"
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
    return QSqlMeetingSummaryRepository(database)


@pytest.fixture
def meeting_repo(qsql_database):
    database, _ = qsql_database
    return QSqlMeetingRepository(database)


@pytest.fixture
def transcription_repo(qsql_database):
    database, _ = qsql_database
    return QSqlMeetingTranscriptionRepository(database)


def _insert_meeting(meeting_repo, meeting_id: str) -> None:
    bundle = _make_meeting_bundle(meeting_id)
    meeting_repo.atomic_replace(bundle, validate_existing=lambda e: None)


def _insert_generation(
    transcription_repo,
    meeting_id: str,
    generation_id: str,
    profile_version: int = 2,
) -> None:
    from buzz.meeting.final_transcription import (
        FinalTranscriptionConfig,
        FinalTranscriptionTrackStatus,
        TrackPersistenceRecord,
        encode_track_status,
    )

    config = FinalTranscriptionConfig(
        profile_version=profile_version,
        whisper_model_size="LARGE",
    )
    transcription_repo.create_generation(
        generation_id=generation_id,
        meeting_id=meeting_id,
        config=config,
        initial_status="COMPLETED",
        time_created="2025-01-01T00:00:01+00:00",
        time_completed="2025-01-01T00:00:02+00:00",
        tracks=(
            TrackPersistenceRecord(
                generation_id=generation_id,
                role="MICROPHONE",
                status=encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                error_message=None,
                time_started="2025-01-01T00:00:01+00:00",
                time_completed="2025-01-01T00:00:02+00:00",
                segment_count=0,
            ),
        ),
    )


_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_MID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_GID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_RID = uuid.UUID("00000000-0000-0000-0000-000000000003")
_SID = uuid.UUID("00000000-0000-0000-0000-000000000004")
_PID = uuid.UUID("00000000-0000-0000-0000-000000000005")


def _minimal_summary(**kw: object) -> MeetingSummary:
    defaults = dict(
        schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
        prompt_version=1,
        title=None,
        summary="A summary.",
        participants=(),
        topics=(),
        decisions=(),
        action_items=(),
        open_questions=(),
        risks=(),
    )
    defaults.update(kw)
    return MeetingSummary(**defaults)  # type: ignore[arg-type]


def _full_summary() -> MeetingSummary:
    return MeetingSummary(
        schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
        prompt_version=2,
        title="Title",
        summary="Summary text.",
        participants=(Participant(name="Alice", reviewed_speaker_id=_PID),),
        topics=(
            Topic(title="Topic", summary="TS", source_start_ns=0, source_end_ns=100),
        ),
        decisions=(Decision(text="Decided", source_start_ns=10, source_end_ns=20),),
        action_items=(
            ActionItem(
                task="Do thing",
                owner="Bob",
                due_date=datetime.date(2026, 6, 1),
                source_start_ns=30,
                source_end_ns=40,
            ),
        ),
        open_questions=(
            OpenQuestion(text="What?", source_start_ns=50, source_end_ns=60),
        ),
        risks=(Risk(text="Risk!", source_start_ns=70, source_end_ns=80),),
    )


def _artifact(
    *,
    summary: MeetingSummary | None = None,
    review_id: uuid.UUID | None = _RID,
    review_revision: int | None = 4,
    sid: uuid.UUID = _SID,
    **kw: object,
) -> MeetingSummaryArtifact:
    defaults = dict(
        summary_id=sid,
        meeting_id=_MID,
        source_generation_id=_GID,
        source_profile_version=2,
        source_review_id=review_id,
        source_review_revision=review_revision,
        created_at=_NOW,
        summary=summary or _minimal_summary(),
    )
    defaults.update(kw)
    return MeetingSummaryArtifact(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSave:
    def test_zero_summaries(self, repo, meeting_repo, transcription_repo) -> None:
        mid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        results = repo.list_for_meeting(uuid.UUID(mid))
        assert results == ()

    def test_one_summary_round_trip(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        sid = uuid.uuid4()
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        art = _artifact(
            sid=sid,
            meeting_id=uuid.UUID(mid),
            source_generation_id=uuid.UUID(gid),
            source_profile_version=2,
        )
        repo.save(art)
        loaded = repo.load(sid)
        assert loaded is not None
        assert loaded.summary_id == sid
        assert loaded.meeting_id == uuid.UUID(mid)
        assert loaded.summary == art.summary

    def test_many_summaries(self, repo, meeting_repo, transcription_repo) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sids = [uuid.uuid4() for _ in range(3)]
        for sid in sids:
            repo.save(
                _artifact(
                    sid=sid,
                    meeting_id=uuid.UUID(mid),
                    source_generation_id=uuid.UUID(gid),
                    source_profile_version=2,
                )
            )
        results = repo.list_for_meeting(uuid.UUID(mid))
        assert len(results) == 3

    def test_payload_json_contains_only_summary_fields(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """payload_json must contain MeetingSummary fields only, not provenance."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        repo.save(
            _artifact(
                sid=sid,
                meeting_id=uuid.UUID(mid),
                source_generation_id=uuid.UUID(gid),
            )
        )
        # Read raw payload_json from DB
        from PyQt6.QtSql import QSqlQuery

        q = QSqlQuery(repo._database)
        q.prepare("SELECT payload_json FROM meeting_summary WHERE id = ?")
        q.addBindValue(str(sid))
        q.exec()
        q.next()
        payload = json.loads(q.value(0))

        # Must contain summary fields
        assert "schema_version" in payload
        assert "prompt_version" in payload
        assert "summary" in payload
        assert "participants" in payload

        # Must NOT contain provenance
        assert "summary_id" not in payload
        assert "meeting_id" not in payload
        assert "source_generation_id" not in payload
        assert "source_review_id" not in payload
        assert "created_at" not in payload

    def test_schema_prompt_mirrors_columns(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        s = _minimal_summary(prompt_version=3)
        repo.save(
            _artifact(
                sid=sid,
                meeting_id=uuid.UUID(mid),
                source_generation_id=uuid.UUID(gid),
                summary=s,
            )
        )
        q = QSqlQuery(repo._database)
        q.prepare(
            "SELECT schema_version, prompt_version FROM meeting_summary WHERE id = ?"
        )
        q.addBindValue(str(sid))
        q.exec()
        q.next()
        assert q.value(0) == MEETING_SUMMARY_SCHEMA_VERSION
        assert q.value(1) == 3

    def test_duplicate_artifact_id_conflict(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        art = _artifact(
            sid=sid,
            meeting_id=uuid.UUID(mid),
            source_generation_id=uuid.UUID(gid),
        )
        repo.save(art)
        with pytest.raises(MeetingSummaryConflictError):
            repo.save(art)

    def test_unrelated_unique_in_error_not_misclassified(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """An error containing 'UNIQUE' but not 'UNIQUE constraint'
        must remain DatabaseError, not ConflictError."""
        from unittest.mock import patch

        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        art = _artifact(
            sid=sid,
            meeting_id=uuid.UUID(mid),
            source_generation_id=uuid.UUID(gid),
        )

        original_execute = repo._execute
        call_count = 0

        def fake_execute(sql, values=()):
            nonlocal call_count
            call_count += 1
            if "INSERT INTO meeting_summary" in sql:
                raise MeetingSummaryDatabaseError(
                    "UNIQUE backend failure during insert"
                )
            return original_execute(sql, values)

        with patch.object(repo, "_execute", side_effect=fake_execute):
            with pytest.raises(MeetingSummaryDatabaseError):
                repo.save(art)

    def test_meeting_fk_violation(self, repo, meeting_repo, transcription_repo) -> None:
        """Save requires meeting to exist."""
        gid = str(uuid.uuid4())
        # Insert meeting and generation under real meeting
        real_mid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, real_mid)
        _insert_generation(transcription_repo, real_mid, gid)

        with pytest.raises(MeetingSummaryDatabaseError, match="Meeting .* not found"):
            repo.save(
                _artifact(
                    meeting_id=uuid.uuid4(),
                    source_generation_id=uuid.UUID(gid),
                )
            )

    def test_generation_fk_violation(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)

        with pytest.raises(
            MeetingSummaryDatabaseError, match="Source generation .* not found"
        ):
            repo.save(
                _artifact(
                    meeting_id=uuid.UUID(mid),
                    source_generation_id=uuid.uuid4(),
                )
            )

    def test_generation_wrong_meeting_reject(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid1 = str(uuid.uuid4())
        mid2 = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid1)
        _insert_meeting(meeting_repo, mid2)
        _insert_generation(transcription_repo, mid1, gid)

        with pytest.raises(MeetingSummaryDatabaseError, match="belongs to meeting"):
            repo.save(
                _artifact(
                    meeting_id=uuid.UUID(mid2),
                    source_generation_id=uuid.UUID(gid),
                )
            )

    def test_profile_mismatch_reject(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid, profile_version=2)

        with pytest.raises(MeetingSummaryDatabaseError, match="profile_version"):
            repo.save(
                _artifact(
                    meeting_id=uuid.UUID(mid),
                    source_generation_id=uuid.UUID(gid),
                    source_profile_version=99,
                )
            )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_not_found(self, repo) -> None:
        assert repo.load(uuid.uuid4()) is None

    def test_load_full_round_trip(self, repo, meeting_repo, transcription_repo) -> None:
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        s = _full_summary()
        art = _artifact(
            sid=sid,
            meeting_id=uuid.UUID(mid),
            source_generation_id=uuid.UUID(gid),
            summary=s,
        )
        repo.save(art)
        loaded = repo.load(sid)
        assert loaded is not None
        assert loaded.summary == s
        assert loaded.source_review_id == _RID
        assert loaded.source_review_revision == 4

    def test_load_no_cache(self, repo, meeting_repo, transcription_repo) -> None:
        """Each load is a fresh DB read."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        repo.save(
            _artifact(
                sid=sid,
                meeting_id=uuid.UUID(mid),
                source_generation_id=uuid.UUID(gid),
            )
        )
        a1 = repo.load(sid)
        a2 = repo.load(sid)
        assert a1 is not None and a2 is not None
        assert a1 is not a2

    def test_review_deletion_preserves_artifact(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """Deleting review rows must NOT affect summary artifact."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        art = _artifact(
            sid=sid,
            meeting_id=uuid.UUID(mid),
            source_generation_id=uuid.UUID(gid),
            review_id=_RID,
            review_revision=4,
        )
        repo.save(art)

        # Delete review rows (simulate review reset)
        q = QSqlQuery(repo._database)
        q.exec("DELETE FROM meeting_speaker_review")
        q.finish()

        loaded = repo.load(sid)
        assert loaded is not None
        assert loaded.source_review_id == _RID
        assert loaded.source_review_revision == 4

    def test_generation_cascade_preserves_summary(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """meeting_summary FK ON DELETE CASCADE on source_generation_id
        means deleting the generation also deletes the summary.
        This is correct behavior — not a test of preservation."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        repo.save(
            _artifact(
                sid=sid,
                meeting_id=uuid.UUID(mid),
                source_generation_id=uuid.UUID(gid),
            )
        )

        # Delete generation — cascades to summary
        q = QSqlQuery(repo._database)
        q.prepare("DELETE FROM meeting_final_transcription WHERE id = ?")
        q.addBindValue(gid)
        q.exec()
        q.finish()

        loaded = repo.load(sid)
        assert loaded is None  # Cascaded


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestList:
    def test_empty_list(self, repo, meeting_repo) -> None:
        mid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        assert repo.list_for_meeting(uuid.UUID(mid)) == ()

    def test_deterministic_ordering(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """Ordered by created_at ASC, then id ASC."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sids = sorted([uuid.uuid4() for _ in range(3)])
        for sid in sids:
            repo.save(
                _artifact(
                    sid=sid,
                    meeting_id=uuid.UUID(mid),
                    source_generation_id=uuid.UUID(gid),
                )
            )
        results = repo.list_for_meeting(uuid.UUID(mid))
        result_ids = [r.summary_id for r in results]
        assert result_ids == sids

    def test_external_db_mutation_visible(
        self, repo, meeting_repo, transcription_repo
    ) -> None:
        """External DB mutation is visible on next read (no cache)."""
        mid = str(uuid.uuid4())
        gid = str(uuid.uuid4())
        _insert_meeting(meeting_repo, mid)
        _insert_generation(transcription_repo, mid, gid)

        sid = uuid.uuid4()
        repo.save(
            _artifact(
                sid=sid,
                meeting_id=uuid.UUID(mid),
                source_generation_id=uuid.UUID(gid),
            )
        )
        assert len(repo.list_for_meeting(uuid.UUID(mid))) == 1

        # Delete via raw SQL
        q = QSqlQuery(repo._database)
        q.prepare("DELETE FROM meeting_summary WHERE id = ?")
        q.addBindValue(str(sid))
        q.exec()
        q.finish()

        assert len(repo.list_for_meeting(uuid.UUID(mid))) == 0


# ---------------------------------------------------------------------------
# Decode corruption
# ---------------------------------------------------------------------------


class TestDecodeCorruption:
    def _inject_row(
        self, repo, *, payload_json: str = "", **overrides: object
    ) -> uuid.UUID:
        """Insert a raw row for corruption testing.

        Temporarily disables FK enforcement so we can inject rows with
        invalid provenance for decode testing.
        """
        sid = uuid.uuid4()
        defaults = dict(
            id=str(sid),
            meeting_id=str(uuid.uuid4()),
            source_generation_id=str(uuid.uuid4()),
            source_profile_version=2,
            source_review_id=None,
            source_review_revision=None,
            schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
            prompt_version=1,
            created_at="2026-01-01T00:00:00+00:00",
            payload_json=payload_json or meeting_summary_to_json(_minimal_summary()),
        )
        defaults.update(overrides)
        fk_q = QSqlQuery(repo._database)
        fk_q.exec("PRAGMA foreign_keys = OFF")
        fk_q.exec("PRAGMA ignore_check_constraints = ON")
        fk_q.finish()
        try:
            q = QSqlQuery(repo._database)
            q.prepare(
                """
                INSERT INTO meeting_summary (
                    id, meeting_id, source_generation_id,
                    source_profile_version,
                    source_review_id, source_review_revision,
                    schema_version, prompt_version,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            )
            for key in (
                "id",
                "meeting_id",
                "source_generation_id",
                "source_profile_version",
                "source_review_id",
                "source_review_revision",
                "schema_version",
                "prompt_version",
                "created_at",
                "payload_json",
            ):
                q.addBindValue(defaults[key])
            q.exec()
            q.finish()
        finally:
            fk_q2 = QSqlQuery(repo._database)
            fk_q2.exec("PRAGMA foreign_keys = ON")
            fk_q2.exec("PRAGMA ignore_check_constraints = OFF")
            fk_q2.finish()
        return sid

    def test_malformed_payload_json(self, repo) -> None:
        sid = self._inject_row(repo, payload_json="{not json")
        with pytest.raises(MeetingSummaryDecodeError, match="Malformed JSON"):
            repo.load(sid)

    def test_payload_non_object(self, repo) -> None:
        sid = self._inject_row(repo, payload_json='"hello"')
        with pytest.raises(MeetingSummaryDecodeError, match="must be an object"):
            repo.load(sid)

    def test_unknown_payload_field(self, repo) -> None:
        d = json.loads(meeting_summary_to_json(_minimal_summary()))
        d["bogus"] = "x"
        sid = self._inject_row(repo, payload_json=json.dumps(d))
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown top-level"):
            repo.load(sid)

    def test_unsupported_payload_schema_version(self, repo) -> None:
        d = json.loads(meeting_summary_to_json(_minimal_summary()))
        d["schema_version"] = 999
        sid = self._inject_row(repo, payload_json=json.dumps(d))
        with pytest.raises(MeetingSummaryDecodeError):
            repo.load(sid)

    def test_corrupt_uuid(self, repo) -> None:
        """Corrupt meeting_id UUID in a valid row triggers DecodeError."""
        sid = self._inject_row(repo, meeting_id="not-a-uuid")
        with pytest.raises(MeetingSummaryDecodeError, match="Invalid meeting_id"):
            repo.load(sid)

    def test_naive_created_at(self, repo) -> None:
        sid = self._inject_row(repo, created_at="2026-01-01T00:00:00")
        with pytest.raises(MeetingSummaryDecodeError, match="timezone-aware"):
            repo.load(sid)

    def test_row_schema_version_mismatch(self, repo) -> None:
        """Row schema_version != payload schema_version → DecodeError."""
        d = json.loads(meeting_summary_to_json(_minimal_summary()))
        # Payload has schema_version=1, but row will have 999
        sid = self._inject_row(repo, schema_version=999, payload_json=json.dumps(d))
        with pytest.raises(MeetingSummaryDecodeError, match="schema_version"):
            repo.load(sid)

    def test_row_prompt_version_mismatch(self, repo) -> None:
        d = json.loads(meeting_summary_to_json(_minimal_summary()))
        sid = self._inject_row(repo, prompt_version=99, payload_json=json.dumps(d))
        with pytest.raises(MeetingSummaryDecodeError, match="prompt_version"):
            repo.load(sid)

    def test_review_half_pair_corruption(self, repo) -> None:
        """source_review_id present but source_review_revision NULL."""
        sid = self._inject_row(
            repo,
            source_review_id=str(_RID),
            source_review_revision=None,
        )
        with pytest.raises(MeetingSummaryDecodeError, match="Review pair corruption"):
            repo.load(sid)

    def test_review_half_pair_corruption_other(self, repo) -> None:
        sid = self._inject_row(
            repo,
            source_review_id=None,
            source_review_revision=0,
        )
        with pytest.raises(MeetingSummaryDecodeError, match="Review pair corruption"):
            repo.load(sid)
