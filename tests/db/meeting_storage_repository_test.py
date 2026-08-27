from __future__ import annotations

import gc
import sqlite3
import uuid
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.db.meeting_storage_repository import QSqlMeetingRepository
from buzz.meeting.meeting_storage import (
    MeetingErrorPersistenceRecord,
    MeetingPersistenceBundle,
    MeetingStorage,
    MeetingStorageConflictError,
    MeetingStorageDatabaseError,
    MeetingStorageDecodeError,
    MeetingTimingPersistenceRecord,
    MeetingTrackPersistenceRecord,
)


def make_bundle(
    session_id: str | None = None,
    *,
    state: str = "COMPLETED",
    outcome: str | None = "COMPLETE",
    sample_count: int = 20,
    remote_source_kind: str = "SYSTEM",
) -> MeetingPersistenceBundle:
    meeting_id = session_id or str(uuid.uuid4())
    tracks = ()
    timings = ()
    errors = ()
    if outcome is not None:
        tracks = tuple(
            MeetingTrackPersistenceRecord(
                role=role,
                relative_path=f"{meeting_id}/{filename}",
                sample_rate=10 if role == "MICROPHONE" else 20,
                sample_count=sample_count if role == "MICROPHONE" else 30,
                recording_state="STOPPED",
                published=1 if role == "MICROPHONE" else 0,
                complete=1,
                timing_basis="host_callback_arrival",
            )
            for role, filename in (
                ("MICROPHONE", "microphone.wav"),
                ("REMOTE", "remote.wav"),
            )
        )
        timings = (
            MeetingTimingPersistenceRecord("MICROPHONE", 0, 10, -5),
            MeetingTimingPersistenceRecord("REMOTE", 0, 20, 7),
        )
        errors = (
            MeetingErrorPersistenceRecord(
                "MICROPHONE", 0, "RECORDER", "builtins", "ValueError", "failure"
            ),
        )
    return MeetingPersistenceBundle(
        session_id=meeting_id,
        remote_source_kind=remote_source_kind,
        session_state=state,
        created_at="2025-01-01T00:00:00+00:00",
        started_at="2025-01-01T00:01:00+00:00",
        ended_at="2025-01-01T00:02:00+00:00",
        duration_ns=60_000_000_000,
        audio_state="STOPPED",
        audio_outcome=outcome,
        tracks=tracks,
        timings=timings,
        errors=errors,
    )


@pytest.fixture
def qsql_database(tmp_path: Path, qt_application):
    database_path = tmp_path / "meeting.sqlite"
    schema = Path("buzz/schema.sql").read_text()
    connection = sqlite3.connect(database_path)
    connection.executescript(schema)
    connection.close()
    name = f"meeting-storage-{uuid.uuid4()}"
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


@pytest.fixture(scope="module")
def qt_application():
    application = QCoreApplication.instance()
    owns_application = application is None
    if application is None:
        application = QCoreApplication([])
    yield application
    if owns_application:
        application.quit()


@pytest.mark.parametrize("outcome", ["COMPLETE", "PARTIAL", "FAILED"])
def test_roundtrip_two_track_aggregate(qsql_database, outcome: str) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    bundle = make_bundle(outcome=outcome)
    repository.atomic_replace(bundle, validate_existing=lambda existing: None)
    assert repository.load_bundle(bundle.session_id) == bundle


def test_created_roundtrip_has_no_children_and_missing_is_none(qsql_database) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    bundle = make_bundle(state="CREATED", outcome=None)
    repository.atomic_replace(bundle, validate_existing=lambda existing: None)
    assert repository.load_bundle(bundle.session_id) == bundle
    assert repository.load_bundle(str(uuid.uuid4())) is None


def test_application_source_roundtrip(qsql_database) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    bundle = make_bundle(remote_source_kind="APPLICATION")
    repository.atomic_replace(bundle, validate_existing=lambda existing: None)
    assert repository.load_bundle(bundle.session_id) == bundle


def execute_sql(database: QSqlDatabase, sql: str, values: tuple = ()) -> None:
    query = QSqlQuery(database)
    assert query.prepare(sql), query.lastError().text()
    for value in values:
        query.addBindValue(value)
    assert query.exec(), query.lastError().text()
    query.finish()
    del query


def insert_orphan_track(database: QSqlDatabase, meeting_id: str) -> None:
    execute_sql(
        database,
        """
        INSERT INTO meeting_audio_track (
            meeting_id, role, relative_path, sample_rate, sample_count,
            recording_state, published, complete, timing_basis
        ) VALUES (?, 'MICROPHONE', ?, 10, 20, 'STOPPED', 1, 1,
                  'host_callback_arrival')
        """,
        (meeting_id, f"{meeting_id}/microphone.wav"),
    )


def insert_orphan_timing(database: QSqlDatabase, meeting_id: str) -> None:
    execute_sql(
        database,
        """
        INSERT INTO meeting_audio_timing_anchor (
            meeting_id, role, ordinal, sample_end, callback_arrival_offset_ns
        ) VALUES (?, 'MICROPHONE', 0, 10, -1)
        """,
        (meeting_id,),
    )


def insert_orphan_error(database: QSqlDatabase, meeting_id: str) -> None:
    execute_sql(
        database,
        """
        INSERT INTO meeting_audio_error (
            meeting_id, role, ordinal, stage, exception_module,
            exception_name, message
        ) VALUES (?, 'MICROPHONE', 0, 'RECORDER', 'builtins',
                  'RuntimeError', 'orphan')
        """,
        (meeting_id,),
    )


@pytest.mark.parametrize(
    "child_kinds", [("track",), ("timing",), ("error",), ("track", "timing", "error")]
)
def test_parentless_children_reach_facade_decode(
    qsql_database, tmp_path: Path, child_kinds: tuple[str, ...]
) -> None:
    database, _ = qsql_database
    execute_sql(database, "PRAGMA foreign_keys = OFF")
    session_id = uuid.uuid4()
    inserters = {
        "track": insert_orphan_track,
        "timing": insert_orphan_timing,
        "error": insert_orphan_error,
    }
    for child_kind in child_kinds:
        inserters[child_kind](database, str(session_id))

    storage = MeetingStorage(QSqlMeetingRepository(database), root=tmp_path)
    with pytest.raises(MeetingStorageDecodeError):
        storage.load(session_id)


def test_parent_with_malformed_children_still_reaches_facade_decode(
    qsql_database, tmp_path: Path
) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    bundle = make_bundle(state="CREATED", outcome=None)
    repository.atomic_replace(bundle, validate_existing=lambda existing: None)
    insert_orphan_track(database, bundle.session_id)

    with pytest.raises(MeetingStorageDecodeError):
        MeetingStorage(repository, root=tmp_path).load(uuid.UUID(bundle.session_id))


def test_complete_replacement_does_not_leave_stale_children(qsql_database) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    first = make_bundle(state="FAILED")
    repository.atomic_replace(first, validate_existing=lambda existing: None)
    second = make_bundle(first.session_id, state="FAILED", sample_count=12)
    repository.atomic_replace(second, validate_existing=lambda existing: None)
    assert repository.load_bundle(first.session_id) == second


def test_semantic_exception_rolls_back_and_is_same_object(qsql_database) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    bundle = make_bundle()
    error = MeetingStorageConflictError("stale")

    def reject(existing) -> None:
        raise error

    with pytest.raises(MeetingStorageConflictError) as caught:
        repository.atomic_replace(bundle, validate_existing=reject)
    assert caught.value is error
    assert repository.load_bundle(bundle.session_id) is None


def test_mid_transaction_exec_failure_preserves_old_aggregate_and_audio(
    qsql_database, tmp_path: Path
) -> None:
    database, _ = qsql_database
    repository = QSqlMeetingRepository(database)
    old = make_bundle(state="FAILED")
    repository.atomic_replace(old, validate_existing=lambda existing: None)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"durable-audio")

    trigger = QSqlQuery(database)
    assert trigger.exec(
        """
        CREATE TRIGGER reject_remote BEFORE INSERT ON meeting_audio_track
        WHEN NEW.role = 'REMOTE'
        BEGIN SELECT RAISE(ABORT, 'injected child failure'); END
        """
    )
    trigger.finish()
    del trigger

    new = make_bundle(old.session_id, state="FAILED", sample_count=12)
    with pytest.raises(MeetingStorageDatabaseError) as caught:
        repository.atomic_replace(new, validate_existing=lambda existing: None)
    assert caught.value.commit_outcome_unknown is False
    assert repository.load_bundle(old.session_id) == old
    assert audio.read_bytes() == b"durable-audio"

    drop = QSqlQuery(database)
    assert drop.exec("DROP TRIGGER reject_remote")
    drop.finish()
    del drop
    repository.atomic_replace(new, validate_existing=lambda existing: None)
    assert repository.load_bundle(old.session_id) == new
    check = QSqlQuery(database)
    assert check.exec("PRAGMA foreign_key_check")
    assert not check.next()
    check.finish()
    del check


class _LastError:
    def __init__(self, message: str) -> None:
        self.message = message

    def text(self) -> str:
        return self.message


class _FakeDatabase:
    def __init__(
        self,
        *,
        commit_result: bool = True,
        rollback_result: bool = True,
    ) -> None:
        self.commit_result = commit_result
        self.rollback_result = rollback_result
        self.rollback_called = False
        self.error_message = "injected commit failure"

    def transaction(self) -> bool:
        return True

    def commit(self) -> bool:
        return self.commit_result

    def rollback(self) -> bool:
        self.rollback_called = True
        if not self.rollback_result:
            self.error_message = "injected rollback failure"
        return self.rollback_result

    def lastError(self) -> _LastError:
        return _LastError(self.error_message)


class _SeamRepository(QSqlMeetingRepository):
    def __init__(self, database, *, failure_phase: str | None = None) -> None:
        super().__init__(database)
        self.failure_phase = failure_phase

    def _load_bundle(self, session_id):
        if self.failure_phase == "read":
            raise MeetingStorageDatabaseError("original read query failure")
        return None

    def _upsert_meeting(self, bundle) -> None:
        pass

    def _execute(self, sql, values):
        return None

    def _insert_tracks(self, bundle) -> None:
        if self.failure_phase == "write":
            raise MeetingStorageDatabaseError("original child write failure")
        pass

    def _insert_timings(self, bundle) -> None:
        pass

    def _insert_errors(self, bundle) -> None:
        pass


def test_commit_failure_marks_outcome_unknown_and_attempts_rollback() -> None:
    database = _FakeDatabase(commit_result=False)
    repository = _SeamRepository(database)
    with pytest.raises(MeetingStorageDatabaseError) as caught:
        repository.atomic_replace(
            make_bundle(), validate_existing=lambda existing: None
        )
    assert caught.value.commit_outcome_unknown is True
    assert database.rollback_called


def test_semantic_error_identity_survives_rollback_failure() -> None:
    database = _FakeDatabase(rollback_result=False)
    repository = _SeamRepository(database)
    error = MeetingStorageConflictError("stale")

    def reject(existing) -> None:
        raise error

    with pytest.raises(MeetingStorageConflictError) as caught:
        repository.atomic_replace(make_bundle(), validate_existing=reject)
    assert caught.value is error
    assert database.rollback_called
    assert any("rollback also failed" in note for note in caught.value.__notes__)
    assert any("state is uncertain" in note for note in caught.value.__notes__)


@pytest.mark.parametrize(
    ("failure_phase", "original_message"),
    [
        ("write", "original child write failure"),
        ("read", "original read query failure"),
    ],
)
def test_database_failure_reports_rollback_failure_and_uncertain_state(
    failure_phase: str,
    original_message: str,
) -> None:
    database = _FakeDatabase(rollback_result=False)
    repository = _SeamRepository(database, failure_phase=failure_phase)
    with pytest.raises(MeetingStorageDatabaseError) as caught:
        repository.atomic_replace(
            make_bundle(), validate_existing=lambda existing: None
        )
    assert caught.value.commit_outcome_unknown is False
    assert database.rollback_called
    assert original_message in str(caught.value)
    assert "rollback also failed" in str(caught.value)
    assert "state is uncertain" in str(caught.value)


def test_commit_and_rollback_failure_preserve_both_diagnostics() -> None:
    database = _FakeDatabase(commit_result=False, rollback_result=False)
    repository = _SeamRepository(database)
    with pytest.raises(MeetingStorageDatabaseError) as caught:
        repository.atomic_replace(
            make_bundle(), validate_existing=lambda existing: None
        )
    assert caught.value.commit_outcome_unknown is True
    assert database.rollback_called
    assert "commit meeting transaction" in str(caught.value)
    assert "injected commit failure" in str(caught.value)
    assert "rollback also failed" in str(caught.value)
    assert "injected rollback failure" in str(caught.value)
    assert "state is uncertain" in str(caught.value)
