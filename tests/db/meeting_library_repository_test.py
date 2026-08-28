from __future__ import annotations

import gc
import uuid

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtSql import QSqlDatabase, QSqlQuery

import buzz.db.meeting_library_repository as repository_module
from buzz.db.meeting_library_repository import QSqlMeetingLibraryRepository
from buzz.meeting.meeting_library import MeetingLibraryDatabaseError


CREATE_MEETING_SQL = """
    CREATE TABLE meeting (
        id TEXT PRIMARY KEY NOT NULL,
        remote_source_kind TEXT NOT NULL,
        session_state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        ended_at TEXT,
        duration_ns INTEGER,
        audio_state TEXT NOT NULL,
        audio_outcome TEXT
    )
"""


@pytest.fixture(scope="module")
def qt_core_application():
    application = QCoreApplication.instance()
    owns_application = application is None
    if application is None:
        application = QCoreApplication([])
    yield application
    if owns_application:
        application.quit()


@pytest.fixture
def database(qt_core_application):
    connection_name = f"meeting-library-{uuid.uuid4()}"
    value = QSqlDatabase.addDatabase("QSQLITE", connection_name)
    value.setDatabaseName(":memory:")
    assert value.open(), value.lastError().text()
    execute(value, CREATE_MEETING_SQL)
    yield value
    value.close()
    del value
    gc.collect()
    QSqlDatabase.removeDatabase(connection_name)


def execute(database: QSqlDatabase, sql: str, values: tuple = ()) -> None:
    query = QSqlQuery(database)
    assert query.prepare(sql), query.lastError().text()
    for value in values:
        query.addBindValue(value)
    assert query.exec(), query.lastError().text()
    query.finish()


def insert_meeting(
    database: QSqlDatabase,
    session_id: str | None = None,
    *,
    remote_source_kind: str = "SYSTEM",
    session_state: str = "COMPLETED",
    started_at: str | None = "2025-01-01T00:01:00+00:00",
    duration_ns: int | None = 60_000_000_000,
    audio_state: str = "STOPPED",
    audio_outcome: str | None = "COMPLETE",
) -> str:
    meeting_id = session_id or str(uuid.uuid4())
    execute(
        database,
        """
        INSERT INTO meeting (
            id, remote_source_kind, session_state, created_at, started_at,
            ended_at, duration_ns, audio_state, audio_outcome
        ) VALUES (?, ?, ?, '2025-01-01T00:00:00+00:00', ?, NULL, ?, ?, ?)
        """,
        (
            meeting_id,
            remote_source_kind,
            session_state,
            started_at,
            duration_ns,
            audio_state,
            audio_outcome,
        ),
    )
    return meeting_id


def test_zero_meetings(database) -> None:
    assert QSqlMeetingLibraryRepository(database).list_meetings() == ()


def test_one_meeting_returns_exact_nine_selected_fields(database) -> None:
    meeting_id = insert_meeting(database, started_at=None, duration_ns=None)
    records = QSqlMeetingLibraryRepository(database).list_meetings()
    assert len(records) == 1
    assert records[0].session_id == meeting_id
    assert records[0].remote_source_kind == "SYSTEM"
    assert records[0].session_state == "COMPLETED"
    assert records[0].created_at == "2025-01-01T00:00:00+00:00"
    assert records[0].started_at is None
    assert records[0].ended_at is None
    assert records[0].duration_ns is None
    assert records[0].audio_state == "STOPPED"
    assert records[0].audio_outcome == "COMPLETE"


def test_many_meetings_are_all_returned_including_nonterminal_and_failed(
    database,
) -> None:
    ids = {
        insert_meeting(database, session_state=state, audio_outcome=None)
        for state in ("CREATED", "STARTING", "ACTIVE", "STOPPING", "FAILED")
    }
    records = QSqlMeetingLibraryRepository(database).list_meetings()
    assert {record.session_id for record in records} == ids


class FakeError:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class FakeQuery:
    instances = []
    prepare_calls = 0
    exec_calls = 0
    prepared_sql = []
    prepare_result = True
    exec_result = True
    header_row = (
        "00000000-0000-0000-0000-000000000001",
        "SYSTEM",
        "COMPLETED",
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:01:00+00:00",
        "2025-01-01T00:02:00+00:00",
        60_000_000_000,
        "STOPPED",
        "COMPLETE",
    )

    def __init__(self, database) -> None:
        self.database = database
        self.sql = None
        self._next_index = -1
        self.instances.append(self)

    def prepare(self, sql: str) -> bool:
        type(self).prepare_calls += 1
        type(self).prepared_sql.append(sql)
        self.sql = sql
        return self.prepare_result

    def exec(self) -> bool:
        type(self).exec_calls += 1
        return self.exec_result

    def next(self) -> bool:
        self._next_index += 1
        return self._next_index == 0

    def isNull(self, index: int) -> bool:
        return self.header_row[index] is None

    def value(self, index: int):
        return self.header_row[index]

    def lastError(self) -> FakeError:
        return FakeError("injected query failure")


@pytest.fixture
def fake_query(monkeypatch):
    FakeQuery.instances = []
    FakeQuery.prepare_calls = 0
    FakeQuery.exec_calls = 0
    FakeQuery.prepared_sql = []
    FakeQuery.prepare_result = True
    FakeQuery.exec_result = True
    monkeypatch.setattr(repository_module, "QSqlQuery", FakeQuery)
    return FakeQuery


def test_populated_result_uses_exactly_one_header_select_for_the_whole_call(
    fake_query,
) -> None:
    supplied_database = object()
    repository = QSqlMeetingLibraryRepository(supplied_database)
    records = repository.list_meetings()

    assert len(records) == 1
    record = records[0]
    assert (
        record.session_id,
        record.remote_source_kind,
        record.session_state,
        record.created_at,
        record.started_at,
        record.ended_at,
        record.duration_ns,
        record.audio_state,
        record.audio_outcome,
    ) == fake_query.header_row

    assert len(fake_query.instances) == 1
    assert fake_query.instances[0].database is supplied_database
    assert fake_query.prepare_calls == 1
    assert fake_query.exec_calls == 1
    assert len(fake_query.prepared_sql) == 1
    normalized_sql = " ".join(fake_query.prepared_sql[0].split())
    assert normalized_sql == (
        "SELECT id, remote_source_kind, session_state, created_at, started_at, "
        "ended_at, duration_ns, audio_state, audio_outcome FROM meeting"
    )


def test_prepare_failure_raises_database_error(fake_query) -> None:
    fake_query.prepare_result = False
    with pytest.raises(MeetingLibraryDatabaseError, match="prepare"):
        QSqlMeetingLibraryRepository(object()).list_meetings()


def test_exec_failure_raises_database_error(fake_query) -> None:
    fake_query.exec_result = False
    with pytest.raises(MeetingLibraryDatabaseError, match="execute"):
        QSqlMeetingLibraryRepository(object()).list_meetings()


def test_parent_without_audio_children_is_listed(database) -> None:
    meeting_id = insert_meeting(database, audio_outcome=None)
    records = QSqlMeetingLibraryRepository(database).list_meetings()
    assert [record.session_id for record in records] == [meeting_id]


def test_missing_audio_files_and_corrupt_child_rows_do_not_affect_list(
    database,
) -> None:
    meeting_id = insert_meeting(database, audio_outcome="PARTIAL")
    execute(
        database,
        """
        CREATE TABLE meeting_audio_track (
            meeting_id TEXT, relative_path TEXT, sample_rate TEXT
        )
        """,
    )
    execute(
        database,
        "INSERT INTO meeting_audio_track VALUES (?, ?, 'corrupt')",
        (meeting_id, "directory-that-does-not-exist/missing.wav"),
    )
    records = QSqlMeetingLibraryRepository(database).list_meetings()
    assert len(records) == 1
    assert records[0].session_id == meeting_id


def test_repository_does_not_close_caller_owned_connection(database) -> None:
    QSqlMeetingLibraryRepository(database).list_meetings()
    assert database.isOpen()
    insert_meeting(database)


def test_fresh_database_changes_are_visible_on_next_call(database) -> None:
    repository = QSqlMeetingLibraryRepository(database)
    assert repository.list_meetings() == ()
    meeting_id = insert_meeting(database)
    assert repository.list_meetings()[0].session_id == meeting_id
