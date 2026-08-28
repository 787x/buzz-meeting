"""QSql adapter for the lightweight meetings library projection."""

from __future__ import annotations

from typing import Any

from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.meeting.meeting_library import (
    MeetingLibraryDatabaseError,
    MeetingLibraryRecord,
)


_LIST_MEETINGS_SQL = """
    SELECT
        id,
        remote_source_kind,
        session_state,
        created_at,
        started_at,
        ended_at,
        duration_ns,
        audio_state,
        audio_outcome
    FROM meeting
"""


class QSqlMeetingLibraryRepository:
    """Read meeting headers using a caller-owned QSql connection."""

    def __init__(self, database: QSqlDatabase) -> None:
        self._database = database

    def list_meetings(self) -> tuple[MeetingLibraryRecord, ...]:
        query = QSqlQuery(self._database)
        if not query.prepare(_LIST_MEETINGS_SQL):
            raise self._query_error("Could not prepare meetings library SQL", query)
        if not query.exec():
            raise self._query_error("Could not execute meetings library SQL", query)

        records: list[MeetingLibraryRecord] = []
        while query.next():
            values = tuple(self._nullable_value(query, index) for index in range(9))
            records.append(MeetingLibraryRecord(*values))
        return tuple(records)

    @staticmethod
    def _nullable_value(query: QSqlQuery, index: int) -> Any:
        return None if query.isNull(index) else query.value(index)

    @staticmethod
    def _query_error(message: str, query: QSqlQuery) -> MeetingLibraryDatabaseError:
        return MeetingLibraryDatabaseError(f"{message}: {query.lastError().text()}")


__all__ = ["QSqlMeetingLibraryRepository"]
