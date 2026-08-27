"""QSql persistence adapter for meeting storage primitive bundles."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional

from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.meeting.meeting_storage import (
    ExistingBundleValidator,
    MeetingErrorPersistenceRecord,
    MeetingPersistenceBundle,
    MeetingPersistenceLoadResult,
    MeetingPersistenceReadBundle,
    MeetingStorageDatabaseError,
    MeetingTimingPersistenceRecord,
    MeetingTrackPersistenceRecord,
)


class QSqlMeetingRepository:
    """Store meeting aggregates using a caller-owned QSql connection.

    Methods must run on the thread that owns ``database``.  The adapter is not
    thread-safe, does not use Qt's default connection, and never creates,
    closes, or moves the supplied connection.
    """

    def __init__(self, database: QSqlDatabase) -> None:
        self._database = database

    def atomic_replace(
        self,
        bundle: MeetingPersistenceBundle,
        *,
        validate_existing: ExistingBundleValidator,
    ) -> None:
        if not self._database.transaction():
            raise self._database_error("Could not begin meeting transaction")
        try:
            existing = self._load_bundle(bundle.session_id)
            validate_existing(existing)
            self._upsert_meeting(bundle)
            self._execute(
                "DELETE FROM meeting_audio_timing_anchor WHERE meeting_id = ?",
                (bundle.session_id,),
            )
            self._execute(
                "DELETE FROM meeting_audio_error WHERE meeting_id = ?",
                (bundle.session_id,),
            )
            self._execute(
                "DELETE FROM meeting_audio_track WHERE meeting_id = ?",
                (bundle.session_id,),
            )
            self._insert_tracks(bundle)
            self._insert_timings(bundle)
            self._insert_errors(bundle)
        except Exception as exc:
            self._raise_after_rollback(exc)

        if not self._database.commit():
            commit_error = self._database_error(
                "Could not commit meeting transaction",
                commit_outcome_unknown=True,
            )
            self._raise_after_rollback(commit_error)

    def load_bundle(self, session_id: str) -> Optional[MeetingPersistenceLoadResult]:
        if not self._database.transaction():
            raise self._database_error("Could not begin meeting read transaction")
        try:
            bundle = self._load_bundle(session_id)
        except Exception as exc:
            self._raise_after_rollback(exc)
        if not self._database.commit():
            commit_error = self._database_error(
                "Could not commit meeting read transaction",
                commit_outcome_unknown=True,
            )
            self._raise_after_rollback(commit_error)
        return bundle

    def _load_bundle(self, session_id: str) -> Optional[MeetingPersistenceLoadResult]:
        query = self._execute(
            """
            SELECT id, remote_source_kind, session_state, created_at, started_at,
                   ended_at, duration_ns, audio_state, audio_outcome
            FROM meeting WHERE id = ?
            """,
            (session_id,),
        )
        meeting_values = (
            tuple(self._nullable_value(query, ordinal) for ordinal in range(9))
            if query.next()
            else None
        )

        track_query = self._execute(
            """
            SELECT role, relative_path, sample_rate, sample_count,
                   recording_state, published, complete, timing_basis
            FROM meeting_audio_track
            WHERE meeting_id = ? ORDER BY role
            """,
            (session_id,),
        )
        tracks: list[MeetingTrackPersistenceRecord] = []
        while track_query.next():
            tracks.append(
                MeetingTrackPersistenceRecord(
                    role=track_query.value(0),
                    relative_path=track_query.value(1),
                    sample_rate=track_query.value(2),
                    sample_count=track_query.value(3),
                    recording_state=track_query.value(4),
                    published=track_query.value(5),
                    complete=track_query.value(6),
                    timing_basis=track_query.value(7),
                )
            )

        timing_query = self._execute(
            """
            SELECT role, ordinal, sample_end, callback_arrival_offset_ns
            FROM meeting_audio_timing_anchor
            WHERE meeting_id = ? ORDER BY role, ordinal
            """,
            (session_id,),
        )
        timings: list[MeetingTimingPersistenceRecord] = []
        while timing_query.next():
            timings.append(
                MeetingTimingPersistenceRecord(
                    role=timing_query.value(0),
                    ordinal=timing_query.value(1),
                    sample_end=timing_query.value(2),
                    callback_arrival_offset_ns=timing_query.value(3),
                )
            )

        error_query = self._execute(
            """
            SELECT role, ordinal, stage, exception_module, exception_name,
                   message
            FROM meeting_audio_error
            WHERE meeting_id = ? ORDER BY role, ordinal
            """,
            (session_id,),
        )
        errors: list[MeetingErrorPersistenceRecord] = []
        while error_query.next():
            errors.append(
                MeetingErrorPersistenceRecord(
                    role=error_query.value(0),
                    ordinal=error_query.value(1),
                    stage=error_query.value(2),
                    exception_module=error_query.value(3),
                    exception_name=error_query.value(4),
                    message=error_query.value(5),
                )
            )

        if meeting_values is None:
            if not tracks and not timings and not errors:
                return None
            return MeetingPersistenceReadBundle(
                meeting=None,
                tracks=tuple(tracks),
                timings=tuple(timings),
                errors=tuple(errors),
            )

        return MeetingPersistenceBundle(
            session_id=meeting_values[0],
            remote_source_kind=meeting_values[1],
            session_state=meeting_values[2],
            created_at=meeting_values[3],
            started_at=meeting_values[4],
            ended_at=meeting_values[5],
            duration_ns=meeting_values[6],
            audio_state=meeting_values[7],
            audio_outcome=meeting_values[8],
            tracks=tuple(tracks),
            timings=tuple(timings),
            errors=tuple(errors),
        )

    def _raise_after_rollback(self, error: Exception) -> None:
        rollback_failure = self._rollback_failure()
        if rollback_failure is None:
            raise error

        logging.error(rollback_failure)
        if isinstance(error, MeetingStorageDatabaseError):
            raise MeetingStorageDatabaseError(
                f"{error}; {rollback_failure}",
                commit_outcome_unknown=error.commit_outcome_unknown,
            ) from error

        error.add_note(rollback_failure)
        raise error

    def _rollback_failure(self) -> Optional[str]:
        try:
            if self._database.rollback():
                return None
            details = self._database.lastError().text()
        except Exception as exc:
            details = f"rollback raised {type(exc).__name__}: {exc}"
        return (
            "Database rollback also failed; transaction/connection state is "
            f"uncertain: {details}"
        )

    def _upsert_meeting(self, bundle: MeetingPersistenceBundle) -> None:
        self._execute(
            """
            INSERT INTO meeting (
                id, remote_source_kind, session_state, created_at, started_at,
                ended_at, duration_ns, audio_state, audio_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                remote_source_kind = excluded.remote_source_kind,
                session_state = excluded.session_state,
                created_at = excluded.created_at,
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                duration_ns = excluded.duration_ns,
                audio_state = excluded.audio_state,
                audio_outcome = excluded.audio_outcome
            """,
            (
                bundle.session_id,
                bundle.remote_source_kind,
                bundle.session_state,
                bundle.created_at,
                bundle.started_at,
                bundle.ended_at,
                bundle.duration_ns,
                bundle.audio_state,
                bundle.audio_outcome,
            ),
        )

    def _insert_tracks(self, bundle: MeetingPersistenceBundle) -> None:
        sql = """
            INSERT INTO meeting_audio_track (
                meeting_id, role, relative_path, sample_rate, sample_count,
                recording_state, published, complete, timing_basis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for track in bundle.tracks:
            self._execute(
                sql,
                (
                    bundle.session_id,
                    track.role,
                    track.relative_path,
                    track.sample_rate,
                    track.sample_count,
                    track.recording_state,
                    track.published,
                    track.complete,
                    track.timing_basis,
                ),
            )

    def _insert_timings(self, bundle: MeetingPersistenceBundle) -> None:
        sql = """
            INSERT INTO meeting_audio_timing_anchor (
                meeting_id, role, ordinal, sample_end,
                callback_arrival_offset_ns
            ) VALUES (?, ?, ?, ?, ?)
        """
        for timing in bundle.timings:
            self._execute(
                sql,
                (
                    bundle.session_id,
                    timing.role,
                    timing.ordinal,
                    timing.sample_end,
                    timing.callback_arrival_offset_ns,
                ),
            )

    def _insert_errors(self, bundle: MeetingPersistenceBundle) -> None:
        sql = """
            INSERT INTO meeting_audio_error (
                meeting_id, role, ordinal, stage, exception_module,
                exception_name, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        for error in bundle.errors:
            self._execute(
                sql,
                (
                    bundle.session_id,
                    error.role,
                    error.ordinal,
                    error.stage,
                    error.exception_module,
                    error.exception_name,
                    error.message,
                ),
            )

    def _execute(self, sql: str, values: Sequence[Any]) -> QSqlQuery:
        query = QSqlQuery(self._database)
        if not query.prepare(sql):
            raise self._query_error("Could not prepare meeting SQL", query)
        for value in values:
            query.addBindValue(value)
        if not query.exec():
            raise self._query_error("Could not execute meeting SQL", query)
        return query

    @staticmethod
    def _nullable_value(query: QSqlQuery, ordinal: int) -> Any:
        return None if query.isNull(ordinal) else query.value(ordinal)

    def _query_error(
        self, message: str, query: QSqlQuery
    ) -> MeetingStorageDatabaseError:
        details = query.lastError().text()
        return MeetingStorageDatabaseError(
            f"{message}: {details}",
            commit_outcome_unknown=False,
        )

    def _database_error(
        self,
        message: str,
        *,
        commit_outcome_unknown: bool = False,
    ) -> MeetingStorageDatabaseError:
        details = self._database.lastError().text()
        return MeetingStorageDatabaseError(
            f"{message}: {details}",
            commit_outcome_unknown=commit_outcome_unknown,
        )


__all__ = ["QSqlMeetingRepository"]
