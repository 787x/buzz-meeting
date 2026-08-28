"""QSql persistence adapter for durable meeting summaries.

The supplied ``QSqlDatabase`` is caller-owned and must be used only from its
owner thread.  Each mutation is one SQLite transaction; this adapter does not
add locks, workers, or caching.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.meeting.meeting_summary import (
    MeetingSummaryArtifact,
    MeetingSummaryConflictError,
    MeetingSummaryDatabaseError,
    MeetingSummaryDecodeError,
    decode_artifact_created_at,
    decode_uuid,
    encode_artifact_created_at,
    meeting_summary_from_json,
)


# ---------------------------------------------------------------------------
# Repository protocol
# ---------------------------------------------------------------------------


class MeetingSummaryRepository(Protocol):
    """Pure persistence boundary for meeting summary artifacts."""

    def save(self, artifact: MeetingSummaryArtifact) -> None:
        ...

    def load(self, summary_id: uuid.UUID) -> MeetingSummaryArtifact | None:
        ...

    def list_for_meeting(
        self, meeting_id: uuid.UUID
    ) -> tuple[MeetingSummaryArtifact, ...]:
        ...


# ---------------------------------------------------------------------------
# QSql implementation
# ---------------------------------------------------------------------------


class QSqlMeetingSummaryRepository:
    """Persist summary artifacts on a caller-owned QSql connection.

    Caller owns connection, thread, and open/close lifecycle.
    Does not use default QSql connection, addDatabase, setup_app_db,
    close database, cache, or async/thread.
    """

    def __init__(self, database: QSqlDatabase) -> None:
        self._database = database

    def save(self, artifact: MeetingSummaryArtifact) -> None:
        """Insert-only save.  Duplicate summary_id raises ConflictError.

        Validates DB provenance: meeting exists, source generation exists,
        generation.meeting_id matches artifact.meeting_id, and
        generation.profile_version matches artifact.source_profile_version.
        """
        if not isinstance(artifact, MeetingSummaryArtifact):
            raise MeetingSummaryDatabaseError("artifact must be MeetingSummaryArtifact")

        summary_id_str = str(artifact.summary_id)
        meeting_id_str = str(artifact.meeting_id)
        generation_id_str = str(artifact.source_generation_id)

        def write() -> None:
            # Validate meeting exists
            meeting_check = self._execute(
                "SELECT id FROM meeting WHERE id = ?",
                (meeting_id_str,),
            )
            if not meeting_check.next():
                raise MeetingSummaryDatabaseError(f"Meeting {meeting_id_str} not found")

            # Validate source generation exists and belongs to meeting
            gen_check = self._execute(
                "SELECT meeting_id, profile_version "
                "FROM meeting_final_transcription WHERE id = ?",
                (generation_id_str,),
            )
            if not gen_check.next():
                raise MeetingSummaryDatabaseError(
                    f"Source generation {generation_id_str} not found"
                )
            gen_meeting_id = gen_check.value(0)
            gen_profile_version = gen_check.value(1)
            if gen_meeting_id != meeting_id_str:
                raise MeetingSummaryDatabaseError(
                    f"Source generation {generation_id_str} belongs to meeting "
                    f"{gen_meeting_id}, not {meeting_id_str}"
                )
            if gen_profile_version != artifact.source_profile_version:
                raise MeetingSummaryDatabaseError(
                    f"Source generation profile_version {gen_profile_version} "
                    f"does not match artifact {artifact.source_profile_version}"
                )

            # Serialize payload (MeetingSummary ONLY, no provenance)
            from buzz.meeting.meeting_summary import meeting_summary_to_json

            payload_json = meeting_summary_to_json(artifact.summary)
            created_at_str = encode_artifact_created_at(artifact.created_at)

            # Insert
            try:
                self._execute(
                    """
                    INSERT INTO meeting_summary (
                        id, meeting_id, source_generation_id,
                        source_profile_version,
                        source_review_id, source_review_revision,
                        schema_version, prompt_version,
                        created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary_id_str,
                        meeting_id_str,
                        generation_id_str,
                        artifact.source_profile_version,
                        (
                            str(artifact.source_review_id)
                            if artifact.source_review_id is not None
                            else None
                        ),
                        artifact.source_review_revision,
                        artifact.summary.schema_version,
                        artifact.summary.prompt_version,
                        created_at_str,
                        payload_json,
                    ),
                )
            except MeetingSummaryDatabaseError as exc:
                if "UNIQUE constraint" in str(exc):
                    raise MeetingSummaryConflictError(
                        f"Summary artifact {summary_id_str} already exists"
                    ) from exc
                raise
            except Exception as exc:
                error_text = str(exc)
                if "UNIQUE constraint" in error_text:
                    raise MeetingSummaryConflictError(
                        f"Summary artifact {summary_id_str} already exists"
                    ) from exc
                raise MeetingSummaryDatabaseError(
                    f"Could not insert summary: {exc}"
                ) from exc

        self._transaction("save", write)

    def load(self, summary_id: uuid.UUID) -> MeetingSummaryArtifact | None:
        """Load artifact by summary_id, or None if not found.

        Fresh DB read, no cache.  Validates persisted envelope structure
        but does not require current generation/review still present.
        """
        if not isinstance(summary_id, uuid.UUID):
            raise MeetingSummaryDatabaseError("summary_id must be uuid.UUID")

        query = self._execute(
            """
            SELECT id, meeting_id, source_generation_id,
                   source_profile_version,
                   source_review_id, source_review_revision,
                   schema_version, prompt_version,
                   created_at, payload_json
            FROM meeting_summary
            WHERE id = ?
            """,
            (str(summary_id),),
        )
        if not query.next():
            return None
        return self._decode_row(query)

    def list_for_meeting(
        self, meeting_id: uuid.UUID
    ) -> tuple[MeetingSummaryArtifact, ...]:
        """Load all artifacts for a meeting, ordered by created_at ASC, id ASC.

        Fresh DB read, no cache.  If any row is corrupt, raises
        DecodeError for the whole list.
        """
        if not isinstance(meeting_id, uuid.UUID):
            raise MeetingSummaryDatabaseError("meeting_id must be uuid.UUID")

        query = self._execute(
            """
            SELECT id, meeting_id, source_generation_id,
                   source_profile_version,
                   source_review_id, source_review_revision,
                   schema_version, prompt_version,
                   created_at, payload_json
            FROM meeting_summary
            WHERE meeting_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (str(meeting_id),),
        )
        results: list[MeetingSummaryArtifact] = []
        while query.next():
            results.append(self._decode_row(query))
        return tuple(results)

    # -- internals -----------------------------------------------------------

    def _decode_row(self, query: QSqlQuery) -> MeetingSummaryArtifact:
        """Decode a single meeting_summary row into an artifact."""
        summary_id_raw = query.value(0)
        meeting_id_raw = query.value(1)
        generation_id_raw = query.value(2)
        profile_version = query.value(3)
        review_id_raw = self._nullable(query, 4)
        review_revision = self._nullable(query, 5)
        row_schema_version = query.value(6)
        row_prompt_version = query.value(7)
        created_at_raw = query.value(8)
        payload_json_raw = query.value(9)

        # Decode UUIDs
        summary_id = decode_uuid(summary_id_raw, "summary_id")
        meeting_id = decode_uuid(meeting_id_raw, "meeting_id")
        generation_id = decode_uuid(generation_id_raw, "source_generation_id")

        # Decode review pair
        source_review_id: uuid.UUID | None = None
        source_review_revision: int | None = None
        if review_id_raw is not None:
            source_review_id = decode_uuid(review_id_raw, "source_review_id")
            if review_revision is None:
                raise MeetingSummaryDecodeError(
                    "Review pair corruption: source_review_id present but "
                    "source_review_revision is NULL"
                )
            source_review_revision = _decode_int(
                review_revision, "source_review_revision", minimum=0
            )
        elif review_revision is not None:
            raise MeetingSummaryDecodeError(
                "Review pair corruption: source_review_revision present but "
                "source_review_id is NULL"
            )

        # Decode created_at
        created_at = decode_artifact_created_at(created_at_raw)

        # Decode payload_json
        if not isinstance(payload_json_raw, str):
            raise MeetingSummaryDecodeError("payload_json must be text")
        try:
            payload_data = meeting_summary_from_json(payload_json_raw)
        except MeetingSummaryDecodeError:
            raise
        except Exception as exc:
            raise MeetingSummaryDecodeError(f"Corrupt payload_json: {exc}") from exc

        # Mirror check: row schema_version == payload schema_version
        if row_schema_version != payload_data.schema_version:
            raise MeetingSummaryDecodeError(
                f"Row schema_version ({row_schema_version}) does not match "
                f"payload schema_version ({payload_data.schema_version})"
            )
        if row_prompt_version != payload_data.prompt_version:
            raise MeetingSummaryDecodeError(
                f"Row prompt_version ({row_prompt_version}) does not match "
                f"payload prompt_version ({payload_data.prompt_version})"
            )

        return MeetingSummaryArtifact(
            summary_id=summary_id,
            meeting_id=meeting_id,
            source_generation_id=generation_id,
            source_profile_version=_decode_int(
                profile_version, "source_profile_version", minimum=1
            ),
            source_review_id=source_review_id,
            source_review_revision=source_review_revision,
            created_at=created_at,
            summary=payload_data,
        )

    @staticmethod
    def _nullable(query: QSqlQuery, index: int) -> Any:
        return None if query.isNull(index) else query.value(index)

    def _execute(self, sql: str, values: Sequence[Any]) -> QSqlQuery:
        query = QSqlQuery(self._database)
        if not query.prepare(sql):
            raise MeetingSummaryDatabaseError(
                f"Could not prepare SQL: {query.lastError().text()}"
            )
        for value in values:
            query.addBindValue(value)
        if not query.exec():
            raise MeetingSummaryDatabaseError(
                f"Could not execute SQL: {query.lastError().text()}"
            )
        return query

    def _transaction(self, action: str, write: object) -> None:
        if not self._database.transaction():
            raise MeetingSummaryDatabaseError(
                f"Could not begin {action} transaction: "
                f"{self._database.lastError().text()}"
            )
        try:
            write()  # type: ignore[operator]
        except Exception:
            self._database.rollback()
            raise
        if not self._database.commit():
            details = self._database.lastError().text()
            self._database.rollback()
            raise MeetingSummaryDatabaseError(f"Could not commit {action}: {details}")


def _decode_int(raw: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise MeetingSummaryDecodeError(f"Invalid {name}: {raw!r}")
    return raw


__all__ = [
    "MeetingSummaryRepository",
    "QSqlMeetingSummaryRepository",
]
