"""QSql persistence adapter for durable meeting speaker reviews.

The supplied ``QSqlDatabase`` is caller-owned and must be used only from its
owner thread.  Each mutation is one SQLite transaction; this adapter does not
add locks, workers, or caching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.meeting.speaker_review import (
    ReviewedSpeakerRecord,
    SpeakerClusterAssignmentRecord,
    SpeakerReviewConflictError,
    SpeakerReviewError,
    SpeakerReviewHeaderRecord,
    SpeakerReviewMutationRecord,
    SpeakerReviewPersistenceBundle,
    SpeakerReviewTrackRecord,
    SpeakerReviewTurnRecord,
    SpeakerWordAttributionRecord,
    SpeakerWordOverrideRecord,
    SpeakerReviewClusterRecord,
)


class QSqlMeetingSpeakerRepository:
    """Persist speaker-review aggregates on a caller-owned QSql connection."""

    def __init__(self, database: QSqlDatabase) -> None:
        self._database = database

    def create_review(self, bundle: SpeakerReviewPersistenceBundle) -> None:
        if (
            self._review_id_for_generation(bundle.header.source_generation_id)
            is not None
        ):
            raise SpeakerReviewConflictError(
                "A canonical speaker review already exists for source generation"
            )

        def write() -> None:
            header = bundle.header
            self._execute(
                """
                INSERT INTO meeting_speaker_review (
                    id, source_generation_id, source_profile_version,
                    source_track_count, mapping_algorithm_version, status, revision,
                    next_speaker_ordinal, time_created, time_updated,
                    time_completed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    header.id,
                    header.source_generation_id,
                    header.source_profile_version,
                    header.source_track_count,
                    header.mapping_algorithm_version,
                    header.status,
                    header.revision,
                    header.next_speaker_ordinal,
                    header.time_created,
                    header.time_updated,
                    header.time_completed,
                ),
            )
            for record in bundle.tracks:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_review_track (
                        review_id, source_generation_id, role,
                        source_track_status, source_word_count, analysis_state,
                        turn_count, diarization_backend,
                        diarization_profile_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.review_id,
                        record.source_generation_id,
                        record.role,
                        record.source_track_status,
                        record.source_word_count,
                        record.analysis_state,
                        record.turn_count,
                        record.diarization_backend,
                        record.diarization_profile_version,
                    ),
                )
            for record in bundle.turns:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_turn (
                        review_id, role, ordinal, speaker_index,
                        local_start_ms, local_end_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.review_id,
                        record.role,
                        record.ordinal,
                        record.speaker_index,
                        record.local_start_ms,
                        record.local_end_ms,
                    ),
                )
            for record in bundle.clusters:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_cluster (
                        review_id, role, speaker_index
                    ) VALUES (?, ?, ?)
                    """,
                    (record.review_id, record.role, record.speaker_index),
                )
            for record in bundle.speakers:
                self._insert_speaker(record)
            for record in bundle.assignments:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_cluster_assignment (
                        review_id, role, speaker_index, reviewed_speaker_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.review_id,
                        record.role,
                        record.speaker_index,
                        record.reviewed_speaker_id,
                    ),
                )
            for record in bundle.attributions:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_word_attribution (
                        review_id, source_generation_id, role, word_ordinal,
                        attribution_status, machine_speaker_index
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.review_id,
                        record.source_generation_id,
                        record.role,
                        record.word_ordinal,
                        record.attribution_status,
                        record.machine_speaker_index,
                    ),
                )
            for record in bundle.overrides:
                self._execute(
                    """
                    INSERT INTO meeting_speaker_word_override (
                        review_id, role, word_ordinal, reviewed_speaker_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.review_id,
                        record.role,
                        record.word_ordinal,
                        record.reviewed_speaker_id,
                    ),
                )

        self._transaction("create_review", write)

    def load_review(self, review_id: str) -> SpeakerReviewPersistenceBundle | None:
        header = self._load_header("id = ?", (review_id,))
        if header is None:
            return None
        return self._load_bundle(header)

    def load_review_for_generation(
        self, generation_id: str
    ) -> SpeakerReviewPersistenceBundle | None:
        header = self._load_header("source_generation_id = ?", (generation_id,))
        if header is None:
            return None
        return self._load_bundle(header)

    def rename_speaker(
        self,
        review_id: str,
        speaker_id: str,
        display_name: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        def write() -> None:
            query = self._execute(
                """
                UPDATE meeting_reviewed_speaker
                SET display_name = ?
                WHERE review_id = ? AND id = ?
                """,
                (display_name, review_id, speaker_id),
            )
            self._expect_one(query, "rename reviewed speaker")
            self._update_header(review_id, mutation)

        self._transaction("rename_speaker", write)

    def create_speaker(
        self,
        speaker: ReviewedSpeakerRecord,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        def write() -> None:
            self._insert_speaker(speaker)
            self._update_header(speaker.review_id, mutation)

        self._transaction("create_speaker", write)

    def merge_speakers(
        self,
        review_id: str,
        source_speaker_id: str,
        target_speaker_id: str,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        def write() -> None:
            self._execute(
                """
                UPDATE meeting_speaker_cluster_assignment
                SET reviewed_speaker_id = ?
                WHERE review_id = ? AND reviewed_speaker_id = ?
                """,
                (target_speaker_id, review_id, source_speaker_id),
            )
            self._execute(
                """
                UPDATE meeting_speaker_word_override
                SET reviewed_speaker_id = ?
                WHERE review_id = ? AND reviewed_speaker_id = ?
                """,
                (target_speaker_id, review_id, source_speaker_id),
            )
            deleted = self._execute(
                """
                DELETE FROM meeting_reviewed_speaker
                WHERE review_id = ? AND id = ?
                """,
                (review_id, source_speaker_id),
            )
            self._expect_one(deleted, "delete merged source speaker")
            self._update_header(review_id, mutation)

        self._transaction("merge_speakers", write)

    def set_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        reviewed_speaker_id: str | None,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        def write() -> None:
            self._execute(
                """
                INSERT INTO meeting_speaker_word_override (
                    review_id, role, word_ordinal, reviewed_speaker_id
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (review_id, role, word_ordinal)
                DO UPDATE SET reviewed_speaker_id = excluded.reviewed_speaker_id
                """,
                (review_id, role, word_ordinal, reviewed_speaker_id),
            )
            self._update_header(review_id, mutation)

        self._transaction("set_word_override", write)

    def clear_word_override(
        self,
        review_id: str,
        role: str,
        word_ordinal: int,
        mutation: SpeakerReviewMutationRecord,
    ) -> None:
        def write() -> None:
            deleted = self._execute(
                """
                DELETE FROM meeting_speaker_word_override
                WHERE review_id = ? AND role = ? AND word_ordinal = ?
                """,
                (review_id, role, word_ordinal),
            )
            self._expect_one(deleted, "clear word override")
            self._update_header(review_id, mutation)

        self._transaction("clear_word_override", write)

    def mark_completed(
        self, review_id: str, mutation: SpeakerReviewMutationRecord
    ) -> None:
        self._transaction(
            "mark_completed", lambda: self._update_header(review_id, mutation)
        )

    def delete_review_for_generation(self, generation_id: str) -> None:
        self._transaction(
            "delete_review_for_generation",
            lambda: self._execute(
                "DELETE FROM meeting_speaker_review WHERE source_generation_id = ?",
                (generation_id,),
            ),
        )

    def _load_bundle(
        self, header: SpeakerReviewHeaderRecord
    ) -> SpeakerReviewPersistenceBundle:
        review_id = header.id
        tracks_query = self._execute(
            """
            SELECT review_id, source_generation_id, role, source_track_status,
                   source_word_count, analysis_state, turn_count,
                   diarization_backend, diarization_profile_version
            FROM meeting_speaker_review_track
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END
            """,
            (review_id,),
        )
        tracks: list[SpeakerReviewTrackRecord] = []
        while tracks_query.next():
            tracks.append(
                SpeakerReviewTrackRecord(
                    review_id=tracks_query.value(0),
                    source_generation_id=tracks_query.value(1),
                    role=tracks_query.value(2),
                    source_track_status=tracks_query.value(3),
                    source_word_count=tracks_query.value(4),
                    analysis_state=tracks_query.value(5),
                    turn_count=tracks_query.value(6),
                    diarization_backend=self._nullable(tracks_query, 7),
                    diarization_profile_version=self._nullable(tracks_query, 8),
                )
            )

        turns_query = self._execute(
            """
            SELECT review_id, role, ordinal, speaker_index,
                   local_start_ms, local_end_ms
            FROM meeting_speaker_turn
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END,
                     ordinal
            """,
            (review_id,),
        )
        turns: list[SpeakerReviewTurnRecord] = []
        while turns_query.next():
            turns.append(
                SpeakerReviewTurnRecord(
                    review_id=turns_query.value(0),
                    role=turns_query.value(1),
                    ordinal=turns_query.value(2),
                    speaker_index=turns_query.value(3),
                    local_start_ms=turns_query.value(4),
                    local_end_ms=turns_query.value(5),
                )
            )

        clusters_query = self._execute(
            """
            SELECT review_id, role, speaker_index
            FROM meeting_speaker_cluster
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END,
                     speaker_index
            """,
            (review_id,),
        )
        clusters: list[SpeakerReviewClusterRecord] = []
        while clusters_query.next():
            clusters.append(
                SpeakerReviewClusterRecord(
                    review_id=clusters_query.value(0),
                    role=clusters_query.value(1),
                    speaker_index=clusters_query.value(2),
                )
            )

        speakers_query = self._execute(
            """
            SELECT review_id, id, ordinal, display_name
            FROM meeting_reviewed_speaker
            WHERE review_id = ?
            ORDER BY ordinal
            """,
            (review_id,),
        )
        speakers: list[ReviewedSpeakerRecord] = []
        while speakers_query.next():
            speakers.append(
                ReviewedSpeakerRecord(
                    review_id=speakers_query.value(0),
                    id=speakers_query.value(1),
                    ordinal=speakers_query.value(2),
                    display_name=self._nullable(speakers_query, 3),
                )
            )

        assignments_query = self._execute(
            """
            SELECT review_id, role, speaker_index, reviewed_speaker_id
            FROM meeting_speaker_cluster_assignment
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END,
                     speaker_index
            """,
            (review_id,),
        )
        assignments: list[SpeakerClusterAssignmentRecord] = []
        while assignments_query.next():
            assignments.append(
                SpeakerClusterAssignmentRecord(
                    review_id=assignments_query.value(0),
                    role=assignments_query.value(1),
                    speaker_index=assignments_query.value(2),
                    reviewed_speaker_id=assignments_query.value(3),
                )
            )

        attributions_query = self._execute(
            """
            SELECT review_id, source_generation_id, role, word_ordinal,
                   attribution_status, machine_speaker_index
            FROM meeting_speaker_word_attribution
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END,
                     word_ordinal
            """,
            (review_id,),
        )
        attributions: list[SpeakerWordAttributionRecord] = []
        while attributions_query.next():
            attributions.append(
                SpeakerWordAttributionRecord(
                    review_id=attributions_query.value(0),
                    source_generation_id=attributions_query.value(1),
                    role=attributions_query.value(2),
                    word_ordinal=attributions_query.value(3),
                    attribution_status=attributions_query.value(4),
                    machine_speaker_index=self._nullable(attributions_query, 5),
                )
            )

        overrides_query = self._execute(
            """
            SELECT review_id, role, word_ordinal, reviewed_speaker_id
            FROM meeting_speaker_word_override
            WHERE review_id = ?
            ORDER BY CASE role WHEN 'MICROPHONE' THEN 0 WHEN 'REMOTE' THEN 1 ELSE 2 END,
                     word_ordinal
            """,
            (review_id,),
        )
        overrides: list[SpeakerWordOverrideRecord] = []
        while overrides_query.next():
            overrides.append(
                SpeakerWordOverrideRecord(
                    review_id=overrides_query.value(0),
                    role=overrides_query.value(1),
                    word_ordinal=overrides_query.value(2),
                    reviewed_speaker_id=self._nullable(overrides_query, 3),
                )
            )

        return SpeakerReviewPersistenceBundle(
            header=header,
            tracks=tuple(tracks),
            turns=tuple(turns),
            clusters=tuple(clusters),
            speakers=tuple(speakers),
            assignments=tuple(assignments),
            attributions=tuple(attributions),
            overrides=tuple(overrides),
        )

    def _load_header(
        self, where_clause: str, values: Sequence[Any]
    ) -> SpeakerReviewHeaderRecord | None:
        query = self._execute(
            f"""
            SELECT id, source_generation_id, source_profile_version,
                   source_track_count, mapping_algorithm_version, status, revision,
                   next_speaker_ordinal, time_created, time_updated,
                   time_completed
            FROM meeting_speaker_review
            WHERE {where_clause}
            """,
            values,
        )
        if not query.next():
            return None
        return SpeakerReviewHeaderRecord(
            id=query.value(0),
            source_generation_id=query.value(1),
            source_profile_version=query.value(2),
            source_track_count=query.value(3),
            mapping_algorithm_version=query.value(4),
            status=query.value(5),
            revision=query.value(6),
            next_speaker_ordinal=query.value(7),
            time_created=query.value(8),
            time_updated=query.value(9),
            time_completed=self._nullable(query, 10),
        )

    def _review_id_for_generation(self, generation_id: str) -> str | None:
        query = self._execute(
            "SELECT id FROM meeting_speaker_review WHERE source_generation_id = ?",
            (generation_id,),
        )
        return query.value(0) if query.next() else None

    def _insert_speaker(self, record: ReviewedSpeakerRecord) -> None:
        self._execute(
            """
            INSERT INTO meeting_reviewed_speaker (
                review_id, id, ordinal, display_name
            ) VALUES (?, ?, ?, ?)
            """,
            (record.review_id, record.id, record.ordinal, record.display_name),
        )

    def _update_header(
        self, review_id: str, mutation: SpeakerReviewMutationRecord
    ) -> None:
        query = self._execute(
            """
            UPDATE meeting_speaker_review
            SET status = ?, revision = ?, next_speaker_ordinal = ?,
                time_updated = ?, time_completed = ?
            WHERE id = ?
            """,
            (
                mutation.status,
                mutation.revision,
                mutation.next_speaker_ordinal,
                mutation.time_updated,
                mutation.time_completed,
                review_id,
            ),
        )
        self._expect_one(query, "update speaker-review header")

    def _transaction(self, action: str, write: Callable[[], object]) -> None:
        if not self._database.transaction():
            raise self._database_error(f"Could not begin {action} transaction")
        try:
            write()
        except Exception:
            self._database.rollback()
            raise
        if not self._database.commit():
            details = self._database.lastError().text()
            self._database.rollback()
            raise SpeakerReviewError(f"Could not commit {action}: {details}")

    @staticmethod
    def _expect_one(query: QSqlQuery, action: str) -> None:
        if query.numRowsAffected() != 1:
            raise SpeakerReviewError(
                f"Could not {action}: affected {query.numRowsAffected()} rows"
            )

    @staticmethod
    def _nullable(query: QSqlQuery, index: int):
        return None if query.isNull(index) else query.value(index)

    def _execute(self, sql: str, values: Sequence[Any]) -> QSqlQuery:
        query = QSqlQuery(self._database)
        if not query.prepare(sql):
            raise self._query_error("Could not prepare SQL", query)
        for value in values:
            query.addBindValue(value)
        if not query.exec():
            raise self._query_error("Could not execute SQL", query)
        return query

    @staticmethod
    def _query_error(message: str, query: QSqlQuery) -> SpeakerReviewError:
        return SpeakerReviewError(f"{message}: {query.lastError().text()}")

    def _database_error(self, message: str) -> SpeakerReviewError:
        return SpeakerReviewError(f"{message}: {self._database.lastError().text()}")


__all__ = ["QSqlMeetingSpeakerRepository"]
