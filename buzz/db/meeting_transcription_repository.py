"""QSql persistence adapter for final-transcription generations, tracks,
and segments.

Methods must run on the thread that owns the supplied ``QSqlDatabase``.
The adapter is not thread-safe, does not use Qt's default connection,
and never creates, closes, or moves the supplied connection.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional

from PyQt6.QtSql import QSqlDatabase, QSqlQuery

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionStatus,
    FinalTranscriptionTrackStatus,
    GenerationPersistenceRecord,
    SegmentPersistenceRecord,
    TrackPersistenceRecord,
    WordPersistenceRecord,
    decode_track_status,
    encode_config,
    encode_generation_status,
    encode_track_status,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


class QSqlMeetingTranscriptionRepository:
    """Store final-transcription aggregates using a caller-owned QSql
    connection.

    All methods must run on the thread that owns ``database``.
    """

    def __init__(self, database: QSqlDatabase) -> None:
        self._database = database

    def create_generation(
        self,
        generation_id: str,
        meeting_id: str,
        config: FinalTranscriptionConfig,
        initial_status: str,
        time_created: str,
        time_completed: Optional[str],
        tracks: tuple[TrackPersistenceRecord, ...],
    ) -> None:
        """Atomically create generation + track rows."""
        if not self._database.transaction():
            raise self._db_error("Could not begin create_generation transaction")
        try:
            config_fields = encode_config(config)
            self._execute(
                """
                INSERT INTO meeting_final_transcription (
                    id, meeting_id, profile_version, status,
                    config_model_type, config_whisper_model_size,
                    config_hugging_face_model_id, config_language,
                    error_message, time_created, time_started, time_completed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    meeting_id,
                    config.profile_version,
                    initial_status,
                    config_fields["config_model_type"],
                    config_fields["config_whisper_model_size"],
                    config_fields["config_hugging_face_model_id"],
                    config_fields["config_language"],
                    None,
                    time_created,
                    None,
                    time_completed,
                ),
            )
            for track in tracks:
                self._execute(
                    """
                    INSERT INTO meeting_final_transcription_track (
                        generation_id, role, status, error_message,
                        time_started, time_completed, segment_count,
                        word_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        track.role,
                        track.status,
                        track.error_message,
                        track.time_started,
                        track.time_completed,
                        track.segment_count,
                        track.word_count,
                    ),
                )
        except Exception:
            self._raise_after_rollback("create_generation failed")
            return  # unreachable

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit create_generation",
                commit_unknown=True,
            )

    def find_generation_by_key(
        self,
        meeting_id: str,
        profile_version: int,
    ) -> Optional[GenerationPersistenceRecord]:
        """Return existing generation for idempotency check."""
        query = self._execute(
            """
            SELECT id, meeting_id, profile_version, status,
                   config_model_type, config_whisper_model_size,
                   config_hugging_face_model_id, config_language,
                   error_message, time_created, time_started, time_completed
            FROM meeting_final_transcription
            WHERE meeting_id = ? AND profile_version = ?
            """,
            (meeting_id, profile_version),
        )
        if not query.next():
            return None
        return self._row_to_generation(query)

    def load_generation(
        self,
        generation_id: str,
    ) -> Optional[GenerationPersistenceRecord]:
        """Load a single generation by ID."""
        query = self._execute(
            """
            SELECT id, meeting_id, profile_version, status,
                   config_model_type, config_whisper_model_size,
                   config_hugging_face_model_id, config_language,
                   error_message, time_created, time_started, time_completed
            FROM meeting_final_transcription
            WHERE id = ?
            """,
            (generation_id,),
        )
        if not query.next():
            return None
        return self._row_to_generation(query)

    def load_tracks(
        self,
        generation_id: str,
    ) -> tuple[TrackPersistenceRecord, ...]:
        """Load all tracks for a generation."""
        query = self._execute(
            """
            SELECT generation_id, role, status, error_message,
                   time_started, time_completed, segment_count, word_count
            FROM meeting_final_transcription_track
            WHERE generation_id = ?
            ORDER BY CASE role
                WHEN 'MICROPHONE' THEN 0
                WHEN 'REMOTE' THEN 1
                ELSE 2
            END
            """,
            (generation_id,),
        )
        tracks: list[TrackPersistenceRecord] = []
        while query.next():
            tracks.append(self._row_to_track(query))
        return tuple(tracks)

    def load_segments(
        self,
        generation_id: str,
        role: str,
    ) -> tuple[SegmentPersistenceRecord, ...]:
        """Load segments for a specific track."""
        query = self._execute(
            """
            SELECT generation_id, role, ordinal, local_start_ms,
                   local_end_ms, start_ns, end_ns, text
            FROM meeting_final_transcription_segment
            WHERE generation_id = ? AND role = ?
            ORDER BY ordinal
            """,
            (generation_id, role),
        )
        segments: list[SegmentPersistenceRecord] = []
        while query.next():
            segments.append(self._row_to_segment(query))
        return tuple(segments)

    def load_words(
        self,
        generation_id: str,
    ) -> tuple[WordPersistenceRecord, ...]:
        """Load words independently so corrupt provenance remains visible."""
        query = self._execute(
            """
            SELECT generation_id, role, ordinal, segment_ordinal,
                   local_start_ms, local_end_ms, start_ns, end_ns, text
            FROM meeting_final_transcription_word
            WHERE generation_id = ?
            ORDER BY role, ordinal
            """,
            (generation_id,),
        )
        words: list[WordPersistenceRecord] = []
        while query.next():
            words.append(self._row_to_word(query))
        return tuple(words)

    def begin_track(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        """Atomically mark track IN_PROGRESS and generation IN_PROGRESS."""
        if not self._database.transaction():
            raise self._db_error("Could not begin begin_track transaction")
        try:
            self._execute(
                """
                UPDATE meeting_final_transcription_track
                SET status = ?, time_started = ?
                WHERE generation_id = ? AND role = ?
                """,
                (
                    encode_track_status(FinalTranscriptionTrackStatus.IN_PROGRESS),
                    now,
                    generation_id,
                    role,
                ),
            )
            # Update generation: set IN_PROGRESS, time_started if NULL
            self._execute(
                """
                UPDATE meeting_final_transcription
                SET status = ?,
                    time_started = COALESCE(time_started, ?)
                WHERE id = ?
                """,
                (
                    encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
                    now,
                    generation_id,
                ),
            )
        except Exception:
            self._raise_after_rollback("begin_track failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit begin_track", commit_unknown=True
            )

    def complete_track(
        self,
        generation_id: str,
        role: str,
        segments: tuple[SegmentPersistenceRecord, ...],
        now: str,
        words: tuple[WordPersistenceRecord, ...] = (),
    ) -> None:
        """Atomically replace results, mark COMPLETED, derive generation."""
        if not self._database.transaction():
            raise self._db_error("Could not begin complete_track transaction")
        try:
            # Delete words before their parent phrase segments.
            self._execute(
                """
                DELETE FROM meeting_final_transcription_word
                WHERE generation_id = ? AND role = ?
                """,
                (generation_id, role),
            )

            # Delete existing segments for this track
            self._execute(
                """
                DELETE FROM meeting_final_transcription_segment
                WHERE generation_id = ? AND role = ?
                """,
                (generation_id, role),
            )

            # Insert new segments
            for seg in segments:
                self._execute(
                    """
                    INSERT INTO meeting_final_transcription_segment (
                        generation_id, role, ordinal, local_start_ms,
                        local_end_ms, start_ns, end_ns, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        role,
                        seg.ordinal,
                        seg.local_start_ms,
                        seg.local_end_ms,
                        seg.start_ns,
                        seg.end_ns,
                        seg.text,
                    ),
                )

            # Insert backend-native words after their parent segments.
            for word in words:
                self._execute(
                    """
                    INSERT INTO meeting_final_transcription_word (
                        generation_id, role, ordinal, segment_ordinal,
                        local_start_ms, local_end_ms, start_ns, end_ns, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        role,
                        word.ordinal,
                        word.segment_ordinal,
                        word.local_start_ms,
                        word.local_end_ms,
                        word.start_ns,
                        word.end_ns,
                        word.text,
                    ),
                )

            # Mark track COMPLETED
            self._execute(
                """
                UPDATE meeting_final_transcription_track
                SET status = ?, time_completed = ?, segment_count = ?,
                    word_count = ?
                WHERE generation_id = ? AND role = ?
                """,
                (
                    encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                    now,
                    len(segments),
                    len(words),
                    generation_id,
                    role,
                ),
            )

            # Derive and update generation status
            self._update_generation_status(generation_id, now)

        except Exception:
            self._raise_after_rollback("complete_track failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit complete_track", commit_unknown=True
            )

    def fail_track(
        self,
        generation_id: str,
        role: str,
        error_message: str,
        now: str,
    ) -> None:
        """Atomically mark track FAILED, derive generation status."""
        if not self._database.transaction():
            raise self._db_error("Could not begin fail_track transaction")
        try:
            self._execute(
                """
                DELETE FROM meeting_final_transcription_word
                WHERE generation_id = ? AND role = ?
                """,
                (generation_id, role),
            )

            # Delete any partial segments
            self._execute(
                """
                DELETE FROM meeting_final_transcription_segment
                WHERE generation_id = ? AND role = ?
                """,
                (generation_id, role),
            )

            self._execute(
                """
                UPDATE meeting_final_transcription_track
                SET status = ?, error_message = ?, time_completed = ?,
                    segment_count = 0, word_count = 0
                WHERE generation_id = ? AND role = ?
                """,
                (
                    encode_track_status(FinalTranscriptionTrackStatus.FAILED),
                    error_message[:4096],
                    now,
                    generation_id,
                    role,
                ),
            )

            self._update_generation_status(generation_id, now)

        except Exception:
            self._raise_after_rollback("fail_track failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit fail_track", commit_unknown=True
            )

    def mark_track_ineligible(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        """Mark track INELIGIBLE, derive generation status."""
        if not self._database.transaction():
            raise self._db_error("Could not begin mark_track_ineligible transaction")
        try:
            self._execute(
                """
                UPDATE meeting_final_transcription_track
                SET status = ?, time_completed = ?
                WHERE generation_id = ? AND role = ?
                """,
                (
                    encode_track_status(FinalTranscriptionTrackStatus.INELIGIBLE),
                    now,
                    generation_id,
                    role,
                ),
            )
            self._update_generation_status(generation_id, now)

        except Exception:
            self._raise_after_rollback("mark_track_ineligible failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit mark_track_ineligible",
                commit_unknown=True,
            )

    def update_generation_status(
        self,
        generation_id: str,
        now: str,
    ) -> None:
        """Derive and update generation status from current track statuses."""
        if not self._database.transaction():
            raise self._db_error("Could not begin update_generation_status transaction")
        try:
            self._update_generation_status(generation_id, now)
        except Exception:
            self._raise_after_rollback("update_generation_status failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit update_generation_status",
                commit_unknown=True,
            )

    def reset_for_retry(
        self,
        generation_id: str,
        desired_track_statuses: dict[str, str],
        now: str,
    ) -> None:
        """Atomically reset non-COMPLETED tracks to desired states.

        ``desired_track_statuses`` maps role strings to exact desired
        status (QUEUED or INELIGIBLE).  COMPLETED tracks preserved.
        """
        if not self._database.transaction():
            raise self._db_error("Could not begin reset_for_retry transaction")
        try:
            # Delete words for all non-COMPLETED tracks first.
            self._execute(
                """
                DELETE FROM meeting_final_transcription_word
                WHERE generation_id = ? AND role IN (
                    SELECT role FROM meeting_final_transcription_track
                    WHERE generation_id = ? AND status != ?
                )
                """,
                (
                    generation_id,
                    generation_id,
                    encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                ),
            )

            # Delete segments for all non-COMPLETED tracks
            self._execute(
                """
                DELETE FROM meeting_final_transcription_segment
                WHERE generation_id = ? AND role IN (
                    SELECT role FROM meeting_final_transcription_track
                    WHERE generation_id = ? AND status != ?
                )
                """,
                (
                    generation_id,
                    generation_id,
                    encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                ),
            )

            # Apply desired status per non-COMPLETED role
            for role, desired_status in desired_track_statuses.items():
                self._execute(
                    """
                    UPDATE meeting_final_transcription_track
                    SET status = ?, error_message = NULL,
                        time_started = NULL, time_completed = NULL,
                        segment_count = 0, word_count = 0
                    WHERE generation_id = ? AND role = ?
                      AND status != ?
                    """,
                    (
                        desired_status,
                        generation_id,
                        role,
                        encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                    ),
                )

            # Recompute generation status
            self._update_generation_status(generation_id, now)

            # Clear generation error and terminal timestamp if non-terminal
            self._execute(
                """
                UPDATE meeting_final_transcription
                SET error_message = NULL,
                    time_completed = CASE
                        WHEN status IN (?, ?) THEN NULL
                        ELSE time_completed
                    END,
                    time_started = CASE
                        WHEN status = ? THEN NULL
                        ELSE time_started
                    END
                WHERE id = ?
                """,
                (
                    encode_generation_status(FinalTranscriptionStatus.QUEUED),
                    encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
                    encode_generation_status(FinalTranscriptionStatus.QUEUED),
                    generation_id,
                ),
            )

        except Exception:
            self._raise_after_rollback("reset_for_retry failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit reset_for_retry", commit_unknown=True
            )

    def load_recoverable_generations(
        self,
    ) -> tuple[GenerationPersistenceRecord, ...]:
        """Load QUEUED and IN_PROGRESS generations for recovery."""
        query = self._execute(
            """
            SELECT id, meeting_id, profile_version, status,
                   config_model_type, config_whisper_model_size,
                   config_hugging_face_model_id, config_language,
                   error_message, time_created, time_started, time_completed
            FROM meeting_final_transcription
            WHERE status IN (?, ?)
            ORDER BY time_created, id
            """,
            (
                encode_generation_status(FinalTranscriptionStatus.QUEUED),
                encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
            ),
        )
        results: list[GenerationPersistenceRecord] = []
        while query.next():
            results.append(self._row_to_generation(query))
        return tuple(results)

    def reset_in_progress_tracks(
        self,
        generation_id: str,
    ) -> None:
        """Reset IN_PROGRESS tracks to QUEUED for recovery."""
        if not self._database.transaction():
            raise self._db_error("Could not begin reset_in_progress_tracks transaction")
        try:
            self._execute(
                """
                UPDATE meeting_final_transcription_track
                SET status = ?, error_message = NULL,
                    time_started = NULL, time_completed = NULL,
                    segment_count = 0, word_count = 0
                WHERE generation_id = ?
                  AND status = ?
                """,
                (
                    encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    generation_id,
                    encode_track_status(FinalTranscriptionTrackStatus.IN_PROGRESS),
                ),
            )

            # Delete words for tracks just reset to QUEUED.
            self._execute(
                """
                DELETE FROM meeting_final_transcription_word
                WHERE generation_id = ? AND role IN (
                    SELECT role FROM meeting_final_transcription_track
                    WHERE generation_id = ? AND status = ?
                )
                """,
                (
                    generation_id,
                    generation_id,
                    encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                ),
            )

            # Delete segments for reset tracks
            self._execute(
                """
                DELETE FROM meeting_final_transcription_segment
                WHERE generation_id = ? AND role IN (
                    SELECT role FROM meeting_final_transcription_track
                    WHERE generation_id = ? AND status = ?
                )
                """,
                (
                    generation_id,
                    generation_id,
                    encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                ),
            )

            # Recompute generation status
            self._update_generation_status(generation_id, utc_now_iso())

        except Exception:
            self._raise_after_rollback("reset_in_progress_tracks failed")
            return

        if not self._database.commit():
            self._raise_after_rollback(
                "Could not commit reset_in_progress_tracks",
                commit_unknown=True,
            )

    # -- internal helpers ---------------------------------------------------

    def _update_generation_status(
        self,
        generation_id: str,
        now: str,
    ) -> None:
        """Derive generation status from current track statuses.

        Must be called within an active transaction.
        """
        query = self._execute(
            """
            SELECT status FROM meeting_final_transcription_track
            WHERE generation_id = ?
            """,
            (generation_id,),
        )
        track_statuses: list[FinalTranscriptionTrackStatus] = []
        while query.next():
            track_statuses.append(decode_track_status(query.value(0)))

        from buzz.meeting.final_transcription import derive_generation_status

        new_status = derive_generation_status(tuple(track_statuses))
        is_terminal = new_status in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
            FinalTranscriptionStatus.FAILED,
        )

        self._execute(
            """
            UPDATE meeting_final_transcription
            SET status = ?,
                time_completed = CASE WHEN ? THEN ? ELSE time_completed END
            WHERE id = ?
            """,
            (
                encode_generation_status(new_status),
                is_terminal,
                now,
                generation_id,
            ),
        )

    def _row_to_generation(self, query: QSqlQuery) -> GenerationPersistenceRecord:
        return GenerationPersistenceRecord(
            id=query.value(0),
            meeting_id=query.value(1),
            profile_version=query.value(2),
            status=query.value(3),
            config_model_type=query.value(4),
            config_whisper_model_size=self._nullable(query, 5),
            config_hugging_face_model_id=query.value(6) or "",
            config_language=self._nullable(query, 7),
            error_message=self._nullable(query, 8),
            time_created=query.value(9),
            time_started=self._nullable(query, 10),
            time_completed=self._nullable(query, 11),
        )

    def _row_to_track(self, query: QSqlQuery) -> TrackPersistenceRecord:
        return TrackPersistenceRecord(
            generation_id=query.value(0),
            role=query.value(1),
            status=query.value(2),
            error_message=self._nullable(query, 3),
            time_started=self._nullable(query, 4),
            time_completed=self._nullable(query, 5),
            segment_count=query.value(6),
            word_count=query.value(7),
        )

    def _row_to_segment(self, query: QSqlQuery) -> SegmentPersistenceRecord:
        return SegmentPersistenceRecord(
            generation_id=query.value(0),
            role=query.value(1),
            ordinal=query.value(2),
            local_start_ms=query.value(3),
            local_end_ms=query.value(4),
            start_ns=query.value(5),
            end_ns=query.value(6),
            text=query.value(7),
        )

    def _row_to_word(self, query: QSqlQuery) -> WordPersistenceRecord:
        return WordPersistenceRecord(
            generation_id=query.value(0),
            role=query.value(1),
            ordinal=query.value(2),
            segment_ordinal=query.value(3),
            local_start_ms=query.value(4),
            local_end_ms=query.value(5),
            start_ns=query.value(6),
            end_ns=query.value(7),
            text=query.value(8),
        )

    @staticmethod
    def _nullable(query: QSqlQuery, index: int) -> Optional[str]:
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

    def _raise_after_rollback(
        self,
        message: str,
        *,
        commit_unknown: bool = False,
    ) -> None:
        try:
            self._database.rollback()
        except Exception as rb_exc:
            logger.error("Rollback also failed: %s", rb_exc)
        raise self._db_error(message, commit_unknown=commit_unknown)

    def _query_error(self, message: str, query: QSqlQuery) -> Exception:
        details = query.lastError().text()
        return Exception(f"{message}: {details}")

    def _db_error(
        self,
        message: str,
        *,
        commit_unknown: bool = False,
    ) -> Exception:
        details = self._database.lastError().text()
        return Exception(f"{message}: {details}")


__all__ = ["QSqlMeetingTranscriptionRepository"]
