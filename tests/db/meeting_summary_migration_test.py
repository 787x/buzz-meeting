"""Additive migration from the explicit PR17 schema to PR18."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from buzz.db.migrator import dumb_migrate_db


PR17_SCHEMA = """
CREATE TABLE transcription (
    id TEXT PRIMARY KEY,
    error_message TEXT,
    export_formats TEXT,
    file TEXT,
    output_folder TEXT,
    progress DOUBLE PRECISION DEFAULT 0.0,
    language TEXT,
    model_type TEXT,
    source TEXT,
    status TEXT,
    task TEXT,
    time_ended TIMESTAMP,
    time_queued TIMESTAMP NOT NULL,
    time_started TIMESTAMP,
    url TEXT,
    whisper_model_size TEXT,
    hugging_face_model_id TEXT,
    word_level_timings BOOLEAN DEFAULT FALSE,
    extract_speech BOOLEAN DEFAULT FALSE,
    name TEXT,
    notes TEXT
);

CREATE TABLE transcription_segment (
    id INTEGER PRIMARY KEY,
    end_time INT DEFAULT 0,
    start_time INT DEFAULT 0,
    text TEXT NOT NULL,
    translation TEXT DEFAULT '',
    transcription_id TEXT,
    FOREIGN KEY (transcription_id) REFERENCES transcription(id) ON DELETE CASCADE
);
CREATE INDEX idx_transcription_id ON transcription_segment(transcription_id);

CREATE TABLE meeting (
    id TEXT PRIMARY KEY NOT NULL,
    remote_source_kind TEXT NOT NULL,
    session_state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    audio_state TEXT NOT NULL,
    audio_outcome TEXT
);

CREATE TABLE meeting_audio_track (
    meeting_id TEXT NOT NULL,
    role TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
    recording_state TEXT NOT NULL,
    published INTEGER NOT NULL CHECK (published IN (0, 1)),
    complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
    timing_basis TEXT NOT NULL,
    PRIMARY KEY (meeting_id, role),
    UNIQUE (meeting_id, relative_path),
    FOREIGN KEY (meeting_id) REFERENCES meeting(id) ON DELETE CASCADE
);

CREATE TABLE meeting_audio_timing_anchor (
    meeting_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    sample_end INTEGER NOT NULL CHECK (sample_end > 0),
    callback_arrival_offset_ns INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, role, ordinal),
    FOREIGN KEY (meeting_id, role)
        REFERENCES meeting_audio_track(meeting_id, role) ON DELETE CASCADE
);

CREATE TABLE meeting_audio_error (
    meeting_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    stage TEXT NOT NULL,
    exception_module TEXT NOT NULL,
    exception_name TEXT NOT NULL,
    message TEXT NOT NULL CHECK (length(message) <= 4096),
    PRIMARY KEY (meeting_id, role, ordinal),
    FOREIGN KEY (meeting_id, role)
        REFERENCES meeting_audio_track(meeting_id, role) ON DELETE CASCADE
);

CREATE TABLE meeting_final_transcription (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL CHECK (profile_version > 0),
    status TEXT NOT NULL,
    config_model_type TEXT NOT NULL,
    config_whisper_model_size TEXT,
    config_hugging_face_model_id TEXT NOT NULL DEFAULT '',
    config_language TEXT,
    error_message TEXT CHECK (error_message IS NULL OR length(error_message) <= 4096),
    time_created TEXT NOT NULL,
    time_started TEXT,
    time_completed TEXT,
    UNIQUE (meeting_id, profile_version),
    FOREIGN KEY (meeting_id) REFERENCES meeting(id) ON DELETE CASCADE
);

CREATE TABLE meeting_final_transcription_track (
    generation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT CHECK (error_message IS NULL OR length(error_message) <= 4096),
    time_started TEXT,
    time_completed TEXT,
    segment_count INTEGER NOT NULL DEFAULT 0 CHECK (segment_count >= 0),
    word_count INTEGER NOT NULL DEFAULT 0 CHECK (word_count >= 0),
    PRIMARY KEY (generation_id, role),
    FOREIGN KEY (generation_id)
        REFERENCES meeting_final_transcription(id) ON DELETE CASCADE
);

CREATE TABLE meeting_final_transcription_segment (
    generation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    local_start_ms INTEGER NOT NULL CHECK (local_start_ms >= 0),
    local_end_ms INTEGER NOT NULL,
    start_ns INTEGER NOT NULL,
    end_ns INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (generation_id, role, ordinal),
    FOREIGN KEY (generation_id, role)
        REFERENCES meeting_final_transcription_track(generation_id, role)
        ON DELETE CASCADE,
    CHECK (local_end_ms >= local_start_ms),
    CHECK (end_ns >= start_ns)
);

CREATE TABLE meeting_final_transcription_word (
    generation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    segment_ordinal INTEGER NOT NULL CHECK (segment_ordinal >= 0),
    local_start_ms INTEGER NOT NULL CHECK (local_start_ms >= 0),
    local_end_ms INTEGER NOT NULL,
    start_ns INTEGER NOT NULL,
    end_ns INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (generation_id, role, ordinal),
    FOREIGN KEY (generation_id, role)
        REFERENCES meeting_final_transcription_track(generation_id, role)
        ON DELETE CASCADE,
    FOREIGN KEY (generation_id, role, segment_ordinal)
        REFERENCES meeting_final_transcription_segment(generation_id, role, ordinal)
        ON DELETE CASCADE,
    CHECK (local_end_ms >= local_start_ms),
    CHECK (end_ns >= start_ns)
);

CREATE TABLE meeting_speaker_review (
    id TEXT PRIMARY KEY NOT NULL,
    source_generation_id TEXT NOT NULL,
    source_profile_version INTEGER NOT NULL
        CHECK (source_profile_version > 0),
    source_track_count INTEGER NOT NULL CHECK (source_track_count >= 0),
    mapping_algorithm_version INTEGER NOT NULL
        CHECK (mapping_algorithm_version > 0),
    status TEXT NOT NULL
        CHECK (status IN ('UNREVIEWED', 'IN_PROGRESS', 'COMPLETED')),
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    next_speaker_ordinal INTEGER NOT NULL DEFAULT 0
        CHECK (next_speaker_ordinal >= 0),
    time_created TEXT NOT NULL,
    time_updated TEXT NOT NULL,
    time_completed TEXT,
    UNIQUE (source_generation_id),
    UNIQUE (id, source_generation_id),
    FOREIGN KEY (source_generation_id)
        REFERENCES meeting_final_transcription(id)
        ON DELETE CASCADE,
    CHECK (time_updated >= time_created),
    CHECK (
        (
            status = 'UNREVIEWED'
            AND revision = 0
            AND time_updated = time_created
            AND time_completed IS NULL
        )
        OR (
            status = 'IN_PROGRESS'
            AND revision >= 1
            AND time_completed IS NULL
        )
        OR (
            status = 'COMPLETED'
            AND revision >= 1
            AND time_completed IS NOT NULL
            AND time_completed = time_updated
        )
    )
);

CREATE TABLE meeting_speaker_review_track (
    review_id TEXT NOT NULL,
    source_generation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('MICROPHONE', 'REMOTE')),
    source_track_status TEXT NOT NULL
        CHECK (
            source_track_status IN (
                'QUEUED', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'INELIGIBLE'
            )
        ),
    source_word_count INTEGER NOT NULL CHECK (source_word_count >= 0),
    analysis_state TEXT NOT NULL
        CHECK (analysis_state IN ('NOT_PROVIDED', 'COMPLETED')),
    turn_count INTEGER NOT NULL CHECK (turn_count >= 0),
    diarization_backend TEXT,
    diarization_profile_version INTEGER,
    PRIMARY KEY (review_id, role),
    FOREIGN KEY (review_id, source_generation_id)
        REFERENCES meeting_speaker_review(id, source_generation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (source_generation_id, role)
        REFERENCES meeting_final_transcription_track(generation_id, role)
        ON DELETE CASCADE,
    CHECK (
        (
            analysis_state = 'NOT_PROVIDED'
            AND turn_count = 0
            AND diarization_backend IS NULL
            AND diarization_profile_version IS NULL
        )
        OR (
            analysis_state = 'COMPLETED'
            AND diarization_backend IS NOT NULL
            AND diarization_profile_version IS NOT NULL
            AND diarization_profile_version > 0
        )
    )
);

CREATE TABLE meeting_speaker_turn (
    review_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    speaker_index INTEGER NOT NULL CHECK (speaker_index >= 0),
    local_start_ms INTEGER NOT NULL CHECK (local_start_ms >= 0),
    local_end_ms INTEGER NOT NULL,
    PRIMARY KEY (review_id, role, ordinal),
    FOREIGN KEY (review_id, role)
        REFERENCES meeting_speaker_review_track(review_id, role)
        ON DELETE CASCADE,
    CHECK (local_end_ms >= local_start_ms)
);

CREATE TABLE meeting_speaker_cluster (
    review_id TEXT NOT NULL,
    role TEXT NOT NULL,
    speaker_index INTEGER NOT NULL CHECK (speaker_index >= 0),
    PRIMARY KEY (review_id, role, speaker_index),
    FOREIGN KEY (review_id, role)
        REFERENCES meeting_speaker_review_track(review_id, role)
        ON DELETE CASCADE
);

CREATE TABLE meeting_reviewed_speaker (
    review_id TEXT NOT NULL,
    id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    display_name TEXT
        CHECK (
            display_name IS NULL
            OR (length(display_name) >= 1 AND length(display_name) <= 256)
        ),
    PRIMARY KEY (review_id, id),
    UNIQUE (id),
    UNIQUE (review_id, ordinal),
    FOREIGN KEY (review_id)
        REFERENCES meeting_speaker_review(id)
        ON DELETE CASCADE
);

CREATE TABLE meeting_speaker_cluster_assignment (
    review_id TEXT NOT NULL,
    role TEXT NOT NULL,
    speaker_index INTEGER NOT NULL,
    reviewed_speaker_id TEXT NOT NULL,
    PRIMARY KEY (review_id, role, speaker_index),
    FOREIGN KEY (review_id, role, speaker_index)
        REFERENCES meeting_speaker_cluster(review_id, role, speaker_index)
        ON DELETE CASCADE,
    FOREIGN KEY (review_id, reviewed_speaker_id)
        REFERENCES meeting_reviewed_speaker(review_id, id)
        ON DELETE NO ACTION
);

CREATE TABLE meeting_speaker_word_attribution (
    review_id TEXT NOT NULL,
    source_generation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    word_ordinal INTEGER NOT NULL CHECK (word_ordinal >= 0),
    attribution_status TEXT NOT NULL
        CHECK (
            attribution_status IN ('ASSIGNED', 'NO_OVERLAP', 'AMBIGUOUS')
        ),
    machine_speaker_index INTEGER,
    PRIMARY KEY (review_id, role, word_ordinal),
    FOREIGN KEY (review_id, source_generation_id)
        REFERENCES meeting_speaker_review(id, source_generation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (review_id, role)
        REFERENCES meeting_speaker_review_track(review_id, role)
        ON DELETE CASCADE,
    FOREIGN KEY (source_generation_id, role, word_ordinal)
        REFERENCES meeting_final_transcription_word(generation_id, role, ordinal)
        ON DELETE CASCADE,
    FOREIGN KEY (review_id, role, machine_speaker_index)
        REFERENCES meeting_speaker_cluster(review_id, role, speaker_index),
    CHECK (
        (
            attribution_status = 'ASSIGNED'
            AND machine_speaker_index IS NOT NULL
        )
        OR (
            attribution_status IN ('NO_OVERLAP', 'AMBIGUOUS')
            AND machine_speaker_index IS NULL
        )
    )
);

CREATE TABLE meeting_speaker_word_override (
    review_id TEXT NOT NULL,
    role TEXT NOT NULL,
    word_ordinal INTEGER NOT NULL,
    reviewed_speaker_id TEXT,
    PRIMARY KEY (review_id, role, word_ordinal),
    FOREIGN KEY (review_id, role, word_ordinal)
        REFERENCES meeting_speaker_word_attribution(review_id, role, word_ordinal)
        ON DELETE CASCADE,
    FOREIGN KEY (review_id, reviewed_speaker_id)
        REFERENCES meeting_reviewed_speaker(review_id, id)
        ON DELETE NO ACTION
);
"""


NEW_TABLE = "meeting_summary"
NEW_INDEXES = (
    "idx_meeting_summary_meeting_created",
    "idx_meeting_summary_source_generation",
)


def test_pr17_to_pr18_migration_is_additive_idempotent_and_fk_clean(tmp_path):
    database = sqlite3.connect(tmp_path / "pr17.sqlite")
    database.execute("PRAGMA foreign_keys = ON")
    database.executescript(PR17_SCHEMA)

    # Insert representative PR17 data
    meeting_id = "00000000-0000-0000-0000-000000000101"
    generation_id = "00000000-0000-0000-0000-000000000102"
    database.execute(
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
    database.execute(
        """
        INSERT INTO meeting_audio_track (
            meeting_id, role, relative_path, sample_rate, sample_count,
            recording_state, published, complete, timing_basis
        ) VALUES (?, 'MICROPHONE', 'meeting/mic.wav', 16000, 16000,
                  'STOPPED', 1, 1, 'host_callback_arrival')
        """,
        (meeting_id,),
    )
    database.execute(
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
            generation_id,
            meeting_id,
            "2026-01-01T00:00:03+00:00",
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
        ),
    )
    database.execute(
        """
        INSERT INTO meeting_final_transcription_track (
            generation_id, role, status, error_message, time_started,
            time_completed, segment_count, word_count
        ) VALUES (?, 'MICROPHONE', 'COMPLETED', NULL, ?, ?, 1, 1)
        """,
        (
            generation_id,
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
        ),
    )
    database.execute(
        """
        INSERT INTO meeting_final_transcription_segment (
            generation_id, role, ordinal, local_start_ms, local_end_ms,
            start_ns, end_ns, text
        ) VALUES (?, 'MICROPHONE', 0, 0, 500, 0, 500000000, 'hello')
        """,
        (generation_id,),
    )
    database.execute(
        """
        INSERT INTO meeting_final_transcription_word (
            generation_id, role, ordinal, segment_ordinal, local_start_ms,
            local_end_ms, start_ns, end_ns, text
        ) VALUES (?, 'MICROPHONE', 0, 0, 0, 400, 0, 400000000, 'hello')
        """,
        (generation_id,),
    )
    review_id = "00000000-0000-0000-0000-000000000103"
    database.execute(
        """
        INSERT INTO meeting_speaker_review (
            id, source_generation_id, source_profile_version,
            source_track_count, mapping_algorithm_version,
            status, revision, next_speaker_ordinal,
            time_created, time_updated, time_completed
        ) VALUES (?, ?, 2, 1, 1, 'UNREVIEWED', 0, 0,
                  '2026-01-01T00:00:06+00:00',
                  '2026-01-01T00:00:06+00:00', NULL)
        """,
        (review_id, generation_id),
    )
    database.commit()

    # Snapshot all old rows
    old_rows = {
        table: database.execute(f"SELECT * FROM {table}").fetchall()
        for table in (
            "meeting",
            "meeting_audio_track",
            "meeting_final_transcription",
            "meeting_final_transcription_track",
            "meeting_final_transcription_segment",
            "meeting_final_transcription_word",
            "meeting_speaker_review",
        )
    }

    latest_schema = Path("buzz/schema.sql").read_text()

    # Migrate
    assert dumb_migrate_db(database, latest_schema)

    # All old rows unchanged
    for table, expected in old_rows.items():
        assert database.execute(f"SELECT * FROM {table}").fetchall() == expected

    # meeting_summary table exists
    existing_tables = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert NEW_TABLE in existing_tables

    # Both indexes exist
    existing_indexes = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    for idx in NEW_INDEXES:
        assert idx in existing_indexes, f"Missing index: {idx}"

    # meeting_summary initially empty
    assert database.execute("SELECT COUNT(*) FROM meeting_summary").fetchone()[0] == 0

    # foreign_key_check clean
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []

    # Second migration is a no-op
    assert not dumb_migrate_db(database, latest_schema)
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []

    database.close()
