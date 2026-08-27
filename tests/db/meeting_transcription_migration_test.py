"""Tests for PR11 schema migration from PR10 schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from buzz.db.migrator import dumb_migrate_db


# Schema as it was at PR10 (before PR11 changes)
PR10_SCHEMA = """
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
    duration_ns INTEGER
        CHECK (duration_ns IS NULL OR duration_ns >= 0),
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
    FOREIGN KEY (meeting_id)
        REFERENCES meeting(id)
        ON DELETE CASCADE
);

CREATE TABLE meeting_audio_timing_anchor (
    meeting_id TEXT NOT NULL,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    sample_end INTEGER NOT NULL CHECK (sample_end > 0),
    callback_arrival_offset_ns INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, role, ordinal),
    FOREIGN KEY (meeting_id, role)
        REFERENCES meeting_audio_track(meeting_id, role)
        ON DELETE CASCADE
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
        REFERENCES meeting_audio_track(meeting_id, role)
        ON DELETE CASCADE
);
"""


PR11_SCHEMA = (
    PR10_SCHEMA
    + """
CREATE TABLE meeting_final_transcription (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL CHECK (profile_version > 0),
    status TEXT NOT NULL,
    config_model_type TEXT NOT NULL,
    config_whisper_model_size TEXT,
    config_hugging_face_model_id TEXT NOT NULL DEFAULT '',
    config_language TEXT,
    error_message TEXT CHECK (
        error_message IS NULL OR length(error_message) <= 4096
    ),
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
    error_message TEXT CHECK (
        error_message IS NULL OR length(error_message) <= 4096
    ),
    time_started TEXT,
    time_completed TEXT,
    segment_count INTEGER NOT NULL DEFAULT 0 CHECK (segment_count >= 0),
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
"""
)


def test_migration_from_pr10_preserves_rows(tmp_path: Path) -> None:
    """Migrating from PR10 schema to PR11 schema preserves all existing
    rows and adds the three new final-transcription tables."""
    database_path = tmp_path / "migration.sqlite"
    database = sqlite3.connect(database_path)
    database.executescript(PR10_SCHEMA)

    # Insert PR10 data
    transcription = (
        "id-1",
        None,
        "txt",
        "input.wav",
        "out",
        0.25,
        "en",
        "whisper",
        "file",
        "completed",
        "transcribe",
        "2025-01-01T00:03:00",
        "2025-01-01T00:00:00",
        "2025-01-01T00:01:00",
        None,
        "tiny",
        None,
        1,
        0,
        "name",
        "notes",
    )
    database.execute(
        "INSERT INTO transcription VALUES ("
        + ",".join("?" for _ in transcription)
        + ")",
        transcription,
    )
    segment = (7, 20, 10, "hello", "", "id-1")
    database.execute(
        "INSERT INTO transcription_segment VALUES (?, ?, ?, ?, ?, ?)",
        segment,
    )

    meeting_id = "meeting-1"
    database.execute(
        "INSERT INTO meeting VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meeting_id,
            "SYSTEM",
            "COMPLETED",
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:01:00+00:00",
            "2025-01-01T00:02:00+00:00",
            60_000_000_000,
            "STOPPED",
            "COMPLETE",
        ),
    )
    for role, path, sr, sc, pub in (
        ("MICROPHONE", f"{meeting_id}/microphone.wav", 16000, 160000, 1),
        ("REMOTE", f"{meeting_id}/remote.wav", 16000, 160000, 1),
    ):
        database.execute(
            "INSERT INTO meeting_audio_track VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                meeting_id,
                role,
                path,
                sr,
                sc,
                "STOPPED",
                pub,
                1,
                "host_callback_arrival",
            ),
        )
    database.execute(
        "INSERT INTO meeting_audio_timing_anchor VALUES (?, ?, ?, ?, ?)",
        (meeting_id, "MICROPHONE", 0, 16000, 0),
    )
    database.execute(
        "INSERT INTO meeting_audio_error VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            meeting_id,
            "MICROPHONE",
            0,
            "RECORDER",
            "builtins",
            "ValueError",
            "test error",
        ),
    )
    database.commit()

    # Migrate
    schema = Path("buzz/schema.sql").read_text()
    assert dumb_migrate_db(database, schema)

    # Verify all original data preserved
    assert database.execute("SELECT * FROM transcription").fetchone() == transcription
    assert database.execute("SELECT * FROM transcription_segment").fetchone() == segment

    meeting_row = database.execute(
        "SELECT * FROM meeting WHERE id = ?", (meeting_id,)
    ).fetchone()
    assert meeting_row is not None
    assert meeting_row[0] == meeting_id

    track_rows = database.execute(
        "SELECT * FROM meeting_audio_track WHERE meeting_id = ?",
        (meeting_id,),
    ).fetchall()
    assert len(track_rows) == 2

    # Verify new tables exist
    tables = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "meeting_final_transcription" in tables
    assert "meeting_final_transcription_track" in tables
    assert "meeting_final_transcription_segment" in tables
    assert "meeting_final_transcription_word" in tables

    # Verify FK integrity
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []

    # Verify repeatable migration
    assert not dumb_migrate_db(database, schema)
    assert database.execute("SELECT * FROM transcription").fetchone() == transcription
    assert database.execute("SELECT * FROM transcription_segment").fetchone() == segment

    database.close()


def test_migration_from_pr11_adds_empty_word_table_and_preserves_v1(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pr11-to-pr12.sqlite"
    database = sqlite3.connect(database_path)
    database.execute("PRAGMA foreign_keys = ON")
    database.executescript(PR11_SCHEMA)

    meeting_id = "pr11-meeting"
    database.execute(
        "INSERT INTO meeting VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meeting_id,
            "SYSTEM",
            "COMPLETED",
            "2025-01-01T00:00:00+00:00",
            None,
            None,
            None,
            "STOPPED",
            "COMPLETE",
        ),
    )
    database.execute(
        """INSERT INTO meeting_final_transcription
        (id, meeting_id, profile_version, status, config_model_type,
         config_whisper_model_size, config_hugging_face_model_id,
         config_language, time_created, time_completed)
        VALUES (?, ?, 1, 'COMPLETED', 'WHISPER_CPP', 'CUSTOM', '', NULL, ?, ?)""",
        (
            "pr11-v1",
            meeting_id,
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:01:00+00:00",
        ),
    )
    for role in ("MICROPHONE", "REMOTE"):
        database.execute(
            """INSERT INTO meeting_final_transcription_track
            (generation_id, role, status, time_completed, segment_count)
            VALUES ('pr11-v1', ?, 'COMPLETED', ?, 1)""",
            (role, "2025-01-01T00:01:00+00:00"),
        )
        database.execute(
            """INSERT INTO meeting_final_transcription_segment
            VALUES ('pr11-v1', ?, 0, 0, 1000, -1000000000, 0, ?)""",
            (role, f"old {role}"),
        )
    old_generation = database.execute(
        "SELECT * FROM meeting_final_transcription WHERE id = 'pr11-v1'"
    ).fetchone()
    old_tracks = database.execute(
        "SELECT * FROM meeting_final_transcription_track ORDER BY role"
    ).fetchall()
    old_segments = database.execute(
        "SELECT * FROM meeting_final_transcription_segment ORDER BY role, ordinal"
    ).fetchall()
    database.commit()

    schema = Path("buzz/schema.sql").read_text()
    assert dumb_migrate_db(database, schema)

    assert (
        database.execute(
            "SELECT * FROM meeting_final_transcription WHERE id = 'pr11-v1'"
        ).fetchone()
        == old_generation
    )
    migrated_tracks = database.execute(
        "SELECT * FROM meeting_final_transcription_track ORDER BY role"
    ).fetchall()
    # Old tracks had no word_count column; migrated tracks have word_count=0
    assert len(migrated_tracks) == len(old_tracks)
    for migrated, old in zip(migrated_tracks, old_tracks):
        assert migrated[:7] == old  # original columns preserved
        assert migrated[7] == 0  # word_count defaults to 0
    assert (
        database.execute(
            "SELECT * FROM meeting_final_transcription_segment ORDER BY role, ordinal"
        ).fetchall()
        == old_segments
    )
    assert (
        database.execute("SELECT * FROM meeting_final_transcription_word").fetchall()
        == []
    )

    # word_count defaults to 0 for migrated PR11 rows
    for row in database.execute(
        "SELECT word_count FROM meeting_final_transcription_track "
        "WHERE generation_id = 'pr11-v1'"
    ).fetchall():
        assert row[0] == 0

    database.execute(
        """INSERT INTO meeting_final_transcription
        (id, meeting_id, profile_version, status, config_model_type,
         config_whisper_model_size, config_hugging_face_model_id,
         config_language, time_created)
        VALUES ('pr12-v2', ?, 2, 'IN_PROGRESS', 'FASTER_WHISPER',
                'SMALL', '', NULL, ?)""",
        (meeting_id, "2025-01-02T00:00:00+00:00"),
    )
    database.execute(
        """INSERT INTO meeting_final_transcription_track
        (generation_id, role, status, segment_count)
        VALUES ('pr12-v2', 'MICROPHONE', 'IN_PROGRESS', 0)"""
    )
    database.execute(
        """INSERT INTO meeting_final_transcription_segment
        VALUES ('pr12-v2', 'MICROPHONE', 0, 0, 1000,
                -1000000000, 0, 'phrase')"""
    )
    database.execute(
        """INSERT INTO meeting_final_transcription_word
        VALUES ('pr12-v2', 'MICROPHONE', 0, 0, 100, 500,
                -900000000, -500000000, 'word')"""
    )
    database.commit()

    assert database.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not dumb_migrate_db(database, schema)
    assert database.execute(
        "SELECT text FROM meeting_final_transcription_word"
    ).fetchone() == ("word",)
    database.close()


def test_new_tables_accept_valid_data(tmp_path: Path) -> None:
    """Verify the new tables accept valid data and enforce constraints."""
    database_path = tmp_path / "constraints.sqlite"
    database = sqlite3.connect(database_path)
    schema = Path("buzz/schema.sql").read_text()
    database.executescript(schema)

    meeting_id = "m1"
    database.execute(
        "INSERT INTO meeting VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meeting_id,
            "SYSTEM",
            "COMPLETED",
            "2025-01-01T00:00:00+00:00",
            None,
            None,
            None,
            "STOPPED",
            None,
        ),
    )

    gen_id = "g1"
    database.execute(
        """INSERT INTO meeting_final_transcription
        (id, meeting_id, profile_version, status,
         config_model_type, config_whisper_model_size,
         config_hugging_face_model_id, config_language,
         time_created)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            gen_id,
            meeting_id,
            1,
            "QUEUED",
            "FASTER_WHISPER",
            "TINY",
            "",
            None,
            "2025-01-01T00:00:00+00:00",
        ),
    )

    database.execute(
        """INSERT INTO meeting_final_transcription_track
        (generation_id, role, status, segment_count)
        VALUES (?, ?, ?, ?)""",
        (gen_id, "MICROPHONE", "QUEUED", 0),
    )

    database.execute(
        """INSERT INTO meeting_final_transcription_segment
        (generation_id, role, ordinal, local_start_ms, local_end_ms,
         start_ns, end_ns, text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (gen_id, "MICROPHONE", 0, 0, 1000, 0, 1_000_000_000, "hello"),
    )

    database.commit()

    seg = database.execute(
        "SELECT * FROM meeting_final_transcription_segment"
    ).fetchone()
    assert seg is not None
    assert seg[7] == "hello"  # text column

    assert database.execute("PRAGMA foreign_key_check").fetchall() == []

    database.close()
