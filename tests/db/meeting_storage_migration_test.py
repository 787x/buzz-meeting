import sqlite3
from pathlib import Path

from buzz.db.migrator import dumb_migrate_db


PRE_PR10_SCHEMA = """
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
"""


def test_pre_pr10_migration_preserves_rows_adds_tables_and_is_repeatable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.sqlite"
    database = sqlite3.connect(database_path)
    database.executescript(PRE_PR10_SCHEMA)
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
    segment = (7, 20, 10, "hello", "你好", "id-1")
    database.execute(
        "INSERT INTO transcription_segment VALUES (?, ?, ?, ?, ?, ?)", segment
    )
    database.commit()

    schema = Path("buzz/schema.sql").read_text()
    assert dumb_migrate_db(database, schema)
    assert database.execute("SELECT * FROM transcription").fetchone() == transcription
    assert database.execute("SELECT * FROM transcription_segment").fetchone() == segment
    tables = {
        row[0]
        for row in database.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {
        "meeting",
        "meeting_audio_track",
        "meeting_audio_timing_anchor",
        "meeting_audio_error",
    } <= tables
    assert database.execute("PRAGMA foreign_key_check").fetchall() == []

    assert not dumb_migrate_db(database, schema)
    assert database.execute("SELECT * FROM transcription").fetchone() == transcription
    assert database.execute("SELECT * FROM transcription_segment").fetchone() == segment
    database.close()
