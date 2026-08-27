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

CREATE TABLE meeting_final_transcription (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL CHECK (profile_version > 0),
    status TEXT NOT NULL,
    config_model_type TEXT NOT NULL,
    config_whisper_model_size TEXT,
    config_hugging_face_model_id TEXT NOT NULL DEFAULT '',
    config_language TEXT,
    error_message TEXT
        CHECK (
            error_message IS NULL
            OR length(error_message) <= 4096
        ),
    time_created TEXT NOT NULL,
    time_started TEXT,
    time_completed TEXT,
    UNIQUE (meeting_id, profile_version),
    FOREIGN KEY (meeting_id)
        REFERENCES meeting(id)
        ON DELETE CASCADE
);

CREATE TABLE meeting_final_transcription_track (
    generation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
        CHECK (
            error_message IS NULL
            OR length(error_message) <= 4096
        ),
    time_started TEXT,
    time_completed TEXT,
    segment_count INTEGER NOT NULL DEFAULT 0
        CHECK (segment_count >= 0),
    word_count INTEGER NOT NULL DEFAULT 0
        CHECK (word_count >= 0),
    PRIMARY KEY (generation_id, role),
    FOREIGN KEY (generation_id)
        REFERENCES meeting_final_transcription(id)
        ON DELETE CASCADE
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
        REFERENCES meeting_final_transcription_segment(
            generation_id, role, ordinal
        )
        ON DELETE CASCADE,
    CHECK (local_end_ms >= local_start_ms),
    CHECK (end_ns >= start_ns)
);
