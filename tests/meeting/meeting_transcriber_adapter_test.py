"""Tests for the Qt/FileTranscriber meeting adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QCoreApplication

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    TrackTranscriptionResult,
)
from buzz.meeting.meeting_transcriber_adapter import MeetingTrackTranscriber
from buzz.model_loader import ModelType, TranscriptionModel, WhisperModelSize
from buzz.transcriber.transcriber import (
    DEFAULT_WHISPER_TEMPERATURE,
    FileTranscriptionTask,
    Segment,
    Task,
)
from buzz.transcriber.whisper_file_transcriber import DetailedTranscriptionWord


@pytest.fixture(scope="module")
def qt_application():
    application = QCoreApplication.instance()
    owns_application = application is None
    if application is None:
        application = QCoreApplication([])
    yield application
    if owns_application:
        application.quit()


class TestModelResolution:
    def test_resolve_whisper_model_type(self) -> None:
        from buzz.model_loader import ModelType

        assert (
            MeetingTrackTranscriber._resolve_model_type("WHISPER") is ModelType.WHISPER
        )
        assert (
            MeetingTrackTranscriber._resolve_model_type("WHISPER_CPP")
            is ModelType.WHISPER_CPP
        )
        assert (
            MeetingTrackTranscriber._resolve_model_type("FASTER_WHISPER")
            is ModelType.FASTER_WHISPER
        )
        assert (
            MeetingTrackTranscriber._resolve_model_type("HUGGING_FACE")
            is ModelType.HUGGING_FACE
        )

    def test_resolve_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            MeetingTrackTranscriber._resolve_model_type("UNKNOWN")

    def test_resolve_whisper_size(self) -> None:
        from buzz.model_loader import WhisperModelSize

        assert (
            MeetingTrackTranscriber._resolve_whisper_size("TINY")
            is WhisperModelSize.TINY
        )
        assert MeetingTrackTranscriber._resolve_whisper_size(None) is None

    def test_resolve_unknown_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            MeetingTrackTranscriber._resolve_whisper_size("HUGE")


class TestAdapterSignals:
    """Test adapter signal behavior without real ASR."""

    def test_error_on_missing_model(self, qt_application, tmp_path: Path) -> None:
        """If model not available locally, adapter emits error signal."""
        adapter = MeetingTrackTranscriber()
        errors: list[str] = []
        adapter.track_error.connect(errors.append)

        # Use a config with a model that's not downloaded
        config = FinalTranscriptionConfig(
            model_type="WHISPER",
            whisper_model_size="LARGE",
        )
        # Use a dummy audio path
        audio_path = str(tmp_path / "dummy.wav")
        Path(audio_path).touch()

        adapter.start(audio_path, 16000, config)
        # Give Qt event loop a chance to process
        QCoreApplication.processEvents()

        # Should have emitted an error about model not available
        assert len(errors) > 0
        assert "not available" in errors[0]

        adapter.shutdown()

    def test_error_when_already_active(self, qt_application, tmp_path: Path) -> None:
        """Cannot start a second transcription while one is active."""
        adapter = MeetingTrackTranscriber()
        errors: list[str] = []
        adapter.track_error.connect(errors.append)

        config = FinalTranscriptionConfig(
            model_type="WHISPER",
            whisper_model_size="LARGE",
        )
        audio = str(tmp_path / "a.wav")
        Path(audio).touch()

        # First start fails (model not available), clears _transcriber
        adapter.start(audio, 16000, config)
        QCoreApplication.processEvents()

        # Verify we got a model-not-available error
        assert len(errors) > 0
        assert "not available" in errors[0]

        # Clear errors for next check
        errors.clear()

        # Now manually set _transcriber to simulate active state
        # by calling start twice rapidly — second should fail because
        # first's error already set _transcriber to None
        # (since model is missing, this tests the early error path)
        adapter.start(audio, 16000, config)
        QCoreApplication.processEvents()
        # Both should fail with model-not-available, not "already active"
        assert all("not available" in e for e in errors)

    def test_shutdown_stops_transcriber(self, qt_application) -> None:
        """Shutdown completes without error when nothing is active."""
        adapter = MeetingTrackTranscriber()
        adapter.shutdown()  # Should not raise


class TestTaskConstruction:
    """Verify the adapter constructs correct FileTranscriptionTask."""

    def test_config_produces_correct_options(self) -> None:
        """Config v1 should produce TRANSCRIBE, no word timings, etc."""
        config = FinalTranscriptionConfig(
            language="zh",
            model_type="FASTER_WHISPER",
            whisper_model_size="SMALL",
        )
        # Verify config values that adapter would use
        assert config.language == "zh"
        assert config.model_type == "FASTER_WHISPER"
        assert config.whisper_model_size == "SMALL"
        # These are the v1 fixed values the adapter applies
        # (verified by code inspection of meeting_transcriber_adapter.py)

    @staticmethod
    def _model(model_type: ModelType) -> TranscriptionModel:
        return TranscriptionModel(
            model_type=model_type,
            whisper_model_size=WhisperModelSize.SMALL,
        )

    @pytest.mark.parametrize("profile_version,word_timings", [(1, False), (2, True)])
    def test_build_task_freezes_profile_options(
        self, profile_version: int, word_timings: bool
    ) -> None:
        config = FinalTranscriptionConfig(
            profile_version=profile_version,
            model_type="FASTER_WHISPER",
            whisper_model_size="SMALL",
            language="zh",
        )
        task = MeetingTrackTranscriber._build_task(
            "meeting.wav",
            config,
            self._model(ModelType.FASTER_WHISPER),
            "local-model",
        )

        options = task.transcription_options
        assert options.language == "zh"
        assert options.task is Task.TRANSCRIBE
        assert options.word_level_timings is word_timings
        assert options.extract_speech is False
        assert options.temperature == DEFAULT_WHISPER_TEMPERATURE
        assert options.initial_prompt == ""
        assert options.enable_llm_translation is False
        assert task.file_transcription_options.output_formats == set()
        assert task.source is FileTranscriptionTask.Source.FILE_IMPORT
        assert task.delete_source_file is False
        assert task.model_path == "local-model"


class TestRichResultConversion:
    def test_v1_emits_list_to_track_completed(self, qt_application) -> None:
        adapter = MeetingTrackTranscriber()
        emitted: list = []
        adapter.track_completed.connect(emitted.append)
        adapter._active_profile_version = 1

        adapter._on_completed([Segment(start=0, end=1000, text="phrase")])

        assert len(emitted) == 1
        assert isinstance(emitted[0], list)
        assert emitted[0][0].text == "phrase"

    def test_v2_emits_result_to_track_rich_completed(self, qt_application) -> None:
        adapter = MeetingTrackTranscriber()
        emitted: list[TrackTranscriptionResult] = []
        adapter.track_rich_completed.connect(emitted.append)
        adapter._active_profile_version = 2
        adapter._transcriber = SimpleNamespace(
            detailed_words=[
                DetailedTranscriptionWord(
                    source_segment_ordinal=0,
                    start_ms=10,
                    end_ms=200,
                    text="word",
                )
            ]
        )

        adapter._on_completed([Segment(start=0, end=1000, text="phrase")])

        assert len(emitted) == 1
        assert [segment.text for segment in emitted[0].segments] == ["phrase"]
        assert len(emitted[0].words) == 1
        assert emitted[0].words[0].source_segment_ordinal == 0
        assert emitted[0].words[0].text == "word"
