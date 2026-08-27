"""Qt adapter wrapping FileTranscriber for meeting final transcription.

Owns one concrete FileTranscriber lifecycle on a QThread.
Returns pure segment DTOs to the caller thread via Qt signals.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    TrackTranscriptionInputSegment,
)
from buzz.model_loader import ModelType, TranscriptionModel, WhisperModelSize
from buzz.transcriber.transcriber import (
    DEFAULT_WHISPER_TEMPERATURE,
    FileTranscriptionOptions,
    FileTranscriptionTask,
    Segment,
    Task,
    TranscriptionOptions,
)
from buzz.transcriber.whisper_file_transcriber import WhisperFileTranscriber

logger = logging.getLogger(__name__)


class MeetingTrackTranscriber(QObject):
    """Run FileTranscriber for one meeting track on a dedicated QThread.

    Emits ``track_completed`` with pure segment DTOs or ``track_error``
    with an error message.  Uses ``WhisperFileTranscriber`` only — no
    OpenAI API backend in PR11.

    Usage::

        adapter = MeetingTrackTranscriber()
        adapter.track_completed.connect(my_handler)
        adapter.start(audio_path, sample_rate, config)
        # ...
        adapter.shutdown()
    """

    track_completed = pyqtSignal(list)  # list[TrackTranscriptionInputSegment]
    track_error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._transcriber: Optional[WhisperFileTranscriber] = None
        self._thread: Optional[QThread] = None
        self._shutdown_requested = False

    def start(
        self,
        audio_path: str,
        sample_rate: int,
        config: FinalTranscriptionConfig,
    ) -> None:
        """Start transcription of one audio file.

        Constructs a FileTranscriptionTask with PR11 v1 fixed options:
        FILE_IMPORT, empty output_formats, TRANSCRIBE, no word timings,
        no speech extraction.
        """
        if self._shutdown_requested:
            self.track_error.emit("Shutdown requested")
            return

        if self._transcriber is not None:
            self.track_error.emit("Transcriber already active")
            return

        # Resolve model path from persisted config
        model_type = self._resolve_model_type(config.model_type)
        whisper_size = self._resolve_whisper_size(config.whisper_model_size)
        model = TranscriptionModel(
            model_type=model_type,
            whisper_model_size=whisper_size,
            hugging_face_model_id=config.hugging_face_model_id or "",
        )
        model_path = model.get_local_model_path()
        if model_path is None:
            self.track_error.emit(
                f"Model not available locally: {config.model_type}/"
                f"{config.whisper_model_size or config.hugging_face_model_id}"
            )
            return

        # Build task with PR11 v1 fixed options
        transcription_options = TranscriptionOptions(
            language=config.language,
            task=Task.TRANSCRIBE,
            model=model,
            word_level_timings=False,
            extract_speech=False,
            temperature=DEFAULT_WHISPER_TEMPERATURE,
            initial_prompt="",
            openai_access_token="",
            enable_llm_translation=False,
            silence_threshold=0.0025,
        )

        file_transcription_options = FileTranscriptionOptions(
            file_paths=None,
            url=None,
            output_formats=set(),
        )

        task = FileTranscriptionTask(
            transcription_options=transcription_options,
            file_transcription_options=file_transcription_options,
            model_path=model_path,
            file_path=str(audio_path),
            source=FileTranscriptionTask.Source.FILE_IMPORT,
        )

        # Ensure extract_speech is False (safety)
        task.transcription_options.extract_speech = False

        self._transcriber = WhisperFileTranscriber(task=task)
        self._thread = QThread(self)
        self._transcriber.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._transcriber.run)
        self._transcriber.completed.connect(self._on_completed)
        self._transcriber.error.connect(self._on_error)
        self._transcriber.completed.connect(self._thread.quit)
        self._transcriber.error.connect(self._thread.quit)
        self._transcriber.completed.connect(self._transcriber.deleteLater)
        self._transcriber.error.connect(self._transcriber.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def shutdown(self) -> None:
        """Request graceful shutdown.

        If a transcription is in progress, stops it. Leaves durable
        state as IN_PROGRESS for recovery.
        """
        self._shutdown_requested = True
        if self._transcriber is not None:
            try:
                self._transcriber.stop()
            except Exception:
                logger.exception("Error stopping transcriber")
        if self._thread is not None and self._thread.isRunning():
            if not self._thread.wait(10000):
                logger.warning("Transcriber thread did not finish in 10s, terminating")
                self._thread.terminate()
                self._thread.wait(2000)
        self._transcriber = None
        self._thread = None

    @pyqtSlot(list)
    def _on_completed(self, segments: list[Segment]) -> None:
        """Convert FileTranscriber segments to pure DTOs."""
        self._transcriber = None
        self._thread = None

        if self._shutdown_requested:
            return  # Suppress callback during intentional shutdown

        result = [
            TrackTranscriptionInputSegment(
                start_ms=seg.start,
                end_ms=seg.end,
                text=seg.text,
            )
            for seg in segments
        ]
        self.track_completed.emit(result)

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        """Forward error message."""
        self._transcriber = None
        self._thread = None

        if self._shutdown_requested:
            return  # Suppress callback during intentional shutdown

        self.track_error.emit(error)

    @staticmethod
    def _resolve_model_type(model_type_str: str) -> ModelType:
        """Convert persisted model_type string to ModelType enum."""
        mapping = {
            "WHISPER": ModelType.WHISPER,
            "WHISPER_CPP": ModelType.WHISPER_CPP,
            "FASTER_WHISPER": ModelType.FASTER_WHISPER,
            "HUGGING_FACE": ModelType.HUGGING_FACE,
        }
        result = mapping.get(model_type_str)
        if result is None:
            raise ValueError(f"Unsupported model_type: {model_type_str!r}")
        return result

    @staticmethod
    def _resolve_whisper_size(
        size_str: Optional[str],
    ) -> Optional[WhisperModelSize]:
        """Convert persisted whisper_model_size string to enum."""
        if size_str is None:
            return None
        try:
            return WhisperModelSize[size_str]
        except KeyError:
            raise ValueError(f"Unknown whisper_model_size: {size_str!r}")


__all__ = ["MeetingTrackTranscriber"]
