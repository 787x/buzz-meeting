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
    TrackTranscriptionInputWord,
    TrackTranscriptionResult,
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
from buzz.transcriber.whisper_file_transcriber import (
    DetailedWhisperFileTranscriber,
    WhisperFileTranscriber,
)

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

    track_completed = pyqtSignal(object)  # list[TrackTranscriptionInputSegment]
    track_rich_completed = pyqtSignal(object)  # TrackTranscriptionResult
    track_error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._transcriber: Optional[WhisperFileTranscriber] = None
        self._thread: Optional[QThread] = None
        self._shutdown_requested = False
        self._active_profile_version: Optional[int] = None

    def start(
        self,
        audio_path: str,
        sample_rate: int,
        config: FinalTranscriptionConfig,
    ) -> None:
        """Start transcription of one audio file.

        Constructs a FileTranscriptionTask with fixed profile semantics.
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

        task = self._build_task(str(audio_path), config, model, model_path)

        transcriber_class = (
            DetailedWhisperFileTranscriber
            if config.profile_version == 2
            else WhisperFileTranscriber
        )
        self._transcriber = transcriber_class(task=task)
        self._active_profile_version = config.profile_version
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
        self._active_profile_version = None

    @pyqtSlot(list)
    def _on_completed(self, segments: list[Segment]) -> None:
        """Convert backend output to the pure meeting result boundary."""
        profile_version = self._active_profile_version
        detailed_words = (
            tuple(self._transcriber.detailed_words)
            if profile_version == 2 and self._transcriber is not None
            else ()
        )
        self._transcriber = None
        self._thread = None
        self._active_profile_version = None

        if self._shutdown_requested:
            return  # Suppress callback during intentional shutdown

        if profile_version == 2:
            result = TrackTranscriptionResult(
                segments=tuple(
                    TrackTranscriptionInputSegment(
                        start_ms=seg.start,
                        end_ms=seg.end,
                        text=seg.text,
                    )
                    for seg in segments
                ),
                words=tuple(
                    TrackTranscriptionInputWord(
                        source_segment_ordinal=word.source_segment_ordinal,
                        start_ms=word.start_ms,
                        end_ms=word.end_ms,
                        text=word.text,
                    )
                    for word in detailed_words
                ),
            )
            self.track_rich_completed.emit(result)
        else:
            self.track_completed.emit(
                [
                    TrackTranscriptionInputSegment(
                        start_ms=seg.start,
                        end_ms=seg.end,
                        text=seg.text,
                    )
                    for seg in segments
                ]
            )

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        """Forward error message."""
        self._transcriber = None
        self._thread = None
        self._active_profile_version = None

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

    @staticmethod
    def _build_task(
        audio_path: str,
        config: FinalTranscriptionConfig,
        model: TranscriptionModel,
        model_path: str,
    ) -> FileTranscriptionTask:
        """Build a task containing only the frozen profile semantics."""
        transcription_options = TranscriptionOptions(
            language=config.language,
            task=Task.TRANSCRIBE,
            model=model,
            word_level_timings=config.profile_version == 2,
            extract_speech=False,
            temperature=DEFAULT_WHISPER_TEMPERATURE,
            initial_prompt="",
            openai_access_token="",
            enable_llm_translation=False,
            silence_threshold=0.0025,
        )
        return FileTranscriptionTask(
            transcription_options=transcription_options,
            file_transcription_options=FileTranscriptionOptions(
                file_paths=None,
                url=None,
                output_formats=set(),
            ),
            model_path=model_path,
            file_path=audio_path,
            source=FileTranscriptionTask.Source.FILE_IMPORT,
            delete_source_file=False,
        )


__all__ = ["MeetingTrackTranscriber"]
