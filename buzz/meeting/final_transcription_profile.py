"""Versioned meeting final-transcription profiles.

Pure Python: no Qt, Settings, model loading, or runtime path resolution.
"""

from __future__ import annotations

from buzz.meeting.final_transcription import FinalTranscriptionConfig


class HighQualityFinalTranscriptionProfile:
    """Factory for the frozen high-quality local profile."""

    VERSION = 2

    @staticmethod
    def create(
        *,
        model_type: str,
        whisper_model_size: str,
        language: str | None = None,
    ) -> FinalTranscriptionConfig:
        return FinalTranscriptionConfig(
            profile_version=HighQualityFinalTranscriptionProfile.VERSION,
            model_type=model_type,
            whisper_model_size=whisper_model_size,
            hugging_face_model_id="",
            language=language,
        )


__all__ = ["HighQualityFinalTranscriptionProfile"]
