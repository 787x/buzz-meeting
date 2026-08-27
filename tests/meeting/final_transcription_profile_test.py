"""Contract tests for the frozen high-quality final-transcription profile."""

from __future__ import annotations

import pytest

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionConfigError,
)
from buzz.meeting.final_transcription_profile import (
    HighQualityFinalTranscriptionProfile,
)


@pytest.mark.parametrize("model_type", ["WHISPER", "FASTER_WHISPER"])
def test_factory_returns_exact_v2_config(model_type: str) -> None:
    config = HighQualityFinalTranscriptionProfile.create(
        model_type=model_type,
        whisper_model_size="LARGEV3",
        language="zh",
    )

    assert config == FinalTranscriptionConfig(
        profile_version=2,
        model_type=model_type,
        whisper_model_size="LARGEV3",
        hugging_face_model_id="",
        language="zh",
    )
    assert HighQualityFinalTranscriptionProfile.VERSION == 2


def test_factory_preserves_none_language() -> None:
    config = HighQualityFinalTranscriptionProfile.create(
        model_type="WHISPER",
        whisper_model_size="TINY",
        language=None,
    )

    assert config.language is None


@pytest.mark.parametrize(
    "model_type,size",
    [
        ("WHISPER_CPP", "SMALL"),
        ("HUGGING_FACE", None),
        ("OPEN_AI_WHISPER_API", "SMALL"),
        ("UNKNOWN", "SMALL"),
        ("WHISPER", "CUSTOM"),
        ("FASTER_WHISPER", "LUMII"),
        ("WHISPER", "UNKNOWN"),
    ],
)
def test_v2_rejects_unsupported_identity(model_type: str, size: str | None) -> None:
    kwargs: dict[str, object] = {
        "profile_version": 2,
        "model_type": model_type,
        "whisper_model_size": size,
    }
    if model_type == "HUGGING_FACE":
        kwargs["hugging_face_model_id"] = "openai/whisper-large-v3"

    with pytest.raises(FinalTranscriptionConfigError):
        FinalTranscriptionConfig(**kwargs)


def test_factory_requires_explicit_model_size() -> None:
    with pytest.raises(TypeError):
        HighQualityFinalTranscriptionProfile.create(model_type="WHISPER")


@pytest.mark.parametrize(
    "size",
    [
        "TINY",
        "TINYEN",
        "BASE",
        "BASEEN",
        "SMALL",
        "SMALLEN",
        "MEDIUM",
        "MEDIUMEN",
        "LARGE",
        "LARGEV2",
        "LARGEV3",
        "LARGEV3TURBO",
    ],
)
def test_v2_accepts_every_frozen_standard_size(size: str) -> None:
    assert (
        FinalTranscriptionConfig(
            profile_version=2,
            model_type="FASTER_WHISPER",
            whisper_model_size=size,
        ).whisper_model_size
        == size
    )


def test_v1_still_accepts_cpp_hf_custom_and_lumii() -> None:
    configs = (
        FinalTranscriptionConfig(model_type="WHISPER_CPP", whisper_model_size="CUSTOM"),
        FinalTranscriptionConfig(model_type="WHISPER", whisper_model_size="LUMII"),
        FinalTranscriptionConfig(
            model_type="HUGGING_FACE",
            whisper_model_size=None,
            hugging_face_model_id="org/model",
        ),
    )

    assert all(config.profile_version == 1 for config in configs)
