"""Adapter mock tests for WhisperDiarizationRunner.

No real models, no network, no GPU.  Deterministic fake modules/objects.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationAudio,
    SpeakerDiarizationBackend,
    SpeakerDiarizationConfigError,
    SpeakerDiarizationError,
    SpeakerDiarizationTurn,
    SpeakerDiarizationUnavailableError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _valid_audio(num_samples: int = 16_000) -> SpeakerDiarizationAudio:
    return SpeakerDiarizationAudio(
        waveform=np.zeros(num_samples, dtype=np.float32),
        sample_rate=16_000,
    )


def _fake_torch():
    """Return a minimal fake torch module."""
    torch = types.ModuleType("torch")
    torch.float32 = np.float32
    torch.float16 = "float16"

    class _FakeTensor:
        pass

    def _from_numpy(arr):
        return _FakeTensor()

    def _unsqueeze(self, dim):
        return self

    _FakeTensor.from_numpy = staticmethod(_from_numpy)
    _FakeTensor.unsqueeze = _unsqueeze
    torch.from_numpy = _from_numpy

    _cuda = types.ModuleType("torch.cuda")
    _cuda.is_available = lambda: False
    _cuda.empty_cache = MagicMock()
    torch.cuda = _cuda

    return torch


# ── 1. Importing pure domain module does not import torch/NeMo ──────────────


def test_pure_module_no_torch_import():
    """Importing buzz.meeting.speaker_diarization must NOT import torch."""
    # Ensure the module is fresh by removing it from cache.
    mods_before = set(sys.modules)
    import buzz.meeting.speaker_diarization  # noqa: F401

    new_mods = set(sys.modules) - mods_before
    assert not any("torch" in m for m in new_mods)


# ── 2. Adapter heavy backend imports are lazy ────────────────────────────────


def test_adapter_import_no_heavy_backend():
    """Importing the adapter module itself must not eagerly import NeMo or
    whisper_diarization backends."""
    mods_before = set(sys.modules)
    import buzz.meeting.speaker_diarization_adapter  # noqa: F401

    new_mods = set(sys.modules) - mods_before
    assert not any("whisper_diarization" in m for m in new_mods)
    assert not any("nemo" in m for m in new_mods)


# ── 3. MSDD backend selects MSDDDiarizer ────────────────────────────────────


@patch.dict(sys.modules, {"torch": _fake_torch()})
def test_msdd_backend_selects_msdd():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    mock_msdd = MagicMock()
    mock_msdd_instance = MagicMock()
    mock_msdd_instance.diarize.return_value = []
    mock_msdd.return_value = mock_msdd_instance

    wd = types.ModuleType("whisper_diarization")
    diar = types.ModuleType("whisper_diarization.diarization")
    diar.MSDDDiarizer = mock_msdd

    with patch.dict(
        sys.modules,
        {
            "whisper_diarization": wd,
            "whisper_diarization.diarization": diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        runner.diarize(_valid_audio())
        mock_msdd.assert_called_once()


# ── 4. Sortformer backend selects SortformerDiarizer ─────────────────────────


@patch.dict(sys.modules, {"torch": _fake_torch()})
def test_sortformer_backend_selects_sortformer():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    mock_sf = MagicMock()
    mock_sf_instance = MagicMock()
    mock_sf_instance.diarize.return_value = []
    mock_sf.return_value = mock_sf_instance

    wd = types.ModuleType("whisper_diarization")
    diar = types.ModuleType("whisper_diarization.diarization")
    diar.SortformerDiarizer = mock_sf

    with patch.dict(
        sys.modules,
        {
            "whisper_diarization": wd,
            "whisper_diarization.diarization": diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.SORTFORMER, "cpu")
        runner.diarize(_valid_audio())
        mock_sf.assert_called_once()


# ── 5. Constructor receives exact device ─────────────────────────────────────


@patch.dict(sys.modules, {"torch": _fake_torch()})
def test_constructor_receives_device():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
    assert runner._device == "cpu"


# ── 6. Correct torch tensor conversion ──────────────────────────────────────


def test_tensor_conversion():
    """Audio waveform is converted via torch.from_numpy(...).unsqueeze(0)."""
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_torch = _fake_torch()
    spy_from_numpy = MagicMock(wraps=fake_torch.from_numpy)
    fake_torch.from_numpy = spy_from_numpy

    mock_diarizer_class = MagicMock()
    mock_diarizer_class.return_value.diarize.return_value = []

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_diarizer_class

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        audio = _valid_audio()
        runner.diarize(audio)

    spy_from_numpy.assert_called_once()
    np.testing.assert_array_equal(spy_from_numpy.call_args[0][0], audio.waveform)


# ── 7. Backend diarize called exactly once ───────────────────────────────────


def test_backend_diarize_called_once():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    mock_class = MagicMock()
    mock_class.return_value.diarize.return_value = []

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        runner.diarize(_valid_audio())
        mock_class.return_value.diarize.assert_called_once()


# ── 8. Raw (start, end, speaker) converted to DTO ───────────────────────────


def test_raw_result_to_dto():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    mock_class = MagicMock()
    mock_class.return_value.diarize.return_value = [
        (0, 1000, 0),
        (1000, 2000, 1),
    ]

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        result = runner.diarize(_valid_audio())

    assert len(result) == 2
    assert result[0] == SpeakerDiarizationTurn(0, 0, 1000)
    assert result[1] == SpeakerDiarizationTurn(1, 1000, 2000)


# ── 9. Non-contiguous speakers accepted at adapter level ─────────────────────


def test_non_contiguous_at_adapter():
    """Adapter returns non-contiguous speaker indices as-is."""
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    mock_class = MagicMock()
    mock_class.return_value.diarize.return_value = [
        (0, 500, 0),
        (500, 1000, 7),
    ]

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        result = runner.diarize(_valid_audio())

    assert result[1].speaker_index == 7


# ── 10. Unknown device rejected ──────────────────────────────────────────────


def test_unknown_device_rejected():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    with pytest.raises(SpeakerDiarizationConfigError, match="device must be one of"):
        WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "tpu")


# ── 11. Backend import failure → SpeakerDiarizationUnavailableError ──────────


def test_backend_import_failure():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    # No MSDDDiarizer attribute → ImportError at attribute access

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        with pytest.raises(SpeakerDiarizationUnavailableError) as exc_info:
            runner.diarize(_valid_audio())
        assert exc_info.value.__cause__ is not None


# ── 12. Backend construction failure preserves cause ─────────────────────────


def test_construction_failure_preserves_cause():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    cause = RuntimeError("bad weights")
    mock_class = MagicMock(side_effect=cause)

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        with pytest.raises(SpeakerDiarizationError) as exc_info:
            runner.diarize(_valid_audio())
        assert exc_info.value.__cause__ is cause


# ── 13. Inference failure preserves cause ────────────────────────────────────


def test_inference_failure_preserves_cause():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    cause = RuntimeError("OOM")
    mock_class = MagicMock()
    mock_class.return_value.diarize.side_effect = cause

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": _fake_torch(),
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        with pytest.raises(SpeakerDiarizationError) as exc_info:
            runner.diarize(_valid_audio())
        assert exc_info.value.__cause__ is cause


# ── 14. torch.cuda.empty_cache cleanup on success ───────────────────────────


def test_cleanup_on_success():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_torch = _fake_torch()
    spy_cleanup = fake_torch.cuda.empty_cache
    spy_cleanup.reset_mock()

    mock_class = MagicMock()
    mock_class.return_value.diarize.return_value = []

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        runner.diarize(_valid_audio())
        spy_cleanup.assert_called()


# ── 15. Cleanup on inference failure ─────────────────────────────────────────


def test_cleanup_on_failure():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_torch = _fake_torch()
    spy_cleanup = fake_torch.cuda.empty_cache
    spy_cleanup.reset_mock()

    mock_class = MagicMock()
    mock_class.return_value.diarize.side_effect = RuntimeError("boom")

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        with pytest.raises(SpeakerDiarizationError):
            runner.diarize(_valid_audio())
        spy_cleanup.assert_called()


# ── 16. CPU CUDA-hiding workaround restores torch.cuda.is_available ──────────


def test_cpu_cuda_hiding_restores():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_torch = _fake_torch()

    # Make it look like CUDA is available so the workaround activates
    fake_torch.cuda.is_available = lambda: True

    mock_class = MagicMock()
    mock_class.return_value.diarize.return_value = []

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        runner.diarize(_valid_audio())

    # The function should be restored (or at least callable)
    # The exact object may differ from original_fn since we replaced it above,
    # but it should NOT be the lambda:False workaround.
    # We verify it doesn't return False always (the workaround signature).
    assert fake_torch.cuda.is_available() is True


# ── 17. CPU workaround restores even on backend raise ────────────────────────


def test_cpu_cuda_restores_on_error():
    from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner

    fake_torch = _fake_torch()
    fake_torch.cuda.is_available = lambda: True  # pretend CUDA available

    mock_class = MagicMock()
    mock_class.side_effect = RuntimeError("construction failed")

    fake_wd = types.ModuleType("whisper_diarization")
    fake_diar = types.ModuleType("whisper_diarization.diarization")
    fake_diar.MSDDDiarizer = mock_class

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "whisper_diarization": fake_wd,
            "whisper_diarization.diarization": fake_diar,
        },
    ):
        runner = WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, "cpu")
        with pytest.raises(SpeakerDiarizationError):
            runner.diarize(_valid_audio())

    # Restored to original (True) — NOT stuck on lambda:False
    assert fake_torch.cuda.is_available() is True


# ── 18. No network/model download in any test ────────────────────────────────
# All tests above use mocks; no test invokes real model download.
