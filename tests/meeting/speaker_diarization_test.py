"""Pure speaker diarization domain tests.

No torch, NeMo, Qt, network, filesystem, or sleep.
"""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationAudio,
    SpeakerDiarizationConfigError,
    SpeakerDiarizationError,
    SpeakerDiarizationTurn,
    SpeakerDiarizationUnavailableError,
    SpeakerDiarizationService,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _valid_audio(
    num_samples: int = 16_000,
    sample_rate: int = 16_000,
) -> SpeakerDiarizationAudio:
    return SpeakerDiarizationAudio(
        waveform=np.zeros(num_samples, dtype=np.float32),
        sample_rate=sample_rate,
    )


def _fake_runner(results: Sequence | None = None, *, side_effect=None):
    """Return a mock runner that yields *results* or raises *side_effect*."""
    runner = MagicMock()
    if side_effect is not None:
        runner.diarize.side_effect = side_effect
    else:
        runner.diarize.return_value = results if results is not None else []
    return runner


# ── 1. Valid single speaker ──────────────────────────────────────────────────


def test_single_speaker():
    runner = _fake_runner([SpeakerDiarizationTurn(0, 0, 1000)])
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == (SpeakerDiarizationTurn(0, 0, 1000),)


# ── 2. Multiple speakers ────────────────────────────────────────────────────


def test_multiple_speakers():
    turns = [
        SpeakerDiarizationTurn(0, 0, 500),
        SpeakerDiarizationTurn(1, 500, 1000),
    ]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == tuple(turns)


# ── 3. Non-contiguous speaker indices accepted ──────────────────────────────


def test_non_contiguous_speaker_indices():
    turns = [
        SpeakerDiarizationTurn(0, 0, 500),
        SpeakerDiarizationTurn(7, 500, 1000),
    ]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == tuple(turns)


# ── 4. Overlap preserved ────────────────────────────────────────────────────


def test_overlap_preserved():
    turns = [
        SpeakerDiarizationTurn(0, 0, 1000),
        SpeakerDiarizationTurn(1, 500, 1500),
    ]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == tuple(turns)


# ── 5. Duplicate turns preserved ────────────────────────────────────────────


def test_duplicate_turns_preserved():
    t = SpeakerDiarizationTurn(0, 0, 1000)
    turns = [t, t]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert len(result) == 2


# ── 6. Zero-length turn allowed ─────────────────────────────────────────────


def test_zero_length_turn():
    turns = [SpeakerDiarizationTurn(0, 500, 500)]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == (SpeakerDiarizationTurn(0, 500, 500),)


# ── 7. Adjacent same-speaker turns not merged ───────────────────────────────


def test_adjacent_same_speaker_not_merged():
    turns = [
        SpeakerDiarizationTurn(0, 0, 500),
        SpeakerDiarizationTurn(0, 500, 1000),
    ]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert len(result) == 2


# ── 8. Empty result valid ───────────────────────────────────────────────────


def test_empty_result():
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == ()


# ── 9. Unsorted input stable-sorted by start_ms ────────────────────────────


def test_sorted_by_start_ms():
    turns = [
        SpeakerDiarizationTurn(1, 1000, 1500),
        SpeakerDiarizationTurn(0, 0, 500),
    ]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result[0].start_ms == 0
    assert result[1].start_ms == 1000


# ── 10. Equal-start tie preserves runner order ──────────────────────────────


def test_equal_start_preserves_runner_order():
    a = SpeakerDiarizationTurn(0, 100, 200)
    b = SpeakerDiarizationTurn(1, 100, 300)
    turns = [a, b]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == (a, b)


def test_equal_start_reverse_preserves_runner_order():
    a = SpeakerDiarizationTurn(1, 100, 300)
    b = SpeakerDiarizationTurn(0, 100, 200)
    turns = [a, b]
    runner = _fake_runner(turns)
    svc = SpeakerDiarizationService(runner)
    result = svc.diarize(_valid_audio())
    assert result == (a, b)


# ── 11. Negative start rejected ─────────────────────────────────────────────


def test_negative_start_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(0, -1, 100)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(SpeakerDiarizationConfigError, match="start_ms must be >= 0"):
        svc.diarize(_valid_audio())


# ── 12. end < start rejected ────────────────────────────────────────────────


def test_end_less_than_start_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(0, 200, 100)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="end_ms must be >= start_ms"
    ):
        svc.diarize(_valid_audio())


# ── 13. Negative speaker rejected ───────────────────────────────────────────


def test_negative_speaker_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(-1, 0, 100)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="speaker_index must be >= 0"
    ):
        svc.diarize(_valid_audio())


# ── 14. bool start rejected ─────────────────────────────────────────────────


def test_bool_start_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(0, True, 100)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="start_ms must be an int, not bool"
    ):
        svc.diarize(_valid_audio())


# ── 15. bool end rejected ───────────────────────────────────────────────────


def test_bool_end_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(0, 0, True)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="end_ms must be an int, not bool"
    ):
        svc.diarize(_valid_audio())


# ── 16. bool speaker rejected ───────────────────────────────────────────────


def test_bool_speaker_rejected():
    runner = _fake_runner([SpeakerDiarizationTurn(True, 0, 100)])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="speaker_index must be an int, not bool"
    ):
        svc.diarize(_valid_audio())


# ── 17. Runner called exactly once ──────────────────────────────────────────


def test_runner_called_exactly_once():
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    svc.diarize(_valid_audio())
    runner.diarize.assert_called_once()


# ── 18. Audio object not mutated ────────────────────────────────────────────


def test_audio_not_mutated():
    waveform = np.ones(16_000, dtype=np.float32)
    original = waveform.copy()
    audio = SpeakerDiarizationAudio(waveform=waveform, sample_rate=16_000)
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    svc.diarize(audio)
    np.testing.assert_array_equal(waveform, original)


# ── 19. Runner exception becomes domain error ───────────────────────────────


def test_runner_exception_wrapped():
    cause = RuntimeError("backend exploded")
    runner = _fake_runner(side_effect=cause)
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationError, match="Runner diarization failed"
    ) as exc_info:
        svc.diarize(_valid_audio())
    assert exc_info.value.__cause__ is cause


# ── 19b. Domain subtypes propagate unchanged ─────────────────────────────────


def test_domain_subtype_propagates_unchanged():
    """SpeakerDiarizationUnavailableError from runner is not re-wrapped."""
    cause = SpeakerDiarizationUnavailableError("missing backend")
    runner = _fake_runner(side_effect=cause)
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationUnavailableError, match="missing backend"
    ) as exc_info:
        svc.diarize(_valid_audio())
    # Same object, not a new wrapper
    assert exc_info.value is cause


# ── Audio validation tests ──────────────────────────────────────────────────


# 20. sample_rate 16000 accepted
def test_valid_sample_rate():
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    svc.diarize(_valid_audio(sample_rate=16_000))


# 21. Wrong sample rate rejected
def test_wrong_sample_rate_rejected():
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="sample_rate must be 16000"
    ):
        svc.diarize(_valid_audio(sample_rate=44_100))


# 22. bool sample_rate rejected
def test_bool_sample_rate_rejected():
    audio = SpeakerDiarizationAudio(
        waveform=np.zeros(16_000, dtype=np.float32),
        sample_rate=True,
    )
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="sample_rate must be an int, not bool"
    ):
        svc.diarize(audio)


# 23. non-ndarray waveform rejected
def test_non_ndarray_rejected():
    audio = SpeakerDiarizationAudio(waveform=[0.0] * 16_000, sample_rate=16_000)
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="waveform must be a numpy.ndarray"
    ):
        svc.diarize(audio)


# 24. 2-D/stereo ndarray rejected
def test_2d_ndarray_rejected():
    audio = SpeakerDiarizationAudio(
        waveform=np.zeros((2, 16_000), dtype=np.float32),
        sample_rate=16_000,
    )
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="waveform must be one-dimensional"
    ):
        svc.diarize(audio)


# 25. float64 rejected
def test_float64_rejected():
    audio = SpeakerDiarizationAudio(
        waveform=np.zeros(16_000, dtype=np.float64),
        sample_rate=16_000,
    )
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    with pytest.raises(
        SpeakerDiarizationConfigError, match="waveform dtype must be float32"
    ):
        svc.diarize(audio)


# 26. float32 one-dimensional accepted
def test_float32_1d_accepted():
    runner = _fake_runner([])
    svc = SpeakerDiarizationService(runner)
    svc.diarize(_valid_audio())
