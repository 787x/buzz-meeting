"""Pure speaker diarization domain — non-Qt, non-Settings, non-NeMo, non-torch.

Provides DTOs, error taxonomy, runner protocol, backend enum, and a thin
validation service that delegates actual inference to a concrete runner.

Only NumPy is used; no filesystem I/O, network, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np


# ── DTOs ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpeakerDiarizationAudio:
    """Mono 16 kHz float32 waveform handed to the diarization service."""

    waveform: np.ndarray
    sample_rate: int


@dataclass(frozen=True, slots=True)
class SpeakerDiarizationTurn:
    """One speaker time range scoped to a single diarization invocation."""

    speaker_index: int
    start_ms: int
    end_ms: int


# ── Backend enum ──────────────────────────────────────────────────────────────


class SpeakerDiarizationBackend(Enum):
    MSDD = "msdd"
    SORTFORMER = "sortformer"


# ── Errors ────────────────────────────────────────────────────────────────────


class SpeakerDiarizationError(Exception):
    """Base error for speaker diarization."""


class SpeakerDiarizationConfigError(SpeakerDiarizationError):
    """Invalid public inputs or configuration."""


class SpeakerDiarizationUnavailableError(SpeakerDiarizationError):
    """Concrete backend, import, or model availability failure."""


# ── Runner protocol ──────────────────────────────────────────────────────────


class SpeakerDiarizationRunner(Protocol):
    """Narrow interface for a concrete diarization backend."""

    def diarize(
        self,
        audio: SpeakerDiarizationAudio,
    ) -> Sequence[SpeakerDiarizationTurn]:
        ...


# ── Validation helpers ────────────────────────────────────────────────────────

_REQUIRED_SAMPLE_RATE = 16_000


def _validate_audio(audio: SpeakerDiarizationAudio) -> None:
    """Validate a SpeakerDiarizationAudio input.

    Raises SpeakerDiarizationConfigError on any violation.
    """
    # sample_rate
    if isinstance(audio.sample_rate, bool):
        raise SpeakerDiarizationConfigError("sample_rate must be an int, not bool")
    if not isinstance(audio.sample_rate, int):
        raise SpeakerDiarizationConfigError("sample_rate must be an int")
    if audio.sample_rate != _REQUIRED_SAMPLE_RATE:
        raise SpeakerDiarizationConfigError(
            f"sample_rate must be {_REQUIRED_SAMPLE_RATE}, " f"got {audio.sample_rate}"
        )

    # waveform
    if not isinstance(audio.waveform, np.ndarray):
        raise SpeakerDiarizationConfigError("waveform must be a numpy.ndarray")
    if audio.waveform.ndim != 1:
        raise SpeakerDiarizationConfigError(
            f"waveform must be one-dimensional, got ndim={audio.waveform.ndim}"
        )
    if audio.waveform.dtype != np.float32:
        raise SpeakerDiarizationConfigError(
            f"waveform dtype must be float32, got {audio.waveform.dtype}"
        )


def _validate_turn(turn: SpeakerDiarizationTurn) -> None:
    """Validate a single SpeakerDiarizationTurn.

    Raises SpeakerDiarizationConfigError on any violation.
    """
    if isinstance(turn.speaker_index, bool):
        raise SpeakerDiarizationConfigError("speaker_index must be an int, not bool")
    if not isinstance(turn.speaker_index, int):
        raise SpeakerDiarizationConfigError("speaker_index must be an int")
    if turn.speaker_index < 0:
        raise SpeakerDiarizationConfigError(
            f"speaker_index must be >= 0, got {turn.speaker_index}"
        )

    if isinstance(turn.start_ms, bool):
        raise SpeakerDiarizationConfigError("start_ms must be an int, not bool")
    if not isinstance(turn.start_ms, int):
        raise SpeakerDiarizationConfigError("start_ms must be an int")
    if turn.start_ms < 0:
        raise SpeakerDiarizationConfigError(
            f"start_ms must be >= 0, got {turn.start_ms}"
        )

    if isinstance(turn.end_ms, bool):
        raise SpeakerDiarizationConfigError("end_ms must be an int, not bool")
    if not isinstance(turn.end_ms, int):
        raise SpeakerDiarizationConfigError("end_ms must be an int")
    if turn.end_ms < turn.start_ms:
        raise SpeakerDiarizationConfigError(
            f"end_ms must be >= start_ms, got "
            f"end_ms={turn.end_ms} < start_ms={turn.start_ms}"
        )


# ── Service ───────────────────────────────────────────────────────────────────


class SpeakerDiarizationService:
    """Thin validation-and-sort layer around a SpeakerDiarizationRunner.

    Responsibilities:
    1. validate audio
    2. invoke runner exactly once
    3. materialise result
    4. validate each turn
    5. stable sort by start_ms
    6. return immutable tuple

    No retries, no sleeps, no Qt, no Settings, no persistence.
    """

    def __init__(self, runner: SpeakerDiarizationRunner) -> None:
        self._runner = runner

    def diarize(
        self,
        audio: SpeakerDiarizationAudio,
    ) -> tuple[SpeakerDiarizationTurn, ...]:
        """Run diarization on a single audio input and return validated turns."""
        _validate_audio(audio)

        try:
            raw_turns = self._runner.diarize(audio)
        except SpeakerDiarizationError:
            raise
        except Exception as exc:
            raise SpeakerDiarizationError("Runner diarization failed") from exc

        for turn in raw_turns:
            _validate_turn(turn)

        # Stable sort by start_ms; equal start_ms preserves original order.
        sorted_turns = sorted(raw_turns, key=lambda t: t.start_ms)

        return tuple(sorted_turns)
