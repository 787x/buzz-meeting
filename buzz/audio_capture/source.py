from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import numpy as np


AudioFrameCallback = Callable[[np.ndarray], None]
AudioErrorCallback = Callable[[Exception], None]


class AudioSourceError(RuntimeError):
    """Raised when an audio source cannot be started."""


class AudioSource(ABC):
    """A source of mono PCM audio for live transcription.

    Audio callbacks receive a one-dimensional ``numpy.float32`` array with
    shape ``(frames,)``. Callbacks may run on any capture thread and have no Qt
    thread affinity. A source delivers callbacks in order, one at a time.

    The callback buffer is borrowed and is guaranteed to remain valid only for
    the duration of the callback. Consumers that retain samples after the
    callback returns must copy them. Consumers should return quickly and must
    not perform I/O or transcription in the callback.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The sample rate of delivered PCM audio, in Hz."""

    @abstractmethod
    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        """Start capture, returning once the source is active."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release its resources.

        Calling this before ``start()`` or more than once is a safe no-op.
        """
