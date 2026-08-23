import threading
from typing import Optional

import numpy as np

from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)


class FakeAudioSource(AudioSource):
    """Deterministic tests-only implementation of the AudioSource contract."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self._sample_rate = sample_rate
        self._on_audio: Optional[AudioFrameCallback] = None
        self._on_error: Optional[AudioErrorCallback] = None
        self._active = False
        self._lock = threading.Lock()
        self.start_count = 0
        self.stop_count = 0
        self.started_event = threading.Event()
        self.stopped_event = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def started(self) -> bool:
        with self._lock:
            return self._active

    @property
    def stopped(self) -> bool:
        return self.stopped_event.is_set()

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        with self._lock:
            if self._active:
                raise AudioSourceError("Audio source is already active")
            self._on_audio = on_audio
            self._on_error = on_error
            self._active = True
            self.start_count += 1
            self.stopped_event.clear()
            self.started_event.set()

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._on_audio = None
            self._on_error = None
            self.stop_count += 1
            self.stopped_event.set()

    def deliver(self, samples: np.ndarray) -> None:
        if samples.dtype != np.float32 or samples.ndim != 1:
            raise ValueError("FakeAudioSource expects mono float32 samples")

        with self._lock:
            if not self._active or self._on_audio is None:
                raise RuntimeError("Audio source is not active")
            on_audio = self._on_audio

        on_audio(samples)

    def fail(self, error: Exception) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("Audio source is not active")
            on_error = self._on_error

        if on_error is not None:
            on_error(error)
