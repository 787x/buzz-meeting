import threading
from typing import Any, Optional

import sounddevice

from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)


class SoundDeviceAudioSource(AudioSource):
    """Capture mono microphone audio with ``sounddevice.InputStream``."""

    def __init__(
        self,
        device_index: Optional[int],
        sample_rate: int,
        sounddevice_module: Any = sounddevice,
    ) -> None:
        self.device_index = device_index
        self._sample_rate = sample_rate
        self._sounddevice = sounddevice_module
        self._stream = None
        self._active = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        with self._lock:
            if self._active:
                raise AudioSourceError("Audio source is already active")

            stream = None

            def stream_callback(in_data, frame_count, time_info, status):
                # Keep the current sounddevice behavior: status is ignored and
                # callback exceptions are left to sounddevice/PortAudio.
                on_audio(in_data.reshape(-1))

            try:
                stream = self._sounddevice.InputStream(
                    samplerate=self.sample_rate,
                    device=self.device_index,
                    dtype="float32",
                    channels=1,
                    callback=stream_callback,
                )
                stream.start()
            except Exception as exc:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                raise AudioSourceError(str(exc)) from exc

            self._stream = stream
            self._active = True

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            if stream is None:
                return

            try:
                stream.stop()
            finally:
                try:
                    stream.close()
                finally:
                    self._stream = None
                    self._active = False
