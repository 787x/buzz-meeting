from collections import deque
from typing import Deque

import numpy as np


class LiveSegmenter:
    """Deterministically split continuous mono PCM into speech utterances."""

    ANALYSIS_FRAME_SECONDS = 0.020
    SPEECH_CONFIRMATION_SECONDS = 0.100
    NATURAL_PAUSE_SECONDS = 0.600
    MIN_ENDPOINT_SECONDS = 1.500
    PRE_ROLL_SECONDS = 0.200
    FORCED_SEARCH_SECONDS = 2.500
    FORCED_QUIET_RATIO = 0.75
    SPEECH_OFF_RATIO = 0.60
    _INGEST_FRAMES = 50

    def __init__(
        self,
        sample_rate: int,
        speech_threshold: float,
        *,
        max_utterance_seconds: float = 12.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if speech_threshold < 0:
            raise ValueError("speech_threshold must not be negative")
        if max_utterance_seconds <= 0:
            raise ValueError("max_utterance_seconds must be positive")

        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.max_utterance_seconds = max_utterance_seconds

        self._frame_samples = self._seconds_to_samples(
            self.ANALYSIS_FRAME_SECONDS,
        )
        self._confirmation_samples = self._seconds_to_samples(
            self.SPEECH_CONFIRMATION_SECONDS,
        )
        self._natural_pause_samples = self._seconds_to_samples(
            self.NATURAL_PAUSE_SECONDS,
        )
        self._minimum_endpoint_samples = self._seconds_to_samples(
            self.MIN_ENDPOINT_SECONDS,
        )
        self._pre_roll_samples = self._seconds_to_samples(self.PRE_ROLL_SECONDS)
        self._forced_search_samples = self._seconds_to_samples(
            self.FORCED_SEARCH_SECONDS,
        )
        self._max_utterance_samples = self._seconds_to_samples(
            max_utterance_seconds,
        )
        self._max_buffered_samples = (
            self._max_utterance_samples + self._frame_samples - 1
        )

        positive_floor = float(np.finfo(np.float32).eps)
        self._speech_on = max(float(speech_threshold), positive_floor)
        self._speech_off = self.SPEECH_OFF_RATIO * self._speech_on

        self._chunks: Deque[tuple[int, np.ndarray]] = deque()
        self._energy_frames: Deque[tuple[int, int, float]] = deque()
        self._buffer_start = 0
        self._buffered_samples = 0
        self._total_samples = 0
        self._next_frame_start = 0
        self._analysis_tail = np.empty(0, dtype=np.float32)

        self._has_speech = False
        self._speech_run_samples = 0
        self._silence_run_samples = 0
        self._utterance_start = 0
        self._pause_start: int | None = None
        self._last_speech_end: int | None = None

    @property
    def buffered_sample_count(self) -> int:
        return self._buffered_samples

    @property
    def max_buffered_sample_count(self) -> int:
        return self._max_buffered_samples

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        self._validate_samples(samples)
        if samples.size == 0:
            return []

        utterances: list[np.ndarray] = []
        offset = 0
        max_ingest_samples = self._INGEST_FRAMES * self._frame_samples

        while offset < samples.size:
            available = self._max_buffered_samples - self._buffered_samples
            if available <= 0:
                raise RuntimeError("LiveSegmenter buffer invariant violated")

            take = min(samples.size - offset, max_ingest_samples, available)
            owned = np.array(samples[offset:offset + take], dtype=np.float32, copy=True)
            self._append_chunk(owned)
            self._analyse(owned, utterances)
            offset += take

        return utterances

    def flush(self) -> list[np.ndarray]:
        utterances: list[np.ndarray] = []

        if self._analysis_tail.size:
            start = self._next_frame_start
            end = self._total_samples
            energy = self._rms(self._analysis_tail)
            self._analysis_tail = np.empty(0, dtype=np.float32)
            self._next_frame_start = end
            self._process_frame(start, end, energy, utterances)

        if self._has_speech:
            cut = self._total_samples
            if self._last_speech_end is not None and self._silence_run_samples:
                cut = min(
                    cut,
                    self._last_speech_end + self._natural_pause_samples // 2,
                )
            if cut > self._utterance_start:
                utterances.append(self._materialize(self._utterance_start, cut))

        self._reset()
        return utterances

    def _seconds_to_samples(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.sample_rate)))

    @staticmethod
    def _validate_samples(samples: np.ndarray) -> None:
        if not isinstance(samples, np.ndarray):
            raise ValueError("samples must be a numpy array")
        if samples.dtype != np.float32:
            raise ValueError("samples must have dtype numpy.float32")
        if samples.ndim != 1:
            raise ValueError("samples must be mono with shape (frames,)")

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        values = samples.astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(values * values)))

    def _append_chunk(self, samples: np.ndarray) -> None:
        start = self._total_samples
        self._chunks.append((start, samples))
        self._total_samples += samples.size
        self._buffered_samples += samples.size

    def _analyse(
        self,
        samples: np.ndarray,
        utterances: list[np.ndarray],
    ) -> None:
        if self._analysis_tail.size:
            analysis = np.concatenate((self._analysis_tail, samples))
        else:
            analysis = samples

        frame_count = analysis.size // self._frame_samples
        analysed_count = frame_count * self._frame_samples
        if frame_count:
            frames = analysis[:analysed_count].reshape(
                frame_count,
                self._frame_samples,
            )
            frame_values = frames.astype(np.float64, copy=False)
            energies = np.sqrt(np.mean(frame_values * frame_values, axis=1))

            for energy in energies:
                start = self._next_frame_start
                end = start + self._frame_samples
                self._next_frame_start = end
                self._process_frame(start, end, float(energy), utterances)

        self._analysis_tail = np.array(
            analysis[analysed_count:],
            dtype=np.float32,
            copy=True,
        )

    def _process_frame(
        self,
        start: int,
        end: int,
        energy: float,
        utterances: list[np.ndarray],
    ) -> None:
        self._energy_frames.append((start, end, energy))
        frame_length = end - start

        if not self._has_speech:
            if energy > self._speech_on:
                self._speech_run_samples += frame_length
            else:
                self._speech_run_samples = 0

            idle_keep = self._pre_roll_samples + self._speech_run_samples
            self._discard_before(max(self._buffer_start, end - idle_keep))

            if self._speech_run_samples >= self._confirmation_samples:
                self._has_speech = True
                self._utterance_start = self._buffer_start
                self._last_speech_end = end
                self._silence_run_samples = 0
                self._pause_start = None
            return

        if energy >= self._speech_off:
            self._last_speech_end = end
            self._silence_run_samples = 0
            self._pause_start = None
        else:
            if self._silence_run_samples == 0:
                self._pause_start = start
            self._silence_run_samples += frame_length

        utterance_length = end - self._utterance_start
        if (
            self._silence_run_samples >= self._natural_pause_samples
            and self._pause_start is not None
        ):
            cut = self._pause_start + self._natural_pause_samples // 2
            self._emit_natural(cut, utterances)
            return

        if utterance_length >= self._max_utterance_samples:
            deadline = self._utterance_start + self._max_utterance_samples
            cut = self._forced_cut_point(deadline)
            self._emit_forced(cut, end, utterances)

    def _forced_cut_point(self, deadline: int) -> int:
        search_start = max(
            self._utterance_start + self._minimum_endpoint_samples,
            deadline - self._forced_search_samples,
        )
        candidates = [
            frame
            for frame in self._energy_frames
            if search_start <= (frame[0] + frame[1]) // 2 <= deadline
        ]
        if not candidates:
            return deadline

        energies = np.fromiter(
            (frame[2] for frame in candidates),
            dtype=np.float64,
            count=len(candidates),
        )
        minimum = float(energies.min())
        median = float(np.median(energies))
        if not (
            minimum < self._speech_off
            or minimum < self.FORCED_QUIET_RATIO * median
        ):
            return deadline

        # Choose the latest equal minimum so tie-breaking is stable and explicit.
        minimum_index = int(np.flatnonzero(energies == minimum)[-1])
        selected = candidates[minimum_index]
        return (selected[0] + selected[1]) // 2

    def _emit_natural(
        self,
        cut: int,
        utterances: list[np.ndarray],
    ) -> None:
        utterances.append(self._materialize(self._utterance_start, cut))
        self._discard_before(cut)
        self._clear_speech_state()

    def _emit_forced(
        self,
        cut: int,
        analysed_end: int,
        utterances: list[np.ndarray],
    ) -> None:
        utterances.append(self._materialize(self._utterance_start, cut))
        self._discard_before(cut)

        remaining_frames = [
            frame for frame in self._energy_frames if frame[1] > cut
        ]
        self._has_speech = True
        self._speech_run_samples = self._confirmation_samples
        self._utterance_start = cut
        self._pause_start = None
        self._silence_run_samples = 0
        self._last_speech_end = cut

        last_active_end: int | None = None
        for frame_start, frame_end, energy in remaining_frames:
            if energy >= self._speech_off:
                last_active_end = frame_end

        if last_active_end is not None:
            self._last_speech_end = last_active_end
            self._silence_run_samples = max(0, analysed_end - last_active_end)
            if self._silence_run_samples:
                self._pause_start = last_active_end
        else:
            self._silence_run_samples = max(0, analysed_end - cut)
            self._pause_start = cut

    def _materialize(self, start: int, end: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for chunk_start, chunk in self._chunks:
            chunk_end = chunk_start + chunk.size
            if chunk_end <= start:
                continue
            if chunk_start >= end:
                break
            local_start = max(start, chunk_start) - chunk_start
            local_end = min(end, chunk_end) - chunk_start
            parts.append(chunk[local_start:local_end])

        if not parts:
            return np.empty(0, dtype=np.float32)
        if len(parts) == 1:
            return np.array(parts[0], dtype=np.float32, copy=True)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _discard_before(self, position: int) -> None:
        position = min(max(position, self._buffer_start), self._total_samples)
        removed = position - self._buffer_start
        if removed == 0:
            return

        while self._chunks:
            chunk_start, chunk = self._chunks[0]
            chunk_end = chunk_start + chunk.size
            if chunk_end <= position:
                self._chunks.popleft()
                continue
            if chunk_start < position:
                offset = position - chunk_start
                self._chunks[0] = (position, chunk[offset:])
            break

        self._buffer_start = position
        self._buffered_samples -= removed
        while self._energy_frames and self._energy_frames[0][1] <= position:
            self._energy_frames.popleft()

    def _clear_speech_state(self) -> None:
        self._has_speech = False
        self._speech_run_samples = 0
        self._silence_run_samples = 0
        self._utterance_start = self._buffer_start
        self._pause_start = None
        self._last_speech_end = None

    def _reset(self) -> None:
        self._chunks.clear()
        self._energy_frames.clear()
        self._buffer_start = 0
        self._buffered_samples = 0
        self._total_samples = 0
        self._next_frame_start = 0
        self._analysis_tail = np.empty(0, dtype=np.float32)
        self._clear_speech_state()
