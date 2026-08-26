from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Callable
from enum import Enum, auto
from typing import Optional

import numpy as np

from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderError,
    MeetingRecorderState,
    MeetingRecordingResult,
)


class MeetingAudioFanoutState(Enum):
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class LiveSubscriptionAudioSource(AudioSource):
    """An AudioSource view over the live branch of a meeting fan-out.

    This adapter owns only a live subscription. Stopping it guarantees that no
    later live callback starts, but deliberately does not stop meeting capture
    or the durable recorder owned by ``MeetingAudioFanout``.
    """

    def __init__(self, fanout: "MeetingAudioFanout") -> None:
        self._fanout = fanout

    @property
    def sample_rate(self) -> int:
        return self._fanout.sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        self._fanout._start_live_subscription(on_audio, on_error)

    def stop(self) -> None:
        self._fanout._stop_live_subscription()


class MeetingAudioFanout:
    """Fan one selected AudioSource out to archival and live consumers.

    The fan-out, not the live transcriber, owns the underlying source and
    recorder lifecycles. The live subscription can therefore end after a live
    backend failure without ending the meeting recording.
    """

    def __init__(
        self,
        source: AudioSource,
        recorder: MeetingRecorder,
        *,
        on_source_error: Optional[AudioErrorCallback] = None,
        on_live_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        if recorder.sample_rate != source.sample_rate:
            raise ValueError(
                "Meeting recorder sample rate must match the underlying audio source"
            )

        self.source = source
        self.recorder = recorder
        self.live_source = LiveSubscriptionAudioSource(self)
        self._external_source_error = on_source_error
        self._external_live_error = on_live_error

        self._condition = threading.Condition()
        self._state = MeetingAudioFanoutState.CREATED
        self._delivery_open = False
        self._controller_in_flight = 0
        self._controller_callback_threads: Counter[int] = Counter()
        self._source_stop_pending = False
        self._source_error: Optional[Exception] = None
        self._source_runtime_failed = False
        self._source_start_failed = False
        self._source_start_attempted = False
        self._source_cleanup_complete = True
        self._source_cleanup_error: Optional[Exception] = None
        self._stop_requested = False
        self._recorder_start_in_progress = False
        self._source_start_in_progress = False
        self._stop_result: Optional[MeetingRecordingResult] = None
        self._stop_error: Optional[Exception] = None
        self._stop_in_progress = False
        self._stop_waiting_for_recorder = False

        self._live_active = False
        self._live_on_audio: Optional[AudioFrameCallback] = None
        self._live_on_error: Optional[AudioErrorCallback] = None
        self._live_in_flight = 0
        self._live_callback_threads: Counter[int] = Counter()

    @property
    def sample_rate(self) -> int:
        return self.source.sample_rate

    @property
    def state(self) -> MeetingAudioFanoutState:
        with self._condition:
            return self._state

    @property
    def source_error(self) -> Optional[Exception]:
        with self._condition:
            return self._source_error

    def start(self) -> None:
        with self._condition:
            if self._state != MeetingAudioFanoutState.CREATED:
                raise AudioSourceError(
                    f"Cannot start meeting audio fan-out in state {self._state.name}"
                )
            self._state = MeetingAudioFanoutState.STARTING
            self._recorder_start_in_progress = True

        try:
            self.recorder.start()
        except Exception:
            with self._condition:
                self._recorder_start_in_progress = False
                self._state = MeetingAudioFanoutState.FAILED
                self._condition.notify_all()
            raise

        with self._condition:
            self._recorder_start_in_progress = False
            if self._stop_requested:
                startup_cancelled = True
            else:
                startup_cancelled = False
                self._delivery_open = True
                self._source_start_attempted = True
                self._source_cleanup_complete = False
                self._source_start_in_progress = True
            self._condition.notify_all()

        if startup_cancelled:
            self.stop()
            raise AudioSourceError("Meeting audio fan-out was stopped during startup")

        try:
            self.source.start(self._on_audio, self._on_source_error)
        except Exception as exc:
            with self._condition:
                self._source_start_in_progress = False
                self._source_start_failed = True
                self._delivery_open = False
                self._stop_requested = True
                self._state = MeetingAudioFanoutState.FAILED
                if self._source_error is None:
                    self._source_error = exc
                self._condition.notify_all()
            try:
                self.stop()
            except Exception:
                logging.exception(
                    "Could not fully clean up meeting audio source startup failure"
                )
            if isinstance(exc, AudioSourceError):
                raise
            raise AudioSourceError(str(exc)) from exc

        with self._condition:
            self._source_start_in_progress = False
            if self._source_error is None and not self._stop_requested:
                self._state = MeetingAudioFanoutState.RUNNING
                self._condition.notify_all()
                return
            source_error = self._source_error
            stopped_during_start = self._stop_requested
            self._condition.notify_all()

        try:
            self.stop()
        except Exception:
            if source_error is None:
                raise
            logging.exception("Could not fully clean up failed meeting audio startup")
        if source_error is None and stopped_during_start:
            raise AudioSourceError("Meeting audio fan-out was stopped during startup")
        assert source_error is not None
        if isinstance(source_error, AudioSourceError):
            raise source_error
        raise AudioSourceError(str(source_error)) from source_error

    def stop(self) -> MeetingRecordingResult:
        cached_stop_result: Optional[MeetingRecordingResult] = None
        while True:
            with self._condition:
                if self._state == MeetingAudioFanoutState.CREATED:
                    result = self.recorder.stop()
                    self._stop_requested = True
                    self._state = MeetingAudioFanoutState.STOPPED
                    self._stop_result = result
                    self._condition.notify_all()
                    return self._stop_result
                if self._source_cleanup_complete and self._stop_result is not None:
                    cached_stop_result = self._stop_result
                    break
                if self._stop_in_progress:
                    if self._stop_waiting_for_recorder:
                        reentrant_recorder_stop = True
                        cleanup_error = self._source_cleanup_error
                        break
                    self._condition.wait_for(lambda: not self._stop_in_progress)
                    continue

                reentrant_recorder_stop = False
                cleanup_error = None
                self._stop_in_progress = True
                self._stop_requested = True
                self._delivery_open = False
                if self._state != MeetingAudioFanoutState.FAILED:
                    self._state = MeetingAudioFanoutState.STOPPING
                self._condition.notify_all()
                break

        if cached_stop_result is not None:
            # A terminal recording result is cached before writer-owned failure
            # cleanup. recorder.stop() is caller-sensitive: it returns directly
            # on the writer thread, but external callers wait for and reap it.
            self.recorder.stop()
            return cached_stop_result

        if reentrant_recorder_stop:
            result = self.recorder.stop()
            with self._condition:
                if self._stop_result is None:
                    self._stop_result = result
                self._condition.notify_all()
            if cleanup_error is not None:
                self._raise_source_error(cleanup_error)
            return result

        source_stop_error: Optional[Exception] = None
        source_error_callback: Optional[AudioErrorCallback] = None
        operation_error: Optional[Exception] = None
        result: Optional[MeetingRecordingResult] = None
        try:
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._recorder_start_in_progress
                    and not self._source_start_in_progress
                )
                should_stop_source = (
                    self._source_start_attempted and not self._source_cleanup_complete
                )
                may_cancel_empty_start = (
                    self._source_start_failed or not self._source_start_attempted
                )

            if should_stop_source:
                try:
                    self.source.stop()
                except Exception as exc:
                    source_stop_error = exc
                    with self._condition:
                        self._source_cleanup_error = exc
                        self._state = MeetingAudioFanoutState.FAILED
                        if self._source_error is None:
                            self._source_error = exc
                            source_error_callback = self._external_source_error
                else:
                    with self._condition:
                        self._source_cleanup_complete = True
                        self._source_cleanup_error = None

            self._wait_for_controller_callbacks()
            cancel_empty_start = (
                may_cancel_empty_start and self.recorder.accepted_sample_count == 0
            )
            with self._condition:
                self._stop_waiting_for_recorder = True
                self._condition.notify_all()
            if cancel_empty_start:
                try:
                    result = self.recorder.cancel_empty_start()
                except MeetingRecorderError:
                    result = self.recorder.stop()
            else:
                result = self.recorder.stop()

            with self._condition:
                self._stop_result = result
                if (
                    self._source_cleanup_complete
                    and not self._source_runtime_failed
                    and not self._source_start_failed
                    and result.state != MeetingRecorderState.FAILED
                ):
                    self._state = MeetingAudioFanoutState.STOPPED
                else:
                    self._state = MeetingAudioFanoutState.FAILED
                self._condition.notify_all()
        except Exception as exc:
            operation_error = exc
            with self._condition:
                self._stop_error = exc
                self._state = MeetingAudioFanoutState.FAILED
                self._condition.notify_all()
        finally:
            with self._condition:
                self._stop_waiting_for_recorder = False
                self._stop_in_progress = False
                self._condition.notify_all()

        if source_stop_error is not None:
            self._safe_error_callback(
                source_error_callback,
                source_stop_error,
                "meeting source",
            )
        if source_stop_error is not None:
            self._raise_source_error(source_stop_error)
        if operation_error is not None:
            raise operation_error
        assert result is not None
        return result

    def _on_audio(self, samples: np.ndarray) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if not self._delivery_open:
                return
            self._controller_in_flight += 1
            self._controller_callback_threads[thread_id] += 1

        request_recorder_stop = False
        try:
            try:
                self.recorder.enqueue(samples)
            except MeetingRecorderError:
                # Invalid producer data is a recorder contract failure, but it
                # must not escape into the underlying source or stop live audio.
                logging.exception("Meeting recorder rejected source audio")

            with self._condition:
                if self._live_active and self._live_on_audio is not None:
                    live_callback = self._live_on_audio
                    self._live_in_flight += 1
                    self._live_callback_threads[thread_id] += 1
                else:
                    live_callback = None

            if live_callback is not None:
                try:
                    live_callback(samples)
                except Exception as exc:
                    self._handle_live_callback_failure(exc)
                finally:
                    with self._condition:
                        self._live_in_flight -= 1
                        self._live_callback_threads[thread_id] -= 1
                        if self._live_callback_threads[thread_id] == 0:
                            del self._live_callback_threads[thread_id]
                        self._condition.notify_all()
        finally:
            with self._condition:
                self._controller_in_flight -= 1
                self._controller_callback_threads[thread_id] -= 1
                if self._controller_callback_threads[thread_id] == 0:
                    del self._controller_callback_threads[thread_id]
                if self._source_stop_pending and self._controller_in_flight == 0:
                    self._source_stop_pending = False
                    request_recorder_stop = True
                self._condition.notify_all()
            if request_recorder_stop:
                self.recorder.request_stop()

    def _on_source_error(self, error: Exception) -> None:
        live_error_callback: Optional[AudioErrorCallback]
        external_error_callback: Optional[AudioErrorCallback]
        request_recorder_stop = False
        with self._condition:
            if self._source_error is not None:
                return
            self._source_error = error
            self._source_runtime_failed = True
            self._stop_requested = True
            self._delivery_open = False
            self._state = MeetingAudioFanoutState.FAILED
            if self._controller_in_flight == 0:
                request_recorder_stop = True
            else:
                self._source_stop_pending = True
            live_error_callback = self._live_on_error if self._live_active else None
            external_error_callback = self._external_source_error
            self._condition.notify_all()

        if request_recorder_stop:
            self.recorder.request_stop()
        self._safe_error_callback(live_error_callback, error, "live source")
        self._safe_error_callback(external_error_callback, error, "meeting source")

    @staticmethod
    def _raise_source_error(error: Exception) -> None:
        if isinstance(error, AudioSourceError):
            raise error
        raise AudioSourceError(str(error)) from error

    def _handle_live_callback_failure(self, cause: Exception) -> None:
        error = AudioSourceError(f"Live audio callback failed: {cause}")
        with self._condition:
            if not self._live_active:
                return
            live_error_callback = self._live_on_error
            external_error_callback = self._external_live_error
            self._live_active = False
            self._live_on_audio = None
            self._live_on_error = None
            self._condition.notify_all()

        self._safe_error_callback(live_error_callback, error, "live subscription")
        self._safe_error_callback(external_error_callback, error, "external live")

    def _start_live_subscription(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback],
    ) -> None:
        with self._condition:
            if self._state != MeetingAudioFanoutState.RUNNING:
                detail = (
                    f": {self._source_error}" if self._source_error is not None else ""
                )
                raise AudioSourceError(
                    "Meeting audio fan-out is not active"
                    f" (state={self._state.name}){detail}"
                )
            if self._live_active:
                raise AudioSourceError("Live audio subscription is already active")
            self._live_on_audio = on_audio
            self._live_on_error = on_error
            self._live_active = True

    def _stop_live_subscription(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if not self._live_active and self._live_in_flight == 0:
                self._live_on_audio = None
                self._live_on_error = None
                return
            self._live_active = False
            self._live_on_audio = None
            self._live_on_error = None
            own_callbacks = self._live_callback_threads.get(thread_id, 0)
            self._condition.wait_for(lambda: self._live_in_flight <= own_callbacks)

    def _wait_for_controller_callbacks(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            own_callbacks = self._controller_callback_threads.get(thread_id, 0)
            self._condition.wait_for(
                lambda: self._controller_in_flight <= own_callbacks
            )

    @staticmethod
    def _safe_error_callback(
        callback: Optional[Callable[[Exception], None]],
        error: Exception,
        label: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(error)
        except Exception:
            logging.exception("%s error callback failed", label.capitalize())


__all__ = [
    "LiveSubscriptionAudioSource",
    "MeetingAudioFanout",
    "MeetingAudioFanoutState",
]
