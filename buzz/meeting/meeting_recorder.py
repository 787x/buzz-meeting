from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import threading
from typing import Optional, Protocol

import numpy as np
import soundfile


class MeetingRecorderState(Enum):
    CREATED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class MeetingRecorderError(RuntimeError):
    """Base error for meeting-recorder failures."""


class MeetingRecorderStateError(MeetingRecorderError):
    """Raised when the recorder API is used in an invalid lifecycle state."""


class MeetingRecorderInputError(MeetingRecorderError):
    """Raised when a producer violates the mono float32 input contract."""


class MeetingRecorderOperationalError(MeetingRecorderError):
    """A terminal failure that prevents a complete archive from being published."""


@dataclass(frozen=True)
class MeetingRecordingResult:
    output_path: Path
    sample_rate: int
    sample_count: int
    duration_seconds: float
    state: MeetingRecorderState
    error: Optional[MeetingRecorderOperationalError]
    published: bool


class _ArchiveWriter(Protocol):
    def write(self, pcm16: np.ndarray) -> None:
        ...

    def flush(self) -> None:
        ...

    def finalize(self) -> None:
        ...

    def publish(self) -> None:
        ...

    def discard(self) -> None:
        ...

    def close_after_failure(self) -> None:
        ...


_WriterFactory = Callable[[Path, int], _ArchiveWriter]
_ErrorCallback = Callable[[MeetingRecorderOperationalError], None]


def _sync_directory_best_effort(directory: Path) -> None:
    """Best-effort post-commit directory durability on POSIX."""

    try:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        logging.warning(
            "Meeting audio was published, but its directory could not be synced",
            exc_info=True,
        )


def _publish_posix_no_replace(partial_path: Path, output_path: Path) -> None:
    """Use the production POSIX hard-link publish path without replacement."""

    # POSIX rename replaces an existing destination. A same-directory hard link
    # instead gives us atomic O_EXCL-like destination creation.
    os.link(partial_path, output_path)
    try:
        partial_path.unlink()
    except Exception:
        logging.warning(
            "Meeting audio was published, but its partial link could not be removed",
            exc_info=True,
        )
    _sync_directory_best_effort(output_path.parent)


def _publish_no_replace(partial_path: Path, output_path: Path) -> None:
    """Atomically create output_path without replacing an existing file.

    Returning marks the publish commit point. Any post-commit cleanup or
    directory-sync failure is deliberately best effort and cannot make the
    already-visible final file unpublished.
    """

    if os.name == "nt":
        # Windows rename is atomic within a volume and fails if the destination
        # already exists.
        os.rename(partial_path, output_path)
        return

    _publish_posix_no_replace(partial_path, output_path)


class _SoundFileRF64Writer:
    """Writer-thread-owned RF64/PCM16 file lifecycle."""

    def __init__(self, output_path: Path, sample_rate: int) -> None:
        self.output_path = output_path
        self.partial_path = output_path.with_name(output_path.name + ".partial")
        self._raw_file = None
        self._sound_file = None

        if not soundfile.check_format("RF64", "PCM_16"):
            raise MeetingRecorderOperationalError(
                "The current SoundFile runtime does not support RF64 with PCM_16"
            )
        if not output_path.parent.is_dir():
            raise MeetingRecorderOperationalError(
                f"Recording output directory does not exist: {output_path.parent}"
            )
        if output_path.exists():
            raise MeetingRecorderOperationalError(
                f"Recording output already exists: {output_path}"
            )
        if self.partial_path.exists():
            raise MeetingRecorderOperationalError(
                f"Recording partial output already exists: {self.partial_path}"
            )

        try:
            self._raw_file = self.partial_path.open("xb")
            self._sound_file = soundfile.SoundFile(
                self._raw_file,
                mode="w",
                samplerate=sample_rate,
                channels=1,
                subtype="PCM_16",
                format="RF64",
                closefd=False,
            )
            if self._sound_file.format != "RF64":
                raise MeetingRecorderOperationalError(
                    "SoundFile opened the meeting archive with a non-RF64 format"
                )
            if self._sound_file.subtype != "PCM_16":
                raise MeetingRecorderOperationalError(
                    "SoundFile opened the meeting archive with a non-PCM16 subtype"
                )
        except Exception:
            self.close_after_failure()
            raise

    def write(self, pcm16: np.ndarray) -> None:
        assert self._sound_file is not None
        self._sound_file.write(pcm16)

    def flush(self) -> None:
        assert self._sound_file is not None
        assert self._raw_file is not None
        self._sound_file.flush()
        self._raw_file.flush()

    def finalize(self) -> None:
        assert self._sound_file is not None
        assert self._raw_file is not None

        primary_error: Optional[Exception] = None
        try:
            self._sound_file.flush()
        except Exception as exc:
            primary_error = exc
        try:
            self._sound_file.close()
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
        finally:
            self._sound_file = None

        try:
            self._raw_file.flush()
            os.fsync(self._raw_file.fileno())
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
        try:
            self._raw_file.close()
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
        finally:
            self._raw_file = None

        if primary_error is not None:
            raise primary_error

    def publish(self) -> None:
        _publish_no_replace(self.partial_path, self.output_path)

    def discard(self) -> None:
        self.partial_path.unlink(missing_ok=True)

    def close_after_failure(self) -> None:
        if self._sound_file is not None:
            try:
                self._sound_file.close()
            except Exception:
                logging.exception("Could not close failed meeting SoundFile")
            finally:
                self._sound_file = None
        if self._raw_file is not None:
            try:
                self._raw_file.close()
            except Exception:
                logging.exception("Could not close failed meeting audio file")
            finally:
                self._raw_file = None


class MeetingRecorder:
    """Durably archive one mono float32 audio source on a dedicated writer thread.

    ``enqueue()`` is intended for an audio callback. It performs one owned
    float32 copy and bounded bookkeeping, but never performs file I/O or PCM16
    conversion. Operational failures return ``False`` and are reported through
    ``on_error`` exactly once so a live consumer can continue independently.
    """

    _WRITER_THREAD_NAME = "meeting-audio-writer"
    _PRIVATE_FLUSH_SECONDS = 5.0

    def __init__(
        self,
        output_path: os.PathLike[str] | str,
        sample_rate: int,
        *,
        max_buffer_seconds: float = 60.0,
        on_error: Optional[_ErrorCallback] = None,
        _writer_factory: Optional[_WriterFactory] = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if max_buffer_seconds <= 0:
            raise ValueError("max_buffer_seconds must be positive")

        output_path = Path(output_path)
        if output_path.suffix.lower() != ".wav":
            raise ValueError(
                "Meeting recording output_path must use the .wav extension"
            )

        max_buffered_samples = round(sample_rate * max_buffer_seconds)
        if max_buffered_samples <= 0:
            raise ValueError("max_buffer_seconds produces an empty sample queue")

        self.output_path = output_path
        self.sample_rate = sample_rate
        self.max_buffer_seconds = max_buffer_seconds
        self.max_buffered_samples = max_buffered_samples
        self._on_error = on_error
        self._writer_factory = _writer_factory or _SoundFileRF64Writer

        self._condition = threading.Condition()
        self._state = MeetingRecorderState.CREATED
        self._queue: deque[np.ndarray] = deque()
        self._buffered_sample_count = 0
        self._accepted_sample_count = 0
        self._written_sample_count = 0
        self._producer_in_flight = 0
        self._start_in_progress = False
        self._stop_requested = False
        self._cancel_requested = False
        self._abort_writer = False
        self._error: Optional[MeetingRecorderOperationalError] = None
        self._error_notified = False
        self._result: Optional[MeetingRecordingResult] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_ready = threading.Event()
        self._writer_done = threading.Event()
        self._writer_opened = False
        self._writer_start_error: Optional[MeetingRecorderOperationalError] = None

    @property
    def state(self) -> MeetingRecorderState:
        with self._condition:
            return self._state

    @property
    def error(self) -> Optional[MeetingRecorderOperationalError]:
        with self._condition:
            return self._error

    @property
    def accepted_sample_count(self) -> int:
        with self._condition:
            return self._accepted_sample_count

    @property
    def buffered_sample_count(self) -> int:
        with self._condition:
            return self._buffered_sample_count

    def start(self) -> None:
        with self._condition:
            if self._state != MeetingRecorderState.CREATED or self._start_in_progress:
                raise MeetingRecorderStateError(
                    f"Cannot start meeting recorder in state {self._state.name}"
                )
            self._start_in_progress = True
            self._writer_ready.clear()
            self._writer_done.clear()
            self._writer_opened = False
            self._writer_start_error = None
            writer_thread = threading.Thread(
                target=self._run_writer,
                name=self._WRITER_THREAD_NAME,
                daemon=False,
            )
            self._writer_thread = writer_thread

        try:
            writer_thread.start()
        except Exception as exc:
            error = self._operational_error(
                "Could not start the meeting audio writer thread", exc
            )
            with self._condition:
                self._start_in_progress = False
                self._writer_start_error = error
                should_notify = self._terminal_writer_failure_locked(error)
                self._writer_ready.set()
            if should_notify:
                self._notify_error(error)
            raise error from exc

        self._writer_ready.wait()
        with self._condition:
            error = self._writer_start_error
            opened = self._writer_opened
        if error is not None:
            writer_thread.join()
            raise error
        if not opened:
            raise MeetingRecorderOperationalError(
                "Meeting audio writer exited without reporting its startup outcome"
            )

    def enqueue(self, samples: np.ndarray) -> bool:
        validation_error = self._validate_samples(samples)
        if validation_error is not None:
            with self._condition:
                if self._state == MeetingRecorderState.FAILED:
                    return False
                if self._state != MeetingRecorderState.RUNNING:
                    raise MeetingRecorderStateError(
                        f"Cannot enqueue meeting audio in state {self._state.name}"
                    )
                should_notify = self._fail_locked(
                    validation_error,
                    abort_writer=False,
                )
            if should_notify:
                self._notify_error(validation_error)
            raise MeetingRecorderInputError(str(validation_error))

        sample_count = int(samples.size)
        with self._condition:
            if self._state == MeetingRecorderState.FAILED:
                return False
            if self._state != MeetingRecorderState.RUNNING:
                raise MeetingRecorderStateError(
                    f"Cannot enqueue meeting audio in state {self._state.name}"
                )
            if sample_count == 0:
                return True
            if sample_count > self.max_buffered_samples:
                error = MeetingRecorderOperationalError(
                    "Incoming audio block exceeds the entire meeting recorder buffer: "
                    f"samples={sample_count}, capacity={self.max_buffered_samples}"
                )
                should_notify = self._fail_locked(error, abort_writer=False)
                oversized = True
            elif self._buffered_sample_count + sample_count > self.max_buffered_samples:
                error = MeetingRecorderOperationalError(
                    "Meeting recorder buffer overflow: "
                    f"buffered={self._buffered_sample_count}, incoming={sample_count}, "
                    f"capacity={self.max_buffered_samples}"
                )
                should_notify = self._fail_locked(error, abort_writer=False)
                oversized = True
            else:
                self._buffered_sample_count += sample_count
                self._producer_in_flight += 1
                error = None
                should_notify = False
                oversized = False

        if oversized:
            if should_notify:
                self._notify_error(error)
            return False

        try:
            owned = np.array(samples, dtype=np.float32, order="C", copy=True)
        except Exception as exc:
            copy_error = self._operational_error(
                "Could not copy meeting audio into the recorder queue", exc
            )
            with self._condition:
                self._buffered_sample_count -= sample_count
                self._producer_in_flight -= 1
                should_notify = self._fail_locked(copy_error, abort_writer=False)
                self._condition.notify_all()
            if should_notify:
                self._notify_error(copy_error)
            return False

        with self._condition:
            self._producer_in_flight -= 1
            if self._state == MeetingRecorderState.FAILED:
                self._buffered_sample_count -= sample_count
                self._condition.notify_all()
                return False
            # A reservation made while RUNNING remains accepted if stop was
            # requested while the copy was in progress. The writer waits for
            # producer reservations before completing shutdown.
            self._queue.append(owned)
            self._accepted_sample_count += sample_count
            self._condition.notify_all()
        return True

    def request_stop(self) -> None:
        with self._condition:
            if self._state == MeetingRecorderState.CREATED:
                return
            if self._state == MeetingRecorderState.RUNNING:
                self._state = MeetingRecorderState.STOPPING
            if self._state in (
                MeetingRecorderState.STOPPING,
                MeetingRecorderState.FAILED,
            ):
                self._stop_requested = True
                self._condition.notify_all()

    def stop(self) -> MeetingRecordingResult:
        with self._condition:
            while self._start_in_progress:
                self._condition.wait()
            cached_result = self._result
            if cached_result is not None:
                writer_thread = self._writer_thread
            elif self._state == MeetingRecorderState.CREATED:
                return self._make_result_locked(
                    state=MeetingRecorderState.CREATED,
                    sample_count=0,
                    published=False,
                )
            else:
                if self._state == MeetingRecorderState.RUNNING:
                    self._state = MeetingRecorderState.STOPPING
                self._stop_requested = True
                self._condition.notify_all()
                writer_thread = self._writer_thread

        self._wait_for_writer_cleanup(writer_thread)

        with self._condition:
            assert self._result is not None
            return self._result

    def cancel_empty_start(self) -> MeetingRecordingResult:
        """Delete an empty partial after an underlying source start failure.

        Accepted audio is never deleted. If source startup delivered any audio,
        callers must use ``stop()`` so that prefix is finalized and published.
        """

        with self._condition:
            if self._result is not None:
                return self._result
            if self._state != MeetingRecorderState.RUNNING:
                raise MeetingRecorderStateError(
                    f"Cannot cancel meeting recorder in state {self._state.name}"
                )
            if (
                self._accepted_sample_count != 0
                or self._producer_in_flight != 0
                or self._queue
            ):
                raise MeetingRecorderStateError(
                    "Cannot cancel a meeting recorder after audio was accepted"
                )
            self._cancel_requested = True
            self._stop_requested = True
            self._state = MeetingRecorderState.STOPPING
            self._condition.notify_all()
            writer_thread = self._writer_thread

        assert writer_thread is not None
        self._wait_for_writer_cleanup(writer_thread)
        with self._condition:
            assert self._result is not None
            return self._result

    def _run_writer(self) -> None:
        writer: Optional[_ArchiveWriter] = None
        try:
            try:
                writer = self._writer_factory(self.output_path, self.sample_rate)
            except Exception as exc:
                error = self._operational_error(
                    "Could not open the RF64 meeting audio archive", exc
                )
                with self._condition:
                    self._start_in_progress = False
                    self._writer_start_error = error
                    should_notify = self._terminal_writer_failure_locked(error)
                    self._writer_ready.set()
                    self._condition.notify_all()
                if should_notify:
                    self._notify_error(error)
                return

            with self._condition:
                self._state = MeetingRecorderState.RUNNING
                self._writer_opened = True
                self._start_in_progress = False
                self._writer_ready.set()
                self._condition.notify_all()

            flush_sample_interval = max(
                1,
                round(self.sample_rate * self._PRIVATE_FLUSH_SECONDS),
            )
            samples_since_flush = 0
            writer_failed = False

            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: bool(self._queue)
                        or (self._stop_requested and self._producer_in_flight == 0)
                        or self._abort_writer
                    )
                    if self._abort_writer:
                        writer_failed = True
                        break
                    if self._queue:
                        block = self._queue.popleft()
                    elif self._stop_requested and self._producer_in_flight == 0:
                        break
                    else:
                        continue

                try:
                    pcm16 = self._float32_to_pcm16(block)
                    writer.write(pcm16)
                    samples_since_flush += block.size
                    if samples_since_flush >= flush_sample_interval:
                        writer.flush()
                        samples_since_flush = 0
                except Exception as exc:
                    error = self._operational_error(
                        "Could not write meeting audio", exc
                    )
                    with self._condition:
                        should_notify = self._terminal_writer_failure_locked(error)
                    if should_notify:
                        self._notify_error(error)
                    writer_failed = True
                    break

                with self._condition:
                    self._written_sample_count += block.size
                    self._buffered_sample_count -= block.size
                    self._condition.notify_all()

            if writer_failed:
                self._wait_for_producers_and_clear_queue()
                writer.close_after_failure()
                return

            with self._condition:
                cancel_requested = self._cancel_requested
                archive_failed = self._state == MeetingRecorderState.FAILED

            if cancel_requested:
                try:
                    writer.finalize()
                    writer.discard()
                except Exception as exc:
                    error = self._operational_error(
                        "Could not remove the empty meeting audio partial", exc
                    )
                    with self._condition:
                        should_notify = self._terminal_writer_failure_locked(error)
                    if should_notify:
                        self._notify_error(error)
                    writer.close_after_failure()
                    return
                with self._condition:
                    self._state = MeetingRecorderState.STOPPED
                    self._finish_locked(published=False)
                return

            if archive_failed:
                # Overflow/input failure already invalidated the archive. Drain
                # accepted PCM best-effort and finalize only the partial file.
                try:
                    writer.finalize()
                except Exception:
                    logging.exception("Could not finalize failed meeting audio partial")
                    writer.close_after_failure()
                with self._condition:
                    self._finish_locked(published=False)
                return

            try:
                writer.finalize()
                writer.publish()
            except Exception as exc:
                error = self._operational_error(
                    "Could not finalize and publish meeting audio", exc
                )
                with self._condition:
                    should_notify = self._terminal_writer_failure_locked(error)
                if should_notify:
                    self._notify_error(error)
                writer.close_after_failure()
                return

            with self._condition:
                self._state = MeetingRecorderState.STOPPED
                self._finish_locked(published=True)
        finally:
            self._writer_ready.set()
            with self._condition:
                self._start_in_progress = False
                self._condition.notify_all()
            self._writer_done.set()

    def _wait_for_writer_cleanup(
        self,
        writer_thread: Optional[threading.Thread],
    ) -> None:
        if writer_thread is None or threading.current_thread() is writer_thread:
            return
        # A Thread object whose start() failed cannot be joined.
        if writer_thread.ident is None:
            return
        self._writer_done.wait()
        writer_thread.join()

    def _wait_for_producers_and_clear_queue(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._producer_in_flight == 0)
            self._queue.clear()
            self._buffered_sample_count = 0
            self._condition.notify_all()

    def _fail_locked(
        self,
        error: MeetingRecorderOperationalError,
        *,
        abort_writer: bool,
    ) -> bool:
        should_notify = self._error is None and not self._error_notified
        if self._error is None:
            self._error = error
        self._state = MeetingRecorderState.FAILED
        self._stop_requested = True
        self._abort_writer = self._abort_writer or abort_writer
        if should_notify:
            self._error_notified = True
        self._condition.notify_all()
        return should_notify

    def _notify_error(self, error: MeetingRecorderOperationalError) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(error)
        except Exception:
            logging.exception("Meeting recorder error callback failed")

    def _terminal_writer_failure_locked(
        self,
        error: MeetingRecorderOperationalError,
    ) -> bool:
        """Cache a coherent terminal failure before invoking user callbacks."""

        should_notify = self._fail_locked(error, abort_writer=True)
        self._finish_locked(published=False)
        return should_notify

    def _finish_locked(self, *, published: bool) -> None:
        if self._result is not None:
            return
        sample_count = (
            self._accepted_sample_count
            if self._state == MeetingRecorderState.STOPPED and published
            else self._written_sample_count
        )
        self._result = self._make_result_locked(
            state=self._state,
            sample_count=sample_count,
            published=published,
        )
        self._condition.notify_all()

    def _make_result_locked(
        self,
        *,
        state: MeetingRecorderState,
        sample_count: int,
        published: bool,
    ) -> MeetingRecordingResult:
        return MeetingRecordingResult(
            output_path=self.output_path,
            sample_rate=self.sample_rate,
            sample_count=sample_count,
            duration_seconds=sample_count / self.sample_rate,
            state=state,
            error=self._error,
            published=published,
        )

    @staticmethod
    def _validate_samples(
        samples: np.ndarray,
    ) -> Optional[MeetingRecorderOperationalError]:
        if not isinstance(samples, np.ndarray):
            return MeetingRecorderOperationalError(
                "Meeting recorder samples must be a numpy array"
            )
        if samples.dtype != np.float32:
            return MeetingRecorderOperationalError(
                "Meeting recorder samples must have dtype numpy.float32"
            )
        if samples.ndim != 1:
            return MeetingRecorderOperationalError(
                "Meeting recorder samples must be mono with shape (frames,)"
            )
        return None

    @staticmethod
    def _float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
        finite = np.nan_to_num(
            samples,
            copy=True,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        np.clip(finite, -1.0, 1.0, out=finite)
        pcm = np.empty(finite.shape, dtype=np.int16)
        negative = finite < 0
        pcm[negative] = np.rint(finite[negative] * 32768.0).astype(np.int16)
        pcm[~negative] = np.rint(finite[~negative] * 32767.0).astype(np.int16)
        return pcm

    @staticmethod
    def _operational_error(
        message: str,
        cause: Exception,
    ) -> MeetingRecorderOperationalError:
        if isinstance(cause, MeetingRecorderOperationalError):
            return cause
        return MeetingRecorderOperationalError(f"{message}: {cause}")
