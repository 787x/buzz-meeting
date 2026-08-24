import logging
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
from typing import BinaryIO, Callable, Optional

import numpy as np

from buzz.assets import APP_BASE_DIR
from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)


class _WindowsHelperAudioSource(AudioSource):
    """Shared lifecycle and PCM transport for Windows native helper sources."""

    SAMPLE_RATE = 16_000
    _HELPER_NAME = "buzz-windows-audio-capture.exe"
    _HEADER = struct.Struct("<4sHHIHH")
    _MAGIC = b"BZWA"
    _PROTOCOL_VERSION = 1
    _SAMPLE_FORMAT_FLOAT32_LE = 1
    _READ_SIZE = 4_096
    _STDERR_LIMIT = 64 * 1_024
    _THREAD_JOIN_TIMEOUT = 2.0
    _SOURCE_LABEL = "Windows system audio"
    _SOURCE_SHORT_LABEL = "System audio"
    _THREAD_PREFIX = "windows-system-audio"

    def __init__(
        self,
        helper_path: Optional[os.PathLike[str] | str] = None,
        *,
        handshake_timeout: float = 5.0,
        graceful_stop_timeout: float = 5.0,
        terminate_timeout: float = 2.0,
        kill_timeout: float = 2.0,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        platform_name: Optional[str] = None,
    ) -> None:
        self._helper_path = (
            Path(helper_path)
            if helper_path is not None
            else Path(APP_BASE_DIR) / "native" / "windows" / self._HELPER_NAME
        )
        self._handshake_timeout = handshake_timeout
        self._graceful_stop_timeout = graceful_stop_timeout
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._process_factory = process_factory
        self._platform_name = platform_name if platform_name is not None else sys.platform

        self._condition = threading.Condition()
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._on_audio: Optional[AudioFrameCallback] = None
        self._on_error: Optional[AudioErrorCallback] = None
        self._starting = False
        self._active = False
        self._stopping = False
        self._stop_signaled = False
        self._callback_in_flight = False
        self._error_reported = False
        self._startup_error: Optional[AudioSourceError] = None
        self._startup_event = threading.Event()
        self._delivery_enabled = threading.Event()
        self._stderr_tail = bytearray()
        self._generation = 0

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def helper_path(self) -> Path:
        return self._helper_path

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        self._preflight()

        with self._condition:
            if self._starting or self._active or self._process is not None:
                raise AudioSourceError(f"{self._SOURCE_SHORT_LABEL} source is already active")
            self._generation += 1
            generation = self._generation
            startup_event = threading.Event()
            delivery_enabled = threading.Event()
            self._starting = True
            self._stopping = False
            self._stop_signaled = False
            self._callback_in_flight = False
            self._error_reported = False
            self._startup_error = None
            self._stderr_tail.clear()
            self._startup_event = startup_event
            self._delivery_enabled = delivery_enabled
            self._on_audio = on_audio
            self._on_error = on_error

        command = self._command()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = self._process_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=creation_flags,
            )
        except Exception as exc:
            self._cancel_unlaunched_start(generation)
            raise AudioSourceError(
                f"Could not start {self._SOURCE_LABEL} capture: {exc}"
            ) from exc

        if process.stdin is None or process.stdout is None or process.stderr is None:
            try:
                self._cleanup_unowned_process(process)
            finally:
                self._cancel_unlaunched_start(generation)
            raise AudioSourceError(f"{self._SOURCE_LABEL.capitalize()} helper pipes are unavailable.")

        try:
            reader_thread = threading.Thread(
                target=self._read_stdout,
                args=(process, generation, startup_event, delivery_enabled),
                name=f"{self._THREAD_PREFIX}-reader",
            )
            stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(process, process.stderr, generation),
                name=f"{self._THREAD_PREFIX}-stderr",
            )
        except Exception as exc:
            try:
                self._cleanup_unowned_process(process)
            except AudioSourceError:
                logging.exception(
                    "%s cleanup failed after thread creation failure",
                    self._SOURCE_LABEL.capitalize(),
                )
            finally:
                self._cancel_unlaunched_start(generation)
            raise AudioSourceError(
                f"Could not create {self._SOURCE_LABEL} worker threads: {exc}"
            ) from exc
        accepted = False
        thread_start_error: Optional[Exception] = None
        with self._condition:
            if (
                self._generation == generation
                and self._starting
                and not self._stopping
                and self._process is None
            ):
                accepted = True
                self._process = process
                self._reader_thread = reader_thread
                self._stderr_thread = stderr_thread
                try:
                    stderr_thread.start()
                    reader_thread.start()
                except Exception as exc:
                    thread_start_error = exc

        if not accepted:
            self._cleanup_unowned_process(process)
            raise AudioSourceError(f"{self._SOURCE_LABEL.capitalize()} capture startup was canceled.")

        if thread_start_error is not None:
            try:
                self._shutdown()
            except AudioSourceError:
                logging.exception(
                    "%s cleanup failed after thread startup failure",
                    self._SOURCE_LABEL.capitalize(),
                )
            raise AudioSourceError(
                f"Could not start {self._SOURCE_LABEL} worker threads: {thread_start_error}"
            ) from thread_start_error

        if not startup_event.wait(self._handshake_timeout):
            startup_error = AudioSourceError(
                f"Timed out while starting {self._SOURCE_LABEL} capture."
            )
            self._complete_failed_start(process)
            raise startup_error

        with self._condition:
            startup_error = (
                self._startup_error
                if self._generation == generation and self._process is process
                else None
            )

        if startup_error is not None:
            self._complete_failed_start(process)
            raise startup_error

        if not self._activate_started_session(
            process,
            generation,
            delivery_enabled,
        ):
            self._complete_failed_start(process)
            raise AudioSourceError(f"{self._SOURCE_LABEL.capitalize()} capture startup was canceled.")

    def _preflight(self) -> None:
        if self._platform_name != "win32":
            raise AudioSourceError("System audio capture is only available on Windows.")
        if not self._helper_path.is_file():
            raise AudioSourceError(
                "The Windows system audio helper is missing. Reinstall Buzz or rebuild "
                "the native helper."
            )

    def _command(self) -> list[str]:
        return [str(self._helper_path), "--mode", "system"]

    def stop(self) -> None:
        self._shutdown()

    def _activate_started_session(
        self,
        process: subprocess.Popen,
        generation: int,
        delivery_enabled: threading.Event,
    ) -> bool:
        with self._condition:
            if (
                self._generation != generation
                or self._process is not process
                or self._delivery_enabled is not delivery_enabled
                or self._stopping
                or self._startup_error is not None
                or self._error_reported
                or process.poll() is not None
            ):
                return False
            self._starting = False
            self._active = True
            delivery_enabled.set()
            return True

    def _read_stdout(
        self,
        process: subprocess.Popen,
        generation: int,
        startup_event: threading.Event,
        delivery_enabled: threading.Event,
    ) -> None:
        assert process.stdout is not None
        try:
            header = self._read_exact(process.stdout, self._HEADER.size)
            if len(header) != self._HEADER.size:
                self._set_startup_error(
                    self._helper_error(process, startup=True),
                    process,
                    generation,
                    startup_event,
                )
                return
            self._validate_header(header)
            startup_event.set()
            delivery_enabled.wait()

            carry = b""
            while True:
                chunk = process.stdout.read(self._READ_SIZE)
                if not chunk:
                    break
                with self._condition:
                    if (
                        self._generation != generation
                        or self._process is not process
                        or self._stopping
                    ):
                        continue

                data = carry + chunk
                complete_size = len(data) - (len(data) % np.dtype("<f4").itemsize)
                if complete_size == 0:
                    carry = data
                    continue

                sample_bytes = data[:complete_size]
                carry = data[complete_size:]
                samples = np.frombuffer(sample_bytes, dtype="<f4")
                self._deliver(samples, process, generation)

            with self._condition:
                stopping = (
                    self._generation != generation
                    or self._process is not process
                    or self._stopping
                )
            if not stopping:
                if carry:
                    error = AudioSourceError(
                        f"{self._SOURCE_LABEL.capitalize()} transport ended with an "
                        "incomplete sample."
                    )
                else:
                    error = self._helper_error(process, startup=False)
                self._report_runtime_failure(error, process, generation)
        except AudioSourceError as exc:
            if not startup_event.is_set():
                self._set_startup_error(
                    exc,
                    process,
                    generation,
                    startup_event,
                )
            else:
                self._report_runtime_failure(exc, process, generation)
        except Exception as exc:
            error = AudioSourceError(
                f"{self._SOURCE_LABEL.capitalize()} transport failed: {exc}"
            )
            if not startup_event.is_set():
                self._set_startup_error(
                    error,
                    process,
                    generation,
                    startup_event,
                )
            else:
                self._report_runtime_failure(error, process, generation)
        finally:
            startup_event.set()

    def _deliver(
        self,
        samples: np.ndarray,
        process: subprocess.Popen,
        generation: int,
    ) -> None:
        with self._condition:
            if (
                self._generation != generation
                or self._process is not process
                or not self._active
                or self._stopping
                or self._on_audio is None
            ):
                return
            callback = self._on_audio
            self._callback_in_flight = True

        callback_error: Optional[Exception] = None
        try:
            callback(samples)
        except Exception as exc:
            callback_error = exc
        finally:
            with self._condition:
                self._callback_in_flight = False
                self._condition.notify_all()

        if callback_error is not None:
            self._report_runtime_failure(
                AudioSourceError(
                    f"{self._SOURCE_SHORT_LABEL} callback failed: {callback_error}"
                ),
                process,
                generation,
            )

    def _drain_stderr(
        self,
        process: subprocess.Popen,
        stderr: BinaryIO,
        generation: int,
    ) -> None:
        try:
            while True:
                chunk = stderr.read(4_096)
                if not chunk:
                    return
                with self._condition:
                    if (
                        self._generation != generation
                        or self._process is not process
                    ):
                        continue
                    self._stderr_tail.extend(chunk)
                    excess = len(self._stderr_tail) - self._STDERR_LIMIT
                    if excess > 0:
                        del self._stderr_tail[:excess]
        except Exception as exc:
            logging.debug("%s stderr drain ended: %s", self._SOURCE_LABEL, exc)

    def _set_startup_error(
        self,
        error: AudioSourceError,
        process: subprocess.Popen,
        generation: int,
        startup_event: threading.Event,
    ) -> None:
        with self._condition:
            if (
                self._generation == generation
                and self._process is process
                and self._startup_error is None
            ):
                self._startup_error = error
        startup_event.set()

    def _report_runtime_failure(
        self,
        error: AudioSourceError,
        process: subprocess.Popen,
        generation: int,
    ) -> None:
        with self._condition:
            if (
                self._generation != generation
                or self._process is not process
                or self._stopping
                or self._error_reported
            ):
                return
            self._error_reported = True
            self._active = False
            callback = self._on_error
            process = self._process

        try:
            self._signal_helper_stop(process)
        except Exception:
            logging.exception(
                "Could not signal %s helper after runtime failure", self._SOURCE_LABEL
            )
        logging.error("%s capture failed: %s", self._SOURCE_LABEL.capitalize(), error)
        if callback is not None:
            try:
                callback(error)
            except Exception:
                logging.exception("%s error callback failed", self._SOURCE_LABEL)

    def _shutdown(self) -> None:
        with self._condition:
            process = self._process
            if process is None:
                self._generation += 1
                self._reset_after_stop_locked()
                self._condition.notify_all()
                return
            if self._stopping:
                self._condition.wait_for(lambda: self._process is not process)
                return

            reader_thread = self._reader_thread
            stderr_thread = self._stderr_thread
            self._stopping = True
            self._starting = False
            self._active = False
            self._generation += 1
            self._startup_event.set()
            self._delivery_enabled.set()

        shutdown_error: Optional[Exception] = None
        try:
            try:
                self._signal_helper_stop(process)
            except Exception as exc:
                shutdown_error = exc

            try:
                self._terminate_process(process)
            except Exception as exc:
                if shutdown_error is None:
                    shutdown_error = exc
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:
                        if shutdown_error is None:
                            shutdown_error = exc

            current_thread = threading.current_thread()
            if reader_thread is not current_thread:
                with self._condition:
                    while self._callback_in_flight:
                        self._condition.wait()

            for name, thread in (
                ("reader", reader_thread),
                ("stderr", stderr_thread),
            ):
                if (
                    thread is None
                    or thread is current_thread
                    or thread.ident is None
                ):
                    continue
                try:
                    thread.join(timeout=self._THREAD_JOIN_TIMEOUT)
                    if thread.is_alive() and shutdown_error is None:
                        shutdown_error = AudioSourceError(
                            f"{self._SOURCE_LABEL.capitalize()} {name} thread did not stop."
                        )
                except Exception as exc:
                    if shutdown_error is None:
                        shutdown_error = exc

            with self._condition:
                self._reset_after_stop_locked()
                self._condition.notify_all()

        if shutdown_error is not None:
            if isinstance(shutdown_error, AudioSourceError):
                raise shutdown_error
            raise AudioSourceError(
                f"Could not stop {self._SOURCE_LABEL} capture: {shutdown_error}"
            ) from shutdown_error

    def _signal_helper_stop(self, process: Optional[subprocess.Popen]) -> None:
        if process is None:
            return
        with self._condition:
            if self._stop_signaled:
                return
            self._stop_signaled = True
            stdin = process.stdin
        if stdin is None:
            return
        try:
            stdin.write(b"\x00")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                stdin.close()
            except (OSError, ValueError):
                pass

    def _terminate_process(self, process: subprocess.Popen) -> None:
        primary_error: Optional[Exception] = None
        exited = False
        try:
            process.wait(timeout=self._graceful_stop_timeout)
            exited = True
        except subprocess.TimeoutExpired:
            pass
        except Exception as exc:
            primary_error = exc

        if not exited:
            try:
                process.terminate()
            except Exception as exc:
                if primary_error is None:
                    primary_error = exc

            try:
                process.wait(timeout=self._terminate_timeout)
                exited = True
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:
                if primary_error is None:
                    primary_error = exc

        if not exited:
            try:
                process.kill()
            except Exception as exc:
                if primary_error is None:
                    primary_error = exc

            try:
                process.wait(timeout=self._kill_timeout)
                exited = True
            except subprocess.TimeoutExpired:
                if primary_error is None:
                    primary_error = AudioSourceError(
                        f"{self._SOURCE_LABEL.capitalize()} helper did not exit after "
                        "it was killed."
                    )
            except Exception as exc:
                if primary_error is None:
                    primary_error = exc

        if primary_error is not None:
            if isinstance(primary_error, AudioSourceError):
                raise primary_error
            raise AudioSourceError(
                f"{self._SOURCE_LABEL.capitalize()} helper shutdown failed: "
                f"{primary_error}"
            ) from primary_error

    def _cancel_unlaunched_start(self, generation: int) -> None:
        with self._condition:
            if self._generation != generation:
                return
            self._generation += 1
            self._reset_after_stop_locked()
            self._condition.notify_all()

    def _complete_failed_start(self, process: subprocess.Popen) -> None:
        with self._condition:
            if self._process is not process:
                return
            if self._stopping:
                self._condition.wait_for(lambda: self._process is not process)
                return

        try:
            self._shutdown()
        except AudioSourceError:
            logging.exception("%s startup cleanup failed", self._SOURCE_LABEL)

    def _cleanup_unowned_process(self, process: subprocess.Popen) -> None:
        cleanup_error: Optional[Exception] = None
        try:
            self._signal_process_stop(process)
        except Exception as exc:
            cleanup_error = exc

        try:
            self._terminate_process(process)
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc

        if cleanup_error is not None:
            if isinstance(cleanup_error, AudioSourceError):
                raise cleanup_error
            raise AudioSourceError(
                f"Could not clean up {self._SOURCE_LABEL} helper: {cleanup_error}"
            ) from cleanup_error

    @staticmethod
    def _signal_process_stop(process: subprocess.Popen) -> None:
        stdin = process.stdin
        if stdin is None:
            return
        try:
            stdin.write(b"\x00")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                stdin.close()
            except (OSError, ValueError):
                pass

    def _helper_error(
        self,
        process: subprocess.Popen,
        *,
        startup: bool,
    ) -> AudioSourceError:
        return_code = process.poll()
        with self._condition:
            native_detail = bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()

        friendly_messages = {
            10: "Windows could not initialize system audio capture.",
            11: "No Windows default system output is available.",
            12: "Buzz could not open the default system output as 16 kHz mono audio.",
            13: "Windows could not start system audio capture.",
            14: "Windows system audio capture failed.",
            15: "The Windows system output device became unavailable.",
            16: "The Windows Audio service stopped or is unavailable.",
            17: "The Windows system audio transport was interrupted.",
            18: "The Windows system audio helper failed unexpectedly.",
        }
        if return_code in friendly_messages:
            message = friendly_messages[return_code]
        elif startup:
            message = "Windows system audio capture ended before it was ready."
        else:
            message = "Windows system audio capture ended unexpectedly."

        if native_detail:
            logging.error(
                "%s helper diagnostics: %s", self._SOURCE_LABEL.capitalize(), native_detail
            )
        return AudioSourceError(message)

    @classmethod
    def _validate_header(cls, header: bytes) -> None:
        magic, version, header_size, sample_rate, channels, sample_format = (
            cls._HEADER.unpack(header)
        )
        if magic != cls._MAGIC:
            raise AudioSourceError(
                f"{cls._SOURCE_LABEL.capitalize()} helper sent an invalid handshake."
            )
        if version != cls._PROTOCOL_VERSION or header_size != cls._HEADER.size:
            raise AudioSourceError(
                f"{cls._SOURCE_LABEL.capitalize()} helper uses an unsupported "
                "protocol version."
            )
        if sample_rate != cls.SAMPLE_RATE:
            raise AudioSourceError(
                f"{cls._SOURCE_LABEL.capitalize()} helper did not provide 16 kHz audio."
            )
        if channels != 1:
            raise AudioSourceError(
                f"{cls._SOURCE_LABEL.capitalize()} helper did not provide mono audio."
            )
        if sample_format != cls._SAMPLE_FORMAT_FLOAT32_LE:
            raise AudioSourceError(
                f"{cls._SOURCE_LABEL.capitalize()} helper did not provide float32 audio."
            )

    @staticmethod
    def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < byte_count:
            chunk = stream.read(byte_count - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def _reset_after_stop_locked(self) -> None:
        self._startup_event.set()
        self._delivery_enabled.set()
        self._process = None
        self._reader_thread = None
        self._stderr_thread = None
        self._on_audio = None
        self._on_error = None
        self._starting = False
        self._active = False
        self._stopping = False
        self._stop_signaled = False
        self._callback_in_flight = False
        self._error_reported = False
        self._startup_error = None
        self._startup_event = threading.Event()
        self._delivery_enabled = threading.Event()
