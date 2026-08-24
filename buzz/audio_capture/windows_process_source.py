import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional

from buzz.assets import APP_BASE_DIR
from buzz.audio_capture._windows_helper_source import _WindowsHelperAudioSource
from buzz.audio_capture.source import AudioSourceError


class WindowsProcessAudioSource(_WindowsHelperAudioSource):
    """Captures audio rendered by a Windows process and its child process tree."""

    MIN_WINDOWS_BUILD = 20_348
    _SOURCE_LABEL = "Windows process audio"
    _SOURCE_SHORT_LABEL = "Process audio"
    _THREAD_PREFIX = "windows-process-audio"

    def __init__(
        self,
        process_id: int,
        helper_path: Optional[os.PathLike[str] | str] = None,
        *,
        handshake_timeout: float = 10.0,
        graceful_stop_timeout: float = 5.0,
        terminate_timeout: float = 2.0,
        kill_timeout: float = 2.0,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        platform_name: Optional[str] = None,
        windows_build: Optional[int] = None,
    ) -> None:
        if type(process_id) is not int or not 1 <= process_id <= 0xFFFFFFFF:  # noqa: E721
            raise ValueError("process_id must be an integer between 1 and 4294967295")

        resolved_helper_path = (
            Path(helper_path)
            if helper_path is not None
            else Path(APP_BASE_DIR) / "native" / "windows" / self._HELPER_NAME
        )
        self._process_id = process_id
        self._windows_build = windows_build
        super().__init__(
            helper_path=resolved_helper_path,
            handshake_timeout=handshake_timeout,
            graceful_stop_timeout=graceful_stop_timeout,
            terminate_timeout=terminate_timeout,
            kill_timeout=kill_timeout,
            process_factory=process_factory,
            platform_name=platform_name,
        )

    @property
    def process_id(self) -> int:
        return self._process_id

    def _preflight(self) -> None:
        if self._platform_name != "win32":
            raise AudioSourceError("Process audio capture is only available on Windows.")

        windows_build = self._windows_build
        if windows_build is None:
            windows_build = sys.getwindowsversion().build
        if windows_build < self.MIN_WINDOWS_BUILD:
            raise AudioSourceError(
                "Process audio capture requires Windows 10 build 20348 or later."
            )

        if not self._helper_path.is_file():
            raise AudioSourceError(
                "The Windows process audio helper is missing. Reinstall Buzz or rebuild "
                "the native helper."
            )

    def _command(self) -> list[str]:
        return [
            str(self._helper_path),
            "--mode",
            "process",
            "--pid",
            str(self._process_id),
        ]

    def _helper_error(
        self,
        process: subprocess.Popen,
        *,
        startup: bool,
    ) -> AudioSourceError:
        return_code = process.poll()
        with self._condition:
            native_detail = (
                bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
            )

        friendly_messages = {
            10: "Windows could not initialize process audio capture.",
            11: "The selected process is unavailable for audio capture.",
            12: "Buzz could not open process audio as 16 kHz mono audio.",
            13: "Windows could not start process audio capture.",
            14: "Windows process audio capture failed.",
            15: "The selected process audio stream became unavailable.",
            16: "The Windows Audio service stopped or is unavailable.",
            17: "The Windows process audio transport was interrupted.",
            18: "The Windows process audio helper failed unexpectedly.",
            19: "Process audio capture requires Windows 10 build 20348 or later.",
            20: "Unable to capture audio from the selected process.",
            21: "Timed out while starting process audio capture.",
        }
        if return_code in friendly_messages:
            message = friendly_messages[return_code]
        elif startup:
            message = "Windows process audio capture ended before it was ready."
        else:
            message = "Windows process audio capture ended unexpectedly."

        if native_detail:
            logging.error(
                "Windows process audio helper diagnostics: %s", native_detail
            )
        return AudioSourceError(message)
