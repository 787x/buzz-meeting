import os
from pathlib import Path
import subprocess
import threading  # noqa: F401 - retained as a test patch point for PR4 compatibility
from typing import Callable, Optional

from buzz.assets import APP_BASE_DIR
from buzz.audio_capture._windows_helper_source import _WindowsHelperAudioSource


class WindowsSystemAudioSource(_WindowsHelperAudioSource):
    """Captures the Windows default system output through the native helper."""

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
        resolved_helper_path = (
            Path(helper_path)
            if helper_path is not None
            else Path(APP_BASE_DIR) / "native" / "windows" / self._HELPER_NAME
        )
        super().__init__(
            helper_path=resolved_helper_path,
            handshake_timeout=handshake_timeout,
            graceful_stop_timeout=graceful_stop_timeout,
            terminate_timeout=terminate_timeout,
            kill_timeout=kill_timeout,
            process_factory=process_factory,
            platform_name=platform_name,
        )
