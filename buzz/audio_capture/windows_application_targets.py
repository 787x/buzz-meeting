"""Windows application targets for process-loopback audio capture.

Window enumeration belongs to the targeting/UI path.  Audio capture continues to
be performed by :class:`WindowsProcessAudioSource` after a target is resolved to
a conservative process-tree root.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import ntpath
import os
import sys
from typing import Callable, Protocol


_MAXIMUM_PROCESS_ID = 0xFFFFFFFF
_MAX_PARENT_HOPS = 16
_MAX_PATH = 260
_MAX_PROCESS_IMAGE_PATH = 32_768

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TH32CS_SNAPPROCESS = 0x00000002
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_APPWINDOW = 0x00040000
_DWMWA_CLOAKED = 14

_ERROR_NO_MORE_FILES = 18
_ERROR_INSUFFICIENT_BUFFER = 122
_APPMODEL_ERROR_NO_APPLICATION = 15_703


class WindowsApplicationTargetError(RuntimeError):
    """Raised when Windows cannot perform a complete target-list refresh."""


@dataclass(frozen=True, slots=True)
class WindowsApplicationAudioTarget:
    hwnd: int
    window_title: str
    window_pid: int
    capture_pid: int
    process_name: str | None
    executable_path: str | None
    app_user_model_id: str | None

    def __post_init__(self) -> None:
        if not _is_valid_pid(self.capture_pid):
            raise ValueError(
                "capture_pid must be an integer between 1 and 4294967295"
            )

    @property
    def selection_key(self) -> tuple[int, int]:
        """Session-only identity used to preserve a UI selection after refresh."""

        return self.hwnd, self.window_pid


@dataclass(frozen=True, slots=True)
class _ProcessSnapshotEntry:
    pid: int
    parent_pid: int
    exe_name: str | None


@dataclass(frozen=True, slots=True)
class _ProcessMetadata:
    executable_path: str | None = None
    app_user_model_id: str | None = None


@dataclass(frozen=True, slots=True)
class _WindowCandidate:
    hwnd: int
    title: str
    pid: int


class _WindowsApi(Protocol):
    def enum_windows(self) -> list[int]: ...

    def is_window(self, hwnd: int) -> bool: ...

    def is_window_visible(self, hwnd: int) -> bool: ...

    def is_window_cloaked(self, hwnd: int) -> bool: ...

    def get_window_title(self, hwnd: int) -> str: ...

    def get_window_pid(self, hwnd: int) -> tuple[int, int]: ...

    def get_window_extended_style(self, hwnd: int) -> int: ...

    def snapshot_processes(self) -> dict[int, _ProcessSnapshotEntry]: ...

    def get_process_metadata(self, pid: int) -> _ProcessMetadata: ...


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * _MAX_PATH),
    ]


class _CtypesWindowsApi:
    """Narrow, lazily loaded ctypes adapter around the required Win32 APIs."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowsApplicationTargetError(
                "Application audio targets are only available on Windows."
            )

        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        self._wndenumproc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        self._user32.EnumWindows.argtypes = [
            self._wndenumproc_type,
            wintypes.LPARAM,
        ]
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD

        if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_long):
            get_window_long = self._user32.GetWindowLongW
            get_window_long.restype = wintypes.LONG
        else:
            get_window_long = self._user32.GetWindowLongPtrW
            get_window_long.restype = ctypes.c_ssize_t
        get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
        self._get_window_long = get_window_long

        self._dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._dwmapi.DwmGetWindowAttribute.restype = wintypes.LONG

        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.GetApplicationUserModelId.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.UINT),
            wintypes.LPWSTR,
        ]
        self._kernel32.GetApplicationUserModelId.restype = wintypes.LONG
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.CreateToolhelp32Snapshot.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self._kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self._kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESSENTRY32W),
        ]
        self._kernel32.Process32FirstW.restype = wintypes.BOOL
        self._kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESSENTRY32W),
        ]
        self._kernel32.Process32NextW.restype = wintypes.BOOL

    def enum_windows(self) -> list[int]:
        hwnds: list[int] = []
        callback_error: list[BaseException] = []

        def collect(hwnd: int, _lparam: int) -> bool:
            try:
                hwnds.append(int(hwnd))
                return True
            except BaseException as error:
                # Exceptions must never unwind through the native callback boundary.
                callback_error.append(error)
                return False

        callback = self._wndenumproc_type(collect)
        ctypes.set_last_error(0)
        succeeded = self._user32.EnumWindows(callback, 0)
        # Keep the callback strongly referenced through the complete synchronous call.
        _ = callback
        if callback_error:
            raise WindowsApplicationTargetError(
                "Unable to enumerate application windows."
            ) from callback_error[0]
        if not succeeded:
            error_code = ctypes.get_last_error()
            raise WindowsApplicationTargetError(
                f"Unable to enumerate application windows (Win32 error {error_code})."
            )
        return hwnds

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def is_window_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(hwnd))

    def is_window_cloaked(self, hwnd: int) -> bool:
        cloaked = wintypes.DWORD()
        result = self._dwmapi.DwmGetWindowAttribute(
            hwnd,
            _DWMWA_CLOAKED,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        if result != 0:
            raise OSError(result, "DwmGetWindowAttribute failed")
        return bool(cloaked.value)

    def get_window_title(self, hwnd: int) -> str:
        # The title can grow between the length and read calls.  Retry once with
        # a fresh length while keeping the work bounded.
        for attempt in range(2):
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buffer = ctypes.create_unicode_buffer(length + 1)
            copied = self._user32.GetWindowTextW(hwnd, buffer, len(buffer))
            if copied <= 0:
                return ""
            if copied < length or attempt == 1:
                return buffer.value
        return ""

    def get_window_pid(self, hwnd: int) -> tuple[int, int]:
        pid = wintypes.DWORD()
        thread_id = self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(thread_id), int(pid.value)

    def get_window_extended_style(self, hwnd: int) -> int:
        ctypes.set_last_error(0)
        style = self._get_window_long(hwnd, _GWL_EXSTYLE)
        error_code = ctypes.get_last_error()
        if style == 0 and error_code != 0:
            raise OSError(error_code, "GetWindowLongPtrW failed")
        return int(style)

    def snapshot_processes(self) -> dict[int, _ProcessSnapshotEntry]:
        snapshot = self._kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if snapshot == invalid_handle:
            error_code = ctypes.get_last_error()
            raise WindowsApplicationTargetError(
                f"Unable to inspect running processes (Win32 error {error_code})."
            )

        processes: dict[int, _ProcessSnapshotEntry] = {}
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            ctypes.set_last_error(0)
            has_entry = self._kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            if not has_entry:
                error_code = ctypes.get_last_error()
                if error_code == _ERROR_NO_MORE_FILES:
                    return processes
                raise WindowsApplicationTargetError(
                    f"Unable to inspect running processes (Win32 error {error_code})."
                )

            while has_entry:
                pid = int(entry.th32ProcessID)
                processes[pid] = _ProcessSnapshotEntry(
                    pid=pid,
                    parent_pid=int(entry.th32ParentProcessID),
                    exe_name=entry.szExeFile or None,
                )
                ctypes.set_last_error(0)
                has_entry = self._kernel32.Process32NextW(
                    snapshot, ctypes.byref(entry)
                )

            error_code = ctypes.get_last_error()
            if error_code not in (0, _ERROR_NO_MORE_FILES):
                raise WindowsApplicationTargetError(
                    f"Unable to inspect running processes (Win32 error {error_code})."
                )
            return processes
        finally:
            self._kernel32.CloseHandle(snapshot)

    def get_process_metadata(self, pid: int) -> _ProcessMetadata:
        process = self._kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not process:
            return _ProcessMetadata()

        try:
            return _ProcessMetadata(
                executable_path=self._query_process_image_path(process),
                app_user_model_id=self._query_application_user_model_id(process),
            )
        finally:
            self._kernel32.CloseHandle(process)

    def _query_process_image_path(self, process: int) -> str | None:
        buffer = ctypes.create_unicode_buffer(_MAX_PROCESS_IMAGE_PATH)
        size = wintypes.DWORD(len(buffer))
        if not self._kernel32.QueryFullProcessImageNameW(
            process, 0, buffer, ctypes.byref(size)
        ):
            return None
        return buffer.value or None

    def _query_application_user_model_id(self, process: int) -> str | None:
        length = wintypes.UINT()
        result = self._kernel32.GetApplicationUserModelId(
            process, ctypes.byref(length), None
        )
        if result == _APPMODEL_ERROR_NO_APPLICATION:
            return None
        if result != _ERROR_INSUFFICIENT_BUFFER or length.value == 0:
            return None

        buffer = ctypes.create_unicode_buffer(length.value)
        result = self._kernel32.GetApplicationUserModelId(
            process, ctypes.byref(length), buffer
        )
        if result != 0:
            return None
        return buffer.value or None


class _ProcessMetadataCache:
    def __init__(self, api: _WindowsApi) -> None:
        self._api = api
        self._cache: dict[int, _ProcessMetadata] = {}

    def get(self, pid: int) -> _ProcessMetadata:
        if pid not in self._cache:
            try:
                self._cache[pid] = self._api.get_process_metadata(pid)
            except OSError:
                self._cache[pid] = _ProcessMetadata()
        return self._cache[pid]


def _is_valid_pid(pid: int) -> bool:
    return type(pid) is int and 1 <= pid <= _MAXIMUM_PROCESS_ID  # noqa: E721


def _normalize_executable_path(path: str | None) -> str | None:
    if not path or not ntpath.isabs(path):
        return None
    return ntpath.normcase(ntpath.normpath(path))


def _can_climb_to_parent(
    child: _ProcessMetadata,
    parent: _ProcessMetadata,
) -> bool:
    child_aumid = child.app_user_model_id
    parent_aumid = parent.app_user_model_id
    if child_aumid and parent_aumid and child_aumid != parent_aumid:
        return False

    child_path = _normalize_executable_path(child.executable_path)
    parent_path = _normalize_executable_path(parent.executable_path)
    if child_path and parent_path and child_path == parent_path:
        return True

    return bool(child_aumid and parent_aumid and child_aumid == parent_aumid)


def _resolve_capture_pid(
    window_pid: int,
    processes: dict[int, _ProcessSnapshotEntry],
    metadata_for: Callable[[int], _ProcessMetadata],
    *,
    max_parent_hops: int = _MAX_PARENT_HOPS,
) -> int:
    current_pid = window_pid
    visited = {current_pid}

    for _ in range(max_parent_hops):
        process = processes.get(current_pid)
        if process is None:
            break
        parent_pid = process.parent_pid
        if (
            not _is_valid_pid(parent_pid)
            or parent_pid == current_pid
            or parent_pid in visited
            or parent_pid not in processes
        ):
            break
        if not _can_climb_to_parent(
            metadata_for(current_pid), metadata_for(parent_pid)
        ):
            break
        current_pid = parent_pid
        visited.add(current_pid)

    return current_pid


def _is_unsafe_shared_capture_root(
    metadata: _ProcessMetadata,
    snapshot_entry: _ProcessSnapshotEntry,
    *,
    system_root: str | None = None,
) -> bool:
    executable_path = _normalize_executable_path(metadata.executable_path)
    if executable_path is None:
        return (snapshot_entry.exe_name or "").casefold() in {
            "explorer.exe",
            "applicationframehost.exe",
        }

    root = system_root or os.environ.get("SystemRoot") or r"C:\Windows"
    unsafe_paths = {
        _normalize_executable_path(ntpath.join(root, "explorer.exe")),
        _normalize_executable_path(
            ntpath.join(root, "System32", "ApplicationFrameHost.exe")
        ),
    }
    return executable_path in unsafe_paths


def _read_window_candidate(
    api: _WindowsApi,
    hwnd: int,
    *,
    current_pid: int,
) -> _WindowCandidate | None:
    try:
        if not api.is_window(hwnd) or not api.is_window_visible(hwnd):
            return None
        if api.is_window_cloaked(hwnd):
            return None

        title = api.get_window_title(hwnd)
        if not title or not title.strip():
            return None

        thread_id, pid = api.get_window_pid(hwnd)
        if thread_id == 0 or not _is_valid_pid(pid) or pid == current_pid:
            return None

        extended_style = api.get_window_extended_style(hwnd)
        is_tool_window = bool(extended_style & _WS_EX_TOOLWINDOW)
        is_app_window = bool(extended_style & _WS_EX_APPWINDOW)
        if is_tool_window and not is_app_window:
            return None
    except (OSError, ValueError):
        return None

    return _WindowCandidate(hwnd=hwnd, title=title, pid=pid)


def _process_name(
    metadata: _ProcessMetadata,
    snapshot_entry: _ProcessSnapshotEntry,
) -> str | None:
    if metadata.executable_path:
        name = ntpath.basename(metadata.executable_path)
        if name:
            return name
    return snapshot_entry.exe_name


def _list_windows_application_audio_targets(
    api: _WindowsApi,
    *,
    current_pid: int,
) -> list[WindowsApplicationAudioTarget]:
    processes = api.snapshot_processes()
    hwnds = api.enum_windows()
    metadata = _ProcessMetadataCache(api)
    targets: list[WindowsApplicationAudioTarget] = []

    for hwnd in hwnds:
        candidate = _read_window_candidate(api, hwnd, current_pid=current_pid)
        if candidate is None:
            continue
        snapshot_entry = processes.get(candidate.pid)
        if snapshot_entry is None:
            continue

        capture_pid = _resolve_capture_pid(candidate.pid, processes, metadata.get)
        if not _is_valid_pid(capture_pid):
            continue
        if _is_unsafe_shared_capture_root(
            metadata.get(capture_pid), processes[capture_pid]
        ):
            continue

        window_metadata = metadata.get(candidate.pid)
        targets.append(
            WindowsApplicationAudioTarget(
                hwnd=candidate.hwnd,
                window_title=candidate.title,
                window_pid=candidate.pid,
                capture_pid=capture_pid,
                process_name=_process_name(window_metadata, snapshot_entry),
                executable_path=window_metadata.executable_path,
                app_user_model_id=window_metadata.app_user_model_id,
            )
        )

    return sorted(
        targets,
        key=lambda target: (
            (target.process_name or "").casefold(),
            target.window_title.casefold(),
            target.window_pid,
            target.hwnd,
        ),
    )


def list_windows_application_audio_targets() -> list[WindowsApplicationAudioTarget]:
    """Return usable visible Windows application targets in deterministic order."""

    if sys.platform != "win32":
        return []
    try:
        api = _CtypesWindowsApi()
        return _list_windows_application_audio_targets(api, current_pid=os.getpid())
    except WindowsApplicationTargetError:
        raise
    except OSError as error:
        raise WindowsApplicationTargetError(
            "Unable to refresh application audio targets."
        ) from error


def _validate_windows_application_audio_target(
    target: WindowsApplicationAudioTarget,
    api: _WindowsApi,
    *,
    current_pid: int,
) -> bool:
    if not _is_valid_pid(target.window_pid) or not _is_valid_pid(target.capture_pid):
        return False

    candidate = _read_window_candidate(api, target.hwnd, current_pid=current_pid)
    if candidate is None or candidate.pid != target.window_pid:
        return False

    try:
        processes = api.snapshot_processes()
    except (OSError, WindowsApplicationTargetError):
        return False
    if target.window_pid not in processes:
        return False

    metadata = _ProcessMetadataCache(api)
    capture_pid = _resolve_capture_pid(target.window_pid, processes, metadata.get)
    if capture_pid != target.capture_pid:
        return False
    return not _is_unsafe_shared_capture_root(
        metadata.get(capture_pid), processes[capture_pid]
    )


def validate_windows_application_audio_target(
    target: WindowsApplicationAudioTarget,
) -> bool:
    """Revalidate a target before capture, including HWND reuse and root changes."""

    if sys.platform != "win32":
        return False
    try:
        return _validate_windows_application_audio_target(
            target,
            _CtypesWindowsApi(),
            current_pid=os.getpid(),
        )
    except (OSError, WindowsApplicationTargetError):
        return False
