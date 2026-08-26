from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import sys

import pytest

import buzz.audio_capture.windows_application_targets as targets_module
from buzz.audio_capture.windows_application_targets import (
    WindowsApplicationAudioTarget,
    WindowsApplicationTargetError,
    _CtypesWindowsApi,
    _ProcessMetadata,
    _ProcessSnapshotEntry,
    _can_climb_to_parent,
    _is_unsafe_shared_capture_root,
    _list_windows_application_audio_targets,
    _resolve_capture_pid,
    _validate_windows_application_audio_target,
)


@dataclass
class _Window:
    pid: int
    title: str = "Document"
    thread_id: int = 10
    exists: bool = True
    visible: bool = True
    cloaked: bool = False
    style: int = 0


class _FakeWindowsApi:
    def __init__(self) -> None:
        self.windows: dict[int, _Window] = {100: _Window(pid=20)}
        self.processes: dict[int, _ProcessSnapshotEntry] = {
            20: _ProcessSnapshotEntry(20, 10, "app.exe"),
            10: _ProcessSnapshotEntry(10, 4, "launcher.exe"),
            4: _ProcessSnapshotEntry(4, 0, "system.exe"),
        }
        self.metadata: dict[int, _ProcessMetadata] = {
            20: _ProcessMetadata(r"C:\Apps\app.exe", None),
            10: _ProcessMetadata(r"C:\Launcher\launcher.exe", None),
            4: _ProcessMetadata(r"C:\Windows\System32\system.exe", None),
        }
        self.metadata_errors: set[int] = set()
        self.metadata_calls: dict[int, int] = {}
        self.snapshot_calls = 0
        self.enum_error: Exception | None = None
        self.snapshot_error: Exception | None = None

    def enum_windows(self) -> list[int]:
        if self.enum_error is not None:
            raise self.enum_error
        return list(self.windows)

    def is_window(self, hwnd: int) -> bool:
        return self.windows[hwnd].exists

    def is_window_visible(self, hwnd: int) -> bool:
        return self.windows[hwnd].visible

    def is_window_cloaked(self, hwnd: int) -> bool:
        return self.windows[hwnd].cloaked

    def get_window_title(self, hwnd: int) -> str:
        return self.windows[hwnd].title

    def get_window_pid(self, hwnd: int) -> tuple[int, int]:
        window = self.windows[hwnd]
        return window.thread_id, window.pid

    def get_window_extended_style(self, hwnd: int) -> int:
        return self.windows[hwnd].style

    def snapshot_processes(self) -> dict[int, _ProcessSnapshotEntry]:
        self.snapshot_calls += 1
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return dict(self.processes)

    def get_process_metadata(self, pid: int) -> _ProcessMetadata:
        self.metadata_calls[pid] = self.metadata_calls.get(pid, 0) + 1
        if pid in self.metadata_errors:
            raise PermissionError("access denied")
        return self.metadata.get(pid, _ProcessMetadata())


def _list(api: _FakeWindowsApi) -> list[WindowsApplicationAudioTarget]:
    return _list_windows_application_audio_targets(api, current_pid=999)


def _target(**overrides: object) -> WindowsApplicationAudioTarget:
    values = {
        "hwnd": 100,
        "window_title": "Document",
        "window_pid": 20,
        "capture_pid": 20,
        "process_name": "app.exe",
        "executable_path": r"C:\Apps\app.exe",
        "app_user_model_id": None,
    }
    values.update(overrides)
    return WindowsApplicationAudioTarget(**values)  # type: ignore[arg-type]


def test_target_has_session_selection_key() -> None:
    assert _target().selection_key == (100, 20)


@pytest.mark.parametrize("capture_pid", [True, 0, -1, 0x1_0000_0000])
def test_target_rejects_invalid_capture_pid(capture_pid: object) -> None:
    with pytest.raises(ValueError, match="capture_pid"):
        _target(capture_pid=capture_pid)


def test_lists_a_visible_valid_window() -> None:
    api = _FakeWindowsApi()

    assert _list(api) == [_target()]
    assert api.snapshot_calls == 1


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("exists", False),
        ("visible", False),
        ("cloaked", True),
        ("title", ""),
        ("title", " \t "),
        ("thread_id", 0),
        ("pid", 0),
    ],
)
def test_filters_invalid_window_candidates(change: str, value: object) -> None:
    api = _FakeWindowsApi()
    setattr(api.windows[100], change, value)

    assert _list(api) == []


def test_filters_buzz_own_pid() -> None:
    api = _FakeWindowsApi()

    assert _list_windows_application_audio_targets(api, current_pid=20) == []


def test_filters_tool_window_unless_it_is_also_an_app_window() -> None:
    api = _FakeWindowsApi()
    api.windows[100].style = 0x00000080
    assert _list(api) == []

    api.windows[100].style |= 0x00040000
    assert _list(api) == [_target()]


def test_process_metadata_failure_keeps_window_with_snapshot_name() -> None:
    api = _FakeWindowsApi()
    api.metadata_errors.add(20)

    assert _list(api) == [
        _target(executable_path=None, app_user_model_id=None)
    ]


def test_missing_process_path_keeps_window() -> None:
    api = _FakeWindowsApi()
    api.metadata[20] = _ProcessMetadata(None, None)

    assert _list(api) == [_target(executable_path=None)]


def test_process_disappearing_from_snapshot_drops_only_that_window() -> None:
    api = _FakeWindowsApi()
    api.windows[101] = _Window(pid=21, title="Other")
    api.processes[21] = _ProcessSnapshotEntry(21, 0, "other.exe")
    api.metadata[21] = _ProcessMetadata(r"C:\Apps\other.exe", None)
    del api.processes[20]

    assert _list(api) == [
        _target(
            hwnd=101,
            window_title="Other",
            window_pid=21,
            capture_pid=21,
            process_name="other.exe",
            executable_path=r"C:\Apps\other.exe",
        )
    ]


@pytest.mark.parametrize(
    "title",
    ["会议记录", "Call 🎙️", "Player \U0001f3b5", "Cafe\u0301"],
)
def test_preserves_unicode_window_titles(title: str) -> None:
    api = _FakeWindowsApi()
    api.windows[100].title = title

    assert _list(api)[0].window_title == title


def test_preserves_multiple_windows_for_the_same_process() -> None:
    api = _FakeWindowsApi()
    api.windows = {
        100: _Window(pid=20, title="Meet"),
        101: _Window(pid=20, title="YouTube"),
    }

    listed = _list(api)

    assert [(target.hwnd, target.window_title) for target in listed] == [
        (100, "Meet"),
        (101, "YouTube"),
    ]
    assert api.metadata_calls == {20: 1, 10: 1}


def test_sorts_deterministically_by_process_title_pid_and_hwnd() -> None:
    api = _FakeWindowsApi()
    api.windows = {
        104: _Window(pid=24, title="Zulu"),
        103: _Window(pid=23, title="alpha"),
        102: _Window(pid=22, title="Beta"),
        101: _Window(pid=21, title="beta"),
    }
    api.processes = {
        pid: _ProcessSnapshotEntry(pid, 0, exe)
        for pid, exe in [(21, "z.exe"), (22, "z.exe"), (23, "a.exe"), (24, "a.exe")]
    }
    api.metadata = {
        pid: _ProcessMetadata(fr"C:\Apps\{entry.exe_name}", None)
        for pid, entry in api.processes.items()
    }

    assert [target.hwnd for target in _list(api)] == [103, 104, 101, 102]


def test_core_enumeration_and_snapshot_errors_remain_global() -> None:
    api = _FakeWindowsApi()
    api.snapshot_error = WindowsApplicationTargetError("snapshot")
    with pytest.raises(WindowsApplicationTargetError, match="snapshot"):
        _list(api)

    api.snapshot_error = None
    api.enum_error = WindowsApplicationTargetError("windows")
    with pytest.raises(WindowsApplicationTargetError, match="windows"):
        _list(api)


def test_non_windows_public_list_is_safe_and_does_not_load_dlls(monkeypatch) -> None:
    monkeypatch.setattr(targets_module.sys, "platform", "linux")
    monkeypatch.setattr(
        targets_module,
        "_CtypesWindowsApi",
        lambda: pytest.fail("Win32 DLL adapter must not be constructed"),
    )

    assert targets_module.list_windows_application_audio_targets() == []
    assert not targets_module.validate_windows_application_audio_target(_target())


def test_same_exact_full_path_parent_climbs() -> None:
    processes = {
        20: _ProcessSnapshotEntry(20, 10, "app.exe"),
        10: _ProcessSnapshotEntry(10, 0, "app.exe"),
    }
    metadata = {
        20: _ProcessMetadata(r"C:\Apps\APP.exe", None),
        10: _ProcessMetadata(r"c:\apps\.\app.exe", None),
    }

    assert _resolve_capture_pid(20, processes, metadata.__getitem__) == 10


def test_same_basename_in_different_paths_does_not_climb() -> None:
    assert not _can_climb_to_parent(
        _ProcessMetadata(r"C:\Apps\app.exe", None),
        _ProcessMetadata(r"D:\Other\app.exe", None),
    )


def test_same_aumid_direct_parent_climbs() -> None:
    processes = {
        20: _ProcessSnapshotEntry(20, 10, "renderer.exe"),
        10: _ProcessSnapshotEntry(10, 0, "host.exe"),
    }
    metadata = {
        20: _ProcessMetadata(r"C:\Apps\renderer.exe", "Vendor.App"),
        10: _ProcessMetadata(r"C:\Apps\host.exe", "Vendor.App"),
    }

    assert _resolve_capture_pid(20, processes, metadata.__getitem__) == 10


def test_different_nonempty_aumid_blocks_even_same_path() -> None:
    assert not _can_climb_to_parent(
        _ProcessMetadata(r"C:\Apps\app.exe", "Vendor.One"),
        _ProcessMetadata(r"C:\Apps\app.exe", "Vendor.Two"),
    )


def test_one_missing_aumid_and_different_path_does_not_climb() -> None:
    assert not _can_climb_to_parent(
        _ProcessMetadata(r"C:\Apps\app.exe", "Vendor.App"),
        _ProcessMetadata(r"C:\Launcher\launcher.exe", None),
    )


@pytest.mark.parametrize(
    ("parent_path", "child_path"),
    [
        (r"C:\Windows\explorer.exe", r"C:\Apps\app.exe"),
        (r"C:\Python\python.exe", r"C:\Apps\app.exe"),
    ],
)
def test_different_executable_parent_does_not_expand_capture(
    parent_path: str, child_path: str
) -> None:
    processes = {
        20: _ProcessSnapshotEntry(20, 10, "app.exe"),
        10: _ProcessSnapshotEntry(10, 0, "parent.exe"),
    }
    metadata = {
        20: _ProcessMetadata(child_path, None),
        10: _ProcessMetadata(parent_path, None),
    }

    assert _resolve_capture_pid(20, processes, metadata.__getitem__) == 20


def test_electron_like_same_path_chain_resolves_to_root_without_sibling_search() -> None:
    processes = {
        30: _ProcessSnapshotEntry(30, 20, "app.exe"),
        20: _ProcessSnapshotEntry(20, 10, "app.exe"),
        10: _ProcessSnapshotEntry(10, 0, "app.exe"),
        40: _ProcessSnapshotEntry(40, 10, "helper.exe"),
    }
    metadata = {
        10: _ProcessMetadata(r"C:\Apps\app.exe", None),
        20: _ProcessMetadata(r"C:\Apps\app.exe", None),
        30: _ProcessMetadata(r"C:\Apps\app.exe", None),
        40: _ProcessMetadata(r"C:\Apps\helper.exe", None),
    }

    assert _resolve_capture_pid(30, processes, metadata.__getitem__) == 10


@pytest.mark.parametrize(
    "processes",
    [
        {20: _ProcessSnapshotEntry(20, 99, "app.exe")},
        {20: _ProcessSnapshotEntry(20, 20, "app.exe")},
        {
            20: _ProcessSnapshotEntry(20, 10, "app.exe"),
            10: _ProcessSnapshotEntry(10, 20, "app.exe"),
        },
    ],
)
def test_resolver_stops_for_missing_self_or_cyclic_parent(
    processes: dict[int, _ProcessSnapshotEntry],
) -> None:
    def metadata(_pid: int) -> _ProcessMetadata:
        return _ProcessMetadata(r"C:\Apps\app.exe", None)

    assert _resolve_capture_pid(20, processes, metadata) in processes


def test_resolver_hop_limit_is_bounded() -> None:
    processes = {
        pid: _ProcessSnapshotEntry(pid, pid - 1, "app.exe")
        for pid in range(1, 22)
    }
    processes[1] = _ProcessSnapshotEntry(1, 0, "app.exe")
    def metadata(_pid: int) -> _ProcessMetadata:
        return _ProcessMetadata(r"C:\Apps\app.exe", None)

    assert _resolve_capture_pid(21, processes, metadata, max_parent_hops=16) == 5


def test_only_exact_windows_shared_host_paths_are_unsafe() -> None:
    assert _is_unsafe_shared_capture_root(
        _ProcessMetadata(r"C:\Windows\explorer.exe", None),
        _ProcessSnapshotEntry(20, 0, "not-explorer.exe"),
        system_root=r"C:\Windows",
    )
    assert _is_unsafe_shared_capture_root(
        _ProcessMetadata(
            r"C:\Windows\System32\ApplicationFrameHost.exe", None
        ),
        _ProcessSnapshotEntry(20, 0, "not-application-frame-host.exe"),
        system_root=r"C:\Windows",
    )
    assert not _is_unsafe_shared_capture_root(
        _ProcessMetadata(r"D:\Tools\explorer.exe", None),
        _ProcessSnapshotEntry(20, 0, "explorer.exe"),
        system_root=r"C:\Windows",
    )
    assert not _is_unsafe_shared_capture_root(
        _ProcessMetadata(None, None),
        _ProcessSnapshotEntry(20, 0, "app.exe"),
        system_root=r"C:\Windows",
    )


@pytest.mark.parametrize("exe_name", ["explorer.exe", "ApplicationFrameHost.exe"])
def test_missing_shared_host_path_falls_back_to_snapshot_name(exe_name: str) -> None:
    api = _FakeWindowsApi()
    api.processes[20] = _ProcessSnapshotEntry(20, 0, exe_name)
    api.metadata[20] = _ProcessMetadata(None, None)

    assert _list(api) == []


def test_available_third_party_explorer_path_overrides_snapshot_name_fallback() -> None:
    api = _FakeWindowsApi()
    api.processes[20] = _ProcessSnapshotEntry(20, 0, "explorer.exe")
    api.metadata[20] = _ProcessMetadata(r"D:\ThirdParty\explorer.exe", None)

    assert _list(api) == [
        _target(
            process_name="explorer.exe",
            executable_path=r"D:\ThirdParty\explorer.exe",
        )
    ]


def test_unsafe_shared_host_window_is_not_listed() -> None:
    api = _FakeWindowsApi()
    api.processes[20] = _ProcessSnapshotEntry(20, 0, "ApplicationFrameHost.exe")
    api.metadata[20] = _ProcessMetadata(
        r"C:\Windows\System32\ApplicationFrameHost.exe", "Windows.App"
    )

    assert _list(api) == []


def test_validation_accepts_same_hwnd_pid_and_changed_title() -> None:
    api = _FakeWindowsApi()
    api.windows[100].title = "Renamed document"

    assert _validate_windows_application_audio_target(
        _target(), api, current_pid=999
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda api: setattr(api.windows[100], "exists", False),
        lambda api: setattr(api.windows[100], "pid", 21),
        lambda api: setattr(api.windows[100], "visible", False),
        lambda api: setattr(api.windows[100], "cloaked", True),
    ],
)
def test_validation_rejects_destroyed_reused_or_no_longer_candidate_window(
    mutate,
) -> None:
    api = _FakeWindowsApi()
    mutate(api)

    assert not _validate_windows_application_audio_target(
        _target(), api, current_pid=999
    )


def test_validation_rejects_changed_capture_root() -> None:
    api = _FakeWindowsApi()
    api.processes[10] = _ProcessSnapshotEntry(10, 0, "app.exe")
    api.metadata[10] = _ProcessMetadata(r"C:\Apps\app.exe", None)

    assert not _validate_windows_application_audio_target(
        _target(capture_pid=20), api, current_pid=999
    )
    assert _validate_windows_application_audio_target(
        _target(capture_pid=10), api, current_pid=999
    )


def test_validation_metadata_loss_is_valid_only_when_root_does_not_change() -> None:
    api = _FakeWindowsApi()
    api.metadata_errors.add(20)
    assert _validate_windows_application_audio_target(
        _target(capture_pid=20), api, current_pid=999
    )

    api.processes[10] = _ProcessSnapshotEntry(10, 0, "app.exe")
    api.metadata[10] = _ProcessMetadata(r"C:\Apps\app.exe", None)
    assert not _validate_windows_application_audio_target(
        _target(capture_pid=10), api, current_pid=999
    )


def test_validation_rejects_unsafe_capture_root() -> None:
    api = _FakeWindowsApi()
    api.processes[20] = _ProcessSnapshotEntry(20, 0, "explorer.exe")
    api.metadata[20] = _ProcessMetadata(r"C:\Windows\explorer.exe", None)

    assert not _validate_windows_application_audio_target(
        _target(capture_pid=20), api, current_pid=999
    )


@pytest.mark.parametrize("exe_name", ["explorer.exe", "ApplicationFrameHost.exe"])
def test_validation_rejects_shared_host_when_metadata_path_becomes_unavailable(
    exe_name: str,
) -> None:
    api = _FakeWindowsApi()
    api.processes[20] = _ProcessSnapshotEntry(20, 0, exe_name)
    api.metadata_errors.add(20)

    assert not _validate_windows_application_audio_target(
        _target(capture_pid=20), api, current_pid=999
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 ABI test")
def test_ctypes_win32_abi_and_get_window_long_signature_are_pointer_safe() -> None:
    pointer_size = ctypes.sizeof(ctypes.c_void_p)

    assert ctypes.sizeof(wintypes.HWND) == pointer_size
    assert ctypes.sizeof(wintypes.HANDLE) == pointer_size
    assert ctypes.sizeof(wintypes.LPARAM) == pointer_size
    assert ctypes.sizeof(ctypes.c_ssize_t) == pointer_size
    assert ctypes.sizeof(wintypes.DWORD) == 4
    assert ctypes.sizeof(wintypes.BOOL) == 4
    assert ctypes.sizeof(wintypes.LONG) == 4  # HRESULT

    api = _CtypesWindowsApi()
    expected_name = "GetWindowLongPtrW" if pointer_size == 8 else "GetWindowLongW"
    expected_restype = ctypes.c_ssize_t if pointer_size == 8 else wintypes.LONG

    assert api._get_window_long is getattr(api._user32, expected_name)
    assert api._get_window_long.argtypes == [wintypes.HWND, ctypes.c_int]
    assert api._get_window_long.restype is expected_restype
    assert api._dwmapi.DwmGetWindowAttribute.restype is wintypes.LONG


class _FakeDwmApi:
    def __init__(self, *, result: int, cloaked: int) -> None:
        self.result = result
        self.cloaked = cloaked

    def DwmGetWindowAttribute(
        self, _hwnd: int, _attribute: int, output_pointer, _output_size: int
    ) -> int:
        output_pointer._obj.value = self.cloaked
        return self.result


@pytest.mark.parametrize(("cloaked", "expected"), [(0, False), (1, True), (7, True)])
def test_dwm_success_uses_cloak_output(cloaked: int, expected: bool) -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    api._dwmapi = _FakeDwmApi(result=0, cloaked=cloaked)

    assert api.is_window_cloaked(100) is expected


def test_dwm_hresult_failure_is_not_treated_as_cloak_output() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    api._dwmapi = _FakeDwmApi(result=-2_147_467_259, cloaked=1)

    with pytest.raises(OSError, match="DwmGetWindowAttribute failed"):
        api.is_window_cloaked(100)


class _CallbackExceptionHwnd:
    def __int__(self) -> int:
        raise ValueError("callback conversion failed")


class _CallbackStoppingUser32:
    callback_result: bool | None = None

    def EnumWindows(self, callback, lparam: int) -> bool:
        self.callback_result = callback(_CallbackExceptionHwnd(), lparam)
        return False


def test_enum_windows_callback_exception_is_raised_after_native_call() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    user32 = _CallbackStoppingUser32()
    api._user32 = user32
    api._wndenumproc_type = lambda callback: callback

    with pytest.raises(WindowsApplicationTargetError) as error:
        api.enum_windows()

    assert user32.callback_result is False
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "callback conversion failed"


class _FailingEnumWindowsUser32:
    def EnumWindows(self, _callback, _lparam: int) -> bool:
        ctypes.set_last_error(5)
        return False


def test_enum_windows_api_failure_is_distinct_from_callback_stop() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    api._user32 = _FailingEnumWindowsUser32()
    api._wndenumproc_type = lambda callback: callback

    with pytest.raises(WindowsApplicationTargetError, match="Win32 error 5") as error:
        api.enum_windows()

    assert error.value.__cause__ is None


class _FakeKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []
        self._returned_process = False

    def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
        return 123

    def Process32FirstW(self, _snapshot: int, entry_pointer) -> bool:
        entry = entry_pointer._obj
        entry.th32ProcessID = 20
        entry.th32ParentProcessID = 10
        entry.szExeFile = "app.exe"
        self._returned_process = True
        return True

    def Process32NextW(self, _snapshot: int, _entry_pointer) -> bool:
        targets_module.ctypes.set_last_error(targets_module._ERROR_NO_MORE_FILES)
        return False

    def CloseHandle(self, handle: int) -> bool:
        self.closed.append(handle)
        return True


def test_toolhelp_snapshot_handle_is_closed() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _FakeKernel32()
    api._kernel32 = kernel32

    assert api.snapshot_processes() == {
        20: _ProcessSnapshotEntry(20, 10, "app.exe")
    }
    assert kernel32.closed == [123]


class _InvalidSnapshotKernel32:
    def __init__(self) -> None:
        self.first_calls = 0
        self.closed: list[int] = []

    def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
        ctypes.set_last_error(5)
        return ctypes.c_void_p(-1).value

    def Process32FirstW(self, _snapshot: int, _entry_pointer) -> bool:
        self.first_calls += 1
        return False

    def CloseHandle(self, handle: int) -> bool:
        self.closed.append(handle)
        return True


def test_toolhelp_invalid_handle_does_not_enter_enumeration_or_close() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _InvalidSnapshotKernel32()
    api._kernel32 = kernel32

    with pytest.raises(WindowsApplicationTargetError, match="Win32 error 5"):
        api.snapshot_processes()

    assert kernel32.first_calls == 0
    assert kernel32.closed == []


class _FirstFailureKernel32(_FakeKernel32):
    def Process32FirstW(self, _snapshot: int, _entry_pointer) -> bool:
        ctypes.set_last_error(5)
        return False


def test_toolhelp_first_failure_still_closes_snapshot_once() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _FirstFailureKernel32()
    api._kernel32 = kernel32

    with pytest.raises(WindowsApplicationTargetError, match="Win32 error 5"):
        api.snapshot_processes()

    assert kernel32.closed == [123]


class _NextFailureKernel32(_FakeKernel32):
    def Process32NextW(self, _snapshot: int, _entry_pointer) -> bool:
        ctypes.set_last_error(5)
        return False


def test_toolhelp_next_failure_still_closes_snapshot_once() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _NextFailureKernel32()
    api._kernel32 = kernel32

    with pytest.raises(WindowsApplicationTargetError, match="Win32 error 5"):
        api.snapshot_processes()

    assert kernel32.closed == [123]


class _FakeMetadataKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []

    def OpenProcess(self, _rights: int, _inherit: bool, _pid: int) -> int:
        return 456

    def CloseHandle(self, handle: int) -> bool:
        self.closed.append(handle)
        return True


def test_process_handle_is_closed_when_metadata_query_fails(monkeypatch) -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _FakeMetadataKernel32()
    api._kernel32 = kernel32
    monkeypatch.setattr(
        api,
        "_query_process_image_path",
        lambda _process: (_ for _ in ()).throw(OSError("process exited")),
    )

    with pytest.raises(OSError, match="process exited"):
        api.get_process_metadata(20)
    assert kernel32.closed == [456]


class _GrowingTitleUser32:
    def __init__(self) -> None:
        self.length_calls = 0
        self.read_calls = 0

    def GetWindowTextLengthW(self, _hwnd: int) -> int:
        self.length_calls += 1
        return 2 if self.length_calls == 1 else 3

    def GetWindowTextW(self, _hwnd: int, buffer, _size: int) -> int:
        self.read_calls += 1
        buffer.value = "会议" if self.read_calls == 1 else "会议中"
        return 2 if self.read_calls == 1 else 3


def test_ctypes_title_read_retries_once_when_title_fills_buffer() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    user32 = _GrowingTitleUser32()
    api._user32 = user32

    assert api.get_window_title(100) == "会议中"
    assert user32.length_calls == 2
    assert user32.read_calls == 2


class _NoApplicationIdentityKernel32:
    def __init__(self) -> None:
        self.calls = 0

    def GetApplicationUserModelId(self, _process, _length, _buffer) -> int:
        self.calls += 1
        return targets_module._APPMODEL_ERROR_NO_APPLICATION


def test_desktop_process_without_aumid_is_normal() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _NoApplicationIdentityKernel32()
    api._kernel32 = kernel32

    assert api._query_application_user_model_id(456) is None
    assert kernel32.calls == 1


class _ApplicationIdentityKernel32:
    def __init__(self) -> None:
        self.calls = 0

    def GetApplicationUserModelId(self, _process, length_pointer, buffer) -> int:
        self.calls += 1
        if buffer is None:
            length_pointer._obj.value = len("Vendor.App") + 1
            return targets_module._ERROR_INSUFFICIENT_BUFFER
        buffer.value = "Vendor.App"
        return 0


def test_ctypes_aumid_query_uses_bounded_two_call_buffer_pattern() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _ApplicationIdentityKernel32()
    api._kernel32 = kernel32

    assert api._query_application_user_model_id(456) == "Vendor.App"
    assert kernel32.calls == 2


class _DeniedMetadataKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []

    def OpenProcess(self, _rights: int, _inherit: bool, _pid: int) -> int:
        return 0

    def CloseHandle(self, handle: int) -> bool:
        self.closed.append(handle)
        return True


def test_open_process_denied_returns_best_effort_empty_metadata() -> None:
    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel32 = _DeniedMetadataKernel32()
    api._kernel32 = kernel32

    assert api.get_process_metadata(20) == _ProcessMetadata()
    assert kernel32.closed == []
