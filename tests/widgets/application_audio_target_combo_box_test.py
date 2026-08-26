from __future__ import annotations

from pytestqt.qtbot import QtBot

from buzz.audio_capture.windows_application_targets import (
    WindowsApplicationAudioTarget,
)
from buzz.widgets.application_audio_target_combo_box import (
    ApplicationAudioTargetComboBox,
)


def _target(
    hwnd: int,
    title: str,
    *,
    pid: int | None = None,
    capture_pid: int | None = None,
    process_name: str | None = "app.exe",
) -> WindowsApplicationAudioTarget:
    window_pid = pid if pid is not None else hwnd + 100
    return WindowsApplicationAudioTarget(
        hwnd=hwnd,
        window_title=title,
        window_pid=window_pid,
        capture_pid=capture_pid if capture_pid is not None else window_pid,
        process_name=process_name,
        executable_path=None,
        app_user_model_id=None,
    )


def test_starts_with_selection_placeholder(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)

    assert combo.count() == 1
    assert combo.currentText() == "Select an application…"
    assert combo.selected_target is None
    assert combo.selection_key is None


def test_set_targets_sorts_and_displays_process_stem_and_title(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    alpha = _target(2, "Zulu", process_name="Alpha.exe")
    beta = _target(1, "会议 🎙️", process_name="beta.exe")

    combo.set_targets([beta, alpha])

    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Select an application…",
        "Alpha — Zulu",
        "beta — 会议 🎙️",
    ]
    assert combo.targets == (alpha, beta)
    assert combo.selected_target is None


def test_missing_process_name_displays_only_unicode_title(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    target = _target(1, "Cafe\u0301 \U0001f3b5", process_name=None)

    combo.set_targets([target])

    assert combo.itemText(1) == "Cafe\u0301 \U0001f3b5"


def test_selected_target_and_key_follow_current_item(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    target = _target(7, "Document", pid=42)
    combo.set_targets([target])

    combo.setCurrentIndex(1)

    assert combo.selected_target is target
    assert combo.selection_key == (7, 42)


def test_refresh_preserves_same_hwnd_and_pid(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    original = _target(7, "Old title", pid=42)
    combo.set_targets([original])
    combo.setCurrentIndex(1)
    refreshed = _target(7, "New title", pid=42)

    combo.set_targets([refreshed])

    assert combo.selected_target is refreshed


def test_refresh_loses_selection_when_hwnd_or_pid_changes(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    combo.set_targets([_target(7, "Old", pid=42)])
    combo.setCurrentIndex(1)

    combo.set_targets([_target(7, "Replacement", pid=43)])

    assert combo.currentIndex() == 0
    assert combo.selected_target is None


def test_refresh_never_selects_first_new_target(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)

    combo.set_targets([_target(1, "First")])

    assert combo.currentIndex() == 0
    assert combo.selected_target is None


def test_empty_and_refresh_error_have_distinct_states(qtbot: QtBot) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)

    combo.set_targets([])
    assert combo.currentText() == "No applications available"

    combo.set_refresh_error()
    assert combo.currentText() == "Unable to refresh applications"
    assert combo.targets == ()


def test_duplicate_capture_pid_windows_remain_independent_and_hide_ids(
    qtbot: QtBot,
) -> None:
    combo = ApplicationAudioTargetComboBox()
    qtbot.add_widget(combo)
    meet = _target(7, "Meet", pid=42, capture_pid=10)
    video = _target(8, "YouTube", pid=42, capture_pid=10)

    combo.set_targets([video, meet])

    assert combo.count() == 3
    labels = [combo.itemText(index) for index in (1, 2)]
    assert labels == ["app — Meet", "app — YouTube"]
    assert all("10" not in label and "42" not in label for label in labels)
