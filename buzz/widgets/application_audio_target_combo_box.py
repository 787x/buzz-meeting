from __future__ import annotations

import ntpath
from typing import Iterable

from PyQt6.QtWidgets import QComboBox, QWidget

from buzz.audio_capture.windows_application_targets import (
    WindowsApplicationAudioTarget,
)
from buzz.locale import _


class ApplicationAudioTargetComboBox(QComboBox):
    """Displays window context while retaining application-level target data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._targets: list[WindowsApplicationAudioTarget] = []
        self._set_status_item(_("Select an application…"))

    @property
    def selected_target(self) -> WindowsApplicationAudioTarget | None:
        target = self.currentData()
        if isinstance(target, WindowsApplicationAudioTarget):
            return target
        return None

    @property
    def selection_key(self) -> tuple[int, int] | None:
        target = self.selected_target
        return None if target is None else target.selection_key

    @property
    def targets(self) -> tuple[WindowsApplicationAudioTarget, ...]:
        return tuple(self._targets)

    def set_targets(
        self,
        targets: Iterable[WindowsApplicationAudioTarget],
        *,
        preserve_selection: bool = True,
    ) -> None:
        previous_key = self.selection_key if preserve_selection else None
        self._targets = sorted(
            targets,
            key=lambda target: (
                (target.process_name or "").casefold(),
                target.window_title.casefold(),
                target.window_pid,
                target.hwnd,
            ),
        )

        previous_signals_blocked = self.blockSignals(True)
        try:
            self.clear()
            if not self._targets:
                self.addItem(_("No applications available"), None)
            else:
                self.addItem(_("Select an application…"), None)
                restored_index = 0
                for target in self._targets:
                    self.addItem(self._display_label(target), target)
                    if target.selection_key == previous_key:
                        restored_index = self.count() - 1
                self.setCurrentIndex(restored_index)
        finally:
            self.blockSignals(previous_signals_blocked)

        self.currentIndexChanged.emit(self.currentIndex())

    def set_refresh_error(self) -> None:
        self._targets = []
        self._set_status_item(_("Unable to refresh applications"))

    def reset_to_placeholder(self) -> None:
        self._targets = []
        self._set_status_item(_("Select an application…"))

    def _set_status_item(self, text: str) -> None:
        previous_signals_blocked = self.blockSignals(True)
        try:
            self.clear()
            self.addItem(text, None)
            self.setCurrentIndex(0)
        finally:
            self.blockSignals(previous_signals_blocked)
        self.currentIndexChanged.emit(self.currentIndex())

    @staticmethod
    def _display_label(target: WindowsApplicationAudioTarget) -> str:
        process_name = target.process_name
        if process_name:
            display_name = ntpath.splitext(ntpath.basename(process_name))[0]
            if display_name:
                return f"{display_name} — {target.window_title}"
        return target.window_title
