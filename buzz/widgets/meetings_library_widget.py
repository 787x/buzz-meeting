"""Accessible table window for browsing durable meeting headers."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from buzz.locale import _
from buzz.meeting.meeting_library import (
    MeetingLibraryEntry,
    MeetingLibraryError,
    MeetingLibraryService,
)
from buzz.widgets.meeting_presentation import (
    format_audio_status,
    format_duration,
    format_meeting_datetime,
    format_meeting_state,
    format_remote_source,
)


class MeetingLibraryTableModel(QAbstractTableModel):
    """Qt table model backed only by immutable meeting library entries."""

    _HEADERS = (
        lambda: _("Date"),
        lambda: _("Duration"),
        lambda: _("Source"),
        lambda: _("Meeting Status"),
        lambda: _("Audio Status"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: tuple[MeetingLibraryEntry, ...] = ()

    def replace_entries(self, entries: tuple[MeetingLibraryEntry, ...]) -> None:
        self.beginResetModel()
        self._entries = entries
        self.endResetModel()

    def meeting_at(self, row: int) -> MeetingLibraryEntry:
        if row < 0 or row >= len(self._entries):
            raise IndexError("meeting row is out of range")
        return self._entries[row]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._HEADERS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
            and 0 <= section < len(self._HEADERS)
        ):
            return self._HEADERS[section]()
        return None

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        entry = self.meeting_at(index.row())
        values = (
            format_meeting_datetime(entry.display_at),
            format_duration(entry.duration_seconds),
            format_remote_source(entry.remote_source_kind),
            format_meeting_state(entry.session_state),
            format_audio_status(entry.audio_state, entry.audio_outcome),
        )
        return values[index.column()]


class _MeetingTableView(QTableView):
    open_requested = pyqtSignal(QModelIndex)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            selected_rows = self.selectionModel().selectedRows()
            if selected_rows:
                self.open_requested.emit(selected_rows[0])
            event.accept()
            return
        super().keyPressEvent(event)


class MeetingsLibraryWidget(QWidget):
    """Reusable meetings window whose caller owns refresh timing."""

    meeting_open_requested = pyqtSignal(object)

    def __init__(
        self,
        service: MeetingLibraryService,
        parent: QWidget | None = None,
        flags: Qt.WindowType = Qt.WindowType.Widget,
    ) -> None:
        super().__init__(parent, flags)
        self._service = service
        self.setWindowTitle(_("Meetings"))

        self.table_model = MeetingLibraryTableModel(self)
        self.table_view = _MeetingTableView(self)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.setSortingEnabled(False)
        self.table_view.doubleClicked.connect(self._request_open)
        self.table_view.open_requested.connect(self._request_open)

        self.state_label = QLabel(self)
        self.state_label.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self.state_label)
        layout.addWidget(self.table_view)

    def selected_meeting_id(self) -> uuid.UUID | None:
        selected_rows = self.table_view.selectionModel().selectedRows()
        if not selected_rows:
            return None
        return self.table_model.meeting_at(selected_rows[0].row()).session_id

    def _request_open(self, index: QModelIndex) -> None:
        if not index.isValid() or index.model() is not self.table_model:
            return
        try:
            entry = self.table_model.meeting_at(index.row())
        except IndexError:
            return
        self.meeting_open_requested.emit(entry.session_id)

    def refresh(self) -> None:
        selected_id = self.selected_meeting_id()
        try:
            entries = self._service.list_meetings()
        except MeetingLibraryError:
            logging.exception("Could not load meetings library")
            self.state_label.setText(_("Could not load meetings."))
            self.state_label.show()
            return

        self.table_model.replace_entries(entries)
        if entries:
            self.state_label.clear()
            self.state_label.hide()
        else:
            self.state_label.setText(_("No meetings yet."))
            self.state_label.show()

        self.table_view.clearSelection()
        if selected_id is None:
            return
        for row, entry in enumerate(entries):
            if entry.session_id == selected_id:
                self.table_view.selectRow(row)
                return


__all__ = ["MeetingLibraryTableModel", "MeetingsLibraryWidget"]
