"""Accessible table window for browsing durable meeting headers."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSignalBlocker,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
from buzz.meeting.meeting_session import MeetingRemoteSourceKind, MeetingSessionState
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
    new_meeting_requested = pyqtSignal()

    def __init__(
        self,
        service: MeetingLibraryService,
        parent: QWidget | None = None,
        flags: Qt.WindowType = Qt.WindowType.Widget,
    ) -> None:
        super().__init__(parent, flags)
        self._service = service
        self._entries: tuple[MeetingLibraryEntry, ...] = ()
        self._has_loaded = False
        self._load_failed = False
        self.setWindowTitle(_("Meetings"))
        self.resize(900, 500)

        self.new_meeting_button = QPushButton(_("New Meeting"), self)
        self.new_meeting_button.clicked.connect(self.new_meeting_requested.emit)
        self.open_meeting_button = QPushButton(_("Open Meeting"), self)
        self.open_meeting_button.setEnabled(False)
        self.open_meeting_button.clicked.connect(self._open_selected_meeting)
        self.refresh_button = QPushButton(_("Refresh"), self)
        self.refresh_button.clicked.connect(self.refresh)

        self.source_filter = QComboBox(self)
        self.source_filter.addItem(_("All"), None)
        for source in MeetingRemoteSourceKind:
            self.source_filter.addItem(format_remote_source(source), source)
        self.state_filter = QComboBox(self)
        self.state_filter.addItem(_("All"), None)
        for state in MeetingSessionState:
            self.state_filter.addItem(format_meeting_state(state), state)
        self.reset_filters_button = QPushButton(_("Reset Filters"), self)
        self.reset_filters_button.setEnabled(False)
        self.reset_filters_button.clicked.connect(self._reset_filters)
        self.source_filter.currentIndexChanged.connect(self._apply_filters)
        self.state_filter.currentIndexChanged.connect(self._apply_filters)

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
        self.table_view.selectionModel().selectionChanged.connect(
            self._update_open_button
        )
        header = self.table_view.horizontalHeader()
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(3, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

        self.state_label = QLabel(self)
        self.state_label.setWordWrap(True)
        self.state_label.hide()

        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        actions.addWidget(self.new_meeting_button)
        actions.addWidget(self.open_meeting_button)
        actions.addStretch()
        actions.addWidget(self.refresh_button)
        layout.addLayout(actions)
        filters = QHBoxLayout()
        for text, combo in (
            (_("Source"), self.source_filter),
            (_("Meeting Status"), self.state_filter),
        ):
            label = QLabel(text, self)
            label.setBuddy(combo)
            filters.addWidget(label)
            filters.addWidget(combo)
        filters.addStretch()
        filters.addWidget(self.reset_filters_button)
        layout.addLayout(filters)
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

    def _open_selected_meeting(self) -> None:
        selected_id = self.selected_meeting_id()
        if selected_id is not None:
            self.meeting_open_requested.emit(selected_id)

    def _update_open_button(self) -> None:
        self.open_meeting_button.setEnabled(self.selected_meeting_id() is not None)

    def _reset_filters(self) -> None:
        with QSignalBlocker(self.source_filter), QSignalBlocker(self.state_filter):
            self.source_filter.setCurrentIndex(0)
            self.state_filter.setCurrentIndex(0)
        self._apply_filters()

    def _apply_filters(self) -> None:
        selected_id = self.selected_meeting_id()
        source = self.source_filter.currentData()
        state = self.state_filter.currentData()
        entries = tuple(
            entry
            for entry in self._entries
            if (source is None or entry.remote_source_kind == source)
            and (state is None or entry.session_state == state)
        )
        self.reset_filters_button.setEnabled(source is not None or state is not None)
        self.table_model.replace_entries(entries)
        self.table_view.clearSelection()
        self.table_view.setCurrentIndex(QModelIndex())
        for row, entry in enumerate(entries):
            if entry.session_id == selected_id:
                self.table_view.selectRow(row)
                break
        self._update_open_button()
        self._update_state_label()

    def _update_state_label(self) -> None:
        if self._load_failed:
            message = _("Could not load meetings.")
        elif not self._has_loaded:
            message = ""
        elif not self._entries:
            message = _("No meetings yet.")
        elif self.table_model.rowCount() == 0:
            message = _("No matching meetings.")
        else:
            message = ""
        self.state_label.setText(message)
        self.state_label.setVisible(bool(message))

    def refresh(self) -> None:
        try:
            entries = self._service.list_meetings()
        except MeetingLibraryError:
            logging.exception("Could not load meetings library")
            self._load_failed = True
            self._update_state_label()
            return

        self._entries = entries
        self._has_loaded = True
        self._load_failed = False
        self._apply_filters()


__all__ = ["MeetingLibraryTableModel", "MeetingsLibraryWidget"]
