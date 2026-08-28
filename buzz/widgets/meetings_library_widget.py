"""Accessible table window for browsing durable meeting headers."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from buzz.locale import _
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
)
from buzz.meeting.meeting_library import (
    MeetingLibraryEntry,
    MeetingLibraryError,
    MeetingLibraryService,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionState,
)


def _format_date(entry: MeetingLibraryEntry) -> str:
    return entry.display_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(entry: MeetingLibraryEntry) -> str:
    if entry.duration_seconds is None:
        return ""
    seconds = int(entry.duration_seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_seconds = divmod(seconds, 3600)
    minutes = remaining_seconds // 60
    return f"{hours}h {minutes:02d}m"


_SOURCE_LABELS: dict[MeetingRemoteSourceKind, Callable[[], str]] = {
    MeetingRemoteSourceKind.SYSTEM: lambda: _("System audio"),
    MeetingRemoteSourceKind.APPLICATION: lambda: _("Application audio"),
}

_MEETING_STATUS_LABELS: dict[MeetingSessionState, Callable[[], str]] = {
    MeetingSessionState.CREATED: lambda: _("Created"),
    MeetingSessionState.STARTING: lambda: _("Starting"),
    MeetingSessionState.ACTIVE: lambda: _("Active"),
    MeetingSessionState.STOPPING: lambda: _("Stopping"),
    MeetingSessionState.COMPLETED: lambda: _("Completed"),
    MeetingSessionState.FAILED: lambda: _("Failed"),
}

_AUDIO_STATE_LABELS: dict[MeetingAudioTracksState, Callable[[], str]] = {
    MeetingAudioTracksState.CREATED: lambda: _("Created"),
    MeetingAudioTracksState.STARTING: lambda: _("Starting"),
    MeetingAudioTracksState.RUNNING: lambda: _("Running"),
    MeetingAudioTracksState.DEGRADED: lambda: _("Degraded"),
    MeetingAudioTracksState.STOPPING: lambda: _("Stopping"),
    MeetingAudioTracksState.STOPPED: lambda: _("Stopped"),
    MeetingAudioTracksState.FAILED: lambda: _("Failed"),
}

_AUDIO_OUTCOME_LABELS: dict[MeetingAudioTracksOutcome, Callable[[], str]] = {
    MeetingAudioTracksOutcome.COMPLETE: lambda: _("Complete"),
    MeetingAudioTracksOutcome.PARTIAL: lambda: _("Partial"),
    MeetingAudioTracksOutcome.FAILED: lambda: _("Failed"),
}


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
            _format_date(entry),
            _format_duration(entry),
            _SOURCE_LABELS[entry.remote_source_kind](),
            _MEETING_STATUS_LABELS[entry.session_state](),
            (
                _AUDIO_OUTCOME_LABELS[entry.audio_outcome]()
                if entry.audio_outcome is not None
                else _AUDIO_STATE_LABELS[entry.audio_state]()
            ),
        )
        return values[index.column()]


class MeetingsLibraryWidget(QWidget):
    """Reusable meetings window whose caller owns refresh timing."""

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
        self.table_view = QTableView(self)
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.setSortingEnabled(False)

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
