from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QApplication

import buzz.widgets.meetings_library_widget as widget_module
from buzz.locale import _
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
)
from buzz.meeting.meeting_library import (
    MeetingLibraryDatabaseError,
    MeetingLibraryEntry,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionState,
)
from buzz.widgets.meetings_library_widget import (
    MeetingLibraryTableModel,
    MeetingsLibraryWidget,
)


class FakeService:
    def __init__(self, results=()) -> None:
        self.results = list(results)
        self.calls = 0

    def list_meetings(self) -> tuple[MeetingLibraryEntry, ...]:
        self.calls += 1
        result = self.results.pop(0) if self.results else ()
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


def make_entry(
    number: int = 1,
    *,
    remote_source_kind=MeetingRemoteSourceKind.SYSTEM,
    session_state=MeetingSessionState.COMPLETED,
    created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    started_at=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
    ended_at=datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc),
    duration_ns=60_000_000_000,
    audio_state=MeetingAudioTracksState.STOPPED,
    audio_outcome=MeetingAudioTracksOutcome.COMPLETE,
) -> MeetingLibraryEntry:
    return MeetingLibraryEntry(
        session_id=uuid.UUID(f"00000000-0000-0000-0000-{number:012d}"),
        remote_source_kind=remote_source_kind,
        session_state=session_state,
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        duration_ns=duration_ns,
        audio_state=audio_state,
        audio_outcome=audio_outcome,
    )


def make_widget(qtbot, service: FakeService) -> MeetingsLibraryWidget:
    widget = MeetingsLibraryWidget(service=service)
    qtbot.add_widget(widget)
    return widget


def cell(widget: MeetingsLibraryWidget, row: int, column: int):
    return widget.table_model.index(row, column).data()


def test_constructor_does_not_refresh(qtbot) -> None:
    service = FakeService()
    widget = make_widget(qtbot, service)
    assert service.calls == 0
    assert widget.table_model.rowCount() == 0
    assert widget.state_label.isHidden()


def test_first_explicit_refresh_calls_service_once_and_shows_empty_state(qtbot) -> None:
    service = FakeService([()])
    widget = make_widget(qtbot, service)
    widget.refresh()
    assert service.calls == 1
    assert widget.table_model.rowCount() == 0
    assert widget.state_label.text() == _("No meetings yet.")
    assert not widget.state_label.isHidden()


def test_model_has_exactly_five_localized_headers() -> None:
    model = MeetingLibraryTableModel()
    assert model.columnCount() == 5
    assert [
        model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        for column in range(5)
    ] == [
        _("Date"),
        _("Duration"),
        _("Source"),
        _("Meeting Status"),
        _("Audio Status"),
    ]


def test_one_row_renders_date_duration_source_and_status(qtbot) -> None:
    entry = make_entry()
    widget = make_widget(qtbot, FakeService([(entry,)]))
    widget.refresh()
    assert widget.table_model.rowCount() == 1
    assert [cell(widget, 0, column) for column in range(5)] == [
        entry.display_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "1m 00s",
        _("System audio"),
        _("Completed"),
        _("Complete"),
    ]


def test_multiple_rows_render(qtbot) -> None:
    entries = (make_entry(1), make_entry(2), make_entry(3))
    widget = make_widget(qtbot, FakeService([entries]))
    widget.refresh()
    assert widget.table_model.rowCount() == 3
    assert widget.table_model.meeting_at(2) == entries[2]


@pytest.mark.parametrize(
    ("duration_ns", "expected"),
    [
        (None, ""),
        (42_000_000_000, "42s"),
        (312_000_000_000, "5m 12s"),
        (3_780_000_000_000, "1h 03m"),
    ],
)
def test_duration_formatting(qtbot, duration_ns, expected) -> None:
    widget = make_widget(qtbot, FakeService())
    widget.table_model.replace_entries((make_entry(duration_ns=duration_ns),))
    assert cell(widget, 0, 1) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (MeetingRemoteSourceKind.SYSTEM, "System audio"),
        (MeetingRemoteSourceKind.APPLICATION, "Application audio"),
    ],
)
def test_source_labels(qtbot, source, expected) -> None:
    widget = make_widget(qtbot, FakeService())
    widget.table_model.replace_entries((make_entry(remote_source_kind=source),))
    assert cell(widget, 0, 2) == _(expected)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (MeetingSessionState.CREATED, "Created"),
        (MeetingSessionState.STARTING, "Starting"),
        (MeetingSessionState.ACTIVE, "Active"),
        (MeetingSessionState.STOPPING, "Stopping"),
        (MeetingSessionState.COMPLETED, "Completed"),
        (MeetingSessionState.FAILED, "Failed"),
    ],
)
def test_meeting_status_labels(qtbot, status, expected) -> None:
    widget = make_widget(qtbot, FakeService())
    widget.table_model.replace_entries((make_entry(session_state=status),))
    assert cell(widget, 0, 3) == _(expected)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (MeetingAudioTracksState.CREATED, "Created"),
        (MeetingAudioTracksState.STARTING, "Starting"),
        (MeetingAudioTracksState.RUNNING, "Running"),
        (MeetingAudioTracksState.DEGRADED, "Degraded"),
        (MeetingAudioTracksState.STOPPING, "Stopping"),
        (MeetingAudioTracksState.STOPPED, "Stopped"),
        (MeetingAudioTracksState.FAILED, "Failed"),
    ],
)
def test_audio_state_labels_when_outcome_is_absent(qtbot, state, expected) -> None:
    widget = make_widget(qtbot, FakeService())
    widget.table_model.replace_entries(
        (make_entry(audio_state=state, audio_outcome=None),)
    )
    assert cell(widget, 0, 4) == _(expected)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (MeetingAudioTracksOutcome.COMPLETE, "Complete"),
        (MeetingAudioTracksOutcome.PARTIAL, "Partial"),
        (MeetingAudioTracksOutcome.FAILED, "Failed"),
    ],
)
def test_audio_outcome_takes_precedence(qtbot, outcome, expected) -> None:
    widget = make_widget(qtbot, FakeService())
    widget.table_model.replace_entries(
        (
            make_entry(
                audio_state=MeetingAudioTracksState.RUNNING, audio_outcome=outcome
            ),
        )
    )
    assert cell(widget, 0, 4) == _(expected)


def test_meeting_at_has_deterministic_index_error_policy() -> None:
    model = MeetingLibraryTableModel()
    entry = make_entry()
    model.replace_entries((entry,))
    assert model.meeting_at(0) is entry
    with pytest.raises(IndexError):
        model.meeting_at(-1)
    with pytest.raises(IndexError):
        model.meeting_at(1)


def test_table_selection_is_single_row_and_returns_selected_uuid(qtbot) -> None:
    entries = (make_entry(1), make_entry(2))
    widget = make_widget(qtbot, FakeService([entries]))
    widget.refresh()
    assert (
        widget.table_view.selectionBehavior()
        == QAbstractItemView.SelectionBehavior.SelectRows
    )
    assert (
        widget.table_view.selectionMode()
        == QAbstractItemView.SelectionMode.SingleSelection
    )
    assert (
        widget.table_view.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    )
    assert not widget.table_view.isSortingEnabled()
    assert widget.selected_meeting_id() is None
    widget.table_view.selectRow(1)
    assert widget.selected_meeting_id() == entries[1].session_id


def test_successful_refresh_replaces_rows_with_fresh_values(qtbot) -> None:
    first = (make_entry(1),)
    second = (make_entry(2), make_entry(3))
    service = FakeService([first, second])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.refresh()
    assert service.calls == 2
    assert widget.table_model.rowCount() == 2
    assert widget.table_model.meeting_at(0) == second[0]


def test_selection_is_restored_by_uuid_after_successful_refresh(qtbot) -> None:
    selected = make_entry(2)
    service = FakeService([(make_entry(1), selected), (selected, make_entry(3))])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.table_view.selectRow(1)
    widget.refresh()
    assert widget.selected_meeting_id() == selected.session_id


def test_removed_selected_uuid_clears_selection(qtbot) -> None:
    service = FakeService([(make_entry(1),), (make_entry(2),)])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.table_view.selectRow(0)
    widget.refresh()
    assert widget.selected_meeting_id() is None


def test_first_load_failure_shows_error_with_empty_model(qtbot) -> None:
    widget = make_widget(
        qtbot, FakeService([MeetingLibraryDatabaseError("database unavailable")])
    )
    widget.refresh()
    assert widget.table_model.rowCount() == 0
    assert widget.state_label.text() == _("Could not load meetings.")
    assert not widget.state_label.isHidden()


def test_failure_after_valid_rows_retains_exact_rows_and_selection(qtbot) -> None:
    entries = (make_entry(1), make_entry(2))
    service = FakeService(
        [entries, MeetingLibraryDatabaseError("database unavailable")]
    )
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.table_view.selectRow(1)
    widget.refresh()
    assert widget.table_model.rowCount() == 2
    assert tuple(widget.table_model.meeting_at(row) for row in range(2)) == entries
    assert widget.selected_meeting_id() == entries[1].session_id
    assert widget.state_label.text() == _("Could not load meetings.")


def test_subsequent_success_clears_error(qtbot) -> None:
    entry = make_entry()
    service = FakeService(
        [MeetingLibraryDatabaseError("database unavailable"), (entry,)]
    )
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.refresh()
    assert widget.state_label.text() == ""
    assert widget.state_label.isHidden()
    assert widget.table_model.meeting_at(0) is entry


def test_widget_has_no_context_detail_or_mutation_features(qtbot) -> None:
    widget = make_widget(qtbot, FakeService())
    assert widget.actions() == []
    for name in (
        "delete_meeting",
        "rename_meeting",
        "open_detail",
        "open_transcript",
    ):
        assert not hasattr(widget, name)
    assert hasattr(widget, "meeting_open_requested")


def test_valid_double_click_emits_one_uuid(qtbot) -> None:
    entry = make_entry()
    widget = make_widget(qtbot, FakeService([(entry,)]))
    widget.refresh()
    received = []
    widget.meeting_open_requested.connect(received.append)
    index = widget.table_model.index(0, 0)

    widget.table_view.doubleClicked.emit(index)

    assert received == [entry.session_id]


@pytest.mark.parametrize("key", [Qt.Key.Key_Enter, Qt.Key.Key_Return])
def test_enter_and_return_emit_selected_uuid_once(qtbot, key) -> None:
    entry = make_entry()
    widget = make_widget(qtbot, FakeService([(entry,)]))
    widget.refresh()
    widget.table_view.selectRow(0)
    received = []
    widget.meeting_open_requested.connect(received.append)

    qtbot.keyClick(widget.table_view, key)

    assert received == [entry.session_id]


def test_open_paths_ignore_no_selection_and_invalid_index(qtbot) -> None:
    entry = make_entry()
    widget = make_widget(qtbot, FakeService([(entry,)]))
    widget.refresh()
    received = []
    widget.meeting_open_requested.connect(received.append)

    qtbot.keyClick(widget.table_view, Qt.Key.Key_Return)
    widget.table_view.doubleClicked.emit(widget.table_model.index(99, 0))

    assert received == []


def test_widget_module_has_no_qsql_dependency() -> None:
    source = inspect.getsource(widget_module)
    assert "QtSql" not in source
    assert "QSql" not in source
    assert "QSqlTableModel" not in source
