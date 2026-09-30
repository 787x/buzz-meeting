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


def visible_entries(widget):
    return tuple(
        widget.table_model.meeting_at(row)
        for row in range(widget.table_model.rowCount())
    )


def test_visible_actions_and_open_enabled_only_with_selection(qtbot) -> None:
    entries = (make_entry(1), make_entry(2))
    widget = make_widget(qtbot, FakeService([entries]))
    widget.refresh()
    assert widget.new_meeting_button.text() == _("New Meeting")
    assert widget.open_meeting_button.text() == _("Open Meeting")
    assert widget.refresh_button.text() == _("Refresh")
    assert widget.new_meeting_button.isEnabled()
    assert widget.refresh_button.isEnabled()
    assert not widget.open_meeting_button.isEnabled()
    received = []
    widget.meeting_open_requested.connect(received.append)
    widget.open_meeting_button.click()
    assert received == []
    widget.table_view.selectRow(1)
    assert widget.open_meeting_button.isEnabled()
    widget.open_meeting_button.click()
    assert received == [entries[1].session_id]
    widget.table_view.clearSelection()
    assert not widget.open_meeting_button.isEnabled()
    widget.open_meeting_button.click()
    assert received == [entries[1].session_id]


def test_new_meeting_requests_once_without_reading_or_opening(qtbot) -> None:
    service = FakeService()
    widget = make_widget(qtbot, service)
    requests, opened = [], []
    widget.new_meeting_requested.connect(lambda: requests.append(True))
    widget.meeting_open_requested.connect(opened.append)
    widget.new_meeting_button.click()
    assert requests == [True]
    assert opened == []
    assert service.calls == 0


def test_refresh_button_reads_once_and_restores_uuid_after_row_moves(qtbot) -> None:
    first, selected, new = make_entry(1), make_entry(2), make_entry(3)
    service = FakeService([(first, selected), (selected, new), (new,)])
    widget = make_widget(qtbot, service)
    widget.refresh_button.click()
    assert service.calls == 1
    widget.table_view.selectRow(1)
    opened = []
    widget.meeting_open_requested.connect(opened.append)
    widget.refresh_button.click()
    assert service.calls == 2
    assert widget.selected_meeting_id() == selected.session_id
    assert widget.open_meeting_button.isEnabled()
    widget.open_meeting_button.click()
    assert opened == [selected.session_id]
    widget.refresh_button.click()
    assert service.calls == 3
    assert widget.selected_meeting_id() is None
    assert not widget.open_meeting_button.isEnabled()
    qtbot.keyClick(widget.table_view, Qt.Key.Key_Return)
    assert opened == [selected.session_id]


def test_filter_options_use_only_existing_source_and_state_values(qtbot) -> None:
    widget = make_widget(qtbot, FakeService())
    assert [widget.source_filter.itemData(i) for i in range(3)] == [
        None,
        MeetingRemoteSourceKind.SYSTEM,
        MeetingRemoteSourceKind.APPLICATION,
    ]
    assert [widget.source_filter.itemText(i) for i in range(3)] == [
        _("All"),
        _("System audio"),
        _("Application audio"),
    ]
    assert [
        widget.state_filter.itemData(i) for i in range(widget.state_filter.count())
    ] == [None, *MeetingSessionState]


@pytest.mark.parametrize("source", list(MeetingRemoteSourceKind))
def test_source_filter_is_local_and_keeps_service_order(qtbot, source) -> None:
    entries = (
        make_entry(3, remote_source_kind=MeetingRemoteSourceKind.APPLICATION),
        make_entry(2),
        make_entry(4, remote_source_kind=MeetingRemoteSourceKind.APPLICATION),
        make_entry(1),
    )
    service = FakeService([entries])
    widget = make_widget(qtbot, service)
    widget.refresh()
    assert visible_entries(widget) == entries
    opened = []
    widget.meeting_open_requested.connect(opened.append)
    widget.source_filter.setCurrentIndex(widget.source_filter.findData(source))
    assert visible_entries(widget) == tuple(
        entry for entry in entries if entry.remote_source_kind == source
    )
    assert service.calls == 1
    assert opened == []
    assert widget.reset_filters_button.isEnabled()
    widget.reset_filters_button.click()
    assert visible_entries(widget) == entries
    assert service.calls == 1
    assert not widget.reset_filters_button.isEnabled()


@pytest.mark.parametrize("state", list(MeetingSessionState))
def test_state_filter_is_local_and_keeps_service_order(qtbot, state) -> None:
    entries = tuple(
        make_entry(i + 1, session_state=value)
        for i, value in enumerate(reversed(list(MeetingSessionState)))
    ) + (make_entry(10, session_state=state),)
    service = FakeService([entries])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.state_filter.setCurrentIndex(widget.state_filter.findData(state))
    assert visible_entries(widget) == tuple(
        entry for entry in entries if entry.session_state == state
    )
    assert service.calls == 1
    widget.reset_filters_button.click()
    assert visible_entries(widget) == entries
    assert service.calls == 1


def test_combined_filters_and_empty_states(qtbot) -> None:
    application = make_entry(1, remote_source_kind=MeetingRemoteSourceKind.APPLICATION)
    failed = make_entry(2, session_state=MeetingSessionState.FAILED)
    service = FakeService([(application, failed), ()])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.source_filter.setCurrentIndex(
        widget.source_filter.findData(MeetingRemoteSourceKind.APPLICATION)
    )
    widget.state_filter.setCurrentIndex(
        widget.state_filter.findData(MeetingSessionState.FAILED)
    )
    assert visible_entries(widget) == ()
    assert widget.state_label.text() == _("No matching meetings.")
    assert not widget.state_label.isHidden()
    assert not widget.open_meeting_button.isEnabled()
    assert service.calls == 1
    widget.refresh()
    assert widget.state_label.text() == _("No meetings yet.")
    assert widget.source_filter.currentData() == MeetingRemoteSourceKind.APPLICATION
    assert widget.state_filter.currentData() == MeetingSessionState.FAILED
    widget.reset_filters_button.click()
    assert widget.source_filter.currentIndex() == 0
    assert widget.state_filter.currentIndex() == 0
    assert widget.state_label.text() == _("No meetings yet.")
    assert service.calls == 2


@pytest.mark.parametrize("filter_name", ["source_filter", "state_filter"])
def test_hidden_selection_cannot_open_after_filter_or_reset(qtbot, filter_name) -> None:
    selected = make_entry(1)
    remaining = make_entry(
        2,
        remote_source_kind=MeetingRemoteSourceKind.APPLICATION,
        session_state=MeetingSessionState.FAILED,
    )
    widget = make_widget(qtbot, FakeService([(selected, remaining)]))
    widget.refresh()
    widget.table_view.selectRow(0)
    opened = []
    widget.meeting_open_requested.connect(opened.append)
    combo = getattr(widget, filter_name)
    value = (
        MeetingRemoteSourceKind.APPLICATION
        if filter_name == "source_filter"
        else MeetingSessionState.FAILED
    )
    combo.setCurrentIndex(combo.findData(value))
    assert visible_entries(widget) == (remaining,)
    assert widget.selected_meeting_id() is None
    assert not widget.table_view.currentIndex().isValid()
    assert not widget.open_meeting_button.isEnabled()
    widget.open_meeting_button.click()
    qtbot.keyClick(widget.table_view, Qt.Key.Key_Return)
    widget.reset_filters_button.click()
    assert widget.selected_meeting_id() is None
    assert not widget.open_meeting_button.isEnabled()
    qtbot.keyClick(widget.table_view, Qt.Key.Key_Enter)
    assert opened == []
    widget.table_view.selectRow(1)
    widget.open_meeting_button.click()
    assert opened == [remaining.session_id]


def test_visible_selection_survives_filter_and_filtered_refresh(qtbot) -> None:
    selected = make_entry(2, session_state=MeetingSessionState.FAILED)
    first = make_entry(1, session_state=MeetingSessionState.FAILED)
    service = FakeService([(first, selected), (selected, first)])
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.table_view.selectRow(1)
    widget.state_filter.setCurrentIndex(
        widget.state_filter.findData(MeetingSessionState.FAILED)
    )
    assert widget.selected_meeting_id() == selected.session_id
    widget.refresh()
    assert widget.selected_meeting_id() == selected.session_id
    assert widget.table_view.selectionModel().selectedRows()[0].row() == 0
    assert widget.open_meeting_button.isEnabled()
    assert service.calls == 2


def test_refresh_clears_selection_if_updated_header_no_longer_matches(qtbot) -> None:
    from dataclasses import replace

    selected = make_entry(1)
    changed = replace(selected, remote_source_kind=MeetingRemoteSourceKind.APPLICATION)
    widget = make_widget(qtbot, FakeService([(selected,), (changed,)]))
    widget.refresh()
    widget.source_filter.setCurrentIndex(
        widget.source_filter.findData(MeetingRemoteSourceKind.SYSTEM)
    )
    widget.table_view.selectRow(0)
    widget.refresh()
    assert widget.selected_meeting_id() is None
    assert not widget.open_meeting_button.isEnabled()
    assert widget.state_label.text() == _("No matching meetings.")


def test_failure_retains_filters_rows_and_selection_then_recovers(qtbot) -> None:
    selected = make_entry(2, session_state=MeetingSessionState.FAILED)
    entries = (make_entry(1), selected)
    service = FakeService(
        [entries, MeetingLibraryDatabaseError("unavailable"), (selected,)]
    )
    widget = make_widget(qtbot, service)
    widget.refresh()
    widget.source_filter.setCurrentIndex(
        widget.source_filter.findData(MeetingRemoteSourceKind.SYSTEM)
    )
    widget.state_filter.setCurrentIndex(
        widget.state_filter.findData(MeetingSessionState.FAILED)
    )
    widget.table_view.selectRow(0)
    widget.refresh_button.click()
    assert visible_entries(widget) == (selected,)
    assert widget.selected_meeting_id() == selected.session_id
    assert widget.open_meeting_button.isEnabled()
    assert widget.source_filter.currentData() == MeetingRemoteSourceKind.SYSTEM
    assert widget.state_filter.currentData() == MeetingSessionState.FAILED
    assert widget.state_label.text() == _("Could not load meetings.")
    widget.reset_filters_button.click()
    assert visible_entries(widget) == entries
    assert widget.state_label.text() == _("Could not load meetings.")
    assert service.calls == 2
    widget.state_filter.setCurrentIndex(
        widget.state_filter.findData(MeetingSessionState.FAILED)
    )
    widget.refresh_button.click()
    assert service.calls == 3
    assert visible_entries(widget) == (selected,)
    assert widget.selected_meeting_id() == selected.session_id
    assert widget.state_filter.currentData() == MeetingSessionState.FAILED
    assert widget.state_label.text() == ""
    assert widget.state_label.isHidden()
