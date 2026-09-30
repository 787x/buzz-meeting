"""Recorder presentation with real lifecycle/storage and controlled audio only."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from buzz.audio_capture.source import AudioSourceError
from buzz.audio_capture.windows_application_targets import WindowsApplicationAudioTarget
from buzz.db.meeting_library_repository import QSqlMeetingLibraryRepository
from buzz.db.meeting_storage_repository import QSqlMeetingRepository
from buzz.meeting.meeting_audio_tracks import MeetingAudioTracksOutcome
from buzz.meeting.meeting_library import MeetingLibraryService
from buzz.meeting.meeting_session import MeetingRemoteSourceKind
from buzz.meeting.meeting_storage import MeetingStorage, MeetingStorageDatabaseError
from buzz.meeting.meeting_workflow import MeetingWorkflow, MeetingWorkflowState
from buzz.widgets.meeting_capture_widget import MeetingCaptureWidget
from buzz.widgets.meeting_mode import MeetingModeController
from tests.meeting.meeting_workflow_test import ControlledAudioSource


PREFIX = "buzz.widgets.meeting_capture_widget."


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


@pytest.fixture
def recorder(db, tmp_path, qtbot, monkeypatch):
    monkeypatch.setattr(
        "buzz.widgets.audio_devices_combo_box.AudioDevicesComboBox.get_audio_devices",
        lambda _: [(7, "Test microphone"), (8, "Second microphone")],
    )
    monkeypatch.setattr(
        "buzz.widgets.audio_devices_combo_box.AudioDevicesComboBox.get_default_device_id",
        lambda _: 7,
    )
    target = WindowsApplicationAudioTarget(
        1, "Test meeting", 2, 3, "meeting.exe", None, None
    )
    discover = Mock(return_value=[target])
    validate = Mock(return_value=True)
    monkeypatch.setattr(PREFIX + "list_windows_application_audio_targets", discover)
    monkeypatch.setattr(PREFIX + "validate_windows_application_audio_target", validate)
    mic, remote = ControlledAudioSource(), ControlledAudioSource()
    microphone_factory = Mock(return_value=mic)
    system_factory = Mock(return_value=remote)
    application_factory = Mock(return_value=remote)
    monkeypatch.setattr(PREFIX + "SoundDeviceAudioSource", microphone_factory)
    monkeypatch.setattr(PREFIX + "WindowsSystemAudioSource", system_factory)
    monkeypatch.setattr(PREFIX + "WindowsProcessAudioSource", application_factory)
    clock = [100.0]
    monkeypatch.setattr(PREFIX + "time", SimpleNamespace(monotonic=lambda: clock[0]))
    storage = MeetingStorage(QSqlMeetingRepository(db), root=tmp_path / "meetings")
    save = storage.save
    controller = MeetingModeController(MeetingWorkflow(storage))
    widget = MeetingCaptureWidget(controller)
    qtbot.addWidget(widget)
    widget.show()
    yield SimpleNamespace(
        widget=widget,
        controller=controller,
        storage=storage,
        mic=mic,
        remote=remote,
        target=target,
        discover=discover,
        validate=validate,
        microphone_factory=microphone_factory,
        system_factory=system_factory,
        application_factory=application_factory,
        clock=clock,
    )
    # Release any controlled blockers even when a presentation assertion fails.
    storage.save = save
    for source in (mic, remote):
        source.allow_start.set()
        source.allow_stop.set()
        source.stop_errors = []
    if controller.active:
        controller.end()
        qtbot.waitUntil(lambda: not controller.active, timeout=10000)
    controller._poll.stop()
    widget._timer.stop()


def start(recorder, qtbot):
    recorder.widget.start_button.click()
    qtbot.waitUntil(lambda: recorder.widget.status_label.text() == "Recording")
    recorder.mic.deliver(np.full(3200, 0.1, dtype=np.float32))
    recorder.remote.deliver(np.full(3200, 0.2, dtype=np.float32))


def assert_locked(widget):
    for control in (
        widget.microphone,
        widget.remote,
        widget.target,
        widget.refresh_targets,
        widget.model_size,
        widget.start_button,
    ):
        assert not control.isEnabled()


def test_initial_setup_and_system_application_visibility(recorder):
    w = recorder.widget
    assert w.windowTitle() == "New Meeting"
    assert w.setup_group.title() == "Recording Setup"
    assert w.recording_group.title() == "Recording"
    assert w.setup_group.geometry().bottom() < w.recording_group.geometry().top()
    assert w.elapsed_label.text() == "00:00:00"
    assert w.elapsed_label.font().pointSize() > w.status_label.font().pointSize()
    assert w.status_label.text() == "Ready"
    assert w.start_button.text() == "Start Meeting"
    assert w.start_button.isEnabled()
    assert w.stop_button.text() == "End && Save"
    assert not w.stop_button.isEnabled()
    assert not w.open_button.isEnabled() and not w.open_button.isVisible()
    assert not w.error_label.isVisible()
    assert w.model_size.currentText() == "SMALL"
    assert "after" in w.model_help.text()
    assert "remains saved" in w.model_help.text()
    assert not w.application_label.isVisible()
    assert not w.target.isVisible() and not w.refresh_targets.isVisible()
    recorder.discover.assert_not_called()

    w.remote.setCurrentIndex(1)
    assert w.application_label.isVisible()
    assert w.target.isVisible() and w.target.isEnabled()
    assert w.refresh_targets.isVisible() and w.refresh_targets.isEnabled()
    recorder.discover.assert_called_once_with()
    w.target.setCurrentIndex(1)
    w._render()
    recorder.discover.assert_called_once()
    w.remote.setCurrentIndex(0)
    assert not w.target.isVisible() and not w.refresh_targets.isVisible()
    assert w.target.selected_target is recorder.target
    recorder.discover.assert_called_once()
    assert recorder.controller.workflow.state is MeetingWorkflowState.IDLE


def test_missing_microphone_cannot_start(recorder):
    w = recorder.widget
    w.microphone.setCurrentIndex(-1)
    w.start_button.click()
    assert w.error_label.isVisible()
    assert w.error_label.text() == "Choose an available microphone"
    assert w.status_label.text() == "Ready"
    recorder.microphone_factory.assert_not_called()
    recorder.system_factory.assert_not_called()
    assert not recorder.controller.active


def test_model_help_remains_readable_in_compact_application_setup(recorder):
    w = recorder.widget
    w.resize(w.minimumWidth(), w.minimumHeight())
    w.remote.setCurrentIndex(1)
    QApplication.processEvents()
    assert w.model_help.isVisible()
    assert w.model_help.height() >= w.model_help.heightForWidth(w.model_help.width())


@pytest.mark.parametrize("kind", list(MeetingRemoteSourceKind))
def test_start_uses_selected_configuration_and_locks_in_flight(recorder, qtbot, kind):
    w, c = recorder.widget, recorder.controller
    w.microphone.setCurrentIndex(1)
    w.model_size.setCurrentText("BASE")
    w.remote.setCurrentIndex(w.remote.findData(kind))
    if kind is MeetingRemoteSourceKind.APPLICATION:
        w.target.setCurrentIndex(1)
    selection = (
        w.microphone.currentData(),
        w.microphone.currentIndex(),
        w.remote.currentData(),
        w.target.selected_target,
        w.model_size.currentText(),
    )
    recorder.mic.pause_start = True
    w.start_button.click()
    qtbot.waitUntil(recorder.mic.start_entered.is_set)
    assert w.status_label.text() == "Starting…"
    assert w.elapsed_label.text() == "00:00:00"
    assert_locked(w)
    assert not w.stop_button.isEnabled()
    w.start_button.click()
    w._start()
    w.stop_button.click()
    recorder.microphone_factory.assert_called_once_with(8, 16000)
    assert not recorder.mic.stop_entered.is_set()
    if kind is MeetingRemoteSourceKind.SYSTEM:
        recorder.system_factory.assert_called_once_with()
        recorder.application_factory.assert_not_called()
    else:
        recorder.validate.assert_called_once_with(recorder.target)
        recorder.application_factory.assert_called_once_with(process_id=3)
        recorder.system_factory.assert_not_called()
    assert w.config.whisper_model_size == "BASE"
    recorder.mic.allow_start.set()
    qtbot.waitUntil(lambda: w.status_label.text() == "Recording")
    assert c.workflow.snapshot().remote_source_kind is kind
    assert_locked(w)
    assert w.stop_button.isEnabled() and w.stop_button.text() == "End && Save"
    with qtbot.waitSignal(c.saved):
        w.stop_button.click()
    assert selection == (
        w.microphone.currentData(),
        w.microphone.currentIndex(),
        w.remote.currentData(),
        w.target.selected_target,
        w.model_size.currentText(),
    )
    assert w.microphone.isEnabled() and w.remote.isEnabled()
    assert w.model_size.isEnabled() and w.start_button.isEnabled()


def test_elapsed_progress_final_duration_and_second_attempt_reset(recorder, qtbot):
    w, c = recorder.widget, recorder.controller
    start(recorder, qtbot)
    assert w.elapsed_label.text() == "00:00:00"
    recorder.clock[0] += 65
    w._timer.timeout.emit()
    assert w.elapsed_label.text() == "00:01:05"
    with qtbot.waitSignal(c.saved):
        w.stop_button.click()
    expected = int(c.duration_seconds)
    assert w.elapsed_label.text() == (
        f"{expected // 3600:02}:{expected // 60 % 60:02}:{expected % 60:02}"
    )
    assert c.duration_seconds == recorder.storage.load(c.saved_id).duration_ns / 1e9
    first_id = c.saved_id
    # Make stale duration easy to detect without waiting for a long real meeting.
    c.duration_seconds = 3661
    c.changed.emit()
    assert w.elapsed_label.text() == "01:01:01"
    assert w.open_button.isVisible() and w.open_button.isEnabled()
    recorder.mic.pause_start = True
    recorder.mic.allow_start.clear()
    recorder.mic.start_entered.clear()
    w.start_button.click()
    qtbot.waitUntil(recorder.mic.start_entered.is_set)
    assert w.elapsed_label.text() == "00:00:00"
    assert c.saved_id is None
    assert not w.open_button.isVisible() and not w.open_button.isEnabled()
    recorder.mic.allow_start.set()
    qtbot.waitUntil(lambda: w.status_label.text() == "Recording")
    recorder.clock[0] += 2
    w._timer.timeout.emit()
    assert w.elapsed_label.text() == "00:00:02"
    with qtbot.waitSignal(c.saved):
        w.stop_button.click()
    assert c.saved_id != first_id
    with qtbot.waitSignal(w.open_requested) as opened:
        w.open_button.click()
    assert opened.args == [c.saved_id]


def test_stopping_disables_duplicate_operations_and_saves_once(recorder, qtbot):
    w, c = recorder.widget, recorder.controller
    start(recorder, qtbot)
    # The existing signal already owns the bound end slot; observe its operation.
    stop = Mock(wraps=c.workflow.stop_capture)
    c.workflow.stop_capture = stop
    save = Mock(wraps=recorder.storage.save)
    recorder.storage.save = save
    recorder.mic.pause_stop = True
    w.stop_button.click()
    qtbot.waitUntil(recorder.mic.stop_entered.is_set)
    assert w.status_label.text() == "Stopping and saving…"
    assert_locked(w)
    assert not w.stop_button.isEnabled()
    w.stop_button.click()
    w.start_button.click()
    stop.assert_called_once_with()
    save.assert_not_called()
    recorder.mic.allow_stop.set()
    qtbot.waitUntil(lambda: not c.active)
    save.assert_called_once()
    assert w.status_label.text() == "Meeting saved"
    assert w.open_button.isVisible() and w.open_button.isEnabled()
    with qtbot.waitSignal(w.open_requested) as opened:
        w.open_button.click()
    assert opened.args == [c.saved_id]


def test_degraded_recording_and_partial_save_preserve_microphone(recorder, qtbot):
    w, c = recorder.widget, recorder.controller
    start(recorder, qtbot)
    recorder.remote.fail(AudioSourceError("remote lost"))
    qtbot.waitUntil(lambda: w.status_label.text() == "Recording — degraded audio")
    assert_locked(w)
    assert w.stop_button.isEnabled() and w.stop_button.text() == "End && Save"
    recorder.mic.deliver(np.full(1600, 0.1, dtype=np.float32))
    with qtbot.waitSignal(c.saved):
        w.stop_button.click()
    assert w.status_label.text() == "Meeting saved — partial audio"
    stored = recorder.storage.load(c.saved_id)
    assert stored.audio_outcome is MeetingAudioTracksOutcome.PARTIAL
    assert stored.microphone.sample_count == 4800
    assert w.open_button.isEnabled()


def test_save_failure_retry_keeps_identity_and_stores_once(recorder, qtbot, db):
    w, c = recorder.widget, recorder.controller
    start(recorder, qtbot)
    identity = c.workflow.session_id
    original_save = recorder.storage.save
    save = Mock(side_effect=MeetingStorageDatabaseError("Disk unavailable <details>"))
    recorder.storage.save = save
    w.stop_button.click()
    qtbot.waitUntil(lambda: w.status_label.text() == "Save failed — retry required")
    assert c.workflow.state is MeetingWorkflowState.AWAITING_PERSISTENCE
    assert_locked(w)
    assert w.stop_button.isEnabled() and w.stop_button.text() == "Retry Saving"
    assert w.error_label.isVisible()
    assert "Disk unavailable <details>" in w.error_label.text()
    assert not w.open_button.isEnabled()
    w._timer.timeout.emit()
    save.assert_called_once()
    assert c.workflow.session_id == identity
    assert recorder.storage.load(identity) is None
    save.side_effect = original_save
    with qtbot.waitSignal(c.saved):
        w.stop_button.click()
    assert c.saved_id == identity
    assert w.status_label.text() == "Meeting saved"
    assert w.open_button.isEnabled()
    library = MeetingLibraryService(QSqlMeetingLibraryRepository(db)).list_meetings()
    assert [meeting.session_id for meeting in library] == [identity]
    assert recorder.mic.stop_count == recorder.remote.stop_count == 1


def test_saved_cleanup_required_blocks_start_and_requires_explicit_retry(
    recorder, qtbot
):
    w, c = recorder.widget, recorder.controller
    start(recorder, qtbot)
    recorder.mic.stop_errors = [RuntimeError("stop failed")]
    w.stop_button.click()
    qtbot.waitUntil(lambda: c.saved_id is not None)
    identity = c.saved_id
    assert c.workflow.state is MeetingWorkflowState.CLEANUP_REQUIRED
    assert w.status_label.text() == "Recording preserved — cleanup still required"
    assert_locked(w)
    assert w.stop_button.isEnabled()
    assert w.stop_button.text() == "Retry Ending Meeting"
    assert w.error_label.isVisible()
    assert w.open_button.isVisible() and w.open_button.isEnabled()
    w._timer.timeout.emit()
    assert c.workflow.state is MeetingWorkflowState.CLEANUP_REQUIRED
    with qtbot.waitSignal(w.open_requested) as opened:
        w.open_button.click()
    assert opened.args == [identity]
    recorder.mic.pause_stop = True
    recorder.mic.stop_entered.clear()
    w.stop_button.click()
    qtbot.waitUntil(recorder.mic.stop_entered.is_set)
    assert w.status_label.text() == "Retrying cleanup…"
    assert not w.stop_button.isEnabled() and not w.start_button.isEnabled()
    recorder.mic.allow_stop.set()
    qtbot.waitUntil(lambda: not c.active)
    assert c.saved_id == identity
    assert w.start_button.isEnabled()


def test_failed_capture_saved_outcome_is_visible(recorder, qtbot):
    w, c = recorder.widget, recorder.controller
    recorder.mic.start_error = RuntimeError("microphone unavailable")
    with qtbot.waitSignal(c.saved):
        w.start_button.click()
    assert w.status_label.text() == "Meeting saved — audio failed"
    assert w.error_label.isVisible()
    assert w.elapsed_label.text() == "00:00:00"
    assert w.open_button.isEnabled()
    assert (
        recorder.storage.load(c.saved_id).audio_outcome
        is MeetingAudioTracksOutcome.FAILED
    )


def test_application_refresh_failure_and_recovery_do_not_change_workflow(recorder):
    w, c = recorder.widget, recorder.controller
    recorder.discover.side_effect = RuntimeError("Application list unavailable")
    w.remote.setCurrentIndex(1)
    assert w.remote.currentData() is MeetingRemoteSourceKind.APPLICATION
    assert w.target.selected_target is None
    assert w.error_label.isVisible()
    assert w.error_label.text() == "Application list unavailable"
    assert w.refresh_targets.isVisible() and w.refresh_targets.isEnabled()
    w.start_button.click()
    assert "Refresh applications" in w.error_label.text()
    assert c.workflow.state is MeetingWorkflowState.IDLE
    recorder.system_factory.assert_not_called()
    recorder.application_factory.assert_not_called()
    recorder.discover.side_effect = None
    w.refresh_targets.click()
    assert not w.error_label.isVisible()
    assert w.target.selected_target is None
    w.target.setCurrentIndex(1)
    assert w.target.selected_target is recorder.target
    assert w.remote.currentData() is MeetingRemoteSourceKind.APPLICATION
    assert c.workflow.state is MeetingWorkflowState.IDLE


def test_stale_application_target_is_validated_before_start(recorder):
    w = recorder.widget
    w.remote.setCurrentIndex(1)
    w.target.setCurrentIndex(1)
    recorder.validate.return_value = False
    w.start_button.click()
    assert w.error_label.isVisible()
    assert "choose an available application" in w.error_label.text()
    recorder.validate.assert_called_once_with(recorder.target)
    recorder.application_factory.assert_not_called()
    assert not recorder.controller.active
