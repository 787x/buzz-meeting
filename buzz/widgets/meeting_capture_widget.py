"""Small meeting capture window; lifecycle belongs to MeetingWorkflow."""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from buzz.audio_capture.sounddevice_source import SoundDeviceAudioSource
from buzz.audio_capture.windows_application_targets import (
    list_windows_application_audio_targets,
    validate_windows_application_audio_target,
)
from buzz.audio_capture.windows_process_source import WindowsProcessAudioSource
from buzz.audio_capture.windows_system_source import WindowsSystemAudioSource
from buzz.meeting.final_transcription import FinalTranscriptionConfig
from buzz.meeting.meeting_session import MeetingRemoteSourceKind
from buzz.meeting.meeting_workflow import MeetingWorkflowState
from buzz.widgets.application_audio_target_combo_box import (
    ApplicationAudioTargetComboBox,
)
from buzz.widgets.audio_devices_combo_box import AudioDevicesComboBox


class MeetingCaptureWidget(QWidget):
    open_requested = pyqtSignal(object)

    def __init__(self, controller, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.controller = controller
        self.config = FinalTranscriptionConfig(whisper_model_size="SMALL")
        self._close_pending = False
        self._started_at = None
        self.setWindowTitle("New Meeting")
        self.microphone = AudioDevicesComboBox(self)
        self.remote = QComboBox(self)
        self.remote.addItem("System Audio", MeetingRemoteSourceKind.SYSTEM)
        self.remote.addItem("Application Audio", MeetingRemoteSourceKind.APPLICATION)
        self.target = ApplicationAudioTargetComboBox(self)
        self.refresh_targets = QPushButton("Refresh applications", self)
        self.model_size = QComboBox(self)
        self.model_size.addItems(["TINY", "BASE", "SMALL", "MEDIUM", "LARGEV3"])
        self.model_size.setCurrentText("SMALL")
        self.model_help = QLabel(
            "Final transcription runs after the recording is saved. "
            "If the model is unavailable, the recording remains saved "
            "and transcription can be retried later.",
            self,
        )
        self.model_help.setWordWrap(True)
        self.model_help.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.start_button = QPushButton("Start Meeting", self)
        self.stop_button = QPushButton("End && Save", self)
        self.open_button = QPushButton("Open Meeting", self)
        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        status_font = self.status_label.font()
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.error_label = QLabel(self)
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.elapsed_label = QLabel("00:00:00", self)
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed_label.setAccessibleName("Elapsed recording time")
        elapsed_font = self.elapsed_label.font()
        elapsed_font.setPointSize(32)
        self.elapsed_label.setFont(elapsed_font)

        layout = QVBoxLayout(self)
        self.setup_group = QGroupBox("Recording Setup", self)
        setup_layout = QFormLayout(self.setup_group)
        setup_layout.addRow("Microphone", self.microphone)
        setup_layout.addRow("Meeting audio", self.remote)
        self.application_label = QLabel("Application", self)
        self.application_controls = QWidget(self)
        application_layout = QHBoxLayout(self.application_controls)
        application_layout.setContentsMargins(0, 0, 0, 0)
        application_layout.addWidget(self.target, 1)
        application_layout.addWidget(self.refresh_targets)
        setup_layout.addRow(self.application_label, self.application_controls)
        final_group = QGroupBox("Final transcription", self)
        final_layout = QFormLayout(final_group)
        final_layout.addRow("Model", self.model_size)
        final_layout.addRow(self.model_help)
        setup_layout.addRow(final_group)
        layout.addWidget(self.setup_group)

        self.recording_group = QGroupBox("Recording", self)
        recording_layout = QVBoxLayout(self.recording_group)
        recording_layout.addWidget(self.elapsed_label)
        recording_layout.addWidget(self.status_label)
        recording_layout.addWidget(self.error_label)
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        recording_layout.addLayout(actions)
        recording_layout.addWidget(self.open_button)
        layout.addWidget(self.recording_group)
        self.setMinimumWidth(560)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(controller.end)
        self.open_button.clicked.connect(
            lambda: self.open_requested.emit(controller.saved_id)
        )
        self.refresh_targets.clicked.connect(self._refresh_targets)
        self.remote.currentIndexChanged.connect(self._source_changed)
        controller.changed.connect(self._render)
        controller.released.connect(self._released)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._render)
        self._timer.start()
        self._render()

    def _source_changed(self):
        if self.remote.currentData() is MeetingRemoteSourceKind.APPLICATION:
            self._refresh_targets()
        self._render()

    def _refresh_targets(self):
        try:
            self.target.set_targets(list_windows_application_audio_targets())
            self.controller.error = ""
        except Exception as exc:
            self.target.set_refresh_error()
            self.controller.error = str(exc)
        self._render()

    def _start(self):
        if self.controller.active:
            return
        try:
            index = self.microphone.currentIndex()
            if index < 0:
                raise ValueError("Choose an available microphone")
            microphone = SoundDeviceAudioSource(
                self.microphone.audio_devices[index][0], 16000
            )
            kind = self.remote.currentData()
            if kind is MeetingRemoteSourceKind.SYSTEM:
                remote = WindowsSystemAudioSource()
            else:
                target = self.target.selected_target
                if target is None or not validate_windows_application_audio_target(
                    target
                ):
                    raise ValueError(
                        "Refresh applications and choose an available application"
                    )
                remote = WindowsProcessAudioSource(process_id=target.capture_pid)
            self.config = FinalTranscriptionConfig(
                whisper_model_size=self.model_size.currentText()
            )
            self.controller.start(microphone, remote, kind)
        except Exception as exc:
            self.controller.error = str(exc)
            self._render()

    def _render(self):
        active = self.controller.active
        for widget in (self.microphone, self.remote, self.model_size):
            widget.setEnabled(not active)
        application = self.remote.currentData() is MeetingRemoteSourceKind.APPLICATION
        self.application_label.setVisible(application)
        self.application_controls.setVisible(application)
        self.target.setEnabled(not active and application)
        self.refresh_targets.setEnabled(not active and application)
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active and self.controller.worker is None)
        self.stop_button.setText(
            {
                MeetingWorkflowState.AWAITING_PERSISTENCE: "Retry Saving",
                MeetingWorkflowState.CLEANUP_REQUIRED: "Retry Ending Meeting",
            }.get(self.controller.workflow.state, "End && Save")
        )
        self.open_button.setEnabled(self.controller.saved_id is not None)
        self.open_button.setVisible(self.controller.saved_id is not None)
        self.status_label.setText(self.controller.status)
        self.error_label.setText(self.controller.error)
        self.error_label.setVisible(bool(self.controller.error))
        # Starting is emitted synchronously by the controller, including when
        # a reused window starts again. Never carry the previous attempt's clock.
        if self.controller.status == "Starting…":
            self._started_at = None
        if self.controller.status.startswith("Recording") and self._started_at is None:
            self._started_at = time.monotonic()
        elapsed = 0
        if self.controller.duration_seconds is not None:
            elapsed = int(self.controller.duration_seconds)
        elif active and self._started_at is not None:
            elapsed = int(time.monotonic() - self._started_at)
        self.elapsed_label.setText(
            f"{elapsed // 3600:02}:{elapsed // 60 % 60:02}:{elapsed % 60:02}"
        )

    def confirm_end(self):
        return (
            QMessageBox.question(
                self,
                "End meeting?",
                "End and save this meeting before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _released(self):
        if self._close_pending:
            self._close_pending = False
            self.close()

    def closeEvent(self, event):
        if self.controller.active:
            event.ignore()
            if self.confirm_end():
                self._close_pending = True
                self.controller.end()
            else:
                self._close_pending = False
            return
        self._close_pending = False
        super().closeEvent(event)
