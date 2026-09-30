"""Product regression: real capture ownership, QSql, orchestration and exports.

Only audio hardware, ASR execution, AI execution and dialogs are substituted.
The legacy (non-meeting) transcription service is unused in these scenarios.
"""

from dataclasses import replace
from threading import Event, get_ident
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from PyQt6.QtCore import Qt
from PyQt6.QtSql import QSqlQuery
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from buzz.audio_capture.source import AudioSourceError
from buzz.locale import _ as translate
from buzz.db.meeting_library_repository import QSqlMeetingLibraryRepository
from buzz.db.meeting_speaker_repository import QSqlMeetingSpeakerRepository
from buzz.db.meeting_storage_repository import QSqlMeetingRepository
from buzz.db.meeting_summary_repository import QSqlMeetingSummaryRepository
from buzz.db.meeting_transcription_repository import QSqlMeetingTranscriptionRepository
from buzz.db.service.transcription_service import TranscriptionService
from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionReadService,
    FinalTranscriptionStatus,
    TrackTranscriptionInputSegment,
    TrackTranscriptionInputWord,
    TrackTranscriptionResult,
)
from buzz.meeting.meeting_audio_tracks import MeetingAudioTracksOutcome
from buzz.meeting.meeting_detail import (
    MeetingDetailService,
    MeetingDetailSpeakerReviewState,
)
from buzz.meeting.meeting_library import MeetingLibraryService
from buzz.meeting.meeting_notes import MeetingNotesService, NotesError
from buzz.meeting.meeting_session import MeetingSessionState
from buzz.meeting.meeting_storage import MeetingStorage
from buzz.meeting.meeting_summary import (
    MeetingSummaryFreshness,
    meeting_summary_to_json,
)
from buzz.meeting.meeting_workflow import MeetingWorkflow
from buzz.meeting.speaker_review import MeetingSpeakerReviewService
from buzz.meeting.speaker_diarization import SpeakerDiarizationService
from buzz.meeting.speaker_mapping import SpeakerAttributionStatus, map_words_to_speakers
from buzz.widgets.main_window import MainWindow
from buzz.widgets.meeting_final_transcription import MeetingFinalTranscription
from buzz.widgets.meeting_mode import MeetingModeController
from buzz.widgets.meeting_notes_controller import MeetingNotesController
from buzz.widgets.meeting_notes_panel import ManualResponseDialog
from buzz.widgets.meeting_speaker_generation import MeetingSpeakerGeneration
from tests.meeting.meeting_workflow_test import ControlledAudioSource
from tests.widgets.meeting_mode_test import Adapter
from tests.widgets.meeting_notes_test import ControlledProvider, config
from tests.widgets.meeting_speaker_generation_test import ControlledRunner


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


@pytest.fixture
def product(db, tmp_path, qtbot, monkeypatch):
    storage = MeetingStorage(QSqlMeetingRepository(db), root=tmp_path / "meetings")
    repository = QSqlMeetingTranscriptionRepository(db)
    reader = FinalTranscriptionReadService(repository)
    reviews = MeetingSpeakerReviewService(QSqlMeetingSpeakerRepository(db), reader)
    detail = MeetingDetailService(storage, reader, reviews)
    speaker_runner = ControlledRunner()
    speakers = MeetingSpeakerGeneration(
        detail,
        reader,
        reviews,
        service_factory=lambda _: SpeakerDiarizationService(speaker_runner),
    )
    summaries = QSqlMeetingSummaryRepository(db)
    provider = ControlledProvider()
    requests = []
    summarize = provider.summarize

    def record_request(request):
        requests.append(request)
        return summarize(request)

    provider.summarize = record_request
    notes = MeetingNotesController(
        MeetingNotesService(detail, summaries), provider_factory=lambda _: provider
    )
    capture = MeetingModeController(MeetingWorkflow(storage))
    adapter = Adapter()
    final = MeetingFinalTranscription(storage, repository, adapter)
    mic, remote = ControlledAudioSource(), ControlledAudioSource()
    monkeypatch.setattr(
        "buzz.widgets.audio_devices_combo_box.AudioDevicesComboBox.get_audio_devices",
        lambda _: [(7, "Synthetic microphone")],
    )
    monkeypatch.setattr(
        "buzz.widgets.audio_devices_combo_box.AudioDevicesComboBox.get_default_device_id",
        lambda _: 7,
    )
    monkeypatch.setattr(
        "buzz.widgets.meeting_capture_widget.SoundDeviceAudioSource", lambda *a: mic
    )
    monkeypatch.setattr(
        "buzz.widgets.meeting_capture_widget.WindowsSystemAudioSource", lambda: remote
    )
    monkeypatch.setattr(
        "buzz.widgets.main_window.PluginManager.initialize", lambda _: None
    )
    window = MainWindow(
        Mock(spec=TranscriptionService),
        MeetingLibraryService(QSqlMeetingLibraryRepository(db)),
        detail,
        reviews,
        Mock(),
        meeting_controller=capture,
        meeting_final=final,
        meeting_notes=notes,
        meeting_speakers=speakers,
    )
    window.new_meeting_action.trigger()
    assert window.meeting_capture_widget.windowTitle() == "New Meeting"
    qtbot.addWidget(window)
    result = SimpleNamespace(
        storage=storage,
        reader=reader,
        summaries=summaries,
        detail=detail,
        notes=notes,
        capture=capture,
        adapter=adapter,
        mic=mic,
        remote=remote,
        final=final,
        provider=provider,
        requests=requests,
        window=window,
        db=db,
        root=tmp_path,
        speakers=speakers,
        speaker_runner=speaker_runner,
        reviews=reviews,
        repository=repository,
    )
    yield result
    speaker_runner.release.set()
    qtbot.waitUntil(lambda: not speakers.busy, timeout=15000)
    provider.allow_result.set()
    provider.allow_cleanup.set()
    qtbot.waitUntil(lambda: not notes.busy, timeout=15000)
    if capture.active:
        capture.end()
        qtbot.waitUntil(lambda: not capture.active)
    # Release controlled external work even when a mutation trips an oracle
    # before the scenario reaches its normal completion step.
    while final.pending:
        qtbot.waitUntil(lambda: final._asr_call is not None or not final.pending)
        if final._asr_call is not None:
            adapter.track_error.emit("fixture cleanup")
    final.close()
    window.meeting_capture_widget.close()
    window.close()


def reviewable_final(product, qtbot):
    """Prepare v2 through the existing final-transcription pipeline and adapter."""
    p = product
    identity, stored = record(p, qtbot)
    transcribe(p, qtbot)
    p.final.request(
        identity,
        FinalTranscriptionConfig(profile_version=2, whisper_model_size="SMALL"),
    )
    for index in (3, 4):
        qtbot.waitUntil(lambda: len(p.adapter.calls) == index)
        path = Path(p.adapter.calls[-1][0])
        assert path in (stored.microphone.path, stored.remote.path)
        p.adapter.track_rich_completed.emit(
            TrackTranscriptionResult(
                segments=(
                    TrackTranscriptionInputSegment(10, 100, f"From {path.stem}"),
                ),
                words=(
                    TrackTranscriptionInputWord(0, 10, 40, "From"),
                    TrackTranscriptionInputWord(0, 40, 100, path.stem),
                ),
            )
        )
    qtbot.waitUntil(lambda: not p.final.pending)
    source = authoritative(p, identity)
    assert source.final_generation.profile_version == 2
    assert source.speaker_review_state is MeetingDetailSpeakerReviewState.ABSENT
    assert not p.speaker_runner.calls, "opening/final completion started diarization"
    assert p.window.meeting_detail_widget.generate_review_button.isEnabled()
    return identity, stored, source


def test_product_explicit_speaker_generation_persisted_reload(
    product, qtbot, monkeypatch
):
    p = product
    identity, stored, source = reviewable_final(p, qtbot)
    audio_before = {
        t.path: t.path.read_bytes() for t in (stored.microphone, stored.remote)
    }
    words_before = p.reader.load_words(source.final_generation.generation_id)
    owner = get_ident()
    calls = []
    create = p.reviews.create_review

    def save(*args):
        assert get_ident() == owner and p.db.isOpen()
        calls.append(args)
        return create(*args)

    monkeypatch.setattr(p.reviews, "create_review", save)
    widget = p.window.meeting_detail_widget
    widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    assert p.speaker_runner.calls[0][1] != owner
    assert not widget.generate_review_button.isEnabled()
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert len(calls) == 1
    assert calls[0][0] == source.final_generation.generation_id
    # Fresh repositories/services must recover the same canonical review.
    reader = FinalTranscriptionReadService(QSqlMeetingTranscriptionRepository(p.db))
    reviews = MeetingSpeakerReviewService(QSqlMeetingSpeakerRepository(p.db), reader)
    detail = MeetingDetailService(
        MeetingStorage(QSqlMeetingRepository(p.db), root=p.root / "meetings"),
        reader,
        reviews,
    )
    reloaded = detail.load(identity)
    assert reloaded.speaker_review_state is MeetingDetailSpeakerReviewState.FRESH
    assert (
        reloaded.speaker_review.source_generation_id
        == source.final_generation.generation_id
    )
    assert tuple(w.word for w in reloaded.speaker_review.words) == words_before
    assert all(
        w.machine_status is SpeakerAttributionStatus.ASSIGNED
        for w in reloaded.speaker_review.words
    )
    assert widget._snapshot.speaker_review == reloaded.speaker_review
    assert widget.word_model.rowCount() == len(words_before)
    assert widget.review_state_label.text() == "Available"
    assert not widget.generate_review_button.isEnabled()
    assert p.storage.load(identity) == stored
    assert p.reader.load_words(source.final_generation.generation_id) == words_before
    assert all(path.read_bytes() == original for path, original in audio_before.items())
    np.testing.assert_allclose(p.speaker_runner.calls[0][0].waveform, 0.1, atol=0.0001)
    np.testing.assert_allclose(p.speaker_runner.calls[1][0].waveform, 0.2, atol=0.0001)


def test_product_workspace_transcript_speakers_manual_notes_and_minutes(
    product, qtbot, monkeypatch
):
    p = product
    identity, stored, source = reviewable_final(p, qtbot)
    widget = p.window.meeting_detail_widget
    assert [widget.tabs.tabText(i) for i in range(widget.tabs.count())] == [
        translate(name) for name in ("Transcript", "Speakers", "AI Notes", "Info")
    ]
    assert widget.tabs.currentIndex() == 0
    assert widget.transcript_edit.toPlainText() == "\n\n".join(
        segment.text for segment in source.transcript.segments
    )
    words = p.reader.load_words(source.final_generation.generation_id)
    audio = {t.path: t.path.read_bytes() for t in (stored.microphone, stored.remote)}
    for area in (1, 2, 3, 0, 1):
        widget.tabs.setCurrentIndex(area)
    assert not p.speaker_runner.calls and not p.requests
    assert not p.notes.manual_contexts
    assert widget.audio_table.rowCount() == 2
    widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    widget.tabs.setCurrentIndex(3)
    widget.tabs.setCurrentIndex(1)
    assert "Generating" in widget.generation_status_label.text()
    assert not widget.generate_review_button.isEnabled()
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert widget.tabs.currentIndex() == 1
    widget.name_edit.setText("Meeting participant")
    widget.save_name_button.click()
    # Notes use names from explicit word assignments in the existing workflow.
    for row, word in enumerate(widget._snapshot.speaker_review.words):
        if word.word.source_role is stored.microphone.role:
            widget.word_table.selectRow(row)
            widget.assign_speaker_combo.setCurrentIndex(0)
            widget.assign_button.click()
    widget.complete_button.click()
    persisted_review = p.detail.load(identity).speaker_review
    assert persisted_review.speakers[0].display_name == "Meeting participant"
    assert not widget.complete_button.isEnabled()
    assert widget.tabs.currentIndex() == 1

    widget.tabs.setCurrentIndex(2)
    panel = widget.notes_panel
    copied = []
    monkeypatch.setattr(panel, "_copy_to_clipboard", copied.append)
    panel.actions["Copy AI Request"].click()
    assert len(copied) == 1 and "Meeting participant" in copied[0]
    assert not p.requests and not p.summaries.list_for_meeting(identity)

    def import_manual(dialog):
        dialog.input.setPlainText(meeting_summary_to_json(p.provider.result))
        dialog.strict.click()
        return dialog.result()

    monkeypatch.setattr(ManualResponseDialog, "exec", import_manual)
    panel.actions["Import AI Response"].click()
    (artifact,) = p.summaries.list_for_meeting(identity)
    assert artifact.source_review_id == persisted_review.id
    assert artifact.source_review_revision == persisted_review.revision
    assert panel.selected_id == artifact.summary_id
    widget.refresh()
    assert widget.tabs.currentWidget() is panel
    assert panel.selected_id == artifact.summary_id
    destination = p.root / "workspace-minutes.md"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(destination), "Markdown (*.md)"),
    )
    panel.actions["Export Minutes"].click()
    assert artifact.summary.summary in destination.read_text(encoding="utf-8")
    assert p.storage.load(identity) == stored
    assert p.detail.load(identity).transcript == source.transcript
    assert p.reader.load_words(source.final_generation.generation_id) == words
    assert all(path.read_bytes() == original for path, original in audio.items())
    assert not p.requests


def test_product_speaker_source_words_race_does_not_persist(product, qtbot):
    p = product
    identity, _, source = reviewable_final(p, qtbot)
    p.window.meeting_detail_widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    query = QSqlQuery(p.db)
    query.prepare(
        "UPDATE meeting_final_transcription_word SET text = ? WHERE generation_id = ?"
    )
    query.addBindValue("Durable source changed")
    query.addBindValue(str(source.final_generation.generation_id))
    assert query.exec(), query.lastError().text()
    assert all(
        w.text == "Durable source changed"
        for w in p.reader.load_words(source.final_generation.generation_id)
    )
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert (
        p.reviews.load_review_for_generation(source.final_generation.generation_id)
        is None
    )
    assert (
        p.detail.load(identity).speaker_review_state
        is MeetingDetailSpeakerReviewState.ABSENT
    )
    assert (
        "Source changed"
        in p.window.meeting_detail_widget.generation_status_label.text()
    )


def test_product_close_waits_for_speaker_work_and_thread_cleanup(
    product, qtbot, monkeypatch, qapp
):
    p = product
    identity, _, source = reviewable_final(p, qtbot)
    p.window.show()
    widget = p.window.meeting_detail_widget
    widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    worker, thread = p.speakers.worker, p.speakers.worker_thread
    cleanup_entered, allow_cleanup = Event(), Event()
    closed = []

    def hold_cleanup(*args):
        cleanup_entered.set()
        assert allow_cleanup.wait(10)

    worker.outcome.connect(hold_cleanup, Qt.ConnectionType.DirectConnection)

    def close_database():
        assert p.speakers.worker is p.speakers.worker_thread is None
        assert (
            p.reviews.load_review_for_generation(source.final_generation.generation_id)
            is not None
        )
        closed.append(identity)
        p.db.close()

    monkeypatch.setattr(qapp, "close_database", close_database, raising=False)
    try:
        assert not p.window.close()
        assert p.db.isOpen() and not closed
        assert p.speakers.parent() is p.window
        assert p.speakers.worker is worker and p.speakers.worker_thread is thread
        p.speaker_runner.release.set()
        qtbot.waitUntil(cleanup_entered.is_set)
        qtbot.waitUntil(
            lambda: p.reviews.load_review_for_generation(
                source.final_generation.generation_id
            )
            is not None
        )
        assert p.speakers.busy and not thread.wait(0)
        assert not p.window.close()
        assert p.db.isOpen() and not closed
        assert p.speakers.worker is worker and p.speakers.worker_thread is thread
        allow_cleanup.set()
        qtbot.waitUntil(lambda: bool(closed))
        assert closed == [identity]
        assert not p.db.isOpen() and not p.window.isVisible()
    finally:
        allow_cleanup.set()
        p.speaker_runner.release.set()
        qtbot.waitUntil(lambda: not p.speakers.busy)
        assert p.db.open()
        monkeypatch.setattr(qapp, "close_database", lambda: None)


def test_product_concurrent_review_keeps_user_edits(product, qtbot):
    p = product
    identity, _, source = reviewable_final(p, qtbot)
    p.window.meeting_detail_widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    words = p.reader.load_words(source.final_generation.generation_id)
    other = p.reviews.create_review(
        source.final_generation.generation_id, (), map_words_to_speakers(words, ())
    )
    protected = p.reviews.create_speaker(other.id, "Protected participant")
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert (
        p.reviews.load_review_for_generation(source.final_generation.generation_id)
        == protected
    )
    assert p.detail.load(identity).speaker_review == protected
    assert p.window.meeting_detail_widget._snapshot.speaker_review == protected


def test_product_speaker_save_failure_rolls_back_then_explicit_retry(product, qtbot):
    p = product
    identity, stored, source = reviewable_final(p, qtbot)
    words = p.reader.load_words(source.final_generation.generation_id)
    query = QSqlQuery(p.db)
    assert query.exec(
        "CREATE TRIGGER fail_speaker_save BEFORE INSERT ON meeting_speaker_turn BEGIN SELECT RAISE(FAIL, 'controlled save failure'); END"
    ), query.lastError().text()
    widget = p.window.meeting_detail_widget
    widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert (
        p.reviews.load_review_for_generation(source.final_generation.generation_id)
        is None
    )
    assert "controlled save failure" in widget.generation_status_label.text()
    assert widget.generate_review_button.isEnabled()
    assert p.storage.load(identity) == stored
    assert p.reader.load_words(source.final_generation.generation_id) == words
    assert query.exec("DROP TRIGGER fail_speaker_save"), query.lastError().text()
    widget.generate_review_button.click()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert (
        p.detail.load(identity).speaker_review_state
        is MeetingDetailSpeakerReviewState.FRESH
    )


def test_product_open_second_meeting_during_speaker_generation(product, qtbot):
    p = product
    first, _, source = reviewable_final(p, qtbot)
    p.window.meeting_detail_widget.generate_review_button.click()
    qtbot.waitUntil(p.speaker_runner.entered.is_set)
    second, _ = record(p, qtbot)
    transcribe(p, qtbot, start=4)
    assert second != first
    assert p.window.meeting_detail_widget._current_meeting_id == second
    p.speaker_runner.release.set()
    qtbot.waitUntil(lambda: not p.speakers.busy)
    assert (
        p.detail.load(first).speaker_review.source_generation_id
        == source.final_generation.generation_id
    )
    assert p.detail.load(second).speaker_review is None
    assert p.window.meeting_detail_widget._snapshot.speaker_review is None
    p.window.on_meeting_open_requested(first)
    assert p.window.meeting_detail_widget._snapshot.speaker_review is not None


def record(product, qtbot, *, degraded=False):
    p = product
    before = len(p.adapter.calls)
    mic, remote = p.mic, p.remote
    p.window.meeting_capture_widget.start_button.click()
    qtbot.waitUntil(lambda: p.capture.status == "Recording")
    mic.deliver(np.ones(3200, dtype=np.float32) * 0.1)
    remote.deliver(np.ones(3200, dtype=np.float32) * 0.2)
    identity = p.capture.workflow.session_id
    assert not p.requests, "AI started before explicit request"
    assert len(p.adapter.calls) == before, "ASR started before stop/save"
    if degraded:
        remote.fail(AudioSourceError("remote disappeared"))
        qtbot.waitUntil(lambda: "degraded" in p.capture.status)
        mic.deliver(np.ones(1600, dtype=np.float32) * 0.1)
    p.window.meeting_capture_widget.stop_button.click()
    qtbot.waitUntil(lambda: not p.capture.active)
    assert p.capture.saved_id == identity, "saved meeting identity changed"
    stored = p.storage.load(identity)
    assert stored is not None, "meeting persistence skipped"
    assert stored.state is MeetingSessionState.COMPLETED
    assert mic.stop_entered.is_set() and remote.stop_entered.is_set()
    assert not p.requests, "stop implicitly triggered AI"
    assert p.window.meeting_detail_widget is not None
    for track, count in (
        (stored.microphone, 4800 if degraded else 3200),
        (stored.remote, 3200),
    ):
        assert track.sample_count == count
        audio = sf.info(track.path)
        assert audio.frames == count
        assert audio.samplerate == track.sample_rate
        pcm, _ = sf.read(track.path, dtype="float32")
        expected = 0.1 if track == stored.microphone else 0.2
        np.testing.assert_allclose(pcm, expected, atol=0.0001)
    return identity, stored


def transcribe(product, qtbot, *, fail=False, start=0):
    p = product
    for index in range(start + 1, start + 3):
        qtbot.waitUntil(lambda: len(p.adapter.calls) == index)
        path = Path(p.adapter.calls[-1][0])
        assert path.is_file(), "ASR received non-durable audio"
        if fail:
            p.adapter.track_error.emit("controlled model failure")
        else:
            p.adapter.track_completed.emit(
                [TrackTranscriptionInputSegment(0, 100, f"Plan from {path.stem}.")]
            )
    qtbot.waitUntil(lambda: not p.final.pending)


def authoritative(product, identity):
    snapshot = product.detail.load(identity)
    assert snapshot.meeting.session_id == identity
    assert snapshot.final_generation.meeting_id == identity
    assert snapshot.transcript is not None, "authoritative transcript unavailable"
    assert snapshot.transcript.meeting_id == identity, "wrong transcript meeting"
    assert snapshot.transcript.generation_id == snapshot.final_generation.generation_id
    assert snapshot.final_generation.status is FinalTranscriptionStatus.COMPLETED
    return snapshot


def manual(product, identity, text):
    rendered = product.notes.copy_request(identity)
    snapshot = authoritative(product, identity)
    for segment in snapshot.transcript.segments:
        assert segment.text in rendered
    value = replace(product.provider.result, summary=text)
    artifact = product.notes.import_response(identity, meeting_summary_to_json(value))
    assert artifact.meeting_id == identity, "manual context identity changed"
    assert artifact.source_generation_id == snapshot.final_generation.generation_id
    assert product.summaries.load(artifact.summary_id) == artifact
    return artifact


def test_product_happy_path_selected_persisted_minutes(product, qtbot, monkeypatch):
    p = product
    identity, stored = record(p, qtbot)
    transcribe(p, qtbot)
    source = authoritative(p, identity)
    p.notes.submit(identity, config())
    qtbot.waitUntil(p.provider.entered.is_set)
    assert p.requests == [p.notes.prepare(identity).request]
    assert [e.text for e in p.requests[0].transcript] == [
        s.text for s in source.transcript.segments
    ]
    p.provider.allow_result.set()
    p.provider.allow_cleanup.set()
    qtbot.waitUntil(lambda: not p.notes.busy)
    history = p.summaries.list_for_meeting(identity)
    assert len(history) == 1, "API result not persisted for originating meeting"
    (first,) = history
    assert first.source_generation_id == source.final_generation.generation_id
    assert first.source_profile_version == source.final_generation.profile_version
    assert first.source_review_id is None and first.source_review_revision is None
    assert first.meeting_id == identity
    second = manual(p, identity, "SECOND NOTES MUST NOT BE EXPORTED")
    assert {a.summary_id for a in p.notes.history(identity)} == {
        first.summary_id,
        second.summary_id,
    }, "summary history overwritten"
    # Reload through new adapters: no controller cache can stand in for storage.
    assert QSqlMeetingSummaryRepository(p.db).load(first.summary_id) == first
    reloaded = MeetingStorage(QSqlMeetingRepository(p.db), root=p.root / "meetings")
    assert reloaded.load(identity) == stored
    assert authoritative(p, identity) == source
    panel = p.window.meeting_detail_widget.notes_panel
    panel.refresh()
    assert (
        panel.history.currentData() == panel.selected_id
    ), "persisted selection missing from history UI"
    panel.history.setCurrentIndex(
        next(
            i
            for i in range(panel.history.count())
            if panel.history.itemData(i) == first.summary_id
        )
    )
    assert panel.selected_id == first.summary_id
    panel.refresh()
    assert panel.history.currentData() == first.summary_id
    destination = p.root / "selected.md"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(destination), "Markdown (*.md)"),
    )
    panel.export()
    exported = destination.read_bytes()
    assert first.summary.summary.encode() in exported, "selected artifact not exported"
    assert second.summary.summary.encode() not in exported, "latest replaced selection"
    p.notes.export(identity, first.summary_id, p.root / "again.md")
    assert (p.root / "again.md").read_bytes() == exported


def test_product_remote_degradation_preserves_microphone(product, qtbot):
    identity, stored = record(product, qtbot, degraded=True)
    assert stored.audio_outcome is MeetingAudioTracksOutcome.PARTIAL, "PARTIAL coerced"
    assert stored.microphone.sample_count == 4800
    assert "partial" in product.capture.status
    # Drain eligible tracks without assuming damaged remote audio is eligible.
    handled = 0
    while product.final.pending:
        qtbot.waitUntil(
            lambda: len(product.adapter.calls) > handled or not product.final.pending
        )
        if len(product.adapter.calls) > handled:
            handled += 1
            product.adapter.track_completed.emit(
                [TrackTranscriptionInputSegment(0, 100, "usable")]
            )
    assert product.storage.load(identity) == stored


def test_product_final_failure_explicit_retry_same_generation(product, qtbot):
    p = product
    identity, stored = record(p, qtbot)
    transcribe(p, qtbot, fail=True)
    failed = p.reader.load_generation_for_meeting(identity, 1)
    assert failed.status is FinalTranscriptionStatus.FAILED
    assert p.storage.load(identity) == stored
    with pytest.raises(NotesError):
        p.notes.prepare(identity)
    assert len(p.adapter.calls) == 2
    p.final.retry(failed.generation_id)
    transcribe(p, qtbot, start=2)
    source = authoritative(p, identity)
    assert (
        source.final_generation.generation_id == failed.generation_id
    ), "retry generation changed"
    assert p.storage.load(identity) == stored


def test_product_api_failure_manual_fallback_preserves_source(product, qtbot):
    p = product
    identity, stored = record(p, qtbot)
    transcribe(p, qtbot)
    before = authoritative(p, identity)
    existing = manual(p, identity, "Earlier history")
    p.provider.error = RuntimeError("controlled API failure")
    p.notes.submit(identity, config())
    qtbot.waitUntil(p.provider.entered.is_set)
    p.provider.allow_result.set()
    p.provider.allow_cleanup.set()
    qtbot.waitUntil(lambda: not p.notes.busy)
    assert p.notes.history(identity) == (existing,), "API failure corrupted history"
    assert p.storage.load(identity) == stored, "API failure mutated meeting"
    assert authoritative(p, identity) == before, "API failure mutated transcript"
    recovered = manual(p, identity, "Manual recovery")
    assert p.notes.manual_contexts[identity] == p.notes.prepare(identity)
    assert {a.summary_id for a in p.notes.history(identity)} == {
        existing.summary_id,
        recovered.summary_id,
    }
    p.notes.export(identity, recovered.summary_id, p.root / "recovered.txt")
    assert "Manual recovery" in (p.root / "recovered.txt").read_text(encoding="utf-8")
    assert authoritative(p, identity) == before


def test_product_stale_history_requires_acknowledgement(product, qtbot, monkeypatch):
    p = product
    identity, _ = record(p, qtbot)
    transcribe(p, qtbot)
    artifact = manual(p, identity, "Original notes")
    assert p.notes.freshness(artifact) is MeetingSummaryFreshness.FRESH
    # A newer authoritative profile is persisted by the real scheduler. While
    # its ASR is outstanding, the old summary must already cease to be fresh.
    p.final.request(
        identity,
        FinalTranscriptionConfig(profile_version=2, whisper_model_size="SMALL"),
    )
    qtbot.waitUntil(
        lambda: p.reader.load_generation_for_meeting(identity, 2) is not None
    )
    assert (
        p.notes.freshness(artifact) is not MeetingSummaryFreshness.FRESH
    ), "stale shown fresh"
    assert p.notes.history(identity) == (artifact,)
    destination = p.root / "stale.md"
    with pytest.raises(NotesError, match="Acknowledge"):
        p.notes.export(identity, artifact.summary_id, destination)
    panel = p.window.meeting_detail_widget.notes_panel
    panel.refresh()
    assert "Fresh" not in panel.provenance.text()
    dialog = Mock(return_value=(str(destination), "Markdown (*.md)"))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialog)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    panel.export()
    dialog.assert_not_called()
    assert not destination.exists()
    p.notes.export(identity, artifact.summary_id, destination, acknowledged=True)
    assert "Original notes" in destination.read_text(encoding="utf-8")
    transcribe(p, qtbot, fail=True, start=2)


def test_product_close_waits_for_sql_persistence_and_provider_release(
    product, qtbot, monkeypatch, qapp
):
    p = product
    identity, _ = record(p, qtbot)
    transcribe(p, qtbot)
    order = []
    save = p.summaries.save

    def persist(artifact):
        assert p.db.isOpen()
        save(artifact)
        order.append("persisted")

    def close_database():
        assert (
            p.notes.worker is None and p.notes.worker_thread is None
        ), "database closed with provider ownership"
        assert len(p.summaries.list_for_meeting(identity)) == 1
        order.append("database closed")
        p.db.close()

    monkeypatch.setattr(p.summaries, "save", persist)
    monkeypatch.setattr(qapp, "close_database", close_database, raising=False)
    p.window.show()
    p.notes.submit(identity, config())
    qtbot.waitUntil(p.provider.entered.is_set)
    worker, thread = p.notes.worker, p.notes.worker_thread
    order.append("close requested")
    assert not p.window.close()
    assert (
        p.db.isOpen() and p.notes.parent() is p.window
    ), "database closed with provider ownership"
    assert p.notes.worker is worker and p.notes.worker_thread is thread
    p.provider.allow_result.set()
    qtbot.waitUntil(p.provider.cleanup_entered.is_set)
    assert order == ["close requested", "persisted"]
    assert p.summaries.list_for_meeting(identity)[0].meeting_id == identity
    assert not p.window.close()  # shutdown(False) must retain ownership.
    assert p.db.isOpen() and p.notes.worker is worker
    p.provider.allow_cleanup.set()
    qtbot.waitUntil(lambda: not p.window.isVisible())
    assert order == ["close requested", "persisted", "database closed"]
    assert not p.db.isOpen()
    assert p.db.open()  # Permit normal fixture verification/teardown.
    monkeypatch.setattr(qapp, "close_database", lambda: None)
