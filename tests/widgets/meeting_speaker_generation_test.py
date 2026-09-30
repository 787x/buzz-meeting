"""Deterministic worker, source-race and widget tests without model inference."""

from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication

from buzz.meeting.final_transcription import (
    FinalTranscriptionStatus,
    FinalTranscriptionTrack,
    FinalTranscriptionTrackStatus,
)
from buzz.meeting.meeting_audio_tracks import MeetingTrackRole
from buzz.meeting.meeting_detail import MeetingDetailSpeakerReviewState
from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationAudio,
    SpeakerDiarizationService,
    SpeakerDiarizationTurn,
)
from buzz.meeting.speaker_mapping import (
    SpeakerAttributionStatus,
    map_words_to_speakers,
)
from buzz.meeting.speaker_review import (
    MeetingSpeakerReviewService,
    SpeakerReviewAnalysisState,
    SpeakerReviewDecodeError,
    SpeakerReviewError,
    SpeakerReviewStaleError,
)
from buzz.widgets.meeting_detail_widget import MeetingDetailWidget
from buzz.widgets.meeting_speaker_generation import MeetingSpeakerGeneration
from tests.meeting.speaker_review_test import (
    FakeFinalTranscriptSource,
    FakeMeetingSpeakerRepository,
)
from tests.widgets.meeting_detail_widget_test import (
    MEETING_ID,
    PreviewPlayer,
    generation,
    reviewed_word,
    snapshot,
)


@pytest.fixture(scope="session")
def qapp_cls():
    return QApplication


class ControlledRunner:
    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.calls = []
        self.error = None

    def diarize(self, audio):
        self.calls.append((audio, get_ident()))
        self.entered.set()
        if not self.release.wait(10):
            raise RuntimeError("test backend was not released")
        if self.error:
            raise self.error
        return (SpeakerDiarizationTurn(7, 0, 1000),)


@pytest.fixture
def generation_case(qtbot):
    owner = get_ident()
    source = FakeFinalTranscriptSource(
        replace(
            generation(),
            tracks=tuple(
                FinalTranscriptionTrack(role, FinalTranscriptionTrackStatus.COMPLETED)
                for role in MeetingTrackRole
            ),
        ),
        (reviewed_word(0, None, False).word,),
    )
    original_words, original_generation = source.load_words, source.load_generation

    def words(identity):
        assert get_ident() == owner
        return original_words(identity)

    def load_generation(identity):
        assert get_ident() == owner
        return original_generation(identity)

    source.load_words = Mock(side_effect=words)
    source.load_generation = Mock(side_effect=load_generation)
    repo = FakeMeetingSpeakerRepository()
    reviews = MeetingSpeakerReviewService(repo, source)
    create = reviews.create_review

    def persist(*args):
        assert get_ident() == owner
        return create(*args)

    reviews.create_review = Mock(side_effect=persist)
    base = snapshot(
        generation_value=source.generation,
        review_state=MeetingDetailSpeakerReviewState.ABSENT,
    )
    case = SimpleNamespace(
        owner=owner,
        source=source,
        repo=repo,
        reviews=reviews,
        base=base,
        create=create,
        paths=[],
        runner=ControlledRunner(),
    )

    def load(identity):
        assert get_ident() == owner
        value = case.base
        if identity != MEETING_ID:
            return replace(value, meeting=replace(value.meeting, session_id=identity))
        state, review = value.speaker_review_state, None
        try:
            review = reviews.load_review_for_generation(source.generation.generation_id)
            if review is not None:
                state = MeetingDetailSpeakerReviewState.FRESH
        except SpeakerReviewStaleError:
            state = MeetingDetailSpeakerReviewState.STALE
        except SpeakerReviewDecodeError:
            state = MeetingDetailSpeakerReviewState.CORRUPT
        except SpeakerReviewError:
            state = MeetingDetailSpeakerReviewState.LOAD_FAILED
        return replace(
            value,
            final_generation=source.generation,
            speaker_review_state=state,
            speaker_review=review,
        )

    case.detail = SimpleNamespace(load=load)

    def decode(path):
        assert get_ident() != owner
        case.paths.append(path)
        return SpeakerDiarizationAudio(np.zeros(160, dtype=np.float32), 16000)

    def service(force_cpu):
        assert get_ident() != owner
        return SpeakerDiarizationService(case.runner)

    case.controller = MeetingSpeakerGeneration(
        case.detail, source, reviews, decoder=decode, service_factory=service
    )
    case.widget = MeetingDetailWidget(
        case.detail,
        reviews,
        lambda _: PreviewPlayer(),
        speaker_generation=case.controller,
    )
    qtbot.addWidget(case.widget)
    case.widget.open_meeting(MEETING_ID)
    yield case
    case.runner.release.set()
    qtbot.waitUntil(lambda: not case.controller.busy, timeout=15000)
    assert case.controller.close()


def trigger(case, qtbot):
    case.widget.generate_review_button.click()
    qtbot.waitUntil(case.runner.entered.is_set)


def finish(case, qtbot):
    case.runner.release.set()
    qtbot.waitUntil(lambda: not case.controller.busy, timeout=15000)
    assert case.controller.worker is case.controller.worker_thread is None


def test_explicit_generation_off_gui_responsive_single_request_and_canonical_reload(
    generation_case, qtbot, monkeypatch
):
    case = generation_case
    assert not case.runner.calls
    assert case.widget.generate_review_button.isEnabled()
    # Display text is deliberately unrelated to the durable word rows.
    case.widget.transcript_edit.setPlainText("This must never become source words")
    from buzz.widgets import meeting_speaker_generation as module

    mapper = Mock(wraps=map_words_to_speakers)
    monkeypatch.setattr(module, "map_words_to_speakers", mapper)
    trigger(case, qtbot)
    assert case.controller.busy
    assert "Generating" in case.widget.generation_status_label.text()
    assert not case.widget.generate_review_button.isEnabled()
    assert case.runner.calls[0][1] != case.owner
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(get_ident()))
    qtbot.waitUntil(lambda: bool(ticks))
    assert ticks == [case.owner]
    case.widget.generate_review_button.click()
    assert case.controller.submit(MEETING_ID) is False
    assert len(case.runner.calls) == 1
    case.reviews.create_review.assert_not_called()
    finish(case, qtbot)
    mapper.assert_called_once()
    assert mapper.call_args.args[0] == case.source.words
    case.reviews.create_review.assert_called_once()
    assert (
        case.reviews.create_review.call_args.args[0]
        == case.source.generation.generation_id
    )
    assert case.paths == [
        case.base.meeting.microphone.path,
        case.base.meeting.remote.path,
    ]
    reloaded = MeetingSpeakerReviewService(
        case.repo, case.source
    ).load_review_for_generation(case.source.generation.generation_id)
    assert reloaded.words[0].word == case.source.words[0]
    assert reloaded.words[0].machine_status is SpeakerAttributionStatus.ASSIGNED
    assert case.widget._snapshot.speaker_review == reloaded
    assert case.widget.word_model.rowCount() == 1
    assert not case.widget.generate_review_button.isEnabled()


@pytest.mark.parametrize(
    "failure", ["decode", "diarize", "map", "persist", "source_read"]
)
def test_failure_preserves_sources_clears_busy_and_allows_explicit_retry(
    generation_case, qtbot, monkeypatch, failure
):
    from buzz.widgets import meeting_speaker_generation as module

    case = generation_case
    initial_generation, initial_words = case.source.generation, case.source.words
    with monkeypatch.context() as patch:
        if failure == "decode":
            patch.setattr(
                case.controller,
                "_decoder",
                Mock(side_effect=RuntimeError("decode failed")),
            )
        elif failure == "diarize":
            case.runner.error = RuntimeError("controlled backend failure")
        elif failure == "map":
            patch.setattr(
                module,
                "map_words_to_speakers",
                Mock(side_effect=RuntimeError("mapping failed")),
            )
        elif failure == "persist":
            patch.setattr(
                case.repo,
                "create_review",
                Mock(side_effect=SpeakerReviewError("save failed")),
            )
        else:
            patch.setattr(
                case.source,
                "load_words",
                Mock(side_effect=RuntimeError("word read failed")),
            )
        case.widget.generate_review_button.click()
        finish(case, qtbot)
        assert case.repo.bundle is None
        assert "Could not generate" in case.widget.generation_status_label.text()
        assert case.widget.generate_review_button.isEnabled()
        assert case.source.generation == initial_generation
        assert case.source.words == initial_words
    case.runner.error = None
    case.widget.generate_review_button.click()
    finish(case, qtbot)
    assert case.repo.bundle is not None


@pytest.mark.parametrize(
    "race", ["words", "generation", "new_generation", "audio", "existing_review"]
)
def test_source_or_existing_review_race_never_saves_or_overwrites(
    generation_case, qtbot, race
):
    case = generation_case
    trigger(case, qtbot)
    existing = None
    if race == "words":
        case.source.words = (
            replace(case.source.words[0], text="changed durable word"),
        )
    elif race == "generation":
        case.source.generation = replace(
            case.source.generation, status=FinalTranscriptionStatus.IN_PROGRESS
        )
    elif race == "new_generation":
        import uuid

        case.source.generation = replace(
            case.source.generation, generation_id=uuid.UUID(int=999)
        )
    elif race == "audio":
        case.base = replace(
            case.base,
            meeting=replace(
                case.base.meeting,
                microphone=replace(case.base.meeting.microphone, sample_count=64000),
            ),
        )
    else:
        existing = case.create(
            case.source.generation.generation_id,
            (),
            map_words_to_speakers(case.source.words, ()),
        )
        existing = case.repo.bundle
    finish(case, qtbot)
    case.reviews.create_review.assert_not_called()
    assert case.repo.bundle == existing
    assert "Source changed" in case.widget.generation_status_label.text()


@pytest.mark.parametrize("state", ["fresh", "stale", "corrupt", "load_failed"])
def test_existing_review_in_every_state_is_preserved_and_never_generates(
    generation_case, monkeypatch, state
):
    case = generation_case
    case.create(
        case.source.generation.generation_id,
        (),
        map_words_to_speakers(case.source.words, ()),
    )
    if state == "stale":
        case.source.words = (replace(case.source.words[0], text="already stale"),)
    elif state == "corrupt":
        case.repo.bundle = replace(
            case.repo.bundle, header=replace(case.repo.bundle.header, status="CORRUPT")
        )
    elif state == "load_failed":
        monkeypatch.setattr(
            case.repo,
            "load_review_for_generation",
            Mock(side_effect=SpeakerReviewError("cannot read review")),
        )
    bundle = case.repo.bundle
    case.widget.refresh()
    assert not case.widget.generate_review_button.isEnabled()
    case.widget.generate_review_button.click()
    assert case.controller.submit(MEETING_ID) is False
    assert not case.controller.busy
    assert not case.runner.calls
    case.reviews.create_review.assert_not_called()
    assert case.repo.bundle == bundle


@pytest.mark.parametrize(
    "status",
    [FinalTranscriptionTrackStatus.FAILED, FinalTranscriptionTrackStatus.INELIGIBLE],
)
def test_partial_generation_diarizes_only_completed_track(
    generation_case, qtbot, status
):
    case = generation_case
    case.source.generation = replace(
        case.source.generation,
        status=FinalTranscriptionStatus.PARTIAL,
        tracks=(
            case.source.generation.tracks[0],
            replace(
                case.source.generation.tracks[1],
                status=status,
            ),
        ),
    )
    case.base = replace(
        case.base,
        meeting=replace(
            case.base.meeting,
            remote=replace(case.base.meeting.remote, asset_exists_at_load=False),
        ),
    )
    case.widget.refresh()
    trigger(case, qtbot)
    finish(case, qtbot)
    assert case.paths == [case.base.meeting.microphone.path]
    review = case.widget._snapshot.speaker_review
    failed = next(t for t in review.tracks if t.source_role is MeetingTrackRole.REMOTE)
    assert failed.analysis_state is SpeakerReviewAnalysisState.NOT_PROVIDED
    assert failed.turn_count == failed.source_word_count == 0


def test_open_another_meeting_keeps_result_and_failure_associated_with_origin(
    generation_case, qtbot
):
    import uuid

    case = generation_case
    trigger(case, qtbot)
    second = uuid.UUID(int=111)
    case.widget.open_meeting(second)
    assert "another meeting" in case.widget.generation_status_label.text()
    assert not case.widget.generate_review_button.isEnabled()
    finish(case, qtbot)
    assert case.repo.bundle.header.source_generation_id == str(
        case.source.generation.generation_id
    )
    assert case.widget._current_meeting_id == second
    assert case.widget._snapshot.meeting.session_id == second
    assert case.widget._snapshot.speaker_review is None
    assert not case.widget.generation_status_label.text()
    case.widget.open_meeting(MEETING_ID)
    assert case.widget._snapshot.speaker_review is not None


def test_ownership_and_busy_outlive_outcome_until_thread_really_finishes(
    generation_case, qtbot
):
    case = generation_case
    trigger(case, qtbot)
    entered, release = Event(), Event()

    def hold_after_outcome(*args):
        entered.set()
        assert release.wait(10)

    case.controller.worker.outcome.connect(
        hold_after_outcome, Qt.ConnectionType.DirectConnection
    )
    worker, thread = case.controller.worker, case.controller.worker_thread
    try:
        case.runner.release.set()
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(lambda: case.repo.bundle is not None)
        assert case.controller.close() is False
        assert (
            case.controller.worker is worker and case.controller.worker_thread is thread
        )
        assert not thread.wait(0)
        assert case.controller.busy
        assert case.controller.submit(MEETING_ID) is False
    finally:
        release.set()
        finish(case, qtbot)
    assert case.controller.close()


def test_controller_rejects_non_owner_thread(generation_case, qtbot):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        operation = pool.submit(generation_case.controller.submit, MEETING_ID)
        with pytest.raises(RuntimeError, match="owner thread"):
            operation.result(timeout=5)


@pytest.mark.parametrize("force_cpu", [False, True])
@pytest.mark.parametrize("cuda_available", [False, True])
def test_default_runner_keeps_msdd_and_respects_cpu_preference(
    monkeypatch, force_cpu, cuda_available
):
    import sys
    from buzz.widgets import meeting_speaker_generation as module
    from buzz.meeting.speaker_diarization import SpeakerDiarizationBackend

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available)),
    )
    runner = Mock()
    monkeypatch.setattr(module, "WhisperDiarizationRunner", runner)
    assert isinstance(module._diarization_service(force_cpu), SpeakerDiarizationService)
    device = "cuda" if cuda_available and not force_cpu else "cpu"
    runner.assert_called_once_with(SpeakerDiarizationBackend.MSDD, device)


def test_missing_completed_source_audio_fails_without_starting_worker(generation_case):
    case = generation_case
    case.base = replace(
        case.base,
        meeting=replace(
            case.base.meeting,
            microphone=replace(
                case.base.meeting.microphone, asset_exists_at_load=False
            ),
        ),
    )
    case.widget.refresh()
    case.widget.generate_review_button.click()
    assert not case.controller.busy
    assert not case.runner.calls
    assert case.repo.bundle is None
    assert "no stored audio" in case.widget.generation_status_label.text()
    assert case.widget.generate_review_button.isEnabled()
