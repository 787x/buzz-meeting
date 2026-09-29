"""Explicit, application-owned generation of a durable meeting Speaker Review.

Only immutable source DTOs cross the worker boundary. Reads, source revalidation
and review persistence stay on the controller's QSql owner thread.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from buzz.meeting.final_transcription import (
    FinalTranscriptionGeneration,
    FinalTranscriptionStatus,
    FinalTranscriptionTrackStatus,
    MeetingTranscriptWord,
)
from buzz.meeting.meeting_detail import (
    MeetingDetailSnapshot,
    MeetingDetailSpeakerReviewState,
    MeetingDetailTranscriptState,
)
from buzz.meeting.meeting_storage import StoredMeetingAudioTrack
from buzz.meeting.speaker_diarization import (
    SpeakerDiarizationAudio,
    SpeakerDiarizationBackend,
    SpeakerDiarizationService,
)
from buzz.meeting.speaker_diarization_adapter import WhisperDiarizationRunner
from buzz.meeting.speaker_mapping import MeetingTrackSpeakerTurns, map_words_to_speakers
from buzz.meeting.speaker_review import SpeakerReviewTrackAnalysis
from buzz.settings.settings import Settings


def can_generate_review(snapshot: MeetingDetailSnapshot) -> bool:
    generation = snapshot.final_generation
    return (
        snapshot.transcript_state is MeetingDetailTranscriptState.AVAILABLE
        and snapshot.speaker_review_state is MeetingDetailSpeakerReviewState.ABSENT
        and generation is not None
        and generation.profile_version == 2
        and generation.status
        in (FinalTranscriptionStatus.COMPLETED, FinalTranscriptionStatus.PARTIAL)
    )


def _decode_audio(path):
    from faster_whisper import decode_audio

    return SpeakerDiarizationAudio(decode_audio(str(path), sampling_rate=16000), 16000)


def _diarization_service(force_cpu):
    import torch

    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    # Preserve the legacy dialog's default; there is no meeting-specific setting.
    return SpeakerDiarizationService(
        WhisperDiarizationRunner(SpeakerDiarizationBackend.MSDD, device)
    )


@dataclass(frozen=True)
class _Source:
    meeting_id: uuid.UUID
    generation: FinalTranscriptionGeneration
    words: tuple[MeetingTranscriptWord, ...]
    audio: tuple[StoredMeetingAudioTrack, ...]


def _source_audio(snapshot):
    tracks = {
        track.role: track
        for track in (snapshot.meeting.microphone, snapshot.meeting.remote)
        if track is not None
    }
    result = []
    for track in snapshot.final_generation.tracks:
        if track.status is not FinalTranscriptionTrackStatus.COMPLETED:
            continue
        audio = tracks.get(track.role)
        if audio is None or not audio.published or not audio.asset_exists_at_load:
            raise RuntimeError("Completed transcript track has no stored audio.")
        result.append(audio)
    return tuple(result)


class _GenerationWorker(QObject):
    outcome = pyqtSignal(object, str)
    finished = pyqtSignal()

    def __init__(self, source, decoder, service_factory, force_cpu):
        super().__init__()
        self.source = source
        self.decoder = decoder
        self.service_factory = service_factory
        self.force_cpu = force_cpu

    @pyqtSlot()
    def run(self):
        try:
            analyses = []
            service = self.service_factory(self.force_cpu)
            for audio in self.source.audio:
                turns = service.diarize(self.decoder(audio.path))
                analyses.append(
                    SpeakerReviewTrackAnalysis(
                        audio.role, turns, SpeakerDiarizationBackend.MSDD, 1
                    )
                )
            attributed = map_words_to_speakers(
                self.source.words,
                tuple(
                    MeetingTrackSpeakerTurns(a.source_role, a.turns) for a in analyses
                ),
            )
            self.outcome.emit((tuple(analyses), attributed), "")
        except Exception as exc:
            logging.exception("Meeting Speaker Review preparation failed")
            self.outcome.emit(None, f"Speaker Review generation failed: {exc}")
        finally:
            # Request exit; the owner still retains this worker until thread join.
            self.finished.emit()


class MeetingSpeakerGeneration(QObject):
    changed = pyqtSignal(object)  # originating meeting, never the selected window
    idle = pyqtSignal()

    def __init__(
        self,
        detail_service,
        reader,
        reviews,
        parent=None,
        *,
        decoder=_decode_audio,
        service_factory=_diarization_service,
    ):
        super().__init__(parent)
        self._detail = detail_service
        self._reader = reader
        self._reviews = reviews
        self._decoder = decoder
        self._service_factory = service_factory
        self.worker = None
        self.worker_thread = None
        self._source = None
        self._delivered = False
        self.closing = False
        self._status_meeting = None
        self._status = ""

    def _owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Speaker Review generation requires its owner thread")

    @property
    def busy(self):
        return self._source is not None

    def message_for(self, meeting_id):
        if self.busy:
            return (
                "Generating Speaker Review…"
                if self._source.meeting_id == meeting_id
                else "Speaker Review generation is running for another meeting."
            )
        return self._status if self._status_meeting == meeting_id else ""

    def _status_for(self, meeting_id, message):
        self._status_meeting, self._status = meeting_id, message
        self.changed.emit(meeting_id)

    def submit(self, meeting_id):
        self._owner()
        if self.closing or self.busy:
            return False
        try:
            snapshot = self._detail.load(meeting_id)
            if not can_generate_review(snapshot):
                raise RuntimeError(
                    "This meeting is not eligible for a new Speaker Review."
                )
            generation = snapshot.final_generation
            if generation.meeting_id != meeting_id:
                raise RuntimeError("Final transcription belongs to another meeting.")
            source = _Source(
                meeting_id,
                generation,
                self._reader.load_words(generation.generation_id),
                _source_audio(snapshot),
            )
            force_cpu = os.getenv(
                "BUZZ_FORCE_CPU", "false"
            ).lower() == "true" or Settings().value(Settings.Key.FORCE_CPU, False)
            self.worker_thread = QThread(self)
            self.worker = _GenerationWorker(
                source, self._decoder, self._service_factory, force_cpu
            )
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)
            self.worker.outcome.connect(self._outcome)
            self.worker.finished.connect(
                self.worker_thread.quit, Qt.ConnectionType.DirectConnection
            )
            self.worker_thread.finished.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self._finished)
            self._source = source
            self._delivered = False
            self.worker_thread.start()
        except Exception as exc:
            self._status_for(meeting_id, f"Could not generate Speaker Review: {exc}")
            return False
        self.changed.emit(meeting_id)
        return True

    @pyqtSlot(object, str)
    def _outcome(self, result, error):
        self._owner()
        source = self._source
        try:
            if error:
                raise RuntimeError(error)
            current = self._detail.load(source.meeting_id)
            if (
                not can_generate_review(current)
                or current.final_generation != source.generation
                or self._reader.load_words(source.generation.generation_id)
                != source.words
                or _source_audio(current) != source.audio
            ):
                raise RuntimeError(
                    "Source changed or a review already exists. Result was not saved."
                )
            # The existing boundary also checks conflicts and exact word equality.
            self._reviews.create_review(source.generation.generation_id, *result)
            self._status_for(source.meeting_id, "Speaker Review saved.")
        except Exception as exc:
            logging.exception("Meeting Speaker Review generation or save failed")
            self._status_for(
                source.meeting_id, f"Could not generate Speaker Review: {exc}"
            )
        finally:
            self._delivered = True

    @pyqtSlot()
    def _finished(self):
        self._owner()
        if self.worker_thread is None:
            return
        # finished can arrive before a queued outcome or native thread teardown.
        if not self._delivered or not self.worker_thread.wait(0):
            QTimer.singleShot(10, self._finished)
            return
        meeting_id = self._source.meeting_id
        self.worker = None
        self.worker_thread.deleteLater()
        self.worker_thread = None
        self._source = None
        self.changed.emit(meeting_id)
        self.idle.emit()

    def close(self):
        self._owner()
        self.closing = True
        return not self.busy
