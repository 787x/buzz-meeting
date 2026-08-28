"""Pure read model for one durable meeting and its derived review data."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from buzz.meeting.final_transcription import (
    FinalTranscriptionDecodeError,
    FinalTranscriptionError,
    FinalTranscriptionGeneration,
    FinalTranscriptionReadService,
    MeetingTranscript,
)
from buzz.meeting.meeting_storage import (
    MeetingStorageDatabaseError,
    MeetingStorageDecodeError,
    MeetingStorageError,
    MeetingStorageFilesystemError,
    StoredMeeting,
)
from buzz.meeting.speaker_review import (
    MeetingSpeakerReview,
    MeetingSpeakerReviewService,
    SpeakerReviewDecodeError,
    SpeakerReviewError,
    SpeakerReviewStaleError,
)


class MeetingDetailError(Exception):
    """Base error for loading a meeting detail snapshot."""


class MeetingDetailNotFoundError(MeetingDetailError):
    """The selected meeting does not exist."""


class MeetingDetailLoadError(MeetingDetailError):
    """The primary stored meeting could not be decoded or loaded."""

    def __init__(self, message: str, *, corrupt: bool = False) -> None:
        super().__init__(message)
        self.corrupt = corrupt


class MeetingDetailTranscriptState(Enum):
    NOT_AVAILABLE = auto()
    AVAILABLE = auto()
    CORRUPT = auto()
    LOAD_FAILED = auto()


class MeetingDetailSpeakerReviewState(Enum):
    NOT_APPLICABLE = auto()
    ABSENT = auto()
    FRESH = auto()
    STALE = auto()
    CORRUPT = auto()
    LOAD_FAILED = auto()


@dataclass(frozen=True, slots=True)
class MeetingDetailSnapshot:
    meeting: StoredMeeting
    transcript_state: MeetingDetailTranscriptState
    final_generation: FinalTranscriptionGeneration | None
    transcript: MeetingTranscript | None
    speaker_review_state: MeetingDetailSpeakerReviewState
    speaker_review: MeetingSpeakerReview | None

    def __post_init__(self) -> None:
        if not isinstance(self.meeting, StoredMeeting):
            raise ValueError("meeting must be StoredMeeting")
        if self.transcript_state is MeetingDetailTranscriptState.NOT_AVAILABLE and (
            self.final_generation is not None or self.transcript is not None
        ):
            raise ValueError("NOT_AVAILABLE cannot expose transcription data")
        if (
            self.transcript_state is MeetingDetailTranscriptState.AVAILABLE
            and self.final_generation is None
        ):
            raise ValueError("AVAILABLE requires a final generation")
        if (
            self.transcript_state
            in (
                MeetingDetailTranscriptState.CORRUPT,
                MeetingDetailTranscriptState.LOAD_FAILED,
            )
            and self.transcript is not None
        ):
            raise ValueError("failed transcript reads cannot expose a transcript")
        if (self.speaker_review is not None) != (
            self.speaker_review_state is MeetingDetailSpeakerReviewState.FRESH
        ):
            raise ValueError("speaker review is exposed if and only if it is fresh")


class MeetingLoader(Protocol):
    def load(self, session_id: uuid.UUID) -> StoredMeeting | None:
        ...


class MeetingDetailService:
    """Compose fresh storage, final-transcription, and review reads."""

    def __init__(
        self,
        meeting_storage: MeetingLoader,
        final_transcription_reader: FinalTranscriptionReadService,
        speaker_review_service: MeetingSpeakerReviewService,
    ) -> None:
        self._meeting_storage = meeting_storage
        self._reader = final_transcription_reader
        self._speaker_reviews = speaker_review_service

    def load(self, session_id: uuid.UUID) -> MeetingDetailSnapshot:
        try:
            meeting = self._meeting_storage.load(session_id)
        except MeetingStorageDecodeError as exc:
            raise MeetingDetailLoadError(
                "Stored meeting data is corrupt", corrupt=True
            ) from exc
        except (MeetingStorageFilesystemError, MeetingStorageDatabaseError) as exc:
            raise MeetingDetailLoadError("Could not load stored meeting") from exc
        except MeetingStorageError as exc:
            raise MeetingDetailLoadError("Could not load stored meeting") from exc
        if meeting is None:
            raise MeetingDetailNotFoundError(f"Meeting {session_id} not found")

        generation: FinalTranscriptionGeneration | None = None
        transcript: MeetingTranscript | None = None
        transcript_state = MeetingDetailTranscriptState.NOT_AVAILABLE
        try:
            generation = self._reader.load_generation_for_meeting(session_id, 2)
            if generation is None:
                generation = self._reader.load_generation_for_meeting(session_id, 1)
            if generation is not None:
                transcript = self._reader.load_transcript(generation.generation_id)
                transcript_state = MeetingDetailTranscriptState.AVAILABLE
        except FinalTranscriptionDecodeError:
            transcript = None
            transcript_state = MeetingDetailTranscriptState.CORRUPT
        except FinalTranscriptionError:
            transcript = None
            transcript_state = MeetingDetailTranscriptState.LOAD_FAILED

        review_state = MeetingDetailSpeakerReviewState.NOT_APPLICABLE
        review: MeetingSpeakerReview | None = None
        if generation is not None and generation.profile_version == 2:
            try:
                review = self._speaker_reviews.load_review_for_generation(
                    generation.generation_id
                )
                review_state = (
                    MeetingDetailSpeakerReviewState.ABSENT
                    if review is None
                    else MeetingDetailSpeakerReviewState.FRESH
                )
            except SpeakerReviewStaleError:
                review_state = MeetingDetailSpeakerReviewState.STALE
            except SpeakerReviewDecodeError:
                review_state = MeetingDetailSpeakerReviewState.CORRUPT
            except SpeakerReviewError:
                review_state = MeetingDetailSpeakerReviewState.LOAD_FAILED

        return MeetingDetailSnapshot(
            meeting=meeting,
            transcript_state=transcript_state,
            final_generation=generation,
            transcript=transcript,
            speaker_review_state=review_state,
            speaker_review=review,
        )


__all__ = [
    "MeetingDetailError",
    "MeetingDetailLoadError",
    "MeetingDetailNotFoundError",
    "MeetingDetailService",
    "MeetingDetailSnapshot",
    "MeetingDetailSpeakerReviewState",
    "MeetingDetailTranscriptState",
]
