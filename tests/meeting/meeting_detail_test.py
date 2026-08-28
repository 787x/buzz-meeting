from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from unittest.mock import Mock

import pytest

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionDecodeError,
    FinalTranscriptionGeneration,
    FinalTranscriptionStateError,
    FinalTranscriptionStatus,
    MeetingTranscript,
    MeetingTranscriptSegment,
)
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
    MeetingTrackRole,
)
from buzz.meeting.meeting_detail import (
    MeetingDetailLoadError,
    MeetingDetailNotFoundError,
    MeetingDetailService,
    MeetingDetailSpeakerReviewState,
    MeetingDetailTranscriptState,
)
from buzz.meeting.meeting_recorder import MeetingRecorderState
from buzz.meeting.meeting_session import MeetingRemoteSourceKind, MeetingSessionState
from buzz.meeting.meeting_storage import (
    MeetingStorageDatabaseError,
    MeetingStorageDecodeError,
    MeetingStorageFilesystemError,
    StoredMeeting,
    StoredMeetingAudioTrack,
)
from buzz.meeting.speaker_review import (
    SpeakerReviewDecodeError,
    SpeakerReviewError,
    SpeakerReviewStaleError,
)


MEETING_ID = uuid.UUID(int=17)


def make_track(role: MeetingTrackRole, *, asset_exists: bool = True):
    return StoredMeetingAudioTrack(
        role=role,
        relative_path=PurePosixPath(f"{role.name.lower()}.wav"),
        path=Path(f"C:/meetings/{role.name.lower()}.wav"),
        sample_rate=16_000,
        sample_count=32_000,
        recording_state=MeetingRecorderState.STOPPED,
        published=True,
        complete=True,
        timing_basis="host_callback_arrival",
        timing_anchors=(),
        errors=(),
        asset_exists_at_load=asset_exists,
    )


def make_meeting(*, asset_exists: bool = True) -> StoredMeeting:
    return StoredMeeting(
        session_id=MEETING_ID,
        remote_source_kind=MeetingRemoteSourceKind.SYSTEM,
        state=MeetingSessionState.COMPLETED,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
        duration_ns=2_000_000_000,
        audio_state=MeetingAudioTracksState.STOPPED,
        audio_outcome=MeetingAudioTracksOutcome.COMPLETE,
        microphone=make_track(MeetingTrackRole.MICROPHONE, asset_exists=asset_exists),
        remote=make_track(MeetingTrackRole.REMOTE),
    )


def make_generation(
    profile: int,
    status: FinalTranscriptionStatus = FinalTranscriptionStatus.COMPLETED,
) -> FinalTranscriptionGeneration:
    config = (
        FinalTranscriptionConfig(profile_version=2, whisper_model_size="LARGE")
        if profile == 2
        else FinalTranscriptionConfig()
    )
    return FinalTranscriptionGeneration(
        generation_id=uuid.UUID(int=100 + profile),
        meeting_id=MEETING_ID,
        profile_version=profile,
        status=status,
        config=config,
        tracks=(),
    )


def make_transcript(generation: FinalTranscriptionGeneration) -> MeetingTranscript:
    return MeetingTranscript(
        generation.generation_id,
        MEETING_ID,
        generation.status,
        (
            MeetingTranscriptSegment(
                0,
                MeetingTrackRole.MICROPHONE,
                0,
                0,
                100,
                0,
                100_000_000,
                "hello",
            ),
        ),
    )


class Storage:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def load(self, meeting_id):
        self.calls.append(meeting_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Reader:
    def __init__(self, generations=(), transcript=None) -> None:
        self.generations = dict(generations)
        self.transcript = transcript
        self.discovery_calls = []
        self.transcript_calls = []

    def load_generation_for_meeting(self, meeting_id, profile):
        self.discovery_calls.append((meeting_id, profile))
        result = self.generations.get(profile)
        if isinstance(result, Exception):
            raise result
        return result

    def load_transcript(self, generation_id):
        self.transcript_calls.append(generation_id)
        if isinstance(self.transcript, Exception):
            raise self.transcript
        return self.transcript


class Reviews:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = []

    def load_review_for_generation(self, generation_id):
        self.calls.append(generation_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def service(storage_result, reader=None, reviews=None):
    return MeetingDetailService(
        Storage(storage_result), reader or Reader(), reviews or Reviews()
    )


def test_meeting_not_found_and_exact_single_load() -> None:
    storage = Storage(None)
    detail = MeetingDetailService(storage, Reader(), Reviews())
    with pytest.raises(MeetingDetailNotFoundError):
        detail.load(MEETING_ID)
    assert storage.calls == [MEETING_ID]


@pytest.mark.parametrize(
    ("error", "corrupt"),
    [
        (MeetingStorageDecodeError("bad"), True),
        (MeetingStorageFilesystemError("bad"), False),
        (MeetingStorageDatabaseError("bad"), False),
    ],
)
def test_storage_failures_remain_classifiable(error, corrupt) -> None:
    with pytest.raises(MeetingDetailLoadError) as captured:
        service(error).load(MEETING_ID)
    assert captured.value.corrupt is corrupt


def test_no_generation_and_missing_asset_are_valid() -> None:
    reader = Reader()
    snapshot = service(make_meeting(asset_exists=False), reader).load(MEETING_ID)
    assert snapshot.meeting.microphone is not None
    assert snapshot.meeting.microphone.asset_exists_at_load is False
    assert snapshot.transcript_state is MeetingDetailTranscriptState.NOT_AVAILABLE
    assert snapshot.final_generation is None
    assert snapshot.transcript is None
    assert (
        snapshot.speaker_review_state is MeetingDetailSpeakerReviewState.NOT_APPLICABLE
    )
    assert reader.discovery_calls == [(MEETING_ID, 2), (MEETING_ID, 1)]


@pytest.mark.parametrize("profile", [1, 2])
def test_one_generation_is_selected(profile) -> None:
    generation = make_generation(profile)
    transcript = make_transcript(generation)
    reader = Reader(((profile, generation),), transcript)
    snapshot = service(make_meeting(), reader).load(MEETING_ID)
    assert snapshot.final_generation is generation
    assert snapshot.transcript is transcript
    assert snapshot.transcript_state is MeetingDetailTranscriptState.AVAILABLE
    assert reader.discovery_calls == (
        [(MEETING_ID, 2)] if profile == 2 else [(MEETING_ID, 2), (MEETING_ID, 1)]
    )


@pytest.mark.parametrize(
    "v2_status",
    [
        FinalTranscriptionStatus.COMPLETED,
        FinalTranscriptionStatus.PARTIAL,
        FinalTranscriptionStatus.FAILED,
        FinalTranscriptionStatus.QUEUED,
        FinalTranscriptionStatus.IN_PROGRESS,
    ],
)
def test_v2_always_wins_over_completed_v1(v2_status) -> None:
    v2 = make_generation(2, v2_status)
    v1 = make_generation(1)
    transcript = (
        make_transcript(v2)
        if v2_status
        in (
            FinalTranscriptionStatus.COMPLETED,
            FinalTranscriptionStatus.PARTIAL,
        )
        else None
    )
    reader = Reader(((2, v2), (1, v1)), transcript)
    snapshot = service(make_meeting(), reader).load(MEETING_ID)
    assert snapshot.final_generation is v2
    assert snapshot.transcript is transcript
    assert reader.discovery_calls == [(MEETING_ID, 2)]


@pytest.mark.parametrize(
    ("error", "state"),
    [
        (FinalTranscriptionDecodeError("bad"), MeetingDetailTranscriptState.CORRUPT),
        (FinalTranscriptionStateError("bad"), MeetingDetailTranscriptState.LOAD_FAILED),
    ],
)
def test_transcript_failures_preserve_meeting_header(error, state) -> None:
    reader = Reader(((2, error),))
    snapshot = service(make_meeting(), reader).load(MEETING_ID)
    assert snapshot.meeting.session_id == MEETING_ID
    assert snapshot.transcript_state is state
    assert snapshot.transcript is None


@pytest.mark.parametrize(
    ("review_result", "state"),
    [
        (None, MeetingDetailSpeakerReviewState.ABSENT),
        (Mock(name="review"), MeetingDetailSpeakerReviewState.FRESH),
        (SpeakerReviewStaleError("bad"), MeetingDetailSpeakerReviewState.STALE),
        (SpeakerReviewDecodeError("bad"), MeetingDetailSpeakerReviewState.CORRUPT),
        (SpeakerReviewError("bad"), MeetingDetailSpeakerReviewState.LOAD_FAILED),
    ],
)
def test_v2_review_states_never_expose_failed_aggregate(review_result, state) -> None:
    generation = make_generation(2)
    reviews = Reviews(review_result)
    snapshot = service(
        make_meeting(), Reader(((2, generation),), make_transcript(generation)), reviews
    ).load(MEETING_ID)
    assert snapshot.speaker_review_state is state
    assert (snapshot.speaker_review is not None) is (
        state is MeetingDetailSpeakerReviewState.FRESH
    )
    assert reviews.calls == [generation.generation_id]


def test_v1_review_is_not_loaded() -> None:
    generation = make_generation(1)
    reviews = Reviews(Mock())
    snapshot = service(
        make_meeting(), Reader(((1, generation),), make_transcript(generation)), reviews
    ).load(MEETING_ID)
    assert (
        snapshot.speaker_review_state is MeetingDetailSpeakerReviewState.NOT_APPLICABLE
    )
    assert reviews.calls == []


def test_every_call_is_fresh_and_sees_changed_generation() -> None:
    reader = Reader()
    detail = service(make_meeting(), reader)
    assert detail.load(MEETING_ID).final_generation is None
    generation = make_generation(2, FinalTranscriptionStatus.QUEUED)
    reader.generations[2] = generation
    assert detail.load(MEETING_ID).final_generation is generation
    assert reader.discovery_calls == [
        (MEETING_ID, 2),
        (MEETING_ID, 1),
        (MEETING_ID, 2),
    ]
