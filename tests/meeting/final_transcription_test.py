"""Tests for the final-transcription domain and service."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Optional

import pytest

from buzz.meeting.final_transcription import (
    FinalTranscriptionConfig,
    FinalTranscriptionConfigError,
    FinalTranscriptionConflictError,
    FinalTranscriptionDecodeError,
    FinalTranscriptionEligibilityError,
    FinalTranscriptionError,
    FinalTranscriptionGeneration,
    FinalTranscriptionService,
    FinalTranscriptionStateError,
    FinalTranscriptionStatus,
    FinalTranscriptionTrackStatus,
    GenerationPersistenceRecord,
    SegmentPersistenceRecord,
    TrackPersistenceRecord,
    TrackTranscriptionInputSegment,
    TrackTranscriptionInputWord,
    TrackTranscriptionResult,
    WordPersistenceRecord,
    check_track_eligibility,
    decode_generation_status,
    decode_role,
    decode_track_status,
    derive_generation_status,
    encode_config,
    encode_datetime,
    encode_generation_status,
    encode_role,
    encode_track_status,
    is_terminal,
)
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
    MeetingTrackRole,
)
from buzz.meeting.meeting_recorder import MeetingRecorderState
from buzz.meeting.meeting_session import MeetingSessionState
from buzz.meeting.meeting_storage import (
    MeetingRemoteSourceKind,
    StoredMeeting,
    StoredMeetingAudioTrack,
    StoredMeetingTimingAnchor,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _make_stored_track(
    role: MeetingTrackRole,
    *,
    published: bool = True,
    sample_count: int = 160000,
    asset_exists: bool = True,
    sample_rate: int = 16000,
    anchors: tuple[StoredMeetingTimingAnchor, ...] = (
        StoredMeetingTimingAnchor(sample_end=16000, callback_arrival_offset_ns=0),
    ),
    complete: bool = True,
) -> StoredMeetingAudioTrack:
    from pathlib import Path, PurePosixPath

    return StoredMeetingAudioTrack(
        role=role,
        relative_path=PurePosixPath(
            "test/microphone.wav"
            if role is MeetingTrackRole.MICROPHONE
            else "test/remote.wav"
        ),
        path=Path(f"test/{role.name.lower()}.wav"),
        sample_rate=sample_rate,
        sample_count=sample_count,
        recording_state=MeetingRecorderState.STOPPED,
        published=published,
        complete=complete,
        timing_basis="host_callback_arrival",
        timing_anchors=anchors,
        errors=(),
        asset_exists_at_load=asset_exists,
    )


def _make_stored_meeting(
    *,
    state: MeetingSessionState = MeetingSessionState.COMPLETED,
    mic_published: bool = True,
    mic_samples: int = 160000,
    remote_published: bool = True,
    remote_samples: int = 160000,
    mic_anchors: tuple[StoredMeetingTimingAnchor, ...] = (
        StoredMeetingTimingAnchor(sample_end=16000, callback_arrival_offset_ns=0),
    ),
    remote_anchors: tuple[StoredMeetingTimingAnchor, ...] = (
        StoredMeetingTimingAnchor(
            sample_end=16000, callback_arrival_offset_ns=100_000_000
        ),
    ),
) -> StoredMeeting:
    return StoredMeeting(
        session_id=uuid.uuid4(),
        remote_source_kind=MeetingRemoteSourceKind.SYSTEM,
        state=state,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        duration_ns=3_600_000_000_000,
        audio_state=MeetingAudioTracksState.STOPPED,
        audio_outcome=MeetingAudioTracksOutcome.COMPLETE,
        microphone=_make_stored_track(
            MeetingTrackRole.MICROPHONE,
            published=mic_published,
            sample_count=mic_samples,
            anchors=mic_anchors,
        ),
        remote=_make_stored_track(
            MeetingTrackRole.REMOTE,
            published=remote_published,
            sample_count=remote_samples,
            anchors=remote_anchors,
        ),
    )


class FakeMeetingStorage:
    """In-memory meeting storage for tests."""

    def __init__(self) -> None:
        self._meetings: dict[uuid.UUID, StoredMeeting] = {}

    def add(self, meeting: StoredMeeting) -> None:
        self._meetings[meeting.session_id] = meeting

    def load(self, meeting_id: uuid.UUID) -> Optional[StoredMeeting]:
        return self._meetings.get(meeting_id)


class FakeRepository:
    """In-memory repository implementing MeetingTranscriptionRepository."""

    def __init__(self) -> None:
        self._generations: dict[str, GenerationPersistenceRecord] = {}
        self._tracks: dict[str, list[TrackPersistenceRecord]] = {}
        self._segments: dict[tuple[str, str], list[SegmentPersistenceRecord]] = {}
        self._words: dict[tuple[str, str], list[WordPersistenceRecord]] = {}
        self._fail_complete = False
        self._fail_create = False

    def create_generation(
        self,
        generation_id: str,
        meeting_id: str,
        config: FinalTranscriptionConfig,
        initial_status: str,
        time_created: str,
        time_completed: Optional[str],
        tracks: tuple[TrackPersistenceRecord, ...],
    ) -> None:
        if self._fail_create:
            raise RuntimeError("Simulated DB failure")
        config_fields = encode_config(config)
        self._generations[generation_id] = GenerationPersistenceRecord(
            id=generation_id,
            meeting_id=meeting_id,
            profile_version=config.profile_version,
            status=initial_status,
            **config_fields,
            error_message=None,
            time_created=time_created,
            time_started=None,
            time_completed=time_completed,
        )
        self._tracks[generation_id] = list(tracks)

    def find_generation_by_key(
        self,
        meeting_id: str,
        profile_version: int,
    ) -> Optional[GenerationPersistenceRecord]:
        for gen in self._generations.values():
            if gen.meeting_id == meeting_id and gen.profile_version == profile_version:
                return gen
        return None

    def load_generation(
        self,
        generation_id: str,
    ) -> Optional[GenerationPersistenceRecord]:
        return self._generations.get(generation_id)

    def load_tracks(
        self,
        generation_id: str,
    ) -> tuple[TrackPersistenceRecord, ...]:
        tracks = self._tracks.get(generation_id, [])
        return tuple(tracks)

    def load_segments(
        self,
        generation_id: str,
        role: str,
    ) -> tuple[SegmentPersistenceRecord, ...]:
        key = (generation_id, role)
        return tuple(self._segments.get(key, []))

    def load_words(
        self,
        generation_id: str,
    ) -> tuple[WordPersistenceRecord, ...]:
        return tuple(
            word
            for (stored_generation_id, _), words in self._words.items()
            if stored_generation_id == generation_id
            for word in words
        )

    def begin_track(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            if tr.role == role:
                tracks[i] = replace(
                    tr,
                    status=encode_track_status(
                        FinalTranscriptionTrackStatus.IN_PROGRESS
                    ),
                    time_started=now,
                )
                break
        gen = self._generations.get(generation_id)
        if gen is not None:
            self._generations[generation_id] = replace(
                gen,
                status=encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
                time_started=gen.time_started or now,
            )

    def complete_track(
        self,
        generation_id: str,
        role: str,
        segments: tuple[SegmentPersistenceRecord, ...],
        now: str,
        words: tuple[WordPersistenceRecord, ...] = (),
    ) -> None:
        if self._fail_complete:
            raise RuntimeError("Simulated segment insert failure")

        key = (generation_id, role)
        self._segments[key] = [
            replace(segment, generation_id=generation_id, role=role)
            for segment in segments
        ]
        self._words[key] = [
            replace(word, generation_id=generation_id, role=role) for word in words
        ]

        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            if tr.role == role:
                tracks[i] = replace(
                    tr,
                    status=encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                    time_completed=now,
                    segment_count=len(segments),
                    word_count=len(words),
                )
                break

        self._derive_generation_status(generation_id, now)

    def fail_track(
        self,
        generation_id: str,
        role: str,
        error_message: str,
        now: str,
    ) -> None:
        key = (generation_id, role)
        self._segments.pop(key, None)
        self._words.pop(key, None)

        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            if tr.role == role:
                tracks[i] = replace(
                    tr,
                    status=encode_track_status(FinalTranscriptionTrackStatus.FAILED),
                    error_message=error_message[:4096],
                    time_completed=now,
                    segment_count=0,
                    word_count=0,
                )
                break

        self._derive_generation_status(generation_id, now)

    def mark_track_ineligible(
        self,
        generation_id: str,
        role: str,
        now: str,
    ) -> None:
        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            if tr.role == role:
                tracks[i] = replace(
                    tr,
                    status=encode_track_status(
                        FinalTranscriptionTrackStatus.INELIGIBLE
                    ),
                    time_completed=now,
                )
                break
        self._derive_generation_status(generation_id, now)

    def update_generation_status(
        self,
        generation_id: str,
        now: str,
    ) -> None:
        self._derive_generation_status(generation_id, now)

    def reset_for_retry(
        self,
        generation_id: str,
        desired_track_statuses: dict[str, str],
        now: str,
    ) -> None:
        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            ts = decode_track_status(tr.status)
            if ts is FinalTranscriptionTrackStatus.COMPLETED:
                continue
            desired = desired_track_statuses.get(
                tr.role,
                encode_track_status(FinalTranscriptionTrackStatus.INELIGIBLE),
            )
            tracks[i] = replace(
                tr,
                status=desired,
                error_message=None,
                time_started=None,
                time_completed=None,
                segment_count=0,
                word_count=0,
            )
            self._segments.pop((generation_id, tr.role), None)
            self._words.pop((generation_id, tr.role), None)

        self._derive_generation_status(generation_id, now)
        gen = self._generations.get(generation_id)
        if gen is not None:
            new_status = decode_generation_status(gen.status)
            if new_status in (
                FinalTranscriptionStatus.QUEUED,
                FinalTranscriptionStatus.IN_PROGRESS,
            ):
                self._generations[generation_id] = replace(
                    gen, time_completed=None, error_message=None
                )
            else:
                self._generations[generation_id] = replace(gen, error_message=None)

    def load_recoverable_generations(
        self,
    ) -> tuple[GenerationPersistenceRecord, ...]:
        return tuple(
            gen
            for gen in self._generations.values()
            if gen.status
            in (
                encode_generation_status(FinalTranscriptionStatus.QUEUED),
                encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
            )
        )

    def reset_in_progress_tracks(
        self,
        generation_id: str,
    ) -> None:
        tracks = self._tracks.get(generation_id, [])
        for i, tr in enumerate(tracks):
            ts = decode_track_status(tr.status)
            if ts is FinalTranscriptionTrackStatus.IN_PROGRESS:
                tracks[i] = replace(
                    tr,
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                    word_count=0,
                )
                self._segments.pop((generation_id, tr.role), None)
                self._words.pop((generation_id, tr.role), None)
        self._derive_generation_status(generation_id, "2025-01-01T00:00:00+00:00")

    def _derive_generation_status(self, generation_id: str, now: str) -> None:
        from buzz.meeting.final_transcription import derive_generation_status

        gen = self._generations.get(generation_id)
        if gen is None:
            return
        tracks = self._tracks.get(generation_id, [])
        statuses = tuple(decode_track_status(tr.status) for tr in tracks)
        new_status = derive_generation_status(statuses)
        terminal = is_terminal(new_status)
        self._generations[generation_id] = replace(
            gen,
            status=encode_generation_status(new_status),
            time_completed=now if terminal else gen.time_completed,
        )


class FakeRunner:
    """Fake TranscriptionRunner returning predefined rich results."""

    def __init__(self) -> None:
        self._results: deque[TrackTranscriptionResult | Exception] = deque()
        self._calls: list[tuple[str, int, FinalTranscriptionConfig]] = []
        self._shutdown_called = False

    def enqueue_result(
        self,
        result: (
            TrackTranscriptionResult
            | tuple[TrackTranscriptionInputSegment, ...]
            | Exception
        ),
    ) -> None:
        self._results.append(result)

    def transcribe_track(
        self,
        audio_path: str,
        sample_rate: int,
        config: FinalTranscriptionConfig,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> TrackTranscriptionResult:
        self._calls.append((audio_path, sample_rate, config))
        if not self._results:
            raise RuntimeError("FakeRunner: no more results enqueued")
        result = self._results.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self) -> None:
        self._shutdown_called = True


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestFinalTranscriptionConfig:
    def test_default_valid(self) -> None:
        cfg = FinalTranscriptionConfig()
        assert cfg.profile_version == 1
        assert cfg.model_type == "FASTER_WHISPER"
        assert cfg.whisper_model_size == "TINY"

    def test_all_whisper_types(self) -> None:
        for mt in ("WHISPER", "WHISPER_CPP", "FASTER_WHISPER"):
            cfg = FinalTranscriptionConfig(model_type=mt, whisper_model_size="SMALL")
            assert cfg.model_type == mt

    def test_hugging_face_requires_id(self) -> None:
        cfg = FinalTranscriptionConfig(
            model_type="HUGGING_FACE",
            whisper_model_size=None,
            hugging_face_model_id="openai/whisper-large-v3",
        )
        assert cfg.hugging_face_model_id == "openai/whisper-large-v3"

    def test_rejects_openai_api(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(model_type="OPEN_AI_WHISPER_API")

    def test_rejects_unknown_profile(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError, match="profile_version"):
            FinalTranscriptionConfig(profile_version=99)

    def test_whisper_requires_size(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError, match="whisper_model_size"):
            FinalTranscriptionConfig(model_type="WHISPER", whisper_model_size=None)

    def test_whisper_rejects_hf_id(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(
                model_type="WHISPER",
                whisper_model_size="TINY",
                hugging_face_model_id="some/model",
            )

    def test_hf_rejects_size(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(
                model_type="HUGGING_FACE",
                whisper_model_size="TINY",
                hugging_face_model_id="some/model",
            )

    def test_hf_requires_nonempty_id(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(
                model_type="HUGGING_FACE",
                whisper_model_size=None,
                hugging_face_model_id="",
            )

    def test_rejects_unknown_size(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(model_type="WHISPER", whisper_model_size="HUGE")

    def test_rejects_empty_language(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError, match="language"):
            FinalTranscriptionConfig(language="")

    def test_none_language_ok(self) -> None:
        cfg = FinalTranscriptionConfig(language=None)
        assert cfg.language is None

    def test_nonempty_language_ok(self) -> None:
        cfg = FinalTranscriptionConfig(language="zh")
        assert cfg.language == "zh"

    def test_config_equality(self) -> None:
        a = FinalTranscriptionConfig(language="en")
        b = FinalTranscriptionConfig(language="en")
        assert a == b

    def test_config_inequality(self) -> None:
        a = FinalTranscriptionConfig(language="en")
        b = FinalTranscriptionConfig(language="zh")
        assert a != b


# ---------------------------------------------------------------------------
# Enum codec tests
# ---------------------------------------------------------------------------


class TestEnumCodecs:
    def test_generation_status_roundtrip(self) -> None:
        for status in FinalTranscriptionStatus:
            encoded = encode_generation_status(status)
            decoded = decode_generation_status(encoded)
            assert decoded is status

    def test_track_status_roundtrip(self) -> None:
        for status in FinalTranscriptionTrackStatus:
            encoded = encode_track_status(status)
            decoded = decode_track_status(encoded)
            assert decoded is status

    def test_role_roundtrip(self) -> None:
        for role in MeetingTrackRole:
            encoded = encode_role(role)
            decoded = decode_role(encoded)
            assert decoded is role

    def test_unknown_generation_status(self) -> None:
        from buzz.meeting.final_transcription import FinalTranscriptionDecodeError

        with pytest.raises(FinalTranscriptionDecodeError):
            decode_generation_status("UNKNOWN")

    def test_unknown_track_status(self) -> None:
        from buzz.meeting.final_transcription import FinalTranscriptionDecodeError

        with pytest.raises(FinalTranscriptionDecodeError):
            decode_track_status("UNKNOWN")

    def test_unknown_role(self) -> None:
        from buzz.meeting.final_transcription import FinalTranscriptionDecodeError

        with pytest.raises(FinalTranscriptionDecodeError):
            decode_role("SPEAKER")


# ---------------------------------------------------------------------------
# Generation status derivation
# ---------------------------------------------------------------------------


class TestDeriveGenerationStatus:
    def test_all_completed(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.COMPLETED,
                    FinalTranscriptionTrackStatus.COMPLETED,
                )
            )
            is FinalTranscriptionStatus.COMPLETED
        )

    def test_one_completed_one_failed(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.COMPLETED,
                    FinalTranscriptionTrackStatus.FAILED,
                )
            )
            is FinalTranscriptionStatus.PARTIAL
        )

    def test_one_completed_one_ineligible(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.COMPLETED,
                    FinalTranscriptionTrackStatus.INELIGIBLE,
                )
            )
            is FinalTranscriptionStatus.PARTIAL
        )

    def test_both_failed(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.FAILED,
                    FinalTranscriptionTrackStatus.FAILED,
                )
            )
            is FinalTranscriptionStatus.FAILED
        )

    def test_both_ineligible(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.INELIGIBLE,
                    FinalTranscriptionTrackStatus.INELIGIBLE,
                )
            )
            is FinalTranscriptionStatus.FAILED
        )

    def test_any_queued(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.COMPLETED,
                    FinalTranscriptionTrackStatus.QUEUED,
                )
            )
            is FinalTranscriptionStatus.QUEUED
        )

    def test_any_in_progress(self) -> None:
        assert (
            derive_generation_status(
                (
                    FinalTranscriptionTrackStatus.IN_PROGRESS,
                    FinalTranscriptionTrackStatus.QUEUED,
                )
            )
            is FinalTranscriptionStatus.IN_PROGRESS
        )


class TestIsTerminal:
    def test_terminal(self) -> None:
        assert is_terminal(FinalTranscriptionStatus.COMPLETED)
        assert is_terminal(FinalTranscriptionStatus.PARTIAL)
        assert is_terminal(FinalTranscriptionStatus.FAILED)

    def test_not_terminal(self) -> None:
        assert not is_terminal(FinalTranscriptionStatus.QUEUED)
        assert not is_terminal(FinalTranscriptionStatus.IN_PROGRESS)


# ---------------------------------------------------------------------------
# Track eligibility
# ---------------------------------------------------------------------------


class TestCheckTrackEligibility:
    def test_eligible_track(self) -> None:
        track = _make_stored_track(MeetingTrackRole.MICROPHONE)
        assert check_track_eligibility(track) is None

    def test_unpublished(self) -> None:
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, published=False)
        assert check_track_eligibility(track) is not None

    def test_zero_samples(self) -> None:
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, sample_count=0)
        assert check_track_eligibility(track) is not None

    def test_missing_asset(self) -> None:
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, asset_exists=False)
        assert check_track_eligibility(track) is not None

    def test_zero_anchors(self) -> None:
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, anchors=())
        assert check_track_eligibility(track) is not None

    def test_nonmonotonic_anchors(self) -> None:
        anchors = (
            StoredMeetingTimingAnchor(
                sample_end=16000, callback_arrival_offset_ns=2_000_000_000
            ),
            StoredMeetingTimingAnchor(
                sample_end=32000, callback_arrival_offset_ns=1_000_000_000
            ),
        )
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, anchors=anchors)
        assert check_track_eligibility(track) is not None

    def test_incomplete_still_eligible(self) -> None:
        """complete=False but published+nonempty is eligible."""
        track = _make_stored_track(MeetingTrackRole.MICROPHONE, complete=False)
        assert check_track_eligibility(track) is None


# ---------------------------------------------------------------------------
# Service: request
# ---------------------------------------------------------------------------


class TestServiceRequest:
    def _make_service(
        self,
        meeting: StoredMeeting,
        runner: Optional[FakeRunner] = None,
    ) -> tuple[FinalTranscriptionService, FakeRepository, FakeRunner]:
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = runner or FakeRunner()
        # Enqueue default success results for both tracks
        if not run._results:
            run.enqueue_result(
                (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="hello"),)
            )
            run.enqueue_result(
                (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="world"),)
            )
        service = FinalTranscriptionService(storage, repo, run)
        return service, repo, run

    def test_creates_generation(self) -> None:
        meeting = _make_stored_meeting()
        service, repo, _ = self._make_service(meeting)
        config = FinalTranscriptionConfig()
        gen = service.request(meeting.session_id, config)

        assert gen.status in (
            FinalTranscriptionStatus.QUEUED,
            FinalTranscriptionStatus.IN_PROGRESS,
        )
        assert gen.meeting_id == meeting.session_id
        assert len(gen.tracks) == 2

    def test_idempotent_same_config(self) -> None:
        meeting = _make_stored_meeting()
        service, repo, _ = self._make_service(meeting)
        config = FinalTranscriptionConfig()
        gen1 = service.request(meeting.session_id, config)
        gen2 = service.request(meeting.session_id, config)
        assert gen1.generation_id == gen2.generation_id

    def test_conflict_different_config(self) -> None:
        meeting = _make_stored_meeting()
        service, repo, _ = self._make_service(meeting)
        config1 = FinalTranscriptionConfig(language="en")
        service.request(meeting.session_id, config1)
        config2 = FinalTranscriptionConfig(language="zh")
        with pytest.raises(FinalTranscriptionConflictError):
            service.request(meeting.session_id, config2)

    def test_rejects_nonexistent_meeting(self) -> None:
        storage = FakeMeetingStorage()
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        with pytest.raises(FinalTranscriptionEligibilityError, match="not found"):
            service.request(uuid.uuid4(), FinalTranscriptionConfig())

    def test_rejects_noncompleted_meeting(self) -> None:
        meeting = _make_stored_meeting(state=MeetingSessionState.FAILED)
        service, _, _ = self._make_service(meeting)
        with pytest.raises(FinalTranscriptionEligibilityError, match="FAILED"):
            service.request(meeting.session_id, FinalTranscriptionConfig())


# ---------------------------------------------------------------------------
# Service: eligibility
# ---------------------------------------------------------------------------


class TestServiceEligibility:
    def test_both_tracks_eligible(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        assert gen.status is not FinalTranscriptionStatus.FAILED

    def test_one_track_ineligible(self) -> None:
        meeting = _make_stored_meeting(remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        roles = {t.role for t in gen.tracks}
        assert MeetingTrackRole.MICROPHONE in roles
        assert MeetingTrackRole.REMOTE in roles

    def test_both_tracks_ineligible(self) -> None:
        meeting = _make_stored_meeting(mic_published=False, remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        assert gen.status is FinalTranscriptionStatus.FAILED

    def test_audio_failed_but_usable_prefix(self) -> None:
        """Even if meeting was PARTIAL, published nonempty tracks are eligible."""
        meeting = _make_stored_meeting(
            mic_published=True,
            mic_samples=80000,
            remote_published=True,
            remote_samples=80000,
        )
        # Both tracks still eligible despite shorter recording
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        assert gen.status is not FinalTranscriptionStatus.FAILED


# ---------------------------------------------------------------------------
# Service: lifecycle
# ---------------------------------------------------------------------------


class TestServiceLifecycle:
    def test_mic_before_remote(self) -> None:
        """Microphone is transcribed before remote in stable role order."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        service.request(meeting.session_id, FinalTranscriptionConfig())
        assert len(run._calls) == 2

    def test_mic_failure_still_runs_remote(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(RuntimeError("ASR failed"))
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        service.request(meeting.session_id, FinalTranscriptionConfig())
        # Both tracks should have been attempted
        assert len(run._calls) == 2

    def test_both_complete_generation_completed(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        # Reload to get final state
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.COMPLETED

    def test_one_complete_one_failed_is_partial(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(RuntimeError("remote ASR failed"))
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.PARTIAL

    def test_empty_asr_result_valid(self) -> None:
        """Empty segment list is valid successful ASR."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(())  # empty
        run.enqueue_result(())  # empty
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.COMPLETED
        for track in gen.tracks:
            assert track.segment_count == 0

    def test_transcript_none_for_failed(self) -> None:
        meeting = _make_stored_meeting(mic_published=False, remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        transcript = service.load_transcript(gen.generation_id)
        assert transcript is None


# ---------------------------------------------------------------------------
# Service: transcript projection
# ---------------------------------------------------------------------------


class TestTranscriptProjection:
    def test_overlaps_preserved(self) -> None:
        """Overlapping segments from different tracks are preserved."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        # MIC: 0-2000ms
        run.enqueue_result(
            (
                TrackTranscriptionInputSegment(
                    start_ms=0, end_ms=2000, text="mic speech"
                ),
            )
        )
        # REMOTE: 1000-3000ms (overlaps with MIC)
        run.enqueue_result(
            (
                TrackTranscriptionInputSegment(
                    start_ms=1000, end_ms=3000, text="remote speech"
                ),
            )
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        transcript = service.load_transcript(gen.generation_id)
        assert transcript is not None
        assert len(transcript.segments) == 2
        # Verify overlap preserved: both segments exist with their timestamps
        segs = transcript.segments
        assert segs[0].text == "mic speech"
        assert segs[1].text == "remote speech"

    def test_stable_order(self) -> None:
        """Same-time segments ordered by role then ordinal."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        # Both at same time
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="remote"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        transcript = service.load_transcript(gen.generation_id)
        assert transcript is not None
        assert transcript.segments[0].source_role is MeetingTrackRole.MICROPHONE
        assert transcript.segments[1].source_role is MeetingTrackRole.REMOTE

    def test_merged_ordinals_sequential(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (
                TrackTranscriptionInputSegment(start_ms=0, end_ms=500, text="a"),
                TrackTranscriptionInputSegment(start_ms=600, end_ms=1000, text="b"),
            )
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=300, end_ms=700, text="c"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        transcript = service.load_transcript(gen.generation_id)
        assert transcript is not None
        for i, seg in enumerate(transcript.segments):
            assert seg.merged_ordinal == i


# ---------------------------------------------------------------------------
# Service: retry
# ---------------------------------------------------------------------------


class TestServiceRetry:
    def _setup_failed_generation(
        self,
    ) -> tuple[
        FinalTranscriptionService,
        FakeRepository,
        FakeRunner,
        FinalTranscriptionGeneration,
    ]:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(RuntimeError("mic failed"))
        run.enqueue_result(RuntimeError("remote failed"))
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.FAILED
        return service, repo, run, gen

    def test_retry_failed(self) -> None:
        service, repo, run, gen = self._setup_failed_generation()
        # Enqueue new success results
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        retried = service.retry(gen.generation_id)
        assert retried.generation_id == gen.generation_id
        retried = service.load_generation(gen.generation_id)
        assert retried is not None
        assert retried.status is FinalTranscriptionStatus.COMPLETED

    def test_retry_preserves_completed_tracks(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        # MIC succeeds, REMOTE fails
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(RuntimeError("remote failed"))
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.PARTIAL

        # Retry: only remote should be retried
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        retried = service.retry(gen.generation_id)
        retried = service.load_generation(gen.generation_id)
        assert retried is not None
        assert retried.status is FinalTranscriptionStatus.COMPLETED

    def test_retry_completed_rejected(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.COMPLETED
        with pytest.raises(FinalTranscriptionStateError):
            service.retry(gen.generation_id)

    def test_retry_no_new_generation_id(self) -> None:
        service, _, run, gen = self._setup_failed_generation()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        retried = service.retry(gen.generation_id)
        assert retried.generation_id == gen.generation_id


# ---------------------------------------------------------------------------
# Service: recovery
# ---------------------------------------------------------------------------


class TestServiceRecovery:
    def test_recover_queued(self) -> None:
        """Simulate a generation left QUEUED after crash (before scheduling)."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()

        # Manually create a QUEUED generation (simulating crash before run)
        config = FinalTranscriptionConfig()
        gen_id = str(uuid.uuid4())
        repo.create_generation(
            gen_id,
            str(meeting.session_id),
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="REMOTE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )

        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        recovered = service.recover_pending()
        assert len(recovered) == 1
        assert recovered[0] == uuid.UUID(gen_id)
        assert len(run._calls) == 2

    def test_recover_in_progress_resets_tracks(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        # Manually create an IN_PROGRESS generation
        config = FinalTranscriptionConfig()
        gen_id = str(uuid.uuid4())
        repo.create_generation(
            gen_id,
            str(meeting.session_id),
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(
                        FinalTranscriptionTrackStatus.IN_PROGRESS
                    ),
                    error_message=None,
                    time_started="2025-01-01T00:00:00+00:00",
                    time_completed=None,
                    segment_count=0,
                ),
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="REMOTE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )
        # Set generation to IN_PROGRESS
        repo._generations[gen_id] = replace(
            repo._generations[gen_id],
            status=encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
        )

        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        recovered = service.recover_pending()
        assert len(recovered) == 1
        # Both tracks should be scheduled
        assert len(run._calls) == 2

    def test_repeated_recover_no_duplicate(self) -> None:
        """Simulate a QUEUED generation and verify repeated recover
        doesn't duplicate in-memory queue entries."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()

        config = FinalTranscriptionConfig()
        gen_id = str(uuid.uuid4())
        repo.create_generation(
            gen_id,
            str(meeting.session_id),
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="REMOTE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.QUEUED),
                    error_message=None,
                    time_started=None,
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )

        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        # First recover schedules and runs
        service.recover_pending()
        assert len(run._calls) == 2
        # Second recover: generation now COMPLETED, nothing to do
        service.recover_pending()
        assert len(run._calls) == 2  # no additional calls

    def test_failed_not_auto_retried(self) -> None:
        meeting = _make_stored_meeting(mic_published=False, remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        service.request(meeting.session_id, FinalTranscriptionConfig())

        # New service for recovery
        run2 = FakeRunner()
        service2 = FinalTranscriptionService(storage, repo, run2)
        recovered = service2.recover_pending()
        assert len(recovered) == 0

    def test_completed_tracks_not_rerun_after_crash(self) -> None:
        """MIC completed, REMOTE pending → only REMOTE reruns."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()

        config = FinalTranscriptionConfig()
        gen_id = str(uuid.uuid4())
        repo.create_generation(
            gen_id,
            str(meeting.session_id),
            config,
            encode_generation_status(FinalTranscriptionStatus.QUEUED),
            "2025-01-01T00:00:00+00:00",
            None,
            (
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="MICROPHONE",
                    status=encode_track_status(FinalTranscriptionTrackStatus.COMPLETED),
                    error_message=None,
                    time_started="2025-01-01T00:00:00+00:00",
                    time_completed="2025-01-01T00:01:00+00:00",
                    segment_count=1,
                ),
                TrackPersistenceRecord(
                    generation_id=gen_id,
                    role="REMOTE",
                    status=encode_track_status(
                        FinalTranscriptionTrackStatus.IN_PROGRESS
                    ),
                    error_message=None,
                    time_started="2025-01-01T00:01:00+00:00",
                    time_completed=None,
                    segment_count=0,
                ),
            ),
        )
        repo._generations[gen_id] = replace(
            repo._generations[gen_id],
            status=encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
        )

        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        service.recover_pending()
        # Only REMOTE should be scheduled (1 call)
        assert len(run._calls) == 1


# ---------------------------------------------------------------------------
# Service: shutdown
# ---------------------------------------------------------------------------


class TestServiceShutdown:
    def test_shutdown_called(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        service.shutdown()
        assert run._shutdown_called

    def test_no_scheduling_after_shutdown(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        service.shutdown()
        # request after shutdown should create generation but not run ASR
        service.request(meeting.session_id, FinalTranscriptionConfig())
        assert len(run._calls) == 0


# ---------------------------------------------------------------------------
# Service: atomic completion
# ---------------------------------------------------------------------------


class TestAtomicCompletion:
    def test_segment_insert_failure_leaves_track_noncompleted(self) -> None:
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        repo._fail_complete = True
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        # MIC should fail to persist, stays IN_PROGRESS
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        # Generation should not be COMPLETED
        assert gen.status is not FinalTranscriptionStatus.COMPLETED


# ---------------------------------------------------------------------------
# Critical probes: H1 atomic creation
# ---------------------------------------------------------------------------


class TestH1AtomicCreation:
    def test_all_ineligible_creation_is_failed_immediately(self) -> None:
        """H1: After the single creation transaction, generation is FAILED
        when both tracks are INELIGIBLE.  Never observes QUEUED."""
        meeting = _make_stored_meeting(mic_published=False, remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())

        # Reload from repo — must be FAILED, never QUEUED
        gen_rec = repo.load_generation(str(gen.generation_id))
        assert gen_rec is not None
        assert gen_rec.status == encode_generation_status(
            FinalTranscriptionStatus.FAILED
        )
        assert gen_rec.time_completed is not None
        # Tracks both INELIGIBLE
        tracks = repo.load_tracks(gen_rec.id)
        for tr in tracks:
            assert tr.status == encode_track_status(
                FinalTranscriptionTrackStatus.INELIGIBLE
            )

    def test_one_eligible_one_ineligible_is_queued(self) -> None:
        """H1: MIC QUEUED + REMOTE INELIGIBLE → generation starts QUEUED.
        After synchronous execution, becomes PARTIAL."""
        meeting = _make_stored_meeting(remote_published=False)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_rec = repo.load_generation(str(gen.generation_id))
        assert gen_rec is not None
        # After synchronous execution: MIC COMPLETED + REMOTE INELIGIBLE = PARTIAL
        assert gen_rec.status == encode_generation_status(
            FinalTranscriptionStatus.PARTIAL
        )
        # time_completed set (terminal)
        assert gen_rec.time_completed is not None
        tracks = repo.load_tracks(gen_rec.id)
        mic = next(t for t in tracks if t.role == "MICROPHONE")
        rem = next(t for t in tracks if t.role == "REMOTE")
        assert mic.status == encode_track_status(
            FinalTranscriptionTrackStatus.COMPLETED
        )
        assert rem.status == encode_track_status(
            FinalTranscriptionTrackStatus.INELIGIBLE
        )

    def test_both_eligible_is_queued(self) -> None:
        """H1: Both QUEUED → generation QUEUED initially, then COMPLETED
        after synchronous execution."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="a"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="b"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_rec = repo.load_generation(str(gen.generation_id))
        assert gen_rec is not None
        # After synchronous execution: both COMPLETED
        assert gen_rec.status == encode_generation_status(
            FinalTranscriptionStatus.COMPLETED
        )

    def test_creation_rollback_on_repo_failure(self) -> None:
        """H1: If repo create fails, no generation/tracks are persisted."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        repo._fail_create = True
        run = FakeRunner()
        service = FinalTranscriptionService(storage, repo, run)
        with pytest.raises(RuntimeError, match="Simulated DB failure"):
            service.request(meeting.session_id, FinalTranscriptionConfig())
        assert len(repo._generations) == 0
        assert len(repo._tracks) == 0


# ---------------------------------------------------------------------------
# Critical probes: H2 atomic retry
# ---------------------------------------------------------------------------


class TestH2AtomicRetry:
    def test_ineligible_to_queued_when_asset_restored(self) -> None:
        """H2: INELIGIBLE → QUEUED when asset/timing is restored."""
        meeting_id = uuid.uuid4()
        # Initially remote is unpublished (ineligible)
        meeting = _make_stored_meeting(remote_published=False)
        meeting = replace(meeting, session_id=meeting_id)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        # MIC succeeds
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None
        assert gen.status is FinalTranscriptionStatus.PARTIAL
        # REMOTE was INELIGIBLE
        rem_track = next(t for t in gen.tracks if t.role is MeetingTrackRole.REMOTE)
        assert rem_track.status is FinalTranscriptionTrackStatus.INELIGIBLE

        # Now restore remote eligibility
        restored_meeting = _make_stored_meeting()  # both published
        restored_meeting = replace(restored_meeting, session_id=meeting_id)
        storage._meetings[meeting_id] = restored_meeting

        # Enqueue REMOTE result for retry
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )

        retried = service.retry(gen.generation_id)
        retried = service.load_generation(gen.generation_id)
        assert retried is not None
        assert retried.generation_id == gen.generation_id  # same ID

        mic_track = next(
            t for t in retried.tracks if t.role is MeetingTrackRole.MICROPHONE
        )
        rem_track = next(t for t in retried.tracks if t.role is MeetingTrackRole.REMOTE)

        # MIC preserved COMPLETED
        assert mic_track.status is FinalTranscriptionTrackStatus.COMPLETED
        # REMOTE now QUEUED (eligibility restored)
        assert rem_track.status in (
            FinalTranscriptionTrackStatus.QUEUED,
            FinalTranscriptionTrackStatus.COMPLETED,
        )

    def test_still_ineligible_retry_stays_ineligible(self) -> None:
        """H2: INELIGIBLE remains INELIGIBLE when still unusable."""
        meeting_id = uuid.uuid4()
        meeting = _make_stored_meeting(remote_published=False)
        meeting = replace(meeting, session_id=meeting_id)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None

        # Retry without restoring remote
        retried = service.retry(gen.generation_id)
        retried = service.load_generation(gen.generation_id)
        assert retried is not None

        rem_track = next(t for t in retried.tracks if t.role is MeetingTrackRole.REMOTE)
        assert rem_track.status is FinalTranscriptionTrackStatus.INELIGIBLE
        assert retried.status is FinalTranscriptionStatus.PARTIAL

    def test_completed_role_preserved_exactly(self) -> None:
        """H2: COMPLETED role's segments and metadata are untouched by retry."""
        meeting_id = uuid.uuid4()
        meeting = _make_stored_meeting(remote_published=False)
        meeting = replace(meeting, session_id=meeting_id)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None

        # Get MIC segments before retry
        mic_segs_before = repo.load_segments(str(gen.generation_id), "MICROPHONE")
        assert len(mic_segs_before) > 0

        # Retry (remote still ineligible)
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service.retry(gen.generation_id)

        # MIC segments must be identical
        mic_segs_after = repo.load_segments(str(gen.generation_id), "MICROPHONE")
        assert len(mic_segs_after) == len(mic_segs_before)
        for before, after in zip(mic_segs_before, mic_segs_after):
            assert before.text == after.text
            assert before.start_ns == after.start_ns
            assert before.end_ns == after.end_ns

    def test_retry_transaction_failure_preserves_old_state(self) -> None:
        """H2: If retry transaction fails, old aggregate is preserved."""
        meeting_id = uuid.uuid4()
        meeting = _make_stored_meeting(remote_published=False)
        meeting = replace(meeting, session_id=meeting_id)
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        service = FinalTranscriptionService(storage, repo, run)
        gen = service.request(meeting_id, FinalTranscriptionConfig())
        gen = service.load_generation(gen.generation_id)
        assert gen is not None

        # Snapshot state before retry
        tracks_before = repo.load_tracks(str(gen.generation_id))
        gen_status_before = gen.status

        # Make repo fail on retry
        def failing_reset(*args, **kwargs):
            raise RuntimeError("Simulated retry failure")

        repo.reset_for_retry = failing_reset  # type: ignore

        with pytest.raises(RuntimeError, match="Simulated retry failure"):
            service.retry(gen.generation_id)

        # State must be unchanged
        tracks_after = repo.load_tracks(str(gen.generation_id))
        assert len(tracks_after) == len(tracks_before)
        for before, after in zip(tracks_before, tracks_after):
            assert before.status == after.status
        gen_rec = repo.load_generation(str(gen.generation_id))
        assert gen_rec is not None
        assert gen_rec.status == encode_generation_status(gen_status_before)


# ---------------------------------------------------------------------------
# Critical probes: injected clock
# ---------------------------------------------------------------------------


class TestInjectedClock:
    def test_all_lifecycle_uses_injected_clock(self) -> None:
        """Verify creation, begin, completion, and failure timestamps
        all come from the injected clock."""
        fixed_time = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        call_count = [0]

        def fake_clock() -> datetime:
            call_count[0] += 1
            return fixed_time

        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        run = FakeRunner()
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="mic"),)
        )
        run.enqueue_result(
            (TrackTranscriptionInputSegment(start_ms=0, end_ms=1000, text="rem"),)
        )
        service = FinalTranscriptionService(storage, repo, run, clock=fake_clock)
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        assert call_count[0] > 0

        # Reload and check timestamps
        gen_rec = repo.load_generation(str(gen.generation_id))
        assert gen_rec is not None
        assert gen_rec.time_created == encode_datetime(fixed_time)

        # Check track timestamps
        tracks = repo.load_tracks(str(gen.generation_id))
        for tr in tracks:
            if tr.status == encode_track_status(
                FinalTranscriptionTrackStatus.COMPLETED
            ):
                assert tr.time_started == encode_datetime(fixed_time)
                assert tr.time_completed == encode_datetime(fixed_time)


# ---------------------------------------------------------------------------
# Timestamp codec
# ---------------------------------------------------------------------------


class TestTimestampCodec:
    def test_encode_aware_utc(self) -> None:
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        encoded = encode_datetime(dt)
        assert encoded is not None
        assert "2025-06-15" in encoded

    def test_encode_naive_raises(self) -> None:
        dt = datetime(2025, 6, 15, 12, 0, 0)
        with pytest.raises(FinalTranscriptionError):
            encode_datetime(dt)

    def test_encode_none_returns_none(self) -> None:
        assert encode_datetime(None) is None


# ---------------------------------------------------------------------------
# PR12 v2 rich-result contracts
# ---------------------------------------------------------------------------


def _v2_config(*, language: Optional[str] = None) -> FinalTranscriptionConfig:
    return FinalTranscriptionConfig(
        profile_version=2,
        model_type="FASTER_WHISPER",
        whisper_model_size="SMALL",
        language=language,
    )


def _rich_result(text: str) -> TrackTranscriptionResult:
    return TrackTranscriptionResult(
        segments=(
            TrackTranscriptionInputSegment(
                start_ms=0,
                end_ms=1000,
                text=text,
            ),
        ),
        words=(
            TrackTranscriptionInputWord(
                source_segment_ordinal=0,
                start_ms=100,
                end_ms=600,
                text=text,
            ),
        ),
    )


def _v2_service() -> (
    tuple[StoredMeeting, FinalTranscriptionService, FakeRepository, FakeRunner]
):
    meeting = _make_stored_meeting()
    storage = FakeMeetingStorage()
    storage.add(meeting)
    repository = FakeRepository()
    runner = FakeRunner()
    return (
        meeting,
        FinalTranscriptionService(storage, repository, runner),
        repository,
        runner,
    )


class TestV2GenerationIdentity:
    def test_v1_and_v2_coexist(self) -> None:
        meeting, service, repository, runner = _v2_service()
        runner.enqueue_result((TrackTranscriptionInputSegment(0, 1000, "v1 mic"),))
        runner.enqueue_result((TrackTranscriptionInputSegment(0, 1000, "v1 remote"),))
        v1 = service.request(meeting.session_id, FinalTranscriptionConfig())
        runner.enqueue_result(_rich_result("v2 mic"))
        runner.enqueue_result(_rich_result("v2 remote"))
        v2 = service.request(meeting.session_id, _v2_config())

        assert v1.generation_id != v2.generation_id
        assert repository.find_generation_by_key(str(meeting.session_id), 1)
        assert repository.find_generation_by_key(str(meeting.session_id), 2)

    def test_same_v2_is_idempotent_and_different_config_conflicts(self) -> None:
        meeting, service, _, runner = _v2_service()
        runner.enqueue_result(_rich_result("mic"))
        runner.enqueue_result(_rich_result("remote"))
        config = _v2_config(language=None)

        first = service.request(meeting.session_id, config)
        second = service.request(meeting.session_id, config)

        assert second.generation_id == first.generation_id
        with pytest.raises(FinalTranscriptionConflictError):
            service.request(meeting.session_id, _v2_config(language="zh"))


class TestV2WordMapping:
    def test_words_are_durable_ordered_and_keep_negative_overlap(self) -> None:
        meeting, service, _, runner = _v2_service()
        overlapping = TrackTranscriptionResult(
            segments=(TrackTranscriptionInputSegment(0, 1000, "phrase"),),
            words=(
                TrackTranscriptionInputWord(0, 100, 600, "one"),
                TrackTranscriptionInputWord(0, 500, 900, "two"),
            ),
        )
        runner.enqueue_result(overlapping)
        runner.enqueue_result(_rich_result("remote"))

        generation = service.request(meeting.session_id, _v2_config())
        words = service.load_words(generation.generation_id)

        mic_words = [
            word for word in words if word.source_role is MeetingTrackRole.MICROPHONE
        ]
        assert [word.source_word_ordinal for word in mic_words] == [0, 1]
        assert [word.source_segment_ordinal for word in mic_words] == [0, 0]
        assert mic_words[0].start_ns == -900_000_000
        assert mic_words[0].end_ns > mic_words[1].start_ns
        assert [word.text for word in mic_words] == ["one", "two"]

    def test_parent_crossing_is_allowed(self) -> None:
        meeting, service, _, runner = _v2_service()
        crossing = TrackTranscriptionResult(
            segments=(TrackTranscriptionInputSegment(100, 900, "phrase"),),
            words=(TrackTranscriptionInputWord(0, 50, 950, "crossing"),),
        )
        runner.enqueue_result(crossing)
        runner.enqueue_result(_rich_result("remote"))

        generation = service.request(meeting.session_id, _v2_config())

        assert len(service.load_words(generation.generation_id)) == 2


class TestV2Completeness:
    def test_empty_phrase_and_words_is_successful(self) -> None:
        meeting, service, _, runner = _v2_service()
        empty = TrackTranscriptionResult(segments=(), words=())
        runner.enqueue_result(empty)
        runner.enqueue_result(empty)

        generation = service.request(meeting.session_id, _v2_config())
        loaded = service.load_generation(generation.generation_id)

        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.COMPLETED
        assert service.load_words(generation.generation_id) == ()

    @pytest.mark.parametrize(
        "bad_result",
        [
            TrackTranscriptionResult(
                segments=(TrackTranscriptionInputSegment(0, 100, "phrase"),),
                words=(),
            ),
            TrackTranscriptionResult(
                segments=(TrackTranscriptionInputSegment(0, 100, "phrase"),),
                words=(TrackTranscriptionInputWord(1, 0, 100, "orphan"),),
            ),
            TrackTranscriptionResult(
                segments=(TrackTranscriptionInputSegment(0, 100, "phrase"),),
                words=(TrackTranscriptionInputWord(0, 100, 99, "bad"),),
            ),
        ],
    )
    def test_invalid_v2_result_fails_track(
        self, bad_result: TrackTranscriptionResult
    ) -> None:
        meeting, service, _, runner = _v2_service()
        runner.enqueue_result(bad_result)
        runner.enqueue_result(_rich_result("remote"))

        generation = service.request(meeting.session_id, _v2_config())
        loaded = service.load_generation(generation.generation_id)

        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.PARTIAL
        assert loaded.tracks[0].status is FinalTranscriptionTrackStatus.FAILED

    def test_v1_rejects_runner_words(self) -> None:
        meeting, service, _, runner = _v2_service()
        runner.enqueue_result(_rich_result("unexpected"))
        runner.enqueue_result((TrackTranscriptionInputSegment(0, 100, "remote"),))

        generation = service.request(meeting.session_id, FinalTranscriptionConfig())
        loaded = service.load_generation(generation.generation_id)

        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.PARTIAL
        assert loaded.tracks[0].status is FinalTranscriptionTrackStatus.FAILED


class TestV2WordCorruption:
    @staticmethod
    def _word(
        generation_id: str,
        *,
        role: str = "MICROPHONE",
        ordinal: int = 0,
        segment_ordinal: int = 0,
    ) -> WordPersistenceRecord:
        return WordPersistenceRecord(
            generation_id=generation_id,
            role=role,
            ordinal=ordinal,
            segment_ordinal=segment_ordinal,
            local_start_ms=0,
            local_end_ms=100,
            start_ns=-1_000_000_000,
            end_ns=-900_000_000,
            text="word",
        )

    def test_v1_word_row_is_corruption(self) -> None:
        meeting, service, repository, runner = _v2_service()
        runner.enqueue_result(())
        runner.enqueue_result(())
        generation = service.request(meeting.session_id, FinalTranscriptionConfig())
        generation_id = str(generation.generation_id)
        repository._words[(generation_id, "MICROPHONE")] = [self._word(generation_id)]

        with pytest.raises(FinalTranscriptionDecodeError, match="version 1"):
            service.load_words(generation.generation_id)

    @pytest.mark.parametrize(
        "role,ordinal,segment_ordinal",
        [
            ("UNKNOWN", 0, 0),
            ("MICROPHONE", 2, 0),
            ("MICROPHONE", 0, 9),
        ],
    )
    def test_corrupt_role_gap_or_parent_is_rejected(
        self, role: str, ordinal: int, segment_ordinal: int
    ) -> None:
        meeting, service, repository, runner = _v2_service()
        runner.enqueue_result(_rich_result("mic"))
        runner.enqueue_result(_rich_result("remote"))
        generation = service.request(meeting.session_id, _v2_config())
        generation_id = str(generation.generation_id)
        repository._words[(generation_id, "MICROPHONE")] = [
            self._word(
                generation_id,
                role=role,
                ordinal=ordinal,
                segment_ordinal=segment_ordinal,
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_words(generation.generation_id)

    def test_word_for_missing_generation_is_rejected(self) -> None:
        _, service, repository, _ = _v2_service()
        missing = uuid.uuid4()
        repository._words[(str(missing), "MICROPHONE")] = [self._word(str(missing))]

        with pytest.raises(FinalTranscriptionDecodeError, match="missing generation"):
            service.load_words(missing)


class TestV2RetryRecovery:
    def test_retry_preserves_completed_role_phrase_and_words(self) -> None:
        meeting, service, repository, runner = _v2_service()
        runner.enqueue_result(_rich_result("mic"))
        runner.enqueue_result(RuntimeError("remote failed"))
        generation = service.request(meeting.session_id, _v2_config())
        mic_words_before = tuple(
            word
            for word in service.load_words(generation.generation_id)
            if word.source_role is MeetingTrackRole.MICROPHONE
        )
        mic_segments_before = repository.load_segments(
            str(generation.generation_id), "MICROPHONE"
        )
        runner.enqueue_result(_rich_result("remote"))

        service.retry(generation.generation_id)

        mic_words_after = tuple(
            word
            for word in service.load_words(generation.generation_id)
            if word.source_role is MeetingTrackRole.MICROPHONE
        )
        assert mic_words_after == mic_words_before
        assert (
            repository.load_segments(str(generation.generation_id), "MICROPHONE")
            == mic_segments_before
        )
        assert len(runner._calls) == 3

    def test_recovery_does_not_rerun_completed_v2_role(self) -> None:
        meeting, service, repository, runner = _v2_service()
        runner.enqueue_result(_rich_result("mic"))
        runner.enqueue_result(RuntimeError("remote interrupted"))
        generation = service.request(meeting.session_id, _v2_config())
        generation_id = str(generation.generation_id)
        mic_words_before = tuple(
            word
            for word in service.load_words(generation.generation_id)
            if word.source_role is MeetingTrackRole.MICROPHONE
        )
        tracks = repository._tracks[generation_id]
        for index, track in enumerate(tracks):
            if track.role == "REMOTE":
                tracks[index] = replace(
                    track,
                    status=encode_track_status(
                        FinalTranscriptionTrackStatus.IN_PROGRESS
                    ),
                    time_completed=None,
                )
        repository._generations[generation_id] = replace(
            repository._generations[generation_id],
            status=encode_generation_status(FinalTranscriptionStatus.IN_PROGRESS),
            time_completed=None,
        )
        recovered_runner = FakeRunner()
        recovered_runner.enqueue_result(_rich_result("remote"))
        recovered_service = FinalTranscriptionService(
            service._meeting_storage, repository, recovered_runner
        )

        assert recovered_service.recover_pending() == (generation.generation_id,)

        assert len(recovered_runner._calls) == 1
        mic_words_after = tuple(
            word
            for word in recovered_service.load_words(generation.generation_id)
            if word.source_role is MeetingTrackRole.MICROPHONE
        )
        assert mic_words_after == mic_words_before


# ---------------------------------------------------------------------------
# H3: Legacy v1 runner ABI compatibility
# ---------------------------------------------------------------------------


class TestV1LegacyRunner:
    def test_strict_tuple_runner_completes_v1(self) -> None:
        """A strict PR11 runner returning a raw tuple must complete v1."""

        class LegacyV1Runner:
            def transcribe_track(
                self, audio_path, sample_rate, config, on_progress=None
            ):
                return (
                    TrackTranscriptionInputSegment(
                        start_ms=0, end_ms=1000, text="legacy phrase"
                    ),
                )

            def shutdown(self):
                pass

        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        repo = FakeRepository()
        service = FinalTranscriptionService(storage, repo, LegacyV1Runner())

        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        loaded = service.load_generation(gen.generation_id)

        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.COMPLETED
        assert service.load_words(gen.generation_id) == ()
        mic_track = next(
            t for t in loaded.tracks if t.role is MeetingTrackRole.MICROPHONE
        )
        assert mic_track.segment_count >= 1


# ---------------------------------------------------------------------------
# H1 / M3: v1 word_count invariant and corruption detection
# ---------------------------------------------------------------------------


class TestV1WordCountInvariant:
    def test_v1_completed_has_word_count_zero(self) -> None:
        meeting, service, repo, runner = _v2_service()
        runner.enqueue_result(())
        runner.enqueue_result(())
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        tracks = repo.load_tracks(str(gen.generation_id))
        for tr in tracks:
            assert tr.word_count == 0

    def test_v1_illegal_word_row_rejected_by_load_generation(self) -> None:
        meeting, service, repo, runner = _v2_service()
        runner.enqueue_result(())
        runner.enqueue_result(())
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        repo._words[(gen_id, "MICROPHONE")] = [
            WordPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                segment_ordinal=0,
                local_start_ms=0,
                local_end_ms=100,
                start_ns=-1_000_000_000,
                end_ns=-900_000_000,
                text="illegal",
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError, match="version 1"):
            service.load_generation(gen.generation_id)

    def test_v1_illegal_word_row_rejected_by_load_transcript(self) -> None:
        meeting, service, repo, runner = _v2_service()
        runner.enqueue_result(())
        runner.enqueue_result(())
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        repo._words[(gen_id, "MICROPHONE")] = [
            WordPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                segment_ordinal=0,
                local_start_ms=0,
                local_end_ms=100,
                start_ns=-1_000_000_000,
                end_ns=-900_000_000,
                text="illegal",
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError, match="version 1"):
            service.load_transcript(gen.generation_id)

    def test_v1_illegal_word_row_rejected_by_load_words(self) -> None:
        meeting, service, repo, runner = _v2_service()
        runner.enqueue_result(())
        runner.enqueue_result(())
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        repo._words[(gen_id, "MICROPHONE")] = [
            WordPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                segment_ordinal=0,
                local_start_ms=0,
                local_end_ms=100,
                start_ns=-1_000_000_000,
                end_ns=-900_000_000,
                text="illegal",
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError, match="version 1"):
            service.load_words(gen.generation_id)


# ---------------------------------------------------------------------------
# H1: v2 word count corruption probes
# ---------------------------------------------------------------------------


class TestV2WordCountIntegrity:
    @staticmethod
    def _corrupt_word(
        gen_id: str, *, ordinal: int, segment_ordinal: int = 0
    ) -> WordPersistenceRecord:
        return WordPersistenceRecord(
            generation_id=gen_id,
            role="MICROPHONE",
            ordinal=ordinal,
            segment_ordinal=segment_ordinal,
            local_start_ms=0,
            local_end_ms=100,
            start_ns=-1_000_000_000,
            end_ns=-900_000_000,
            text=f"word{ordinal}",
        )

    def _create_valid_v2(self, repo) -> tuple[str, FinalTranscriptionService]:
        """Create a valid v2 generation with 4 words on MICROPHONE."""
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        service = FinalTranscriptionService(storage, repo, runner)

        # 4 words all in segment 0
        words = tuple(
            TrackTranscriptionInputWord(
                source_segment_ordinal=0,
                start_ms=i * 100,
                end_ms=(i + 1) * 100,
                text=f"word{i}",
            )
            for i in range(4)
        )
        result = TrackTranscriptionResult(
            segments=(TrackTranscriptionInputSegment(0, 500, "phrase"),),
            words=words,
        )
        runner.enqueue_result(result)
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))

        gen = service.request(meeting.session_id, _v2_config())
        gen_id = str(gen.generation_id)
        assert service.load_generation(gen.generation_id) is not None
        return gen_id, service

    def test_tail_word_deletion_detected(self) -> None:
        """H1: deleting last word ordinal → DecodeError."""
        repo = FakeRepository()
        gen_id, service = self._create_valid_v2(repo)
        # Remove ordinal 3, leave 0,1,2
        repo._words[(gen_id, "MICROPHONE")] = [
            self._corrupt_word(gen_id, ordinal=o) for o in range(3)
        ]
        # Track still says word_count=4; clear cache to force re-validation
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(uuid.UUID(gen_id))

    def test_middle_word_deletion_detected(self) -> None:
        """H1: deleting middle word ordinal → DecodeError."""
        repo = FakeRepository()
        gen_id, service = self._create_valid_v2(repo)
        # Remove ordinal 1, leave 0,2,3
        repo._words[(gen_id, "MICROPHONE")] = [
            self._corrupt_word(gen_id, ordinal=o) for o in (0, 2, 3)
        ]
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(uuid.UUID(gen_id))

    def test_extra_word_without_count_update_detected(self) -> None:
        """H1: inserting extra word without updating word_count → DecodeError."""
        repo = FakeRepository()
        gen_id, service = self._create_valid_v2(repo)
        # Add ordinal 4 without changing track.word_count
        repo._words[(gen_id, "MICROPHONE")] = [
            self._corrupt_word(gen_id, ordinal=o) for o in range(5)
        ]
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(uuid.UUID(gen_id))

    def test_empty_v2_transcript_valid(self) -> None:
        """H1: segment_count=0, word_count=0, zero phrases/words → valid."""
        repo = FakeRepository()
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))
        service = FinalTranscriptionService(storage, repo, runner)
        gen = service.request(meeting.session_id, _v2_config())
        loaded = service.load_generation(gen.generation_id)
        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.COMPLETED
        assert service.load_words(gen.generation_id) == ()


# ---------------------------------------------------------------------------
# H2: Per-phrase word coverage
# ---------------------------------------------------------------------------


class TestV2PhraseCoverage:
    def test_nonempty_phrase_without_words_rejected_before_persist(self) -> None:
        """H2: v2 result with phrase but words only for another phrase → FAIL."""
        meeting, service, _, runner = _v2_service()
        # Phrase 0 = "hello", Phrase 1 = "world"
        # Words only reference phrase 0
        bad_result = TrackTranscriptionResult(
            segments=(
                TrackTranscriptionInputSegment(0, 100, "hello"),
                TrackTranscriptionInputSegment(200, 300, "world"),
            ),
            words=(TrackTranscriptionInputWord(0, 10, 90, "hello"),),
        )
        runner.enqueue_result(bad_result)
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))

        gen = service.request(meeting.session_id, _v2_config())
        loaded = service.load_generation(gen.generation_id)
        assert loaded is not None
        mic = next(t for t in loaded.tracks if t.role is MeetingTrackRole.MICROPHONE)
        assert mic.status is FinalTranscriptionTrackStatus.FAILED

    def test_all_nonempty_phrases_covered_accepted(self) -> None:
        """H2: v2 result where every nonempty phrase has words → succeeds."""
        meeting, service, _, runner = _v2_service()
        good_result = TrackTranscriptionResult(
            segments=(
                TrackTranscriptionInputSegment(0, 100, "hello"),
                TrackTranscriptionInputSegment(200, 300, "world"),
            ),
            words=(
                TrackTranscriptionInputWord(0, 10, 90, "hello"),
                TrackTranscriptionInputWord(1, 210, 290, "world"),
            ),
        )
        runner.enqueue_result(good_result)
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))

        gen = service.request(meeting.session_id, _v2_config())
        loaded = service.load_generation(gen.generation_id)
        assert loaded is not None
        assert loaded.status is FinalTranscriptionStatus.COMPLETED

    def test_coverage_corruption_detected_on_load(self) -> None:
        """H2: start valid, then remove words for phrase 1 → load rejects."""
        meeting, service, repo, runner = _v2_service()
        good_result = TrackTranscriptionResult(
            segments=(
                TrackTranscriptionInputSegment(0, 100, "hello"),
                TrackTranscriptionInputSegment(200, 300, "world"),
            ),
            words=(
                TrackTranscriptionInputWord(0, 10, 90, "hello"),
                TrackTranscriptionInputWord(1, 210, 290, "world"),
            ),
        )
        runner.enqueue_result(good_result)
        runner.enqueue_result(TrackTranscriptionResult(segments=(), words=()))

        gen = service.request(meeting.session_id, _v2_config())
        gen_id = str(gen.generation_id)
        # Valid initially
        assert service.load_generation(gen.generation_id) is not None

        # Remove all words for phrase 1, keep phrase 0's word
        repo._words[(gen_id, "MICROPHONE")] = [
            WordPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                segment_ordinal=0,
                local_start_ms=10,
                local_end_ms=90,
                start_ns=-900_000_000,
                end_ns=-100_000_000,
                text="hello",
            )
        ]
        # word_count still 2 but only 1 word → count mismatch
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(gen.generation_id)


# ---------------------------------------------------------------------------
# M2: Phrase tail integrity
# ---------------------------------------------------------------------------


class TestPhraseCountIntegrity:
    def test_phrase_tail_deletion_detected(self) -> None:
        """M2: segment_count=3, delete ordinal 2 → DecodeError."""
        repo = FakeRepository()
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        runner.enqueue_result(())
        runner.enqueue_result(())
        service = FinalTranscriptionService(storage, repo, runner)

        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        # Manually set segment_count=3 and insert 3 segments
        tracks = repo._tracks[gen_id]
        for i, tr in enumerate(tracks):
            if tr.role == "MICROPHONE":
                tracks[i] = replace(tr, segment_count=3)
        repo._segments[(gen_id, "MICROPHONE")] = [
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=o,
                local_start_ms=o * 1000,
                local_end_ms=(o + 1) * 1000,
                start_ns=o * 1_000_000_000,
                end_ns=(o + 1) * 1_000_000_000,
                text=f"seg{o}",
            )
            for o in range(3)
        ]
        # Valid initially
        assert service.load_generation(gen.generation_id) is not None

        # Delete ordinal 2, leaving 0,1
        repo._segments[(gen_id, "MICROPHONE")] = [
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=o,
                local_start_ms=o * 1000,
                local_end_ms=(o + 1) * 1000,
                start_ns=o * 1_000_000_000,
                end_ns=(o + 1) * 1_000_000_000,
                text=f"seg{o}",
            )
            for o in range(2)
        ]

        with pytest.raises(FinalTranscriptionDecodeError, match="Phrase count"):
            service.load_generation(gen.generation_id)

    def test_phrase_middle_deletion_detected(self) -> None:
        """M2: segment_count=3, delete ordinal 1 leaving 0,2 → DecodeError."""
        repo = FakeRepository()
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        runner.enqueue_result(())
        runner.enqueue_result(())
        service = FinalTranscriptionService(storage, repo, runner)

        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        tracks = repo._tracks[gen_id]
        for i, tr in enumerate(tracks):
            if tr.role == "MICROPHONE":
                tracks[i] = replace(tr, segment_count=3)
        repo._segments[(gen_id, "MICROPHONE")] = [
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=o,
                local_start_ms=o * 1000,
                local_end_ms=(o + 1) * 1000,
                start_ns=o * 1_000_000_000,
                end_ns=(o + 1) * 1_000_000_000,
                text=f"seg{o}",
            )
            for o in (0, 2)
        ]

        with pytest.raises(FinalTranscriptionDecodeError, match="Phrase"):
            service.load_generation(gen.generation_id)


# ---------------------------------------------------------------------------
# M1: v2 must require explicit model size
# ---------------------------------------------------------------------------


class TestV2ExplicitModelSize:
    def test_v1_default_still_tiny(self) -> None:
        cfg = FinalTranscriptionConfig()
        assert cfg.profile_version == 1
        assert cfg.whisper_model_size == "TINY"

    def test_v1_whisper_omitted_size_still_tiny(self) -> None:
        cfg = FinalTranscriptionConfig(model_type="WHISPER")
        assert cfg.whisper_model_size == "TINY"

    def test_v2_whisper_omitted_size_rejected(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError, match="explicit"):
            FinalTranscriptionConfig(profile_version=2, model_type="WHISPER")

    def test_v2_faster_omitted_size_rejected(self) -> None:
        with pytest.raises(FinalTranscriptionConfigError, match="explicit"):
            FinalTranscriptionConfig(profile_version=2, model_type="FASTER_WHISPER")

    def test_v2_explicit_tiny_accepted(self) -> None:
        cfg = FinalTranscriptionConfig(
            profile_version=2,
            model_type="WHISPER",
            whisper_model_size="TINY",
        )
        assert cfg.whisper_model_size == "TINY"

    def test_v1_hf_omitted_size_preserved(self) -> None:
        """v1 HUGGING_FACE with omitted whisper_model_size → error (same as before)."""
        with pytest.raises(FinalTranscriptionConfigError):
            FinalTranscriptionConfig(
                model_type="HUGGING_FACE",
                hugging_face_model_id="org/model",
            )

    def test_v1_hf_explicit_none_preserved(self) -> None:
        cfg = FinalTranscriptionConfig(
            model_type="HUGGING_FACE",
            whisper_model_size=None,
            hugging_face_model_id="org/model",
        )
        assert cfg.whisper_model_size is None


# ---------------------------------------------------------------------------
# M3: Aggregate corruption consistency across all load APIs
# ---------------------------------------------------------------------------


class TestAggregateValidatorConsistency:
    def test_word_tail_deletion_rejected_by_all_load_apis(self) -> None:
        """M3: word corruption → load_generation, load_transcript, load_words
        all reject."""
        repo = FakeRepository()
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        service = FinalTranscriptionService(storage, repo, runner)

        # v1 empty result
        runner.enqueue_result(())
        runner.enqueue_result(())
        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        # Valid
        assert service.load_generation(gen.generation_id) is not None

        # Inject v1 word corruption
        repo._words[(gen_id, "MICROPHONE")] = [
            WordPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                segment_ordinal=0,
                local_start_ms=0,
                local_end_ms=100,
                start_ns=-1_000_000_000,
                end_ns=-900_000_000,
                text="illegal",
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(gen.generation_id)
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_transcript(gen.generation_id)
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_words(gen.generation_id)

    def test_phrase_tail_deletion_rejected_by_all_load_apis(self) -> None:
        """M3: phrase corruption → all load APIs reject."""
        repo = FakeRepository()
        meeting = _make_stored_meeting()
        storage = FakeMeetingStorage()
        storage.add(meeting)
        runner = FakeRunner()
        runner.enqueue_result(())
        runner.enqueue_result(())
        service = FinalTranscriptionService(storage, repo, runner)

        gen = service.request(meeting.session_id, FinalTranscriptionConfig())
        gen_id = str(gen.generation_id)
        # Set segment_count=2, insert 1 segment
        tracks = repo._tracks[gen_id]
        for i, tr in enumerate(tracks):
            if tr.role == "MICROPHONE":
                tracks[i] = replace(tr, segment_count=2)
        repo._segments[(gen_id, "MICROPHONE")] = [
            SegmentPersistenceRecord(
                generation_id=gen_id,
                role="MICROPHONE",
                ordinal=0,
                local_start_ms=0,
                local_end_ms=1000,
                start_ns=0,
                end_ns=1_000_000_000,
                text="only_one",
            )
        ]

        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_generation(gen.generation_id)
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_transcript(gen.generation_id)
        with pytest.raises(FinalTranscriptionDecodeError):
            service.load_words(gen.generation_id)
