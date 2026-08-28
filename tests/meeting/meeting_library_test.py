from __future__ import annotations

import uuid
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
)
from buzz.meeting.meeting_library import (
    MeetingLibraryDecodeError,
    MeetingLibraryEntry,
    MeetingLibraryRecord,
    MeetingLibraryService,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionState,
)


class FakeRepository:
    def __init__(self, records: tuple[MeetingLibraryRecord, ...] = ()) -> None:
        self.records = records
        self.calls = 0

    def list_meetings(self) -> tuple[MeetingLibraryRecord, ...]:
        self.calls += 1
        return self.records


def make_record(
    session_id: str = "00000000-0000-0000-0000-000000000001",
    **overrides,
) -> MeetingLibraryRecord:
    values = {
        "session_id": session_id,
        "remote_source_kind": "SYSTEM",
        "session_state": "COMPLETED",
        "created_at": "2025-01-01T00:00:00+00:00",
        "started_at": "2025-01-01T00:01:00+00:00",
        "ended_at": "2025-01-01T00:02:00+00:00",
        "duration_ns": 60_000_000_000,
        "audio_state": "STOPPED",
        "audio_outcome": "COMPLETE",
    }
    values.update(overrides)
    return MeetingLibraryRecord(**values)


def decode(record: MeetingLibraryRecord) -> MeetingLibraryEntry:
    return MeetingLibraryService(FakeRepository((record,))).list_meetings()[0]


def test_empty_repository_returns_tuple() -> None:
    assert MeetingLibraryService(FakeRepository()).list_meetings() == ()


def test_one_record_decodes_every_field() -> None:
    entry = decode(make_record())
    assert entry == MeetingLibraryEntry(
        session_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        remote_source_kind=MeetingRemoteSourceKind.SYSTEM,
        session_state=MeetingSessionState.COMPLETED,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        started_at=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, 0, 2, tzinfo=timezone.utc),
        duration_ns=60_000_000_000,
        audio_state=MeetingAudioTracksState.STOPPED,
        audio_outcome=MeetingAudioTracksOutcome.COMPLETE,
    )


def test_multiple_records_are_all_returned() -> None:
    records = tuple(
        make_record(f"00000000-0000-0000-0000-{value:012d}") for value in (1, 2, 3)
    )
    assert len(MeetingLibraryService(FakeRepository(records)).list_meetings()) == 3


@pytest.mark.parametrize("factory", [make_record, lambda: decode(make_record())])
def test_records_and_entries_are_frozen(factory) -> None:
    value = factory()
    with pytest.raises(FrozenInstanceError):
        value.duration_ns = None


def test_display_at_prefers_started_at_and_falls_back_to_created_at() -> None:
    started = decode(make_record())
    created = decode(make_record(started_at=None))
    assert started.display_at == started.started_at
    assert created.display_at == created.created_at


@pytest.mark.parametrize(
    ("duration_ns", "expected"),
    [(None, None), (1_500_000_000, 1.5)],
)
def test_duration_seconds(duration_ns, expected) -> None:
    assert decode(make_record(duration_ns=duration_ns)).duration_seconds == expected


@pytest.mark.parametrize("source_kind", list(MeetingRemoteSourceKind))
def test_all_remote_source_kinds_decode(source_kind) -> None:
    assert (
        decode(make_record(remote_source_kind=source_kind.name)).remote_source_kind
        is source_kind
    )


@pytest.mark.parametrize("session_state", list(MeetingSessionState))
def test_all_session_states_decode(session_state) -> None:
    assert (
        decode(make_record(session_state=session_state.name)).session_state
        is session_state
    )


@pytest.mark.parametrize("audio_state", list(MeetingAudioTracksState))
def test_all_audio_states_decode(audio_state) -> None:
    assert decode(make_record(audio_state=audio_state.name)).audio_state is audio_state


@pytest.mark.parametrize("audio_outcome", [None, *list(MeetingAudioTracksOutcome)])
def test_all_audio_outcomes_and_none_decode(audio_outcome) -> None:
    raw = None if audio_outcome is None else audio_outcome.name
    assert decode(make_record(audio_outcome=raw)).audio_outcome is audio_outcome


@pytest.mark.parametrize(
    "session_id",
    [
        "malformed",
        "00000000000000000000000000000001",
        "{00000000-0000-0000-0000-000000000001}",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        uuid.uuid4(),
    ],
)
def test_noncanonical_or_malformed_uuid_is_rejected(session_id) -> None:
    with pytest.raises(MeetingLibraryDecodeError):
        decode(make_record(session_id=session_id))


@pytest.mark.parametrize(
    ("field_name", "raw"),
    [
        ("remote_source_kind", "UNKNOWN"),
        ("session_state", "UNKNOWN"),
        ("audio_state", "UNKNOWN"),
        ("audio_outcome", "UNKNOWN"),
        ("remote_source_kind", 1),
    ],
)
def test_unknown_or_nontext_enum_is_rejected(field_name, raw) -> None:
    with pytest.raises(MeetingLibraryDecodeError):
        decode(make_record(**{field_name: raw}))


@pytest.mark.parametrize(
    ("field_name", "raw"),
    [
        ("created_at", "not-a-date"),
        ("created_at", "2025-01-01T00:00:00"),
        ("started_at", "not-a-date"),
        ("ended_at", "2025-01-01T00:00:00"),
    ],
)
def test_malformed_or_naive_datetime_is_rejected(field_name, raw) -> None:
    with pytest.raises(MeetingLibraryDecodeError):
        decode(make_record(**{field_name: raw}))


def test_aware_non_utc_offset_is_normalized_to_utc() -> None:
    entry = decode(
        make_record(
            created_at="2025-01-01T09:00:00+09:00",
            started_at="2025-01-01T10:00:00+09:00",
            ended_at=None,
        )
    )
    assert entry.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert entry.started_at == datetime(2025, 1, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("duration_ns", [-1, True, 1.0, "1"])
def test_invalid_duration_is_rejected(duration_ns) -> None:
    with pytest.raises(MeetingLibraryDecodeError):
        decode(make_record(duration_ns=duration_ns))


def test_corrupt_nth_record_fails_the_whole_call() -> None:
    records = (make_record(), make_record(session_id="corrupt"), make_record())
    with pytest.raises(MeetingLibraryDecodeError):
        MeetingLibraryService(FakeRepository(records)).list_meetings()


def test_every_call_reads_repository_fresh_without_cache() -> None:
    repository = FakeRepository((make_record(),))
    service = MeetingLibraryService(repository)
    first = service.list_meetings()
    repository.records = (make_record("00000000-0000-0000-0000-000000000002"),)
    second = service.list_meetings()
    assert repository.calls == 2
    assert first != second


def test_ordering_uses_display_at_descending() -> None:
    earlier = make_record(
        "00000000-0000-0000-0000-000000000001",
        started_at="2025-01-01T01:00:00+00:00",
    )
    later = make_record(
        "00000000-0000-0000-0000-000000000002",
        started_at="2025-01-01T02:00:00+00:00",
    )
    entries = MeetingLibraryService(FakeRepository((earlier, later))).list_meetings()
    assert [str(entry.session_id) for entry in entries] == [
        later.session_id,
        earlier.session_id,
    ]


def test_ordering_uses_created_at_descending_as_display_tie_breaker() -> None:
    display_at = "2025-01-02T00:00:00+00:00"
    older = make_record(
        "00000000-0000-0000-0000-000000000001",
        created_at="2025-01-01T00:00:00+00:00",
        started_at=display_at,
    )
    newer = make_record(
        "00000000-0000-0000-0000-000000000002",
        created_at="2025-01-01T01:00:00+00:00",
        started_at=display_at,
    )
    entries = MeetingLibraryService(FakeRepository((older, newer))).list_meetings()
    assert [str(entry.session_id) for entry in entries] == [
        newer.session_id,
        older.session_id,
    ]


def test_ordering_uses_canonical_uuid_text_ascending_as_final_tie_breaker() -> None:
    larger = make_record("00000000-0000-0000-0000-000000000002")
    smaller = make_record("00000000-0000-0000-0000-000000000001")
    entries = MeetingLibraryService(FakeRepository((larger, smaller))).list_meetings()
    assert [str(entry.session_id) for entry in entries] == [
        smaller.session_id,
        larger.session_id,
    ]


def test_ordering_uses_decoded_utc_chronology_not_raw_iso_text() -> None:
    lexically_later_but_chronologically_earlier = make_record(
        "00000000-0000-0000-0000-000000000001",
        started_at="2025-01-01T09:00:00+09:00",
    )
    chronologically_later = make_record(
        "00000000-0000-0000-0000-000000000002",
        started_at="2025-01-01T01:00:00+00:00",
    )
    entries = MeetingLibraryService(
        FakeRepository(
            (
                lexically_later_but_chronologically_earlier,
                chronologically_later,
            )
        )
    ).list_meetings()
    assert [str(entry.session_id) for entry in entries] == [
        chronologically_later.session_id,
        lexically_later_but_chronologically_earlier.session_id,
    ]


def test_no_cross_field_lifecycle_rules_are_invented() -> None:
    entry = decode(
        make_record(
            session_state="FAILED",
            started_at=None,
            ended_at=None,
            duration_ns=None,
            audio_state="CREATED",
            audio_outcome=None,
        )
    )
    assert entry.session_state is MeetingSessionState.FAILED
    assert entry.display_at == entry.created_at
