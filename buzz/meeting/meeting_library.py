"""Lightweight read projection for the durable meetings library."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
)
from buzz.meeting.meeting_session import (
    MeetingRemoteSourceKind,
    MeetingSessionState,
)


class MeetingLibraryError(Exception):
    """Base error for loading the meetings library."""


class MeetingLibraryDecodeError(MeetingLibraryError):
    """Raised when a persisted meeting header cannot be decoded strictly."""


class MeetingLibraryDatabaseError(MeetingLibraryError):
    """Raised when the meetings library query cannot be executed."""


@dataclass(frozen=True, slots=True)
class MeetingLibraryRecord:
    """Raw primitive meeting-header fields returned by persistence."""

    session_id: object
    remote_source_kind: object
    session_state: object
    created_at: object
    started_at: object
    ended_at: object
    duration_ns: object
    audio_state: object
    audio_outcome: object


@dataclass(frozen=True, slots=True)
class MeetingLibraryEntry:
    """Decoded meeting header displayed by the meetings library."""

    session_id: uuid.UUID
    remote_source_kind: MeetingRemoteSourceKind
    session_state: MeetingSessionState
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    duration_ns: int | None
    audio_state: MeetingAudioTracksState
    audio_outcome: MeetingAudioTracksOutcome | None

    @property
    def display_at(self) -> datetime:
        return self.started_at or self.created_at

    @property
    def duration_seconds(self) -> float | None:
        if self.duration_ns is None:
            return None
        return self.duration_ns / 1_000_000_000


class MeetingLibraryRepository(Protocol):
    """Persistence boundary for meeting header collection reads."""

    def list_meetings(self) -> tuple[MeetingLibraryRecord, ...]:
        ...


class MeetingLibraryService:
    """Decode and semantically order fresh meeting header reads."""

    def __init__(self, repository: MeetingLibraryRepository) -> None:
        self._repository = repository

    def list_meetings(self) -> tuple[MeetingLibraryEntry, ...]:
        entries = [self._decode(record) for record in self._repository.list_meetings()]
        entries.sort(key=lambda entry: str(entry.session_id))
        entries.sort(key=lambda entry: entry.created_at, reverse=True)
        entries.sort(key=lambda entry: entry.display_at, reverse=True)
        return tuple(entries)

    @classmethod
    def _decode(cls, record: MeetingLibraryRecord) -> MeetingLibraryEntry:
        return MeetingLibraryEntry(
            session_id=cls._decode_uuid(record.session_id),
            remote_source_kind=cls._decode_enum(
                record.remote_source_kind,
                MeetingRemoteSourceKind,
                "remote_source_kind",
            ),
            session_state=cls._decode_enum(
                record.session_state,
                MeetingSessionState,
                "session_state",
            ),
            created_at=cls._decode_datetime(record.created_at, "created_at"),
            started_at=cls._decode_optional_datetime(record.started_at, "started_at"),
            ended_at=cls._decode_optional_datetime(record.ended_at, "ended_at"),
            duration_ns=cls._decode_duration(record.duration_ns),
            audio_state=cls._decode_enum(
                record.audio_state,
                MeetingAudioTracksState,
                "audio_state",
            ),
            audio_outcome=(
                None
                if record.audio_outcome is None
                else cls._decode_enum(
                    record.audio_outcome,
                    MeetingAudioTracksOutcome,
                    "audio_outcome",
                )
            ),
        )

    @staticmethod
    def _decode_uuid(raw: object) -> uuid.UUID:
        if not isinstance(raw, str):
            raise MeetingLibraryDecodeError("session_id must be canonical UUID text")
        try:
            value = uuid.UUID(raw)
        except ValueError as exc:
            raise MeetingLibraryDecodeError(
                "session_id must be canonical UUID text"
            ) from exc
        if str(value) != raw:
            raise MeetingLibraryDecodeError("session_id must be canonical UUID text")
        return value

    @staticmethod
    def _decode_enum(raw: object, enum_type: type, field_name: str):
        if not isinstance(raw, str):
            raise MeetingLibraryDecodeError(f"{field_name} must be text")
        try:
            return enum_type[raw]
        except KeyError as exc:
            raise MeetingLibraryDecodeError(f"Unknown {field_name}: {raw!r}") from exc

    @staticmethod
    def _decode_datetime(raw: object, field_name: str) -> datetime:
        if not isinstance(raw, str):
            raise MeetingLibraryDecodeError(f"{field_name} must be text")
        try:
            value = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise MeetingLibraryDecodeError(
                f"Malformed persisted {field_name}"
            ) from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise MeetingLibraryDecodeError(
                f"Persisted {field_name} must be timezone-aware"
            )
        return value.astimezone(timezone.utc)

    @classmethod
    def _decode_optional_datetime(cls, raw: object, field_name: str) -> datetime | None:
        if raw is None:
            return None
        return cls._decode_datetime(raw, field_name)

    @staticmethod
    def _decode_duration(raw: object) -> int | None:
        if raw is None:
            return None
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise MeetingLibraryDecodeError("duration_ns must be an integer")
        if raw < 0:
            raise MeetingLibraryDecodeError("duration_ns must be nonnegative")
        return raw


__all__ = [
    "MeetingLibraryDatabaseError",
    "MeetingLibraryDecodeError",
    "MeetingLibraryEntry",
    "MeetingLibraryError",
    "MeetingLibraryRecord",
    "MeetingLibraryRepository",
    "MeetingLibraryService",
]
