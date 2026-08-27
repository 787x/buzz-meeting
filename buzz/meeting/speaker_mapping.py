"""Pure speaker-mapping module — maps ASR word timestamps to speaker turns.

No Qt, Settings, torch, NeMo, DB, filesystem, or network.
All types are frozen dataclasses or pure Python enums.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from buzz.meeting.final_transcription import MeetingTranscriptWord
from buzz.meeting.meeting_audio_tracks import MeetingTrackRole
from buzz.meeting.speaker_diarization import SpeakerDiarizationTurn


# ── Errors ────────────────────────────────────────────────────────────────────


class SpeakerMappingError(Exception):
    """Base error for speaker-mapping failures."""


class SpeakerMappingConfigError(SpeakerMappingError):
    """Raised for invalid structural inputs."""


# ── DTOs ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MeetingTrackSpeakerTurns:
    """One diarization result set for one meeting source role."""

    source_role: MeetingTrackRole
    turns: tuple[SpeakerDiarizationTurn, ...]


@dataclass(frozen=True, slots=True)
class MeetingSpeakerKey:
    """Machine speaker-cluster identity — not a human identity."""

    source_role: MeetingTrackRole
    speaker_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_role, MeetingTrackRole):
            raise SpeakerMappingConfigError(
                f"source_role must be MeetingTrackRole, got {type(self.source_role)}"
            )
        if isinstance(self.speaker_index, bool):
            raise SpeakerMappingConfigError("speaker_index must be an int, not bool")
        if not isinstance(self.speaker_index, int):
            raise SpeakerMappingConfigError(
                f"speaker_index must be an int, got {type(self.speaker_index)}"
            )
        if self.speaker_index < 0:
            raise SpeakerMappingConfigError(
                f"speaker_index must be >= 0, got {self.speaker_index}"
            )


class SpeakerAttributionStatus(Enum):
    """Why a word was or was not attributed to a speaker."""

    ASSIGNED = auto()
    NO_OVERLAP = auto()
    AMBIGUOUS = auto()


@dataclass(frozen=True, slots=True)
class SpeakerAttributedWord:
    """One word with its speaker attribution."""

    word: MeetingTranscriptWord
    speaker: MeetingSpeakerKey | None
    status: SpeakerAttributionStatus

    def __post_init__(self) -> None:
        if self.status is SpeakerAttributionStatus.ASSIGNED:
            if self.speaker is None:
                raise ValueError("ASSIGNED status requires a non-None speaker")
        elif self.status in (
            SpeakerAttributionStatus.NO_OVERLAP,
            SpeakerAttributionStatus.AMBIGUOUS,
        ):
            if self.speaker is not None:
                raise ValueError(f"{self.status.name} status requires speaker=None")


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_int_not_bool(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise SpeakerMappingConfigError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise SpeakerMappingConfigError(f"{name} must be an int, got {type(value)}")
    return value


def _validate_word(word: object) -> MeetingTranscriptWord:
    if not isinstance(word, MeetingTranscriptWord):
        raise SpeakerMappingConfigError(
            f"word must be MeetingTranscriptWord, got {type(word)}"
        )
    if not isinstance(word.source_role, MeetingTrackRole):
        raise SpeakerMappingConfigError(
            f"word.source_role must be MeetingTrackRole, got {type(word.source_role)}"
        )
    local_start_ms = _validate_int_not_bool(word.local_start_ms, "word.local_start_ms")
    if local_start_ms < 0:
        raise SpeakerMappingConfigError(
            f"word.local_start_ms must be >= 0, got {local_start_ms}"
        )
    local_end_ms = _validate_int_not_bool(word.local_end_ms, "word.local_end_ms")
    if local_end_ms < local_start_ms:
        raise SpeakerMappingConfigError(
            f"word.local_end_ms must be >= local_start_ms, "
            f"got {local_end_ms} < {local_start_ms}"
        )
    return word


def _validate_turn(turn: object) -> None:
    if not isinstance(turn, SpeakerDiarizationTurn):
        raise SpeakerMappingConfigError(
            f"turn must be SpeakerDiarizationTurn, got {type(turn)}"
        )
    speaker_index = _validate_int_not_bool(turn.speaker_index, "turn.speaker_index")
    if speaker_index < 0:
        raise SpeakerMappingConfigError(
            f"turn.speaker_index must be >= 0, got {speaker_index}"
        )
    start_ms = _validate_int_not_bool(turn.start_ms, "turn.start_ms")
    if start_ms < 0:
        raise SpeakerMappingConfigError(f"turn.start_ms must be >= 0, got {start_ms}")
    end_ms = _validate_int_not_bool(turn.end_ms, "turn.end_ms")
    if end_ms < start_ms:
        raise SpeakerMappingConfigError(
            f"turn.end_ms must be >= start_ms, got {end_ms} < {start_ms}"
        )


def _validate_diarization_entry(entry: object) -> MeetingTrackSpeakerTurns:
    if not isinstance(entry, MeetingTrackSpeakerTurns):
        raise SpeakerMappingConfigError(
            f"diarization entry must be MeetingTrackSpeakerTurns, " f"got {type(entry)}"
        )
    if not isinstance(entry.source_role, MeetingTrackRole):
        raise SpeakerMappingConfigError(
            f"source_role must be MeetingTrackRole, got {type(entry.source_role)}"
        )
    if not isinstance(entry.turns, tuple):
        raise SpeakerMappingConfigError(f"turns must be tuple, got {type(entry.turns)}")
    for i, turn in enumerate(entry.turns):
        try:
            _validate_turn(turn)
        except SpeakerMappingConfigError as exc:
            raise SpeakerMappingConfigError(
                f"turn[{i}] in role {entry.source_role.name}: {exc}"
            ) from exc
    return entry


# ── Interval math ─────────────────────────────────────────────────────────────


def _positive_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Half-open interval overlap: max(0, min(a_end, b_end) - max(a_start, b_start))."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _merge_interval_lengths(intervals: list[tuple[int, int]]) -> int:
    """Merge overlapping/adjacent half-open intervals, return total length."""
    if not intervals:
        return 0
    # Stable sort by start
    intervals.sort(key=lambda iv: iv[0])
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            # Overlapping or adjacent — extend
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


# ── Public API ────────────────────────────────────────────────────────────────


def map_words_to_speakers(
    words: Sequence[MeetingTranscriptWord],
    diarization: Sequence[MeetingTrackSpeakerTurns],
) -> tuple[SpeakerAttributedWord, ...]:
    """Map ASR words to diarization speaker turns.

    Pure function.  No async, no callbacks, no progress, no persistence.
    Caller supplies already-produced DTOs.
    """
    # ── Validate diarization entries ──────────────────────────────────────
    validated_entries: list[MeetingTrackSpeakerTurns] = []
    seen_roles: set[MeetingTrackRole] = set()
    for entry in diarization:
        validated = _validate_diarization_entry(entry)
        if validated.source_role in seen_roles:
            raise SpeakerMappingConfigError(
                f"Duplicate role set for {validated.source_role.name}"
            )
        seen_roles.add(validated.source_role)
        validated_entries.append(validated)

    # ── Validate words ────────────────────────────────────────────────────
    validated_words: list[MeetingTranscriptWord] = []
    for word in words:
        validated_words.append(_validate_word(word))

    # Empty words → empty result
    if not validated_words:
        return ()

    # ── Bucket and sort turns by role ─────────────────────────────────────
    turns_by_role: dict[MeetingTrackRole, list[SpeakerDiarizationTurn]] = {}
    for entry in validated_entries:
        # Stable sort by start_ms only; equal start_ms preserves input order
        sorted_turns = sorted(entry.turns, key=lambda t: t.start_ms)
        turns_by_role[entry.source_role] = sorted_turns

    # ── Map each word ─────────────────────────────────────────────────────
    result: list[SpeakerAttributedWord] = []
    for word in validated_words:
        role = word.source_role
        role_turns = turns_by_role.get(role)

        if not role_turns:
            # Missing or empty role turn set → NO_OVERLAP
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.NO_OVERLAP,
                )
            )
            continue

        # Zero-duration word → always NO_OVERLAP
        if word.local_start_ms == word.local_end_ms:
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.NO_OVERLAP,
                )
            )
            continue

        # Collect per-speaker intersection intervals
        # turns are sorted by start_ms; break once turn.start >= word.end
        speaker_intervals: dict[int, list[tuple[int, int]]] = {}
        for turn in role_turns:
            if turn.start_ms >= word.local_end_ms:
                break
            if turn.end_ms <= word.local_start_ms:
                continue

            overlap = _positive_overlap(
                word.local_start_ms,
                word.local_end_ms,
                turn.start_ms,
                turn.end_ms,
            )
            if overlap <= 0:
                continue

            iv_start = max(word.local_start_ms, turn.start_ms)
            iv_end = min(word.local_end_ms, turn.end_ms)
            speaker_intervals.setdefault(turn.speaker_index, []).append(
                (iv_start, iv_end)
            )

        if not speaker_intervals:
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.NO_OVERLAP,
                )
            )
            continue

        # Compute unique coverage per speaker
        speaker_coverage: dict[int, int] = {}
        for speaker_idx, intervals in speaker_intervals.items():
            speaker_coverage[speaker_idx] = _merge_interval_lengths(intervals)

        # Find winner
        if not speaker_coverage:
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.NO_OVERLAP,
                )
            )
            continue

        max_coverage = max(speaker_coverage.values())
        if max_coverage <= 0:
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.NO_OVERLAP,
                )
            )
            continue

        winners = [idx for idx, cov in speaker_coverage.items() if cov == max_coverage]

        if len(winners) >= 2:
            # Tie — AMBIGUOUS
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=None,
                    status=SpeakerAttributionStatus.AMBIGUOUS,
                )
            )
        else:
            # Unique winner — ASSIGNED
            result.append(
                SpeakerAttributedWord(
                    word=word,
                    speaker=MeetingSpeakerKey(
                        source_role=role,
                        speaker_index=winners[0],
                    ),
                    status=SpeakerAttributionStatus.ASSIGNED,
                )
            )

    return tuple(result)
