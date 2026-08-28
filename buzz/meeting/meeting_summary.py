"""Pure meeting-summary domain, serialization, and freshness.

No Qt, QSql, network, or provider imports.  All types are frozen
dataclasses or pure Python protocols.  Serialization boundary covers
only the provider-independent ``MeetingSummary`` payload; the durable
``MeetingSummaryArtifact`` envelope is never exposed in that JSON
contract.
"""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

MEETING_SUMMARY_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MeetingSummaryError(Exception):
    """Base error for meeting-summary failures."""


class MeetingSummaryValidationError(MeetingSummaryError):
    """Direct caller/domain invariant failure."""


class MeetingSummaryVersionError(MeetingSummaryError):
    """Unsupported schema_version."""


class MeetingSummaryDecodeError(MeetingSummaryError):
    """Serialized/persisted representation invalid."""


class MeetingSummaryDatabaseError(MeetingSummaryError):
    """QSql/transaction failure."""


class MeetingSummaryConflictError(MeetingSummaryError):
    """Duplicate summary artifact ID."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeetingSummaryValidationError(
            f"{name} must be a non-bool int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise MeetingSummaryValidationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise MeetingSummaryValidationError(f"{name} must be str")
    if not value.strip():
        raise MeetingSummaryValidationError(f"{name} must not be empty/whitespace")
    return value


def _check_optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MeetingSummaryValidationError(f"{name} must be str or None")
    if not value.strip():
        raise MeetingSummaryValidationError(
            f"{name} must not be empty/whitespace when present"
        )
    return value


def _validate_timestamp_pair(
    start: int | None, end: int | None, name: str
) -> tuple[int | None, int | None]:
    """Validate a source_start_ns / source_end_ns pair."""
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise MeetingSummaryValidationError(
            f"{name}: both source_start_ns and source_end_ns must be present "
            "or both must be None"
        )
    start = _require_int(start, f"{name}.source_start_ns")
    end = _require_int(end, f"{name}.source_end_ns")
    if end < start:
        raise MeetingSummaryValidationError(
            f"{name}: source_end_ns ({end}) must be >= source_start_ns ({start})"
        )
    return start, end


def _validate_aware_utc(dt: datetime.datetime, name: str) -> datetime.datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise MeetingSummaryValidationError(f"{name} must be timezone-aware")
    return dt.astimezone(datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Nested DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Participant:
    name: str | None
    reviewed_speaker_id: uuid.UUID | None

    def __post_init__(self) -> None:
        if self.name is None and self.reviewed_speaker_id is None:
            raise MeetingSummaryValidationError(
                "Participant must have at least one of name or reviewed_speaker_id"
            )
        if self.name is not None:
            _check_optional_text(self.name, "Participant.name")
        if self.reviewed_speaker_id is not None:
            if not isinstance(self.reviewed_speaker_id, uuid.UUID):
                raise MeetingSummaryValidationError(
                    "Participant.reviewed_speaker_id must be uuid.UUID or None"
                )


@dataclass(frozen=True, slots=True)
class Topic:
    title: str
    summary: str | None
    source_start_ns: int | None
    source_end_ns: int | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.title, "Topic.title")
        _check_optional_text(self.summary, "Topic.summary")
        s, e = _validate_timestamp_pair(
            self.source_start_ns, self.source_end_ns, "Topic"
        )
        object.__setattr__(self, "source_start_ns", s)
        object.__setattr__(self, "source_end_ns", e)


@dataclass(frozen=True, slots=True)
class Decision:
    text: str
    source_start_ns: int | None
    source_end_ns: int | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.text, "Decision.text")
        s, e = _validate_timestamp_pair(
            self.source_start_ns, self.source_end_ns, "Decision"
        )
        object.__setattr__(self, "source_start_ns", s)
        object.__setattr__(self, "source_end_ns", e)


@dataclass(frozen=True, slots=True)
class ActionItem:
    task: str
    owner: str | None
    due_date: datetime.date | None
    source_start_ns: int | None
    source_end_ns: int | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.task, "ActionItem.task")
        _check_optional_text(self.owner, "ActionItem.owner")
        if self.due_date is not None and not isinstance(self.due_date, datetime.date):
            raise MeetingSummaryValidationError(
                "ActionItem.due_date must be datetime.date or None"
            )
        s, e = _validate_timestamp_pair(
            self.source_start_ns, self.source_end_ns, "ActionItem"
        )
        object.__setattr__(self, "source_start_ns", s)
        object.__setattr__(self, "source_end_ns", e)


@dataclass(frozen=True, slots=True)
class Risk:
    text: str
    source_start_ns: int | None
    source_end_ns: int | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.text, "Risk.text")
        s, e = _validate_timestamp_pair(
            self.source_start_ns, self.source_end_ns, "Risk"
        )
        object.__setattr__(self, "source_start_ns", s)
        object.__setattr__(self, "source_end_ns", e)


@dataclass(frozen=True, slots=True)
class OpenQuestion:
    text: str
    source_start_ns: int | None
    source_end_ns: int | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.text, "OpenQuestion.text")
        s, e = _validate_timestamp_pair(
            self.source_start_ns, self.source_end_ns, "OpenQuestion"
        )
        object.__setattr__(self, "source_start_ns", s)
        object.__setattr__(self, "source_end_ns", e)


# ---------------------------------------------------------------------------
# MeetingSummary — provider-independent payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    """Provider-independent structured meeting summary payload.

    This is the future PR19–23 structured result contract.  It does NOT
    contain summary_id, meeting_id, or any application-owned provenance.
    """

    schema_version: int
    prompt_version: int
    title: str | None
    summary: str
    participants: tuple[Participant, ...]
    topics: tuple[Topic, ...]
    decisions: tuple[Decision, ...]
    action_items: tuple[ActionItem, ...]
    open_questions: tuple[OpenQuestion, ...]
    risks: tuple[Risk, ...]

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "schema_version")
        if self.schema_version != MEETING_SUMMARY_SCHEMA_VERSION:
            raise MeetingSummaryVersionError(
                f"Unsupported schema_version: {self.schema_version}"
            )
        _require_int(self.prompt_version, "prompt_version", minimum=1)
        if self.title is not None:
            _check_optional_text(self.title, "MeetingSummary.title")
        _require_nonempty_text(self.summary, "MeetingSummary.summary")
        _require_tuple(self.participants, "participants")
        _require_tuple(self.topics, "topics")
        _require_tuple(self.decisions, "decisions")
        _require_tuple(self.action_items, "action_items")
        _require_tuple(self.open_questions, "open_questions")
        _require_tuple(self.risks, "risks")


def _require_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise MeetingSummaryValidationError(f"{name} must be a tuple")


# ---------------------------------------------------------------------------
# MeetingSummaryArtifact — durable envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeetingSummaryArtifact:
    """Durable provenance envelope wrapping a ``MeetingSummary``."""

    summary_id: uuid.UUID
    meeting_id: uuid.UUID
    source_generation_id: uuid.UUID
    source_profile_version: int
    source_review_id: uuid.UUID | None
    source_review_revision: int | None
    created_at: datetime.datetime
    summary: MeetingSummary

    def __post_init__(self) -> None:
        _require_uuid(self.summary_id, "summary_id")
        _require_uuid(self.meeting_id, "meeting_id")
        _require_uuid(self.source_generation_id, "source_generation_id")
        _require_int(self.source_profile_version, "source_profile_version", minimum=1)

        # Review pair: both None or both present
        has_id = self.source_review_id is not None
        has_rev = self.source_review_revision is not None
        if has_id != has_rev:
            raise MeetingSummaryValidationError(
                "source_review_id and source_review_revision must both be "
                "present or both be None"
            )
        if self.source_review_id is not None:
            _require_uuid(self.source_review_id, "source_review_id")
        if self.source_review_revision is not None:
            _require_int(
                self.source_review_revision,
                "source_review_revision",
                minimum=0,
            )

        # created_at: must be aware, normalize to UTC
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise MeetingSummaryValidationError("created_at must be timezone-aware")
        object.__setattr__(
            self, "created_at", _validate_aware_utc(self.created_at, "created_at")
        )

        if not isinstance(self.summary, MeetingSummary):
            raise MeetingSummaryValidationError("summary must be MeetingSummary")

        # Cross-validation: participant reviewed_speaker_id ↔ review provenance
        _validate_participant_review_consistency(
            self.summary.participants,
            self.source_review_id,
            self.source_review_revision,
        )


def _require_uuid(value: object, name: str) -> uuid.UUID:
    if not isinstance(value, uuid.UUID):
        raise MeetingSummaryValidationError(
            f"{name} must be uuid.UUID, got {type(value).__name__}"
        )
    return value


def _validate_participant_review_consistency(
    participants: tuple[Participant, ...],
    review_id: uuid.UUID | None,
    review_revision: int | None,
) -> None:
    has_review = review_id is not None
    for p in participants:
        if p.reviewed_speaker_id is not None and not has_review:
            raise MeetingSummaryValidationError(
                "Participant has reviewed_speaker_id but artifact has no "
                "review provenance"
            )
        if p.reviewed_speaker_id is None and has_review:
            # This is fine — a participant may have name only even when
            # review provenance exists.
            pass


# ---------------------------------------------------------------------------
# Canonical serialization — MeetingSummary ONLY
# ---------------------------------------------------------------------------


def meeting_summary_to_dict(summary: MeetingSummary) -> dict[str, object]:
    """Convert a ``MeetingSummary`` to a canonical dict."""
    if not isinstance(summary, MeetingSummary):
        raise MeetingSummaryValidationError("summary must be MeetingSummary")
    return {
        "schema_version": summary.schema_version,
        "prompt_version": summary.prompt_version,
        "title": summary.title,
        "summary": summary.summary,
        "participants": [_participant_to_dict(p) for p in summary.participants],
        "topics": [_topic_to_dict(t) for t in summary.topics],
        "decisions": [_decision_to_dict(d) for d in summary.decisions],
        "action_items": [_action_item_to_dict(a) for a in summary.action_items],
        "open_questions": [_open_question_to_dict(q) for q in summary.open_questions],
        "risks": [_risk_to_dict(r) for r in summary.risks],
    }


def _participant_to_dict(p: Participant) -> dict[str, object]:
    return {
        "name": p.name,
        "reviewed_speaker_id": (
            str(p.reviewed_speaker_id) if p.reviewed_speaker_id is not None else None
        ),
    }


def _topic_to_dict(t: Topic) -> dict[str, object]:
    return {
        "title": t.title,
        "summary": t.summary,
        "source_start_ns": t.source_start_ns,
        "source_end_ns": t.source_end_ns,
    }


def _decision_to_dict(d: Decision) -> dict[str, object]:
    return {
        "text": d.text,
        "source_start_ns": d.source_start_ns,
        "source_end_ns": d.source_end_ns,
    }


def _action_item_to_dict(a: ActionItem) -> dict[str, object]:
    return {
        "task": a.task,
        "owner": a.owner,
        "due_date": a.due_date.isoformat() if a.due_date is not None else None,
        "source_start_ns": a.source_start_ns,
        "source_end_ns": a.source_end_ns,
    }


def _open_question_to_dict(q: OpenQuestion) -> dict[str, object]:
    return {
        "text": q.text,
        "source_start_ns": q.source_start_ns,
        "source_end_ns": q.source_end_ns,
    }


def _risk_to_dict(r: Risk) -> dict[str, object]:
    return {
        "text": r.text,
        "source_start_ns": r.source_start_ns,
        "source_end_ns": r.source_end_ns,
    }


def meeting_summary_from_dict(data: object) -> MeetingSummary:
    """Deserialize a ``MeetingSummary`` from a dict.

    Rejects unknown fields, missing fields, wrong types, and
    non-object inputs.
    """
    if not isinstance(data, dict):
        raise MeetingSummaryDecodeError("Top-level value must be an object")

    _KNOWN_KEYS = {
        "schema_version",
        "prompt_version",
        "title",
        "summary",
        "participants",
        "topics",
        "decisions",
        "action_items",
        "open_questions",
        "risks",
    }
    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown top-level keys: {unknown}")

    try:
        schema_version = _require_int(data["schema_version"], "schema_version")
    except KeyError as exc:
        raise MeetingSummaryDecodeError(f"Missing required key: {exc}") from exc
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc

    if schema_version != MEETING_SUMMARY_SCHEMA_VERSION:
        raise MeetingSummaryVersionError(
            f"Unsupported schema_version: {schema_version}"
        )

    try:
        prompt_version = _require_int(
            data["prompt_version"], "prompt_version", minimum=1
        )
    except KeyError as exc:
        raise MeetingSummaryDecodeError(f"Missing required key: {exc}") from exc
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc

    title = _get_optional_str(data, "title")
    summary = _get_required_str(data, "summary")

    try:
        participants = tuple(
            _participant_from_dict(p) for p in _get_list(data, "participants")
        )
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid participants: {exc}") from exc

    try:
        topics = tuple(_topic_from_dict(t) for t in _get_list(data, "topics"))
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid topics: {exc}") from exc

    try:
        decisions = tuple(_decision_from_dict(d) for d in _get_list(data, "decisions"))
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid decisions: {exc}") from exc

    try:
        action_items = tuple(
            _action_item_from_dict(a) for a in _get_list(data, "action_items")
        )
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid action_items: {exc}") from exc

    try:
        open_questions = tuple(
            _open_question_from_dict(q) for q in _get_list(data, "open_questions")
        )
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid open_questions: {exc}") from exc

    try:
        risks = tuple(_risk_from_dict(r) for r in _get_list(data, "risks"))
    except (KeyError, MeetingSummaryDecodeError):
        raise
    except Exception as exc:
        raise MeetingSummaryDecodeError(f"Invalid risks: {exc}") from exc

    try:
        return MeetingSummary(
            schema_version=schema_version,
            prompt_version=prompt_version,
            title=title,
            summary=summary,
            participants=participants,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            open_questions=open_questions,
            risks=risks,
        )
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _get_required_str(data: dict[str, object], key: str) -> str:
    try:
        value = data[key]
    except KeyError as exc:
        raise MeetingSummaryDecodeError(f"Missing required key: {key}") from exc
    if not isinstance(value, str):
        raise MeetingSummaryDecodeError(f"{key} must be str")
    return value


def _get_optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MeetingSummaryDecodeError(f"{key} must be str or null")
    return value


def _get_list(data: dict[str, object], key: str) -> list[object]:
    try:
        value = data[key]
    except KeyError as exc:
        raise MeetingSummaryDecodeError(f"Missing required key: {key}") from exc
    if not isinstance(value, list):
        raise MeetingSummaryDecodeError(f"{key} must be an array")
    return value


def _get_optional_str_nested(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise MeetingSummaryDecodeError(f"{key} must be str or null")
    return value


def _get_optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeetingSummaryDecodeError(f"{key} must be int or null")
    return value


def _get_optional_uuid(data: dict[str, object], key: str) -> uuid.UUID | None:
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise MeetingSummaryDecodeError(f"{key} must be a UUID string or null")
    try:
        value = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise MeetingSummaryDecodeError(f"Invalid {key}: {raw!r}") from exc
    if str(value) != raw:
        raise MeetingSummaryDecodeError(f"Non-canonical {key}: {raw!r}")
    return value


def _get_optional_date(data: dict[str, object], key: str) -> datetime.date | None:
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise MeetingSummaryDecodeError(f"{key} must be a date string or null")
    try:
        value = datetime.date.fromisoformat(raw)
    except ValueError as exc:
        raise MeetingSummaryDecodeError(f"Invalid {key}: {raw!r}") from exc
    if value.isoformat() != raw:
        raise MeetingSummaryDecodeError(
            f"Non-canonical {key}: {raw!r} (expected {value.isoformat()!r})"
        )
    return value


def _participant_from_dict(d: object) -> Participant:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("Participant must be an object")
    known = {"name", "reviewed_speaker_id"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown Participant keys: {unknown}")
    name = _get_optional_str_nested(d, "name")
    sid = _get_optional_uuid(d, "reviewed_speaker_id")
    try:
        return Participant(name=name, reviewed_speaker_id=sid)
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _topic_from_dict(d: object) -> Topic:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("Topic must be an object")
    known = {"title", "summary", "source_start_ns", "source_end_ns"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown Topic keys: {unknown}")
    try:
        title = _get_required_str(d, "title")
    except MeetingSummaryDecodeError:
        raise
    summary = _get_optional_str_nested(d, "summary")
    start = _get_optional_int(d, "source_start_ns")
    end = _get_optional_int(d, "source_end_ns")
    try:
        return Topic(
            title=title,
            summary=summary,
            source_start_ns=start,
            source_end_ns=end,
        )
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _decision_from_dict(d: object) -> Decision:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("Decision must be an object")
    known = {"text", "source_start_ns", "source_end_ns"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown Decision keys: {unknown}")
    text = _get_required_str(d, "text")
    start = _get_optional_int(d, "source_start_ns")
    end = _get_optional_int(d, "source_end_ns")
    try:
        return Decision(text=text, source_start_ns=start, source_end_ns=end)
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _action_item_from_dict(d: object) -> ActionItem:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("ActionItem must be an object")
    known = {"task", "owner", "due_date", "source_start_ns", "source_end_ns"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown ActionItem keys: {unknown}")
    task = _get_required_str(d, "task")
    owner = _get_optional_str_nested(d, "owner")
    due_date = _get_optional_date(d, "due_date")
    start = _get_optional_int(d, "source_start_ns")
    end = _get_optional_int(d, "source_end_ns")
    try:
        return ActionItem(
            task=task,
            owner=owner,
            due_date=due_date,
            source_start_ns=start,
            source_end_ns=end,
        )
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _open_question_from_dict(d: object) -> OpenQuestion:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("OpenQuestion must be an object")
    known = {"text", "source_start_ns", "source_end_ns"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown OpenQuestion keys: {unknown}")
    text = _get_required_str(d, "text")
    start = _get_optional_int(d, "source_start_ns")
    end = _get_optional_int(d, "source_end_ns")
    try:
        return OpenQuestion(text=text, source_start_ns=start, source_end_ns=end)
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


def _risk_from_dict(d: object) -> Risk:
    if not isinstance(d, dict):
        raise MeetingSummaryDecodeError("Risk must be an object")
    known = {"text", "source_start_ns", "source_end_ns"}
    unknown = set(d) - known
    if unknown:
        raise MeetingSummaryDecodeError(f"Unknown Risk keys: {unknown}")
    text = _get_required_str(d, "text")
    start = _get_optional_int(d, "source_start_ns")
    end = _get_optional_int(d, "source_end_ns")
    try:
        return Risk(text=text, source_start_ns=start, source_end_ns=end)
    except MeetingSummaryValidationError as exc:
        raise MeetingSummaryDecodeError(str(exc)) from exc


# JSON boundary


def meeting_summary_to_json(summary: MeetingSummary) -> str:
    """Serialize a ``MeetingSummary`` to canonical JSON text."""
    return json.dumps(
        meeting_summary_to_dict(summary),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def meeting_summary_from_json(text: str) -> MeetingSummary:
    """Deserialize a ``MeetingSummary`` from JSON text."""
    if not isinstance(text, str):
        raise MeetingSummaryDecodeError("JSON text must be str")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MeetingSummaryDecodeError(f"Malformed JSON: {exc}") from exc
    return meeting_summary_from_dict(data)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


class MeetingSummaryFreshness(Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    INDETERMINATE = "INDETERMINATE"


class _GenerationLike(Protocol):
    """Narrow protocol for freshness — avoids importing UI."""

    @property
    def generation_id(self) -> uuid.UUID:
        ...

    @property
    def profile_version(self) -> int:
        ...


class _ReviewLike(Protocol):
    """Narrow protocol for freshness — avoids importing UI."""

    @property
    def id(self) -> uuid.UUID:
        ...

    @property
    def revision(self) -> int:
        ...


def check_freshness(
    artifact: MeetingSummaryArtifact,
    current_generation: _GenerationLike | None,
    current_review: _ReviewLike | None,
) -> MeetingSummaryFreshness:
    """Determine freshness of a summary artifact.

    Pure helper — no DB queries, no side effects, does not mutate the
    artifact.
    """
    # Generation check
    if current_generation is None:
        return MeetingSummaryFreshness.STALE
    if artifact.source_generation_id != current_generation.generation_id:
        return MeetingSummaryFreshness.STALE
    if artifact.source_profile_version != current_generation.profile_version:
        return MeetingSummaryFreshness.STALE

    # Summary used review
    if artifact.source_review_id is not None:
        if current_review is None:
            return MeetingSummaryFreshness.STALE
        if artifact.source_review_id != current_review.id:
            return MeetingSummaryFreshness.STALE
        if artifact.source_review_revision != current_review.revision:
            return MeetingSummaryFreshness.STALE
        return MeetingSummaryFreshness.FRESH

    # Summary used no review
    if current_review is None:
        return MeetingSummaryFreshness.FRESH
    return MeetingSummaryFreshness.INDETERMINATE


# ---------------------------------------------------------------------------
# Artifact payload extraction (used by repository)
# ---------------------------------------------------------------------------


def artifact_payload_dict(artifact: MeetingSummaryArtifact) -> dict[str, object]:
    """Extract the canonical MeetingSummary dict from an artifact.

    Used by the repository to serialize only the payload into
    ``payload_json`` — provenance stays in envelope columns only.
    """
    return meeting_summary_to_dict(artifact.summary)


# ---------------------------------------------------------------------------
# Envelope encoding helpers (used by repository)
# ---------------------------------------------------------------------------


def encode_artifact_created_at(dt: datetime.datetime) -> str:
    """Encode a timezone-aware datetime as canonical UTC ISO-8601."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise MeetingSummaryValidationError("created_at must be timezone-aware")
    return dt.astimezone(datetime.timezone.utc).isoformat()


def decode_artifact_created_at(raw: str) -> datetime.datetime:
    """Decode a UTC ISO-8601 timestamp string."""
    try:
        value = datetime.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise MeetingSummaryDecodeError(f"Malformed created_at: {raw!r}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise MeetingSummaryDecodeError(f"created_at must be timezone-aware: {raw!r}")
    return value.astimezone(datetime.timezone.utc)


def decode_uuid(raw: str, name: str) -> uuid.UUID:
    """Decode a canonical lowercase UUID string."""
    if not isinstance(raw, str):
        raise MeetingSummaryDecodeError(f"{name} must be canonical UUID text")
    try:
        value = uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise MeetingSummaryDecodeError(f"Invalid {name}: {raw!r}") from exc
    if str(value) != raw:
        raise MeetingSummaryDecodeError(f"Non-canonical {name}: {raw!r}")
    return value


__all__ = [
    "ActionItem",
    "Decision",
    "MEETING_SUMMARY_SCHEMA_VERSION",
    "MeetingSummary",
    "MeetingSummaryArtifact",
    "MeetingSummaryConflictError",
    "MeetingSummaryDatabaseError",
    "MeetingSummaryDecodeError",
    "MeetingSummaryError",
    "MeetingSummaryFreshness",
    "MeetingSummaryValidationError",
    "MeetingSummaryVersionError",
    "OpenQuestion",
    "Participant",
    "Risk",
    "Topic",
    "artifact_payload_dict",
    "check_freshness",
    "decode_artifact_created_at",
    "decode_uuid",
    "encode_artifact_created_at",
    "meeting_summary_from_dict",
    "meeting_summary_from_json",
    "meeting_summary_to_dict",
    "meeting_summary_to_json",
]
