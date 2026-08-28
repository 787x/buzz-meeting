"""Tests for the meeting-summary domain, serialization, and freshness."""

from __future__ import annotations

import datetime
import json
import uuid

import pytest

from buzz.meeting.meeting_summary import (
    ActionItem,
    Decision,
    MEETING_SUMMARY_SCHEMA_VERSION,
    MeetingSummary,
    MeetingSummaryArtifact,
    MeetingSummaryDecodeError,
    MeetingSummaryFreshness,
    MeetingSummaryValidationError,
    MeetingSummaryVersionError,
    OpenQuestion,
    Participant,
    Risk,
    Topic,
    check_freshness,
    meeting_summary_from_dict,
    meeting_summary_from_json,
    meeting_summary_to_dict,
    meeting_summary_to_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_GID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_RID = uuid.UUID("00000000-0000-0000-0000-000000000003")
_SID = uuid.UUID("00000000-0000-0000-0000-000000000004")
_PID = uuid.UUID("00000000-0000-0000-0000-000000000005")
_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _minimal_summary(**kw: object) -> MeetingSummary:
    defaults = dict(
        schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
        prompt_version=1,
        title=None,
        summary="A summary.",
        participants=(),
        topics=(),
        decisions=(),
        action_items=(),
        open_questions=(),
        risks=(),
    )
    defaults.update(kw)
    return MeetingSummary(**defaults)  # type: ignore[arg-type]


def _full_summary(**kw: object) -> MeetingSummary:
    defaults = dict(
        schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
        prompt_version=1,
        title="Meeting Title",
        summary="Full summary text.",
        participants=(Participant(name="Alice", reviewed_speaker_id=_PID),),
        topics=(
            Topic(
                title="Budget",
                summary="Budget discussion",
                source_start_ns=0,
                source_end_ns=1000,
            ),
        ),
        decisions=(
            Decision(text="Approved budget", source_start_ns=100, source_end_ns=200),
        ),
        action_items=(
            ActionItem(
                task="Send report",
                owner="Bob",
                due_date=datetime.date(2026, 2, 1),
                source_start_ns=300,
                source_end_ns=400,
            ),
        ),
        open_questions=(
            OpenQuestion(text="What about Q3?", source_start_ns=500, source_end_ns=600),
        ),
        risks=(Risk(text="Budget overrun", source_start_ns=700, source_end_ns=800),),
    )
    defaults.update(kw)
    return MeetingSummary(**defaults)  # type: ignore[arg-type]


def _artifact(
    *,
    review_id: uuid.UUID | None = _RID,
    review_revision: int | None = 4,
    summary: MeetingSummary | None = None,
    **kw: object,
) -> MeetingSummaryArtifact:
    defaults = dict(
        summary_id=_SID,
        meeting_id=_MID,
        source_generation_id=_GID,
        source_profile_version=2,
        source_review_id=review_id,
        source_review_revision=review_revision,
        created_at=_NOW,
        summary=summary or _minimal_summary(),
    )
    defaults.update(kw)
    return MeetingSummaryArtifact(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Frozen / slots
# ---------------------------------------------------------------------------


class TestDataclassProperties:
    def test_meeting_summary_frozen_slots(self) -> None:
        s = _minimal_summary()
        assert type(s).__slots__ == (
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
        )
        with pytest.raises(AttributeError):
            s.title = "new"  # type: ignore[misc]

    def test_meeting_summary_artifact_frozen_slots(self) -> None:
        a = _artifact()
        with pytest.raises(AttributeError):
            a.summary_id = uuid.uuid4()  # type: ignore[misc]

    def test_participant_frozen_slots(self) -> None:
        p = Participant(name="A", reviewed_speaker_id=None)
        with pytest.raises(AttributeError):
            p.name = "B"  # type: ignore[misc]

    def test_topic_frozen_slots(self) -> None:
        t = Topic(title="T", summary=None, source_start_ns=None, source_end_ns=None)
        with pytest.raises(AttributeError):
            t.title = "X"  # type: ignore[misc]

    def test_decision_frozen_slots(self) -> None:
        d = Decision(text="D", source_start_ns=None, source_end_ns=None)
        with pytest.raises(AttributeError):
            d.text = "X"  # type: ignore[misc]

    def test_action_item_frozen_slots(self) -> None:
        a = ActionItem(
            task="T",
            owner=None,
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        with pytest.raises(AttributeError):
            a.task = "X"  # type: ignore[misc]

    def test_risk_frozen_slots(self) -> None:
        r = Risk(text="R", source_start_ns=None, source_end_ns=None)
        with pytest.raises(AttributeError):
            r.text = "X"  # type: ignore[misc]

    def test_open_question_frozen_slots(self) -> None:
        q = OpenQuestion(text="Q", source_start_ns=None, source_end_ns=None)
        with pytest.raises(AttributeError):
            q.text = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MeetingSummary validation
# ---------------------------------------------------------------------------


class TestMeetingSummaryValidation:
    def test_minimal_valid(self) -> None:
        s = _minimal_summary()
        assert s.schema_version == MEETING_SUMMARY_SCHEMA_VERSION
        assert s.prompt_version == 1
        assert s.title is None
        assert s.participants == ()
        assert s.topics == ()

    def test_fully_populated(self) -> None:
        s = _full_summary()
        assert s.title == "Meeting Title"
        assert len(s.participants) == 1
        assert len(s.topics) == 1
        assert len(s.decisions) == 1
        assert len(s.action_items) == 1
        assert len(s.open_questions) == 1
        assert len(s.risks) == 1

    def test_tuple_collections(self) -> None:
        s = _minimal_summary()
        assert isinstance(s.participants, tuple)
        assert isinstance(s.topics, tuple)
        assert isinstance(s.decisions, tuple)
        assert isinstance(s.action_items, tuple)
        assert isinstance(s.open_questions, tuple)
        assert isinstance(s.risks, tuple)

    def test_unicode(self) -> None:
        s = _minimal_summary(summary="日本語のサマリー 🎉")
        assert s.summary == "日本語のサマリー 🎉"

    def test_newlines_preserved(self) -> None:
        s = _minimal_summary(summary="line1\nline2\nline3")
        assert s.summary == "line1\nline2\nline3"

    def test_title_none(self) -> None:
        s = _minimal_summary(title=None)
        assert s.title is None

    def test_empty_collections(self) -> None:
        s = _minimal_summary()
        assert s.participants == ()
        assert s.topics == ()
        assert s.decisions == ()
        assert s.action_items == ()
        assert s.open_questions == ()
        assert s.risks == ()

    def test_schema_version_bool_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="non-bool int"):
            _minimal_summary(schema_version=True)  # type: ignore[arg-type]

    def test_wrong_schema_version(self) -> None:
        with pytest.raises(MeetingSummaryVersionError):
            _minimal_summary(schema_version=999)

    def test_prompt_version_zero_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match=">= 1"):
            _minimal_summary(prompt_version=0)

    def test_prompt_version_negative_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match=">= 1"):
            _minimal_summary(prompt_version=-1)

    def test_prompt_version_bool_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="non-bool int"):
            _minimal_summary(prompt_version=True)  # type: ignore[arg-type]

    def test_required_whitespace_only_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must not be empty"):
            _minimal_summary(summary="   ")

    def test_optional_present_whitespace_only_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must not be empty"):
            _minimal_summary(title="   ")

    def test_leading_trailing_whitespace_preserved(self) -> None:
        s = _minimal_summary(summary="  hello  ")
        assert s.summary == "  hello  "

    def test_null_collection_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must be a tuple"):
            _minimal_summary(participants=None)  # type: ignore[arg-type]

    def test_list_collection_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must be a tuple"):
            _minimal_summary(participants=[])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Participant
# ---------------------------------------------------------------------------


class TestParticipant:
    def test_both_none_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="at least one"):
            Participant(name=None, reviewed_speaker_id=None)

    def test_name_only(self) -> None:
        p = Participant(name="Alice", reviewed_speaker_id=None)
        assert p.name == "Alice"
        assert p.reviewed_speaker_id is None

    def test_speaker_id_only(self) -> None:
        p = Participant(name=None, reviewed_speaker_id=_PID)
        assert p.name is None
        assert p.reviewed_speaker_id == _PID

    def test_both_present(self) -> None:
        p = Participant(name="Alice", reviewed_speaker_id=_PID)
        assert p.name == "Alice"
        assert p.reviewed_speaker_id == _PID

    def test_name_whitespace_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must not be empty"):
            Participant(name="  ", reviewed_speaker_id=_PID)

    def test_speaker_id_with_artifact_no_review_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="no review provenance"):
            _artifact(
                review_id=None,
                review_revision=None,
                summary=_minimal_summary(
                    participants=(Participant(name=None, reviewed_speaker_id=_PID),),
                ),
            )


# ---------------------------------------------------------------------------
# Topic / Decision / ActionItem / Risk / OpenQuestion
# ---------------------------------------------------------------------------


class TestNestedModels:
    def test_topic_valid(self) -> None:
        t = Topic(title="T", summary="S", source_start_ns=0, source_end_ns=100)
        assert t.title == "T"
        assert t.source_start_ns == 0

    def test_topic_timestamps_both_none(self) -> None:
        t = Topic(title="T", summary=None, source_start_ns=None, source_end_ns=None)
        assert t.source_start_ns is None

    def test_topic_timestamp_half_present_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="both"):
            Topic(title="T", summary=None, source_start_ns=0, source_end_ns=None)

    def test_topic_bool_timestamp_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="non-bool int"):
            Topic(title="T", summary=None, source_start_ns=True, source_end_ns=False)  # type: ignore[arg-type]

    def test_topic_negative_timestamp_accepted(self) -> None:
        t = Topic(title="T", summary=None, source_start_ns=-500, source_end_ns=-100)
        assert t.source_start_ns == -500

    def test_topic_end_lt_start_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match=">="):
            Topic(title="T", summary=None, source_start_ns=100, source_end_ns=50)

    def test_decision_valid(self) -> None:
        d = Decision(text="D", source_start_ns=None, source_end_ns=None)
        assert d.text == "D"

    def test_decision_empty_text_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="must not be empty"):
            Decision(text="", source_start_ns=None, source_end_ns=None)

    def test_action_item_owner_none(self) -> None:
        a = ActionItem(
            task="T",
            owner=None,
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.owner is None

    def test_action_item_due_date_none(self) -> None:
        a = ActionItem(
            task="T",
            owner="Bob",
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.due_date is None

    def test_action_item_due_date_present(self) -> None:
        d = datetime.date(2026, 3, 15)
        a = ActionItem(
            task="T",
            owner=None,
            due_date=d,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.due_date == d

    def test_risk_valid(self) -> None:
        r = Risk(text="R", source_start_ns=None, source_end_ns=None)
        assert r.text == "R"

    def test_open_question_valid(self) -> None:
        q = OpenQuestion(text="Q", source_start_ns=None, source_end_ns=None)
        assert q.text == "Q"


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------


class TestArtifactValidation:
    def test_valid_artifact(self) -> None:
        a = _artifact()
        assert a.summary_id == _SID
        assert a.source_review_id == _RID
        assert a.source_review_revision == 4

    def test_review_pair_half_present_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="both be present"):
            _artifact(review_id=_RID, review_revision=None)

    def test_review_pair_half_present_other_reject(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match="both be present"):
            _artifact(review_id=None, review_revision=0)

    def test_source_profile_version_invalid(self) -> None:
        with pytest.raises(MeetingSummaryValidationError, match=">= 1"):
            _artifact(source_profile_version=0)

    def test_created_at_naive_reject(self) -> None:
        naive = datetime.datetime(2026, 1, 1)
        with pytest.raises(MeetingSummaryValidationError, match="timezone-aware"):
            _artifact(created_at=naive)

    def test_aware_non_utc_normalized(self) -> None:
        est = datetime.timezone(datetime.timedelta(hours=-5))
        dt_est = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=est)
        a = _artifact(created_at=dt_est)
        assert a.created_at.utcoffset() == datetime.timedelta(0)
        assert a.created_at.hour == 17  # 12 EST = 17 UTC

    def test_no_review_artifact(self) -> None:
        a = _artifact(review_id=None, review_revision=None)
        assert a.source_review_id is None
        assert a.source_review_revision is None


# ---------------------------------------------------------------------------
# Serialization round trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_dict_round_trip_minimal(self) -> None:
        s = _minimal_summary()
        d = meeting_summary_to_dict(s)
        s2 = meeting_summary_from_dict(d)
        assert s == s2

    def test_dict_round_trip_full(self) -> None:
        s = _full_summary()
        d = meeting_summary_to_dict(s)
        s2 = meeting_summary_from_dict(d)
        assert s == s2

    def test_json_round_trip(self) -> None:
        s = _full_summary()
        j = meeting_summary_to_json(s)
        s2 = meeting_summary_from_json(j)
        assert s == s2

    def test_deterministic_json(self) -> None:
        s = _full_summary()
        j1 = meeting_summary_to_json(s)
        j2 = meeting_summary_to_json(s)
        assert j1 == j2

    def test_json_sorted_keys(self) -> None:
        s = _minimal_summary()
        j = meeting_summary_to_json(s)
        d = json.loads(j)
        keys = list(d.keys())
        assert keys == sorted(keys)

    def test_json_compact_separators(self) -> None:
        s = _minimal_summary()
        j = meeting_summary_to_json(s)
        assert ": " not in j
        assert ", " not in j

    def test_json_utf8(self) -> None:
        s = _minimal_summary(summary="日本語")
        j = meeting_summary_to_json(s)
        assert "日本語" in j
        assert "\\u" not in j

    def test_json_no_nan(self) -> None:
        # allow_nan=False is enforced by json.dumps
        s = _minimal_summary()
        j = meeting_summary_to_json(s)
        assert "NaN" not in j

    def test_uuid_canonical_lowercase(self) -> None:
        s = _minimal_summary(
            participants=(Participant(name=None, reviewed_speaker_id=_PID),),
        )
        d = meeting_summary_to_dict(s)
        assert d["participants"][0]["reviewed_speaker_id"] == str(_PID)

    def test_due_date_isoformat(self) -> None:
        s = _minimal_summary(
            action_items=(
                ActionItem(
                    task="T",
                    owner=None,
                    due_date=datetime.date(2026, 3, 15),
                    source_start_ns=None,
                    source_end_ns=None,
                ),
            ),
        )
        d = meeting_summary_to_dict(s)
        assert d["action_items"][0]["due_date"] == "2026-03-15"

    def test_empty_tuple_serializes_to_empty_list(self) -> None:
        s = _minimal_summary()
        d = meeting_summary_to_dict(s)
        assert d["participants"] == []
        assert d["topics"] == []

    def test_timestamps_as_json_integer(self) -> None:
        s = _minimal_summary(
            topics=(
                Topic(title="T", summary=None, source_start_ns=-500, source_end_ns=100),
            ),
        )
        d = meeting_summary_to_dict(s)
        assert d["topics"][0]["source_start_ns"] == -500
        assert isinstance(d["topics"][0]["source_start_ns"], int)

    def test_nullable_values_explicit_null(self) -> None:
        s = _minimal_summary()
        d = meeting_summary_to_dict(s)
        assert d["title"] is None


# ---------------------------------------------------------------------------
# Deserialization errors
# ---------------------------------------------------------------------------


class TestDeserializationErrors:
    def test_unknown_top_level_field(self) -> None:
        d = meeting_summary_to_dict(_minimal_summary())
        d["unknown_field"] = "value"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown top-level keys"):
            meeting_summary_from_dict(d)

    def test_unknown_participant_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                participants=(Participant(name="A", reviewed_speaker_id=None),),
            )
        )
        d["participants"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown Participant"):
            meeting_summary_from_dict(d)

    def test_unknown_topic_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                topics=(
                    Topic(
                        title="T",
                        summary=None,
                        source_start_ns=None,
                        source_end_ns=None,
                    ),
                ),
            )
        )
        d["topics"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown Topic"):
            meeting_summary_from_dict(d)

    def test_unknown_decision_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                decisions=(
                    Decision(text="D", source_start_ns=None, source_end_ns=None),
                ),
            )
        )
        d["decisions"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown Decision"):
            meeting_summary_from_dict(d)

    def test_unknown_action_item_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                action_items=(
                    ActionItem(
                        task="T",
                        owner=None,
                        due_date=None,
                        source_start_ns=None,
                        source_end_ns=None,
                    ),
                ),
            )
        )
        d["action_items"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown ActionItem"):
            meeting_summary_from_dict(d)

    def test_unknown_open_question_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                open_questions=(
                    OpenQuestion(text="Q", source_start_ns=None, source_end_ns=None),
                ),
            )
        )
        d["open_questions"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown OpenQuestion"):
            meeting_summary_from_dict(d)

    def test_unknown_risk_field(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                risks=(Risk(text="R", source_start_ns=None, source_end_ns=None),),
            )
        )
        d["risks"][0]["bogus"] = "x"
        with pytest.raises(MeetingSummaryDecodeError, match="Unknown Risk"):
            meeting_summary_from_dict(d)

    def test_missing_top_level_key(self) -> None:
        d = meeting_summary_to_dict(_minimal_summary())
        del d["summary"]
        with pytest.raises(MeetingSummaryDecodeError, match="Missing required key"):
            meeting_summary_from_dict(d)

    def test_wrong_type_top_level(self) -> None:
        with pytest.raises(MeetingSummaryDecodeError, match="must be an object"):
            meeting_summary_from_dict("not a dict")

    def test_non_object_json(self) -> None:
        with pytest.raises(MeetingSummaryDecodeError, match="must be an object"):
            meeting_summary_from_json('"hello"')

    def test_malformed_json(self) -> None:
        with pytest.raises(MeetingSummaryDecodeError, match="Malformed JSON"):
            meeting_summary_from_json("{invalid json")

    def test_null_collection_reject_decode(self) -> None:
        d = meeting_summary_to_dict(_minimal_summary())
        d["participants"] = None
        with pytest.raises(MeetingSummaryDecodeError, match="must be an array"):
            meeting_summary_from_dict(d)

    def test_unsupported_schema_version_decode(self) -> None:
        d = meeting_summary_to_dict(_minimal_summary())
        d["schema_version"] = 999
        with pytest.raises(MeetingSummaryVersionError, match="Unsupported"):
            meeting_summary_from_dict(d)

    def test_list_decode_behavior(self) -> None:
        """Lists in JSON decode correctly into tuples."""
        d = meeting_summary_to_dict(_minimal_summary())
        result = meeting_summary_from_dict(d)
        assert isinstance(result.participants, tuple)
        assert isinstance(result.topics, tuple)

    def test_unknown_owner_remains_none(self) -> None:
        """ActionItem with no owner stays None — no inference."""
        a = ActionItem(
            task="T",
            owner=None,
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.owner is None

    def test_unknown_due_date_remains_none(self) -> None:
        """ActionItem with no due_date stays None — no inference."""
        a = ActionItem(
            task="T",
            owner=None,
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.due_date is None

    def test_no_automatic_relative_date_parsing(self) -> None:
        """'next Friday' is not parsed into a date."""
        a = ActionItem(
            task="T",
            owner=None,
            due_date=None,
            source_start_ns=None,
            source_end_ns=None,
        )
        assert a.due_date is None


# ---------------------------------------------------------------------------
# MEDIUM-1 regression: canonical due_date decode
# ---------------------------------------------------------------------------


class TestDueDateCanonicalDecode:
    def test_canonical_date_accepted(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                action_items=(
                    ActionItem(
                        task="T",
                        owner=None,
                        due_date=datetime.date(2026, 9, 4),
                        source_start_ns=None,
                        source_end_ns=None,
                    ),
                ),
            )
        )
        s = meeting_summary_from_dict(d)
        assert s.action_items[0].due_date == datetime.date(2026, 9, 4)

    def test_compact_date_rejected(self) -> None:
        """'20260904' is parseable by fromisoformat but non-canonical."""
        d = meeting_summary_to_dict(
            _minimal_summary(
                action_items=(
                    ActionItem(
                        task="T",
                        owner=None,
                        due_date=datetime.date(2026, 9, 4),
                        source_start_ns=None,
                        source_end_ns=None,
                    ),
                ),
            )
        )
        d["action_items"][0]["due_date"] = "20260904"
        with pytest.raises(MeetingSummaryDecodeError, match="Non-canonical"):
            meeting_summary_from_dict(d)

    def test_whitespace_padded_date_rejected(self) -> None:
        d = meeting_summary_to_dict(
            _minimal_summary(
                action_items=(
                    ActionItem(
                        task="T",
                        owner=None,
                        due_date=datetime.date(2026, 9, 4),
                        source_start_ns=None,
                        source_end_ns=None,
                    ),
                ),
            )
        )
        d["action_items"][0]["due_date"] = " 2026-09-04"
        with pytest.raises(MeetingSummaryDecodeError, match="Non-canonical|Invalid"):
            meeting_summary_from_dict(d)


class _FakeGeneration:
    def __init__(self, gid: uuid.UUID, pv: int) -> None:
        self.generation_id = gid
        self.profile_version = pv


class _FakeReview:
    def __init__(self, rid: uuid.UUID, rev: int) -> None:
        self.id = rid
        self.revision = rev


class TestFreshness:
    def test_exact_match_no_review(self) -> None:
        a = _artifact(review_id=None, review_revision=None)
        gen = _FakeGeneration(_GID, 2)
        assert check_freshness(a, gen, None) is MeetingSummaryFreshness.FRESH

    def test_generation_missing(self) -> None:
        a = _artifact()
        assert check_freshness(a, None, None) is MeetingSummaryFreshness.STALE

    def test_generation_id_mismatch(self) -> None:
        a = _artifact()
        gen = _FakeGeneration(uuid.uuid4(), 2)
        assert check_freshness(a, gen, None) is MeetingSummaryFreshness.STALE

    def test_profile_mismatch(self) -> None:
        a = _artifact()
        gen = _FakeGeneration(_GID, 99)
        assert check_freshness(a, gen, None) is MeetingSummaryFreshness.STALE

    def test_review_match(self) -> None:
        a = _artifact(review_id=_RID, review_revision=4)
        gen = _FakeGeneration(_GID, 2)
        review = _FakeReview(_RID, 4)
        assert check_freshness(a, gen, review) is MeetingSummaryFreshness.FRESH

    def test_review_revision_mismatch(self) -> None:
        a = _artifact(review_id=_RID, review_revision=4)
        gen = _FakeGeneration(_GID, 2)
        review = _FakeReview(_RID, 5)
        assert check_freshness(a, gen, review) is MeetingSummaryFreshness.STALE

    def test_review_id_mismatch(self) -> None:
        a = _artifact(review_id=_RID, review_revision=4)
        gen = _FakeGeneration(_GID, 2)
        review = _FakeReview(uuid.uuid4(), 4)
        assert check_freshness(a, gen, review) is MeetingSummaryFreshness.STALE

    def test_summary_used_review_current_missing(self) -> None:
        a = _artifact(review_id=_RID, review_revision=4)
        gen = _FakeGeneration(_GID, 2)
        assert check_freshness(a, gen, None) is MeetingSummaryFreshness.STALE

    def test_no_review_current_absent(self) -> None:
        a = _artifact(review_id=None, review_revision=None)
        gen = _FakeGeneration(_GID, 2)
        assert check_freshness(a, gen, None) is MeetingSummaryFreshness.FRESH

    def test_no_review_current_appears(self) -> None:
        a = _artifact(review_id=None, review_revision=None)
        gen = _FakeGeneration(_GID, 2)
        review = _FakeReview(_RID, 1)
        assert check_freshness(a, gen, review) is MeetingSummaryFreshness.INDETERMINATE

    def test_freshness_does_not_mutate_artifact(self) -> None:
        a = _artifact()
        original_id = a.summary_id
        original_gen = a.source_generation_id
        check_freshness(a, None, None)
        assert a.summary_id == original_id
        assert a.source_generation_id == original_gen
