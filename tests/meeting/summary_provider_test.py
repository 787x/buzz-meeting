"""Pure tests for the summary-provider domain.

No Qt, DB, filesystem, network, or QApplication.  Only
deterministic, pure-Python assertions.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
import uuid

import pytest

from buzz.meeting.meeting_summary import (
    MEETING_SUMMARY_SCHEMA_VERSION,
    MeetingSummary,
    MeetingSummaryArtifact,
    Participant,
)
from buzz.meeting.summary_provider import (
    MeetingSummaryRequest,
    MeetingSummaryTranscriptEntry,
    SummaryProvider,
    SummaryProviderConfigurationError,
    SummaryProviderError,
    SummaryProviderRequestError,
    SummaryProviderResponseError,
    SummaryProviderTransportError,
    validate_summary_provider_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    text: str = "Hello world",
    start_ns: int = 0,
    end_ns: int = 1000,
    speaker_name: str | None = None,
) -> MeetingSummaryTranscriptEntry:
    return MeetingSummaryTranscriptEntry(
        text=text,
        source_start_ns=start_ns,
        source_end_ns=end_ns,
        speaker_name=speaker_name,
    )


def _make_request(
    entries: tuple[MeetingSummaryTranscriptEntry, ...] | None = None,
    schema_version: int = MEETING_SUMMARY_SCHEMA_VERSION,
    prompt_version: int = 1,
) -> MeetingSummaryRequest:
    if entries is None:
        entries = (_make_entry(),)
    return MeetingSummaryRequest(
        schema_version=schema_version,
        prompt_version=prompt_version,
        transcript=entries,
    )


def _make_summary(
    schema_version: int = MEETING_SUMMARY_SCHEMA_VERSION,
    prompt_version: int = 1,
    participants: tuple[Participant, ...] | None = None,
) -> MeetingSummary:
    if participants is None:
        participants = (Participant(name="Alice", reviewed_speaker_id=None),)
    return MeetingSummary(
        schema_version=schema_version,
        prompt_version=prompt_version,
        title="Test summary",
        summary="Summary text",
        participants=participants,
        topics=(),
        decisions=(),
        action_items=(),
        open_questions=(),
        risks=(),
    )


# =========================================================================
# §33  Transcript entry tests
# =========================================================================


class TestMeetingSummaryTranscriptEntry:
    # -- M1-A: exact field set --
    def test_exact_field_set(self) -> None:
        fields = {f.name for f in dataclasses.fields(MeetingSummaryTranscriptEntry)}
        assert fields == {"text", "source_start_ns", "source_end_ns", "speaker_name"}

    # -- L1: immutability with resilient exception handling --
    def test_existing_field_mutation_blocked(self) -> None:
        entry = _make_entry(text="original")
        with pytest.raises((AttributeError, TypeError)):
            entry.text = "changed"  # type: ignore[misc]
        assert entry.text == "original"

    def test_new_field_assignment_blocked(self) -> None:
        entry = _make_entry()
        with pytest.raises((AttributeError, TypeError)):
            entry.nonexistent = "value"  # type: ignore[attr-defined]
        assert not hasattr(entry, "nonexistent")

    def test_valid_basic(self) -> None:
        entry = _make_entry()
        assert entry.text == "Hello world"
        assert entry.source_start_ns == 0
        assert entry.source_end_ns == 1000
        assert entry.speaker_name is None

    def test_unicode_preserved(self) -> None:
        entry = _make_entry(text="会议内容 こんにちは 🎉")
        assert entry.text == "会议内容 こんにちは 🎉"

    def test_newline_preserved(self) -> None:
        entry = _make_entry(text="line1\nline2\nline3")
        assert entry.text == "line1\nline2\nline3"

    def test_leading_trailing_preserved(self) -> None:
        entry = _make_entry(text="  padded  ")
        assert entry.text == "  padded  "

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="empty/whitespace"):
            _make_entry(text="   ")

    def test_negative_start_accepted(self) -> None:
        entry = _make_entry(start_ns=-100, end_ns=0)
        assert entry.source_start_ns == -100

    def test_negative_end_accepted(self) -> None:
        entry = _make_entry(start_ns=-200, end_ns=-100)
        assert entry.source_end_ns == -100

    def test_end_lt_start_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match=">= source_start_ns"):
            _make_entry(start_ns=1000, end_ns=999)

    def test_bool_start_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="non-bool int"):
            _make_entry(start_ns=True, end_ns=1000)  # type: ignore[arg-type]

    def test_bool_end_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="non-bool int"):
            _make_entry(start_ns=0, end_ns=False)  # type: ignore[arg-type]

    def test_speaker_name_none_accepted(self) -> None:
        entry = _make_entry(speaker_name=None)
        assert entry.speaker_name is None

    def test_speaker_name_explicit_preserved(self) -> None:
        entry = _make_entry(speaker_name="Alice")
        assert entry.speaker_name == "Alice"

    def test_speaker_name_whitespace_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="empty/whitespace"):
            _make_entry(speaker_name="   ")


# =========================================================================
# §34  Request tests
# =========================================================================


class TestMeetingSummaryRequest:
    # -- M1-A: exact field set --
    def test_exact_field_set(self) -> None:
        fields = {f.name for f in dataclasses.fields(MeetingSummaryRequest)}
        assert fields == {"schema_version", "prompt_version", "transcript"}

    # -- L1: immutability with resilient exception handling --
    def test_existing_field_mutation_blocked(self) -> None:
        req = _make_request()
        with pytest.raises((AttributeError, TypeError)):
            req.schema_version = 99  # type: ignore[misc]
        assert req.schema_version == MEETING_SUMMARY_SCHEMA_VERSION

    def test_new_field_assignment_blocked(self) -> None:
        req = _make_request()
        with pytest.raises((AttributeError, TypeError)):
            req.nonexistent = "value"  # type: ignore[attr-defined]
        assert not hasattr(req, "nonexistent")

    def test_tuple_transcript_required(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="tuple"):
            MeetingSummaryRequest(
                schema_version=MEETING_SUMMARY_SCHEMA_VERSION,
                prompt_version=1,
                transcript=[_make_entry()],  # type: ignore[arg-type]
            )

    def test_empty_tuple_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="not be empty"):
            _make_request(entries=())

    def test_non_entry_item_rejected(self) -> None:
        with pytest.raises(
            SummaryProviderRequestError, match="MeetingSummaryTranscriptEntry"
        ):
            _make_request(entries=("not an entry",))  # type: ignore[arg-type]

    def test_current_schema_version_accepted(self) -> None:
        req = _make_request()
        assert req.schema_version == MEETING_SUMMARY_SCHEMA_VERSION

    def test_unsupported_schema_version_rejected(self) -> None:
        with pytest.raises(
            SummaryProviderRequestError, match="Unsupported schema_version"
        ):
            _make_request(schema_version=9999)

    def test_schema_bool_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="non-bool int"):
            _make_request(schema_version=True)  # type: ignore[arg-type]

    def test_prompt_version_1_accepted(self) -> None:
        req = _make_request(prompt_version=1)
        assert req.prompt_version == 1

    def test_prompt_version_0_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match=">= 1"):
            _make_request(prompt_version=0)

    def test_prompt_version_negative_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match=">= 1"):
            _make_request(prompt_version=-1)

    def test_prompt_bool_rejected(self) -> None:
        with pytest.raises(SummaryProviderRequestError, match="non-bool int"):
            _make_request(prompt_version=True)  # type: ignore[arg-type]

    def test_chronological_transcript_accepted(self) -> None:
        entries = (
            _make_entry(start_ns=0, end_ns=1000),
            _make_entry(start_ns=1000, end_ns=2000),
            _make_entry(start_ns=2000, end_ns=3000),
        )
        req = _make_request(entries=entries)
        assert len(req.transcript) == 3

    def test_equal_start_timestamps_accepted(self) -> None:
        entries = (
            _make_entry(start_ns=500, end_ns=1000),
            _make_entry(start_ns=500, end_ns=1500),
        )
        req = _make_request(entries=entries)
        assert len(req.transcript) == 2

    # -- M1-B: equal-start order identity preservation --
    def test_equal_start_order_preserved_both_permutations(self) -> None:
        """Regression: if implementation sorts equal-start entries by any
        secondary key (text, end, speaker), one of these permutations
        will fail.  Both must preserve caller order by identity."""
        e_zulu = _make_entry(start_ns=100, end_ns=300, text="zulu", speaker_name="Zulu")
        e_alpha = _make_entry(
            start_ns=100, end_ns=200, text="alpha", speaker_name="Alpha"
        )
        for first, second in ((e_zulu, e_alpha), (e_alpha, e_zulu)):
            req = _make_request(entries=(first, second))
            assert req.transcript[0] is first
            assert req.transcript[1] is second

    def test_start_time_regression_rejected(self) -> None:
        entries = (
            _make_entry(start_ns=2000, end_ns=3000),
            _make_entry(start_ns=1000, end_ns=2000),
        )
        with pytest.raises(
            SummaryProviderRequestError, match="chronological regression"
        ):
            _make_request(entries=entries)

    def test_request_does_not_reorder(self) -> None:
        e1 = _make_entry(start_ns=0, end_ns=100, text="first")
        e2 = _make_entry(start_ns=100, end_ns=200, text="second")
        req = _make_request(entries=(e1, e2))
        assert req.transcript[0] is e1
        assert req.transcript[1] is e2


# =========================================================================
# §35  Internal-ID absence tests
# =========================================================================


class TestInternalIdAbsence:
    """Structurally verify request/entry dataclass fields do NOT include
    internal identity fields."""

    _FORBIDDEN_FIELDS = {
        "meeting_id",
        "generation_id",
        "review_id",
        "review_revision",
        "speaker_id",
        "reviewed_speaker_id",
        "machine_speaker",
        "source_role",
        "track_id",
    }

    def test_entry_no_internal_ids(self) -> None:
        fields = {f.name for f in dataclasses.fields(MeetingSummaryTranscriptEntry)}
        leaked = fields & self._FORBIDDEN_FIELDS
        assert not leaked, f"Entry has forbidden fields: {leaked}"

    def test_request_no_internal_ids(self) -> None:
        fields = {f.name for f in dataclasses.fields(MeetingSummaryRequest)}
        leaked = fields & self._FORBIDDEN_FIELDS
        assert not leaked, f"Request has forbidden fields: {leaked}"


# =========================================================================
# §36  Speaker-name responsibility test
# =========================================================================


class TestSpeakerNameResponsibility:
    """Document that speaker_name structural validation does NOT implement
    string-pattern rejection.  Semantic provenance is the future request
    assembler's responsibility."""

    def test_none_valid(self) -> None:
        entry = _make_entry(speaker_name=None)
        assert entry.speaker_name is None

    def test_nonempty_explicit_text_structurally_valid(self) -> None:
        entry = _make_entry(speaker_name="Alice")
        assert entry.speaker_name == "Alice"

    # -- M1-C: prove no brittle blacklist exists --
    @pytest.mark.parametrize(
        "name",
        [
            "Speaker 1",
            "Speaker 2",
            "MICROPHONE:0",
            "REMOTE:1",
        ],
    )
    def test_machine_label_structurally_accepted(self, name: str) -> None:
        """These strings must be accepted by the DTO.  If production
        later adds a blacklist, this test fails."""
        entry = _make_entry(speaker_name=name)
        assert entry.speaker_name == name


# =========================================================================
# §37  Protocol fake test
# =========================================================================


class _FakeProvider:
    """Test-only fake executable provider."""

    def summarize(self, request: MeetingSummaryRequest) -> MeetingSummary:
        return _make_summary()


class TestProtocolFake:
    def test_protocol_shape_usable(self) -> None:
        provider: SummaryProvider = _FakeProvider()
        req = _make_request()
        result = provider.summarize(req)
        assert isinstance(result, MeetingSummary)

    def test_protocol_is_not_runtime_checkable(self) -> None:
        # Without @runtime_checkable, isinstance() against the protocol
        # should raise TypeError.
        with pytest.raises(TypeError):
            isinstance(_FakeProvider(), SummaryProvider)

    # -- M1-D: lock protocol annotations exactly --
    def test_summarize_signature_parameters(self) -> None:
        sig = inspect.signature(SummaryProvider.summarize)
        params = list(sig.parameters.keys())
        assert params == ["self", "request"]

    def test_summarize_request_annotation(self) -> None:
        hints = typing.get_type_hints(SummaryProvider.summarize)
        assert hints["request"] is MeetingSummaryRequest

    def test_summarize_return_annotation(self) -> None:
        hints = typing.get_type_hints(SummaryProvider.summarize)
        assert hints["return"] is MeetingSummary


# =========================================================================
# §38  Result validation tests
# =========================================================================


class TestValidateSummaryProviderResult:
    def test_matching_result_accepted(self) -> None:
        req = _make_request()
        result = _make_summary()
        validated = validate_summary_provider_result(req, result)
        assert validated is result

    def test_same_object_returned(self) -> None:
        req = _make_request()
        result = _make_summary()
        assert validate_summary_provider_result(req, result) is result

    def test_wrong_schema_version_rejected(self) -> None:
        req = _make_request()
        # Build a result with mismatched schema_version by bypassing
        # MeetingSummary's own validation.
        result = _make_summary()
        object.__setattr__(result, "schema_version", 9999)
        with pytest.raises(SummaryProviderResponseError, match="schema_version"):
            validate_summary_provider_result(req, result)

    def test_wrong_prompt_version_rejected(self) -> None:
        req = _make_request()
        result = _make_summary()
        object.__setattr__(result, "prompt_version", 99)
        with pytest.raises(SummaryProviderResponseError, match="prompt_version"):
            validate_summary_provider_result(req, result)

    def test_non_meeting_summary_rejected(self) -> None:
        req = _make_request()
        with pytest.raises(SummaryProviderResponseError, match="MeetingSummary"):
            validate_summary_provider_result(req, "not a summary")  # type: ignore[arg-type]

    def test_fabricated_reviewed_speaker_id_rejected(self) -> None:
        req = _make_request()
        participant = Participant(name="Alice", reviewed_speaker_id=None)
        # Fabricate a reviewed_speaker_id by bypassing frozen.
        object.__setattr__(participant, "reviewed_speaker_id", uuid.uuid4())
        result = _make_summary(participants=(participant,))
        with pytest.raises(SummaryProviderResponseError, match="reviewed_speaker_id"):
            validate_summary_provider_result(req, result)

    def test_name_with_none_reviewed_speaker_id_accepted(self) -> None:
        req = _make_request()
        participant = Participant(name="Alice", reviewed_speaker_id=None)
        result = _make_summary(participants=(participant,))
        validated = validate_summary_provider_result(req, result)
        assert validated.participants[0].name == "Alice"
        assert validated.participants[0].reviewed_speaker_id is None

    # -- M1-E: MeetingSummaryArtifact explicitly rejected at boundary --
    def test_meeting_summary_artifact_rejected(self) -> None:
        """The persistence envelope must not cross the provider
        result boundary."""
        import datetime

        req = _make_request()
        artifact = MeetingSummaryArtifact(
            summary_id=uuid.uuid4(),
            meeting_id=uuid.uuid4(),
            source_generation_id=uuid.uuid4(),
            source_profile_version=1,
            source_review_id=None,
            source_review_revision=None,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            summary=_make_summary(),
        )
        with pytest.raises(SummaryProviderResponseError, match="MeetingSummary"):
            validate_summary_provider_result(req, artifact)  # type: ignore[arg-type]


# =========================================================================
# M1-F  Public API boundary
# =========================================================================


_EXPECTED_PUBLIC_API = frozenset(
    {
        "MeetingSummaryTranscriptEntry",
        "MeetingSummaryRequest",
        "SummaryProvider",
        "SummaryProviderError",
        "SummaryProviderConfigurationError",
        "SummaryProviderTransportError",
        "SummaryProviderRequestError",
        "SummaryProviderResponseError",
        "validate_summary_provider_result",
    }
)


def _production_public_names() -> set[str]:
    """Collect module-level public names, excluding re-exported
    support symbols and names starting with '_'."""
    import buzz.meeting.summary_provider as mod

    all_declared = getattr(mod, "__all__", None)
    if all_declared is not None:
        return set(all_declared)
    # Fallback: inspect module attributes, ignoring private names
    # and imported standard-library / typing symbols.
    names = set()
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            continue
        # Skip re-exported support types from meeting_summary
        if hasattr(obj, "__module__") and getattr(obj, "__module__", "") not in (
            mod.__name__,
            "",
        ):
            continue
        names.add(name)
    return names


class TestPublicApiBoundary:
    """M1-F: exact public API protection and scope creep detection."""

    def test_exact_public_api_matches(self) -> None:
        actual = _production_public_names()
        assert actual == _EXPECTED_PUBLIC_API, (
            f"Public API changed: added={actual - _EXPECTED_PUBLIC_API}, "
            f"removed={_EXPECTED_PUBLIC_API - actual}"
        )

    # -- Public API scope protection: prohibited concepts --
    def test_no_request_to_json(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "request_to_json"
        ), "request_to_json must not exist (PR21 owns serialization)"

    def test_no_request_to_dict(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "request_to_dict"
        ), "request_to_dict must not exist (PR21 owns serialization)"

    def test_no_request_builder(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "MeetingSummaryRequestBuilder"
        ), "Request builder must not exist in PR19"

    def test_no_meeting_context(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "MeetingSummaryMeetingContext"
        ), "Meeting context must not exist in PR19 request"

    def test_no_meeting_summary_artifact_in_production(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "MeetingSummaryArtifact"
        ), "MeetingSummaryArtifact must not be re-exported from provider module"

    def test_no_manual_provider(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "ManualProvider"
        ), "ManualProvider must not exist in PR19"

    def test_no_pending_result(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(mod, "PendingResult"), "PendingResult must not exist in PR19"

    def test_no_registry(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "SummaryProviderRegistry"
        ), "Provider registry must not exist in PR19"

    def test_no_provider_config(self) -> None:
        import buzz.meeting.summary_provider as mod

        assert not hasattr(
            mod, "SummaryProviderConfig"
        ), "Provider config must not exist in PR19"


# =========================================================================
# §39  Error hierarchy tests
# =========================================================================


class TestErrorHierarchy:
    def test_configuration_is_provider_error(self) -> None:
        assert issubclass(SummaryProviderConfigurationError, SummaryProviderError)

    def test_transport_is_provider_error(self) -> None:
        assert issubclass(SummaryProviderTransportError, SummaryProviderError)

    def test_request_is_provider_error(self) -> None:
        assert issubclass(SummaryProviderRequestError, SummaryProviderError)

    def test_response_is_provider_error(self) -> None:
        assert issubclass(SummaryProviderResponseError, SummaryProviderError)


# =========================================================================
# §40  Import isolation test
# =========================================================================


class TestImportIsolation:
    """Ensure summary_provider.py does not import forbidden modules."""

    _FORBIDDEN_MODULES = frozenset(
        {
            "PyQt5",
            "PyQt6",
            "PySide2",
            "PySide6",
            "PyQt5.QtCore",
            "PyQt6.QtCore",
            "PySide2.QtCore",
            "PySide6.QtCore",
            "qtpy",
            "PyQt5.QtSql",
            "PyQt6.QtSql",
            "PySide2.QtSql",
            "PySide6.QtSql",
            "requests",
            "httpx",
            "openai",
            "socket",
            "keyring",
        }
    )

    def test_no_forbidden_imports(self) -> None:
        import importlib

        mod = importlib.import_module("buzz.meeting.summary_provider")
        source = inspect.getsource(mod)
        for name in self._FORBIDDEN_MODULES:
            # Check for import statements, not just documentation mentions
            assert (
                f"import {name}" not in source
            ), f"summary_provider.py imports forbidden module: {name}"
