"""Tests for strict structured AI meeting-response import."""

from __future__ import annotations

import ast
import datetime
import inspect
import json
import uuid
from collections.abc import Callable

import pytest

from buzz.meeting.meeting_summary import (
    ActionItem,
    Decision,
    MeetingSummary,
    OpenQuestion,
    Participant,
    Risk,
    Topic,
    meeting_summary_to_dict,
    meeting_summary_to_json,
)
from buzz.meeting.meeting_summary_provenance import (
    MeetingSummaryTimestampProvenanceError,
    validate_meeting_summary_timestamp_provenance,
)
from buzz.meeting.portable_ai_response import (
    StructuredAIResponseImportError,
    import_structured_ai_meeting_response,
)
from buzz.meeting.summary_provider import (
    MeetingSummaryRequest,
    MeetingSummaryTranscriptEntry,
)

_IMPORT_ERROR_MESSAGE = "Structured AI meeting response failed strict validation"


def _request(*, prompt_version: int = 1) -> MeetingSummaryRequest:
    return MeetingSummaryRequest(
        schema_version=1,
        prompt_version=prompt_version,
        transcript=(
            MeetingSummaryTranscriptEntry(
                text="First",
                source_start_ns=100,
                source_end_ns=200,
                speaker_name=None,
            ),
            MeetingSummaryTranscriptEntry(
                text="Second",
                source_start_ns=300,
                source_end_ns=400,
                speaker_name="Alice",
            ),
        ),
    )


def _summary(**overrides: object) -> MeetingSummary:
    values: dict[str, object] = {
        "schema_version": 1,
        "prompt_version": 1,
        "title": "规划会议",
        "summary": "团队确认了发布计划。",
        "participants": (Participant(name="艾丽丝", reviewed_speaker_id=None),),
        "topics": (
            Topic(
                title="发布",
                summary="发布范围",
                source_start_ns=100,
                source_end_ns=400,
            ),
        ),
        "decisions": (
            Decision(text="Proceed", source_start_ns=300, source_end_ns=400),
        ),
        "action_items": (
            ActionItem(
                task="Draft",
                owner=None,
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
        ),
        "open_questions": (
            OpenQuestion(text="When?", source_start_ns=100, source_end_ns=200),
        ),
        "risks": (Risk(text="Delay", source_start_ns=None, source_end_ns=None),),
    }
    values.update(overrides)
    return MeetingSummary(**values)  # type: ignore[arg-type]


def _json_with_mutation(mutate: Callable[[dict[str, object]], None]) -> str:
    data = meeting_summary_to_dict(_summary())
    mutate(data)
    return json.dumps(data, ensure_ascii=False)


def _unknown_top_level(data: dict[str, object]) -> None:
    data["unknown"] = "field"


def _unknown_nested(data: dict[str, object]) -> None:
    participants = data["participants"]
    assert isinstance(participants, list)
    participant = participants[0]
    assert isinstance(participant, dict)
    participant["unknown"] = "field"


def test_success_returns_expected_unicode_summary() -> None:
    expected = _summary()

    result = import_structured_ai_meeting_response(
        _request(), meeting_summary_to_json(expected)
    )

    assert result == expected
    assert result.title == "规划会议"
    assert result.summary == "团队确认了发布计划。"
    assert result.participants[0].name == "艾丽丝"


def test_outer_json_whitespace_succeeds() -> None:
    expected = _summary()
    response_text = " \n\t" + meeting_summary_to_json(expected) + "\n "

    assert import_structured_ai_meeting_response(_request(), response_text) == expected


def test_decoder_receives_exact_raw_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response as module

    decoded = _summary()
    raw_response = " \n\tUNIQUE_RAW_RESPONSE_SENTINEL\n "
    received: list[object] = []

    def decode(value: object) -> MeetingSummary:
        received.append(value)
        return decoded

    monkeypatch.setattr(module, "meeting_summary_from_json", decode)

    result = module.import_structured_ai_meeting_response(_request(), raw_response)

    assert received == [raw_response]
    assert received[0] is raw_response
    assert result is decoded


def test_provider_result_validator_receives_exact_request_and_summary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response as module

    request = _request()
    decoded = _summary()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(module, "meeting_summary_from_json", lambda _: decoded)

    def validate(request_arg: object, summary_arg: object) -> MeetingSummary:
        calls.append((request_arg, summary_arg))
        assert summary_arg is decoded
        return decoded

    monkeypatch.setattr(module, "validate_summary_provider_result", validate)

    result = module.import_structured_ai_meeting_response(request, "raw")

    assert calls == [(request, decoded)]
    assert calls[0][0] is request
    assert result is decoded


def test_shared_provenance_validator_receives_exact_request_and_summary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response as module

    request = _request()
    decoded = _summary()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(module, "meeting_summary_from_json", lambda _: decoded)

    def validate(request_arg: object, summary_arg: object) -> None:
        calls.append((request_arg, summary_arg))

    monkeypatch.setattr(
        module,
        "validate_meeting_summary_timestamp_provenance",
        validate,
    )

    result = module.import_structured_ai_meeting_response(request, "raw")

    assert calls == [(request, decoded)]
    assert calls[0][0] is request
    assert calls[0][1] is decoded
    assert result is decoded


@pytest.mark.parametrize(
    "response_text",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("\n\t", id="whitespace"),
        pytest.param(
            "```json\n" + meeting_summary_to_json(_summary()) + "\n```",
            id="markdown-fence",
        ),
        pytest.param(
            "Here is the JSON: " + meeting_summary_to_json(_summary()),
            id="prose-prefix",
        ),
        pytest.param(
            meeting_summary_to_json(_summary()) + "\nHope this helps!",
            id="prose-suffix",
        ),
        pytest.param(
            meeting_summary_to_json(_summary())[:-1] + ",}",
            id="trailing-comma",
        ),
        pytest.param(
            meeting_summary_to_json(_summary()).replace('"', "'"),
            id="single-quotes",
        ),
        pytest.param(meeting_summary_to_json(_summary())[:-1], id="truncated"),
        pytest.param(
            meeting_summary_to_json(_summary())
            + "\n"
            + meeting_summary_to_json(_summary()),
            id="multiple-documents",
        ),
        pytest.param(_json_with_mutation(_unknown_top_level), id="unknown-top-level"),
        pytest.param(_json_with_mutation(_unknown_nested), id="unknown-nested"),
    ],
)
def test_strict_malformed_response_matrix(response_text: str) -> None:
    with pytest.raises(
        StructuredAIResponseImportError,
        match=f"^{_IMPORT_ERROR_MESSAGE}$",
    ):
        import_structured_ai_meeting_response(_request(), response_text)


def test_noncanonical_due_date_rejected_without_normalization() -> None:
    def mutate(data: dict[str, object]) -> None:
        action_items = data["action_items"]
        assert isinstance(action_items, list)
        action_item = action_items[0]
        assert isinstance(action_item, dict)
        action_item["due_date"] = "2026-2-1"

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(_request(), _json_with_mutation(mutate))


def test_wrong_schema_version_rejected_without_rewrite() -> None:
    def mutate(data: dict[str, object]) -> None:
        data["schema_version"] = 2

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(_request(), _json_with_mutation(mutate))


def test_wrong_prompt_version_rejected_without_rewrite() -> None:
    response = meeting_summary_to_json(_summary(prompt_version=2))

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(_request(prompt_version=1), response)


def test_reviewed_speaker_id_rejected_without_clearing() -> None:
    fabricated_id = uuid.uuid4()

    def mutate(data: dict[str, object]) -> None:
        participants = data["participants"]
        assert isinstance(participants, list)
        participant = participants[0]
        assert isinstance(participant, dict)
        participant["reviewed_speaker_id"] = str(fabricated_id)

    response = _json_with_mutation(mutate)

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(_request(), response)
    assert str(fabricated_id) in response


@pytest.mark.parametrize("response_text", [None, b"bytes", {}, []])
def test_wrong_response_type_rejected_without_coercion(response_text: object) -> None:
    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(
            _request(),
            response_text,  # type: ignore[arg-type]
        )


def test_public_error_does_not_leak_raw_response() -> None:
    sentinel = "SUPER_SECRET_RESPONSE_SENTINEL"

    with pytest.raises(StructuredAIResponseImportError) as caught:
        import_structured_ai_meeting_response(_request(), sentinel)

    assert str(caught.value) == _IMPORT_ERROR_MESSAGE
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert caught.value.__cause__ is not None


def test_fabricated_start_boundary_rejected() -> None:
    result = _summary(
        topics=(
            Topic(
                title="Plan",
                summary=None,
                source_start_ns=150,
                source_end_ns=400,
            ),
        )
    )

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(result)
        )


def test_fabricated_end_boundary_rejected() -> None:
    result = _summary(
        topics=(
            Topic(
                title="Plan",
                summary=None,
                source_start_ns=100,
                source_end_ns=350,
            ),
        )
    )

    with pytest.raises(StructuredAIResponseImportError):
        import_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(result)
        )


def test_cross_entry_boundary_span_succeeds() -> None:
    result = _summary(
        topics=(
            Topic(
                title="Plan",
                summary=None,
                source_start_ns=100,
                source_end_ns=400,
            ),
        )
    )

    assert (
        import_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(result)
        )
        == result
    )


def test_null_timestamp_pair_succeeds() -> None:
    result = _summary(
        topics=(
            Topic(
                title="Plan",
                summary=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
        )
    )

    assert (
        import_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(result)
        )
        == result
    )


def test_shared_provenance_error_is_provider_independent_and_safe() -> None:
    result = _summary(
        risks=(Risk(text="Risk", source_start_ns=100, source_end_ns=350),)
    )

    with pytest.raises(
        MeetingSummaryTimestampProvenanceError,
        match="^MeetingSummary timestamp is outside the supplied transcript boundaries$",
    ):
        validate_meeting_summary_timestamp_provenance(_request(), result)


def test_public_apis() -> None:
    import buzz.meeting.meeting_summary_provenance as provenance_module
    import buzz.meeting.portable_ai_response as response_module

    assert provenance_module.__all__ == [
        "MeetingSummaryTimestampProvenanceError",
        "validate_meeting_summary_timestamp_provenance",
    ]
    assert response_module.__all__ == [
        "StructuredAIResponseImportError",
        "import_structured_ai_meeting_response",
    ]


def test_portable_importer_has_no_local_provenance_or_repair_logic() -> None:
    import buzz.meeting.portable_ai_response as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    local_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert local_functions == {"import_structured_ai_meeting_response"}
    assert not any(isinstance(node, ast.SetComp) for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"source_start_ns", "source_end_ns"}
        for node in ast.walk(tree)
    )
    assert ".strip(" not in source
    assert "json.loads" not in source
    assert "meeting_summary_from_dict" not in source
    assert "repair" not in source.lower()


@pytest.mark.parametrize(
    ("module_name", "allowed_imports"),
    [
        pytest.param(
            "buzz.meeting.portable_ai_response",
            {
                "__future__",
                "buzz.meeting.meeting_summary",
                "buzz.meeting.meeting_summary_provenance",
                "buzz.meeting.summary_provider",
            },
            id="portable-response",
        ),
        pytest.param(
            "buzz.meeting.meeting_summary_provenance",
            {
                "__future__",
                "buzz.meeting.meeting_summary",
                "buzz.meeting.summary_provider",
            },
            id="shared-provenance",
        ),
    ],
)
def test_import_isolation(module_name: str, allowed_imports: set[str]) -> None:
    module = __import__(module_name, fromlist=["*"])
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert imports == allowed_imports
    assert {
        "open",
        "Path",
        "QClipboard",
        "QApplication",
        "QSqlDatabase",
        "Settings",
        "MeetingSummaryArtifact",
        "MeetingSummaryRepository",
        "OpenAICompatibleProvider",
    }.isdisjoint(names)


def test_due_date_round_trip_succeeds() -> None:
    expected = _summary(
        action_items=(
            ActionItem(
                task="Ship",
                owner="Alice",
                due_date=datetime.date(2026, 9, 1),
                source_start_ns=300,
                source_end_ns=400,
            ),
        )
    )

    assert (
        import_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(expected)
        )
        == expected
    )
