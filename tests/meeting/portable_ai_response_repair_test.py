"""Tests for bounded structured AI meeting-response representation repair."""

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
from buzz.meeting.portable_ai_response import StructuredAIResponseImportError
from buzz.meeting.portable_ai_response_repair import (
    StructuredAIResponseRepairError,
    import_repaired_structured_ai_meeting_response,
)
from buzz.meeting.summary_provider import (
    MeetingSummaryRequest,
    MeetingSummaryTranscriptEntry,
)

_REPAIR_ERROR_MESSAGE = "Structured AI meeting response could not be repaired safely"


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


def _assert_repair_error(response_text: object) -> None:
    with pytest.raises(
        StructuredAIResponseRepairError,
        match=f"^{_REPAIR_ERROR_MESSAGE}$",
    ):
        import_repaired_structured_ai_meeting_response(
            _request(),
            response_text,  # type: ignore[arg-type]
        )


def test_strict_valid_response_returns_same_semantic_summary() -> None:
    expected = _summary()

    assert (
        import_repaired_structured_ai_meeting_response(
            _request(), meeting_summary_to_json(expected)
        )
        == expected
    )


def test_strict_success_calls_pr22_once_with_original_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    request = _request()
    response_text = " \n\tORIGINAL_RESPONSE_OBJECT\r\n"
    sentinel = _summary()
    calls: list[tuple[object, object]] = []

    def strict_import(request_arg: object, response_arg: object) -> MeetingSummary:
        calls.append((request_arg, response_arg))
        return sentinel

    def forbidden_repair(_: object) -> None:
        raise AssertionError("repair must not run after strict success")

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)
    monkeypatch.setattr(module, "_repair_response_representation", forbidden_repair)

    result = module.import_repaired_structured_ai_meeting_response(
        request, response_text
    )

    assert result is sentinel
    assert calls == [(request, response_text)]
    assert calls[0][0] is request
    assert calls[0][1] is response_text


def test_valid_json_semantic_failure_is_reraised_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    original_error = StructuredAIResponseImportError("semantic rejection")
    response_text = ' \n {"already":"valid"}\r\n '
    calls: list[object] = []

    def strict_import(_: object, response_arg: object) -> MeetingSummary:
        calls.append(response_arg)
        raise original_error

    def forbidden_repair(_: object) -> None:
        raise AssertionError("repair must not run for a complete JSON document")

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)
    monkeypatch.setattr(module, "_repair_response_representation", forbidden_repair)

    with pytest.raises(StructuredAIResponseImportError) as caught:
        module.import_repaired_structured_ai_meeting_response(_request(), response_text)

    assert caught.value is original_error
    assert calls == [response_text]


def test_repaired_success_calls_pr22_with_exact_request_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    request = _request()
    candidate = '{\n  "z": "雪☃ { } \\"quote\\" \\\\ path \\n",\n  "a": 1\n}'
    response_text = f"Here is the exact object:\n{candidate}\nThanks."
    sentinel = _summary()
    calls: list[tuple[object, object]] = []

    def strict_import(request_arg: object, response_arg: object) -> MeetingSummary:
        calls.append((request_arg, response_arg))
        if len(calls) == 1:
            raise StructuredAIResponseImportError("strict failure")
        return sentinel

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)

    result = module.import_repaired_structured_ai_meeting_response(
        request, response_text
    )

    assert result is sentinel
    assert calls == [(request, response_text), (request, candidate)]
    assert all(call[0] is request for call in calls)
    assert calls[0][1] is response_text


def test_repaired_candidate_is_already_a_strict_json_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    candidate = '{"one":1}'
    response_text = f"Result:\n{candidate}\nDone."
    sentinel = _summary()
    calls: list[str] = []

    def strict_import(_: object, response_arg: str) -> MeetingSummary:
        calls.append(response_arg)
        if response_arg != candidate:
            raise StructuredAIResponseImportError("strict failure")
        return sentinel

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)

    assert (
        module.import_repaired_structured_ai_meeting_response(_request(), response_text)
        is sentinel
    )
    assert calls == [response_text, candidate]


def test_one_leading_bom_is_repaired() -> None:
    expected = _summary()

    assert (
        import_repaired_structured_ai_meeting_response(
            _request(), "\ufeff" + meeting_summary_to_json(expected)
        )
        == expected
    )


def test_two_leading_boms_are_rejected() -> None:
    _assert_repair_error("\ufeff\ufeff" + meeting_summary_to_json(_summary()))


def test_bom_inside_json_string_is_preserved() -> None:
    expected = _summary(summary="before\ufeffafter")

    result = import_repaired_structured_ai_meeting_response(
        _request(), meeting_summary_to_json(expected)
    )

    assert result == expected
    assert result.summary == "before\ufeffafter"


def test_nonleading_wrapper_bom_is_rejected() -> None:
    _assert_repair_error(
        "prefix\ufeff\n" + meeting_summary_to_json(_summary()) + "\nsuffix"
    )


@pytest.mark.parametrize(
    "opening",
    ["```", "```json", "```JSON", "```Json", "```jSoN"],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_allowed_fence_forms_are_repaired(opening: str, newline: str) -> None:
    expected = _summary()
    response_text = (
        opening + newline + meeting_summary_to_json(expected) + newline + "```"
    )

    assert (
        import_repaired_structured_ai_meeting_response(_request(), response_text)
        == expected
    )


def test_fence_payload_newline_style_and_text_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    payload = '\r\n { "unicode": "雪", "ordered": 1 }\r\n'
    response_text = "```JSON\r\n" + payload + "```\r\n"
    sentinel = _summary()
    calls: list[str] = []

    def strict_import(_: object, response_arg: str) -> MeetingSummary:
        calls.append(response_arg)
        if len(calls) == 1:
            raise StructuredAIResponseImportError("strict failure")
        return sentinel

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)

    assert (
        module.import_repaired_structured_ai_meeting_response(_request(), response_text)
        is sentinel
    )
    assert calls == [response_text, payload]


@pytest.mark.parametrize("opening", ["~~~json", "~~~JSON", "~~~"])
def test_tilde_opening_fences_are_rejected(opening: str) -> None:
    response_text = opening + "\n" + meeting_summary_to_json(_summary()) + "\n```"

    _assert_repair_error(response_text)


@pytest.mark.parametrize(
    "response_text",
    [
        "``` json\n{}\n```",
        "```\tjson\n{}\n```",
        "```JSON5\n{}\n```",
        "```javascript\n{}\n```",
        "````json\n{}\n```",
        "```json\n{}",
        "```json\n{}\n~~~",
        "```json\n```\n{}\n```",
        "```json\n{}\n```\n```text\nother\n```",
    ],
)
def test_malformed_missing_multiple_and_nested_fences_are_rejected(
    response_text: str,
) -> None:
    _assert_repair_error(response_text)


@pytest.mark.parametrize(
    "response_text",
    [
        "prefix\n" + meeting_summary_to_json(_summary()),
        meeting_summary_to_json(_summary()) + "\nsuffix",
        "prefix\n" + meeting_summary_to_json(_summary()) + "\nsuffix",
    ],
)
def test_single_object_surrounded_by_prose_is_repaired(response_text: str) -> None:
    assert (
        import_repaired_structured_ai_meeting_response(_request(), response_text)
        == _summary()
    )


def test_prose_plus_fence_is_repaired() -> None:
    response_text = (
        "Here you go:\n```json\n"
        + meeting_summary_to_json(_summary())
        + "\n```\nThanks!"
    )

    assert (
        import_repaired_structured_ai_meeting_response(_request(), response_text)
        == _summary()
    )


def test_bom_plus_prose_plus_fence_is_repaired() -> None:
    response_text = (
        "\ufeffHere you go:\r\n```JSON\r\n"
        + meeting_summary_to_json(_summary())
        + "\r\n```\r\nThanks!"
    )

    assert (
        import_repaired_structured_ai_meeting_response(_request(), response_text)
        == _summary()
    )


def test_braces_quotes_backslashes_newline_escape_and_unicode_do_not_confuse_scan() -> (
    None
):
    expected = _summary(
        summary='literal { and } and {not an object}; "quote"; C:\\temp\\x; \\n; 雪'
    )
    response_text = "Here:\n" + meeting_summary_to_json(expected) + "\nDone."

    result = import_repaired_structured_ai_meeting_response(_request(), response_text)

    assert result == expected
    assert result.summary == expected.summary


@pytest.mark.parametrize(
    "response_text",
    [
        meeting_summary_to_json(_summary())
        + "\n"
        + meeting_summary_to_json(_summary(title="Second")),
        "{}\n" + meeting_summary_to_json(_summary()),
        '{"example":true}\n' + meeting_summary_to_json(_summary()),
    ],
)
def test_multiple_distinct_complete_objects_are_rejected(response_text: str) -> None:
    _assert_repair_error(response_text)


@pytest.mark.parametrize(
    "response_text",
    [
        '[{"nested": true}',
        "{ malformed wrapper: " + meeting_summary_to_json(_summary()),
        meeting_summary_to_json(_summary()) + "]",
        "prefix { " + meeting_summary_to_json(_summary()) + " suffix",
    ],
)
def test_structural_noise_does_not_salvage_nested_object(
    response_text: str,
) -> None:
    _assert_repair_error(response_text)


def test_complete_array_wrapper_remains_a_strict_import_error() -> None:
    response_text = "[" + meeting_summary_to_json(_summary()) + "]"

    with pytest.raises(StructuredAIResponseImportError):
        import_repaired_structured_ai_meeting_response(_request(), response_text)


@pytest.mark.parametrize(
    "payload",
    [
        '{"foo": 1,}',
        "{'foo': 1}",
        '{"foo": 1 // comment\n}',
        '{/* comment */ "foo": 1}',
        "{foo: 1}",
        '{"foo": None}',
        '{"foo": True}',
        '{"foo": False}',
        '{"foo": NaN}',
        '{"foo": Infinity}',
        '{"foo": -Infinity}',
        "{foo: 'json5'}",
        "{“foo”: 1}",
        '{"foo": "bad\\q"}',
        '{"foo": 1',
        '{"foo": "missing quote}',
    ],
)
def test_tier_two_malformed_json_is_not_repaired(payload: str) -> None:
    _assert_repair_error("```json\n" + payload + "\n```")


@pytest.mark.parametrize("response_text", [None, b"bytes", {}, []])
def test_non_string_response_is_consistently_a_repair_error(
    response_text: object,
) -> None:
    _assert_repair_error(response_text)


def _wrong_schema(data: dict[str, object]) -> None:
    data["schema_version"] = 2


def _wrong_prompt(data: dict[str, object]) -> None:
    data["prompt_version"] = 2


def _reviewed_speaker(data: dict[str, object]) -> None:
    participants = data["participants"]
    assert isinstance(participants, list)
    participant = participants[0]
    assert isinstance(participant, dict)
    participant["reviewed_speaker_id"] = str(uuid.uuid4())


def _fabricated_timestamp(data: dict[str, object]) -> None:
    topics = data["topics"]
    assert isinstance(topics, list)
    topic = topics[0]
    assert isinstance(topic, dict)
    topic["source_start_ns"] = 150


def _unknown_field(data: dict[str, object]) -> None:
    data["unknown"] = "must remain"


def _noncanonical_date(data: dict[str, object]) -> None:
    action_items = data["action_items"]
    assert isinstance(action_items, list)
    action_item = action_items[0]
    assert isinstance(action_item, dict)
    action_item["due_date"] = "20260904"


def _missing_field(data: dict[str, object]) -> None:
    del data["risks"]


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_schema,
        _wrong_prompt,
        _reviewed_speaker,
        _fabricated_timestamp,
        _unknown_field,
        _noncanonical_date,
        _missing_field,
    ],
)
def test_semantic_invalid_wrapped_response_remains_pr22_import_error(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    response_text = "Here:\n" + _json_with_mutation(mutate) + "\nDone."

    with pytest.raises(StructuredAIResponseImportError) as caught:
        import_repaired_structured_ai_meeting_response(_request(), response_text)

    assert not isinstance(caught.value, StructuredAIResponseRepairError)


def test_noncanonical_valid_json_document_is_not_repaired() -> None:
    expected = _summary(
        action_items=(
            ActionItem(
                task="Ship",
                owner="Alice",
                due_date=datetime.date(2026, 9, 4),
                source_start_ns=300,
                source_end_ns=400,
            ),
        )
    )
    data = meeting_summary_to_dict(expected)
    action_items = data["action_items"]
    assert isinstance(action_items, list)
    action_item = action_items[0]
    assert isinstance(action_item, dict)
    action_item["due_date"] = "09/04/2026"
    response_text = json.dumps(data)

    with pytest.raises(StructuredAIResponseImportError):
        import_repaired_structured_ai_meeting_response(_request(), response_text)


def test_wrapped_pr22_failure_is_propagated_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import buzz.meeting.portable_ai_response_repair as module

    original_error = StructuredAIResponseImportError("strict original")
    semantic_error = StructuredAIResponseImportError("semantic repaired")
    response_text = 'prefix\n{"valid":"object"}\nsuffix'
    calls = 0

    def strict_import(_: object, __: object) -> MeetingSummary:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise original_error
        raise semantic_error

    monkeypatch.setattr(module, "import_structured_ai_meeting_response", strict_import)

    with pytest.raises(StructuredAIResponseImportError) as caught:
        module.import_repaired_structured_ai_meeting_response(_request(), response_text)

    assert caught.value is semantic_error
    assert calls == 2


def test_repair_error_is_private_and_has_no_decode_cause_chain() -> None:
    sentinel = "SUPER_SECRET_REPAIR_RESPONSE_SENTINEL"

    with pytest.raises(StructuredAIResponseRepairError) as caught:
        import_repaired_structured_ai_meeting_response(
            _request(), "```json\n" + sentinel + " {\n```"
        )

    assert str(caught.value) == _REPAIR_ERROR_MESSAGE
    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_same_invalid_input_is_deterministic() -> None:
    errors: list[tuple[type[BaseException], str]] = []
    for _ in range(2):
        with pytest.raises(StructuredAIResponseRepairError) as caught:
            import_repaired_structured_ai_meeting_response(
                _request(), "prefix {'invalid': True} suffix"
            )
        errors.append((type(caught.value), str(caught.value)))

    assert errors[0] == errors[1]


def test_public_api_is_exact() -> None:
    import buzz.meeting.portable_ai_response_repair as module

    assert module.__all__ == [
        "StructuredAIResponseRepairError",
        "import_repaired_structured_ai_meeting_response",
    ]
    assert not hasattr(module, "repair_structured_ai_meeting_response")


def test_unique_object_extractor_uses_raw_decode_not_brace_search() -> None:
    import buzz.meeting.portable_ai_response_repair as module

    tree = ast.parse(inspect.getsource(module))
    extractors = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_extract_unique_json_object"
    ]

    assert len(extractors) == 1
    calls = [node for node in ast.walk(extractors[0]) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "raw_decode"
        for call in calls
    )

    forbidden_methods = {"find", "rfind", "index", "rindex"}
    forbidden_braces = {"{", "}"}
    forbidden_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr in forbidden_methods
        and any(
            isinstance(argument, ast.Constant) and argument.value in forbidden_braces
            for argument in (
                *call.args,
                *(keyword.value for keyword in call.keywords),
            )
        )
    ]
    assert forbidden_calls == []


def test_import_isolation_and_no_forbidden_repairs_or_side_effects() -> None:
    import buzz.meeting.portable_ai_response_repair as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
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

    assert imports == {
        "__future__",
        "json",
        "buzz.meeting.meeting_summary",
        "buzz.meeting.portable_ai_response",
        "buzz.meeting.summary_provider",
    }
    assert {
        "meeting_summary_from_json",
        "meeting_summary_from_dict",
        "validate_summary_provider_result",
        "validate_meeting_summary_timestamp_provenance",
        "OpenAICompatibleProvider",
        "MeetingSummaryArtifact",
        "MeetingSummaryRepository",
        "open",
        "Path",
        "QClipboard",
        "QApplication",
        "QSqlDatabase",
        "Settings",
        "print",
    }.isdisjoint(names)
    assert "json.loads" not in source
    assert "json.dumps" not in source
    assert "logging" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "socket" not in source
    assert "__cause__" not in source
