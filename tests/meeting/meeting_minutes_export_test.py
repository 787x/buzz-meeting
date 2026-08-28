from __future__ import annotations

import ast
from dataclasses import replace
import datetime
import inspect
from pathlib import Path
import uuid
import zipfile
from xml.etree import ElementTree

import pytest

from buzz.meeting import meeting_minutes_export
from buzz.meeting.meeting_minutes_export import (
    MeetingMinutesMetadata,
    render_meeting_minutes_markdown,
    render_meeting_minutes_text,
    write_meeting_minutes_docx,
    write_meeting_minutes_markdown,
    write_meeting_minutes_text,
)
from buzz.meeting.meeting_summary import (
    ActionItem,
    Decision,
    MEETING_SUMMARY_SCHEMA_VERSION,
    MeetingSummary,
    OpenQuestion,
    Participant,
    Risk,
    Topic,
)


_REVIEWED_SPEAKER_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
_SOURCE_START = -987654321012345
_SOURCE_END = 876543210123456
_MEETING_AT = datetime.datetime(
    2026,
    9,
    4,
    9,
    30,
    tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=45)),
)
_METADATA = MeetingMinutesMetadata(
    meeting_at=_MEETING_AT,
    duration_ns=(2 * 3600 + 5 * 60 + 59) * 1_000_000_000,
)
_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NAMESPACES = {"w": _WORD_NAMESPACE}


def _minimal_summary(**changes: object) -> MeetingSummary:
    values = {
        "schema_version": MEETING_SUMMARY_SCHEMA_VERSION,
        "prompt_version": 987654,
        "title": None,
        "summary": "Required summary.",
        "participants": (),
        "topics": (),
        "decisions": (),
        "action_items": (),
        "open_questions": (),
        "risks": (),
    }
    values.update(changes)
    return MeetingSummary(**values)  # type: ignore[arg-type]


def _full_summary(**changes: object) -> MeetingSummary:
    values = {
        "schema_version": MEETING_SUMMARY_SCHEMA_VERSION,
        "prompt_version": 987654,
        "title": "Quarterly **sync**\r\n# launch 中文",
        "summary": (
            "café 😀 مرحبا\r\n"
            "# fake heading\r"
            "- fake bullet\n"
            "+ fake plus\n"
            "> fake quote\n"
            "~~~json\n"
            "```json\n"
            "1. fake ordered\n"
            "2) fake ordered\n"
            "**fake bold** _fake italic_\n"
            "[link](https://example.invalid) <tag> &amp; \\backslash"
        ),
        "participants": (
            Participant(name="Zoë café 😀 مرحبا", reviewed_speaker_id=None),
            Participant(name=None, reviewed_speaker_id=_REVIEWED_SPEAKER_ID),
        ),
        "topics": (
            Topic(
                title="Zeta <topic>",
                summary="Detail\r\n<br> & value",
                source_start_ns=_SOURCE_START,
                source_end_ns=_SOURCE_END,
            ),
            Topic(
                title="Alpha _topic_",
                summary=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
        ),
        "decisions": (
            Decision(
                text="Ship [link](https://example.invalid)",
                source_start_ns=_SOURCE_START,
                source_end_ns=_SOURCE_END,
            ),
            Decision(
                text="Ship [link](https://example.invalid)",
                source_start_ns=None,
                source_end_ns=None,
            ),
        ),
        "action_items": (
            ActionItem(
                task="Send report\nACTION ITEMS",
                owner="Alice *lead*",
                due_date=datetime.date(2026, 9, 4),
                source_start_ns=_SOURCE_START,
                source_end_ns=_SOURCE_END,
            ),
            ActionItem(
                task="- follow up",
                owner=None,
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
        ),
        "open_questions": (
            OpenQuestion(
                text="> unanswered?",
                source_start_ns=_SOURCE_START,
                source_end_ns=_SOURCE_END,
            ),
        ),
        "risks": (
            Risk(
                text="</w:t><w:p><w:r><w:t>attack\nRISKS\n- fake",
                source_start_ns=_SOURCE_START,
                source_end_ns=_SOURCE_END,
            ),
        ),
    }
    values.update(changes)
    return MeetingSummary(**values)  # type: ignore[arg-type]


def _document_root(path: Path) -> ElementTree.Element:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    return ElementTree.fromstring(document_xml)


def _paragraphs(path: Path) -> list[ElementTree.Element]:
    return _document_root(path).findall(".//w:body/w:p", _NAMESPACES)


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", _NAMESPACES))


def _run_details(paragraph: ElementTree.Element) -> list[tuple[str, bool]]:
    details = []
    for run in paragraph.findall("w:r", _NAMESPACES):
        text = run.find("w:t", _NAMESPACES)
        bold = run.find("w:rPr/w:b", _NAMESPACES) is not None
        details.append((text.text or "", bold))
    return details


def _docx_text(path: Path) -> str:
    return "\n".join(_paragraph_text(paragraph) for paragraph in _paragraphs(path))


def _assert_occurs_in_order(output: str, expected: tuple[str, ...]) -> None:
    cursor = 0
    for value in expected:
        position = output.find(value, cursor)
        assert position >= 0, f"{value!r} missing after position {cursor}"
        cursor = position + len(value)


def test_public_api_is_exact() -> None:
    assert meeting_minutes_export.__all__ == [
        "MeetingMinutesMetadata",
        "render_meeting_minutes_markdown",
        "render_meeting_minutes_text",
        "write_meeting_minutes_markdown",
        "write_meeting_minutes_text",
        "write_meeting_minutes_docx",
    ]
    expected_parameters = {
        "render_meeting_minutes_markdown": ("summary", "metadata"),
        "render_meeting_minutes_text": ("summary", "metadata"),
        "write_meeting_minutes_markdown": ("out_path", "summary", "metadata"),
        "write_meeting_minutes_text": ("out_path", "summary", "metadata"),
        "write_meeting_minutes_docx": ("out_path", "summary", "metadata"),
    }
    for name, expected in expected_parameters.items():
        function = getattr(meeting_minutes_export, name)
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == expected
        assert signature.parameters["summary"].annotation == "MeetingSummary"
        assert signature.parameters["metadata"].annotation == "MeetingMinutesMetadata"
        if "out_path" in signature.parameters:
            assert signature.parameters["out_path"].annotation == "str | Path"
        assert not {
            "meeting_id",
            "summary_id",
            "artifact",
            "repository",
        }.intersection(signature.parameters)


def test_metadata_is_frozen_and_slotted() -> None:
    metadata = MeetingMinutesMetadata(_MEETING_AT, None)
    assert type(metadata).__slots__ == ("meeting_at", "duration_ns")
    with pytest.raises(AttributeError):
        metadata.duration_ns = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "meeting_at",
    [
        datetime.datetime(2026, 9, 4, tzinfo=datetime.timezone.utc),
        datetime.datetime(
            2026,
            9,
            4,
            tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
        ),
    ],
)
def test_metadata_accepts_aware_datetime(meeting_at: datetime.datetime) -> None:
    metadata = MeetingMinutesMetadata(meeting_at)
    assert metadata.meeting_at is meeting_at


@pytest.mark.parametrize("meeting_at", ["2026-09-04", None, 1])
def test_metadata_rejects_non_datetime(meeting_at: object) -> None:
    with pytest.raises(TypeError, match="meeting_at must be datetime"):
        MeetingMinutesMetadata(meeting_at)  # type: ignore[arg-type]


def test_metadata_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="meeting_at must be timezone-aware"):
        MeetingMinutesMetadata(datetime.datetime(2026, 9, 4))


class _NaiveTzInfo(datetime.tzinfo):
    def utcoffset(self, dt: datetime.datetime | None) -> None:
        return None


def test_metadata_rejects_datetime_with_none_utcoffset() -> None:
    with pytest.raises(ValueError, match="meeting_at must be timezone-aware"):
        MeetingMinutesMetadata(datetime.datetime(2026, 9, 4, tzinfo=_NaiveTzInfo()))


@pytest.mark.parametrize("duration_ns", [None, 0, 1, 5_000_000_000])
def test_metadata_accepts_valid_duration(duration_ns: int | None) -> None:
    assert MeetingMinutesMetadata(_MEETING_AT, duration_ns).duration_ns == duration_ns


@pytest.mark.parametrize("duration_ns", [True, False, 1.5, "1", object()])
def test_metadata_rejects_invalid_duration_type(duration_ns: object) -> None:
    with pytest.raises(TypeError, match="duration_ns must be int or None"):
        MeetingMinutesMetadata(_MEETING_AT, duration_ns)  # type: ignore[arg-type]


def test_metadata_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_ns must be >= 0"):
        MeetingMinutesMetadata(_MEETING_AT, -1)


@pytest.mark.parametrize(
    ("duration_ns", "formatted"),
    [
        (0, "0s"),
        (59_000_000_000, "59s"),
        (60_000_000_000, "1m 00s"),
        (67_000_000_000, "1m 07s"),
        (3_599_000_000_000, "59m 59s"),
        (3_600_000_000_000, "1h 00m"),
        ((2 * 3600 + 5 * 60 + 59) * 1_000_000_000, "2h 05m"),
        (999_999_999, "0s"),
        (67_999_999_999, "1m 07s"),
    ],
)
def test_duration_format_boundaries(duration_ns: int, formatted: str) -> None:
    metadata = MeetingMinutesMetadata(_MEETING_AT, duration_ns)
    markdown = render_meeting_minutes_markdown(_minimal_summary(), metadata)
    assert f"**Duration:** {formatted}" in markdown


def test_timezone_is_preserved_without_machine_conversion() -> None:
    markdown = render_meeting_minutes_markdown(_minimal_summary(), _METADATA)
    text = render_meeting_minutes_text(_minimal_summary(), _METADATA)
    expected = "2026-09-04 09:30:00+05:45"
    assert expected in markdown
    assert expected in text
    assert "+00:00" not in markdown


def test_minimal_markdown_snapshot_omits_duration_and_empty_sections() -> None:
    metadata = MeetingMinutesMetadata(_MEETING_AT)
    assert render_meeting_minutes_markdown(_minimal_summary(), metadata) == (
        "# Meeting Minutes\n\n"
        "**Date / Start:** 2026-09-04 09:30:00+05:45\n\n"
        "## Summary\n\n"
        "Required summary.\n"
    )


def test_minimal_text_snapshot_omits_duration_and_empty_sections() -> None:
    metadata = MeetingMinutesMetadata(_MEETING_AT)
    assert render_meeting_minutes_text(_minimal_summary(), metadata) == (
        "TITLE\n"
        "  Meeting Minutes\n\n"
        "Date / Start: 2026-09-04 09:30:00+05:45\n\n"
        "SUMMARY\n"
        "  Required summary.\n"
    )


def test_full_markdown_exact_snapshot() -> None:
    assert render_meeting_minutes_markdown(_full_summary(), _METADATA) == (
        "# Quarterly \\*\\*sync\\*\\*<br>\\# launch 中文\n\n"
        "**Date / Start:** 2026-09-04 09:30:00+05:45\n"
        "**Duration:** 2h 05m\n\n"
        "## Summary\n\n"
        "café 😀 مرحبا<br>\\# fake heading<br>\\- fake bullet<br>"
        "\\+ fake plus<br>\\> fake quote<br>\\~~~json<br>"
        "\\`\\`\\`json<br>1\\. fake ordered<br>2\\) fake ordered<br>"
        "\\*\\*fake bold\\*\\* \\_fake italic\\_<br>"
        "\\[link\\](https://example.invalid) \\<tag\\> \\&amp; \\\\backslash\n\n"
        "## Participants\n\n"
        "- Zoë café 😀 مرحبا\n"
        "- Unnamed participant\n\n"
        "## Topics\n\n"
        "- **Zeta \\<topic\\>** — Detail<br>\\<br\\> \\& value\n"
        "- **Alpha \\_topic\\_**\n\n"
        "## Decisions\n\n"
        "- Ship \\[link\\](https://example.invalid)\n"
        "- Ship \\[link\\](https://example.invalid)\n\n"
        "## Action Items\n\n"
        "- Send report<br>ACTION ITEMS — **Owner:** Alice \\*lead\\* — "
        "**Due:** 2026-09-04\n"
        "- \\- follow up\n\n"
        "## Open Questions\n\n"
        "- \\> unanswered?\n\n"
        "## Risks\n\n"
        "- \\</w:t\\>\\<w:p\\>\\<w:r\\>\\<w:t\\>attack<br>RISKS<br>"
        "\\- fake\n"
    )


def test_full_text_exact_snapshot() -> None:
    assert render_meeting_minutes_text(_full_summary(), _METADATA) == (
        "TITLE\n"
        "  Quarterly **sync**\n"
        "  # launch 中文\n\n"
        "Date / Start: 2026-09-04 09:30:00+05:45\n"
        "Duration: 2h 05m\n\n"
        "SUMMARY\n"
        "  café 😀 مرحبا\n"
        "  # fake heading\n"
        "  - fake bullet\n"
        "  + fake plus\n"
        "  > fake quote\n"
        "  ~~~json\n"
        "  ```json\n"
        "  1. fake ordered\n"
        "  2) fake ordered\n"
        "  **fake bold** _fake italic_\n"
        "  [link](https://example.invalid) <tag> &amp; \\backslash\n\n"
        "PARTICIPANTS\n"
        "- Zoë café 😀 مرحبا\n"
        "- Unnamed participant\n\n"
        "TOPICS\n"
        "- Zeta <topic> — Detail\n"
        "  <br> & value\n"
        "- Alpha _topic_\n\n"
        "DECISIONS\n"
        "- Ship [link](https://example.invalid)\n"
        "- Ship [link](https://example.invalid)\n\n"
        "ACTION ITEMS\n"
        "- Send report\n"
        "  ACTION ITEMS — Owner: Alice *lead* — Due: 2026-09-04\n"
        "- - follow up\n\n"
        "OPEN QUESTIONS\n"
        "- > unanswered?\n\n"
        "RISKS\n"
        "- </w:t><w:p><w:r><w:t>attack\n"
        "  RISKS\n"
        "  - fake\n"
    )


def test_markdown_user_content_cannot_add_renderer_structure() -> None:
    output = render_meeting_minutes_markdown(_full_summary(), _METADATA)
    assert output.count("\n# ") == 0
    assert output.count("\n## ") == 7
    assert "<br># fake heading" not in output
    assert "<br>- fake bullet" not in output
    assert "<br>```json" not in output
    assert "[link](https://example.invalid)" not in output
    assert "<tag>" not in output
    assert "\\<br\\>" in output
    assert output.count("<br>") == 15


@pytest.mark.parametrize(
    ("user_text", "escaped_text"),
    [
        ("# fake", "\\# fake"),
        ("#\tfake", "\\#\tfake"),
        (" ## fake", " \\## fake"),
        (" ##\tfake", " \\##\tfake"),
        ("   ### fake", "   \\### fake"),
        ("   ###\tfake", "   \\###\tfake"),
        ("- fake", "\\- fake"),
        ("-\tfake", "\\-\tfake"),
        ("  + fake", "  \\+ fake"),
        ("  +\tfake", "  \\+\tfake"),
        ("1. fake", "1\\. fake"),
        ("1.\tfake", "1\\.\tfake"),
        ("  2) fake", "  2\\) fake"),
        ("  2)\tfake", "  2\\)\tfake"),
        ("   123. fake", "   123\\. fake"),
        ("   123.\tfake", "   123\\.\tfake"),
    ],
)
def test_markdown_block_markers_accept_space_or_tab_separator(
    user_text: str,
    escaped_text: str,
) -> None:
    output = render_meeting_minutes_markdown(
        _minimal_summary(summary=user_text),
        _METADATA,
    )
    assert output.endswith(f"## Summary\n\n{escaped_text}\n")
    assert [line for line in output.splitlines() if line.startswith("#")] == [
        "# Meeting Minutes",
        "## Summary",
    ]
    assert f"## Summary\n\n{user_text}\n" not in output


def test_markdown_tab_fix_does_not_escape_inline_non_structure() -> None:
    output = render_meeting_minutes_markdown(
        _minimal_summary(summary="abc-def\nvalue+value\nversion1.2"),
        _METADATA,
    )
    assert output.endswith("## Summary\n\nabc-def<br>value+value<br>version1.2\n")


def test_txt_multiline_content_is_indented_inside_owning_field() -> None:
    output = render_meeting_minutes_text(_full_summary(), _METADATA)
    assert "\n  SUMMARY\n" not in output
    assert "\n  ACTION ITEMS — Owner:" in output
    assert "\n  RISKS\n  - fake\n" in output
    assert "\n- fake\n" not in output


def test_docx_semantics_order_bold_and_injection_safety(tmp_path: Path) -> None:
    out_path = tmp_path / "minutes.bin"
    write_meeting_minutes_docx(out_path, _full_summary(), _METADATA)

    paragraphs = _paragraphs(out_path)
    texts = [_paragraph_text(paragraph) for paragraph in paragraphs]
    assert texts == [
        "Quarterly **sync**\n# launch 中文",
        "Date / Start: 2026-09-04 09:30:00+05:45",
        "Duration: 2h 05m",
        "Summary",
        _full_summary().summary.replace("\r\n", "\n").replace("\r", "\n"),
        "Participants",
        "• Zoë café 😀 مرحبا",
        "• Unnamed participant",
        "Topics",
        "• Zeta <topic> — Detail\n<br> & value",
        "• Alpha _topic_",
        "Decisions",
        "• Ship [link](https://example.invalid)",
        "• Ship [link](https://example.invalid)",
        "Action Items",
        "• Send report\nACTION ITEMS — Owner: Alice *lead* — Due: 2026-09-04",
        "• - follow up",
        "Open Questions",
        "• > unanswered?",
        "Risks",
        "• </w:t><w:p><w:r><w:t>attack\nRISKS\n- fake",
    ]
    assert _run_details(paragraphs[1]) == [
        ("Date / Start:", True),
        (" 2026-09-04 09:30:00+05:45", False),
    ]
    assert _run_details(paragraphs[2]) == [
        ("Duration:", True),
        (" 2h 05m", False),
    ]
    assert _run_details(paragraphs[9]) == [
        ("• ", False),
        ("Zeta <topic>", True),
        (" — ", False),
        ("Detail\n<br> & value", False),
    ]
    assert _run_details(paragraphs[15]) == [
        ("• ", False),
        ("Send report\nACTION ITEMS", False),
        (" — ", False),
        ("Owner:", True),
        (" Alice *lead*", False),
        (" — ", False),
        ("Due:", True),
        (" 2026-09-04", False),
    ]

    with zipfile.ZipFile(out_path) as archive:
        document_xml = archive.read("word/document.xml")
    ElementTree.fromstring(document_xml)
    assert document_xml.count(b"<w:p>") == 21
    assert b"&amp;lt;/w:t&amp;gt;" not in document_xml


@pytest.mark.parametrize(
    ("field", "markdown_heading", "text_heading"),
    [
        ("participants", "## Participants", "PARTICIPANTS"),
        ("topics", "## Topics", "TOPICS"),
        ("decisions", "## Decisions", "DECISIONS"),
        ("action_items", "## Action Items", "ACTION ITEMS"),
        ("open_questions", "## Open Questions", "OPEN QUESTIONS"),
        ("risks", "## Risks", "RISKS"),
    ],
)
def test_empty_section_matrix_across_formats(
    tmp_path: Path,
    field: str,
    markdown_heading: str,
    text_heading: str,
) -> None:
    summary = replace(_full_summary(), **{field: ()})
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    out_path = tmp_path / f"{field}.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    docx = _docx_text(out_path)

    assert markdown_heading not in markdown
    assert f"\n{text_heading}\n" not in text
    assert f"\n{text_heading.title()}\n" not in docx
    assert "## Summary" in markdown
    assert "\nSUMMARY\n" in text
    assert "\nSummary\n" in docx


def test_cross_format_sections_have_one_exact_order(tmp_path: Path) -> None:
    summary = _full_summary()
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    out_path = tmp_path / "minutes.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    docx = _docx_text(out_path)
    markdown_headings = [
        "## Summary",
        "## Participants",
        "## Topics",
        "## Decisions",
        "## Action Items",
        "## Open Questions",
        "## Risks",
    ]
    text_headings = [
        heading.removeprefix("## ").upper() for heading in markdown_headings
    ]
    docx_headings = [heading.removeprefix("## ") for heading in markdown_headings]
    assert [markdown.index(heading) for heading in markdown_headings] == sorted(
        markdown.index(heading) for heading in markdown_headings
    )
    assert [text.index(heading) for heading in text_headings] == sorted(
        text.index(heading) for heading in text_headings
    )
    assert [docx.index(heading) for heading in docx_headings] == sorted(
        docx.index(heading) for heading in docx_headings
    )


def test_all_collection_order_and_required_duplicates_across_formats(
    tmp_path: Path,
) -> None:
    duplicate_decision = Decision(
        text="Decision Alpha",
        source_start_ns=None,
        source_end_ns=None,
    )
    duplicate_action = ActionItem(
        task="Action Alpha",
        owner=None,
        due_date=None,
        source_start_ns=None,
        source_end_ns=None,
    )
    summary = _minimal_summary(
        participants=tuple(
            Participant(name=name, reviewed_speaker_id=None)
            for name in (
                "Participant Zulu",
                "Participant Alpha",
                "Participant Middle",
            )
        ),
        topics=tuple(
            Topic(
                title=title,
                summary=None,
                source_start_ns=None,
                source_end_ns=None,
            )
            for title in ("Topic Zulu", "Topic Alpha", "Topic Middle")
        ),
        decisions=(
            Decision(
                text="Decision Zulu",
                source_start_ns=None,
                source_end_ns=None,
            ),
            duplicate_decision,
            Decision(
                text="Decision Middle",
                source_start_ns=None,
                source_end_ns=None,
            ),
            duplicate_decision,
        ),
        action_items=(
            ActionItem(
                task="Action Zulu",
                owner=None,
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
            duplicate_action,
            ActionItem(
                task="Action Middle",
                owner=None,
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
            duplicate_action,
        ),
        open_questions=tuple(
            OpenQuestion(text=text, source_start_ns=None, source_end_ns=None)
            for text in ("Question Zulu", "Question Alpha", "Question Middle")
        ),
        risks=tuple(
            Risk(text=text, source_start_ns=None, source_end_ns=None)
            for text in ("Risk Zulu", "Risk Alpha", "Risk Middle")
        ),
    )
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    out_path = tmp_path / "ordered.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    docx = _docx_text(out_path)
    outputs = (markdown, text, docx)

    expected_collections = (
        ("Participant Zulu", "Participant Alpha", "Participant Middle"),
        ("Topic Zulu", "Topic Alpha", "Topic Middle"),
        (
            "Decision Zulu",
            "Decision Alpha",
            "Decision Middle",
            "Decision Alpha",
        ),
        ("Action Zulu", "Action Alpha", "Action Middle", "Action Alpha"),
        ("Question Zulu", "Question Alpha", "Question Middle"),
        ("Risk Zulu", "Risk Alpha", "Risk Middle"),
    )
    for output in outputs:
        for expected in expected_collections:
            _assert_occurs_in_order(output, expected)
        assert output.count("Decision Alpha") == 2
        assert output.count("Action Alpha") == 2


def test_missing_action_owner_and_due_are_not_fabricated(tmp_path: Path) -> None:
    summary = _minimal_summary(
        action_items=(
            ActionItem(
                task="Only task",
                owner=None,
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
        )
    )
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    out_path = tmp_path / "minutes.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    outputs = (markdown, text, _docx_text(out_path))
    for output in outputs:
        assert "Owner:" not in output
        assert "Due:" not in output
        assert "None" not in output
        assert "null" not in output
        assert "Unknown" not in output
        assert "N/A" not in output


def test_action_owner_and_due_are_independently_optional_in_all_formats(
    tmp_path: Path,
) -> None:
    summary = _minimal_summary(
        action_items=(
            ActionItem(
                task="Prepare launch",
                owner="Alice",
                due_date=None,
                source_start_ns=None,
                source_end_ns=None,
            ),
            ActionItem(
                task="Ship report",
                owner=None,
                due_date=datetime.date(2026, 9, 4),
                source_start_ns=None,
                source_end_ns=None,
            ),
        )
    )
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    assert "- Prepare launch — **Owner:** Alice\n" in markdown
    assert "- Ship report — **Due:** 2026-09-04\n" in markdown
    assert "Prepare launch — **Owner:** Alice — **Due:**" not in markdown
    assert "Ship report — **Owner:**" not in markdown
    assert "- Prepare launch — Owner: Alice\n" in text
    assert "- Ship report — Due: 2026-09-04\n" in text
    assert "Prepare launch — Owner: Alice — Due:" not in text
    assert "Ship report — Owner:" not in text

    out_path = tmp_path / "independent-options.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    action_paragraphs = {
        _paragraph_text(paragraph): paragraph
        for paragraph in _paragraphs(out_path)
        if _paragraph_text(paragraph).startswith(("• Prepare", "• Ship"))
    }
    assert _run_details(action_paragraphs["• Prepare launch — Owner: Alice"]) == [
        ("• ", False),
        ("Prepare launch", False),
        (" — ", False),
        ("Owner:", True),
        (" Alice", False),
    ]
    assert _run_details(action_paragraphs["• Ship report — Due: 2026-09-04"]) == [
        ("• ", False),
        ("Ship report", False),
        (" — ", False),
        ("Due:", True),
        (" 2026-09-04", False),
    ]


def test_provenance_versions_and_source_spans_never_leak(tmp_path: Path) -> None:
    summary = _full_summary()
    markdown = render_meeting_minutes_markdown(summary, _METADATA)
    text = render_meeting_minutes_text(summary, _METADATA)
    out_path = tmp_path / "minutes.docx"
    write_meeting_minutes_docx(out_path, summary, _METADATA)
    outputs = (markdown, text, _docx_text(out_path))
    forbidden = (
        str(_REVIEWED_SPEAKER_ID),
        str(_SOURCE_START),
        str(_SOURCE_END),
        "987654",
        "prompt_version",
        "schema_version",
        "source_start_ns",
        "source_end_ns",
    )
    for output in outputs:
        assert "Unnamed participant" in output
        assert not any(value in output for value in forbidden)


@pytest.mark.parametrize(
    ("writer", "renderer", "filename"),
    [
        (
            write_meeting_minutes_markdown,
            render_meeting_minutes_markdown,
            "minutes.data",
        ),
        (write_meeting_minutes_text, render_meeting_minutes_text, "notes.anything"),
    ],
)
def test_text_writers_use_exact_renderer_bytes_utf8_lf_and_no_bom(
    tmp_path: Path,
    writer,
    renderer,
    filename: str,
) -> None:
    out_path = tmp_path / filename
    writer(out_path, _full_summary(), _METADATA)
    raw = out_path.read_bytes()
    assert raw == renderer(_full_summary(), _METADATA).encode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert not (tmp_path / f"{filename}.md").exists()
    assert not (tmp_path / f"{filename}.txt").exists()


def test_markdown_writer_delegates_to_pure_renderer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sentinel = "MARKDOWN_RENDER_SENTINEL\n"
    calls = []

    def render(summary, metadata):
        calls.append((summary, metadata))
        return sentinel

    monkeypatch.setattr(
        meeting_minutes_export,
        "render_meeting_minutes_markdown",
        render,
    )
    summary = _minimal_summary()
    out_path = tmp_path / "delegated.markdown"
    write_meeting_minutes_markdown(out_path, summary, _METADATA)

    assert out_path.read_bytes() == sentinel.encode("utf-8")
    assert len(calls) == 1
    assert calls[0][0] is summary
    assert calls[0][1] is _METADATA


def test_text_writer_delegates_to_pure_renderer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sentinel = "TEXT_RENDER_SENTINEL\n"
    calls = []

    def render(summary, metadata):
        calls.append((summary, metadata))
        return sentinel

    monkeypatch.setattr(
        meeting_minutes_export,
        "render_meeting_minutes_text",
        render,
    )
    summary = _minimal_summary()
    out_path = tmp_path / "delegated.text"
    write_meeting_minutes_text(out_path, summary, _METADATA)

    assert out_path.read_bytes() == sentinel.encode("utf-8")
    assert len(calls) == 1
    assert calls[0][0] is summary
    assert calls[0][1] is _METADATA


@pytest.mark.parametrize(
    "writer",
    [
        write_meeting_minutes_markdown,
        write_meeting_minutes_text,
        write_meeting_minutes_docx,
    ],
)
def test_writers_propagate_missing_parent_without_creating_it(
    tmp_path: Path,
    writer,
) -> None:
    parent = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        writer(parent / "minutes.unusual", _minimal_summary(), _METADATA)
    assert not parent.exists()


def test_writers_overwrite_exact_path_and_accept_str_or_path(tmp_path: Path) -> None:
    markdown_path = tmp_path / "markdown.any"
    text_path = tmp_path / "text.any"
    docx_path = tmp_path / "docx.any"
    for path in (markdown_path, text_path, docx_path):
        path.write_text("old sentinel", encoding="utf-8")

    write_meeting_minutes_markdown(str(markdown_path), _minimal_summary(), _METADATA)
    write_meeting_minutes_text(text_path, _minimal_summary(), _METADATA)
    write_meeting_minutes_docx(str(docx_path), _minimal_summary(), _METADATA)

    assert markdown_path.read_text(encoding="utf-8").startswith("# Meeting Minutes")
    assert text_path.read_text(encoding="utf-8").startswith("TITLE")
    assert _paragraph_text(_paragraphs(docx_path)[0]) == "Meeting Minutes"
    assert not (tmp_path / "docx.any.docx").exists()
    assert not (tmp_path / "markdown.any.md").exists()
    assert not (tmp_path / "text.any.txt").exists()


def test_all_renderers_use_shared_plan_constructor(monkeypatch, tmp_path: Path) -> None:
    original = meeting_minutes_export._build_meeting_minutes_plan
    calls = []

    def recording_builder(summary, metadata):
        calls.append((summary, metadata))
        return original(summary, metadata)

    monkeypatch.setattr(
        meeting_minutes_export,
        "_build_meeting_minutes_plan",
        recording_builder,
    )
    summary = _minimal_summary()
    render_meeting_minutes_markdown(summary, _METADATA)
    render_meeting_minutes_text(summary, _METADATA)
    write_meeting_minutes_docx(tmp_path / "minutes.docx", summary, _METADATA)
    assert calls == [(summary, _METADATA)] * 3


def test_production_source_architecture_gates() -> None:
    source_path = Path(meeting_minutes_export.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    docx_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
            imports.extend(alias.name for alias in node.names)
            if node.module == "buzz.transcriber.docx_writer":
                docx_names.extend(alias.name for alias in node.names)

    forbidden = {
        "MeetingSummaryArtifact",
        "MeetingSummaryRepository",
        "QSqlMeetingSummaryRepository",
        "MeetingStorage",
        "MeetingRepository",
        "SummaryProvider",
        "PyQt6",
        "zipfile",
        "ZipFile",
        "write_plain_docx",
        "logging",
        "json",
        "requests",
        "httpx",
        "openai",
        "Settings",
    }
    assert forbidden.isdisjoint(imports)
    forbidden_import_fragments = (
        "meeting_summary_repository",
        "meeting_storage",
        "meeting_repository",
        "summary_provider",
        "portable_ai",
        "PyQt",
        "buzz.locale",
    )
    assert not any(
        fragment in imported
        for imported in imports
        for fragment in forbidden_import_fragments
    )
    assert docx_names == ["DocxRun", "DocxWriter"]
    assert not any(name.startswith("_") for name in docx_names)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "DocxWriter" in called_names
    assert "<w:" not in source
    assert "astimezone" not in source
    assert "mkdir" not in source
    assert "makedirs" not in source
    assert "meeting_summary_to_" not in source
    assert "meeting_summary_from_" not in source
    assert "list_for_meeting" not in source


def test_production_has_no_artifact_or_repository_facing_api() -> None:
    source = Path(meeting_minutes_export.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert set(public_functions) == {
        "render_meeting_minutes_markdown",
        "render_meeting_minutes_text",
        "write_meeting_minutes_markdown",
        "write_meeting_minutes_text",
        "write_meeting_minutes_docx",
    }
    forbidden_fragments = (
        "artifact",
        "repository",
        "meeting_id",
        "summary_id",
        "freshness",
    )
    for node in public_functions.values():
        parameter_names = [argument.arg for argument in node.args.args]
        assert not any(
            fragment in name
            for name in parameter_names
            for fragment in forbidden_fragments
        )
