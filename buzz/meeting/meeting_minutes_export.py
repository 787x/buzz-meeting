"""Deterministic human-readable exports for structured meeting summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import re

from buzz.meeting.meeting_summary import MeetingSummary
from buzz.transcriber.docx_writer import DocxRun, DocxWriter

__all__ = [
    "MeetingMinutesMetadata",
    "render_meeting_minutes_markdown",
    "render_meeting_minutes_text",
    "write_meeting_minutes_markdown",
    "write_meeting_minutes_text",
    "write_meeting_minutes_docx",
]


@dataclass(frozen=True, slots=True)
class MeetingMinutesMetadata:
    meeting_at: datetime
    duration_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.meeting_at, datetime):
            raise TypeError("meeting_at must be datetime")
        if self.meeting_at.tzinfo is None or self.meeting_at.utcoffset() is None:
            raise ValueError("meeting_at must be timezone-aware")
        if self.duration_ns is not None:
            if isinstance(self.duration_ns, bool) or not isinstance(
                self.duration_ns, int
            ):
                raise TypeError("duration_ns must be int or None")
            if self.duration_ns < 0:
                raise ValueError("duration_ns must be >= 0")


@dataclass(frozen=True, slots=True)
class _TopicPlan:
    title: str
    summary: str | None


@dataclass(frozen=True, slots=True)
class _ActionPlan:
    task: str
    owner: str | None
    due: str | None


class _SectionKind(Enum):
    SUMMARY = auto()
    TEXT_ITEMS = auto()
    TOPICS = auto()
    ACTIONS = auto()


@dataclass(frozen=True, slots=True)
class _MeetingMinutesSection:
    heading: str
    kind: _SectionKind
    content: str | tuple[str, ...] | tuple[_TopicPlan, ...] | tuple[_ActionPlan, ...]


@dataclass(frozen=True, slots=True)
class _MeetingMinutesPlan:
    title: str
    meeting_at: str
    duration: str | None
    sections: tuple[_MeetingMinutesSection, ...]


def _normalize_user_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _format_duration(duration_ns: int) -> str:
    seconds = duration_ns // 1_000_000_000
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_seconds = divmod(seconds, 3600)
    minutes = remaining_seconds // 60
    return f"{hours}h {minutes:02d}m"


def _build_meeting_minutes_plan(
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> _MeetingMinutesPlan:
    title = _normalize_user_text(
        summary.title if summary.title is not None else "Meeting Minutes"
    )
    sections = [
        _MeetingMinutesSection(
            heading="Summary",
            kind=_SectionKind.SUMMARY,
            content=_normalize_user_text(summary.summary),
        )
    ]

    participants = tuple(
        _normalize_user_text(participant.name)
        if participant.name is not None
        else "Unnamed participant"
        for participant in summary.participants
    )
    if participants:
        sections.append(
            _MeetingMinutesSection(
                heading="Participants",
                kind=_SectionKind.TEXT_ITEMS,
                content=participants,
            )
        )

    topics = tuple(
        _TopicPlan(
            title=_normalize_user_text(topic.title),
            summary=(
                _normalize_user_text(topic.summary)
                if topic.summary is not None
                else None
            ),
        )
        for topic in summary.topics
    )
    if topics:
        sections.append(
            _MeetingMinutesSection(
                heading="Topics",
                kind=_SectionKind.TOPICS,
                content=topics,
            )
        )

    decisions = tuple(
        _normalize_user_text(decision.text) for decision in summary.decisions
    )
    if decisions:
        sections.append(
            _MeetingMinutesSection(
                heading="Decisions",
                kind=_SectionKind.TEXT_ITEMS,
                content=decisions,
            )
        )

    actions = tuple(
        _ActionPlan(
            task=_normalize_user_text(action.task),
            owner=(
                _normalize_user_text(action.owner) if action.owner is not None else None
            ),
            due=action.due_date.isoformat() if action.due_date is not None else None,
        )
        for action in summary.action_items
    )
    if actions:
        sections.append(
            _MeetingMinutesSection(
                heading="Action Items",
                kind=_SectionKind.ACTIONS,
                content=actions,
            )
        )

    questions = tuple(
        _normalize_user_text(question.text) for question in summary.open_questions
    )
    if questions:
        sections.append(
            _MeetingMinutesSection(
                heading="Open Questions",
                kind=_SectionKind.TEXT_ITEMS,
                content=questions,
            )
        )

    risks = tuple(_normalize_user_text(risk.text) for risk in summary.risks)
    if risks:
        sections.append(
            _MeetingMinutesSection(
                heading="Risks",
                kind=_SectionKind.TEXT_ITEMS,
                content=risks,
            )
        )

    return _MeetingMinutesPlan(
        title=title,
        meeting_at=metadata.meeting_at.isoformat(sep=" ", timespec="seconds"),
        duration=(
            _format_duration(metadata.duration_ns)
            if metadata.duration_ns is not None
            else None
        ),
        sections=tuple(sections),
    )


_MARKDOWN_INLINE = re.compile(r"[`*_\[\]<>&]")
_MARKDOWN_ORDERED = re.compile(r"^([ ]{0,3}\d+)([.)])(?=[ \t])")
_MARKDOWN_BLOCK = re.compile(r"^([ ]{0,3})(#+|[\-+])(?=[ \t])")
_MARKDOWN_TILDE_FENCE = re.compile(r"^([ ]{0,3})(~~~)")


def _escape_markdown(text: str) -> str:
    escaped_lines = []
    for line in text.split("\n"):
        line = line.replace("\\", "\\\\")
        line = _MARKDOWN_INLINE.sub(lambda match: f"\\{match.group(0)}", line)
        line = _MARKDOWN_ORDERED.sub(r"\1\\\2", line)
        line = _MARKDOWN_BLOCK.sub(r"\1\\\2", line)
        line = _MARKDOWN_TILDE_FENCE.sub(r"\1\\\2", line)
        escaped_lines.append(line)
    return "<br>".join(escaped_lines)


def _markdown_topic(topic: _TopicPlan) -> str:
    result = f"**{_escape_markdown(topic.title)}**"
    if topic.summary is not None:
        result += f" — {_escape_markdown(topic.summary)}"
    return result


def _markdown_action(action: _ActionPlan) -> str:
    result = _escape_markdown(action.task)
    if action.owner is not None:
        result += f" — **Owner:** {_escape_markdown(action.owner)}"
    if action.due is not None:
        result += f" — **Due:** {action.due}"
    return result


def render_meeting_minutes_markdown(
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> str:
    plan = _build_meeting_minutes_plan(summary, metadata)
    blocks = [f"# {_escape_markdown(plan.title)}"]
    metadata_lines = [f"**Date / Start:** {plan.meeting_at}"]
    if plan.duration is not None:
        metadata_lines.append(f"**Duration:** {plan.duration}")
    blocks.append("\n".join(metadata_lines))

    for section in plan.sections:
        blocks.append(f"## {section.heading}")
        if section.kind is _SectionKind.SUMMARY:
            assert isinstance(section.content, str)
            blocks.append(_escape_markdown(section.content))
        elif section.kind is _SectionKind.TOPICS:
            blocks.append(
                "\n".join(f"- {_markdown_topic(topic)}" for topic in section.content)
            )
        elif section.kind is _SectionKind.ACTIONS:
            blocks.append(
                "\n".join(f"- {_markdown_action(action)}" for action in section.content)
            )
        else:
            blocks.append(
                "\n".join(f"- {_escape_markdown(item)}" for item in section.content)
            )
    return "\n\n".join(blocks) + "\n"


def _indent_text(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.split("\n"))


def _text_bullet(text: str) -> str:
    lines = text.split("\n")
    return "\n".join([f"- {lines[0]}", *(f"  {line}" for line in lines[1:])])


def _text_topic(topic: _TopicPlan) -> str:
    text = topic.title
    if topic.summary is not None:
        text += f" — {topic.summary}"
    return _text_bullet(text)


def _text_action(action: _ActionPlan) -> str:
    text = action.task
    if action.owner is not None:
        text += f" — Owner: {action.owner}"
    if action.due is not None:
        text += f" — Due: {action.due}"
    return _text_bullet(text)


def render_meeting_minutes_text(
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> str:
    plan = _build_meeting_minutes_plan(summary, metadata)
    blocks = [f"TITLE\n{_indent_text(plan.title)}"]
    metadata_lines = [f"Date / Start: {plan.meeting_at}"]
    if plan.duration is not None:
        metadata_lines.append(f"Duration: {plan.duration}")
    blocks.append("\n".join(metadata_lines))

    for section in plan.sections:
        heading = section.heading.upper()
        if section.kind is _SectionKind.SUMMARY:
            assert isinstance(section.content, str)
            body = _indent_text(section.content)
        elif section.kind is _SectionKind.TOPICS:
            body = "\n".join(_text_topic(topic) for topic in section.content)
        elif section.kind is _SectionKind.ACTIONS:
            body = "\n".join(_text_action(action) for action in section.content)
        else:
            body = "\n".join(_text_bullet(item) for item in section.content)
        blocks.append(f"{heading}\n{body}")
    return "\n\n".join(blocks) + "\n"


def write_meeting_minutes_markdown(
    out_path: str | Path,
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> None:
    with Path(out_path).open("w", encoding="utf-8", newline="\n") as output:
        output.write(render_meeting_minutes_markdown(summary, metadata))


def write_meeting_minutes_text(
    out_path: str | Path,
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> None:
    with Path(out_path).open("w", encoding="utf-8", newline="\n") as output:
        output.write(render_meeting_minutes_text(summary, metadata))


def _docx_topic_runs(topic: _TopicPlan) -> list[DocxRun]:
    runs = [DocxRun(topic.title, bold=True)]
    if topic.summary is not None:
        runs.extend((DocxRun(" — "), DocxRun(topic.summary)))
    return runs


def _docx_action_runs(action: _ActionPlan) -> list[DocxRun]:
    runs = [DocxRun(action.task)]
    if action.owner is not None:
        runs.extend(
            (
                DocxRun(" — "),
                DocxRun("Owner:", bold=True),
                DocxRun(f" {action.owner}"),
            )
        )
    if action.due is not None:
        runs.extend(
            (
                DocxRun(" — "),
                DocxRun("Due:", bold=True),
                DocxRun(f" {action.due}"),
            )
        )
    return runs


def write_meeting_minutes_docx(
    out_path: str | Path,
    summary: MeetingSummary,
    metadata: MeetingMinutesMetadata,
) -> None:
    plan = _build_meeting_minutes_plan(summary, metadata)
    writer = DocxWriter()
    writer.add_title(plan.title)
    writer.add_paragraph(
        [DocxRun("Date / Start:", bold=True), DocxRun(f" {plan.meeting_at}")]
    )
    if plan.duration is not None:
        writer.add_paragraph(
            [DocxRun("Duration:", bold=True), DocxRun(f" {plan.duration}")]
        )

    for section in plan.sections:
        writer.add_heading(section.heading)
        if section.kind is _SectionKind.SUMMARY:
            assert isinstance(section.content, str)
            writer.add_paragraph(section.content)
        elif section.kind is _SectionKind.TOPICS:
            for topic in section.content:
                writer.add_bullet(_docx_topic_runs(topic))
        elif section.kind is _SectionKind.ACTIONS:
            for action in section.content:
                writer.add_bullet(_docx_action_runs(action))
        else:
            for item in section.content:
                writer.add_bullet(item)
    writer.write(str(out_path))
