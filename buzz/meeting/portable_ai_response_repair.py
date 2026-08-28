"""Bounded representation repair for structured AI meeting responses."""

from __future__ import annotations

import json

from buzz.meeting.meeting_summary import MeetingSummary
from buzz.meeting.portable_ai_response import (
    StructuredAIResponseImportError,
    import_structured_ai_meeting_response,
)
from buzz.meeting.summary_provider import MeetingSummaryRequest

_REPAIR_ERROR_MESSAGE = "Structured AI meeting response could not be repaired safely"
_JSON_DOCUMENT_WHITESPACE = " \t\r\n"
_STRUCTURAL_NOISE = frozenset("{}[]\ufeff")


class StructuredAIResponseRepairError(ValueError):
    """A structured response has no single safe repaired representation."""


def import_repaired_structured_ai_meeting_response(
    request: MeetingSummaryRequest,
    response_text: str,
) -> MeetingSummary:
    """Strictly import a response, repairing only bounded presentation wrappers."""
    try:
        return import_structured_ai_meeting_response(request, response_text)
    except StructuredAIResponseImportError:
        if _decode_complete_json_document(response_text) is not None:
            raise
        repaired_text = _repair_response_representation(response_text)

    if repaired_text is None:
        raise StructuredAIResponseRepairError(_REPAIR_ERROR_MESSAGE) from None

    return import_structured_ai_meeting_response(request, repaired_text)


def _repair_response_representation(response_text: object) -> str | None:
    if not isinstance(response_text, str):
        return None

    candidate_text = response_text
    if candidate_text.startswith("\ufeff"):
        if candidate_text.startswith("\ufeff\ufeff"):
            return None
        candidate_text = candidate_text[1:]

    fence_valid, candidate_text = _unwrap_single_fence(candidate_text)
    if not fence_valid:
        return None

    return _extract_unique_json_object(candidate_text)


def _unwrap_single_fence(text: str) -> tuple[bool, str]:
    lines = _lines_with_offsets(text)
    markers = [line for line in lines if _is_fence_like(line[2])]
    if not markers:
        return True, text
    if len(markers) != 2:
        return False, text

    opening, closing = markers
    if not _is_valid_opening_fence(opening[2]) or opening[3] not in {
        "\n",
        "\r\n",
    }:
        return False, text
    if closing[2] != "```" or closing[0] <= opening[0]:
        return False, text
    if closing[3] not in {"", "\n", "\r\n"}:
        return False, text

    payload = text[opening[1] : closing[0]]
    decoded = _decode_complete_json_document(payload)
    if decoded is None or not isinstance(decoded[0], dict):
        return False, text

    unwrapped = text[: opening[0]] + payload + text[closing[1] :]
    return True, unwrapped


def _lines_with_offsets(text: str) -> list[tuple[int, int, str, str]]:
    lines: list[tuple[int, int, str, str]] = []
    start = 0
    while start < len(text):
        newline = text.find("\n", start)
        if newline == -1:
            lines.append((start, len(text), text[start:], ""))
            break

        content_end = newline
        line_ending = "\n"
        if content_end > start and text[content_end - 1] == "\r":
            content_end -= 1
            line_ending = "\r\n"
        lines.append((start, newline + 1, text[start:content_end], line_ending))
        start = newline + 1
    return lines


def _is_fence_like(line_text: str) -> bool:
    left_aligned = line_text.lstrip(" \t")
    return left_aligned.startswith("```") or left_aligned.startswith("~~~")


def _is_valid_opening_fence(line_text: str) -> bool:
    if line_text == "```":
        return True
    if len(line_text) != 7 or line_text[:3] != "```":
        return False
    return line_text[3:].lower() == "json" and line_text[3:].isascii()


def _extract_unique_json_object(text: str) -> str | None:
    complete = _decode_complete_json_document(text)
    if complete is not None:
        return text if isinstance(complete[0], dict) else None

    decoder = json.JSONDecoder(parse_constant=_reject_nonstandard_json_constant)
    ranges: set[tuple[int, int]] = set()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            ranges.add((start, end))

    maximal_ranges = [
        candidate
        for candidate in ranges
        if not any(
            outer_start <= candidate[0]
            and candidate[1] <= outer_end
            and candidate != (outer_start, outer_end)
            for outer_start, outer_end in ranges
        )
    ]
    if len(maximal_ranges) != 1:
        return None

    start, end = maximal_ranges[0]
    surroundings = text[:start] + text[end:]
    if any(character in _STRUCTURAL_NOISE for character in surroundings):
        return None

    candidate = text[start:end]
    decoded = _decode_complete_json_document(candidate)
    if decoded is None or not isinstance(decoded[0], dict):
        return None
    return candidate


def _decode_complete_json_document(
    text: object,
) -> tuple[object, int, int] | None:
    if not isinstance(text, str):
        return None

    start = _skip_json_whitespace(text, 0)
    try:
        value, end = json.JSONDecoder(
            parse_constant=_reject_nonstandard_json_constant
        ).raw_decode(text, start)
    except (json.JSONDecodeError, ValueError):
        return None
    if _skip_json_whitespace(text, end) != len(text):
        return None
    return value, start, end


def _reject_nonstandard_json_constant(_: str) -> object:
    raise ValueError


def _skip_json_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start] in _JSON_DOCUMENT_WHITESPACE:
        start += 1
    return start


__all__ = [
    "StructuredAIResponseRepairError",
    "import_repaired_structured_ai_meeting_response",
]
