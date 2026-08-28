"""Portable plain-text packaging for a meeting-summary request."""

from __future__ import annotations

from buzz.meeting.meeting_summary_prompt import (
    MEETING_SUMMARY_PROMPT_INSTRUCTIONS,
    render_meeting_summary_request_json,
)
from buzz.meeting.summary_provider import MeetingSummaryRequest


def render_portable_ai_meeting_request(request: MeetingSummaryRequest) -> str:
    """Render a meeting-summary request for use with a generic AI chat."""
    return (
        "MEETING SUMMARY REQUEST\n\n"
        "INSTRUCTIONS\n"
        f"{MEETING_SUMMARY_PROMPT_INSTRUCTIONS}\n\n"
        "INPUT\n"
        f"{render_meeting_summary_request_json(request)}"
    )


__all__ = ["render_portable_ai_meeting_request"]
