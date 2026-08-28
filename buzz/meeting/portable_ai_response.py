"""Strict local import of a structured AI meeting response."""

from __future__ import annotations

from buzz.meeting.meeting_summary import (
    MeetingSummary,
    MeetingSummaryError,
    meeting_summary_from_json,
)
from buzz.meeting.meeting_summary_provenance import (
    MeetingSummaryTimestampProvenanceError,
    validate_meeting_summary_timestamp_provenance,
)
from buzz.meeting.summary_provider import (
    MeetingSummaryRequest,
    SummaryProviderResponseError,
    validate_summary_provider_result,
)

_IMPORT_ERROR_MESSAGE = "Structured AI meeting response failed strict validation"


class StructuredAIResponseImportError(ValueError):
    """Strict structured AI meeting-response import failed."""


def import_structured_ai_meeting_response(
    request: MeetingSummaryRequest,
    response_text: str,
) -> MeetingSummary:
    """Decode and strictly validate a structured AI meeting response."""
    try:
        summary = meeting_summary_from_json(response_text)
        summary = validate_summary_provider_result(request, summary)
        validate_meeting_summary_timestamp_provenance(request, summary)
    except (
        MeetingSummaryError,
        SummaryProviderResponseError,
        MeetingSummaryTimestampProvenanceError,
    ) as exc:
        raise StructuredAIResponseImportError(_IMPORT_ERROR_MESSAGE) from exc
    return summary


__all__ = [
    "StructuredAIResponseImportError",
    "import_structured_ai_meeting_response",
]
