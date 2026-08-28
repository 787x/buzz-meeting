"""Provider-independent MeetingSummary timestamp provenance validation."""

from __future__ import annotations

from buzz.meeting.meeting_summary import MeetingSummary as _MeetingSummary
from buzz.meeting.summary_provider import (
    MeetingSummaryRequest as _MeetingSummaryRequest,
)


class MeetingSummaryTimestampProvenanceError(ValueError):
    """A MeetingSummary timestamp is not grounded in its source transcript."""


def validate_meeting_summary_timestamp_provenance(
    request: _MeetingSummaryRequest,
    summary: _MeetingSummary,
) -> None:
    """Require summary timestamps to use supplied transcript boundaries."""
    allowed_starts = {entry.source_start_ns for entry in request.transcript}
    allowed_ends = {entry.source_end_ns for entry in request.transcript}
    timestamped_items = (
        *summary.topics,
        *summary.decisions,
        *summary.action_items,
        *summary.open_questions,
        *summary.risks,
    )

    for item in timestamped_items:
        if (
            item.source_start_ns is not None
            and item.source_start_ns not in allowed_starts
        ):
            raise MeetingSummaryTimestampProvenanceError(
                "MeetingSummary timestamp is outside the supplied transcript "
                "boundaries"
            )
        if item.source_end_ns is not None and item.source_end_ns not in allowed_ends:
            raise MeetingSummaryTimestampProvenanceError(
                "MeetingSummary timestamp is outside the supplied transcript "
                "boundaries"
            )


__all__ = [
    "MeetingSummaryTimestampProvenanceError",
    "validate_meeting_summary_timestamp_provenance",
]
