"""Provider-independent meeting-summary request DTOs, provider protocol,
error taxonomy, and result-boundary validation.

Pure Python domain module.  No Qt, QSql, network, persistence, or
provider-specific imports.  Allowed dependencies:
``buzz.meeting.meeting_summary`` and standard-library
``typing`` / ``dataclasses``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from buzz.meeting.meeting_summary import (
    MEETING_SUMMARY_SCHEMA_VERSION,
    MeetingSummary,
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryProviderRequestError(
            f"{name} must be a non-bool int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise SummaryProviderRequestError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SummaryProviderRequestError(f"{name} must be str")
    if not value.strip():
        raise SummaryProviderRequestError(f"{name} must not be empty/whitespace")
    return value


def _check_optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SummaryProviderRequestError(f"{name} must be str or None")
    if not value.strip():
        raise SummaryProviderRequestError(
            f"{name} must not be empty/whitespace when present"
        )
    return value


# ---------------------------------------------------------------------------
# Transcript entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeetingSummaryTranscriptEntry:
    """One normalized phrase/utterance for provider input.

    Each entry corresponds to a canonical transcript segment.
    ``speaker_name`` is structurally validated here but semantic
    provenance (populating only from a reviewed speaker's
    ``display_name``, otherwise ``None``) is the responsibility of
    the future request assembler.
    """

    text: str
    source_start_ns: int
    source_end_ns: int
    speaker_name: str | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.text, "text")
        _require_int(self.source_start_ns, "source_start_ns")
        _require_int(self.source_end_ns, "source_end_ns")
        if self.source_end_ns < self.source_start_ns:
            raise SummaryProviderRequestError(
                f"source_end_ns ({self.source_end_ns}) must be >= "
                f"source_start_ns ({self.source_start_ns})"
            )
        _check_optional_text(self.speaker_name, "speaker_name")


# ---------------------------------------------------------------------------
# Request DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeetingSummaryRequest:
    """Provider-independent immutable summary request.

    Contains no meeting metadata, IDs, or persistence state.
    Transcript ordering is canonical; the request validator
    rejects chronological regression but never re-sorts.
    """

    schema_version: int
    prompt_version: int
    transcript: tuple[MeetingSummaryTranscriptEntry, ...]

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "schema_version")
        if self.schema_version != MEETING_SUMMARY_SCHEMA_VERSION:
            raise SummaryProviderRequestError(
                f"Unsupported schema_version: {self.schema_version}"
            )
        _require_int(self.prompt_version, "prompt_version", minimum=1)

        if not isinstance(self.transcript, tuple):
            raise SummaryProviderRequestError("transcript must be a tuple")
        if len(self.transcript) == 0:
            raise SummaryProviderRequestError("transcript must not be empty")
        for i, entry in enumerate(self.transcript):
            if not isinstance(entry, MeetingSummaryTranscriptEntry):
                raise SummaryProviderRequestError(
                    f"transcript[{i}] must be MeetingSummaryTranscriptEntry, "
                    f"got {type(entry).__name__}"
                )

        # Chronological regression check — equal start_ns is valid.
        for i in range(1, len(self.transcript)):
            curr = self.transcript[i]
            prev = self.transcript[i - 1]
            if curr.source_start_ns < prev.source_start_ns:
                raise SummaryProviderRequestError(
                    f"transcript[{i}].source_start_ns ({curr.source_start_ns}) "
                    f"< transcript[{i - 1}].source_start_ns "
                    f"({prev.source_start_ns}): chronological regression"
                )


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class SummaryProviderError(Exception):
    """Base error for summary-provider failures."""


class SummaryProviderConfigurationError(SummaryProviderError):
    """Provider-specific runtime configuration is missing or invalid."""


class SummaryProviderTransportError(SummaryProviderError):
    """No usable remote/provider response due to timeout, connection,
    TLS, DNS, or other transport failure."""


class SummaryProviderRequestError(SummaryProviderError):
    """Request reached an executable provider but was rejected or
    could not be executed."""


class SummaryProviderResponseError(SummaryProviderError):
    """Provider returned content but it cannot cross the trusted
    ``MeetingSummary`` boundary (e.g. version alignment failure
    or fabricated speaker IDs)."""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class SummaryProvider(Protocol):
    """Executable one-call summary provider.

    Synchronous/blocking.  Caller owns execution context and
    threading.  Manual AI round-trip shares the normalized
    request/result schema but is not required to implement this
    synchronous runtime protocol.
    """

    def summarize(self, request: MeetingSummaryRequest) -> MeetingSummary:
        ...


# ---------------------------------------------------------------------------
# Result-boundary validation
# ---------------------------------------------------------------------------


def validate_summary_provider_result(
    request: MeetingSummaryRequest,
    result: MeetingSummary,
) -> MeetingSummary:
    """Validate that a provider result crosses the trusted boundary.

    Checks only provider-boundary invariants: type, schema version
    alignment, prompt version alignment, and no fabricated
    ``reviewed_speaker_id`` values.  Schema-internal validation
    (title, topics, etc.) is owned by ``MeetingSummary`` itself.

    Returns the same ``MeetingSummary`` object on success.
    """
    if not isinstance(result, MeetingSummary):
        raise SummaryProviderResponseError(
            f"result must be MeetingSummary, got {type(result).__name__}"
        )
    if result.schema_version != request.schema_version:
        raise SummaryProviderResponseError(
            f"result.schema_version ({result.schema_version}) != "
            f"request.schema_version ({request.schema_version})"
        )
    if result.prompt_version != request.prompt_version:
        raise SummaryProviderResponseError(
            f"result.prompt_version ({result.prompt_version}) != "
            f"request.prompt_version ({request.prompt_version})"
        )
    for i, participant in enumerate(result.participants):
        if participant.reviewed_speaker_id is not None:
            raise SummaryProviderResponseError(
                f"participants[{i}].reviewed_speaker_id must be None "
                f"(providers cannot fabricate speaker identity)"
            )
    return result


__all__ = [
    "MeetingSummaryRequest",
    "MeetingSummaryTranscriptEntry",
    "SummaryProvider",
    "SummaryProviderConfigurationError",
    "SummaryProviderError",
    "SummaryProviderRequestError",
    "SummaryProviderResponseError",
    "SummaryProviderTransportError",
    "validate_summary_provider_result",
]
