"""Localized presentation helpers shared by meeting windows."""

from __future__ import annotations

from datetime import datetime

from buzz.locale import _
from buzz.meeting.meeting_audio_tracks import (
    MeetingAudioTracksOutcome,
    MeetingAudioTracksState,
)
from buzz.meeting.meeting_session import MeetingRemoteSourceKind, MeetingSessionState


def format_meeting_datetime(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    whole_seconds = int(seconds)
    if whole_seconds < 60:
        return f"{whole_seconds}s"
    if whole_seconds < 3600:
        minutes, remaining_seconds = divmod(whole_seconds, 60)
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_seconds = divmod(whole_seconds, 3600)
    minutes = remaining_seconds // 60
    return f"{hours}h {minutes:02d}m"


def format_remote_source(value: MeetingRemoteSourceKind) -> str:
    return {
        MeetingRemoteSourceKind.SYSTEM: _("System audio"),
        MeetingRemoteSourceKind.APPLICATION: _("Application audio"),
    }[value]


def format_meeting_state(value: MeetingSessionState) -> str:
    return {
        MeetingSessionState.CREATED: _("Created"),
        MeetingSessionState.STARTING: _("Starting"),
        MeetingSessionState.ACTIVE: _("Active"),
        MeetingSessionState.STOPPING: _("Stopping"),
        MeetingSessionState.COMPLETED: _("Completed"),
        MeetingSessionState.FAILED: _("Failed"),
    }[value]


def format_audio_state(value: MeetingAudioTracksState) -> str:
    return {
        MeetingAudioTracksState.CREATED: _("Created"),
        MeetingAudioTracksState.STARTING: _("Starting"),
        MeetingAudioTracksState.RUNNING: _("Running"),
        MeetingAudioTracksState.DEGRADED: _("Degraded"),
        MeetingAudioTracksState.STOPPING: _("Stopping"),
        MeetingAudioTracksState.STOPPED: _("Stopped"),
        MeetingAudioTracksState.FAILED: _("Failed"),
    }[value]


def format_audio_outcome(value: MeetingAudioTracksOutcome) -> str:
    return {
        MeetingAudioTracksOutcome.COMPLETE: _("Complete"),
        MeetingAudioTracksOutcome.PARTIAL: _("Partial"),
        MeetingAudioTracksOutcome.FAILED: _("Failed"),
    }[value]


def format_audio_status(
    state: MeetingAudioTracksState,
    outcome: MeetingAudioTracksOutcome | None,
) -> str:
    return (
        format_audio_state(state) if outcome is None else format_audio_outcome(outcome)
    )


__all__ = [
    "format_audio_outcome",
    "format_audio_state",
    "format_audio_status",
    "format_duration",
    "format_meeting_datetime",
    "format_meeting_state",
    "format_remote_source",
]
