"""Tests for the audio-coordinator timeline mapper."""

from __future__ import annotations

import pytest

from buzz.meeting.final_transcription import (
    TimelineMappingError,
    map_track_time_to_meeting_ns,
)
from buzz.meeting.meeting_storage import StoredMeetingTimingAnchor


def _anchor(sample_end: int, offset_ns: int) -> StoredMeetingTimingAnchor:
    return StoredMeetingTimingAnchor(
        sample_end=sample_end,
        callback_arrival_offset_ns=offset_ns,
    )


class TestZeroAnchors:
    def test_rejects_zero_anchors(self) -> None:
        with pytest.raises(TimelineMappingError, match="Zero timing anchors"):
            map_track_time_to_meeting_ns(0, 16000, ())

    def test_rejects_with_nonzero_local_ms(self) -> None:
        with pytest.raises(TimelineMappingError, match="Zero timing anchors"):
            map_track_time_to_meeting_ns(5000, 16000, ())


class TestOneAnchor:
    def test_constant_offset(self) -> None:
        # Anchor: sample 16000 at offset 100_000_000 ns
        # sample_rate 16000 → anchor_local_ns = 16000 * 1e9 // 16000 = 1_000_000_000
        # offset = 100_000_000 - 1_000_000_000 = -900_000_000
        anchor = _anchor(16000, 100_000_000)
        # local_ms=0 → local_ns=0 → 0 + (-900_000_000) = -900_000_000
        result = map_track_time_to_meeting_ns(0, 16000, (anchor,))
        assert result == -900_000_000

    def test_constant_offset_positive(self) -> None:
        # Anchor: sample 16000 at offset 2_000_000_000 ns
        # anchor_local_ns = 1_000_000_000
        # offset = 2_000_000_000 - 1_000_000_000 = 1_000_000_000
        anchor = _anchor(16000, 2_000_000_000)
        # local_ms=500 → local_ns=500_000_000 → 500_000_000 + 1_000_000_000
        result = map_track_time_to_meeting_ns(500, 16000, (anchor,))
        assert result == 1_500_000_000


class TestTwoAnchors:
    def test_interpolation(self) -> None:
        # Anchor 0: sample 16000, offset 0 ns
        #   local_ns = 16000 * 1e9 // 16000 = 1_000_000_000
        # Anchor 1: sample 32000, offset 2_100_000_000 ns
        #   local_ns = 32000 * 1e9 // 16000 = 2_000_000_000
        # Slope: (2_100_000_000 - 0) / (2_000_000_000 - 1_000_000_000)
        #       = 2.1 (but integer: 2_100_000_000 // 1_000_000_000 = 2)
        anchors = (
            _anchor(16000, 0),
            _anchor(32000, 2_100_000_000),
        )
        # local_ms=1500 → local_ns = 1_500_000_000
        # Between anchors: y = 0 + (1_500_000_000 - 1_000_000_000) * 2_100_000_000 // 1_000_000_000
        # = 500_000_000 * 2_100_000_000 // 1_000_000_000
        # = 1_050_000_000
        result = map_track_time_to_meeting_ns(1500, 16000, anchors)
        assert result == 1_050_000_000

    def test_at_anchor_exact(self) -> None:
        anchors = (
            _anchor(16000, 500_000_000),
            _anchor(32000, 2_500_000_000),
        )
        # At anchor 0: local_ms=1000 → local_ns=1_000_000_000
        # anchor_local_ns = 1_000_000_000
        # x == x0, so y = y0 = 500_000_000
        result = map_track_time_to_meeting_ns(1000, 16000, anchors)
        assert result == 500_000_000


class TestExtrapolation:
    def test_before_first_anchor(self) -> None:
        anchors = (
            _anchor(16000, 1_000_000_000),  # local_ns = 1_000_000_000
            _anchor(32000, 3_000_000_000),  # local_ns = 2_000_000_000
        )
        # Slope: (3e9 - 1e9) / (2e9 - 1e9) = 2
        # local_ms=0 → local_ns=0
        # y = 1e9 + (0 - 1e9) * 2 = 1e9 - 2e9 = -1_000_000_000
        result = map_track_time_to_meeting_ns(0, 16000, anchors)
        assert result == -1_000_000_000

    def test_after_last_anchor(self) -> None:
        anchors = (
            _anchor(16000, 0),  # local_ns = 1_000_000_000
            _anchor(32000, 2_000_000_000),  # local_ns = 2_000_000_000
        )
        # Slope: 2e9 / 1e9 = 2
        # local_ms=3000 → local_ns=3_000_000_000
        # y = 0 + (3e9 - 1e9) * 2 = 4_000_000_000
        result = map_track_time_to_meeting_ns(3000, 16000, anchors)
        assert result == 4_000_000_000


class TestNegativeOffset:
    def test_negative_preserved(self) -> None:
        # Single anchor with negative offset
        anchor = _anchor(16000, -500_000_000)
        # anchor_local_ns = 1_000_000_000
        # offset = -500_000_000 - 1_000_000_000 = -1_500_000_000
        # local_ms=0 → 0 + (-1_500_000_000)
        result = map_track_time_to_meeting_ns(0, 16000, (anchor,))
        assert result == -1_500_000_000

    def test_negative_in_interpolation(self) -> None:
        anchors = (
            _anchor(16000, -1_000_000_000),
            _anchor(32000, 0),
        )
        # local_ms=1000 → at anchor 0 → result = -1e9
        result = map_track_time_to_meeting_ns(1000, 16000, anchors)
        assert result == -1_000_000_000


class TestNonmonotonic:
    def test_nonmonotonic_callback_rejected(self) -> None:
        anchors = (
            _anchor(16000, 2_000_000_000),
            _anchor(32000, 1_000_000_000),  # backward in meeting time
        )
        with pytest.raises(TimelineMappingError, match="Non-monotonic"):
            map_track_time_to_meeting_ns(1500, 16000, anchors)

    def test_equal_callback_rejected(self) -> None:
        anchors = (
            _anchor(16000, 1_000_000_000),
            _anchor(32000, 1_000_000_000),  # equal
        )
        with pytest.raises(TimelineMappingError, match="Non-monotonic"):
            map_track_time_to_meeting_ns(1500, 16000, anchors)


class TestValidation:
    def test_invalid_sample_rate(self) -> None:
        anchor = _anchor(16000, 0)
        with pytest.raises(TimelineMappingError, match="sample_rate"):
            map_track_time_to_meeting_ns(0, 0, (anchor,))

    def test_negative_sample_rate(self) -> None:
        anchor = _anchor(16000, 0)
        with pytest.raises(TimelineMappingError, match="sample_rate"):
            map_track_time_to_meeting_ns(0, -1, (anchor,))

    def test_non_int_local_ms(self) -> None:
        anchor = _anchor(16000, 0)
        with pytest.raises(TimelineMappingError, match="local_ms"):
            map_track_time_to_meeting_ns(1.5, 16000, (anchor,))  # type: ignore

    def test_bool_local_ms(self) -> None:
        anchor = _anchor(16000, 0)
        with pytest.raises(TimelineMappingError, match="local_ms"):
            map_track_time_to_meeting_ns(True, 16000, (anchor,))  # type: ignore

    def test_negative_local_ms(self) -> None:
        anchor = _anchor(16000, 0)
        with pytest.raises(TimelineMappingError, match="local_ms"):
            map_track_time_to_meeting_ns(-1, 16000, (anchor,))


class TestPrecision:
    def test_long_meeting_integer_precision(self) -> None:
        """Two-hour meeting: verify integer interpolation precision."""
        sample_rate = 48000
        # Anchors at 1 hour and 2 hours
        # 1 hour = 3600s * 48000 = 172_800_000 samples
        # 2 hours = 7200s * 48000 = 345_600_000 samples
        # Offsets: 3_600_000_000_000 and 7_200_000_000_000 ns
        anchors = (
            _anchor(172_800_000, 3_600_000_000_000),
            _anchor(345_600_000, 7_200_000_000_000),
        )
        # Map 1.5 hours: local_ms = 5_400_000
        # local_ns = 5_400_000_000_000
        # Interpolation: y = 3_600_000_000_000 +
        #   (5_400_000_000_000 - 3_600_000_000_000) *
        #   (7_200_000_000_000 - 3_600_000_000_000) //
        #   (7_200_000_000_000 - 3_600_000_000_000)
        # = 3_600_000_000_000 + 1_800_000_000_000 = 5_400_000_000_000
        result = map_track_time_to_meeting_ns(5_400_000, sample_rate, anchors)
        assert result == 5_400_000_000_000

    def test_long_meeting_interpolation_no_overflow(self) -> None:
        """Large values must not overflow Python int."""
        sample_rate = 48000
        # Anchors at exact sample-rate boundaries to avoid truncation
        # 1 hour = 3600 * 48000 = 172_800_000 samples → local_ns = 3_600_000_000_000
        # 2 hours = 7200 * 48000 = 345_600_000 samples → local_ns = 7_200_000_000_000
        anchors = (
            _anchor(172_800_000, 3_600_000_000_000),
            _anchor(345_600_000, 7_200_000_000_000),
        )
        # Extrapolate 0.5h before first anchor
        # local_ms = 1_800_000 → local_ns = 1_800_000_000_000
        # y = 3_600_000_000_000 + (1_800_000_000_000 - 3_600_000_000_000)
        #     * (7_200_000_000_000 - 3_600_000_000_000)
        #     // (7_200_000_000_000 - 3_600_000_000_000)
        #   = 3_600_000_000_000 + (-1_800_000_000_000) * 3_600_000_000_000
        #     // 3_600_000_000_000
        #   = 3_600_000_000_000 - 1_800_000_000_000 = 1_800_000_000_000
        result = map_track_time_to_meeting_ns(1_800_000, sample_rate, anchors)
        assert isinstance(result, int)
        assert result == 1_800_000_000_000


class TestMultipleAnchors:
    def test_three_anchors_between(self) -> None:
        anchors = (
            _anchor(16000, 0),
            _anchor(32000, 2_000_000_000),
            _anchor(48000, 4_000_000_000),
        )
        # Between 2nd and 3rd: local_ms=2500 → local_ns=2_500_000_000
        # x0=2e9, x1=3e9, y0=2e9, y1=4e9
        # y = 2e9 + (2.5e9 - 2e9) * (4e9 - 2e9) // (3e9 - 2e9)
        # = 2e9 + 0.5e9 * 2e9 // 1e9 = 2e9 + 1e9 = 3e9
        result = map_track_time_to_meeting_ns(2500, 16000, anchors)
        assert result == 3_000_000_000
