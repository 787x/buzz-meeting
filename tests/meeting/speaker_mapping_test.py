"""Speaker mapping tests — pure, no torch/NeMo/Qt/DB/filesystem/network/sleep."""

from __future__ import annotations

import pytest

from buzz.meeting.final_transcription import MeetingTranscriptWord
from buzz.meeting.meeting_audio_tracks import MeetingTrackRole
from buzz.meeting.speaker_diarization import SpeakerDiarizationTurn
from buzz.meeting.speaker_mapping import (
    MeetingSpeakerKey,
    MeetingTrackSpeakerTurns,
    SpeakerAttributedWord,
    SpeakerAttributionStatus,
    SpeakerMappingConfigError,
    map_words_to_speakers,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

MIC = MeetingTrackRole.MICROPHONE
REMOTE = MeetingTrackRole.REMOTE


def _word(
    role: MeetingTrackRole = MIC,
    start_ms: int = 0,
    end_ms: int = 100,
    *,
    text: str = "hello",
    seg_ord: int = 0,
    word_ord: int = 0,
    start_ns: int = 0,
    end_ns: int = 0,
) -> MeetingTranscriptWord:
    return MeetingTranscriptWord(
        source_role=role,
        source_segment_ordinal=seg_ord,
        source_word_ordinal=word_ord,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        start_ns=start_ns,
        end_ns=end_ns,
        text=text,
    )


def _turn(
    speaker: int = 0,
    start_ms: int = 0,
    end_ms: int = 100,
) -> SpeakerDiarizationTurn:
    return SpeakerDiarizationTurn(
        speaker_index=speaker,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _role_turns(
    role: MeetingTrackRole = MIC,
    turns: tuple[SpeakerDiarizationTurn, ...] = (),
) -> MeetingTrackSpeakerTurns:
    return MeetingTrackSpeakerTurns(source_role=role, turns=turns)


# ══════════════════════════════════════════════════════════════════════════════
# §34. Base tests
# ══════════════════════════════════════════════════════════════════════════════


# 1. empty words → ()
def test_01_empty_words():
    result = map_words_to_speakers(
        words=(),
        diarization=[_role_turns(MIC, (_turn(0, 0, 100),))],
    )
    assert result == ()


# 2. one word / one speaker → ASSIGNED
def test_02_one_word_one_speaker():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert len(result) == 1
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 3. multiple words same speaker
def test_03_multiple_words_same_speaker():
    w1 = _word(MIC, 0, 100)
    w2 = _word(MIC, 50, 150, text="world")
    result = map_words_to_speakers(
        words=[w1, w2],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert len(result) == 2
    assert all(r.status is SpeakerAttributionStatus.ASSIGNED for r in result)
    assert all(r.speaker == MeetingSpeakerKey(MIC, 0) for r in result)


# 4. MIC and REMOTE independently mapped
def test_04_mic_remote_independent():
    mic_word = _word(MIC, 0, 100)
    remote_word = _word(REMOTE, 0, 100)
    result = map_words_to_speakers(
        words=[mic_word, remote_word],
        diarization=[
            _role_turns(MIC, (_turn(0, 0, 200),)),
            _role_turns(REMOTE, (_turn(1, 0, 200),)),
        ],
    )
    assert len(result) == 2
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)
    assert result[1].speaker == MeetingSpeakerKey(REMOTE, 1)


# 5. MIC speaker 0 != REMOTE speaker 0
def test_05_mic_remote_speaker_distinct():
    mic_word = _word(MIC, 0, 100)
    remote_word = _word(REMOTE, 0, 100)
    result = map_words_to_speakers(
        words=[mic_word, remote_word],
        diarization=[
            _role_turns(MIC, (_turn(0, 0, 200),)),
            _role_turns(REMOTE, (_turn(0, 0, 200),)),
        ],
    )
    mic_key = result[0].speaker
    remote_key = result[1].speaker
    assert mic_key is not None and remote_key is not None
    assert mic_key != remote_key
    assert mic_key.speaker_index == 0 and remote_key.speaker_index == 0
    assert mic_key.source_role is MIC and remote_key.source_role is REMOTE


# 6. missing role turn set → NO_OVERLAP
def test_06_missing_role_turns():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(REMOTE, (_turn(0, 0, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP
    assert result[0].speaker is None


# 7. empty role turn set → NO_OVERLAP
def test_07_empty_role_turns():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, ())],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP
    assert result[0].speaker is None


# 8. input word order preserved
def test_08_word_order_preserved():
    words = [
        _word(MIC, 200, 300, text="b"),
        _word(MIC, 0, 100, text="a"),
        _word(MIC, 100, 200, text="c"),
    ]
    result = map_words_to_speakers(
        words=words,
        diarization=[_role_turns(MIC, (_turn(0, 0, 400),))],
    )
    assert result[0].word.text == "b"
    assert result[1].word.text == "a"
    assert result[2].word.text == "c"


# 9. negative meeting start_ns/end_ns preserved
def test_09_negative_meeting_ns_preserved():
    w = _word(MIC, 0, 100, start_ns=-5000, end_ns=-3000)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert result[0].word.start_ns == -5000
    assert result[0].word.end_ns == -3000


# 10. word object preserved inside wrapper (identity check)
def test_10_word_identity_preserved():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert result[0].word is w


# ══════════════════════════════════════════════════════════════════════════════
# §35. Overlap tests
# ══════════════════════════════════════════════════════════════════════════════


# 11. word entirely inside turn
def test_11_word_inside_turn():
    w = _word(MIC, 50, 80)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 12. turn entirely inside word
def test_12_turn_inside_word():
    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 50, 100),))],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 13. partial left overlap
def test_13_partial_left_overlap():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 50, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED


# 14. partial right overlap
def test_14_partial_right_overlap():
    w = _word(MIC, 50, 150)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 100),))],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED


# 15. boundary touch on left → NO_OVERLAP
def test_15_boundary_touch_left():
    w = _word(MIC, 100, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 100),))],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP


# 16. boundary touch on right → NO_OVERLAP
def test_16_boundary_touch_right():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 100, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP


# 17. no overlap → NO_OVERLAP
def test_17_no_overlap():
    w = _word(MIC, 0, 50)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 100, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP


# 18. larger overlap wins
def test_18_larger_overlap_wins():
    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 60),  # 60ms overlap
                    _turn(1, 50, 200),  # 150ms overlap
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 1)


# 19. exact cross-speaker tie → AMBIGUOUS
def test_19_exact_tie():
    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 100),  # 100ms overlap
                    _turn(1, 100, 200),  # 100ms overlap
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS
    assert result[0].speaker is None


# 20. different-speaker overlap with unique maximum → ASSIGNED
def test_20_unique_max():
    w = _word(MIC, 0, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 50),  # 50ms
                    _turn(1, 0, 200),  # 200ms
                    _turn(2, 250, 300),  # 50ms
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 1)


# 21. zero-duration word → NO_OVERLAP
def test_21_zero_duration_word():
    w = _word(MIC, 100, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP
    assert result[0].speaker is None


# 22. zero-duration turn → contributes nothing
def test_22_zero_duration_turn():
    w = _word(MIC, 50, 150)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 100, 100),  # zero-duration
                    _turn(1, 0, 200),  # 200ms overlap
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 1)


# ══════════════════════════════════════════════════════════════════════════════
# §36. Union coverage tests
# ══════════════════════════════════════════════════════════════════════════════


# 23. same speaker split adjacent — speaker 0 wins
def test_23_split_adjacent():
    w = _word(MIC, 100, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 100, 180),  # speaker 0: 80ms
                    _turn(0, 180, 260),  # speaker 0: 80ms → union 160ms
                    _turn(1, 200, 270),  # speaker 1: 70ms
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 24. same-speaker duplicate interval counted once
def test_24_duplicate_interval_counted_once():
    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 200),
                    _turn(0, 0, 200),  # duplicate
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 25. same-speaker overlapping intervals counted by union, not sum
def test_25_overlapping_union_not_sum():
    # word: 100-300
    # speaker 0: 100-220, 180-280
    # union = [100, 280) = 180ms
    w = _word(MIC, 100, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 100, 220),
                    _turn(0, 180, 280),
                ),
            ),
        ],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 26. duplicate same-speaker turn added does not change assignment
def test_26_duplicate_turn_no_change():
    w = _word(MIC, 0, 200)
    without_dup = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 200),
                    _turn(1, 0, 100),
                ),
            ),
        ],
    )
    with_dup = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 200),
                    _turn(0, 0, 200),  # duplicate of speaker 0
                    _turn(1, 0, 100),
                ),
            ),
        ],
    )
    assert without_dup[0].speaker == with_dup[0].speaker
    assert without_dup[0].status == with_dup[0].status


# ══════════════════════════════════════════════════════════════════════════════
# §37. Ambiguity-status tests
# ══════════════════════════════════════════════════════════════════════════════


# 27. no overlap: speaker=None, status=NO_OVERLAP
def test_27_no_overlap_status():
    w = _word(MIC, 0, 50)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 100, 200),))],
    )
    assert result[0].speaker is None
    assert result[0].status is SpeakerAttributionStatus.NO_OVERLAP


# 28. exact tie: speaker=None, status=AMBIGUOUS
def test_28_exact_tie_status():
    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 0, 100),
                    _turn(1, 100, 200),
                ),
            ),
        ],
    )
    assert result[0].speaker is None
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS


# 29. assigned: speaker non-None, status=ASSIGNED
def test_29_assigned_status():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(0, 0, 200),))],
    )
    assert result[0].speaker is not None
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED


# 30. DTO coherence: invalid status/speaker combinations rejected
def test_30_dto_coherence_assigned_requires_speaker():
    with pytest.raises(ValueError, match="ASSIGNED status requires"):
        SpeakerAttributedWord(
            word=_word(), speaker=None, status=SpeakerAttributionStatus.ASSIGNED
        )


def test_30_dto_coherence_no_overlap_requires_none():
    with pytest.raises(ValueError, match="NO_OVERLAP status requires"):
        SpeakerAttributedWord(
            word=_word(),
            speaker=MeetingSpeakerKey(MIC, 0),
            status=SpeakerAttributionStatus.NO_OVERLAP,
        )


def test_30_dto_coherence_ambiguous_requires_none():
    with pytest.raises(ValueError, match="AMBIGUOUS status requires"):
        SpeakerAttributedWord(
            word=_word(),
            speaker=MeetingSpeakerKey(MIC, 0),
            status=SpeakerAttributionStatus.AMBIGUOUS,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §38. Structural validation tests
# ══════════════════════════════════════════════════════════════════════════════


# 31. duplicate MICROPHONE role sets → SpeakerMappingConfigError
def test_31_duplicate_role():
    with pytest.raises(SpeakerMappingConfigError, match="Duplicate role"):
        map_words_to_speakers(
            words=[_word(MIC, 0, 100)],
            diarization=[
                _role_turns(MIC, (_turn(0, 0, 200),)),
                _role_turns(MIC, (_turn(1, 0, 200),)),
            ],
        )


# 32. invalid role-set role → ConfigError
def test_32_invalid_role():
    with pytest.raises(SpeakerMappingConfigError, match="source_role must be"):
        map_words_to_speakers(
            words=[],
            diarization=[
                MeetingTrackSpeakerTurns(source_role="INVALID", turns=()),  # type: ignore[arg-type]
            ],
        )


# 33. turns field not tuple → ConfigError
def test_33_turns_not_tuple():
    with pytest.raises(SpeakerMappingConfigError, match="turns must be tuple"):
        map_words_to_speakers(
            words=[],
            diarization=[
                MeetingTrackSpeakerTurns(
                    source_role=MIC,
                    turns=[_turn(0, 0, 100)],  # type: ignore[arg-type]
                ),
            ],
        )


# 34. non-SpeakerDiarizationTurn element → ConfigError
def test_34_non_turn_element():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn must be SpeakerDiarizationTurn"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                MeetingTrackSpeakerTurns(
                    source_role=MIC,
                    turns=("not a turn",),  # type: ignore[arg-type]
                ),
            ],
        )


# 35. word not MeetingTranscriptWord → ConfigError
def test_35_word_not_transcript_word():
    with pytest.raises(
        SpeakerMappingConfigError, match="word must be MeetingTranscriptWord"
    ):
        map_words_to_speakers(
            words=["not a word"],  # type: ignore[arg-type]
            diarization=[],
        )


# 36. word bool start → ConfigError
def test_36_word_bool_start():
    bad = MeetingTranscriptWord(
        source_role=MIC,
        source_segment_ordinal=0,
        source_word_ordinal=0,
        local_start_ms=True,  # type: ignore[arg-type]
        local_end_ms=100,
        start_ns=0,
        end_ns=0,
        text="x",
    )
    with pytest.raises(
        SpeakerMappingConfigError, match="word.local_start_ms must be an int, not bool"
    ):
        map_words_to_speakers(words=[bad], diarization=[])


# 37. word bool end → ConfigError
def test_37_word_bool_end():
    bad = MeetingTranscriptWord(
        source_role=MIC,
        source_segment_ordinal=0,
        source_word_ordinal=0,
        local_start_ms=0,
        local_end_ms=True,  # type: ignore[arg-type]
        start_ns=0,
        end_ns=0,
        text="x",
    )
    with pytest.raises(
        SpeakerMappingConfigError, match="word.local_end_ms must be an int, not bool"
    ):
        map_words_to_speakers(words=[bad], diarization=[])


# 38. word negative local start → ConfigError
def test_38_word_negative_start():
    bad = MeetingTranscriptWord(
        source_role=MIC,
        source_segment_ordinal=0,
        source_word_ordinal=0,
        local_start_ms=-1,
        local_end_ms=100,
        start_ns=0,
        end_ns=0,
        text="x",
    )
    with pytest.raises(
        SpeakerMappingConfigError, match="word.local_start_ms must be >= 0"
    ):
        map_words_to_speakers(words=[bad], diarization=[])


# 39. word end < start → ConfigError
def test_39_word_end_lt_start():
    bad = MeetingTranscriptWord(
        source_role=MIC,
        source_segment_ordinal=0,
        source_word_ordinal=0,
        local_start_ms=200,
        local_end_ms=100,
        start_ns=0,
        end_ns=0,
        text="x",
    )
    with pytest.raises(
        SpeakerMappingConfigError, match="word.local_end_ms must be >= local_start_ms"
    ):
        map_words_to_speakers(words=[bad], diarization=[])


# 40. turn bool speaker → ConfigError
def test_40_turn_bool_speaker():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn.speaker_index must be an int, not bool"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(True, 0, 100),)),  # type: ignore[arg-type]
            ],
        )


# 41. turn negative speaker → ConfigError
def test_41_turn_negative_speaker():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn.speaker_index must be >= 0"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(-1, 0, 100),)),
            ],
        )


# 42. turn bool start → ConfigError
def test_42_turn_bool_start():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn.start_ms must be an int, not bool"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(0, True, 100),)),  # type: ignore[arg-type]
            ],
        )


# 43. turn negative start → ConfigError
def test_43_turn_negative_start():
    with pytest.raises(SpeakerMappingConfigError, match="turn.start_ms must be >= 0"):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(0, -1, 100),)),
            ],
        )


# 44. turn bool end → ConfigError
def test_44_turn_bool_end():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn.end_ms must be an int, not bool"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(0, 0, True),)),  # type: ignore[arg-type]
            ],
        )


# 45. turn end < start → ConfigError
def test_45_turn_end_lt_start():
    with pytest.raises(
        SpeakerMappingConfigError, match="turn.end_ms must be >= start_ms"
    ):
        map_words_to_speakers(
            words=[],
            diarization=[
                _role_turns(MIC, (SpeakerDiarizationTurn(0, 200, 100),)),
            ],
        )


# 46. speaker 7 accepted
def test_46_speaker_7_accepted():
    w = _word(MIC, 0, 100)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, (_turn(7, 0, 200),))],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 7)


# ══════════════════════════════════════════════════════════════════════════════
# §39. Unsorted / input mutation tests
# ══════════════════════════════════════════════════════════════════════════════


# 47. unsorted turns map correctly
def test_47_unsorted_turns():
    w = _word(MIC, 0, 100)
    turns = [
        _turn(0, 0, 100),
        _turn(1, 0, 100),
    ]
    # Input is sorted, but let's provide reverse order
    unsorted = (turns[1], turns[0])
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, unsorted)],
    )
    # Both speakers have 100ms overlap → tie → AMBIGUOUS
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS


# 48. equal-start turn input order unchanged
def test_48_equal_start_input_order():
    a = _turn(0, 100, 300)
    b = _turn(1, 100, 300)
    c = _turn(2, 100, 300)
    input_turns = (a, b, c)

    w = _word(MIC, 100, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, input_turns)],
    )
    # 3-way tie → AMBIGUOUS
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS


# 49. caller turn tuple unchanged
def test_49_caller_turn_tuple_unchanged():
    turns = (_turn(1, 200, 300), _turn(0, 0, 100))
    original_turns = turns
    w = _word(MIC, 0, 300)
    map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, turns)],
    )
    assert turns is original_turns
    assert turns == (_turn(1, 200, 300), _turn(0, 0, 100))


# 50. caller words sequence unchanged
def test_50_caller_words_unchanged():
    w1 = _word(MIC, 100, 200, text="b")
    w2 = _word(MIC, 0, 100, text="a")
    words = [w1, w2]
    original_texts = [w.text for w in words]
    map_words_to_speakers(
        words=words,
        diarization=[_role_turns(MIC, (_turn(0, 0, 300),))],
    )
    assert [w.text for w in words] == original_texts


# 51. no in-place sort of caller list
def test_51_no_in_place_sort():
    w1 = _word(MIC, 200, 300, text="b")
    w2 = _word(MIC, 0, 100, text="a")
    words_list = [w1, w2]
    map_words_to_speakers(
        words=words_list,
        diarization=[_role_turns(MIC, (_turn(0, 0, 400),))],
    )
    # List order must be unchanged
    assert words_list[0].text == "b"
    assert words_list[1].text == "a"


# ══════════════════════════════════════════════════════════════════════════════
# §40. Property-style deterministic tests (no Hypothesis)
# ══════════════════════════════════════════════════════════════════════════════


# 52. translation invariance: shift word + same-role turns by +K → same result
@pytest.mark.parametrize("shift", [0, 500, 5000, 100_000])
def test_52_translation_invariance(shift: int):
    base_word = _word(MIC, 100, 300)
    base_turns = (_turn(0, 0, 200), _turn(1, 150, 400))

    shifted_word = _word(MIC, 100 + shift, 300 + shift)
    shifted_turns = (
        _turn(0, 0 + shift, 200 + shift),
        _turn(1, 150 + shift, 400 + shift),
    )

    base_result = map_words_to_speakers(
        words=[base_word],
        diarization=[_role_turns(MIC, base_turns)],
    )
    shifted_result = map_words_to_speakers(
        words=[shifted_word],
        diarization=[_role_turns(MIC, shifted_turns)],
    )
    assert base_result[0].status == shifted_result[0].status
    assert base_result[0].speaker == shifted_result[0].speaker


# 53. duplicate same-speaker invariance
@pytest.mark.parametrize("dup_count", [1, 2, 5])
def test_53_duplicate_same_speaker_invariance(dup_count: int):
    w = _word(MIC, 0, 300)
    single_turn = _turn(0, 0, 200)
    turns = tuple(single_turn for _ in range(dup_count))
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, turns)],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 0)


# 54. turn permutation does not change coverage-based assignment (unique winner)
@pytest.mark.parametrize(
    "permutation",
    [
        (0, 1, 2),
        (2, 0, 1),
        (1, 2, 0),
    ],
)
def test_54_turn_permutation_invariance(permutation: tuple[int, int, int]):
    base_turns = [
        _turn(0, 0, 50),  # 50ms
        _turn(1, 0, 200),  # 200ms — clear winner
        _turn(2, 180, 200),  # 20ms
    ]
    permuted_turns = tuple(base_turns[i] for i in permutation)

    w = _word(MIC, 0, 200)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, permuted_turns)],
    )
    assert result[0].status is SpeakerAttributionStatus.ASSIGNED
    assert result[0].speaker == MeetingSpeakerKey(MIC, 1)


# ══════════════════════════════════════════════════════════════════════════════
# §41. MeetingSpeakerKey direct DTO validation
# ══════════════════════════════════════════════════════════════════════════════


# 55. speaker_index=0 valid
def test_55_key_speaker_0_valid():
    key = MeetingSpeakerKey(source_role=MIC, speaker_index=0)
    assert key.source_role is MIC
    assert key.speaker_index == 0


# 56. speaker_index=7 valid
def test_56_key_speaker_7_valid():
    key = MeetingSpeakerKey(source_role=REMOTE, speaker_index=7)
    assert key.source_role is REMOTE
    assert key.speaker_index == 7


# 57. speaker_index=True rejected
def test_57_key_bool_true_rejected():
    with pytest.raises(
        SpeakerMappingConfigError, match="speaker_index must be an int, not bool"
    ):
        MeetingSpeakerKey(source_role=MIC, speaker_index=True)  # type: ignore[arg-type]


# 58. speaker_index=False rejected
def test_58_key_bool_false_rejected():
    with pytest.raises(
        SpeakerMappingConfigError, match="speaker_index must be an int, not bool"
    ):
        MeetingSpeakerKey(source_role=MIC, speaker_index=False)  # type: ignore[arg-type]


# 59. speaker_index=-1 rejected
def test_59_key_negative_rejected():
    with pytest.raises(SpeakerMappingConfigError, match="speaker_index must be >= 0"):
        MeetingSpeakerKey(source_role=MIC, speaker_index=-1)


# 60. speaker_index=1.0 rejected
def test_60_key_float_rejected():
    with pytest.raises(SpeakerMappingConfigError, match="speaker_index must be an int"):
        MeetingSpeakerKey(source_role=MIC, speaker_index=1.0)  # type: ignore[arg-type]


# 61. invalid source_role rejected
def test_61_key_invalid_role_rejected():
    with pytest.raises(
        SpeakerMappingConfigError, match="source_role must be MeetingTrackRole"
    ):
        MeetingSpeakerKey(source_role="INVALID", speaker_index=0)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# §42. Three-way exact tie regression
# ══════════════════════════════════════════════════════════════════════════════


# 62. three-way exact tie → AMBIGUOUS
def test_62_three_way_exact_tie():
    w = _word(MIC, 100, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[
            _role_turns(
                MIC,
                (
                    _turn(0, 100, 200),
                    _turn(1, 100, 200),
                    _turn(7, 100, 200),
                ),
            ),
        ],
    )
    assert result[0].speaker is None
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS


# 63. three-way tie permutation invariant
@pytest.mark.parametrize(
    "permutation",
    [
        (0, 1, 2),
        (2, 0, 1),
        (1, 2, 0),
        (0, 2, 1),
        (1, 0, 2),
        (2, 1, 0),
    ],
)
def test_63_three_way_tie_permutation_invariant(permutation: tuple[int, int, int]):
    base_turns = [
        _turn(0, 100, 200),
        _turn(1, 100, 200),
        _turn(7, 100, 200),
    ]
    permuted_turns = tuple(base_turns[i] for i in permutation)

    w = _word(MIC, 100, 300)
    result = map_words_to_speakers(
        words=[w],
        diarization=[_role_turns(MIC, permuted_turns)],
    )
    assert result[0].speaker is None
    assert result[0].status is SpeakerAttributionStatus.AMBIGUOUS
