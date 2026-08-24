import pytest

import buzz.transcriber.incremental_transcript as incremental_transcript_module
from buzz.transcriber.incremental_transcript import IncrementalTranscript


def test_first_hypothesis_is_provisional():
    transcript = IncrementalTranscript()

    update = transcript.update("we need to review")

    assert update.newly_committed_text == ""
    assert update.provisional_text == "we need to review"
    assert transcript.snapshot() == ""
    assert transcript.snapshot(include_provisional=True) == "we need to review"


def test_exact_english_overlap_commits_previous_through_anchor():
    transcript = IncrementalTranscript()
    transcript.update("we need to review")

    update = transcript.update("review the budget tomorrow")

    assert update.newly_committed_text == "we need to review"
    assert update.provisional_text == " the budget tomorrow"


def test_three_cycle_english_overlap():
    transcript = IncrementalTranscript()
    transcript.update("we need to review")
    transcript.update("review the budget tomorrow")

    update = transcript.update("tomorrow and send the report")

    assert update.newly_committed_text == " the budget tomorrow"
    assert update.provisional_text == " and send the report"
    assert transcript.snapshot(include_provisional=True) == (
        "we need to review the budget tomorrow and send the report"
    )


def test_four_cycle_english_overlap():
    transcript = IncrementalTranscript()
    for hypothesis in (
        "we need to review",
        "review the budget tomorrow",
        "tomorrow and send the report",
        "report before Friday",
    ):
        transcript.update(hypothesis)

    assert transcript.snapshot(include_provisional=True) == (
        "we need to review the budget tomorrow and send the report before Friday"
    )


def test_revised_overlap_uses_conservative_old_prefix():
    transcript = IncrementalTranscript()
    transcript.update("review a budget")

    update = transcript.update("the budget tomorrow")

    assert update.newly_committed_text == "review a budget"
    assert update.provisional_text == " tomorrow"
    assert transcript.snapshot(include_provisional=True) == "review a budget tomorrow"


def test_no_overlap_adds_safe_latin_seam():
    transcript = IncrementalTranscript()
    transcript.update("first agenda item")

    update = transcript.update("the server deployment")

    assert update.newly_committed_text == "first agenda item"
    assert update.provisional_text == " the server deployment"
    assert transcript.snapshot(include_provisional=True) == (
        "first agenda item the server deployment"
    )


def test_no_overlap_does_not_add_space_between_cjk_text():
    transcript = IncrementalTranscript()
    transcript.update("第一项议程")

    update = transcript.update("服务器部署")

    assert update.provisional_text == "服务器部署"
    assert transcript.snapshot(include_provisional=True) == "第一项议程服务器部署"


def test_two_character_cjk_overlap_is_reliable():
    transcript = IncrementalTranscript()
    transcript.update("我们需要审核")

    update = transcript.update("审核明天的预算")

    assert update.newly_committed_text == "我们需要审核"
    assert update.provisional_text == "明天的预算"


def test_three_cycle_cjk_overlap():
    transcript = IncrementalTranscript()
    transcript.update("我们需要审核")
    transcript.update("审核明天的预算")
    transcript.update("明天的预算然后发送报告")

    assert transcript.snapshot(include_provisional=True) == (
        "我们需要审核明天的预算然后发送报告"
    )


def test_mixed_chinese_and_english_overlap():
    transcript = IncrementalTranscript()
    transcript.update("我们明天 review budget")

    update = transcript.update("review budget 后发送报告")

    assert update.newly_committed_text == "我们明天 review budget"
    assert update.provisional_text == " 后发送报告"


def test_case_and_punctuation_are_soft_for_comparison():
    transcript = IncrementalTranscript()
    transcript.update("we should go")

    update = transcript.update("We should go, tomorrow")

    assert update.newly_committed_text == "we should go"
    assert update.provisional_text == ", tomorrow"
    assert transcript.snapshot(include_provisional=True) == "we should go, tomorrow"


def test_whitespace_is_soft_for_open_ai_overlap():
    transcript = IncrementalTranscript()
    transcript.update("Open AI")

    update = transcript.update("OpenAI is ready")

    assert update.newly_committed_text == "Open AI"
    assert update.provisional_text == " is ready"
    assert transcript.snapshot(include_provisional=True) == "Open AI is ready"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("this is very", "very very important", "this is very very important"),
        ("we said that", "that that matters", "we said that that matters"),
        ("we had", "had had enough", "we had had enough"),
    ],
)
def test_legitimate_repeated_words_are_preserved(first, second, expected):
    transcript = IncrementalTranscript()
    transcript.update(first)
    transcript.update(second)

    assert transcript.snapshot(include_provisional=True) == expected


def test_repeated_anchor_occurrences_choose_previous_suffix_and_current_prefix():
    transcript = IncrementalTranscript()
    transcript.update("we said that and that")

    transcript.update("that that matters")

    assert transcript.snapshot(include_provisional=True) == (
        "we said that and that that matters"
    )


def test_repeated_cjk_anchor_occurrences_preserve_old_prefix():
    transcript = IncrementalTranscript()
    transcript.update("审核然后审核")

    transcript.update("审核审核通过")

    assert transcript.snapshot(include_provisional=True) == "审核然后审核审核通过"


def test_repeated_anchor_ranking_is_deterministic():
    results = []
    for _ in range(20):
        transcript = IncrementalTranscript()
        transcript.update("we said that and that")
        transcript.update("that that matters")
        results.append(transcript.snapshot(include_provisional=True))

    assert results == ["we said that and that that matters"] * 20


def test_minimum_score_cjk_anchor_requires_exact_edges():
    transcript = IncrementalTranscript()
    transcript.update("上周会议需要审核")

    transcript.update("需要安排服务器部署")

    assert transcript.snapshot(include_provisional=True) == (
        "上周会议需要审核需要安排服务器部署"
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("Fus", "Fußball", "Fus Fußball"),
        ("proof", "prooﬃce", "proof prooﬃce"),
    ],
)
def test_normalization_expansion_cannot_be_partially_consumed(
    previous,
    current,
    expected,
):
    transcript = IncrementalTranscript()
    transcript.update(previous)

    transcript.update(current)

    assert transcript.snapshot(include_provisional=True) == expected


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("we go", "go go now", "we go go now"),
        ("we had", "had had enough", "we had had enough"),
        ("他说不", "不不可以", "他说不不可以"),
        ("他说对", "对对没错", "他说对对没错"),
    ],
)
def test_short_repeated_edge_token_preserves_second_occurrence(
    previous,
    current,
    expected,
):
    transcript = IncrementalTranscript()
    transcript.update(previous)

    transcript.update(current)

    assert transcript.snapshot(include_provisional=True) == expected


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("hello,", "world", "hello, world"),
        ("hello.", "Next", "hello. Next"),
        ("hello!", "Next", "hello! Next"),
        ("hello?", "Next", "hello? Next"),
        ("hello;", "world", "hello; world"),
        ("hello", ",", "hello,"),
        ("hello", ".", "hello."),
        ("中文", "继续", "中文继续"),
        ("中文", "，", "中文，"),
    ],
)
def test_no_overlap_seam_spacing(previous, current, expected):
    transcript = IncrementalTranscript()
    transcript.update(previous)

    transcript.update(current)

    assert transcript.snapshot(include_provisional=True) == expected


@pytest.mark.parametrize("blank", ["", " ", "\n\t"])
def test_blank_hypothesis_is_a_no_op(blank):
    transcript = IncrementalTranscript()
    transcript.update("existing provisional")

    update = transcript.update(blank)

    assert update.newly_committed_text == ""
    assert update.provisional_text == "existing provisional"
    assert transcript.snapshot() == ""


def test_long_hypothesis_bounds_provisional_text():
    transcript = IncrementalTranscript(max_revision_chars=256)

    update = transcript.update("a" * 300)

    assert update.newly_committed_text == "a" * 44
    assert update.provisional_text == "a" * 256
    assert len(update.provisional_text) == 256


def test_forced_commit_does_not_split_combining_sequence():
    transcript = IncrementalTranscript(max_revision_chars=256)
    text = "a" * 44 + "e\u0301" + "b" * 255

    update = transcript.update(text)

    assert update.newly_committed_text.endswith("e\u0301")
    assert update.provisional_text == "b" * 255
    assert not update.provisional_text.startswith("\u0301")


def test_forced_commit_does_not_split_multiple_combining_marks():
    transcript = IncrementalTranscript(max_revision_chars=256)
    text = "a" * 44 + "e\u0301\u0327" + "b" * 254

    update = transcript.update(text)

    assert update.newly_committed_text.endswith("e\u0301\u0327")
    assert update.provisional_text == "b" * 254


def test_committed_history_is_never_rewritten():
    transcript = IncrementalTranscript()
    transcript.update("we need to review")
    transcript.update("review the budget tomorrow")
    committed_prefix = transcript.snapshot()

    transcript.update("tomorrow and send the report")
    transcript.update("unrelated next topic")

    assert transcript.snapshot().startswith(committed_prefix)
    assert committed_prefix == "we need to review"


def test_finalize_commits_all_provisional_text():
    transcript = IncrementalTranscript()
    transcript.update("final recognized words")

    update = transcript.finalize()

    assert update.newly_committed_text == "final recognized words"
    assert update.provisional_text == ""
    assert transcript.snapshot() == "final recognized words"
    assert transcript.snapshot(include_provisional=True) == "final recognized words"


def test_finalize_is_idempotent():
    transcript = IncrementalTranscript()
    transcript.update("final recognized words")
    transcript.finalize()

    update = transcript.finalize()

    assert update.newly_committed_text == ""
    assert update.provisional_text == ""
    assert transcript.snapshot() == "final recognized words"


def test_reset_starts_a_new_transcript():
    transcript = IncrementalTranscript()
    transcript.update("old recording")
    transcript.finalize()

    transcript.reset()
    update = transcript.update("new recording")

    assert transcript.snapshot() == ""
    assert update.provisional_text == "new recording"
    assert transcript.snapshot(include_provisional=True) == "new recording"


def test_snapshot_can_exclude_or_include_provisional():
    transcript = IncrementalTranscript()
    transcript.update("we need to review")
    transcript.update("review the budget tomorrow")

    assert transcript.snapshot() == "we need to review"
    assert transcript.snapshot(include_provisional=True) == (
        "we need to review the budget tomorrow"
    )


def test_single_cjk_character_is_below_anchor_threshold():
    transcript = IncrementalTranscript()
    transcript.update("测试甲")
    transcript.update("甲继续")

    assert transcript.snapshot(include_provisional=True) == "测试甲甲继续"


def test_non_bmp_text_does_not_disturb_display_mapping():
    transcript = IncrementalTranscript()
    transcript.update("🙂 Open AI")

    update = transcript.update("openai continues 🙂")

    assert update.newly_committed_text == "🙂 Open AI"
    assert update.provisional_text == " continues 🙂"
    assert transcript.snapshot(include_provisional=True) == "🙂 Open AI continues 🙂"


def test_matcher_inputs_are_bounded(monkeypatch):
    seen_calls = []
    original_comparison_units = incremental_transcript_module._comparison_units

    def recording_comparison_units(text, *, max_units=None):
        units = original_comparison_units(text, max_units=max_units)
        seen_calls.append((len(text), max_units, len(units)))
        return units

    monkeypatch.setattr(
        incremental_transcript_module,
        "_comparison_units",
        recording_comparison_units,
    )
    transcript = IncrementalTranscript(
        max_revision_chars=32,
        match_window_units=8,
    )
    transcript.update("a" * 1_000)
    transcript.update("a" * 8 + "b" * 1_000)

    bounded_current_calls = [call for call in seen_calls if call[1] is not None]
    assert bounded_current_calls
    assert all(max_units == 8 for _, max_units, _ in bounded_current_calls)
    assert all(unit_count <= 8 for _, _, unit_count in bounded_current_calls)


def test_thousands_of_updates_keep_revision_and_match_inputs_bounded(monkeypatch):
    previous_lengths = []
    original_find_anchor = IncrementalTranscript._find_anchor

    def recording_find_anchor(self, previous, current):
        previous_lengths.append(len(previous))
        return original_find_anchor(self, previous, current)

    monkeypatch.setattr(
        IncrementalTranscript,
        "_find_anchor",
        recording_find_anchor,
    )
    transcript = IncrementalTranscript(max_revision_chars=32, match_window_units=8)

    for index in range(1_000):
        transcript.update("bravo" if index % 2 else "cider")

    assert len(transcript.snapshot()) > 4_000
    assert len(transcript.update("delta").provisional_text) <= 32
    assert previous_lengths
    assert max(previous_lengths) <= 32
