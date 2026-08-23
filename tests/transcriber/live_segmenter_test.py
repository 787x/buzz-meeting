import numpy as np
import pytest

from buzz.transcriber.live_segmenter import LiveSegmenter


SAMPLE_RATE = 1_000
THRESHOLD = 0.01


def pcm(seconds: float, amplitude: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    count = int(round(seconds * sample_rate))
    if amplitude == 0:
        return np.zeros(count, dtype=np.float32)
    values = np.full(count, amplitude, dtype=np.float32)
    values[1::2] *= -1
    return values


def make_segmenter(
    *,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = 12.0,
) -> LiveSegmenter:
    return LiveSegmenter(
        sample_rate=sample_rate,
        speech_threshold=THRESHOLD,
        max_utterance_seconds=max_seconds,
    )


def collect(segmenter: LiveSegmenter, waveform: np.ndarray) -> list[np.ndarray]:
    return segmenter.push(waveform) + segmenter.flush()


def test_meaningful_pause_produces_natural_endpoint():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(1.2, 0.1), pcm(0.6, 0), pcm(1.0, 0.2)))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 1
    assert len(utterances[0]) == pytest.approx(1_500, abs=20)
    assert np.max(np.abs(utterances[0][:1_200])) == pytest.approx(0.1)


def test_confirmed_short_phrase_emits_at_natural_pause_without_minimum_duration():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(0.1, 0.1), pcm(0.6, 0)))
    utterances: list[np.ndarray] = []
    emitted_at = None

    for offset in range(0, waveform.size, 20):
        utterances.extend(segmenter.push(waveform[offset:offset + 20]))
        if utterances:
            emitted_at = min(offset + 20, waveform.size)
            break

    assert emitted_at is not None
    assert 680 <= emitted_at <= 720
    assert emitted_at < 1_500
    assert len(utterances) == 1
    assert utterances[0].size == 400
    np.testing.assert_array_equal(utterances[0][:100], waveform[:100])


def test_unconfirmed_short_pulse_followed_by_natural_pause_does_not_emit():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(0.08, 0.1), pcm(0.6, 0)))

    assert segmenter.push(waveform) == []
    assert segmenter.flush() == []


def test_short_pause_does_not_produce_endpoint():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(1.2, 0.1), pcm(0.15, 0), pcm(1.0, 0.2)))

    assert segmenter.push(waveform) == []


def test_confirmed_short_phrase_is_not_split_by_short_pause():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(0.12, 0.1), pcm(0.15, 0), pcm(0.5, 0.2)))

    assert segmenter.push(waveform) == []
    utterances = segmenter.flush()

    assert len(utterances) == 1
    np.testing.assert_array_equal(utterances[0], waveform)


def test_continuous_speech_forces_split_at_normal_deadline():
    segmenter = make_segmenter()

    utterances = segmenter.push(pcm(12.1, 0.1))

    assert len(utterances) == 1
    assert utterances[0].size == 12_000


def test_forced_split_prefers_low_energy_notch():
    segmenter = make_segmenter()
    waveform = np.concatenate((
        pcm(10.1, 0.1),
        pcm(0.1, 0.02),
        pcm(1.9, 0.1),
    ))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 1
    assert utterances[0].size == pytest.approx(10_190, abs=20)
    assert utterances[0].size != 12_000


def test_uniform_speech_falls_back_to_exact_deadline():
    segmenter = make_segmenter(max_seconds=3.5)

    utterances = segmenter.push(pcm(3.6, 0.1))

    assert len(utterances) == 1
    assert utterances[0].size == 3_500


def test_all_silence_never_emits():
    segmenter = make_segmenter()

    assert segmenter.push(pcm(30, 0)) == []
    assert segmenter.flush() == []


def test_pulse_shorter_than_speech_confirmation_does_not_emit():
    segmenter = make_segmenter()
    waveform = np.concatenate((pcm(0.08, 0.1), pcm(2, 0)))

    assert collect(segmenter, waveform) == []


def test_confirmed_short_speech_is_returned_by_flush():
    segmenter = make_segmenter()
    speech = pcm(0.3, 0.1)

    assert segmenter.push(speech) == []
    utterances = segmenter.flush()

    assert len(utterances) == 1
    np.testing.assert_array_equal(utterances[0], speech)


def test_arbitrary_block_partition_is_invariant():
    waveform = np.concatenate((
        pcm(1.2, 0.1),
        pcm(0.6, 0),
        pcm(3.8, 0.2),
        pcm(0.8, 0),
        pcm(0.7, 0.15),
    ))
    expected = collect(make_segmenter(max_seconds=3.5), waveform)
    partitioned = make_segmenter(max_seconds=3.5)
    actual: list[np.ndarray] = []
    sizes = [1, 19, 20, 21, 137, 503]
    offset = 0
    index = 0
    while offset < waveform.size:
        size = sizes[index % len(sizes)]
        actual.extend(partitioned.push(waveform[offset:offset + size]))
        offset += size
        index += 1
    actual.extend(partitioned.flush())

    assert len(actual) == len(expected)
    for actual_utterance, expected_utterance in zip(actual, expected):
        np.testing.assert_array_equal(actual_utterance, expected_utterance)


def test_buffer_is_strictly_bounded_during_continuous_speech():
    segmenter = make_segmenter(max_seconds=3.5)
    maximum_seen = 0

    for _ in range(100):
        segmenter.push(pcm(0.2, 0.1))
        maximum_seen = max(maximum_seen, segmenter.buffered_sample_count)

    assert maximum_seen <= segmenter.max_buffered_sample_count


def test_long_stream_state_does_not_grow_with_meeting_duration():
    segmenter = make_segmenter(max_seconds=3.5)
    block = pcm(0.5, 0.1)

    for _ in range(7_200):  # One synthetic hour, streamed in bounded blocks.
        segmenter.push(block)
        assert segmenter.buffered_sample_count <= segmenter.max_buffered_sample_count


def test_low_energy_noise_does_not_fragment_or_grow():
    segmenter = make_segmenter()
    noise = pcm(0.5, THRESHOLD / 2)

    for _ in range(120):
        assert segmenter.push(noise) == []
        assert segmenter.buffered_sample_count <= 300


def test_multiple_natural_utterances():
    segmenter = make_segmenter()
    waveform = np.concatenate((
        pcm(1.2, 0.1), pcm(0.7, 0),
        pcm(1.2, 0.2), pcm(0.7, 0),
        pcm(1.2, 0.3), pcm(0.7, 0),
    ))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 3


def test_natural_endpoint_does_not_repeat_speech_pcm():
    segmenter = make_segmenter()
    waveform = np.concatenate((
        pcm(1.2, 0.1), pcm(0.7, 0), pcm(1.2, 0.2), pcm(0.7, 0),
    ))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 2
    assert np.count_nonzero(np.isclose(np.abs(utterances[0]), 0.2)) == 0
    assert np.count_nonzero(np.isclose(np.abs(utterances[1]), 0.1)) == 0


def test_forced_endpoint_partitions_pcm_without_overlap():
    segmenter = make_segmenter(max_seconds=3.5)
    waveform = pcm(7.2, 0.1)

    utterances = segmenter.push(waveform)
    utterances.extend(segmenter.flush())

    np.testing.assert_array_equal(np.concatenate(utterances), waveform)


def test_forced_early_cut_preserves_exact_remainder_in_later_segments():
    segmenter = make_segmenter(max_seconds=3.5)
    first_push = np.concatenate((
        pcm(2.9, 0.1),
        pcm(0.1, 0.02),
        pcm(0.6, 0.1),
    ))
    second_push = pcm(3.0, 0.2)
    waveform = np.concatenate((first_push, second_push))

    utterances = segmenter.push(first_push)
    assert len(utterances) == 1
    assert utterances[0].size < 3_500

    utterances.extend(segmenter.push(second_push))
    utterances.extend(segmenter.flush())

    assert len(utterances) >= 2
    np.testing.assert_array_equal(np.concatenate(utterances), waveform)


def test_append_and_correct_deadline_remains_bounded_across_multiple_cycles():
    transcription_step_samples = 3_500
    segmenter = make_segmenter(max_seconds=3.5)
    waveform = pcm(14.2, 0.1)

    utterances = segmenter.push(waveform)

    assert len(utterances) == 4
    update_positions = np.cumsum([utterance.size for utterance in utterances])
    update_intervals = np.diff(np.concatenate(([0], update_positions)))
    assert np.all(update_intervals <= transcription_step_samples)
    np.testing.assert_array_equal(
        np.concatenate(utterances),
        waveform[:update_positions[-1]],
    )


def test_emitted_utterance_owns_storage_independent_of_input_and_history():
    segmenter = make_segmenter()
    input_buffer = np.concatenate((pcm(0.2, 0.1), pcm(0.6, 0)))
    original = input_buffer.copy()

    utterances = segmenter.push(input_buffer)

    assert len(utterances) == 1
    utterance = utterances[0]
    input_buffer.fill(0.9)
    np.testing.assert_array_equal(utterance, original[:500])
    assert not np.shares_memory(utterance, input_buffer)
    assert utterance.base is None


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 48_000])
def test_timing_parameters_scale_with_sample_rate(sample_rate):
    segmenter = make_segmenter(sample_rate=sample_rate, max_seconds=3.5)

    utterances = segmenter.push(pcm(3.6, 0.1, sample_rate))

    assert len(utterances) == 1
    assert utterances[0].size == int(3.5 * sample_rate)
    assert segmenter.max_buffered_sample_count == (
        int(3.5 * sample_rate) + int(0.02 * sample_rate) - 1
    )


def test_analysis_frame_split_across_pushes():
    waveform = np.concatenate((pcm(1.2, 0.1), pcm(0.6, 0), pcm(0.5, 0.2)))
    expected = make_segmenter().push(waveform)
    segmenter = make_segmenter()
    actual: list[np.ndarray] = []

    for sample in waveform:
        actual.extend(segmenter.push(np.array([sample], dtype=np.float32)))

    assert len(actual) == len(expected) == 1
    np.testing.assert_array_equal(actual[0], expected[0])


def test_empty_and_invalid_inputs():
    segmenter = make_segmenter()

    assert segmenter.push(np.empty(0, dtype=np.float32)) == []
    with pytest.raises(ValueError, match="dtype"):
        segmenter.push(np.ones(10, dtype=np.float64))
    with pytest.raises(ValueError, match="mono"):
        segmenter.push(np.ones((10, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="numpy"):
        segmenter.push([0.1])  # type: ignore[arg-type]


def test_forced_candidate_tie_breaks_to_latest_minimum():
    segmenter = make_segmenter(max_seconds=3.5)
    waveform = np.concatenate((
        pcm(2.7, 0.1),
        pcm(0.02, 0.02),
        pcm(0.46, 0.1),
        pcm(0.02, 0.02),
        pcm(0.4, 0.1),
    ))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 1
    assert utterances[0].size == pytest.approx(3_190, abs=20)


def test_append_and_correct_deadline_prefers_recent_notch():
    segmenter = make_segmenter(max_seconds=3.5)
    waveform = np.concatenate((pcm(2.9, 0.1), pcm(0.1, 0.02), pcm(0.6, 0.1)))

    utterances = segmenter.push(waveform)

    assert len(utterances) == 1
    assert utterances[0].size == pytest.approx(2_990, abs=20)
    assert utterances[0].size <= 3_500
