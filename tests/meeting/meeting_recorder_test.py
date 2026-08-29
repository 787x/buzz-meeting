import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile

from buzz.meeting import meeting_recorder as meeting_recorder_module
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderInputError,
    MeetingRecorderOperationalError,
    MeetingRecorderState,
    MeetingRecorderStateError,
)


class FakeArchiveWriter:
    def __init__(
        self,
        output_path: Path,
        sample_rate: int,
        *,
        blocked: bool = False,
        fail_at: str | None = None,
        block_failure_cleanup: bool = False,
    ) -> None:
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.blocked = blocked
        self.fail_at = fail_at
        self.block_failure_cleanup = block_failure_cleanup
        self.blocks: list[np.ndarray] = []
        self.calls: list[str] = []
        self.thread_ids: list[int] = []
        self.write_entered = threading.Event()
        self.release_write = threading.Event()
        self.write_condition = threading.Condition()
        self.published = False
        self.discarded = False
        self.failure_cleanup_entered = threading.Event()
        self.release_failure_cleanup = threading.Event()

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.thread_ids.append(threading.get_ident())
        if self.fail_at == name:
            raise OSError(f"{name} failed")

    def write(self, pcm16: np.ndarray) -> None:
        self._record("write")
        self.write_entered.set()
        if self.blocked:
            assert self.release_write.wait(timeout=5)
        with self.write_condition:
            self.blocks.append(pcm16.copy())
            self.write_condition.notify_all()

    def flush(self) -> None:
        self._record("flush")

    def finalize(self) -> None:
        self._record("finalize")
        self._record("close")
        self._record("fsync")

    def publish(self) -> None:
        self._record("publish")
        self._record("rename")
        self.published = True

    def discard(self) -> None:
        self._record("discard")
        self.discarded = True

    def close_after_failure(self) -> None:
        self.calls.append("close_after_failure")
        self.thread_ids.append(threading.get_ident())
        self.failure_cleanup_entered.set()
        if self.block_failure_cleanup:
            assert self.release_failure_cleanup.wait(timeout=5)

    def wait_for_blocks(self, count: int) -> None:
        with self.write_condition:
            assert self.write_condition.wait_for(
                lambda: len(self.blocks) >= count,
                timeout=5,
            )


class FakeWriterFactory:
    def __init__(
        self,
        *,
        blocked: bool = False,
        fail_at: str | None = None,
        block_failure_cleanup: bool = False,
    ):
        self.blocked = blocked
        self.fail_at = fail_at
        self.block_failure_cleanup = block_failure_cleanup
        self.writer: FakeArchiveWriter | None = None

    def __call__(self, output_path: Path, sample_rate: int) -> FakeArchiveWriter:
        self.writer = FakeArchiveWriter(
            output_path,
            sample_rate,
            blocked=self.blocked,
            fail_at=self.fail_at,
            block_failure_cleanup=self.block_failure_cleanup,
        )
        return self.writer


class CountingEvent:
    """Event that deterministically exposes how many callers are waiting."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._condition = threading.Condition()
        self._wait_count = 0

    def clear(self) -> None:
        self._event.clear()

    def set(self) -> None:
        self._event.set()

    def wait(self, timeout=None) -> bool:
        with self._condition:
            self._wait_count += 1
            self._condition.notify_all()
        return self._event.wait(timeout)

    def wait_for_waiters(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(
                lambda: self._wait_count >= count,
                timeout=5,
            )


class PosixPublishArchiveWriter(FakeArchiveWriter):
    """Fake data writer that executes the real POSIX publish implementation."""

    def __init__(self, output_path: Path, sample_rate: int) -> None:
        super().__init__(output_path, sample_rate)
        self.partial_path = output_path.with_name(output_path.name + ".partial")
        self.partial_path.write_bytes(b"partial archive")

    def publish(self) -> None:
        meeting_recorder_module._publish_posix_no_replace(
            self.partial_path,
            self.output_path,
        )


class ControlledReadyEvent:
    """Hold start() after the writer has reported its startup outcome."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.waiting_after_ready = threading.Event()
        self.allow_wait_return = threading.Event()

    def clear(self) -> None:
        self._event.clear()

    def set(self) -> None:
        self._event.set()

    def wait(self, timeout=None) -> bool:
        if not self._event.wait(timeout):
            return False
        self.waiting_after_ready.set()
        return self.allow_wait_return.wait(timeout=5)


def make_fake_recorder(
    tmp_path,
    *,
    sample_rate: int = 10,
    max_buffer_seconds: float = 10.0,
    blocked: bool = False,
    fail_at: str | None = None,
    block_failure_cleanup: bool = False,
    on_error=None,
):
    factory = FakeWriterFactory(
        blocked=blocked,
        fail_at=fail_at,
        block_failure_cleanup=block_failure_cleanup,
    )
    recorder = MeetingRecorder(
        tmp_path / "meeting.wav",
        sample_rate,
        max_buffer_seconds=max_buffer_seconds,
        on_error=on_error,
        _writer_factory=factory,
    )
    return recorder, factory


def test_constructor_requires_final_wav_extension(tmp_path):
    with pytest.raises(ValueError, match=".wav extension"):
        MeetingRecorder(tmp_path / "meeting.rf64", 16_000)


def test_real_soundfile_writes_rf64_pcm16_mono_at_source_rate(tmp_path):
    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 44_100)
    first = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    second = np.array([0.5, 1.0], dtype=np.float32)

    recorder.start()
    assert recorder.enqueue(first)
    assert recorder.enqueue(second)
    result = recorder.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.published
    assert result.sample_count == 5
    assert result.duration_seconds == pytest.approx(5 / 44_100)
    assert output_path.read_bytes()[:4] == b"RF64"
    assert not (tmp_path / "meeting.wav.partial").exists()

    with soundfile.SoundFile(output_path) as audio_file:
        assert audio_file.format == "RF64"
        assert audio_file.subtype == "PCM_16"
        assert audio_file.samplerate == 44_100
        assert audio_file.channels == 1
        assert audio_file.frames == 5
        actual = audio_file.read(dtype="int16")
    np.testing.assert_array_equal(
        actual,
        np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16),
    )


def test_generated_rf64_is_accepted_by_ffprobe_when_available(tmp_path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is not installed")
    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 48_000)
    recorder.start()
    assert recorder.enqueue(np.ones(4_800, dtype=np.float32))
    recorder.stop()

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    assert stream["codec_name"] == "pcm_s16le"
    assert stream["sample_rate"] == "48000"
    assert stream["channels"] == 1
    assert float(stream["duration"]) == pytest.approx(0.1)


def test_generated_rf64_is_accepted_by_existing_whisper_audio_loader(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    from buzz.whisper_audio import load_audio

    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 48_000)
    recorder.start()
    assert recorder.enqueue(np.full(4_800, 0.25, dtype=np.float32))
    recorder.stop()

    decoded = load_audio(str(output_path), sr=16_000)
    assert decoded.dtype == np.float32
    assert decoded.size == 1_600
    assert np.mean(decoded) == pytest.approx(0.25, abs=1 / 32768)


def test_start_synchronously_rejects_runtime_without_rf64_pcm16(tmp_path):
    recorder = MeetingRecorder(tmp_path / "meeting.wav", 16_000)

    with patch(
        "buzz.meeting.meeting_recorder.soundfile.check_format",
        return_value=False,
    ), pytest.raises(MeetingRecorderOperationalError, match="does not support"):
        recorder.start()

    assert recorder.state == MeetingRecorderState.FAILED
    assert not (tmp_path / "meeting.wav.partial").exists()


def test_pcm16_conversion_has_explicit_nan_inf_and_clipping_policy(tmp_path):
    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 16_000)
    samples = np.array(
        [
            -np.inf,
            np.nan,
            np.inf,
            -2.0,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            2.0,
        ],
        dtype=np.float32,
    )

    recorder.start()
    assert recorder.enqueue(samples)
    recorder.stop()

    actual, sample_rate = soundfile.read(output_path, dtype="int16")
    assert sample_rate == 16_000
    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                -32768,
                0,
                32767,
                -32768,
                -32768,
                -16384,
                0,
                16384,
                32767,
                32767,
            ],
            dtype=np.int16,
        ),
    )


def test_borrowed_input_is_copied_and_multiple_blocks_keep_exact_order(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    recorder.start()
    samples = np.array([0.25, -0.25], dtype=np.float32)

    assert recorder.enqueue(samples)
    samples[:] = 1.0
    assert recorder.enqueue(np.array([0.5], dtype=np.float32))
    result = recorder.stop()

    assert result.sample_count == 3
    assert factory.writer is not None
    np.testing.assert_array_equal(
        np.concatenate(factory.writer.blocks),
        np.array([8192, -8192, 16384], dtype=np.int16),
    )


def test_empty_blocks_between_non_empty_blocks_do_not_reach_writer(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    first = np.array([0.25, -0.25], dtype=np.float32)
    second = np.array([0.5], dtype=np.float32)

    recorder.start()
    assert recorder.enqueue(first)
    assert recorder.enqueue(np.empty(0, dtype=np.float32))
    assert recorder.enqueue(np.empty(0, dtype=np.float32))
    assert recorder.enqueue(second)
    result = recorder.stop()

    assert result.sample_count == first.size + second.size
    assert recorder.accepted_sample_count == first.size + second.size
    assert factory.writer is not None
    assert len(factory.writer.blocks) == 2
    np.testing.assert_array_equal(
        np.concatenate(factory.writer.blocks),
        np.array([8192, -8192, 16384], dtype=np.int16),
    )


def test_valid_empty_blocks_are_noops_while_writer_is_blocked(tmp_path):
    errors = []
    recorder, factory = make_fake_recorder(
        tmp_path,
        blocked=True,
        on_error=errors.append,
    )
    recorder.start()
    first = np.array([0.25, -0.25], dtype=np.float32)
    assert recorder.enqueue(first)
    assert factory.writer is not None
    assert factory.writer.write_entered.wait(timeout=5)

    with recorder._condition:
        before = (
            len(recorder._queue),
            recorder._buffered_sample_count,
            recorder._accepted_sample_count,
            recorder._written_sample_count,
            recorder._producer_in_flight,
        )
    writer_calls = list(factory.writer.calls)

    with patch("buzz.meeting.meeting_recorder.np.array") as array_copy:
        for _ in range(10_000):
            assert recorder.enqueue(np.empty(0, dtype=np.float32))
        array_copy.assert_not_called()

    with recorder._condition:
        assert (
            len(recorder._queue),
            recorder._buffered_sample_count,
            recorder._accepted_sample_count,
            recorder._written_sample_count,
            recorder._producer_in_flight,
        ) == before
    assert factory.writer.calls == writer_calls
    assert not factory.writer.published
    assert errors == []
    assert recorder.state == MeetingRecorderState.RUNNING

    factory.writer.release_write.set()
    result = recorder.stop()

    assert result.sample_count == first.size
    assert len(factory.writer.blocks) == 1


def test_valid_empty_block_preserves_enqueue_lifecycle_semantics(tmp_path):
    recorder, _ = make_fake_recorder(tmp_path)
    empty = np.empty(0, dtype=np.float32)

    with pytest.raises(MeetingRecorderStateError, match="state CREATED"):
        recorder.enqueue(empty)

    recorder.start()
    assert recorder.enqueue(empty)
    recorder.request_stop()

    with pytest.raises(MeetingRecorderStateError, match="state STOPPING"):
        recorder.enqueue(empty)

    assert recorder.stop().state == MeetingRecorderState.STOPPED
    with pytest.raises(MeetingRecorderStateError, match="state STOPPED"):
        recorder.enqueue(empty)


def test_stop_before_start_is_noop_and_does_not_consume_recorder(tmp_path):
    recorder, _ = make_fake_recorder(tmp_path)

    before_start = recorder.stop()
    assert before_start.state == MeetingRecorderState.CREATED
    assert recorder.state == MeetingRecorderState.CREATED

    recorder.start()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))
    assert recorder.stop().state == MeetingRecorderState.STOPPED


def test_duplicate_start_is_rejected_and_duplicate_stop_returns_same_result(tmp_path):
    recorder, _ = make_fake_recorder(tmp_path)
    recorder.start()

    with pytest.raises(MeetingRecorderStateError, match="Cannot start"):
        recorder.start()

    first = recorder.stop()
    second = recorder.stop()
    assert second is first


def test_start_uses_explicit_open_outcome_when_concurrent_stop_finishes(tmp_path):
    recorder, _ = make_fake_recorder(tmp_path)
    controlled_ready = ControlledReadyEvent()
    recorder._writer_ready = controlled_ready
    start_errors: list[Exception] = []
    start_thread = threading.Thread(
        target=lambda: _capture_exception(recorder.start, start_errors)
    )
    start_thread.start()
    assert controlled_ready.waiting_after_ready.wait(timeout=5)

    stop_result = []
    stop_thread = threading.Thread(target=lambda: stop_result.append(recorder.stop()))
    stop_thread.start()
    with recorder._condition:
        assert recorder._condition.wait_for(
            lambda: recorder._result is not None,
            timeout=5,
        )
    controlled_ready.allow_wait_return.set()
    start_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert start_errors == []
    assert len(stop_result) == 1
    assert stop_result[0].state == MeetingRecorderState.STOPPED


def _capture_exception(call, errors: list[Exception]) -> None:
    try:
        call()
    except Exception as exc:
        errors.append(exc)


def test_zero_length_recording_is_a_valid_published_rf64_file(tmp_path):
    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 48_000)

    recorder.start()
    result = recorder.stop()

    assert result.sample_count == 0
    assert result.published
    with soundfile.SoundFile(output_path) as audio_file:
        assert audio_file.format == "RF64"
        assert audio_file.frames == 0
        assert audio_file.samplerate == 48_000


def test_ten_consecutive_recorder_assets_start_and_stop_cleanly(tmp_path):
    for index in range(10):
        output_path = tmp_path / f"meeting-{index}.wav"
        recorder = MeetingRecorder(output_path, 16_000)
        recorder.start()
        assert recorder.enqueue(np.full(160, index / 10, dtype=np.float32))
        result = recorder.stop()

        assert result.state == MeetingRecorderState.STOPPED
        assert result.sample_count == 160
        assert output_path.exists()
        assert not output_path.with_name(output_path.name + ".partial").exists()


def test_final_block_is_written_before_stop_returns(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path, blocked=True)
    recorder.start()
    assert recorder.enqueue(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert factory.writer is not None
    assert factory.writer.write_entered.wait(timeout=5)
    stop_finished = threading.Event()

    def stop_recorder():
        recorder.stop()
        stop_finished.set()

    stop_thread = threading.Thread(target=stop_recorder)
    stop_thread.start()
    assert not stop_finished.is_set()

    factory.writer.release_write.set()
    stop_thread.join(timeout=5)
    assert not stop_thread.is_alive()
    assert stop_finished.is_set()
    assert recorder.stop().sample_count == 3


def test_stop_waits_for_producer_copy_and_preserves_reserved_final_block(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    recorder.start()
    copy_entered = threading.Event()
    allow_copy = threading.Event()
    original_array = np.array

    def controlled_copy(*args, **kwargs):
        copy_entered.set()
        assert allow_copy.wait(timeout=5)
        return original_array(*args, **kwargs)

    accepted: list[bool] = []
    with patch(
        "buzz.meeting.meeting_recorder.np.array",
        side_effect=controlled_copy,
    ):
        producer = threading.Thread(
            target=lambda: accepted.append(
                recorder.enqueue(np.ones(3, dtype=np.float32))
            )
        )
        producer.start()
        assert copy_entered.wait(timeout=5)
        stop_finished = threading.Event()
        stop_thread = threading.Thread(
            target=lambda: (recorder.stop(), stop_finished.set())
        )
        stop_thread.start()
        assert not stop_finished.wait(timeout=0.05)
        allow_copy.set()
        producer.join(timeout=5)
        stop_thread.join(timeout=5)

    assert accepted == [True]
    assert stop_finished.is_set()
    assert recorder.stop().sample_count == 3
    assert factory.writer is not None and len(factory.writer.blocks) == 1


def test_copy_failure_releases_reservation_with_other_producer_and_stop(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    recorder.start()
    first_copy_entered = threading.Event()
    allow_first_copy_failure = threading.Event()
    original_array = np.array
    copy_lock = threading.Lock()
    copy_count = 0

    def controlled_copy(*args, **kwargs):
        nonlocal copy_count
        with copy_lock:
            copy_count += 1
            current = copy_count
        if current == 1:
            first_copy_entered.set()
            assert allow_first_copy_failure.wait(timeout=5)
            raise MemoryError("owned copy failed")
        return original_array(*args, **kwargs)

    outcomes: list[bool] = []
    with patch(
        "buzz.meeting.meeting_recorder.np.array",
        side_effect=controlled_copy,
    ):
        producer_a = threading.Thread(
            target=lambda: outcomes.append(
                recorder.enqueue(np.ones(3, dtype=np.float32))
            )
        )
        producer_a.start()
        assert first_copy_entered.wait(timeout=5)
        producer_b = threading.Thread(
            target=lambda: outcomes.append(
                recorder.enqueue(np.ones(2, dtype=np.float32))
            )
        )
        producer_b.start()
        producer_b.join(timeout=5)
        stop_result = []
        stop_thread = threading.Thread(
            target=lambda: stop_result.append(recorder.stop())
        )
        stop_thread.start()
        allow_first_copy_failure.set()
        producer_a.join(timeout=5)
        stop_thread.join(timeout=5)

    assert sorted(outcomes) == [False, True]
    assert len(stop_result) == 1
    assert stop_result[0].state == MeetingRecorderState.FAILED
    assert recorder.buffered_sample_count == 0
    assert recorder._producer_in_flight == 0
    assert factory.writer is not None


def test_writer_inflight_samples_still_count_toward_capacity(tmp_path):
    errors = []
    recorder, factory = make_fake_recorder(
        tmp_path,
        sample_rate=10,
        max_buffer_seconds=1.0,
        blocked=True,
        on_error=errors.append,
    )
    recorder.start()
    assert recorder.enqueue(np.ones(6, dtype=np.float32))
    assert factory.writer is not None
    assert factory.writer.write_entered.wait(timeout=5)

    assert recorder.buffered_sample_count == 6
    assert not recorder.enqueue(np.ones(5, dtype=np.float32))
    assert recorder.state == MeetingRecorderState.FAILED
    assert len(errors) == 1

    factory.writer.release_write.set()
    result = recorder.stop()
    assert result.state == MeetingRecorderState.FAILED
    assert result.error is not None
    assert not result.published
    assert "publish" not in factory.writer.calls


def test_single_oversized_block_fails_before_copy(tmp_path):
    errors = []
    recorder, _ = make_fake_recorder(
        tmp_path,
        sample_rate=4,
        max_buffer_seconds=1.0,
        on_error=errors.append,
    )
    recorder.start()
    samples = np.ones(5, dtype=np.float32)

    with patch("buzz.meeting.meeting_recorder.np.array") as array_copy:
        assert not recorder.enqueue(samples)
        array_copy.assert_not_called()

    result = recorder.stop()
    assert result.state == MeetingRecorderState.FAILED
    assert len(errors) == 1


def test_invalid_input_is_explicit_and_fails_archive_without_escaping_later(tmp_path):
    errors = []
    recorder, _ = make_fake_recorder(tmp_path, on_error=errors.append)
    recorder.start()

    with pytest.raises(MeetingRecorderInputError, match="float32"):
        recorder.enqueue(np.ones(2, dtype=np.float64))

    assert not recorder.enqueue(np.empty(0, dtype=np.float32))
    assert recorder.stop().state == MeetingRecorderState.FAILED
    assert len(errors) == 1


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.empty(0, dtype=np.float64), "float32"),
        (np.empty((0, 1), dtype=np.float32), "mono"),
        ([], "numpy array"),
    ],
)
def test_empty_input_must_still_satisfy_type_and_shape_contract(
    tmp_path,
    samples,
    message,
):
    errors = []
    recorder, _ = make_fake_recorder(tmp_path, on_error=errors.append)
    recorder.start()

    with pytest.raises(MeetingRecorderInputError, match=message):
        recorder.enqueue(samples)

    assert recorder.stop().state == MeetingRecorderState.FAILED
    assert len(errors) == 1


@pytest.mark.parametrize(
    "fail_at",
    ["write", "flush", "finalize", "close", "fsync", "publish", "rename"],
)
def test_writer_stage_failure_is_failed_once_and_never_publishes(
    tmp_path,
    fail_at,
):
    errors = []
    sample_rate = 1 if fail_at == "flush" else 10
    recorder, factory = make_fake_recorder(
        tmp_path,
        sample_rate=sample_rate,
        max_buffer_seconds=10,
        fail_at=fail_at,
        on_error=errors.append,
    )
    recorder.start()
    sample_count = 5 if fail_at == "flush" else 2
    assert recorder.enqueue(np.ones(sample_count, dtype=np.float32))

    result = recorder.stop()

    assert result.state == MeetingRecorderState.FAILED
    assert result.error is errors[0]
    assert len(errors) == 1
    assert not result.published
    assert factory.writer is not None
    assert not factory.writer.published


def test_writer_error_callback_can_reenter_recorder_stop(tmp_path):
    callback_returned = threading.Event()
    callback_results = []
    recorder_ref = {}

    def on_error(_):
        callback_results.append(recorder_ref["recorder"].stop())
        callback_returned.set()

    recorder, _ = make_fake_recorder(
        tmp_path,
        fail_at="write",
        on_error=on_error,
    )
    recorder_ref["recorder"] = recorder
    recorder.start()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))

    assert callback_returned.wait(timeout=5)
    assert recorder._writer_thread is not None
    recorder._writer_thread.join(timeout=5)
    second = recorder.stop()

    assert not recorder._writer_thread.is_alive()
    assert callback_results == [second]
    assert second.state == MeetingRecorderState.FAILED


def test_external_stop_waits_for_cleanup_after_reentrant_writer_stop(tmp_path):
    callback_returned = threading.Event()
    external_returned = threading.Event()
    callback_results = []
    external_results = []
    recorder_ref = {}

    def on_error(_):
        callback_results.append(recorder_ref["recorder"].stop())
        callback_returned.set()

    recorder, factory = make_fake_recorder(
        tmp_path,
        fail_at="write",
        block_failure_cleanup=True,
        on_error=on_error,
    )
    writer_done = CountingEvent()
    recorder._writer_done = writer_done
    recorder_ref["recorder"] = recorder
    recorder.start()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))
    assert callback_returned.wait(timeout=5)
    assert factory.writer is not None
    assert factory.writer.failure_cleanup_entered.wait(timeout=5)

    def stop_externally() -> None:
        external_results.append(recorder.stop())
        external_returned.set()

    external_stop = threading.Thread(target=stop_externally)
    external_stop.start()
    writer_done.wait_for_waiters(1)
    assert not external_returned.is_set()

    factory.writer.release_failure_cleanup.set()
    external_stop.join(timeout=5)

    assert not external_stop.is_alive()
    assert recorder._writer_thread is not None
    assert not recorder._writer_thread.is_alive()
    assert callback_results == external_results
    assert external_results[0].state == MeetingRecorderState.FAILED


def test_duplicate_external_stops_all_wait_for_failed_writer_cleanup(tmp_path):
    recorder, factory = make_fake_recorder(
        tmp_path,
        fail_at="write",
        block_failure_cleanup=True,
    )
    writer_done = CountingEvent()
    recorder._writer_done = writer_done
    recorder.start()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))
    assert factory.writer is not None
    assert factory.writer.failure_cleanup_entered.wait(timeout=5)

    results = []
    returned = [threading.Event(), threading.Event()]

    def stop_externally(index: int) -> None:
        results.append(recorder.stop())
        returned[index].set()

    stop_threads = [
        threading.Thread(target=stop_externally, args=(index,)) for index in range(2)
    ]
    for stop_thread in stop_threads:
        stop_thread.start()
    writer_done.wait_for_waiters(2)
    assert not any(event.is_set() for event in returned)

    factory.writer.release_failure_cleanup.set()
    for stop_thread in stop_threads:
        stop_thread.join(timeout=5)

    assert all(not stop_thread.is_alive() for stop_thread in stop_threads)
    assert recorder._writer_thread is not None
    assert not recorder._writer_thread.is_alive()
    assert len(results) == 2
    assert results[0] is results[1]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("os.fsync", "fsync failed"),
        ("os.rename" if os.name == "nt" else "os.link", "publish failed"),
    ],
)
def test_real_writer_fsync_and_publish_failures_keep_only_partial(
    tmp_path,
    target,
    message,
):
    output_path = tmp_path / "meeting.wav"
    errors = []
    recorder = MeetingRecorder(output_path, 16_000, on_error=errors.append)
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))

    with patch(
        f"buzz.meeting.meeting_recorder.{target}",
        side_effect=OSError(message),
    ):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.FAILED
    assert result.error is errors[0]
    assert len(errors) == 1
    assert not result.published
    assert not output_path.exists()
    assert output_path.with_name("meeting.wav.partial").exists()


def test_publish_never_overwrites_final_created_during_recording(tmp_path):
    output_path = tmp_path / "meeting.wav"
    partial_path = tmp_path / "meeting.wav.partial"
    recorder = MeetingRecorder(output_path, 16_000)
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))
    output_path.write_bytes(b"created by another actor")

    result = recorder.stop()

    assert result.state == MeetingRecorderState.FAILED
    assert not result.published
    assert output_path.read_bytes() == b"created by another actor"
    assert partial_path.exists()


def test_mocked_posix_publish_does_not_replace_race_created_final(tmp_path):
    output_path = tmp_path / "meeting.wav"
    partial_path = tmp_path / "meeting.wav.partial"
    recorder = MeetingRecorder(
        output_path,
        16_000,
        _writer_factory=PosixPublishArchiveWriter,
    )
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))
    output_path.write_bytes(b"created by another actor")

    with patch.object(
        meeting_recorder_module.os,
        "link",
        side_effect=FileExistsError("destination exists"),
    ):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.FAILED
    assert result.error is not None
    assert not result.published
    assert output_path.read_bytes() == b"created by another actor"
    assert partial_path.read_bytes() == b"partial archive"


def test_mocked_posix_partial_unlink_failure_stays_published(tmp_path):
    output_path = tmp_path / "meeting.wav"
    partial_path = tmp_path / "meeting.wav.partial"
    recorder = MeetingRecorder(
        output_path,
        16_000,
        _writer_factory=PosixPublishArchiveWriter,
    )
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))
    original_unlink = Path.unlink

    def fail_partial_unlink(path, *args, **kwargs):
        if path == partial_path:
            raise OSError("partial unlink failed")
        return original_unlink(path, *args, **kwargs)

    with (
        patch.object(Path, "unlink", new=fail_partial_unlink),
        patch.object(meeting_recorder_module, "_sync_directory_best_effort"),
    ):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.error is None
    assert result.published
    assert output_path.read_bytes() == b"partial archive"
    assert partial_path.read_bytes() == b"partial archive"


def test_mocked_posix_directory_sync_failure_stays_published(tmp_path):
    output_path = tmp_path / "meeting.wav"
    partial_path = tmp_path / "meeting.wav.partial"
    recorder = MeetingRecorder(
        output_path,
        16_000,
        _writer_factory=PosixPublishArchiveWriter,
    )
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))

    with patch.object(
        meeting_recorder_module.os,
        "open",
        side_effect=OSError("directory open failed"),
    ):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.error is None
    assert result.published
    assert output_path.read_bytes() == b"partial archive"
    assert not partial_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync semantics")
def test_directory_fsync_failure_after_publish_does_not_unpublish(tmp_path):
    output_path = tmp_path / "meeting.wav"
    recorder = MeetingRecorder(output_path, 16_000)
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))

    with patch(
        "buzz.meeting.meeting_recorder.os.open",
        side_effect=OSError("directory open failed"),
    ):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.published
    assert output_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link publish semantics")
def test_partial_unlink_failure_after_link_commit_remains_published(tmp_path):
    output_path = tmp_path / "meeting.wav"
    partial_path = tmp_path / "meeting.wav.partial"
    recorder = MeetingRecorder(output_path, 16_000)
    recorder.start()
    assert recorder.enqueue(np.ones(160, dtype=np.float32))
    original_unlink = Path.unlink

    def fail_partial_unlink(path, *args, **kwargs):
        if path == partial_path:
            raise OSError("partial unlink failed")
        return original_unlink(path, *args, **kwargs)

    with patch.object(Path, "unlink", new=fail_partial_unlink):
        result = recorder.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.published
    assert output_path.exists()
    assert partial_path.exists()


def test_open_failure_is_synchronous_and_reports_once(tmp_path):
    errors = []

    def failing_factory(output_path, sample_rate):
        raise OSError("open failed")

    recorder = MeetingRecorder(
        tmp_path / "meeting.wav",
        16_000,
        on_error=errors.append,
        _writer_factory=failing_factory,
    )

    with pytest.raises(MeetingRecorderOperationalError, match="open failed"):
        recorder.start()

    assert recorder.state == MeetingRecorderState.FAILED
    assert len(errors) == 1
    assert recorder.stop().error is errors[0]


def test_existing_final_and_partial_are_never_overwritten(tmp_path):
    final_path = tmp_path / "meeting.wav"
    final_path.write_bytes(b"existing")
    recorder = MeetingRecorder(final_path, 16_000)

    with pytest.raises(MeetingRecorderOperationalError, match="already exists"):
        recorder.start()
    assert final_path.read_bytes() == b"existing"

    final_path.unlink()
    partial_path = tmp_path / "meeting.wav.partial"
    partial_path.write_bytes(b"partial")
    recorder = MeetingRecorder(final_path, 16_000)
    with pytest.raises(MeetingRecorderOperationalError, match="partial output"):
        recorder.start()
    assert partial_path.read_bytes() == b"partial"


def test_cancel_empty_start_discards_only_empty_partial(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    recorder.start()

    result = recorder.cancel_empty_start()

    assert result.state == MeetingRecorderState.STOPPED
    assert not result.published
    assert factory.writer is not None
    assert factory.writer.discarded


def test_cancel_empty_start_refuses_to_delete_accepted_audio(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path, blocked=True)
    recorder.start()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))

    with pytest.raises(MeetingRecorderStateError, match="after audio was accepted"):
        recorder.cancel_empty_start()

    assert factory.writer is not None
    factory.writer.release_write.set()
    result = recorder.stop()
    assert result.published
    assert not factory.writer.discarded


def test_long_synthetic_stream_remains_bounded_and_ordered(tmp_path):
    recorder, factory = make_fake_recorder(
        tmp_path,
        sample_rate=100,
        max_buffer_seconds=1.0,
    )
    recorder.start()
    assert factory.writer is not None

    # Ten minutes at 100 Hz, delivered in 100 ms blocks. Waiting for each fake
    # write keeps the test deterministic while still proving duration-independent
    # queue memory and ordering over a long logical stream.
    block_count = 6_000
    for index in range(block_count):
        block = np.full(10, (index % 100) / 100.0, dtype=np.float32)
        assert recorder.enqueue(block)
        factory.writer.wait_for_blocks(index + 1)
        assert recorder.buffered_sample_count <= recorder.max_buffered_samples

    result = recorder.stop()
    assert result.sample_count == 60_000
    assert result.duration_seconds == 600
    assert len(factory.writer.blocks) == block_count


def test_all_writer_operations_run_off_the_producer_thread(tmp_path):
    recorder, factory = make_fake_recorder(tmp_path)
    recorder.start()
    producer_thread_id = threading.get_ident()
    assert recorder.enqueue(np.ones(2, dtype=np.float32))
    recorder.stop()

    assert factory.writer is not None
    assert factory.writer.thread_ids
    assert set(factory.writer.thread_ids) == {
        recorder._writer_thread.ident,
    }
    assert producer_thread_id not in factory.writer.thread_ids
