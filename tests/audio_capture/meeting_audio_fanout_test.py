from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile

from buzz.audio_capture.meeting_audio_fanout import (
    MeetingAudioFanout,
    MeetingAudioFanoutState,
)
from buzz.audio_capture.source import (
    AudioErrorCallback,
    AudioFrameCallback,
    AudioSource,
    AudioSourceError,
)
from buzz.meeting.meeting_recorder import (
    MeetingRecorder,
    MeetingRecorderState,
    MeetingRecordingResult,
)
from buzz.model_loader import ModelType, TranscriptionModel
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import Task, TranscriptionOptions


class FakeRecorder:
    def __init__(self, sample_rate: int = 1_000, events: Optional[list[str]] = None):
        self.sample_rate = sample_rate
        self.events = events if events is not None else []
        self.blocks: list[np.ndarray] = []
        self.started = False
        self.stopped = False
        self.cancelled = False
        self.stop_count = 0
        self.request_stop_count = 0
        self.fail_enqueue = False
        self.pause_start = False
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()
        self._result: Optional[MeetingRecordingResult] = None
        self._lock = threading.Lock()

    @property
    def accepted_sample_count(self) -> int:
        with self._lock:
            return sum(block.size for block in self.blocks)

    def start(self) -> None:
        self.events.append("recorder.start")
        self.start_entered.set()
        if self.pause_start:
            assert self.allow_start.wait(timeout=5)
        self.started = True

    def enqueue(self, samples: np.ndarray) -> bool:
        self.events.append("recorder.enqueue")
        if self.fail_enqueue:
            return False
        with self._lock:
            self.blocks.append(samples.copy())
        return True

    def request_stop(self) -> None:
        self.request_stop_count += 1

    def stop(self) -> MeetingRecordingResult:
        with self._lock:
            if self._result is None:
                self.stop_count += 1
                sample_count = sum(block.size for block in self.blocks)
                self._result = MeetingRecordingResult(
                    output_path=Path("meeting.wav"),
                    sample_rate=self.sample_rate,
                    sample_count=sample_count,
                    duration_seconds=sample_count / self.sample_rate,
                    state=MeetingRecorderState.STOPPED,
                    error=None,
                    published=True,
                )
            self.stopped = True
            return self._result

    def cancel_empty_start(self) -> MeetingRecordingResult:
        assert self.accepted_sample_count == 0
        self.cancelled = True
        return self.stop()


class ControlledAudioSource(AudioSource):
    def __init__(self, sample_rate: int = 1_000, events: Optional[list[str]] = None):
        self._sample_rate = sample_rate
        self.events = events if events is not None else []
        self.on_audio: Optional[AudioFrameCallback] = None
        self.on_error: Optional[AudioErrorCallback] = None
        self.active = False
        self.start_count = 0
        self.stop_count = 0
        self.start_action = None
        self.start_error: Optional[Exception] = None
        self.stop_errors: list[Exception] = []
        self.stop_attempt_count = 0
        self.pause_start = False
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()
        self.callback_obtained = threading.Event()
        self.allow_callback = threading.Event()
        self.pause_before_callback = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(
        self,
        on_audio: AudioFrameCallback,
        on_error: Optional[AudioErrorCallback] = None,
    ) -> None:
        self.events.append("source.start")
        with self._lock:
            self.on_audio = on_audio
            self.on_error = on_error
            self.active = True
            self.start_count += 1
        self.start_entered.set()
        if self.start_action is not None:
            self.start_action(on_audio)
        if self.pause_start:
            assert self.allow_start.wait(timeout=5)
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.events.append("source.stop")
        with self._lock:
            self.stop_attempt_count += 1
            if self.stop_errors:
                raise self.stop_errors.pop(0)
            if not self.active:
                return
            self.active = False
            self.stop_count += 1

    def deliver(self, samples: np.ndarray) -> None:
        with self._lock:
            if not self.active or self.on_audio is None:
                raise RuntimeError("source is not active")
            callback = self.on_audio
        self.callback_obtained.set()
        if self.pause_before_callback:
            assert self.allow_callback.wait(timeout=2)
        callback(samples)

    def fail(self, error: Exception) -> None:
        with self._lock:
            callback = self.on_error
        assert callback is not None
        callback(error)


class BlockingEnqueueRecorder(FakeRecorder):
    def __init__(self, sample_rate: int = 1_000):
        super().__init__(sample_rate)
        self.enqueue_entered = threading.Event()
        self.allow_enqueue = threading.Event()

    def enqueue(self, samples: np.ndarray) -> bool:
        self.enqueue_entered.set()
        assert self.allow_enqueue.wait(timeout=5)
        return super().enqueue(samples)


class FailingArchiveWriter:
    def __init__(self, *, block_failure_cleanup: bool = False) -> None:
        self.block_failure_cleanup = block_failure_cleanup
        self.failure_cleanup_entered = threading.Event()
        self.release_failure_cleanup = threading.Event()

    def write(self, pcm16: np.ndarray) -> None:
        raise OSError("writer failed")

    def flush(self) -> None:
        pass

    def finalize(self) -> None:
        pass

    def publish(self) -> None:
        pass

    def discard(self) -> None:
        pass

    def close_after_failure(self) -> None:
        self.failure_cleanup_entered.set()
        if self.block_failure_cleanup:
            assert self.release_failure_cleanup.wait(timeout=5)


def _start_fanout(
    *, sample_rate: int = 1_000, events: Optional[list[str]] = None
) -> tuple[MeetingAudioFanout, ControlledAudioSource, FakeRecorder]:
    source = ControlledAudioSource(sample_rate, events)
    recorder = FakeRecorder(sample_rate, events)
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]
    fanout.start()
    return fanout, source, recorder


def test_recorder_is_ready_before_source_start_and_rate_must_match():
    events: list[str] = []
    fanout, _, _ = _start_fanout(events=events)
    assert events[:2] == ["recorder.start", "source.start"]
    fanout.stop()

    with pytest.raises(ValueError, match="sample rate"):
        MeetingAudioFanout(ControlledAudioSource(48_000), FakeRecorder(16_000))  # type: ignore[arg-type]


def test_fanout_stop_before_start_consumes_fanout_and_prevents_later_source_start():
    source = ControlledAudioSource()
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]

    result = fanout.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert fanout.state == MeetingAudioFanoutState.STOPPED
    with pytest.raises(AudioSourceError, match="Cannot start"):
        fanout.start()
    assert source.start_count == 0


def test_fanout_can_start_before_live_and_late_subscription_gets_only_later_pcm():
    fanout, source, recorder = _start_fanout()
    early = np.array([0.1, 0.2], dtype=np.float32)
    late = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    live: list[np.ndarray] = []

    source.deliver(early)
    fanout.live_source.start(lambda samples: live.append(samples.copy()))
    source.deliver(late)
    result = fanout.stop()

    assert result.sample_count == 5
    np.testing.assert_array_equal(
        np.concatenate(recorder.blocks), np.concatenate((early, late))
    )
    assert len(live) == 1
    np.testing.assert_array_equal(live[0], late)


def test_archive_enqueue_happens_before_live_callback():
    events: list[str] = []
    fanout, source, _ = _start_fanout(events=events)
    fanout.live_source.start(lambda _: events.append("live.callback"))

    source.deliver(np.ones(2, dtype=np.float32))

    assert events.index("recorder.enqueue") < events.index("live.callback")
    fanout.stop()


def test_live_stop_only_unsubscribes_and_is_idempotent():
    fanout, source, recorder = _start_fanout()
    live: list[np.ndarray] = []
    fanout.live_source.start(lambda samples: live.append(samples.copy()))

    fanout.live_source.stop()
    fanout.live_source.stop()
    source.deliver(np.ones(3, dtype=np.float32))

    assert source.stop_count == 0
    assert recorder.stop_count == 0
    assert live == []
    assert recorder.accepted_sample_count == 3
    fanout.stop()


def test_live_callback_failure_archives_current_block_and_capture_continues():
    live_errors: list[Exception] = []
    external_errors: list[Exception] = []
    source = ControlledAudioSource()
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(  # type: ignore[arg-type]
        source,
        recorder,
        on_live_error=external_errors.append,
    )
    fanout.start()

    def fail_live(_: np.ndarray) -> None:
        raise RuntimeError("backend rejected PCM")

    fanout.live_source.start(fail_live, live_errors.append)
    source.deliver(np.ones(2, dtype=np.float32))
    source.deliver(np.ones(3, dtype=np.float32))

    assert recorder.accepted_sample_count == 5
    assert source.stop_count == 0
    assert recorder.stop_count == 0
    assert len(live_errors) == len(external_errors) == 1
    assert "backend rejected PCM" in str(live_errors[0])
    fanout.stop()


def test_recorder_failure_does_not_stop_or_interrupt_live_delivery():
    fanout, source, recorder = _start_fanout()
    recorder.fail_enqueue = True
    live: list[np.ndarray] = []
    fanout.live_source.start(lambda samples: live.append(samples.copy()))

    source.deliver(np.ones(4, dtype=np.float32))

    assert len(live) == 1
    assert source.stop_count == 0
    fanout.stop()


def test_source_runtime_failure_requests_nonblocking_recorder_stop_and_publishes_prefix():
    live_errors: list[Exception] = []
    source_errors: list[Exception] = []
    source = ControlledAudioSource()
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(  # type: ignore[arg-type]
        source,
        recorder,
        on_source_error=source_errors.append,
    )
    fanout.start()
    fanout.live_source.start(lambda _: None, live_errors.append)
    source.deliver(np.ones(7, dtype=np.float32))

    error = AudioSourceError("capture disconnected")
    source.fail(error)

    assert fanout.state == MeetingAudioFanoutState.FAILED
    assert recorder.request_stop_count == 1
    assert recorder.stop_count == 0
    assert live_errors == [error]
    assert source_errors == [error]
    result = fanout.stop()
    assert result.sample_count == 7
    assert result.published


def test_source_start_failure_cancels_empty_partial():
    source = ControlledAudioSource()
    source.start_error = AudioSourceError("could not start")
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]

    with pytest.raises(AudioSourceError, match="could not start"):
        fanout.start()

    assert recorder.cancelled
    assert recorder.accepted_sample_count == 0
    assert source.stop_count == 1


def test_source_start_failure_after_delivery_preserves_and_publishes_prefix():
    source = ControlledAudioSource()
    prefix = np.arange(5, dtype=np.float32)
    source.start_action = lambda callback: callback(prefix)
    source.start_error = AudioSourceError("late startup failure")
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]

    with pytest.raises(AudioSourceError, match="late startup failure"):
        fanout.start()

    assert not recorder.cancelled
    assert recorder.stopped
    assert recorder.accepted_sample_count == prefix.size
    assert recorder._result is not None and recorder._result.published


def test_source_start_synchronous_callback_is_archived_before_start_returns():
    source = ControlledAudioSource()
    synchronous = np.arange(8, dtype=np.float32)
    source.start_action = lambda callback: callback(synchronous)
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]

    fanout.start()

    assert fanout.state == MeetingAudioFanoutState.RUNNING
    assert recorder.accepted_sample_count == synchronous.size
    np.testing.assert_array_equal(recorder.blocks[0], synchronous)
    fanout.stop()


def test_stop_during_recorder_start_prevents_future_source_start():
    source = ControlledAudioSource()
    recorder = FakeRecorder()
    recorder.pause_start = True
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]
    start_errors: list[Exception] = []
    start_thread = threading.Thread(
        target=lambda: _capture_exception(fanout.start, start_errors)
    )
    start_thread.start()
    assert recorder.start_entered.wait(timeout=5)

    stop_results = []
    stop_thread = threading.Thread(target=lambda: stop_results.append(fanout.stop()))
    stop_thread.start()
    with fanout._condition:
        assert fanout._condition.wait_for(lambda: fanout._stop_requested, timeout=5)
    recorder.allow_start.set()
    start_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert len(start_errors) == 1
    assert "stopped during startup" in str(start_errors[0])
    assert len(stop_results) == 1
    assert source.start_count == 0
    assert fanout.state == MeetingAudioFanoutState.STOPPED


def test_stop_waits_for_committed_source_start_then_stops_it():
    source = ControlledAudioSource()
    source.pause_start = True
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]
    start_errors: list[Exception] = []
    start_thread = threading.Thread(
        target=lambda: _capture_exception(fanout.start, start_errors)
    )
    start_thread.start()
    assert source.start_entered.wait(timeout=5)

    stop_results = []
    stop_thread = threading.Thread(target=lambda: stop_results.append(fanout.stop()))
    stop_thread.start()
    with fanout._condition:
        assert fanout._condition.wait_for(lambda: fanout._stop_requested, timeout=5)
    assert stop_results == []
    source.allow_start.set()
    start_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert len(start_errors) == 1
    assert len(stop_results) == 1
    assert source.start_count == 1
    assert source.stop_attempt_count == 1
    assert source.stop_count == 1
    assert not source.active


def _capture_exception(call, errors: list[Exception]) -> None:
    try:
        call()
    except Exception as exc:
        errors.append(exc)


def test_real_recorder_source_error_publishes_exact_accepted_prefix(tmp_path):
    output_path = tmp_path / "meeting.wav"
    source = ControlledAudioSource()
    recorder = MeetingRecorder(output_path, source.sample_rate)
    fanout = MeetingAudioFanout(source, recorder)
    prefix = np.linspace(-1, 1, 17, dtype=np.float32)
    fanout.start()
    source.deliver(prefix)

    source.fail(AudioSourceError("capture disconnected"))
    result = fanout.stop()

    assert result.state == MeetingRecorderState.STOPPED
    assert result.sample_count == prefix.size
    assert result.published
    with soundfile.SoundFile(output_path) as audio_file:
        assert audio_file.frames == prefix.size
        assert audio_file.format == "RF64"


def test_real_recorder_source_start_failure_cleans_empty_partial(tmp_path):
    output_path = tmp_path / "meeting.wav"
    source = ControlledAudioSource()
    source.start_error = AudioSourceError("could not start")
    recorder = MeetingRecorder(output_path, source.sample_rate)
    fanout = MeetingAudioFanout(source, recorder)

    with pytest.raises(AudioSourceError, match="could not start"):
        fanout.start()

    assert not output_path.exists()
    assert not output_path.with_name("meeting.wav.partial").exists()


def test_source_stop_failure_is_retried_until_cleanup_succeeds():
    fanout, source, recorder = _start_fanout()
    source.stop_errors.append(AudioSourceError("first stop failed"))

    with pytest.raises(AudioSourceError, match="first stop failed"):
        fanout.stop()

    assert source.active
    assert source.stop_attempt_count == 1
    assert fanout.state == MeetingAudioFanoutState.FAILED
    assert not fanout._stop_in_progress
    result = fanout.stop()

    assert result is recorder._result
    assert source.stop_attempt_count == 2
    assert source.stop_count == 1
    assert not source.active
    assert fanout.state == MeetingAudioFanoutState.STOPPED


def test_repeated_source_stop_failure_never_wedges_or_reports_clean_stop():
    fanout, source, _ = _start_fanout()
    source.stop_errors.extend(
        [
            AudioSourceError("stop failed one"),
            AudioSourceError("stop failed two"),
        ]
    )

    with pytest.raises(AudioSourceError, match="stop failed one"):
        fanout.stop()
    assert not fanout._stop_in_progress
    with pytest.raises(AudioSourceError, match="stop failed two"):
        fanout.stop()

    assert source.stop_attempt_count == 2
    assert source.active
    assert fanout.state == MeetingAudioFanoutState.FAILED
    assert not fanout._stop_in_progress


def test_writer_error_callback_can_reenter_fanout_stop_without_wedging(tmp_path):
    source = ControlledAudioSource()
    callback_returned = threading.Event()
    callback_results = []
    callback_errors: list[Exception] = []
    fanout_ref = {}

    def on_recording_error(_):
        try:
            callback_results.append(fanout_ref["fanout"].stop())
        except Exception as exc:
            callback_errors.append(exc)
        finally:
            callback_returned.set()

    recorder = MeetingRecorder(
        tmp_path / "meeting.wav",
        source.sample_rate,
        on_error=on_recording_error,
        _writer_factory=lambda *_: FailingArchiveWriter(),
    )
    fanout = MeetingAudioFanout(source, recorder)
    fanout_ref["fanout"] = fanout
    fanout.start()
    source.deliver(np.ones(4, dtype=np.float32))

    assert callback_returned.wait(timeout=5)
    assert recorder._writer_thread is not None
    recorder._writer_thread.join(timeout=5)
    second = fanout.stop()

    assert callback_errors == []
    assert callback_results == [second]
    assert not recorder._writer_thread.is_alive()
    assert not fanout._stop_in_progress
    assert fanout.state == MeetingAudioFanoutState.FAILED


def test_external_fanout_stop_waits_after_reentrant_cached_result(tmp_path):
    source = ControlledAudioSource()
    writer = FailingArchiveWriter(block_failure_cleanup=True)
    callback_returned = threading.Event()
    external_wait_entered = threading.Event()
    external_returned = threading.Event()
    callback_results = []
    external_results = []
    callback_errors: list[Exception] = []
    fanout_ref = {}

    def on_recording_error(_):
        try:
            callback_results.append(fanout_ref["fanout"].stop())
        except Exception as exc:
            callback_errors.append(exc)
        finally:
            callback_returned.set()

    recorder = MeetingRecorder(
        tmp_path / "meeting.wav",
        source.sample_rate,
        on_error=on_recording_error,
        _writer_factory=lambda *_: writer,
    )
    fanout = MeetingAudioFanout(source, recorder)
    fanout_ref["fanout"] = fanout
    fanout.start()
    source.deliver(np.ones(4, dtype=np.float32))

    assert callback_returned.wait(timeout=5)
    assert writer.failure_cleanup_entered.wait(timeout=5)
    original_wait_for_cleanup = recorder._wait_for_writer_cleanup

    def tracked_wait_for_cleanup(writer_thread) -> None:
        if threading.current_thread() is not writer_thread:
            external_wait_entered.set()
        original_wait_for_cleanup(writer_thread)

    recorder._wait_for_writer_cleanup = tracked_wait_for_cleanup

    def stop_externally() -> None:
        external_results.append(fanout.stop())
        external_returned.set()

    external_stop = threading.Thread(target=stop_externally)
    external_stop.start()
    assert external_wait_entered.wait(timeout=5)
    assert not external_returned.is_set()

    writer.release_failure_cleanup.set()
    external_stop.join(timeout=5)

    assert callback_errors == []
    assert callback_results == external_results
    assert not external_stop.is_alive()
    assert recorder._writer_thread is not None
    assert not recorder._writer_thread.is_alive()
    assert source.stop_count == 1
    assert not source.active
    assert fanout._source_cleanup_complete
    assert not fanout._stop_in_progress


def test_source_error_during_archive_enqueue_preserves_entered_pcm_prefix():
    source = ControlledAudioSource()
    recorder = BlockingEnqueueRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]
    fanout.start()
    block = np.arange(6, dtype=np.float32)
    delivery = threading.Thread(target=source.deliver, args=(block,))
    delivery.start()
    assert recorder.enqueue_entered.wait(timeout=5)

    source.fail(AudioSourceError("capture failed during callback"))
    source.deliver(np.ones(2, dtype=np.float32))
    recorder.allow_enqueue.set()
    delivery.join(timeout=5)
    result = fanout.stop()

    assert result.sample_count == block.size
    assert recorder.accepted_sample_count == block.size
    assert recorder.request_stop_count == 1
    assert len(recorder.blocks) == 1


def test_stop_closes_gate_before_late_source_callback_enters_controller():
    fanout, source, recorder = _start_fanout()
    source.pause_before_callback = True
    delivery = threading.Thread(
        target=source.deliver,
        args=(np.ones(4, dtype=np.float32),),
    )
    delivery.start()
    assert source.callback_obtained.wait(timeout=2)

    stopped = threading.Event()
    stop_thread = threading.Thread(target=lambda: (fanout.stop(), stopped.set()))
    stop_thread.start()
    assert stopped.wait(timeout=2)
    source.allow_callback.set()
    delivery.join(timeout=2)
    stop_thread.join(timeout=2)

    assert stopped.is_set()
    assert recorder.accepted_sample_count == 0


def test_fanout_stop_waits_for_controller_callback_already_in_progress():
    fanout, source, recorder = _start_fanout()
    live_entered = threading.Event()
    allow_live = threading.Event()

    def live_callback(_: np.ndarray) -> None:
        live_entered.set()
        assert allow_live.wait(timeout=2)

    fanout.live_source.start(live_callback)
    delivery = threading.Thread(
        target=source.deliver,
        args=(np.ones(4, dtype=np.float32),),
    )
    delivery.start()
    assert live_entered.wait(timeout=2)

    stopped = threading.Event()
    stop_thread = threading.Thread(target=lambda: (fanout.stop(), stopped.set()))
    stop_thread.start()
    assert not stopped.wait(timeout=0.05)
    assert recorder.stop_count == 0
    allow_live.set()
    delivery.join(timeout=2)
    stop_thread.join(timeout=2)

    assert stopped.is_set()
    assert recorder.accepted_sample_count == 4
    assert recorder.stop_count == 1


def test_live_stop_waits_for_inflight_callback_without_stopping_capture():
    fanout, source, recorder = _start_fanout()
    live_entered = threading.Event()
    allow_live = threading.Event()

    def live_callback(_: np.ndarray) -> None:
        live_entered.set()
        assert allow_live.wait(timeout=2)

    fanout.live_source.start(live_callback)
    delivery = threading.Thread(
        target=source.deliver,
        args=(np.ones(3, dtype=np.float32),),
    )
    delivery.start()
    assert live_entered.wait(timeout=2)

    unsubscribed = threading.Event()
    live_stop = threading.Thread(
        target=lambda: (fanout.live_source.stop(), unsubscribed.set())
    )
    live_stop.start()
    assert not unsubscribed.wait(timeout=0.05)
    assert source.stop_count == recorder.stop_count == 0
    allow_live.set()
    delivery.join(timeout=2)
    live_stop.join(timeout=2)

    assert unsubscribed.is_set()
    assert source.stop_count == recorder.stop_count == 0
    fanout.stop()


def test_concurrent_duplicate_fanout_stop_returns_same_result():
    fanout, _, recorder = _start_fanout()
    gate = threading.Barrier(3)
    results: list[MeetingRecordingResult] = []

    def stop() -> None:
        gate.wait()
        results.append(fanout.stop())

    threads = [threading.Thread(target=stop) for _ in range(2)]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 2
    assert results[0] is results[1]
    assert recorder.stop_count == 1


def test_live_subscription_rejects_invalid_lifecycle_and_duplicate_start():
    source = ControlledAudioSource()
    recorder = FakeRecorder()
    fanout = MeetingAudioFanout(source, recorder)  # type: ignore[arg-type]

    with pytest.raises(AudioSourceError, match="not active"):
        fanout.live_source.start(lambda _: None)
    fanout.start()
    fanout.live_source.start(lambda _: None)
    with pytest.raises(AudioSourceError, match="already active"):
        fanout.live_source.start(lambda _: None)
    fanout.stop()
    fanout.live_source.stop()
    with pytest.raises(AudioSourceError, match="not active"):
        fanout.live_source.start(lambda _: None)


def test_recording_transcriber_backend_failure_stops_only_live_subscription():
    fanout, source, recorder = _start_fanout()
    options = TranscriptionOptions(
        model=TranscriptionModel(model_type=ModelType.WHISPER),
        language="en",
        task=Task.TRANSCRIBE,
    )
    transcriber = RecordingTranscriber(
        transcription_options=options,
        input_device_index=None,
        sample_rate=source.sample_rate,
        model_path="/fake/path",
        sounddevice=MagicMock(),
        audio_source=fanout.live_source,
    )
    backend_called = threading.Event()

    def fail_backend(samples, model, prompt):
        backend_called.set()
        raise RuntimeError("backend failed")

    with patch.object(transcriber, "_load_model", return_value=object()), patch.object(
        transcriber, "_transcribe", side_effect=fail_backend
    ), patch.object(transcriber, "_release_model"):
        worker = threading.Thread(target=transcriber.start)
        worker.start()
        with fanout._condition:
            assert fanout._condition.wait_for(lambda: fanout._live_active, timeout=2)
        source.deliver(np.ones(12 * source.sample_rate, dtype=np.float32))
        assert backend_called.wait(timeout=2)
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert source.stop_count == 0
    assert recorder.stop_count == 0
    assert recorder.accepted_sample_count == 12 * source.sample_rate
    post_backend_block = np.ones(37, dtype=np.float32)
    source.deliver(post_backend_block)
    assert recorder.accepted_sample_count == 12 * source.sample_rate + 37
    assert source.active
    assert recorder.stop_count == 0
    fanout.stop()
    assert source.stop_count == recorder.stop_count == 1
