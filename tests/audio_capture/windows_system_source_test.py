import struct
import subprocess
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.audio_capture.source import AudioSourceError
from buzz.audio_capture.windows_system_source import WindowsSystemAudioSource
from tests.audio_capture.windows_helper_test_support import HEADER, FakeProcess


def make_source(tmp_path, process=None, **kwargs):
    helper_path = tmp_path / "buzz-windows-audio-capture.exe"
    helper_path.touch()
    process = process or FakeProcess()
    process_factory = MagicMock(return_value=process)
    source = WindowsSystemAudioSource(
        helper_path=helper_path,
        process_factory=process_factory,
        platform_name="win32",
        handshake_timeout=0.1,
        graceful_stop_timeout=0.01,
        terminate_timeout=0.01,
        kill_timeout=0.01,
        **kwargs,
    )
    return source, process, process_factory


def start_with_header(source, process, on_audio=lambda samples: None, on_error=None):
    process.stdout.feed(HEADER)
    source.start(on_audio, on_error)


def assert_source_clean(source):
    assert source._active is False
    assert source._starting is False
    assert source._stopping is False
    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None
    assert source._on_audio is None
    assert source._on_error is None


def test_constructor_has_fixed_sample_rate_and_does_not_launch(tmp_path):
    source, _, process_factory = make_source(tmp_path)

    assert source.sample_rate == 16_000
    process_factory.assert_not_called()


def test_default_helper_path_uses_application_base_directory(tmp_path):
    with patch(
        "buzz.audio_capture.windows_system_source.APP_BASE_DIR", str(tmp_path)
    ):
        source = WindowsSystemAudioSource(platform_name="win32")

    assert source.helper_path == (
        tmp_path / "native" / "windows" / "buzz-windows-audio-capture.exe"
    )


def test_non_windows_start_is_rejected_before_launch(tmp_path):
    source, _, process_factory = make_source(tmp_path)
    source._platform_name = "linux"

    with pytest.raises(AudioSourceError, match="only available on Windows"):
        source.start(lambda samples: None)

    process_factory.assert_not_called()


def test_missing_helper_is_reported(tmp_path):
    source = WindowsSystemAudioSource(
        helper_path=tmp_path / "missing.exe",
        platform_name="win32",
    )

    with pytest.raises(AudioSourceError, match="helper is missing"):
        source.start(lambda samples: None)


def test_process_launch_failure_cleans_source_state(tmp_path):
    source, _, process_factory = make_source(tmp_path)
    process_factory.side_effect = OSError("CreateProcess failed")

    with pytest.raises(AudioSourceError, match="CreateProcess failed"):
        source.start(lambda samples: None)

    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None


def test_worker_thread_creation_failure_cleans_process_and_source_state(tmp_path):
    source, process, _ = make_source(tmp_path)

    with patch(
        "buzz.audio_capture.windows_system_source.threading.Thread",
        side_effect=RuntimeError("thread creation failed"),
    ), pytest.raises(AudioSourceError, match="thread creation failed"):
        source.start(lambda samples: None)

    assert process.returncode == 0
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert_source_clean(source)


def test_start_uses_hidden_system_mode_command(tmp_path):
    source, process, process_factory = make_source(tmp_path)
    start_with_header(source, process)

    args, kwargs = process_factory.call_args
    assert args[0] == [str(source.helper_path), "--mode", "system"]
    assert kwargs["stdin"] == subprocess.PIPE
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["bufsize"] == 0
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)

    source.stop()


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (struct.pack("<4sHHIHH", b"NOPE", 1, 16, 16_000, 1, 1), "invalid handshake"),
        (struct.pack("<4sHHIHH", b"BZWA", 2, 16, 16_000, 1, 1), "protocol version"),
        (struct.pack("<4sHHIHH", b"BZWA", 1, 20, 16_000, 1, 1), "protocol version"),
        (struct.pack("<4sHHIHH", b"BZWA", 1, 16, 48_000, 1, 1), "16 kHz"),
        (struct.pack("<4sHHIHH", b"BZWA", 1, 16, 16_000, 2, 1), "mono"),
        (struct.pack("<4sHHIHH", b"BZWA", 1, 16, 16_000, 1, 2), "float32"),
    ],
)
def test_invalid_handshake_fails_start_and_cleans_up(tmp_path, header, message):
    source, process, _ = make_source(tmp_path)
    process.stdout.feed(header)

    with pytest.raises(AudioSourceError, match=message):
        source.start(lambda samples: None)

    assert process.returncode == 0
    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None


def test_partial_handshake_is_reassembled(tmp_path):
    source, process, _ = make_source(tmp_path)
    for byte in HEADER:
        process.stdout.feed(bytes([byte]))

    source.start(lambda samples: None)
    source.stop()


def test_handshake_timeout_terminates_helper_and_cleans_up(tmp_path):
    process = FakeProcess(graceful_exit=False)
    source, process, _ = make_source(tmp_path, process)

    with pytest.raises(AudioSourceError, match="Timed out"):
        source.start(lambda samples: None)

    assert process.terminate_count == 1
    assert source._process is None


def test_helper_exit_before_handshake_is_start_failure_only(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    process.stderr.feed(b"native startup details")
    process.exit(12)

    with pytest.raises(AudioSourceError, match="16 kHz mono"):
        source.start(lambda samples: None, errors.append)

    assert errors == []
    assert source._process is None


def test_pcm_partial_reads_and_float_carry_preserve_sample_order(tmp_path):
    source, process, _ = make_source(tmp_path)
    received = []
    delivered = threading.Event()

    def on_audio(samples):
        received.append(np.array(samples, copy=True))
        if sum(chunk.size for chunk in received) == 4:
            delivered.set()

    start_with_header(source, process, on_audio)
    pcm = np.array([0.1, -0.2, 0.3, -0.4], dtype="<f4").tobytes()
    process.stdout.feed(pcm[:1])
    process.stdout.feed(pcm[1:4])
    process.stdout.feed(pcm[4:9])
    process.stdout.feed(pcm[9:])

    assert delivered.wait(timeout=1)
    combined = np.concatenate(received)
    assert combined.dtype == np.float32
    assert combined.ndim == 1
    np.testing.assert_allclose(combined, [0.1, -0.2, 0.3, -0.4])
    source.stop()


def test_duplicate_start_is_rejected(tmp_path):
    source, process, process_factory = make_source(tmp_path)
    start_with_header(source, process)

    with pytest.raises(AudioSourceError, match="already active"):
        source.start(lambda samples: None)

    assert process_factory.call_count == 1
    source.stop()


def test_concurrent_stop_between_handshake_and_activation_cancels_start(tmp_path):
    source, process, _ = make_source(tmp_path)
    activation_entered = threading.Event()
    release_activation = threading.Event()
    original_activate = source._activate_started_session
    start_errors = []

    def blocked_activate(*args):
        activation_entered.set()
        assert release_activation.wait(timeout=1)
        return original_activate(*args)

    source._activate_started_session = blocked_activate
    process.stdout.feed(HEADER)

    def run_start():
        try:
            source.start(lambda samples: None, lambda error: None)
        except Exception as exc:
            start_errors.append(exc)

    start_thread = threading.Thread(target=run_start)
    start_thread.start()
    assert activation_entered.wait(timeout=1)

    source.stop()
    assert_source_clean(source)
    release_activation.set()
    start_thread.join(timeout=1)

    assert not start_thread.is_alive()
    assert len(start_errors) == 1
    assert isinstance(start_errors[0], AudioSourceError)
    assert "canceled" in str(start_errors[0]).lower()
    assert_source_clean(source)


def test_concurrent_stop_before_process_assignment_cancels_start(tmp_path):
    source, process, _ = make_source(tmp_path)
    factory_entered = threading.Event()
    release_factory = threading.Event()
    start_errors = []

    def blocked_factory(*args, **kwargs):
        factory_entered.set()
        assert release_factory.wait(timeout=1)
        return process

    source._process_factory = blocked_factory

    def run_start():
        try:
            source.start(lambda samples: None)
        except Exception as exc:
            start_errors.append(exc)

    start_thread = threading.Thread(target=run_start)
    start_thread.start()
    assert factory_entered.wait(timeout=1)

    source.stop()
    assert_source_clean(source)
    release_factory.set()
    start_thread.join(timeout=1)

    assert not start_thread.is_alive()
    assert len(start_errors) == 1
    assert isinstance(start_errors[0], AudioSourceError)
    assert process.returncode == 0
    assert_source_clean(source)


def test_stop_before_start_and_duplicate_stop_are_safe(tmp_path):
    source, process, _ = make_source(tmp_path)

    source.stop()
    start_with_header(source, process)
    source.stop()
    source.stop()

    assert process.stdin.writes == [b"\x00"]
    assert source._process is None


def test_stop_uses_terminate_after_graceful_timeout(tmp_path):
    process = FakeProcess(graceful_exit=False, terminate_exit=True)
    source, process, _ = make_source(tmp_path, process)
    start_with_header(source, process)

    source.stop()

    assert process.terminate_count == 1
    assert process.kill_count == 0


def test_stop_uses_kill_after_terminate_timeout(tmp_path):
    process = FakeProcess(graceful_exit=False, terminate_exit=False)
    source, process, _ = make_source(tmp_path, process)
    start_with_header(source, process)

    source.stop()

    assert process.terminate_count == 1
    assert process.kill_count == 1


def test_final_kill_timeout_still_performs_all_local_cleanup(tmp_path):
    process = FakeProcess(
        graceful_exit=False,
        terminate_exit=False,
        kill_exit=False,
    )
    source, process, _ = make_source(tmp_path, process)
    start_with_header(source, process)

    with pytest.raises(AudioSourceError, match="did not exit after it was killed"):
        source.stop()

    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert process.terminate_count == 1
    assert process.kill_count == 1
    assert_source_clean(source)

    source.stop()
    assert process.terminate_count == 1
    assert process.kill_count == 1


@pytest.mark.parametrize(
    ("process", "message"),
    [
        (
            FakeProcess(
                graceful_exit=False,
                terminate_error=RuntimeError("terminate exploded"),
            ),
            "terminate exploded",
        ),
        (
            FakeProcess(
                graceful_exit=False,
                terminate_exit=False,
                kill_error=RuntimeError("kill exploded"),
            ),
            "kill exploded",
        ),
    ],
)
def test_process_shutdown_exception_still_performs_local_cleanup(
    tmp_path,
    process,
    message,
):
    source, process, _ = make_source(tmp_path, process)
    start_with_header(source, process)

    with pytest.raises(AudioSourceError, match=message):
        source.stop()

    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
    assert_source_clean(source)


@pytest.mark.parametrize(
    ("returncode", "message"),
    [
        (0, "ended unexpectedly"),
        (14, "capture failed"),
        (15, "became unavailable"),
        (16, "Audio service"),
        (17, "transport was interrupted"),
    ],
)
def test_runtime_eof_reports_friendly_error_once(tmp_path, returncode, message):
    source, process, _ = make_source(tmp_path)
    errors = []
    error_event = threading.Event()

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_error=on_error)
    process.exit(returncode)

    assert error_event.wait(timeout=1)
    assert len(errors) == 1
    assert message.lower() in errors[0].lower()
    source.stop()
    assert len(errors) == 1


def test_incomplete_final_float_is_transport_error(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    error_event = threading.Event()

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_error=on_error)
    process.stdout.feed(b"\x00\x01\x02")
    process.exit(0)

    assert error_event.wait(timeout=1)
    assert errors == ["Windows system audio transport ended with an incomplete sample."]
    source.stop()


def test_stderr_tail_is_bounded(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    error_event = threading.Event()

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_error=on_error)
    process.stderr.feed(b"x" * (source._STDERR_LIMIT + 10_000))
    process.stderr.feed(b"the final diagnostic")
    assert process.stderr.wait_until_empty()
    process.exit(14)

    assert error_event.wait(timeout=1)
    assert len(source._stderr_tail) <= source._STDERR_LIMIT
    assert bytes(source._stderr_tail).endswith(b"the final diagnostic")
    source.stop()


def test_callback_exception_becomes_one_runtime_error(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    error_event = threading.Event()

    def on_audio(samples):
        raise RuntimeError("consumer exploded")

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_audio, on_error)
    process.stdout.feed(np.array([0.1], dtype="<f4").tobytes())

    assert error_event.wait(timeout=1)
    assert errors == ["System audio callback failed: consumer exploded"]
    source.stop()
    assert len(errors) == 1


def test_callback_failure_and_runtime_eof_report_error_only_once(tmp_path):
    source, process, _ = make_source(tmp_path)
    callback_started = threading.Event()
    release_callback = threading.Event()
    error_event = threading.Event()
    errors = []

    def on_audio(samples):
        callback_started.set()
        assert release_callback.wait(timeout=1)
        raise RuntimeError("callback failed during EOF")

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_audio, on_error)
    process.stdout.feed(np.array([0.1], dtype="<f4").tobytes())
    assert callback_started.wait(timeout=1)
    process.exit(14)
    release_callback.set()

    assert error_event.wait(timeout=1)
    source.stop()
    assert len(errors) == 1


def test_normal_stop_eof_does_not_report_runtime_error(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    start_with_header(source, process, on_error=errors.append)

    source.stop()

    assert errors == []
    assert_source_clean(source)


def test_stop_waits_for_in_flight_callback_and_prevents_later_callbacks(tmp_path):
    source, process, _ = make_source(tmp_path)
    callback_started = threading.Event()
    release_callback = threading.Event()
    received = []

    def on_audio(samples):
        received.append(np.array(samples, copy=True))
        callback_started.set()
        assert release_callback.wait(timeout=1)

    start_with_header(source, process, on_audio)
    process.stdout.feed(np.array([0.25], dtype="<f4").tobytes())
    assert callback_started.wait(timeout=1)

    stop_thread = threading.Thread(target=source.stop)
    stop_thread.start()
    assert process.stdin.closed_event.wait(timeout=1)
    assert stop_thread.is_alive()
    process.stdout.feed(np.array([0.5], dtype="<f4").tobytes())
    release_callback.set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert len(received) == 1
    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None
    assert source._on_audio is None
    assert source._on_error is None
