import subprocess
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from buzz.audio_capture import WindowsProcessAudioSource
from buzz.audio_capture.source import AudioSourceError
from tests.audio_capture.windows_helper_test_support import HEADER, FakeProcess


def make_source(tmp_path, process=None, *, process_id=1234, **kwargs):
    helper_path = tmp_path / "buzz-windows-audio-capture.exe"
    helper_path.touch()
    process = process or FakeProcess()
    process_factory = MagicMock(return_value=process)
    source = WindowsProcessAudioSource(
        process_id,
        helper_path=helper_path,
        process_factory=process_factory,
        platform_name="win32",
        windows_build=20_348,
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


@pytest.mark.parametrize(
    "process_id",
    [True, False, 0, -1, 0x1_0000_0000, 1.5, "1234", None],
)
def test_constructor_rejects_invalid_process_id(process_id):
    with pytest.raises(ValueError, match="process_id"):
        WindowsProcessAudioSource(process_id)


@pytest.mark.parametrize("process_id", [1, 1234, 0xFFFFFFFF])
def test_constructor_accepts_dword_process_id_without_launch(tmp_path, process_id):
    source, _, process_factory = make_source(tmp_path, process_id=process_id)

    assert source.process_id == process_id
    assert source.sample_rate == 16_000
    process_factory.assert_not_called()


def test_default_helper_path_uses_application_base_directory(tmp_path):
    with patch(
        "buzz.audio_capture.windows_process_source.APP_BASE_DIR", str(tmp_path)
    ):
        source = WindowsProcessAudioSource(
            1234,
            platform_name="win32",
            windows_build=20_348,
        )

    assert source.helper_path == (
        tmp_path / "native" / "windows" / "buzz-windows-audio-capture.exe"
    )


def test_non_windows_start_is_rejected_before_launch(tmp_path):
    source, _, process_factory = make_source(tmp_path)
    source._platform_name = "linux"
    source._windows_build = None

    with pytest.raises(AudioSourceError, match="only available on Windows"):
        source.start(lambda samples: None)

    process_factory.assert_not_called()


@pytest.mark.parametrize("windows_build", [0, 19_045, 20_347])
def test_unsupported_windows_build_is_rejected_before_launch(
    tmp_path, windows_build
):
    source, _, process_factory = make_source(tmp_path)
    source._windows_build = windows_build

    with pytest.raises(AudioSourceError, match="build 20348 or later"):
        source.start(lambda samples: None)

    process_factory.assert_not_called()


@pytest.mark.parametrize("windows_build", [20_348, 20_349, 26_100])
def test_supported_windows_build_launches_helper(tmp_path, windows_build):
    source, process, process_factory = make_source(tmp_path)
    source._windows_build = windows_build
    start_with_header(source, process)

    assert process_factory.call_count == 1
    source.stop()


def test_missing_helper_is_reported_before_launch(tmp_path):
    process_factory = MagicMock()
    source = WindowsProcessAudioSource(
        1234,
        helper_path=tmp_path / "missing.exe",
        process_factory=process_factory,
        platform_name="win32",
        windows_build=20_348,
    )

    with pytest.raises(AudioSourceError, match="helper is missing"):
        source.start(lambda samples: None)

    process_factory.assert_not_called()


def test_process_launch_failure_cleans_source_state(tmp_path):
    source, _, process_factory = make_source(tmp_path)
    process_factory.side_effect = OSError("CreateProcess failed")

    with pytest.raises(AudioSourceError, match="CreateProcess failed"):
        source.start(lambda samples: None)

    assert_source_clean(source)


def test_start_uses_exact_hidden_process_mode_command(tmp_path):
    source, process, process_factory = make_source(tmp_path, process_id=0xFFFFFFFF)
    start_with_header(source, process)

    args, kwargs = process_factory.call_args
    assert args[0] == [
        str(source.helper_path),
        "--mode",
        "process",
        "--pid",
        "4294967295",
    ]
    assert kwargs["stdin"] == subprocess.PIPE
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["bufsize"] == 0
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    source.stop()


@pytest.mark.parametrize(
    ("returncode", "message"),
    [
        (19, "requires Windows 10 build 20348"),
        (20, "Unable to capture audio from the selected process"),
        (21, "Timed out while starting process audio capture"),
    ],
)
def test_process_startup_exit_codes_have_friendly_errors(
    tmp_path, returncode, message
):
    source, process, _ = make_source(tmp_path)
    process.stderr.feed(b"pid=1234 HRESULT=0x80004005")
    process.exit(returncode)

    with pytest.raises(AudioSourceError, match=message):
        source.start(lambda samples: None)

    assert_source_clean(source)


def test_partial_pcm_is_delivered_as_mono_float32(tmp_path):
    source, process, _ = make_source(tmp_path)
    received = []
    delivered = threading.Event()

    def on_audio(samples):
        received.append(np.array(samples, copy=True))
        if sum(chunk.size for chunk in received) == 3:
            delivered.set()

    start_with_header(source, process, on_audio)
    pcm = np.array([0.25, -0.5, 0.75], dtype="<f4").tobytes()
    process.stdout.feed(pcm[:3])
    process.stdout.feed(pcm[3:7])
    process.stdout.feed(pcm[7:])

    assert delivered.wait(timeout=1)
    combined = np.concatenate(received)
    assert combined.dtype == np.float32
    assert combined.shape == (3,)
    np.testing.assert_allclose(combined, [0.25, -0.5, 0.75])
    source.stop()


def test_runtime_error_is_reported_once(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    error_event = threading.Event()

    def on_error(error):
        errors.append(str(error))
        error_event.set()

    start_with_header(source, process, on_error=on_error)
    process.exit(14)

    assert error_event.wait(timeout=1)
    assert errors == ["Windows process audio capture failed."]
    source.stop()
    assert len(errors) == 1


def test_normal_stop_does_not_report_runtime_error(tmp_path):
    source, process, _ = make_source(tmp_path)
    errors = []
    start_with_header(source, process, on_error=errors.append)

    source.stop()

    assert errors == []
    assert_source_clean(source)


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
            source.start(lambda samples: None)
        except Exception as exc:
            start_errors.append(exc)

    start_thread = threading.Thread(target=run_start)
    start_thread.start()
    assert activation_entered.wait(timeout=1)

    source.stop()
    release_activation.set()
    start_thread.join(timeout=1)

    assert not start_thread.is_alive()
    assert len(start_errors) == 1
    assert "canceled" in str(start_errors[0]).lower()
    assert_source_clean(source)


def test_stop_waits_for_in_flight_callback(tmp_path):
    source, process, _ = make_source(tmp_path)
    callback_started = threading.Event()
    release_callback = threading.Event()

    def on_audio(samples):
        callback_started.set()
        assert release_callback.wait(timeout=1)

    start_with_header(source, process, on_audio)
    process.stdout.feed(np.array([0.25], dtype="<f4").tobytes())
    assert callback_started.wait(timeout=1)

    stop_thread = threading.Thread(target=source.stop)
    stop_thread.start()
    assert process.stdin.closed_event.wait(timeout=1)
    assert stop_thread.is_alive()
    release_callback.set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert_source_clean(source)


def test_kill_timeout_still_cleans_local_state(tmp_path):
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
    assert_source_clean(source)
