"""Windows audio integration tests: real child process, real OS pipes.

PR27 closes the integration gap between the Python Windows audio source
adapters and the real subprocess/pipe boundary, while still using
deterministic synthetic audio (no hardware, no ASR, no network).

Every scenario spawns a test-local ephemeral Python helper child through
``subprocess.Popen`` with real OS pipes.  The child independently
constructs the BZWA v1 wire protocol and emits deterministic float32
PCM.  No production protocol parser or serializer is used to generate
the expected bytes.
"""

from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from buzz.audio_capture.meeting_audio_fanout import MeetingAudioFanout
from buzz.audio_capture.source import AudioSourceError
from buzz.audio_capture.windows_process_source import WindowsProcessAudioSource
from buzz.audio_capture.windows_system_source import WindowsSystemAudioSource
from buzz.meeting.meeting_recorder import MeetingRecorder

# ---------------------------------------------------------------------------
# Platform gate: tests that require real Windows child processes.
# ---------------------------------------------------------------------------
_WINDOWS_REASON = "Integration tests require real Windows child/pipe semantics"

# ---------------------------------------------------------------------------
# Frozen BZWA v1 wire protocol (independent oracle, no production calls).
# ---------------------------------------------------------------------------
_BZWA_MAGIC = b"BZWA"
_BZWA_VERSION = 1
_BZWA_HEADER_SIZE = 16
_BZWA_SAMPLE_RATE = 16_000
_BZWA_CHANNELS = 1
_BZWA_FORMAT = 1  # float32 LE

# Independent PCM oracle: 6 distinct exactly-representable float32 values.
_EXPECTED_SAMPLES = np.array([0.0, -1.0, 0.5, -0.5, 0.25, -0.25], dtype="<f4")

# Bounded timeouts (deadlock / failure guards, not correctness sleeps).
_HANDSHAKE_TIMEOUT = 2.0
_GRACEFUL_STOP_TIMEOUT = 1.0
_TERMINATE_TIMEOUT = 1.0
_KILL_TIMEOUT = 1.0
_EVENT_TIMEOUT = 5.0
_STDERR_FLOOD_BYTES = 70 * 1024
_STDERR_LATE_MARKER = b"PR27_STDERR_LATE_MARKER"
_READY_LITERAL = "READY"
_STOP_ACK_LITERAL = "STOP_ACK"
_H1_MUTATION_REPETITIONS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_bzwa_header() -> bytes:
    """Independently construct a BZWA v1 16-byte header."""
    return struct.pack(
        "<4sHHIHH",
        _BZWA_MAGIC,
        _BZWA_VERSION,
        _BZWA_HEADER_SIZE,
        _BZWA_SAMPLE_RATE,
        _BZWA_CHANNELS,
        _BZWA_FORMAT,
    )


def _wait_until(predicate, *, timeout: float, message: str) -> None:
    """Poll test-visible state with a bounded deadlock/failure deadline."""
    deadline = time.monotonic() + timeout
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(message)
        time.sleep(min(0.01, remaining))


def _marker_has_literal(path: Path, literal: str) -> bool:
    """Return whether *path* contains the complete lifecycle marker."""
    try:
        return path.read_text(encoding="ascii") == literal
    except FileNotFoundError:
        return False


def _wait_for_lifecycle_marker(
    path: Path,
    literal: str,
    *,
    child: subprocess.Popen,
    errors: list[Exception],
) -> None:
    """Wait for a child marker, stopping early on child/runtime failure."""
    _wait_until(
        lambda: _marker_has_literal(path, literal)
        or child.poll() is not None
        or bool(errors),
        timeout=_EVENT_TIMEOUT,
        message=f"child did not publish {literal} lifecycle marker",
    )
    assert _marker_has_literal(path, literal)


def _capture_running_resources(source, factory: "_RecordingFactory"):
    """Retain and verify the exact process and threads owned by *source*."""
    child = source._process
    reader = source._reader_thread
    stderr = source._stderr_thread
    assert child is not None
    assert reader is not None
    assert stderr is not None
    assert factory.processes[-1] is child
    assert child.poll() is None
    assert reader.is_alive()
    assert stderr.is_alive()
    return child, reader, stderr


def _assert_resources_stopped(source, child, reader, stderr) -> None:
    """Verify cleanup using retained identities, then source-owned state."""
    assert child.poll() is not None
    assert not reader.is_alive()
    assert not stderr.is_alive()
    assert child.stdin is not None and child.stdin.closed
    assert child.stdout is not None and child.stdout.closed
    assert child.stderr is not None and child.stderr.closed
    assert not source._active
    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None


# Lines of the fake helper script, assembled without embedded newlines
# inside Python string literals so that Windows write_text CRLF
# translation cannot corrupt escape sequences.
_HELPER_LINES = [
    '"Ephemeral test-local helper - replaces native exe."',
    "import os, struct, sys",
    "",
    "# -- Parse production-facing arguments --",
    "args = sys.argv[1:]",
    "mode = None",
    "pid = None",
    "i = 0",
    "while i < len(args):",
    '    if args[i] == "--mode" and i + 1 < len(args):',
    "        mode = args[i + 1]",
    "        i += 2",
    '    elif args[i] == "--pid" and i + 1 < len(args):',
    "        pid = int(args[i + 1])",
    "        i += 2",
    "    else:",
    "        i += 1",
    "",
    "# Scenario B: reject missing mode.",
    "if mode is None:",
    "    os._exit(99)",
    "",
    "# Scenario B: process mode must have a valid PID.",
    'if mode == "process" and pid is None:',
    "    os._exit(99)",
    "",
    "# -- Read scenario configuration from environment --",
    'scenario = os.environ.get("BUZZ_TEST_SCENARIO", "default")',
    'n_samples = int(os.environ.get("BUZZ_TEST_PCM_COUNT", "6"))',
    'exit_code = int(os.environ.get("BUZZ_TEST_EXIT_CODE", "0"))',
    'buflen = int(os.environ.get("BUZZ_TEST_STDERR_FLOOD", "0"))',
    'prefix = os.environ.get("BUZZ_TEST_STDOUT_PREFIX", "")',
    'ready_marker = os.environ.get("BUZZ_TEST_READY_MARKER", "")',
    'stop_ack_marker = os.environ.get("BUZZ_TEST_STOP_ACK_MARKER", "")',
    "",
    "def _write_all(stream, data):",
    "    remaining = memoryview(data)",
    "    while remaining:",
    "        written = stream.write(remaining)",
    "        if written is None or written <= 0:",
    '            raise OSError("pipe write made no progress")',
    "        remaining = remaining[written:]",
    "",
    "def _write_marker(path, literal):",
    '    with open(path, "x", encoding="ascii") as marker:',
    "        marker.write(literal)",
    "        marker.flush()",
    "",
    "# -- Construct BZWA header + deterministic PCM (independent oracle) --",
    'header = struct.pack("<4sHHIHH", b"BZWA", 1, 16, 16000, 1, 1)',
    "pcm = struct.pack(",
    '    "<6f",',
    "    0.0, -1.0, 0.5, -0.5, 0.25, -0.25,",
    ")[: n_samples * 4]",
    "",
    "# Write directly to OS fd 1 (real pipe, no Python buffering).",
    'out = os.fdopen(1, "wb", 0)',
    "",
    "# Scenario C: stdout contamination.",
    "if prefix:",
    "    _write_all(out, prefix.encode())",
    "    out.flush()",
    "",
    "# Fragmentation controls via environment.",
    'frag_hdr = os.environ.get("BUZZ_TEST_FRAGMENT_HEADER", "0") != "0"',
    'frag_pcm = os.environ.get("BUZZ_TEST_FRAGMENT_PCM", "0") != "0"',
    "",
    "# -- Write header (possibly fragmented) --",
    "if frag_hdr:",
    "    _write_all(out, header[:4])",
    "    _write_all(out, header[4:8])",
    "    _write_all(out, header[8:12])",
    "    _write_all(out, header[12:])",
    "else:",
    "    _write_all(out, header)",
    "out.flush()",
    "",
    "# -- Scenario E: synchronously fill stderr before writing any PCM. --",
    "if buflen > 0:",
    '    err = os.fdopen(2, "wb", 0)',
    '    _write_all(err, b"D" * buflen)',
    '    _write_all(err, b"PR27_STDERR_LATE_MARKER")',
    "    err.flush()",
    "",
    "# Runtime failure is triggered only after source.start() has returned.",
    "if exit_code != 0:",
    "    _ = sys.stdin.buffer.read(1)",
    "",
    "# -- Write PCM only after any stderr flood has completed. --",
    "if frag_pcm and len(pcm) >= 4:",
    "    _write_all(out, pcm[:1])",
    "    _write_all(out, pcm[1:4])",
    "    _write_all(out, pcm[4:])",
    "else:",
    "    _write_all(out, pcm)",
    "out.flush()",
    "",
    "# Explicit failure scenarios may close stdout and terminate.",
    "if exit_code != 0:",
    '    sys.stderr.write("EXIT_CODE=%d" % exit_code + chr(10))',
    "    sys.stderr.flush()",
    "    out.close()",
    "    os._exit(exit_code)",
    "",
    "# Nominal success proves stdout is open immediately before stdin wait.",
    'if scenario == "old_eof_mutation":',
    "    out.close()",
    "out.flush()",
    "os.fstat(out.fileno())",
    "if ready_marker:",
    '    _write_marker(ready_marker, "READY")',
    'if scenario == "close_after_ready_mutation":',
    "    out.close()",
    "",
    "# Production stop is the only normal event that releases this read.",
    "stop_byte = sys.stdin.buffer.read(1)",
    'if stop_byte != b"\\x00":',
    "    os._exit(98)",
    "",
    "# Prove stdout remained open throughout the wait-for-stop interval.",
    "out.flush()",
    "os.fstat(out.fileno())",
    "if stop_ack_marker:",
    '    _write_marker(stop_ack_marker, "STOP_ACK")',
    "out.close()",
    "os._exit(0)",
]


def _write_fake_helper(directory: Path) -> Path:
    """Create a tiny ephemeral helper child script under *directory*.

    The script parses production-facing arguments (``--mode system`` or
    ``--mode process --pid N``), writes BZWA + PCM to **stdout**, reads
    stdin for the stop signal, and is configured entirely through
    environment variables – never through production CLI flags.
    """
    path = directory / "fake_windows_audio_helper.py"
    path.write_text("\n".join(_HELPER_LINES) + "\n", encoding="utf-8")
    return path


class _RecordingFactory:
    """Callable that records Popen args and delegates to real ``Popen``."""

    def __init__(
        self,
        fake_helper: Path,
        *,
        scenario: str = "default",
        pcm_count: int = 6,
        exit_code: int = 0,
        stderr_flood: int = 0,
        stdout_prefix: str = "",
        fragment_header: bool = False,
        fragment_pcm: bool = False,
    ):
        self._fake_helper = fake_helper
        self._scenario = scenario
        self._pcm_count = pcm_count
        self._exit_code = exit_code
        self._stderr_flood = stderr_flood
        self._stdout_prefix = stdout_prefix
        self._fragment_header = fragment_header
        self._fragment_pcm = fragment_pcm
        self.last_command: list | None = None
        self.last_kwargs: dict | None = None
        self.commands: list[list] = []
        self.processes: list[subprocess.Popen] = []
        self.ready_markers: list[Path] = []
        self.stop_ack_markers: list[Path] = []

    @property
    def call_args(self):
        """Match MagicMock ``(args_tuple, kwargs_dict)`` interface."""
        return ((self.last_command,), self.last_kwargs)

    def __call__(self, command, **kwargs):
        self.last_command = command
        self.last_kwargs = kwargs
        self.commands.append(command)
        env = os.environ.copy()
        env["BUZZ_TEST_SCENARIO"] = self._scenario
        env["BUZZ_TEST_PCM_COUNT"] = str(self._pcm_count)
        env["BUZZ_TEST_EXIT_CODE"] = str(self._exit_code)
        env["BUZZ_TEST_STDERR_FLOOD"] = str(self._stderr_flood)
        env["BUZZ_TEST_STDOUT_PREFIX"] = self._stdout_prefix
        env["BUZZ_TEST_FRAGMENT_HEADER"] = "1" if self._fragment_header else "0"
        env["BUZZ_TEST_FRAGMENT_PCM"] = "1" if self._fragment_pcm else "0"
        invocation = len(self.processes)
        ready_marker = self._fake_helper.parent / f"pr27-ready-{invocation}.marker"
        stop_ack_marker = (
            self._fake_helper.parent / f"pr27-stop-ack-{invocation}.marker"
        )
        env["BUZZ_TEST_READY_MARKER"] = str(ready_marker)
        env["BUZZ_TEST_STOP_ACK_MARKER"] = str(stop_ack_marker)
        self.ready_markers.append(ready_marker)
        self.stop_ack_markers.append(stop_ack_marker)
        translated = [sys.executable, "-u", str(self._fake_helper)] + list(command[1:])
        process = subprocess.Popen(translated, env=env, **kwargs)
        self.processes.append(process)
        return process


def _make_system_source(
    tmp_path: Path,
    *,
    scenario: str = "default",
    pcm_count: int = 6,
    exit_code: int = 0,
    stderr_flood: int = 0,
    stdout_prefix: str = "",
    fragment_header: bool = False,
    fragment_pcm: bool = False,
):
    """Create a ``WindowsSystemAudioSource`` with a real child process."""
    helper_path = tmp_path / "buzz-windows-audio-capture.exe"
    helper_path.touch()
    fake = _write_fake_helper(tmp_path)
    pf = _RecordingFactory(
        fake,
        scenario=scenario,
        pcm_count=pcm_count,
        exit_code=exit_code,
        stderr_flood=stderr_flood,
        stdout_prefix=stdout_prefix,
        fragment_header=fragment_header,
        fragment_pcm=fragment_pcm,
    )
    source = WindowsSystemAudioSource(
        helper_path=helper_path,
        process_factory=pf,
        platform_name="win32",
        handshake_timeout=_HANDSHAKE_TIMEOUT,
        graceful_stop_timeout=_GRACEFUL_STOP_TIMEOUT,
        terminate_timeout=_TERMINATE_TIMEOUT,
        kill_timeout=_KILL_TIMEOUT,
    )
    return source, pf


def _make_process_source(
    tmp_path: Path,
    *,
    process_id: int = 4242,
    scenario: str = "default",
    pcm_count: int = 6,
    exit_code: int = 0,
    stderr_flood: int = 0,
    stdout_prefix: str = "",
    fragment_header: bool = False,
    fragment_pcm: bool = False,
):
    """Create a ``WindowsProcessAudioSource`` with a real child process."""
    helper_path = tmp_path / "buzz-windows-audio-capture.exe"
    helper_path.touch()
    fake = _write_fake_helper(tmp_path)
    pf = _RecordingFactory(
        fake,
        scenario=scenario,
        pcm_count=pcm_count,
        exit_code=exit_code,
        stderr_flood=stderr_flood,
        stdout_prefix=stdout_prefix,
        fragment_header=fragment_header,
        fragment_pcm=fragment_pcm,
    )
    source = WindowsProcessAudioSource(
        process_id,
        helper_path=helper_path,
        process_factory=pf,
        platform_name="win32",
        windows_build=20_348,
        handshake_timeout=_HANDSHAKE_TIMEOUT,
        graceful_stop_timeout=_GRACEFUL_STOP_TIMEOUT,
        terminate_timeout=_TERMINATE_TIMEOUT,
        kill_timeout=_KILL_TIMEOUT,
    )
    return source, pf


# ---------------------------------------------------------------------------
# Scenario A: system source real-pipe success
# ---------------------------------------------------------------------------


def _assert_nominal_system_lifecycle(
    tmp_path: Path,
    *,
    scenario: str,
) -> None:
    """Exercise the complete deterministic nominal system-source oracle."""
    source, pf = _make_system_source(
        tmp_path,
        scenario=scenario,
        pcm_count=6,
        fragment_header=True,
        fragment_pcm=True,
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []
    done = threading.Event()

    def on_audio(samples: np.ndarray) -> None:
        collected.append(samples.copy())
        if sum(s.shape[0] for s in collected) >= len(_EXPECTED_SAMPLES):
            done.set()

    try:
        source.start(on_audio, on_error=errors.append)
        child, reader, stderr = _capture_running_resources(source, pf)
        assert done.wait(timeout=_EVENT_TIMEOUT)

        combined = np.concatenate(collected)
        assert combined.dtype == np.float32
        assert combined.ndim == 1
        assert combined.shape == (len(_EXPECTED_SAMPLES),)
        np.testing.assert_array_equal(combined, _EXPECTED_SAMPLES)

        _wait_for_lifecycle_marker(
            pf.ready_markers[-1],
            _READY_LITERAL,
            child=child,
            errors=errors,
        )
        assert errors == []
        assert child.poll() is None

        args, kwargs = pf.call_args
        assert args[0] == [str(source.helper_path), "--mode", "system"]
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["bufsize"] == 0

        source.stop()
        _wait_for_lifecycle_marker(
            pf.stop_ack_markers[-1],
            _STOP_ACK_LITERAL,
            child=child,
            errors=errors,
        )

        assert child.returncode == 0
        assert errors == []
        _assert_resources_stopped(source, child, reader, stderr)
    finally:
        source.stop()


def _h1_mutation_survives(tmp_path: Path, *, scenario: str) -> bool:
    """Mechanically apply an H1 mutation to the full nominal oracle."""
    tmp_path.mkdir()
    try:
        _assert_nominal_system_lifecycle(tmp_path, scenario=scenario)
    except (AssertionError, AudioSourceError):
        return False
    return True


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_system_source_real_pipe_success(tmp_path: Path) -> None:
    """System source delivers exact PCM through real OS pipes."""
    _assert_nominal_system_lifecycle(tmp_path, scenario="default")


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_old_nominal_eof_mutation_is_killed_50_of_50(
    tmp_path: Path,
    caplog,
) -> None:
    """Closing stdout before stdin wait never satisfies the nominal oracle."""
    caplog.set_level(logging.CRITICAL)
    survivors = sum(
        _h1_mutation_survives(
            tmp_path / f"old-eof-mutation-{repetition}",
            scenario="old_eof_mutation",
        )
        for repetition in range(_H1_MUTATION_REPETITIONS)
    )

    assert survivors == 0


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_close_after_ready_mutation_is_killed(tmp_path: Path, caplog) -> None:
    """READY cannot substitute for the post-stop stdout-open acknowledgement."""
    caplog.set_level(logging.CRITICAL)
    survived = _h1_mutation_survives(
        tmp_path / "close-after-ready-mutation",
        scenario="close_after_ready_mutation",
    )

    assert not survived


# ---------------------------------------------------------------------------
# Scenario B: process source exact PID/mode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_process_source_exact_pid_mode(tmp_path: Path) -> None:
    """Process source passes exact --mode process --pid <N> command."""
    test_pid = 4242
    source, pf = _make_process_source(
        tmp_path,
        process_id=test_pid,
        scenario="default",
        pcm_count=6,
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []
    done = threading.Event()

    def on_audio(samples: np.ndarray) -> None:
        collected.append(samples.copy())
        if sum(s.shape[0] for s in collected) >= len(_EXPECTED_SAMPLES):
            done.set()

    source.start(on_audio, on_error=errors.append)
    child, reader, stderr = _capture_running_resources(source, pf)
    assert done.wait(timeout=_EVENT_TIMEOUT)
    _wait_for_lifecycle_marker(
        pf.ready_markers[-1],
        _READY_LITERAL,
        child=child,
        errors=errors,
    )
    assert errors == []
    assert child.poll() is None

    args, kwargs = pf.call_args
    assert args[0] == [
        str(source.helper_path),
        "--mode",
        "process",
        "--pid",
        str(test_pid),
    ]
    assert kwargs["stdin"] == subprocess.PIPE
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["bufsize"] == 0

    source.stop()
    _wait_for_lifecycle_marker(
        pf.stop_ack_markers[-1],
        _STOP_ACK_LITERAL,
        child=child,
        errors=errors,
    )

    combined = np.concatenate(collected)
    assert combined.dtype == np.float32
    assert combined.ndim == 1
    assert combined.shape == (len(_EXPECTED_SAMPLES),)
    np.testing.assert_array_equal(combined, _EXPECTED_SAMPLES)
    assert child.returncode == 0
    assert errors == []
    _assert_resources_stopped(source, child, reader, stderr)


# ---------------------------------------------------------------------------
# Scenario C: stdout contamination
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_stdout_contamination_rejected(tmp_path: Path) -> None:
    """Text on stdout before BZWA triggers startup failure, no callback."""
    source, pf = _make_system_source(
        tmp_path,
        scenario="default",
        pcm_count=6,
        stdout_prefix="starting capture\n",
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []

    with pytest.raises(AudioSourceError, match="protocol|BZWA|magic|handshake|Invalid"):
        source.start(
            lambda s: collected.append(s.copy()),
            on_error=lambda e: errors.append(e),
        )

    child = pf.processes[-1]
    assert errors == []
    assert child.poll() is not None
    assert child.stdin is not None and child.stdin.closed
    assert child.stdout is not None and child.stdout.closed
    assert child.stderr is not None and child.stderr.closed
    assert not source._active
    assert source._process is None
    assert source._reader_thread is None
    assert source._stderr_thread is None
    assert len(collected) == 0


# ---------------------------------------------------------------------------
# Scenario D: runtime nonzero exit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_runtime_nonzero_exit(tmp_path: Path) -> None:
    """Valid handshake + PCM prefix then non-zero exit → exactly one error."""
    prefix_count = 3
    source, pf = _make_system_source(
        tmp_path,
        scenario="runtime_nonzero",
        pcm_count=prefix_count,
        exit_code=14,
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []
    error_event = threading.Event()

    def on_audio(samples: np.ndarray) -> None:
        collected.append(samples.copy())

    def on_error(exc: Exception) -> None:
        errors.append(exc)
        error_event.set()

    source.start(on_audio, on_error=on_error)
    child, reader, stderr = _capture_running_resources(source, pf)
    assert source._process is child
    assert source._reader_thread is reader
    assert source._stderr_thread is stderr
    assert child.stdin is not None
    child.stdin.write(b"\x01")
    child.stdin.flush()
    assert error_event.wait(timeout=_EVENT_TIMEOUT), "runtime error not reported"

    source.stop()

    assert len(errors) == 1
    combined = np.concatenate(collected)
    assert combined.shape == (prefix_count,)
    np.testing.assert_array_equal(combined, _EXPECTED_SAMPLES[:prefix_count])
    _assert_resources_stopped(source, child, reader, stderr)


# ---------------------------------------------------------------------------
# Scenario E: stderr flood with real pipe
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_stderr_flood_real_pipe(tmp_path: Path) -> None:
    """Stderr flood is fully drained before PCM; retained tail is 64 KiB.

    The child synchronously writes more than 64 KiB of stderr, including
    a unique late marker, before it writes any PCM. PCM delivery therefore
    proves the drain thread prevented real pipe backpressure from blocking
    the child.
    """
    source, pf = _make_system_source(
        tmp_path,
        scenario="stderr_flood",
        pcm_count=6,
        stderr_flood=_STDERR_FLOOD_BYTES,
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []
    done = threading.Event()

    def on_audio(samples: np.ndarray) -> None:
        collected.append(samples.copy())
        if sum(s.shape[0] for s in collected) >= len(_EXPECTED_SAMPLES):
            done.set()

    source.start(on_audio, on_error=errors.append)
    child, reader, stderr = _capture_running_resources(source, pf)
    assert done.wait(timeout=_EVENT_TIMEOUT), "PCM delivery deadlocked by stderr flood"

    _wait_until(
        lambda: len(source._stderr_tail) == source._STDERR_LIMIT
        and bytes(source._stderr_tail).endswith(_STDERR_LATE_MARKER),
        timeout=_EVENT_TIMEOUT,
        message="stderr tail did not retain the complete bounded late tail",
    )
    _wait_for_lifecycle_marker(
        pf.ready_markers[-1],
        _READY_LITERAL,
        child=child,
        errors=errors,
    )

    # Capture stderr tail before stop() clears it via _reset_after_stop_locked.
    stderr_tail = bytes(source._stderr_tail)
    assert errors == []
    assert child.poll() is None
    source.stop()
    _wait_for_lifecycle_marker(
        pf.stop_ack_markers[-1],
        _STOP_ACK_LITERAL,
        child=child,
        errors=errors,
    )

    assert len(stderr_tail) == 65_536
    assert stderr_tail.endswith(_STDERR_LATE_MARKER)

    combined = np.concatenate(collected)
    assert combined.dtype == np.float32
    assert combined.ndim == 1
    assert combined.shape == (len(_EXPECTED_SAMPLES),)
    np.testing.assert_array_equal(combined, _EXPECTED_SAMPLES)
    assert child.returncode == 0
    assert errors == []
    _assert_resources_stopped(source, child, reader, stderr)


# ---------------------------------------------------------------------------
# Scenario F: normal stop signal / child reap
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_normal_stop_reaps_child(tmp_path: Path) -> None:
    """Normal stop() reaps child, terminates threads, no runtime error."""
    source, pf = _make_system_source(
        tmp_path,
        scenario="normal_stop",
        pcm_count=2,
    )
    collected: list[np.ndarray] = []
    errors: list[Exception] = []
    done = threading.Event()

    def on_audio(samples: np.ndarray) -> None:
        collected.append(samples.copy())
        if sum(s.shape[0] for s in collected) >= 2:
            done.set()

    def on_error(exc: Exception) -> None:
        errors.append(exc)

    source.start(on_audio, on_error=on_error)
    child, reader, stderr = _capture_running_resources(source, pf)
    assert done.wait(timeout=_EVENT_TIMEOUT)
    _wait_for_lifecycle_marker(
        pf.ready_markers[-1],
        _READY_LITERAL,
        child=child,
        errors=errors,
    )
    assert errors == []
    assert child.poll() is None

    source.stop()
    _wait_for_lifecycle_marker(
        pf.stop_ack_markers[-1],
        _STOP_ACK_LITERAL,
        child=child,
        errors=errors,
    )

    combined = np.concatenate(collected)
    np.testing.assert_array_equal(combined, _EXPECTED_SAMPLES[:2])
    assert child.returncode == 0
    assert errors == []
    _assert_resources_stopped(source, child, reader, stderr)


# ---------------------------------------------------------------------------
# Scenario G: source → fanout → recorder
# ---------------------------------------------------------------------------


class _ValidatingWriter:
    """Test-local archive writer that records PCM16 blocks."""

    def __init__(self, output_path, sample_rate):
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.blocks: list[np.ndarray] = []

    def write(self, pcm16: np.ndarray) -> None:
        self.blocks.append(pcm16.copy())

    def flush(self) -> None:
        pass

    def finalize(self) -> None:
        pass

    def publish(self) -> None:
        pass

    def discard(self) -> None:
        pass

    def close_after_failure(self) -> None:
        pass


class _ValidatingWriterFactory:
    """Creates and tracks ``_ValidatingWriter`` instances."""

    def __init__(self):
        self.writer: _ValidatingWriter | None = None

    def __call__(self, output_path, sample_rate):
        self.writer = _ValidatingWriter(output_path, sample_rate)
        return self.writer


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_source_to_fanout_to_recorder_exact_samples(tmp_path: Path) -> None:
    """Source -> fanout -> recorder delivers exact sample count and values."""
    source, pf = _make_system_source(
        tmp_path,
        scenario="default",
        pcm_count=6,
        fragment_header=True,
        fragment_pcm=True,
    )
    writer_factory = _ValidatingWriterFactory()
    recorder = MeetingRecorder(
        output_path=tmp_path / "archive.wav",
        sample_rate=16_000,
        _writer_factory=writer_factory,
    )
    source_errors: list[Exception] = []
    fanout = MeetingAudioFanout(
        source,
        recorder,
        on_source_error=source_errors.append,
    )
    fanout.start()
    child, reader, stderr = _capture_running_resources(source, pf)

    # Wait for all PCM to flow through the pipeline.
    _wait_until(
        lambda: recorder.accepted_sample_count == len(_EXPECTED_SAMPLES),
        timeout=_EVENT_TIMEOUT,
        message="recorder did not accept all expected samples",
    )
    _wait_for_lifecycle_marker(
        pf.ready_markers[-1],
        _READY_LITERAL,
        child=child,
        errors=source_errors,
    )
    assert source_errors == []
    assert fanout.source_error is None
    assert child.poll() is None

    fanout.stop()
    _wait_for_lifecycle_marker(
        pf.stop_ack_markers[-1],
        _STOP_ACK_LITERAL,
        child=child,
        errors=source_errors,
    )

    assert writer_factory.writer is not None
    assert recorder.accepted_sample_count == len(_EXPECTED_SAMPLES)

    recorded_pcm16 = np.concatenate(writer_factory.writer.blocks)
    expected_pcm16 = np.array(
        [0, -32768, 16384, -16384, 8192, -8192],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(recorded_pcm16, expected_pcm16)

    assert child.returncode == 0
    assert source_errors == []
    assert fanout.source_error is None
    _assert_resources_stopped(source, child, reader, stderr)
