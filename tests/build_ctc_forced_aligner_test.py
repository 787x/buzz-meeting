import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_ctc_forced_aligner.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_ctc_forced_aligner_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
build_ctc_forced_aligner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_ctc_forced_aligner)


class StrictEncodingStream:
    encoding = "ascii"

    def __init__(self):
        self.parts = []

    def write(self, text):
        text.encode(self.encoding, errors="strict")
        self.parts.append(text)
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


def completed(command, returncode=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


class TestDecodeDiagnostic:
    def test_none_and_empty_are_empty(self):
        assert build_ctc_forced_aligner._decode_diagnostic(None) == ""
        assert build_ctc_forced_aligner._decode_diagnostic(b"") == ""

    def test_valid_utf8_is_readable(self):
        assert (
            build_ctc_forced_aligner._decode_diagnostic(b"hello \xe2\x82\xac")
            == "hello \u20ac"
        )

    def test_invalid_utf8_is_visibly_escaped(self):
        assert (
            build_ctc_forced_aligner._decode_diagnostic(b"before:\xff:after")
            == r"before:\xff:after"
        )

    def test_every_invalid_byte_is_preserved(self):
        assert (
            build_ctc_forced_aligner._decode_diagnostic(b"A\xffB\xfeC\x80D")
            == r"A\xffB\xfeC\x80D"
        )


def test_write_diagnostic_escapes_for_destination_encoding():
    stream = StrictEncodingStream()

    build_ctc_forced_aligner._write_diagnostic(stream, b"price \xe2\x82\xac")

    assert stream.getvalue() == r"price \u20ac" + "\n"


def test_all_subprocess_boundaries_capture_bytes(monkeypatch, tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    forward_patch = patches_dir / "ctc_forced_aligner_a.patch"
    reverse_patch = patches_dir / "ctc_forced_aligner_b.patch"
    forward_patch.write_bytes(b"a")
    reverse_patch.write_bytes(b"b")
    calls = []

    def fake_run(command, **kwargs):
        forbidden = {
            "text",
            "universal_newlines",
            "encoding",
            "errors",
        }
        assert not forbidden.intersection(kwargs)
        assert kwargs["capture_output"] is True
        calls.append((command, kwargs))

        if command[0] != "git":
            return completed(command)
        if "--reverse" in command:
            return completed(command)
        if "--check" in command:
            return completed(
                command,
                returncode=0 if command[-1] == str(forward_patch) else 1,
            )
        return completed(command)

    monkeypatch.setattr(build_ctc_forced_aligner, "PATCHES_DIR", patches_dir)
    monkeypatch.setattr(build_ctc_forced_aligner, "ALIGNER_DIR", tmp_path)
    monkeypatch.setattr(build_ctc_forced_aligner.subprocess, "run", fake_run)

    build_ctc_forced_aligner.build()

    assert len(calls) == 5
    assert any(
        command[:3] == ["git", "apply", "--check"] and "--reverse" not in command
        for command, _ in calls
    )
    assert any(
        command[:2] == ["git", "apply"] and "--check" not in command
        for command, _ in calls
    )
    assert any("--reverse" in command for command, _ in calls)
    assert any(command[-2:] == ["build_ext", "--inplace"] for command, _ in calls)


def test_success_preserves_status_and_stream_routing(monkeypatch, capsys):
    result = completed(
        ["setup.py"],
        stdout=b"OUT_SENTINEL:\xe2\x82\xac\n",
        stderr=b"ERR_SENTINEL:\xff\n",
    )
    monkeypatch.setattr(build_ctc_forced_aligner, "apply_patches", lambda: None)
    monkeypatch.setattr(
        build_ctc_forced_aligner.subprocess,
        "run",
        lambda *args, **kwargs: result,
    )

    build_ctc_forced_aligner.build()

    captured = capsys.readouterr()
    assert result.returncode == 0
    assert "OUT_SENTINEL:\u20ac" in captured.out
    assert "OUT_SENTINEL" not in captured.err
    assert r"ERR_SENTINEL:\xff" in captured.err
    assert "ERR_SENTINEL" not in captured.out


def test_nonzero_status_and_diagnostics_survive_rendering(monkeypatch, capsys):
    failure = subprocess.CalledProcessError(
        17,
        ["setup.py", "build_ext", "--inplace"],
        output=b"OUT_FAILURE:\xe2\x82\xac\n",
        stderr=b"ERR_FAILURE:\xff\n",
    )

    def fail_build(*args, **kwargs):
        assert kwargs["check"] is True
        raise failure

    monkeypatch.setattr(build_ctc_forced_aligner, "apply_patches", lambda: None)
    monkeypatch.setattr(build_ctc_forced_aligner.subprocess, "run", fail_build)
    monkeypatch.setattr(
        build_ctc_forced_aligner.sys,
        "argv",
        [str(SCRIPT_PATH), "--force"],
    )

    with pytest.raises(SystemExit) as raised:
        build_ctc_forced_aligner.main()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert failure.returncode == 17
    assert "exit status 17" in captured.err
    assert "stdout: OUT_FAILURE:\u20ac" in captured.err
    assert r"stderr: ERR_FAILURE:\xff" in captured.err
    assert "OUT_FAILURE" not in captured.out
    assert "ERR_FAILURE" not in captured.out
    assert "b'OUT_FAILURE" not in captured.err
    assert "b'ERR_FAILURE" not in captured.err


def test_both_patch_checks_fail_warns_and_continues(monkeypatch, tmp_path, capsys):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    patch_file = patches_dir / "ctc_forced_aligner_warning.patch"
    patch_file.write_bytes(b"patch")
    results = iter(
        [
            completed(
                ["git", "apply", "--check"],
                returncode=1,
                stderr=b"FORWARD_FAILURE:\xff",
            ),
            completed(
                ["git", "apply", "--check", "--reverse"],
                returncode=2,
                stderr=b"REVERSE_FAILURE",
            ),
        ]
    )
    monkeypatch.setattr(build_ctc_forced_aligner, "PATCHES_DIR", patches_dir)
    monkeypatch.setattr(build_ctc_forced_aligner, "ALIGNER_DIR", tmp_path)
    monkeypatch.setattr(
        build_ctc_forced_aligner.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )

    build_ctc_forced_aligner.apply_patches()

    captured = capsys.readouterr()
    assert "WARNING: could not apply patch" in captured.err
    assert r"FORWARD_FAILURE:\xff" in captured.err
    assert "REVERSE_FAILURE" not in captured.err


def test_actual_patch_apply_failure_is_not_swallowed(monkeypatch, tmp_path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    patch_file = patches_dir / "ctc_forced_aligner_apply.patch"
    patch_file.write_bytes(b"patch")
    failure = subprocess.CalledProcessError(
        17,
        ["git", "apply"],
        output=b"APPLY_OUT:\xff",
        stderr=b"APPLY_ERR:\xfe",
    )
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed(command)
        raise failure

    monkeypatch.setattr(build_ctc_forced_aligner, "PATCHES_DIR", patches_dir)
    monkeypatch.setattr(build_ctc_forced_aligner, "ALIGNER_DIR", tmp_path)
    monkeypatch.setattr(build_ctc_forced_aligner.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        build_ctc_forced_aligner.apply_patches()

    assert raised.value is failure
    assert raised.value.returncode == 17
