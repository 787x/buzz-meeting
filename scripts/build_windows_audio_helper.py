"""Build the native Windows system-audio capture helper with CMake/MSVC."""

from pathlib import Path
import platform
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "native" / "windows_audio_capture"
BUILD_DIR = PROJECT_ROOT / "build" / "windows_audio_capture"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "buzz"
    / "native"
    / "windows"
    / "buzz-windows-audio-capture.exe"
)


def main() -> int:
    if sys.platform != "win32":
        print("The Windows system-audio helper can only be built on Windows.", file=sys.stderr)
        return 1
    if platform.machine().lower() not in ("amd64", "x86_64"):
        print("The Windows system-audio helper requires an x64 build host.", file=sys.stderr)
        return 1

    subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(BUILD_DIR),
            "--fresh",
            "-A",
            "x64",
            "-DBUILD_TESTING=ON",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--config",
            "Release",
            "--parallel",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    built_helper = BUILD_DIR / "Release" / OUTPUT_PATH.name
    if not built_helper.is_file():
        print(f"CMake did not produce the expected helper: {built_helper}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_helper, OUTPUT_PATH)
    if not OUTPUT_PATH.is_file():
        print(f"Failed to copy the helper to: {OUTPUT_PATH}", file=sys.stderr)
        return 1

    print(f"Windows system-audio helper: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
