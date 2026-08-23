# Meeting Fork Development Baseline

This file records facts that have already been verified for the PR0 development
baseline. It is not a test report for every Buzz feature and does not claim that
PR0 reran the full project test suite.

## Verified baseline

| Item | Verified result |
| --- | --- |
| Platform | Windows 11 AMD64 |
| Buzz version | 1.4.5 |
| Python requirement | `>=3.12,<3.13` |
| `uv sync` | PASS |
| Buzz GUI launch | PASS |
| Vulkan initialization | PASS |
| Faster Whisper Large-v3 file transcription | PASS |
| `RecordingTranscriber` focused tests | PASS |
| `whisper.cpp` | Locally built and runnable after supplying the required runtime files |

Development baseline tag: `meeting-dev-baseline-2026-08-23`.

## Known development-environment notes

1. The current local `whisper.cpp` build was produced with MSYS2 UCRT64/MinGW.
   Local development therefore requires the corresponding MinGW runtime DLLs.
2. Installing the optional DeepFilterNet plugin currently produces a warning
   because Rust/Cargo is not installed on this machine. The warning does not
   block the Buzz GUI or the main meeting-development path.
3. Do not claim that the full project test suite passes unless that suite is
   actually run for the task being reported.
