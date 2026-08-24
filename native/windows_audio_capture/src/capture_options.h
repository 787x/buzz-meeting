#pragma once

#include <Windows.h>

namespace buzz::windows_audio {

enum class CaptureMode {
    kSystem,
    kProcess,
    kSelfTest,
};

struct CaptureOptions {
    CaptureMode mode = CaptureMode::kSystem;
    DWORD process_id = 0;
};

bool ParseCaptureOptions(
    int argument_count,
    const wchar_t* const arguments[],
    CaptureOptions* options
);
bool IsProcessLoopbackBuildSupported(DWORD build_number);
bool QueryWindowsBuildNumber(DWORD* build_number);

}  // namespace buzz::windows_audio
