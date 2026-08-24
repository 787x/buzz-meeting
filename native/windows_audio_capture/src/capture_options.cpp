#include "capture_options.h"

#include <winternl.h>

#include <cwchar>
#include <limits>

namespace buzz::windows_audio {
namespace {

constexpr DWORD kMinimumProcessLoopbackBuild = 20'348;

bool ParseProcessId(const wchar_t* text, DWORD* process_id) {
    if (text == nullptr || process_id == nullptr || text[0] == L'\0') {
        return false;
    }

    unsigned long long value = 0;
    for (const wchar_t* current = text; *current != L'\0'; ++current) {
        if (*current < L'0' || *current > L'9') {
            return false;
        }
        const unsigned long long digit =
            static_cast<unsigned long long>(*current - L'0');
        const unsigned long long maximum = std::numeric_limits<DWORD>::max();
        if (value > (maximum - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    if (value == 0) {
        return false;
    }

    *process_id = static_cast<DWORD>(value);
    return true;
}

}  // namespace

bool ParseCaptureOptions(
    int argument_count,
    const wchar_t* const arguments[],
    CaptureOptions* options
) {
    if (arguments == nullptr || options == nullptr) {
        return false;
    }

    if (argument_count == 2 &&
        std::wcscmp(arguments[1], L"--self-test") == 0) {
        *options = {CaptureMode::kSelfTest, 0};
        return true;
    }

    if (argument_count == 3 &&
        std::wcscmp(arguments[1], L"--mode") == 0 &&
        std::wcscmp(arguments[2], L"system") == 0) {
        *options = {CaptureMode::kSystem, 0};
        return true;
    }

    if (argument_count == 5 &&
        std::wcscmp(arguments[1], L"--mode") == 0 &&
        std::wcscmp(arguments[2], L"process") == 0 &&
        std::wcscmp(arguments[3], L"--pid") == 0) {
        DWORD process_id = 0;
        if (!ParseProcessId(arguments[4], &process_id)) {
            return false;
        }
        *options = {CaptureMode::kProcess, process_id};
        return true;
    }

    return false;
}

bool IsProcessLoopbackBuildSupported(DWORD build_number) {
    return build_number >= kMinimumProcessLoopbackBuild;
}

bool QueryWindowsBuildNumber(DWORD* build_number) {
    if (build_number == nullptr) {
        return false;
    }

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll == nullptr) {
        return false;
    }
    using RtlGetVersionFunction = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    const auto rtl_get_version = reinterpret_cast<RtlGetVersionFunction>(
        GetProcAddress(ntdll, "RtlGetVersion")
    );
    if (rtl_get_version == nullptr) {
        return false;
    }

    RTL_OSVERSIONINFOW version{};
    version.dwOSVersionInfoSize = sizeof(version);
    if (rtl_get_version(&version) < 0) {
        return false;
    }
    *build_number = version.dwBuildNumber;
    return true;
}

}  // namespace buzz::windows_audio
