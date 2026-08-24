#include "system_loopback_capture.h"

#include <cstdio>
#include <utility>

namespace buzz::windows_audio {
namespace {

constexpr DWORD kBaseStreamFlags =
    AUDCLNT_STREAMFLAGS_LOOPBACK |
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM;

CaptureResult Failure(ExitCode code, HRESULT result, const char* context) {
    return {code, result, context};
}

}  // namespace

SystemLoopbackCapture::SystemLoopbackCapture() : session_("System audio") {}

SystemLoopbackCapture::~SystemLoopbackCapture() {
    Stop();
}

HRESULT SystemLoopbackCapture::ActivateAndInitialize(DWORD stream_flags) {
    Microsoft::WRL::ComPtr<IAudioClient> candidate;
    const HRESULT result = device_->Activate(
        __uuidof(IAudioClient),
        CLSCTX_ALL,
        nullptr,
        reinterpret_cast<void**>(candidate.GetAddressOf())
    );
    if (FAILED(result)) {
        return result;
    }
    return session_.Initialize(std::move(candidate), stream_flags);
}

CaptureResult SystemLoopbackCapture::InitializeAndStart(
    HANDLE,
    HANDLE audio_ready_event
) {
    Microsoft::WRL::ComPtr<IMMDeviceEnumerator> enumerator;
    HRESULT result = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        nullptr,
        CLSCTX_ALL,
        IID_PPV_ARGS(enumerator.GetAddressOf())
    );
    if (FAILED(result)) {
        return Failure(
            ExitCodeForHresult(result, ExitCode::kNoDefaultEndpoint),
            result,
            "create audio device enumerator"
        );
    }

    result = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
    if (FAILED(result)) {
        return Failure(
            ExitCodeForHresult(result, ExitCode::kNoDefaultEndpoint),
            result,
            "get default system output"
        );
    }

    const DWORD high_quality_flags =
        kBaseStreamFlags | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    const HRESULT first_result = ActivateAndInitialize(high_quality_flags);
    if (FAILED(first_result)) {
        std::fprintf(
            stderr,
            "WASAPI 16 kHz mono initialization with high-quality SRC failed "
            "(HRESULT 0x%08lX); retrying without SRC_DEFAULT_QUALITY.\n",
            static_cast<unsigned long>(first_result)
        );

        result = ActivateAndInitialize(kBaseStreamFlags);
        if (FAILED(result)) {
            std::fprintf(
                stderr,
                "WASAPI 16 kHz mono initialization retry failed "
                "(first HRESULT 0x%08lX, second HRESULT 0x%08lX).\n",
                static_cast<unsigned long>(first_result),
                static_cast<unsigned long>(result)
            );
            return Failure(
                ExitCodeForHresult(result, ExitCode::kAudioClientInitialization),
                result,
                "initialize 16 kHz mono system loopback"
            );
        }
    }

    return session_.Start(audio_ready_event, "start system loopback capture");
}

CaptureResult SystemLoopbackCapture::Capture(
    HANDLE stop_event,
    BoundedPcmQueue* queue
) {
    return session_.Capture(stop_event, queue);
}

void SystemLoopbackCapture::Stop() {
    session_.Stop();
    device_.Reset();
}

}  // namespace buzz::windows_audio
