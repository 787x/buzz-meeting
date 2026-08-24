#pragma once

#include <Windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <propidl.h>
#include <wrl/client.h>

#include <array>
#include <functional>
#include <memory>
#include <mutex>

#include "loopback_capture_session.h"

namespace buzz::windows_audio {

constexpr DWORD kProcessActivationTimeoutMilliseconds = 8'000;
constexpr DWORD kProcessBaseStreamFlags =
    AUDCLNT_STREAMFLAGS_LOOPBACK |
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK |
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM;

struct InitializationAttempt {
    DWORD stream_flags;
    unsigned int activation_sequence;
};

enum class ActivationWaitOutcome {
    kStopped,
    kCompleted,
    kTimedOut,
    kFailed,
};

std::array<InitializationAttempt, 2> BuildInitializationAttempts();
ActivationWaitOutcome ClassifyActivationWait(
    DWORD wait_result,
    bool stop_signaled
);
AUDIOCLIENT_ACTIVATION_PARAMS BuildProcessActivationParameters(DWORD process_id);
PROPVARIANT BuildProcessActivationProperty(
    AUDIOCLIENT_ACTIVATION_PARAMS* parameters
);

class ActivationState {
public:
    ActivationState();
    ~ActivationState();

    ActivationState(const ActivationState&) = delete;
    ActivationState& operator=(const ActivationState&) = delete;

    bool valid() const;
    HANDLE completed_event() const;
    void Complete(
        HRESULT activation_result,
        Microsoft::WRL::ComPtr<IAudioClient> audio_client
    );
    void Abandon();
    Microsoft::WRL::ComPtr<IAudioClient> TakeAudioClient(HRESULT* activation_result);

    bool completed_for_test() const;
    bool abandoned_for_test() const;
    HRESULT result_for_test() const;
    bool has_audio_client_for_test() const;

private:
    mutable std::mutex mutex_;
    HANDLE completed_event_ = nullptr;
    bool completed_ = false;
    bool abandoned_ = false;
    HRESULT activation_result_ = E_PENDING;
    Microsoft::WRL::ComPtr<IAudioClient> audio_client_;
};

using ActivateAudioInterfaceAsyncFunction = std::function<HRESULT(
    LPCWSTR,
    REFIID,
    PROPVARIANT*,
    IActivateAudioInterfaceCompletionHandler*,
    IActivateAudioInterfaceAsyncOperation**
)>;

struct ProcessActivationDependencies {
    ActivateAudioInterfaceAsyncFunction activate;
    std::function<ULONGLONG()> tick_count;
    std::function<DWORD(DWORD, const HANDLE*, BOOL, DWORD)> wait_for_events;
    std::function<void(const std::shared_ptr<ActivationState>&)> state_created;
};

ProcessActivationDependencies BuildProductionProcessActivationDependencies();
Microsoft::WRL::ComPtr<IActivateAudioInterfaceCompletionHandler>
CreateProcessActivationCompletionHandler(
    std::shared_ptr<ActivationState> state
);

class ProcessLoopbackCapture {
public:
    explicit ProcessLoopbackCapture(DWORD process_id);
    ProcessLoopbackCapture(
        DWORD process_id,
        ProcessActivationDependencies activation_dependencies
    );
    ~ProcessLoopbackCapture();

    ProcessLoopbackCapture(const ProcessLoopbackCapture&) = delete;
    ProcessLoopbackCapture& operator=(const ProcessLoopbackCapture&) = delete;

    CaptureResult InitializeAndStart(HANDLE stop_event, HANDLE audio_ready_event);
    CaptureResult Capture(HANDLE stop_event, BoundedPcmQueue* queue);
    void Stop();

private:
    struct ActivationResult {
        CaptureResult result;
        Microsoft::WRL::ComPtr<IAudioClient> audio_client;
        bool stopped = false;
    };

    ActivationResult ActivateFreshClient(
        HANDLE stop_event,
        ULONGLONG activation_deadline
    );

    DWORD process_id_;
    ProcessActivationDependencies activation_dependencies_;
    LoopbackCaptureSession session_;
};

}  // namespace buzz::windows_audio
