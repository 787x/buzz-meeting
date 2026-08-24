#include "process_loopback_capture.h"

#include <mmdeviceapi.h>
#include <wrl/implements.h>

#include <cstdio>
#include <utility>

namespace buzz::windows_audio {
namespace {

using Microsoft::WRL::ClassicCom;
using Microsoft::WRL::ComPtr;
using Microsoft::WRL::FtmBase;
using Microsoft::WRL::Make;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;

CaptureResult Failure(ExitCode code, HRESULT result, const char* context) {
    return {code, result, context};
}

class ActivationCompletionHandler final
    : public RuntimeClass<
          RuntimeClassFlags<ClassicCom>,
          FtmBase,
          IActivateAudioInterfaceCompletionHandler> {
public:
    void SetState(std::shared_ptr<ActivationState> state) {
        state_ = std::move(state);
    }

    IFACEMETHODIMP ActivateCompleted(
        IActivateAudioInterfaceAsyncOperation* operation
    ) override {
        if (!state_) {
            return E_UNEXPECTED;
        }
        if (operation == nullptr) {
            state_->Complete(E_POINTER, nullptr);
            return S_OK;
        }

        HRESULT activation_result = E_FAIL;
        ComPtr<IUnknown> activated_interface;
        const HRESULT get_result = operation->GetActivateResult(
            &activation_result,
            activated_interface.GetAddressOf()
        );
        if (FAILED(get_result)) {
            state_->Complete(get_result, nullptr);
            return S_OK;
        }
        if (FAILED(activation_result)) {
            state_->Complete(activation_result, nullptr);
            return S_OK;
        }

        ComPtr<IAudioClient> audio_client;
        const HRESULT query_result = activated_interface.As(&audio_client);
        state_->Complete(query_result, std::move(audio_client));
        return S_OK;
    }

private:
    std::shared_ptr<ActivationState> state_;
};

}  // namespace

ProcessActivationDependencies BuildProductionProcessActivationDependencies() {
    ProcessActivationDependencies dependencies;
    dependencies.activate = [](
        LPCWSTR device_interface_path,
        REFIID interface_id,
        PROPVARIANT* activation_parameters,
        IActivateAudioInterfaceCompletionHandler* completion_handler,
        IActivateAudioInterfaceAsyncOperation** operation
    ) {
        return ActivateAudioInterfaceAsync(
            device_interface_path,
            interface_id,
            activation_parameters,
            completion_handler,
            operation
        );
    };
    dependencies.tick_count = []() { return GetTickCount64(); };
    dependencies.wait_for_events = [](
        DWORD event_count,
        const HANDLE* events,
        BOOL wait_for_all,
        DWORD timeout
    ) {
        return WaitForMultipleObjects(
            event_count,
            events,
            wait_for_all,
            timeout
        );
    };
    return dependencies;
}

ComPtr<IActivateAudioInterfaceCompletionHandler>
CreateProcessActivationCompletionHandler(
    std::shared_ptr<ActivationState> state
) {
    ComPtr<ActivationCompletionHandler> handler =
        Make<ActivationCompletionHandler>();
    if (!handler) {
        return nullptr;
    }
    handler->SetState(std::move(state));
    return handler;
}

std::array<InitializationAttempt, 2> BuildInitializationAttempts() {
    return {{
        {
            kProcessBaseStreamFlags | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
            1,
        },
        {kProcessBaseStreamFlags, 2},
    }};
}

ActivationWaitOutcome ClassifyActivationWait(
    DWORD wait_result,
    bool stop_signaled
) {
    if (stop_signaled || wait_result == WAIT_OBJECT_0) {
        return ActivationWaitOutcome::kStopped;
    }
    if (wait_result == WAIT_OBJECT_0 + 1) {
        return ActivationWaitOutcome::kCompleted;
    }
    if (wait_result == WAIT_TIMEOUT) {
        return ActivationWaitOutcome::kTimedOut;
    }
    return ActivationWaitOutcome::kFailed;
}

AUDIOCLIENT_ACTIVATION_PARAMS BuildProcessActivationParameters(DWORD process_id) {
    AUDIOCLIENT_ACTIVATION_PARAMS parameters{};
    parameters.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    parameters.ProcessLoopbackParams.TargetProcessId = process_id;
    parameters.ProcessLoopbackParams.ProcessLoopbackMode =
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;
    return parameters;
}

PROPVARIANT BuildProcessActivationProperty(
    AUDIOCLIENT_ACTIVATION_PARAMS* parameters
) {
    PROPVARIANT property{};
    property.vt = VT_BLOB;
    property.blob.cbSize = sizeof(AUDIOCLIENT_ACTIVATION_PARAMS);
    property.blob.pBlobData = reinterpret_cast<BYTE*>(parameters);
    return property;
}

ActivationState::ActivationState() {
    completed_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
}

ActivationState::~ActivationState() {
    if (completed_event_ != nullptr) {
        CloseHandle(completed_event_);
    }
}

bool ActivationState::valid() const {
    return completed_event_ != nullptr;
}

HANDLE ActivationState::completed_event() const {
    return completed_event_;
}

void ActivationState::Complete(
    HRESULT activation_result,
    ComPtr<IAudioClient> audio_client
) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!completed_) {
            completed_ = true;
            activation_result_ = activation_result;
            if (!abandoned_ && SUCCEEDED(activation_result)) {
                audio_client_ = std::move(audio_client);
            }
        }
    }
    if (completed_event_ != nullptr) {
        SetEvent(completed_event_);
    }
}

void ActivationState::Abandon() {
    std::lock_guard<std::mutex> lock(mutex_);
    abandoned_ = true;
    audio_client_.Reset();
}

ComPtr<IAudioClient> ActivationState::TakeAudioClient(
    HRESULT* activation_result
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (activation_result != nullptr) {
        *activation_result = activation_result_;
    }
    if (abandoned_) {
        return nullptr;
    }
    return std::move(audio_client_);
}

bool ActivationState::completed_for_test() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}

bool ActivationState::abandoned_for_test() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return abandoned_;
}

HRESULT ActivationState::result_for_test() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return activation_result_;
}

bool ActivationState::has_audio_client_for_test() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_client_ != nullptr;
}

ProcessLoopbackCapture::ProcessLoopbackCapture(DWORD process_id)
    : ProcessLoopbackCapture(
          process_id,
          BuildProductionProcessActivationDependencies()
      ) {}

ProcessLoopbackCapture::ProcessLoopbackCapture(
    DWORD process_id,
    ProcessActivationDependencies activation_dependencies
)
    : process_id_(process_id),
      activation_dependencies_(std::move(activation_dependencies)),
      session_("Process audio") {}

ProcessLoopbackCapture::~ProcessLoopbackCapture() {
    Stop();
}

ProcessLoopbackCapture::ActivationResult
ProcessLoopbackCapture::ActivateFreshClient(
    HANDLE stop_event,
    ULONGLONG activation_deadline
) {
    if (WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        return {{}, nullptr, true};
    }
    const ULONGLONG before_activation = activation_dependencies_.tick_count();
    if (before_activation >= activation_deadline) {
        return {
            Failure(
                ExitCode::kProcessActivationTimeout,
                HRESULT_FROM_WIN32(ERROR_TIMEOUT),
                "activate process loopback before deadline"
            ),
            nullptr,
            false,
        };
    }

    auto state = std::make_shared<ActivationState>();
    if (!state->valid()) {
        return {
            Failure(
                ExitCode::kInternalFailure,
                HRESULT_FROM_WIN32(GetLastError()),
                "create process activation completion event"
            ),
            nullptr,
            false,
        };
    }
    if (activation_dependencies_.state_created) {
        activation_dependencies_.state_created(state);
    }

    ComPtr<IActivateAudioInterfaceCompletionHandler> handler =
        CreateProcessActivationCompletionHandler(state);
    if (!handler) {
        return {
            Failure(
                ExitCode::kInternalFailure,
                E_OUTOFMEMORY,
                "create process activation completion handler"
            ),
            nullptr,
            false,
        };
    }
    AUDIOCLIENT_ACTIVATION_PARAMS parameters =
        BuildProcessActivationParameters(process_id_);
    PROPVARIANT activation_property =
        BuildProcessActivationProperty(&parameters);
    ComPtr<IActivateAudioInterfaceAsyncOperation> async_operation;
    const HRESULT begin_result = activation_dependencies_.activate(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        __uuidof(IAudioClient),
        &activation_property,
        handler.Get(),
        async_operation.GetAddressOf()
    );
    if (FAILED(begin_result)) {
        state->Abandon();
        std::fprintf(
            stderr,
            "Process loopback activation call failed for PID %lu "
            "(HRESULT 0x%08lX).\n",
            static_cast<unsigned long>(process_id_),
            static_cast<unsigned long>(begin_result)
        );
        return {
            Failure(
                ExitCode::kProcessActivationFailure,
                begin_result,
                "begin process loopback activation"
            ),
            nullptr,
            false,
        };
    }

    const ULONGLONG before_wait = activation_dependencies_.tick_count();
    if (before_wait >= activation_deadline) {
        state->Abandon();
        return {
            Failure(
                ExitCode::kProcessActivationTimeout,
                HRESULT_FROM_WIN32(ERROR_TIMEOUT),
                "wait for process loopback activation"
            ),
            nullptr,
            false,
        };
    }
    const DWORD wait_timeout = static_cast<DWORD>(activation_deadline - before_wait);
    HANDLE wait_events[] = {stop_event, state->completed_event()};
    const DWORD wait_result = activation_dependencies_.wait_for_events(
        2,
        wait_events,
        FALSE,
        wait_timeout
    );

    const bool stop_signaled =
        WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0;
    const ActivationWaitOutcome wait_outcome =
        ClassifyActivationWait(wait_result, stop_signaled);
    if (wait_outcome == ActivationWaitOutcome::kStopped) {
        state->Abandon();
        return {{}, nullptr, true};
    }
    if (wait_outcome == ActivationWaitOutcome::kTimedOut) {
        state->Abandon();
        std::fprintf(
            stderr,
            "Process loopback activation timed out for PID %lu after %lu ms.\n",
            static_cast<unsigned long>(process_id_),
            static_cast<unsigned long>(kProcessActivationTimeoutMilliseconds)
        );
        return {
            Failure(
                ExitCode::kProcessActivationTimeout,
                HRESULT_FROM_WIN32(ERROR_TIMEOUT),
                "wait for process loopback activation"
            ),
            nullptr,
            false,
        };
    }
    if (wait_outcome == ActivationWaitOutcome::kFailed) {
        const HRESULT wait_error = HRESULT_FROM_WIN32(GetLastError());
        state->Abandon();
        return {
            Failure(
                ExitCode::kProcessActivationFailure,
                wait_error,
                "wait for process loopback activation"
            ),
            nullptr,
            false,
        };
    }

    HRESULT activation_result = E_FAIL;
    ComPtr<IAudioClient> audio_client =
        state->TakeAudioClient(&activation_result);
    if (FAILED(activation_result) || !audio_client) {
        if (SUCCEEDED(activation_result)) {
            activation_result = E_NOINTERFACE;
        }
        std::fprintf(
            stderr,
            "Process loopback activation failed for PID %lu "
            "(HRESULT 0x%08lX).\n",
            static_cast<unsigned long>(process_id_),
            static_cast<unsigned long>(activation_result)
        );
        return {
            Failure(
                ExitCode::kProcessActivationFailure,
                activation_result,
                "complete process loopback activation"
            ),
            nullptr,
            false,
        };
    }

    return {{}, std::move(audio_client), false};
}

CaptureResult ProcessLoopbackCapture::InitializeAndStart(
    HANDLE stop_event,
    HANDLE audio_ready_event
) {
    const ULONGLONG activation_deadline =
        activation_dependencies_.tick_count() +
        kProcessActivationTimeoutMilliseconds;
    const auto attempts = BuildInitializationAttempts();

    HRESULT first_initialize_result = S_OK;
    for (const InitializationAttempt& attempt : attempts) {
        ActivationResult activation =
            ActivateFreshClient(stop_event, activation_deadline);
        if (activation.stopped) {
            return {};
        }
        if (!activation.result.ok()) {
            return activation.result;
        }

        const HRESULT initialize_result = session_.Initialize(
            std::move(activation.audio_client),
            attempt.stream_flags
        );
        if (SUCCEEDED(initialize_result)) {
            return session_.Start(
                audio_ready_event,
                "start process loopback capture"
            );
        }

        if (attempt.activation_sequence == 1) {
            first_initialize_result = initialize_result;
            std::fprintf(
                stderr,
                "Process loopback 16 kHz mono initialization with high-quality "
                "SRC failed for PID %lu (HRESULT 0x%08lX); activating a fresh "
                "client and retrying without SRC_DEFAULT_QUALITY.\n",
                static_cast<unsigned long>(process_id_),
                static_cast<unsigned long>(initialize_result)
            );
            continue;
        }

        std::fprintf(
            stderr,
            "Process loopback 16 kHz mono initialization retry failed for PID "
            "%lu (first HRESULT 0x%08lX, second HRESULT 0x%08lX).\n",
            static_cast<unsigned long>(process_id_),
            static_cast<unsigned long>(first_initialize_result),
            static_cast<unsigned long>(initialize_result)
        );
        return Failure(
            ExitCodeForHresult(
                initialize_result,
                ExitCode::kAudioClientInitialization
            ),
            initialize_result,
            "initialize 16 kHz mono process loopback"
        );
    }

    return Failure(
        ExitCode::kInternalFailure,
        E_UNEXPECTED,
        "initialize process loopback capture"
    );
}

CaptureResult ProcessLoopbackCapture::Capture(
    HANDLE stop_event,
    BoundedPcmQueue* queue
) {
    return session_.Capture(stop_event, queue);
}

void ProcessLoopbackCapture::Stop() {
    session_.Stop();
}

}  // namespace buzz::windows_audio
