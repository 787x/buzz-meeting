#include <Windows.h>
#include <audioclient.h>
#include <wrl/implements.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "capture_options.h"
#include "pcm_transport.h"
#include "process_loopback_capture.h"
#include "system_loopback_capture.h"

namespace bwa = buzz::windows_audio;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

using Microsoft::WRL::ClassicCom;
using Microsoft::WRL::ComPtr;
using Microsoft::WRL::Make;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;

struct ScopedHandle {
    HANDLE value = nullptr;

    ~ScopedHandle() {
        if (value != nullptr && value != INVALID_HANDLE_VALUE) {
            CloseHandle(value);
        }
    }

    ScopedHandle() = default;
    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;
};

struct FakeAudioClientStats {
    explicit FakeAudioClientStats(int value) : object_id(value) {}

    int object_id;
    int initialize_calls = 0;
    int start_calls = 0;
    int stop_calls = 0;
    int destroyed = 0;
    DWORD initialize_flags = 0;
};

class FakeAudioCaptureClient final
    : public RuntimeClass<RuntimeClassFlags<ClassicCom>, IAudioCaptureClient> {
public:
    IFACEMETHODIMP GetBuffer(
        BYTE**,
        UINT32*,
        DWORD*,
        UINT64*,
        UINT64*
    ) override {
        return E_NOTIMPL;
    }

    IFACEMETHODIMP ReleaseBuffer(UINT32) override {
        return S_OK;
    }

    IFACEMETHODIMP GetNextPacketSize(UINT32* packet_size) override {
        if (packet_size == nullptr) {
            return E_POINTER;
        }
        *packet_size = 0;
        return S_OK;
    }
};

class FakeAudioClient final
    : public RuntimeClass<RuntimeClassFlags<ClassicCom>, IAudioClient> {
public:
    FakeAudioClient(
        std::shared_ptr<FakeAudioClientStats> stats,
        HRESULT initialize_result,
        HRESULT start_result
    )
        : stats_(std::move(stats)),
          initialize_result_(initialize_result),
          start_result_(start_result),
          capture_client_(Make<FakeAudioCaptureClient>()) {}

    ~FakeAudioClient() {
        ++stats_->destroyed;
    }

    IFACEMETHODIMP Initialize(
        AUDCLNT_SHAREMODE,
        DWORD stream_flags,
        REFERENCE_TIME,
        REFERENCE_TIME,
        const WAVEFORMATEX*,
        LPCGUID
    ) override {
        ++stats_->initialize_calls;
        stats_->initialize_flags = stream_flags;
        return initialize_result_;
    }

    IFACEMETHODIMP GetBufferSize(UINT32*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetStreamLatency(REFERENCE_TIME*) override {
        return E_NOTIMPL;
    }
    IFACEMETHODIMP GetCurrentPadding(UINT32*) override { return E_NOTIMPL; }
    IFACEMETHODIMP IsFormatSupported(
        AUDCLNT_SHAREMODE,
        const WAVEFORMATEX*,
        WAVEFORMATEX** closest_match
    ) override {
        if (closest_match != nullptr) {
            *closest_match = nullptr;
        }
        return E_NOTIMPL;
    }
    IFACEMETHODIMP GetMixFormat(WAVEFORMATEX** format) override {
        if (format != nullptr) {
            *format = nullptr;
        }
        return E_NOTIMPL;
    }
    IFACEMETHODIMP GetDevicePeriod(REFERENCE_TIME*, REFERENCE_TIME*) override {
        return E_NOTIMPL;
    }
    IFACEMETHODIMP Start() override {
        ++stats_->start_calls;
        return start_result_;
    }
    IFACEMETHODIMP Stop() override {
        ++stats_->stop_calls;
        return S_OK;
    }
    IFACEMETHODIMP Reset() override { return S_OK; }
    IFACEMETHODIMP SetEventHandle(HANDLE) override { return S_OK; }
    IFACEMETHODIMP GetService(REFIID interface_id, void** service) override {
        if (service == nullptr) {
            return E_POINTER;
        }
        *service = nullptr;
        if (!IsEqualIID(interface_id, __uuidof(IAudioCaptureClient))) {
            return E_NOINTERFACE;
        }
        return capture_client_->QueryInterface(interface_id, service);
    }

private:
    std::shared_ptr<FakeAudioClientStats> stats_;
    HRESULT initialize_result_;
    HRESULT start_result_;
    ComPtr<IAudioCaptureClient> capture_client_;
};

struct FakeOperationStats {
    int get_activate_result_calls = 0;
};

class FakeActivateAudioInterfaceAsyncOperation final
    : public RuntimeClass<
          RuntimeClassFlags<ClassicCom>,
          IActivateAudioInterfaceAsyncOperation> {
public:
    FakeActivateAudioInterfaceAsyncOperation(
        HRESULT get_result,
        HRESULT activation_result,
        ComPtr<IAudioClient> audio_client,
        std::shared_ptr<FakeOperationStats> stats
    )
        : get_result_(get_result),
          activation_result_(activation_result),
          audio_client_(std::move(audio_client)),
          stats_(std::move(stats)) {}

    IFACEMETHODIMP GetActivateResult(
        HRESULT* activation_result,
        IUnknown** activated_interface
    ) override {
        ++stats_->get_activate_result_calls;
        if (activation_result == nullptr || activated_interface == nullptr) {
            return E_POINTER;
        }
        *activation_result = activation_result_;
        *activated_interface = nullptr;
        if (FAILED(get_result_)) {
            return get_result_;
        }
        if (audio_client_) {
            *activated_interface = audio_client_.Get();
            (*activated_interface)->AddRef();
            audio_client_.Reset();
        }
        return get_result_;
    }

private:
    HRESULT get_result_;
    HRESULT activation_result_;
    ComPtr<IAudioClient> audio_client_;
    std::shared_ptr<FakeOperationStats> stats_;
};

enum class FakeWaitAction {
    kComplete,
    kTimeout,
    kCompleteAndStop,
};

struct FakeActivationPlan {
    HRESULT begin_result = S_OK;
    HRESULT get_result = S_OK;
    HRESULT activation_result = S_OK;
    HRESULT initialize_result = S_OK;
    HRESULT start_result = S_OK;
    FakeWaitAction wait_action = FakeWaitAction::kComplete;
    ULONGLONG elapsed_milliseconds = 0;
    std::shared_ptr<FakeAudioClientStats> client_stats;
    std::shared_ptr<FakeOperationStats> operation_stats =
        std::make_shared<FakeOperationStats>();
};

class FakeActivationController {
public:
    explicit FakeActivationController(
        std::vector<FakeActivationPlan> plans,
        bool retain_state = false
    )
        : plans_(std::move(plans)), retain_state_(retain_state) {}

    bwa::ProcessActivationDependencies BuildDependencies() {
        bwa::ProcessActivationDependencies dependencies;
        dependencies.activate = [this](
            LPCWSTR device_interface_path,
            REFIID interface_id,
            PROPVARIANT* activation_parameters,
            IActivateAudioInterfaceCompletionHandler* completion_handler,
            IActivateAudioInterfaceAsyncOperation** operation
        ) {
            return Activate(
                device_interface_path,
                interface_id,
                activation_parameters,
                completion_handler,
                operation
            );
        };
        dependencies.tick_count = [this]() { return now_; };
        dependencies.wait_for_events = [this](
            DWORD event_count,
            const HANDLE* events,
            BOOL wait_for_all,
            DWORD timeout
        ) {
            return Wait(event_count, events, wait_for_all, timeout);
        };
        dependencies.state_created = [this](
            const std::shared_ptr<bwa::ActivationState>& state
        ) {
            last_state_ = state;
            if (retain_state_) {
                retained_state_ = state;
            }
        };
        return dependencies;
    }

    int activation_calls() const { return activation_calls_; }
    const std::vector<DWORD>& wait_timeouts() const { return wait_timeouts_; }
    bool first_client_released_before_second_activation() const {
        return first_client_released_before_second_activation_;
    }
    std::shared_ptr<bwa::ActivationState> last_state() const {
        return retained_state_ != nullptr ? retained_state_ : last_state_.lock();
    }
    bool state_alive() const { return !last_state_.expired(); }
    HRESULT CompletePending() {
        if (!current_handler_ || !current_operation_) {
            return E_UNEXPECTED;
        }
        return current_handler_->ActivateCompleted(current_operation_.Get());
    }
    void ReleasePending() {
        current_handler_.Reset();
        current_operation_.Reset();
        current_plan_ = nullptr;
    }

private:
    HRESULT Activate(
        LPCWSTR device_interface_path,
        REFIID interface_id,
        PROPVARIANT* activation_parameters,
        IActivateAudioInterfaceCompletionHandler* completion_handler,
        IActivateAudioInterfaceAsyncOperation** operation
    ) {
        Check(
            device_interface_path != nullptr &&
                std::wcscmp(
                    device_interface_path,
                    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK
                ) == 0,
            "fake activation receives the process-loopback virtual device"
        );
        Check(
            IsEqualIID(interface_id, __uuidof(IAudioClient)),
            "fake activation requests IAudioClient"
        );
        Check(
            activation_parameters != nullptr &&
                activation_parameters->vt == VT_BLOB,
            "fake activation receives the process activation blob"
        );
        Check(
            completion_handler != nullptr && operation != nullptr,
            "fake activation receives completion and operation outputs"
        );

        ++activation_calls_;
        const std::size_t plan_index =
            static_cast<std::size_t>(activation_calls_ - 1);
        if (plan_index >= plans_.size()) {
            return E_UNEXPECTED;
        }
        if (activation_calls_ == 2 && !plans_.empty() &&
            plans_[0].client_stats != nullptr) {
            first_client_released_before_second_activation_ =
                plans_[0].client_stats->destroyed == 1;
        }

        current_plan_ = &plans_[plan_index];
        if (FAILED(current_plan_->begin_result)) {
            return current_plan_->begin_result;
        }

        ComPtr<IAudioClient> client;
        if (current_plan_->client_stats != nullptr) {
            client = Make<FakeAudioClient>(
                current_plan_->client_stats,
                current_plan_->initialize_result,
                current_plan_->start_result
            );
        }
        current_operation_ = Make<FakeActivateAudioInterfaceAsyncOperation>(
            current_plan_->get_result,
            current_plan_->activation_result,
            std::move(client),
            current_plan_->operation_stats
        );
        current_handler_ = completion_handler;
        return current_operation_.CopyTo(operation);
    }

    DWORD Wait(
        DWORD event_count,
        const HANDLE* events,
        BOOL wait_for_all,
        DWORD timeout
    ) {
        wait_timeouts_.push_back(timeout);
        if (current_plan_ == nullptr) {
            return WAIT_FAILED;
        }
        now_ += current_plan_->elapsed_milliseconds;
        if (current_plan_->wait_action == FakeWaitAction::kTimeout) {
            return WAIT_TIMEOUT;
        }

        const HRESULT callback_result =
            current_handler_->ActivateCompleted(current_operation_.Get());
        Check(
            SUCCEEDED(callback_result),
            "production completion handler accepts fake activation completion"
        );
        if (current_plan_->wait_action == FakeWaitAction::kCompleteAndStop) {
            SetEvent(events[0]);
        }
        const DWORD wait_result = WaitForMultipleObjects(
            event_count,
            events,
            wait_for_all,
            0
        );
        current_handler_.Reset();
        current_operation_.Reset();
        current_plan_ = nullptr;
        return wait_result;
    }

    std::vector<FakeActivationPlan> plans_;
    ULONGLONG now_ = 0;
    int activation_calls_ = 0;
    bool first_client_released_before_second_activation_ = false;
    std::vector<DWORD> wait_timeouts_;
    bool retain_state_ = false;
    FakeActivationPlan* current_plan_ = nullptr;
    ComPtr<IActivateAudioInterfaceCompletionHandler> current_handler_;
    ComPtr<IActivateAudioInterfaceAsyncOperation> current_operation_;
    std::weak_ptr<bwa::ActivationState> last_state_;
    std::shared_ptr<bwa::ActivationState> retained_state_;
};

struct TransportTrace {
    int writer_starts = 0;
    int valid_header_count = 0;
    int pcm_write_count = 0;
    std::size_t header_bytes = 0;
};

bool RunTransportGate(
    const bwa::CaptureResult& startup_result,
    HANDLE stop_event,
    TransportTrace* trace
) {
    return bwa::StartTransportAfterCaptureStart(
        startup_result,
        stop_event,
        [trace]() {
            ++trace->writer_starts;
            const bwa::ProtocolHeader header = bwa::SerializeProtocolHeader();
            trace->header_bytes += header.size();
            if (header.size() == bwa::kProtocolHeaderSize &&
                std::memcmp(header.data(), "BZWA", 4) == 0) {
                ++trace->valid_header_count;
            }
            ++trace->pcm_write_count;
        }
    );
}

struct StartupRun {
    bwa::CaptureResult result;
    TransportTrace transport;
};

StartupRun RunProcessStartup(
    FakeActivationController* controller,
    HANDLE stop_event
) {
    ScopedHandle audio_ready_event;
    audio_ready_event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Check(
        audio_ready_event.value != nullptr,
        "process startup test creates an audio-ready event"
    );

    bwa::ProcessLoopbackCapture capture(4242, controller->BuildDependencies());
    StartupRun run;
    run.result = capture.InitializeAndStart(
        stop_event,
        audio_ready_event.value
    );
    RunTransportGate(run.result, stop_event, &run.transport);
    capture.Stop();
    return run;
}

void TestProtocolHeader() {
    const bwa::ProtocolHeader header = bwa::SerializeProtocolHeader();
    const std::uint8_t expected[] = {
        'B', 'Z', 'W', 'A',
        1, 0,
        16, 0,
        0x80, 0x3e, 0, 0,
        1, 0,
        1, 0,
    };
    Check(
        std::memcmp(header.data(), expected, sizeof(expected)) == 0,
        "protocol header has the exact little-endian layout"
    );
}

void TestTargetFormat() {
    const WAVEFORMATEX format = bwa::BuildTargetFormat();
    Check(format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT, "target is IEEE float");
    Check(format.nSamplesPerSec == 16'000, "target sample rate is 16 kHz");
    Check(format.nChannels == 1, "target has one channel");
    Check(format.wBitsPerSample == 32, "target samples have 32 bits");
    Check(format.nBlockAlign == 4, "target block alignment is four bytes");
    Check(format.nAvgBytesPerSec == 64'000, "target rate is 64 KiB per second");
}

bool ParseOptions(
    const std::vector<const wchar_t*>& arguments,
    bwa::CaptureOptions* options
) {
    return bwa::ParseCaptureOptions(
        static_cast<int>(arguments.size()),
        arguments.data(),
        options
    );
}

void TestCommandLineParsing() {
    bwa::CaptureOptions options;
    Check(
        ParseOptions({L"helper", L"--mode", L"system"}, &options) &&
            options.mode == bwa::CaptureMode::kSystem &&
            options.process_id == 0,
        "system mode command remains compatible"
    );
    Check(
        ParseOptions(
            {L"helper", L"--mode", L"process", L"--pid", L"4294967295"},
            &options
        ) &&
            options.mode == bwa::CaptureMode::kProcess &&
            options.process_id == 0xFFFFFFFF,
        "process mode accepts an exact DWORD PID"
    );
    Check(
        ParseOptions({L"helper", L"--self-test"}, &options) &&
            options.mode == bwa::CaptureMode::kSelfTest,
        "self-test command remains compatible"
    );

    const std::vector<std::vector<const wchar_t*>> invalid_commands = {
        {L"helper", L"--mode", L"process"},
        {L"helper", L"--mode", L"process", L"--pid", L"0"},
        {L"helper", L"--mode", L"process", L"--pid", L"-1"},
        {L"helper", L"--mode", L"process", L"--pid", L"+1"},
        {L"helper", L"--mode", L"process", L"--pid", L"12x"},
        {L"helper", L"--mode", L"process", L"--pid", L"4294967296"},
        {
            L"helper",
            L"--mode",
            L"process",
            L"--pid",
            L"184467440737095516160000",
        },
        {L"helper", L"--mode", L"process", L"--pid", L"1", L"extra"},
        {L"helper", L"--mode", L"system", L"--pid", L"1"},
        {L"helper", L"--mode", L"unknown"},
    };
    for (const auto& command : invalid_commands) {
        Check(!ParseOptions(command, &options), "malformed command is rejected");
    }
}

void TestProcessBuildGate() {
    Check(
        !bwa::IsProcessLoopbackBuildSupported(20'347),
        "build 20347 is rejected for process loopback"
    );
    Check(
        bwa::IsProcessLoopbackBuildSupported(20'348),
        "build 20348 is accepted for process loopback"
    );
    Check(
        bwa::IsProcessLoopbackBuildSupported(26'100),
        "later Windows builds are accepted for process loopback"
    );
}

void TestProcessActivationParameters() {
    AUDIOCLIENT_ACTIVATION_PARAMS parameters =
        bwa::BuildProcessActivationParameters(0xFEDCBA98);
    Check(
        parameters.ActivationType == AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        "activation type is process loopback"
    );
    Check(
        parameters.ProcessLoopbackParams.TargetProcessId == 0xFEDCBA98,
        "activation target contains the exact DWORD PID"
    );
    Check(
        parameters.ProcessLoopbackParams.ProcessLoopbackMode ==
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
        "activation includes the target process tree"
    );

    const PROPVARIANT property =
        bwa::BuildProcessActivationProperty(&parameters);
    Check(property.vt == VT_BLOB, "activation property is a blob");
    Check(
        property.blob.cbSize == sizeof(AUDIOCLIENT_ACTIVATION_PARAMS),
        "activation property has the exact parameter size"
    );
    Check(
        property.blob.pBlobData == reinterpret_cast<BYTE*>(&parameters),
        "activation property points at the activation parameters"
    );
}

void TestInitializationAttemptPlan() {
    const auto attempts = bwa::BuildInitializationAttempts();
    Check(attempts.size() == 2, "process initialization has exactly two attempts");
    Check(
        attempts[0].activation_sequence == 1 &&
            attempts[1].activation_sequence == 2,
        "each initialization attempt requires a fresh activation sequence"
    );
    Check(
        (attempts[0].stream_flags & bwa::kProcessBaseStreamFlags) ==
            bwa::kProcessBaseStreamFlags &&
            (attempts[0].stream_flags &
             AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY) != 0,
        "first process attempt uses high-quality automatic SRC"
    );
    Check(
        attempts[1].stream_flags == bwa::kProcessBaseStreamFlags,
        "second process attempt drops only SRC_DEFAULT_QUALITY"
    );
}

void TestActivationStateLifecycle() {
    bwa::ActivationState failed_state;
    Check(failed_state.valid(), "activation state owns a completion event");
    failed_state.Complete(E_ACCESSDENIED, nullptr);
    Check(failed_state.completed_for_test(), "activation failure completes state");
    Check(
        failed_state.result_for_test() == E_ACCESSDENIED,
        "activation failure HRESULT is retained"
    );
    Check(
        WaitForSingleObject(failed_state.completed_event(), 0) == WAIT_OBJECT_0,
        "activation completion signals its event"
    );

    bwa::ActivationState completed_state;
    completed_state.Complete(S_OK, nullptr);
    Check(completed_state.completed_for_test(), "activation success completes state");
    Check(
        completed_state.result_for_test() == S_OK,
        "activation success HRESULT is retained"
    );

    bwa::ActivationState abandoned_state;
    abandoned_state.Abandon();
    abandoned_state.Complete(S_OK, nullptr);
    Check(abandoned_state.abandoned_for_test(), "stopped activation is abandoned");
    Check(
        abandoned_state.completed_for_test(),
        "late completion safely finishes abandoned state"
    );
    Check(
        !abandoned_state.has_audio_client_for_test(),
        "late completion cannot publish a client after abandonment"
    );
}

void TestActivationWaitOutcomes() {
    Check(
        bwa::ClassifyActivationWait(WAIT_OBJECT_0 + 1, true) ==
            bwa::ActivationWaitOutcome::kStopped,
        "stop wins when completion and stop are both observed"
    );
    Check(
        bwa::ClassifyActivationWait(WAIT_OBJECT_0, false) ==
            bwa::ActivationWaitOutcome::kStopped,
        "stop event produces the stopped activation state"
    );
    Check(
        bwa::ClassifyActivationWait(WAIT_OBJECT_0 + 1, false) ==
            bwa::ActivationWaitOutcome::kCompleted,
        "completion event produces the completed activation state"
    );
    Check(
        bwa::ClassifyActivationWait(WAIT_TIMEOUT, false) ==
            bwa::ActivationWaitOutcome::kTimedOut,
        "bounded activation wait produces the timeout state"
    );
    Check(
        bwa::ClassifyActivationWait(WAIT_FAILED, false) ==
            bwa::ActivationWaitOutcome::kFailed,
        "failed activation wait produces the failure state"
    );
}

void TestLateCompletionAfterOwnerAbandonment() {
    ScopedHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    auto client_stats = std::make_shared<FakeAudioClientStats>(1);
    FakeActivationPlan plan;
    plan.client_stats = client_stats;
    plan.wait_action = FakeWaitAction::kTimeout;
    FakeActivationController controller({plan});

    const StartupRun run = RunProcessStartup(&controller, stop_event.value);
    Check(
        run.result.exit_code == bwa::ExitCode::kProcessActivationTimeout,
        "owner follows the production activation-timeout path"
    );
    Check(
        controller.state_alive(),
        "only pending handler ownership keeps abandoned state alive"
    );
    {
        const std::shared_ptr<bwa::ActivationState> abandoned_state =
            controller.last_state();
        Check(
            abandoned_state != nullptr && abandoned_state->abandoned_for_test(),
            "owner marks state abandoned before returning"
        );
    }
    Check(
        client_stats->destroyed == 0,
        "fake async operation owns returned client before late completion"
    );
    Check(
        SUCCEEDED(controller.CompletePending()),
        "real completion handler processes fake operation after owner destruction"
    );
    std::shared_ptr<bwa::ActivationState> completed_state =
        controller.last_state();
    Check(completed_state != nullptr, "late callback state remains inspectable");
    Check(
        completed_state != nullptr && completed_state->completed_for_test(),
        "late callback completes state"
    );
    Check(
        completed_state != nullptr &&
            WaitForSingleObject(completed_state->completed_event(), 0) ==
                WAIT_OBJECT_0,
        "late callback signals the heap-owned completion event"
    );
    Check(
        completed_state != nullptr && completed_state->abandoned_for_test(),
        "late state remains abandoned"
    );
    Check(
        completed_state != nullptr &&
            !completed_state->has_audio_client_for_test(),
        "late callback cannot publish its returned client"
    );
    Check(
        plan.operation_stats->get_activate_result_calls == 1,
        "late callback executes GetActivateResult exactly once"
    );
    Check(client_stats->initialize_calls == 0, "late client is not initialized");
    Check(client_stats->start_calls == 0, "late client is not started");
    Check(client_stats->destroyed == 1, "late returned client is released");

    Check(
        run.transport.valid_header_count == 0 &&
            run.transport.pcm_write_count == 0,
        "late completion emits no header or PCM"
    );
    controller.ReleasePending();
    completed_state.reset();
    Check(
        !controller.state_alive(),
        "state is destroyed after callback and handler ownership release"
    );
}

void TestStopPrecedesSimultaneousActivationCompletion() {
    ScopedHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    Check(stop_event.value != nullptr, "simultaneous test creates stop event");

    auto client_stats = std::make_shared<FakeAudioClientStats>(2);
    FakeActivationPlan plan;
    plan.client_stats = client_stats;
    plan.wait_action = FakeWaitAction::kCompleteAndStop;
    FakeActivationController controller({plan}, true);

    const StartupRun run = RunProcessStartup(&controller, stop_event.value);
    const std::shared_ptr<bwa::ActivationState> state = controller.last_state();
    Check(run.result.ok(), "stop during activation exits without startup error");
    Check(
        WaitForSingleObject(stop_event.value, 0) == WAIT_OBJECT_0,
        "stop and completion events are both ready before owner decision"
    );
    Check(state != nullptr && state->completed_for_test(), "completion ran first");
    Check(
        state != nullptr && state->abandoned_for_test(),
        "owner deterministically abandons when stop is also signaled"
    );
    Check(client_stats->initialize_calls == 0, "stopped client is not initialized");
    Check(client_stats->start_calls == 0, "stopped client is not started");
    Check(client_stats->destroyed == 1, "stopped completed client is released");
    Check(
        run.transport.valid_header_count == 0 &&
            run.transport.pcm_write_count == 0,
        "simultaneous stop/completion emits no transport data"
    );
}

void TestAbsoluteActivationDeadline() {
    ScopedHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);

    auto first_stats = std::make_shared<FakeAudioClientStats>(3);
    FakeActivationPlan first;
    first.client_stats = first_stats;
    first.initialize_result = E_FAIL;
    first.elapsed_milliseconds = 6'000;

    auto second_stats = std::make_shared<FakeAudioClientStats>(4);
    FakeActivationPlan second;
    second.client_stats = second_stats;
    second.wait_action = FakeWaitAction::kTimeout;
    second.elapsed_milliseconds = 2'000;

    FakeActivationController controller({first, second});
    const StartupRun run = RunProcessStartup(&controller, stop_event.value);
    const std::vector<DWORD>& waits = controller.wait_timeouts();
    Check(controller.activation_calls() == 2, "deadline test activates twice");
    Check(
        waits.size() == 2 && waits[0] == 8'000 && waits[1] == 2'000,
        "two fresh attempts share one 8000 ms absolute deadline"
    );
    Check(
        run.result.exit_code == bwa::ExitCode::kProcessActivationTimeout,
        "second activation times out at the total deadline"
    );
    Check(first_stats->initialize_calls == 1, "first attempt consumes 6000 ms");
    Check(second_stats->initialize_calls == 0, "timed-out second client is unused");
    Check(
        run.transport.valid_header_count == 0 &&
            run.transport.pcm_write_count == 0,
        "absolute deadline failure emits no transport data"
    );
}

void TestFreshActivationRetryAndStartBeforeHeader() {
    ScopedHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);

    auto first_stats = std::make_shared<FakeAudioClientStats>(5);
    FakeActivationPlan first;
    first.client_stats = first_stats;
    first.initialize_result = E_FAIL;

    auto second_stats = std::make_shared<FakeAudioClientStats>(6);
    FakeActivationPlan second;
    second.client_stats = second_stats;

    FakeActivationController controller({first, second});
    const StartupRun run = RunProcessStartup(&controller, stop_event.value);
    Check(run.result.ok(), "fresh second activation starts successfully");
    Check(controller.activation_calls() == 2, "Initialize failure reactivates once");
    Check(
        first_stats->object_id != second_stats->object_id,
        "first and second activation return distinct clients A and B"
    );
    Check(
        controller.first_client_released_before_second_activation(),
        "failed client A is released before activating client B"
    );
    Check(first_stats->destroyed == 1, "failed client A remains released");
    Check(second_stats->destroyed == 1, "successful client B releases on Stop");
    Check(
        (first_stats->initialize_flags &
         AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY) != 0,
        "client A initializes with SRC_DEFAULT_QUALITY"
    );
    Check(
        first_stats->initialize_flags ==
            (bwa::kProcessBaseStreamFlags |
             AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY),
        "client A receives the exact first-attempt flags"
    );
    Check(
        second_stats->initialize_flags == bwa::kProcessBaseStreamFlags,
        "client B receives the exact retry flags without high-quality SRC"
    );
    Check(first_stats->start_calls == 0, "failed client A never starts");
    Check(second_stats->start_calls == 1, "successful client B starts once");
    Check(
        run.transport.writer_starts == 1 &&
            run.transport.valid_header_count == 1 &&
            run.transport.header_bytes == bwa::kProtocolHeaderSize &&
            run.transport.pcm_write_count == 1,
        "writer emits one BZWA header only after client B Start succeeds"
    );
}

void CheckNoStartupOutput(const StartupRun& run, const std::string& scenario) {
    Check(
        run.transport.writer_starts == 0,
        scenario + " does not start the writer"
    );
    Check(
        run.transport.header_bytes == 0 &&
            run.transport.valid_header_count == 0,
        scenario + " writes zero BZWA header bytes"
    );
    Check(
        run.transport.pcm_write_count == 0,
        scenario + " writes zero PCM chunks"
    );
}

void TestStartupFailuresWriteNoOutput() {
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        FakeActivationPlan plan;
        plan.begin_result = E_ACCESSDENIED;
        FakeActivationController controller({plan});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            run.result.exit_code == bwa::ExitCode::kProcessActivationFailure,
            "immediate activation failure keeps category 20"
        );
        CheckNoStartupOutput(run, "immediate activation failure");
    }
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        auto stats = std::make_shared<FakeAudioClientStats>(7);
        FakeActivationPlan plan;
        plan.client_stats = stats;
        plan.get_result = E_FAIL;
        FakeActivationController controller({plan});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            plan.operation_stats->get_activate_result_calls == 1,
            "completion handler executes failing GetActivateResult"
        );
        Check(
            run.result.exit_code == bwa::ExitCode::kProcessActivationFailure,
            "GetActivateResult failure keeps category 20"
        );
        CheckNoStartupOutput(run, "GetActivateResult failure");
    }
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        auto stats = std::make_shared<FakeAudioClientStats>(7);
        FakeActivationPlan plan;
        plan.client_stats = stats;
        plan.activation_result = E_ACCESSDENIED;
        FakeActivationController controller({plan});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            plan.operation_stats->get_activate_result_calls == 1,
            "completion failure executes GetActivateResult"
        );
        Check(
            run.result.exit_code == bwa::ExitCode::kProcessActivationFailure,
            "completion activation failure keeps category 20"
        );
        CheckNoStartupOutput(run, "completion activation failure");
    }
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        FakeActivationPlan plan;
        plan.wait_action = FakeWaitAction::kTimeout;
        FakeActivationController controller({plan});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            run.result.exit_code == bwa::ExitCode::kProcessActivationTimeout,
            "activation timeout keeps category 21"
        );
        CheckNoStartupOutput(run, "activation timeout");
    }
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        auto first_stats = std::make_shared<FakeAudioClientStats>(8);
        auto second_stats = std::make_shared<FakeAudioClientStats>(9);
        FakeActivationPlan first;
        first.client_stats = first_stats;
        first.initialize_result = E_FAIL;
        FakeActivationPlan second;
        second.client_stats = second_stats;
        second.initialize_result = E_FAIL;
        FakeActivationController controller({first, second});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            run.result.exit_code == bwa::ExitCode::kAudioClientInitialization,
            "two Initialize failures keep category 12"
        );
        CheckNoStartupOutput(run, "all Initialize attempts failing");
    }
    {
        ScopedHandle stop_event;
        stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        auto stats = std::make_shared<FakeAudioClientStats>(10);
        FakeActivationPlan plan;
        plan.client_stats = stats;
        plan.start_result = E_FAIL;
        FakeActivationController controller({plan});
        const StartupRun run = RunProcessStartup(&controller, stop_event.value);
        Check(
            run.result.exit_code == bwa::ExitCode::kAudioClientStart,
            "Start failure keeps category 13"
        );
        Check(stats->start_calls == 1, "Start failure executes Start once");
        CheckNoStartupOutput(run, "Start failure");
    }
}

void TestQueueOrderingAndOverflow() {
    Check(
        bwa::kQueueCapacityFrames == 32'000,
        "production queue is bounded to two seconds of 16 kHz audio"
    );
    bwa::BoundedPcmQueue queue(5);
    Check(queue.Push({1.0F, 2.0F, 3.0F}) == 0, "first chunk is retained");
    Check(queue.Push({4.0F, 5.0F, 6.0F}) == 3, "oldest chunk is dropped");
    Check(queue.queued_frames() == 3, "queue remains bounded");
    Check(queue.dropped_frames() == 3, "queue tracks dropped frames");
    queue.Close();

    std::vector<float> samples;
    Check(queue.Pop(&samples), "remaining queue chunk can be read");
    Check(
        samples == std::vector<float>({4.0F, 5.0F, 6.0F}),
        "remaining samples preserve order"
    );
    Check(!queue.Pop(&samples), "closed empty queue reaches end of stream");

    bwa::BoundedPcmQueue oversized_queue(3);
    Check(
        oversized_queue.Push({1.0F, 2.0F, 3.0F, 4.0F, 5.0F}) == 2,
        "oversized input drops its oldest prefix"
    );
    oversized_queue.Close();
    Check(oversized_queue.Pop(&samples), "trimmed input remains readable");
    Check(
        samples == std::vector<float>({3.0F, 4.0F, 5.0F}),
        "oversized input keeps the newest samples"
    );
}

void TestSilentPacketsAndFlags() {
    const std::vector<float> silent = bwa::CopyPacketSamples(nullptr, 8, true);
    Check(silent.size() == 8, "silent packet keeps its frame count");
    for (float sample : silent) {
        Check(sample == 0.0F, "silent packet contains only zeros");
    }

    const float input[] = {0.25F, -0.5F};
    const std::vector<float> copied = bwa::CopyPacketSamples(
        reinterpret_cast<const BYTE*>(input),
        2,
        false
    );
    Check(copied == std::vector<float>({0.25F, -0.5F}), "PCM packet is copied");

    const bwa::BufferFlagInfo flags = bwa::ClassifyBufferFlags(
        AUDCLNT_BUFFERFLAGS_SILENT |
        AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY |
        AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR
    );
    Check(flags.silent, "silent flag is classified");
    Check(flags.data_discontinuity, "discontinuity flag is classified");
    Check(flags.timestamp_error, "timestamp error flag is classified");
}

void TestErrorMapping() {
    Check(
        static_cast<int>(bwa::ExitCode::kUnsupportedProcessLoopback) == 19 &&
            static_cast<int>(bwa::ExitCode::kProcessActivationFailure) == 20 &&
            static_cast<int>(bwa::ExitCode::kProcessActivationTimeout) == 21,
        "process errors extend rather than renumber existing exit categories"
    );
    Check(
        bwa::ExitCodeForHresult(
            AUDCLNT_E_DEVICE_INVALIDATED,
            bwa::ExitCode::kCaptureFailure
        ) == bwa::ExitCode::kDeviceInvalidated,
        "device invalidation has a stable category"
    );
    Check(
        bwa::ExitCodeForHresult(
            AUDCLNT_E_SERVICE_NOT_RUNNING,
            bwa::ExitCode::kCaptureFailure
        ) == bwa::ExitCode::kAudioServiceStopped,
        "stopped audio service has a stable category"
    );
    Check(
        bwa::ExitCodeForHresult(E_FAIL, bwa::ExitCode::kCaptureFailure) ==
            bwa::ExitCode::kCaptureFailure,
        "unclassified HRESULT keeps its fallback category"
    );
}

void TestStopWakeup() {
    HANDLE stop_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    Check(stop_event != nullptr, "stop event can be created");
    if (stop_event == nullptr) {
        return;
    }
    std::thread signaler([stop_event]() { SetEvent(stop_event); });
    Check(
        WaitForSingleObject(stop_event, 2'000) == WAIT_OBJECT_0,
        "stop event wakes a native waiter"
    );
    signaler.join();
    CloseHandle(stop_event);
}

void TestClosedPipeWrite() {
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    Check(CreatePipe(&read_pipe, &write_pipe, nullptr, 0) != FALSE, "pipe is created");
    if (read_pipe == nullptr || write_pipe == nullptr) {
        return;
    }
    CloseHandle(read_pipe);
    const std::uint8_t value = 1;
    Check(!bwa::WriteAll(write_pipe, &value, 1), "closed pipe write fails cleanly");
    CloseHandle(write_pipe);
}

}  // namespace

int main() {
    const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    Check(SUCCEEDED(com_result), "native behavior tests initialize the MTA");

    TestProtocolHeader();
    TestTargetFormat();
    TestCommandLineParsing();
    TestProcessBuildGate();
    TestProcessActivationParameters();
    TestInitializationAttemptPlan();
    TestActivationStateLifecycle();
    TestActivationWaitOutcomes();
    TestLateCompletionAfterOwnerAbandonment();
    TestStopPrecedesSimultaneousActivationCompletion();
    TestAbsoluteActivationDeadline();
    TestFreshActivationRetryAndStartBeforeHeader();
    TestStartupFailuresWriteNoOutput();
    TestQueueOrderingAndOverflow();
    TestSilentPacketsAndFlags();
    TestErrorMapping();
    TestStopWakeup();
    TestClosedPipeWrite();

    if (SUCCEEDED(com_result)) {
        CoUninitialize();
    }

    if (failures == 0) {
        std::fprintf(stderr, "All native Windows-audio tests passed.\n");
    }
    return failures == 0 ? 0 : 1;
}
