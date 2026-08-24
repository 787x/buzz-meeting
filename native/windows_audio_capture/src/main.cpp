#include <Windows.h>
#include <objbase.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

#include "capture_options.h"
#include "pcm_transport.h"
#include "process_loopback_capture.h"
#include "system_loopback_capture.h"

namespace bwa = buzz::windows_audio;

namespace {

struct UniqueHandle {
    HANDLE value = nullptr;

    ~UniqueHandle() {
        if (value != nullptr && value != INVALID_HANDLE_VALUE) {
            CloseHandle(value);
        }
    }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    UniqueHandle() = default;
};

struct ComApartment {
    bool initialized = false;

    ~ComApartment() {
        if (initialized) {
            CoUninitialize();
        }
    }

    ComApartment(const ComApartment&) = delete;
    ComApartment& operator=(const ComApartment&) = delete;
    ComApartment() = default;
};

struct CaptureThreadCleanup {
    HANDLE stop_event;
    bwa::BoundedPcmQueue* queue;
    std::thread* control_thread;
    std::thread* writer_thread;
    bool complete = false;

    CaptureThreadCleanup(
        HANDLE stop_event_value,
        bwa::BoundedPcmQueue* queue_value,
        std::thread* control_thread_value,
        std::thread* writer_thread_value
    )
        : stop_event(stop_event_value),
          queue(queue_value),
          control_thread(control_thread_value),
          writer_thread(writer_thread_value) {}

    ~CaptureThreadCleanup() {
        StopAndJoin();
    }

    CaptureThreadCleanup(const CaptureThreadCleanup&) = delete;
    CaptureThreadCleanup& operator=(const CaptureThreadCleanup&) = delete;

    void StopAndJoin() noexcept {
        if (complete) {
            return;
        }
        complete = true;
        SetEvent(stop_event);
        queue->Close();

        if (writer_thread->joinable()) {
            writer_thread->join();
        }

        if (control_thread->joinable()) {
            CancelSynchronousIo(control_thread->native_handle());
            control_thread->join();
        }
    }
};

void ControlThreadMain(HANDLE stop_event) {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    BYTE command = 0;
    DWORD bytes_read = 0;
    if (input != nullptr && input != INVALID_HANDLE_VALUE) {
        ReadFile(input, &command, sizeof(command), &bytes_read, nullptr);
    }
    SetEvent(stop_event);
}

void WriterThreadMain(
    HANDLE stop_event,
    bwa::BoundedPcmQueue* queue,
    std::atomic<int>* writer_exit_code,
    const char* transport_label
) {
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    const bwa::ProtocolHeader header = bwa::SerializeProtocolHeader();
    if (!bwa::WriteAll(output, header.data(), header.size())) {
        writer_exit_code->store(static_cast<int>(bwa::ExitCode::kPipeFailure));
        std::fprintf(
            stderr,
            "Failed to write %s startup handshake.\n",
            transport_label
        );
        SetEvent(stop_event);
        return;
    }

    std::vector<float> samples;
    while (queue->Pop(&samples)) {
        if (!bwa::WriteAll(
                output,
                samples.data(),
                samples.size() * sizeof(float)
            )) {
            writer_exit_code->store(static_cast<int>(bwa::ExitCode::kPipeFailure));
            std::fprintf(stderr, "%s PCM stdout pipe was closed.\n", transport_label);
            SetEvent(stop_event);
            return;
        }
    }
}

bool RunSelfTest() {
    const bwa::ProtocolHeader header = bwa::SerializeProtocolHeader();
    if (std::memcmp(header.data(), "BZWA", 4) != 0 ||
        header.size() != bwa::kProtocolHeaderSize) {
        return false;
    }

    const WAVEFORMATEX format = bwa::BuildTargetFormat();
    if (format.nSamplesPerSec != bwa::kSampleRate ||
        format.nChannels != bwa::kChannelCount ||
        format.wFormatTag != WAVE_FORMAT_IEEE_FLOAT) {
        return false;
    }

    const AUDIOCLIENT_ACTIVATION_PARAMS process_parameters =
        bwa::BuildProcessActivationParameters(1234);
    if (process_parameters.ActivationType !=
            AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK ||
        process_parameters.ProcessLoopbackParams.TargetProcessId != 1234 ||
        process_parameters.ProcessLoopbackParams.ProcessLoopbackMode !=
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE) {
        return false;
    }

    bwa::BoundedPcmQueue queue(4);
    queue.Push({1.0F, 2.0F, 3.0F});
    queue.Push({4.0F, 5.0F});
    queue.Close();
    std::vector<float> samples;
    if (!queue.Pop(&samples) || samples != std::vector<float>({4.0F, 5.0F}) ||
        queue.dropped_frames() != 3) {
        return false;
    }

    UniqueHandle event;
    event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (event.value == nullptr || !SetEvent(event.value) ||
        WaitForSingleObject(event.value, 0) != WAIT_OBJECT_0) {
        return false;
    }
    return true;
}

template <typename CaptureType>
int RunCapture(
    CaptureType* capture,
    const char* transport_label,
    const char* com_label
) {
    UniqueHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    UniqueHandle audio_ready_event;
    audio_ready_event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (stop_event.value == nullptr || audio_ready_event.value == nullptr) {
        std::fprintf(
            stderr,
            "Failed to create %s synchronization events.\n",
            transport_label
        );
        return static_cast<int>(bwa::ExitCode::kInternalFailure);
    }

    bwa::BoundedPcmQueue queue(bwa::kQueueCapacityFrames);
    std::atomic<int> writer_exit_code{static_cast<int>(bwa::ExitCode::kSuccess)};
    std::thread control_thread;
    std::thread writer_thread;
    CaptureThreadCleanup thread_cleanup{
        stop_event.value,
        &queue,
        &control_thread,
        &writer_thread,
    };
    control_thread = std::thread(ControlThreadMain, stop_event.value);

    ComApartment com_apartment;
    const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(com_result)) {
        std::fprintf(
            stderr,
            "Failed to initialize COM for %s (HRESULT 0x%08lX).\n",
            com_label,
            static_cast<unsigned long>(com_result)
        );
        return static_cast<int>(bwa::ExitCode::kComInitialization);
    }
    com_apartment.initialized = true;

    bwa::CaptureResult result = capture->InitializeAndStart(
        stop_event.value,
        audio_ready_event.value
    );
    const bool transport_started = bwa::StartTransportAfterCaptureStart(
        result,
        stop_event.value,
        [&]() {
            writer_thread = std::thread(
                WriterThreadMain,
                stop_event.value,
                &queue,
                &writer_exit_code,
                transport_label
            );
        }
    );
    if (transport_started) {
        result = capture->Capture(stop_event.value, &queue);
    }

    capture->Stop();
    thread_cleanup.StopAndJoin();

    if (!result.ok()) {
        bwa::PrintNativeError(result);
        return static_cast<int>(result.exit_code);
    }
    return writer_exit_code.load();
}

int RunSystemCapture() {
    bwa::SystemLoopbackCapture capture;
    return RunCapture(&capture, "system-audio", "system audio");
}

int RunProcessCapture(DWORD process_id) {
    DWORD windows_build = 0;
    if (!bwa::QueryWindowsBuildNumber(&windows_build)) {
        std::fprintf(
            stderr,
            "Unable to determine the Windows build for process loopback capture.\n"
        );
        return static_cast<int>(bwa::ExitCode::kUnsupportedProcessLoopback);
    }
    if (!bwa::IsProcessLoopbackBuildSupported(windows_build)) {
        std::fprintf(
            stderr,
            "Process audio capture requires Windows 10 build 20348 or later "
            "(current build %lu).\n",
            static_cast<unsigned long>(windows_build)
        );
        return static_cast<int>(bwa::ExitCode::kUnsupportedProcessLoopback);
    }

    bwa::ProcessLoopbackCapture capture(process_id);
    return RunCapture(&capture, "process-audio", "process audio");
}

void PrintUsage() {
    std::fprintf(
        stderr,
        "Usage: buzz-windows-audio-capture.exe --mode system\n"
        "       buzz-windows-audio-capture.exe --mode process --pid <DWORD>\n"
        "       buzz-windows-audio-capture.exe --self-test\n"
    );
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
    try {
        std::vector<const wchar_t*> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.push_back(argv[index]);
        }

        bwa::CaptureOptions options;
        if (!bwa::ParseCaptureOptions(argc, arguments.data(), &options)) {
            PrintUsage();
            return static_cast<int>(bwa::ExitCode::kUsage);
        }

        switch (options.mode) {
        case bwa::CaptureMode::kSelfTest:
            if (!RunSelfTest()) {
                std::fprintf(stderr, "Windows-audio helper self-test failed.\n");
                return static_cast<int>(bwa::ExitCode::kInternalFailure);
            }
            std::fprintf(stderr, "Windows-audio helper self-test passed.\n");
            return 0;
        case bwa::CaptureMode::kSystem:
            return RunSystemCapture();
        case bwa::CaptureMode::kProcess:
            return RunProcessCapture(options.process_id);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Windows-audio helper failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "Windows-audio helper failed with an unknown error.\n");
    }
    return static_cast<int>(bwa::ExitCode::kInternalFailure);
}
