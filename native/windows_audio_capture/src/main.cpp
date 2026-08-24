#include <Windows.h>
#include <objbase.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

#include "pcm_transport.h"
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
    std::atomic<int>* writer_exit_code
) {
    HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    const bwa::ProtocolHeader header = bwa::SerializeProtocolHeader();
    if (!bwa::WriteAll(output, header.data(), header.size())) {
        writer_exit_code->store(static_cast<int>(bwa::ExitCode::kPipeFailure));
        std::fprintf(stderr, "Failed to write system-audio startup handshake.\n");
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
            std::fprintf(stderr, "System-audio PCM stdout pipe was closed.\n");
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

int RunSystemCapture() {
    UniqueHandle stop_event;
    stop_event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    UniqueHandle audio_ready_event;
    audio_ready_event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (stop_event.value == nullptr || audio_ready_event.value == nullptr) {
        std::fprintf(stderr, "Failed to create system-audio synchronization events.\n");
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
            "Failed to initialize COM for system audio (HRESULT 0x%08lX).\n",
            static_cast<unsigned long>(com_result)
        );
        return static_cast<int>(bwa::ExitCode::kComInitialization);
    }
    com_apartment.initialized = true;

    bwa::SystemLoopbackCapture capture;
    bwa::CaptureResult result = capture.InitializeAndStart(audio_ready_event.value);
    if (result.ok() && WaitForSingleObject(stop_event.value, 0) != WAIT_OBJECT_0) {
        writer_thread = std::thread(
            WriterThreadMain,
            stop_event.value,
            &queue,
            &writer_exit_code
        );
        result = capture.Capture(stop_event.value, &queue);
    }

    capture.Stop();
    thread_cleanup.StopAndJoin();

    if (!result.ok()) {
        bwa::PrintNativeError(result);
        return static_cast<int>(result.exit_code);
    }
    return writer_exit_code.load();
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
    try {
        if (argc == 2 && std::wcscmp(argv[1], L"--self-test") == 0) {
            if (!RunSelfTest()) {
                std::fprintf(stderr, "System-audio helper self-test failed.\n");
                return static_cast<int>(bwa::ExitCode::kInternalFailure);
            }
            std::fprintf(stderr, "System-audio helper self-test passed.\n");
            return 0;
        }

        if (argc == 3 && std::wcscmp(argv[1], L"--mode") == 0 &&
            std::wcscmp(argv[2], L"system") == 0) {
            return RunSystemCapture();
        }

        std::fprintf(
            stderr,
            "Usage: buzz-windows-audio-capture.exe --mode system\n"
            "       buzz-windows-audio-capture.exe --self-test\n"
        );
        return static_cast<int>(bwa::ExitCode::kUsage);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "System-audio helper failed: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "System-audio helper failed with an unknown error.\n");
    }
    return static_cast<int>(bwa::ExitCode::kInternalFailure);
}
