#include <Windows.h>
#include <audioclient.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "pcm_transport.h"
#include "system_loopback_capture.h"

namespace bwa = buzz::windows_audio;

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
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
    TestProtocolHeader();
    TestTargetFormat();
    TestQueueOrderingAndOverflow();
    TestSilentPacketsAndFlags();
    TestErrorMapping();
    TestStopWakeup();
    TestClosedPipeWrite();

    if (failures == 0) {
        std::fprintf(stderr, "All native system-audio tests passed.\n");
    }
    return failures == 0 ? 0 : 1;
}
