#pragma once

#include <Windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

#include <cstdint>

#include "pcm_transport.h"

namespace buzz::windows_audio {

enum class ExitCode : int {
    kSuccess = 0,
    kUsage = 2,
    kComInitialization = 10,
    kNoDefaultEndpoint = 11,
    kAudioClientInitialization = 12,
    kAudioClientStart = 13,
    kCaptureFailure = 14,
    kDeviceInvalidated = 15,
    kAudioServiceStopped = 16,
    kPipeFailure = 17,
    kInternalFailure = 18,
};

ExitCode ExitCodeForHresult(HRESULT result, ExitCode fallback);
WAVEFORMATEX BuildTargetFormat();

struct CaptureResult {
    ExitCode exit_code = ExitCode::kSuccess;
    HRESULT hresult = S_OK;
    const char* context = nullptr;

    bool ok() const { return exit_code == ExitCode::kSuccess; }
};

class SystemLoopbackCapture {
public:
    SystemLoopbackCapture() = default;
    ~SystemLoopbackCapture();

    SystemLoopbackCapture(const SystemLoopbackCapture&) = delete;
    SystemLoopbackCapture& operator=(const SystemLoopbackCapture&) = delete;

    CaptureResult InitializeAndStart(HANDLE audio_ready_event);
    CaptureResult Capture(HANDLE stop_event, BoundedPcmQueue* queue);
    void Stop();

private:
    HRESULT ActivateAndInitialize(DWORD stream_flags);
    CaptureResult DrainPackets(BoundedPcmQueue* queue);
    void ReportPacketDiagnostics(const BufferFlagInfo& flags, std::size_t dropped);

    Microsoft::WRL::ComPtr<IMMDevice> device_;
    Microsoft::WRL::ComPtr<IAudioClient> audio_client_;
    Microsoft::WRL::ComPtr<IAudioCaptureClient> capture_client_;
    HANDLE audio_ready_event_ = nullptr;
    bool started_ = false;
    std::uint64_t discontinuity_count_ = 0;
    std::uint64_t timestamp_error_count_ = 0;
    std::uint64_t dropped_frame_count_ = 0;
    ULONGLONG last_diagnostic_tick_ = 0;
};

void PrintNativeError(const CaptureResult& result);

}  // namespace buzz::windows_audio
