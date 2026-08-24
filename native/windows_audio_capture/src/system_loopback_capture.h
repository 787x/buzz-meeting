#pragma once

#include <Windows.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

#include "loopback_capture_session.h"

namespace buzz::windows_audio {

class SystemLoopbackCapture {
public:
    SystemLoopbackCapture();
    ~SystemLoopbackCapture();

    SystemLoopbackCapture(const SystemLoopbackCapture&) = delete;
    SystemLoopbackCapture& operator=(const SystemLoopbackCapture&) = delete;

    CaptureResult InitializeAndStart(HANDLE stop_event, HANDLE audio_ready_event);
    CaptureResult Capture(HANDLE stop_event, BoundedPcmQueue* queue);
    void Stop();

private:
    HRESULT ActivateAndInitialize(DWORD stream_flags);

    Microsoft::WRL::ComPtr<IMMDevice> device_;
    LoopbackCaptureSession session_;
};

}  // namespace buzz::windows_audio
