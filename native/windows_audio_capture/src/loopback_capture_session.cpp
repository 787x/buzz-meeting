#include "loopback_capture_session.h"

#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

namespace buzz::windows_audio {
namespace {

CaptureResult Failure(ExitCode code, HRESULT result, const char* context) {
    return {code, result, context};
}

}  // namespace

ExitCode ExitCodeForHresult(HRESULT result, ExitCode fallback) {
    if (result == AUDCLNT_E_DEVICE_INVALIDATED ||
        result == AUDCLNT_E_RESOURCES_INVALIDATED) {
        return ExitCode::kDeviceInvalidated;
    }
    if (result == AUDCLNT_E_SERVICE_NOT_RUNNING) {
        return ExitCode::kAudioServiceStopped;
    }
    return fallback;
}

WAVEFORMATEX BuildTargetFormat() {
    WAVEFORMATEX format{};
    format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    format.nChannels = static_cast<WORD>(kChannelCount);
    format.nSamplesPerSec = kSampleRate;
    format.wBitsPerSample = 32;
    format.nBlockAlign = static_cast<WORD>(
        format.nChannels * format.wBitsPerSample / 8
    );
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
    format.cbSize = 0;
    return format;
}

LoopbackCaptureSession::LoopbackCaptureSession(const char* diagnostic_label)
    : diagnostic_label_(diagnostic_label) {}

LoopbackCaptureSession::~LoopbackCaptureSession() {
    Stop();
}

HRESULT LoopbackCaptureSession::Initialize(
    Microsoft::WRL::ComPtr<IAudioClient> audio_client,
    DWORD stream_flags
) {
    Stop();
    if (!audio_client) {
        return E_POINTER;
    }

    WAVEFORMATEX target_format = BuildTargetFormat();
    const HRESULT result = audio_client->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        stream_flags,
        0,
        0,
        &target_format,
        nullptr
    );
    if (FAILED(result)) {
        return result;
    }

    audio_client_ = std::move(audio_client);
    return S_OK;
}

CaptureResult LoopbackCaptureSession::Start(
    HANDLE audio_ready_event,
    const char* start_context
) {
    if (!audio_client_ || audio_ready_event == nullptr) {
        return Failure(
            ExitCode::kAudioClientInitialization,
            E_POINTER,
            "prepare WASAPI capture session"
        );
    }
    audio_ready_event_ = audio_ready_event;

    HRESULT result = audio_client_->GetService(
        __uuidof(IAudioCaptureClient),
        reinterpret_cast<void**>(capture_client_.GetAddressOf())
    );
    if (FAILED(result)) {
        return Failure(
            ExitCodeForHresult(result, ExitCode::kAudioClientInitialization),
            result,
            "get WASAPI capture service"
        );
    }

    result = audio_client_->SetEventHandle(audio_ready_event_);
    if (FAILED(result)) {
        return Failure(
            ExitCodeForHresult(result, ExitCode::kAudioClientInitialization),
            result,
            "set WASAPI audio-ready event"
        );
    }

    result = audio_client_->Start();
    if (FAILED(result)) {
        return Failure(
            ExitCodeForHresult(result, ExitCode::kAudioClientStart),
            result,
            start_context
        );
    }
    started_ = true;
    return {};
}

CaptureResult LoopbackCaptureSession::Capture(
    HANDLE stop_event,
    BoundedPcmQueue* queue
) {
    if (queue == nullptr) {
        return Failure(ExitCode::kInternalFailure, E_POINTER, "capture PCM queue");
    }

    HANDLE events[] = {stop_event, audio_ready_event_};
    while (true) {
        const DWORD wait_result = WaitForMultipleObjects(2, events, FALSE, INFINITE);
        if (wait_result == WAIT_OBJECT_0) {
            return {};
        }
        if (wait_result == WAIT_OBJECT_0 + 1) {
            CaptureResult drain_result = DrainPackets(queue);
            if (!drain_result.ok()) {
                return drain_result;
            }
            continue;
        }
        const HRESULT wait_error = HRESULT_FROM_WIN32(GetLastError());
        return Failure(
            ExitCodeForHresult(wait_error, ExitCode::kCaptureFailure),
            wait_error,
            "wait for WASAPI packet"
        );
    }
}

CaptureResult LoopbackCaptureSession::DrainPackets(BoundedPcmQueue* queue) {
    while (true) {
        UINT32 next_packet_size = 0;
        HRESULT result = capture_client_->GetNextPacketSize(&next_packet_size);
        if (FAILED(result)) {
            return Failure(
                ExitCodeForHresult(result, ExitCode::kCaptureFailure),
                result,
                "get next WASAPI packet size"
            );
        }
        if (next_packet_size == 0) {
            return {};
        }

        BYTE* data = nullptr;
        UINT32 frame_count = 0;
        DWORD raw_flags = 0;
        result = capture_client_->GetBuffer(
            &data,
            &frame_count,
            &raw_flags,
            nullptr,
            nullptr
        );
        if (FAILED(result)) {
            return Failure(
                ExitCodeForHresult(result, ExitCode::kCaptureFailure),
                result,
                "read WASAPI packet"
            );
        }

        const BufferFlagInfo flags = ClassifyBufferFlags(raw_flags);
        std::vector<float> samples;
        try {
            samples = CopyPacketSamples(data, frame_count, flags.silent);
        } catch (const std::exception&) {
            capture_client_->ReleaseBuffer(frame_count);
            return Failure(
                ExitCode::kInternalFailure,
                E_INVALIDARG,
                "copy WASAPI packet"
            );
        }

        result = capture_client_->ReleaseBuffer(frame_count);
        if (FAILED(result)) {
            return Failure(
                ExitCodeForHresult(result, ExitCode::kCaptureFailure),
                result,
                "release WASAPI packet"
            );
        }

        const std::size_t dropped = queue->Push(std::move(samples));
        ReportPacketDiagnostics(flags, dropped);
    }
}

void LoopbackCaptureSession::ReportPacketDiagnostics(
    const BufferFlagInfo& flags,
    std::size_t dropped
) {
    if (flags.data_discontinuity) {
        ++discontinuity_count_;
    }
    if (flags.timestamp_error) {
        ++timestamp_error_count_;
    }
    dropped_frame_count_ += dropped;

    if (!flags.data_discontinuity && !flags.timestamp_error && dropped == 0) {
        return;
    }

    const ULONGLONG now = GetTickCount64();
    if (last_diagnostic_tick_ != 0 && now - last_diagnostic_tick_ < 5'000) {
        return;
    }
    last_diagnostic_tick_ = now;
    std::fprintf(
        stderr,
        "%s diagnostics: discontinuities=%llu, timestamp_errors=%llu, "
        "dropped_live_frames=%llu.\n",
        diagnostic_label_,
        static_cast<unsigned long long>(discontinuity_count_),
        static_cast<unsigned long long>(timestamp_error_count_),
        static_cast<unsigned long long>(dropped_frame_count_)
    );
}

void LoopbackCaptureSession::Stop() {
    if (started_ && audio_client_) {
        audio_client_->Stop();
    }
    started_ = false;
    capture_client_.Reset();
    audio_client_.Reset();
    audio_ready_event_ = nullptr;
}

bool StartTransportAfterCaptureStart(
    const CaptureResult& startup_result,
    HANDLE stop_event,
    const std::function<void()>& start_transport
) {
    if (!startup_result.ok() ||
        WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        return false;
    }
    start_transport();
    return true;
}

void PrintNativeError(const CaptureResult& result) {
    const char* context = result.context != nullptr ? result.context : "native audio capture";
    std::fprintf(
        stderr,
        "Failed to %s (HRESULT 0x%08lX, category %d).\n",
        context,
        static_cast<unsigned long>(result.hresult),
        static_cast<int>(result.exit_code)
    );
}

}  // namespace buzz::windows_audio
