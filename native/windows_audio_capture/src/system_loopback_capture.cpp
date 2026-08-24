#include "system_loopback_capture.h"

#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

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

SystemLoopbackCapture::~SystemLoopbackCapture() {
    Stop();
}

HRESULT SystemLoopbackCapture::ActivateAndInitialize(DWORD stream_flags) {
    Microsoft::WRL::ComPtr<IAudioClient> candidate;
    HRESULT result = device_->Activate(
        __uuidof(IAudioClient),
        CLSCTX_ALL,
        nullptr,
        reinterpret_cast<void**>(candidate.GetAddressOf())
    );
    if (FAILED(result)) {
        return result;
    }

    WAVEFORMATEX target_format = BuildTargetFormat();
    result = candidate->Initialize(
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

    audio_client_ = std::move(candidate);
    return S_OK;
}

CaptureResult SystemLoopbackCapture::InitializeAndStart(HANDLE audio_ready_event) {
    audio_ready_event_ = audio_ready_event;

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
        audio_client_.Reset();

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

    result = audio_client_->GetService(
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
            "start system loopback capture"
        );
    }
    started_ = true;
    return {};
}

CaptureResult SystemLoopbackCapture::Capture(
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

CaptureResult SystemLoopbackCapture::DrainPackets(BoundedPcmQueue* queue) {
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

void SystemLoopbackCapture::ReportPacketDiagnostics(
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
        "System audio diagnostics: discontinuities=%llu, timestamp_errors=%llu, "
        "dropped_live_frames=%llu.\n",
        static_cast<unsigned long long>(discontinuity_count_),
        static_cast<unsigned long long>(timestamp_error_count_),
        static_cast<unsigned long long>(dropped_frame_count_)
    );
}

void SystemLoopbackCapture::Stop() {
    if (started_ && audio_client_) {
        audio_client_->Stop();
    }
    started_ = false;
    capture_client_.Reset();
    audio_client_.Reset();
    device_.Reset();
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
