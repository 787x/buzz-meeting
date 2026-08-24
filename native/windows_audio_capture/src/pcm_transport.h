#pragma once

#include <Windows.h>
#include <audioclient.h>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace buzz::windows_audio {

constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::uint32_t kSampleRate = 16'000;
constexpr std::uint16_t kChannelCount = 1;
constexpr std::uint16_t kSampleFormatFloat32LittleEndian = 1;
constexpr std::size_t kProtocolHeaderSize = 16;
constexpr std::size_t kQueueCapacityFrames = kSampleRate * 2;

using ProtocolHeader = std::array<std::uint8_t, kProtocolHeaderSize>;

ProtocolHeader SerializeProtocolHeader();
bool WriteAll(HANDLE handle, const void* data, std::size_t byte_count);

struct BufferFlagInfo {
    bool silent;
    bool data_discontinuity;
    bool timestamp_error;
};

BufferFlagInfo ClassifyBufferFlags(DWORD flags);
std::vector<float> CopyPacketSamples(
    const BYTE* data,
    std::size_t frame_count,
    bool silent
);

class BoundedPcmQueue {
public:
    explicit BoundedPcmQueue(std::size_t capacity_frames);

    std::size_t Push(std::vector<float> samples);
    bool Pop(std::vector<float>* samples);
    void Close();

    std::size_t capacity_frames() const;
    std::size_t queued_frames() const;
    std::uint64_t dropped_frames() const;

private:
    const std::size_t capacity_frames_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<std::vector<float>> chunks_;
    std::size_t queued_frames_ = 0;
    std::uint64_t dropped_frames_ = 0;
    bool closed_ = false;
};

}  // namespace buzz::windows_audio
