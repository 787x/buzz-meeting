#include "pcm_transport.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace buzz::windows_audio {
namespace {

void StoreU16(ProtocolHeader* header, std::size_t offset, std::uint16_t value) {
    (*header)[offset] = static_cast<std::uint8_t>(value & 0xffU);
    (*header)[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
}

void StoreU32(ProtocolHeader* header, std::size_t offset, std::uint32_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        (*header)[offset + byte] =
            static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU);
    }
}

}  // namespace

ProtocolHeader SerializeProtocolHeader() {
    ProtocolHeader header{};
    header[0] = 'B';
    header[1] = 'Z';
    header[2] = 'W';
    header[3] = 'A';
    StoreU16(&header, 4, kProtocolVersion);
    StoreU16(&header, 6, static_cast<std::uint16_t>(header.size()));
    StoreU32(&header, 8, kSampleRate);
    StoreU16(&header, 12, kChannelCount);
    StoreU16(&header, 14, kSampleFormatFloat32LittleEndian);
    return header;
}

bool WriteAll(HANDLE handle, const void* data, std::size_t byte_count) {
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written_total = 0;
    while (written_total < byte_count) {
        const std::size_t remaining = byte_count - written_total;
        const DWORD write_size = static_cast<DWORD>(std::min<std::size_t>(
            remaining,
            std::numeric_limits<DWORD>::max()
        ));
        DWORD written = 0;
        if (!WriteFile(
                handle,
                bytes + written_total,
                write_size,
                &written,
                nullptr
            ) || written == 0) {
            return false;
        }
        written_total += written;
    }
    return true;
}

BufferFlagInfo ClassifyBufferFlags(DWORD flags) {
    return {
        (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0,
        (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) != 0,
        (flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR) != 0,
    };
}

std::vector<float> CopyPacketSamples(
    const BYTE* data,
    std::size_t frame_count,
    bool silent
) {
    std::vector<float> samples(frame_count, 0.0F);
    if (!silent && frame_count != 0) {
        if (data == nullptr) {
            throw std::invalid_argument("A non-silent packet has no sample data");
        }
        std::memcpy(samples.data(), data, frame_count * sizeof(float));
    }
    return samples;
}

BoundedPcmQueue::BoundedPcmQueue(std::size_t capacity_frames)
    : capacity_frames_(capacity_frames) {
    if (capacity_frames == 0) {
        throw std::invalid_argument("PCM queue capacity must be positive");
    }
}

std::size_t BoundedPcmQueue::Push(std::vector<float> samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return samples.size();
    }

    std::size_t dropped = 0;
    if (samples.size() > capacity_frames_) {
        const std::size_t trim = samples.size() - capacity_frames_;
        samples.erase(samples.begin(), samples.begin() + trim);
        dropped += trim;
    }

    while (!chunks_.empty() && queued_frames_ + samples.size() > capacity_frames_) {
        dropped += chunks_.front().size();
        queued_frames_ -= chunks_.front().size();
        chunks_.pop_front();
    }

    if (!samples.empty()) {
        queued_frames_ += samples.size();
        chunks_.push_back(std::move(samples));
        available_.notify_one();
    }
    dropped_frames_ += dropped;
    return dropped;
}

bool BoundedPcmQueue::Pop(std::vector<float>* samples) {
    if (samples == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this]() { return closed_ || !chunks_.empty(); });
    if (chunks_.empty()) {
        return false;
    }

    *samples = std::move(chunks_.front());
    queued_frames_ -= samples->size();
    chunks_.pop_front();
    return true;
}

void BoundedPcmQueue::Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    available_.notify_all();
}

std::size_t BoundedPcmQueue::capacity_frames() const {
    return capacity_frames_;
}

std::size_t BoundedPcmQueue::queued_frames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_frames_;
}

std::uint64_t BoundedPcmQueue::dropped_frames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_frames_;
}

}  // namespace buzz::windows_audio
