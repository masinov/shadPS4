// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "common/types.h"

namespace Vulkan {

enum class GpuCheckpoint : u8 {
    CommandBufferBegin,
    CommandBufferEnd,
    DirectDraw,
    IndirectDraw,
    DirectDispatch,
    IndirectDispatch,
    PredicationReduce,
    Detile,
    TileLinear,
    TileImageDispatch,
    FaultParse,
    BufferUpload,
    BufferDownload,
    BufferCopy,
    BufferMerge,
    Count,
};

struct GpuResourceContext {
    u64 guest_address{};
    u64 size{};
    u32 id{};
};

struct GpuCheckpointContext {
    static constexpr size_t MaxResources = 4;
    static constexpr size_t MaxShaderStages = 6;

    u64 pipeline_hash{};
    std::array<u64, MaxShaderStages> shader_hashes{};
    u64 source_guest_address{};
    u64 destination_guest_address{};
    u64 transfer_size{};
    u64 source_generation{};
    u64 destination_generation{};
    /// Byte offset of the transfer/argument range inside the source cache buffer.
    u64 source_offset{};
    u32 item_count{};
    /// Direct draws only. Indirect draws read their instance count from the argument buffer on
    /// the GPU; the CPU never observes it, so this stays zero and must not be reported as data.
    u32 instance_count{};
    /// Direct dispatches: the actual dimensions. Indirect dispatches: the argument values read
    /// from guest memory at record time; if the GPU writes them later (async-compute patched
    /// dispatches), the executed values may differ — record_time_groups marks that case.
    u32 group_x{};
    u32 group_y{};
    u32 group_z{};
    bool record_time_groups{};
    u16 writable_buffer_count{};
    u16 writable_image_count{};
    u8 color_attachment_count{};
    bool has_depth_attachment{};
    bool has_stencil_attachment{};
    bool uses_dma{};
    bool indexed{};
    bool predicated{};
    /// The draw/dispatch parameters live in a GPU argument buffer described by the source fields.
    bool indirect_arguments{};
    std::array<GpuResourceContext, MaxResources> writable_buffers{};
    std::array<GpuResourceContext, MaxResources> writable_images{};
};

struct GpuCheckpointRecord {
    u64 serial{};
    GpuCheckpoint checkpoint{};
    GpuCheckpointContext context{};
};

class GpuCheckpointTracker {
public:
    static constexpr size_t MaxRecords = 16'384;

    [[nodiscard]] const void* Record(GpuCheckpoint checkpoint,
                                     const GpuCheckpointContext& context = {}) const {
        std::scoped_lock lock{mutex};
        if (records.empty()) {
            records.resize(MaxRecords);
        }
        const u64 serial = ++next_serial;
        records[serial % records.size()] = GpuCheckpointRecord{serial, checkpoint, context};
        return EncodeMarker(serial, checkpoint);
    }

    [[nodiscard]] std::optional<GpuCheckpointRecord> Resolve(const void* marker) const {
        const auto decoded = DecodeMarker(marker);
        if (!decoded) {
            return std::nullopt;
        }
        const auto [serial, checkpoint] = *decoded;
        std::scoped_lock lock{mutex};
        if (records.empty()) {
            return std::nullopt;
        }
        const auto& record = records[serial % records.size()];
        if (record.serial != serial || record.checkpoint != checkpoint) {
            return std::nullopt;
        }
        return record;
    }

private:
    static constexpr u64 MarkerTypeBits = 4;
    static constexpr u64 MarkerTypeMask = (u64{1} << MarkerTypeBits) - 1;

    [[nodiscard]] static const void* EncodeMarker(u64 serial, GpuCheckpoint checkpoint) {
        const auto value = static_cast<std::uintptr_t>((serial << MarkerTypeBits) |
                                                       (static_cast<u64>(checkpoint) + 1));
        return reinterpret_cast<const void*>(value);
    }

    [[nodiscard]] static std::optional<std::pair<u64, GpuCheckpoint>> DecodeMarker(
        const void* marker) {
        const auto value = reinterpret_cast<std::uintptr_t>(marker);
        const u64 type = value & MarkerTypeMask;
        const u64 serial = value >> MarkerTypeBits;
        if (serial == 0 || type == 0 || type > static_cast<u64>(GpuCheckpoint::Count)) {
            return std::nullopt;
        }
        return std::pair{serial, static_cast<GpuCheckpoint>(type - 1)};
    }

    mutable std::mutex mutex;
    mutable std::vector<GpuCheckpointRecord> records;
    mutable u64 next_serial{};
};

enum class BufferAddressState : u8 {
    Live,
    Retired,
};

struct BufferAddressRecord {
    u64 generation{};
    u64 device_address{};
    u64 size{};
    u64 guest_address{};
    u64 allocation_size{};
    u64 retirement_sequence{};
    u64 last_use_tick{};
    u64 retirement_cpu_tick{};
    u64 retirement_scheduled_tick{};
    u32 usage{};
    u32 retirement_reason{};
    BufferAddressState state{};
};

struct BufferAddressMatch : BufferAddressRecord {
    u64 distance{};
    bool contains{};
};

class BufferAddressTracker {
public:
    static constexpr size_t MaxRecentRetiredRecords = 16'384;

    [[nodiscard]] u64 Register(u64 device_address, u64 size, u64 guest_address, u64 allocation_size,
                               u32 usage) const {
        if (device_address == 0 || size == 0) {
            return 0;
        }
        std::scoped_lock lock{mutex};
        const u64 generation = ++sequence;
        live_records.emplace(generation, BufferAddressRecord{
                                             .generation = generation,
                                             .device_address = device_address,
                                             .size = size,
                                             .guest_address = guest_address,
                                             .allocation_size = allocation_size,
                                             .usage = usage,
                                             .state = BufferAddressState::Live,
                                         });
        return generation;
    }

    void Retire(u64 generation, u64 last_use_tick = 0, u64 retirement_cpu_tick = 0,
                u64 retirement_scheduled_tick = 0, u32 retirement_reason = 0) const {
        if (generation == 0) {
            return;
        }
        std::scoped_lock lock{mutex};
        const auto it = live_records.find(generation);
        if (it == live_records.end()) {
            ++missing_retirements;
            return;
        }
        BufferAddressRecord retired = it->second;
        retired.state = BufferAddressState::Retired;
        retired.retirement_sequence = ++sequence;
        retired.last_use_tick = last_use_tick;
        retired.retirement_cpu_tick = retirement_cpu_tick;
        retired.retirement_scheduled_tick = retirement_scheduled_tick;
        retired.retirement_reason = retirement_reason;
        if (recent_retired.empty()) {
            recent_retired.resize(MaxRecentRetiredRecords);
        }
        recent_retired[recent_retired_cursor] = retired;
        recent_retired_cursor = (recent_retired_cursor + 1) % recent_retired.size();
        live_records.erase(it);
    }

    [[nodiscard]] std::vector<BufferAddressMatch> FindAll(u64 address,
                                                          size_t max_matches = 8) const {
        std::scoped_lock lock{mutex};
        std::vector<BufferAddressMatch> matches;
        matches.reserve(max_matches);
        const auto better = [](const auto& lhs, const auto& rhs) {
            if (lhs.contains != rhs.contains) {
                return lhs.contains > rhs.contains;
            }
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            if (lhs.generation != rhs.generation) {
                return lhs.generation > rhs.generation;
            }
            return lhs.state == BufferAddressState::Live &&
                   rhs.state == BufferAddressState::Retired;
        };
        const auto consider = [&](const BufferAddressRecord& record) {
            if (record.generation == 0) {
                return;
            }
            const bool contains =
                address >= record.device_address && address - record.device_address < record.size;
            const u64 distance =
                contains ? 0 : DistanceToRange(address, record.device_address, record.size);
            const BufferAddressMatch candidate{record, distance, contains};
            if (matches.size() < max_matches) {
                matches.push_back(candidate);
                return;
            }
            if (matches.empty()) {
                return;
            }
            size_t worst = 0;
            for (size_t i = 1; i < matches.size(); ++i) {
                if (better(matches[worst], matches[i])) {
                    worst = i;
                }
            }
            if (better(candidate, matches[worst])) {
                matches[worst] = candidate;
            }
        };
        // Live records must never be evicted by diagnostic churn. Only completed lifetime history
        // is bounded, so a fault can always be correlated with every allocation that still exists.
        for (const auto& [generation, record] : live_records) {
            consider(record);
        }
        for (const auto& record : recent_retired) {
            consider(record);
        }
        std::ranges::sort(matches, better);
        return matches;
    }

    [[nodiscard]] size_t LiveCount() const {
        std::scoped_lock lock{mutex};
        return live_records.size();
    }

    [[nodiscard]] u64 MissingRetirementCount() const {
        std::scoped_lock lock{mutex};
        return missing_retirements;
    }

private:
    [[nodiscard]] static constexpr u64 SaturatingEnd(u64 base, u64 size) noexcept {
        return size > std::numeric_limits<u64>::max() - base ? std::numeric_limits<u64>::max()
                                                             : base + size;
    }

    [[nodiscard]] static constexpr u64 DistanceToRange(u64 address, u64 base, u64 size) noexcept {
        if (address < base) {
            return base - address;
        }
        const u64 end = SaturatingEnd(base, size);
        return address >= end ? address - end : 0;
    }

    mutable std::mutex mutex;
    mutable std::map<u64, BufferAddressRecord> live_records;
    mutable std::vector<BufferAddressRecord> recent_retired;
    mutable size_t recent_retired_cursor{};
    mutable u64 sequence{};
    mutable u64 missing_retirements{};
};

} // namespace Vulkan
