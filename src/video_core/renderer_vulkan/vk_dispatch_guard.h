// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

/// Execution-time clamp for indirect dispatch arguments. The arguments live in GPU-written
/// guest buffers, so record-time inspection cannot see the values the dispatch will consume;
/// out-of-range group counts hang the device (the recurring indirect-dispatch TDR class). A
/// one-workgroup pre-pass copies the arguments into a private ring, clamped to the device
/// limit, and counts every clamp into a host-visible counter. Sane arguments pass through
/// byte-identical, so the guard is behavior-preserving for correct content.
class DispatchGuard {
public:
    explicit DispatchGuard(const Instance& instance, Scheduler& scheduler);
    ~DispatchGuard();

    struct ClampedArgs {
        vk::Buffer buffer;
        u64 offset;
    };

    /// Records the clamp pre-pass for the arguments at src_buffer+src_offset and returns the
    /// ring location the indirect dispatch must source instead. Must be called outside a render
    /// pass, before the caller binds its own compute pipeline.
    ClampedArgs Clamp(vk::CommandBuffer cmdbuf, vk::Buffer src_buffer, u64 src_offset,
                      VAddr guest_address);

    /// Same for the four argument words of a non-indexed indirect draw: a vertex or instance
    /// count out of range zeroes the draw. Must be called outside a render pass, before the
    /// caller binds its descriptors and push constants.
    ClampedArgs ClampDraw(vk::CommandBuffer cmdbuf, vk::Buffer src_buffer, u64 src_offset,
                          VAddr guest_address);

    /// Total clamp events observed so far (host-visible counter, no synchronization: the value
    /// trails the GPU slightly, which is fine for statistics).
    [[nodiscard]] u32 ClampCount() const;
    [[nodiscard]] u32 DrawClampCount() const;
    /// Number of draws routed through ClampDraw.
    [[nodiscard]] u64 GuardedDraws() const {
        return guarded_draws;
    }

    /// Logs the arguments the most recent guarded draws and dispatches consumed, newest first.
    /// The ring is host-visible, so this still answers after the device is lost.
    void LogRecent() const;

private:
    enum class Kind : u8 { None, Dispatch, Draw };
    struct SlotInfo {
        VAddr guest_address{};
        Kind kind{Kind::None};
    };

    ClampedArgs Record(vk::CommandBuffer cmdbuf, vk::Pipeline clamp_pipeline,
                       vk::PipelineLayout layout, vk::Buffer src_buffer, u64 src_offset,
                       u32 num_words, u64 counter_offset, Kind kind, VAddr guest_address);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::Buffer args_ring;
    VideoCore::Buffer clamp_counter;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout draw_pipeline_layout;
    vk::UniquePipeline draw_pipeline;
    // Not "slots": Qt defines that as an empty macro, and in a translation unit that includes Qt
    // the member silently vanishes, shrinking the class (and Rasterizer) by 24 bytes.
    std::vector<SlotInfo> slot_info;
    u64 ring_cursor = 0;
    u64 guarded_draws = 0;
};

} // namespace Vulkan
