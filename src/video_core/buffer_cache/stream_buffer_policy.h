// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include "common/types.h"
#include "video_core/memory_gc.h"

namespace VideoCore {

constexpr int StreamLeapThreshold = 16;
constexpr int StreamingScore = StreamLeapThreshold + 1;
constexpr VAddr StreamGrowthMinimum = 2_MB;
constexpr VAddr StreamGrowthMaximum = 16_MB;
constexpr u32 StreamGrowthLookahead = 4;
// Proportional reserve for large streaming ranges. Every replacement of an N-byte buffer costs an
// N-byte allocation plus an N-byte copy while the old allocation is still resident, so the number
// of replacements must fall as N grows or the copy volume becomes quadratic in the streamed size
// (Run 20: 117 GiB allocated for 25 GiB requested). One eighth of the natural size, capped at
// 64 MiB, keeps the tail small relative to the buffer while cutting large-buffer replacements to a
// quarter of the fixed 16 MiB reserve.
constexpr VAddr StreamGrowthProportionalMaximum = 64_MB;
constexpr u32 StreamGrowthProportionShift = 3;
// Sparse cache buffers bind reserve pages only when they are first touched, so their reserve costs
// address space rather than memory. It is still kept modest: a grow replacement under sparse is
// cheap (new buffer + aliased blocks, no copy), while a huge reserve inflates every later span
// (Run 29: a 24 KB request resolved to 306 MB) and widens the window of registered-but-unbound
// pages whose device addresses are null.
constexpr VAddr SparseStreamGrowthMaximum = 64_MB;

[[nodiscard]] constexpr VAddr SparseStreamGrowthDistance(VAddr classic_distance,
                                                         VAddr natural_size) noexcept {
    return std::max(classic_distance, std::min(natural_size, SparseStreamGrowthMaximum));
}

struct StreamGrowthDirections {
    bool left;
    bool right;
};

[[nodiscard]] constexpr StreamGrowthDirections DetermineStreamGrowth(VAddr requested_begin,
                                                                     VAddr requested_end,
                                                                     VAddr overlap_begin,
                                                                     VAddr overlap_end) noexcept {
    return {
        .left = requested_begin<overlap_begin, .right = requested_end> overlap_end,
    };
}

// Reserve several observed outward steps so a streaming workload does not replace and copy the
// entire buffer on every request. Both the minimum and maximum are deliberate: small strides still
// receive useful capacity, while a single unusual request cannot retain an unbounded VRAM tail.
[[nodiscard]] constexpr VAddr DetermineStreamGrowthDistance(VAddr request_size,
                                                            VAddr outward_extension,
                                                            VAddr natural_size = 0) noexcept {
    const VAddr bounded_extension =
        std::min(outward_extension, StreamGrowthMaximum / StreamGrowthLookahead);
    const VAddr lookahead = bounded_extension * StreamGrowthLookahead;
    const VAddr stride_reserve =
        std::clamp(std::max(request_size, lookahead), StreamGrowthMinimum, StreamGrowthMaximum);
    const VAddr proportional_reserve =
        std::min(natural_size >> StreamGrowthProportionShift, StreamGrowthProportionalMaximum);
    return std::max(stride_reserve, proportional_reserve);
}

// Once a range has crossed the streaming threshold, a small amount of spare capacity amortizes a
// full replacement and copy on every outward step. Under pressure, reserve it only when the spare
// capacity is no larger than the natural replacement itself. This keeps tiny ranges from doubling
// their footprint while preserving look-ahead for the 100+ MiB replacement chains where a bounded
// 2--16 MiB tail prevents far more transient allocation and copying than it costs.
[[nodiscard]] constexpr bool ShouldReserveStreamGrowth(u64 used_memory, u64 total_budget,
                                                       u64 natural_size,
                                                       u64 desired_growth) noexcept {
    if (desired_growth == 0) {
        return false;
    }
    return AllocationReclaimTarget(used_memory, total_budget, natural_size, false) == 0 ||
           desired_growth <= natural_size;
}

// Replaced allocations cannot be destroyed until the command buffer containing their copy has
// completed. Bound how much old storage one recording tick may accumulate before advancing the
// timeline. The allocator headroom is an appropriate dynamic limit (128--256 MiB on normal GPUs).
// This is deliberately independent of current memory pressure: the retained old versions are
// what creates the pressure, and a submission is far cheaper than a later emergency collection.
[[nodiscard]] constexpr bool ShouldAdvanceReplacementChain(u64 deferred_bytes,
                                                           u64 next_overlap_bytes,
                                                           u64 total_budget) noexcept {
    if (deferred_bytes == 0 || next_overlap_bytes == 0) {
        return false;
    }
    const u64 limit = AllocationHeadroom(total_budget);
    return deferred_bytes > limit - std::min(limit, next_overlap_bytes);
}

[[nodiscard]] constexpr VAddr GrowStreamBegin(VAddr begin, VAddr distance, VAddr minimum) noexcept {
    const VAddr available = begin > minimum ? begin - minimum : 0;
    return begin - std::min(distance, available);
}

[[nodiscard]] constexpr VAddr GrowStreamEnd(VAddr end, VAddr distance, VAddr maximum) noexcept {
    const VAddr available = end < maximum ? maximum - end : 0;
    return end + std::min(distance, available);
}

[[nodiscard]] constexpr int AccumulateStreamScore(int score, int addend) noexcept {
    const int bounded_score = std::clamp(score, 0, StreamingScore);
    const int bounded_addend = std::clamp(addend, 0, StreamingScore);
    return std::min(StreamingScore, bounded_score + bounded_addend);
}

[[nodiscard]] constexpr int ReplacementStreamScore(int overlap_score, std::size_t overlap_count,
                                                   bool made_stream_leap) noexcept {
    if (made_stream_leap) {
        return StreamingScore;
    }
    return AccumulateStreamScore(
        overlap_score, static_cast<int>(std::min<std::size_t>(overlap_count, StreamingScore)));
}

} // namespace VideoCore
