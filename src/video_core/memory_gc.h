// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "common/types.h"
#include "common/unique_function.h"

namespace VideoCore {

enum class GcPressure : u8 {
    None,
    High,
    Critical,
};

struct GcBudget {
    u64 bytes_remaining{};
    u64 completed_tick{};
    u32 objects_remaining{};
    GcPressure pressure{GcPressure::None};
    bool collect_retirements{};
    bool require_completed{};
    /// A real allocation failure: every reclaimable byte matters, even tiny objects.
    bool emergency{};
    /// Usage exceeds the budget itself (not merely the headroom): evict warmer objects.
    bool overshoot{};

    [[nodiscard]] bool Active() const noexcept {
        return pressure != GcPressure::None && bytes_remaining != 0 && objects_remaining != 0;
    }

    void Reclaim(u64 bytes) noexcept {
        bytes_remaining -= std::min(bytes_remaining, bytes);
        --objects_remaining;
    }
};

struct GcResult {
    u64 reclaimed_bytes{};
    u64 latest_use_tick{};
    u32 inspected_objects{};
    u32 evicted_objects{};
    u32 skipped_gpu_modified{};
    u32 skipped_gpu_modified_unique{};
    u32 skipped_bound{};
    u32 skipped_in_flight{};
    u32 skipped_pending{};
    u32 skipped_unallocated{};
    u32 skipped_small{};
    std::vector<Common::UniqueFunction<void>> retirements;

    void QueueRetirement(u64 use_tick, Common::UniqueFunction<void>&& retirement) {
        latest_use_tick = std::max(latest_use_tick, use_tick);
        retirements.emplace_back(std::move(retirement));
    }

    /// Physically releases the selected resources. Returns how many retirements executed.
    u32 ExecuteRetirements() {
        const u32 count = static_cast<u32>(retirements.size());
        for (auto& retirement : retirements) {
            retirement();
        }
        retirements.clear();
        return count;
    }
};

// Eviction ages are measured in guest submissions (Rasterizer::OnSubmit advances both caches'
// epochs once per submission). They must not be measured in collector invocations: allocation
// preflight can run the collector many times inside one frame, and an epoch that advances per
// call makes a resource used every frame look "old" between two draws of the same frame. That
// evicts hot streaming buffers which are then rebuilt piecewise and re-merged with full copies.
[[nodiscard]] constexpr u64 GcMinimumAge(GcPressure pressure, bool overshoot = false) noexcept {
    // Epochs advance once per guest submission (tens per second). The previous 8/32 made assets
    // idle for a fraction of a second evictable; Run 31 showed that as visible eviction/re-upload
    // bursts once the collector could actually reach large objects. Roughly two / eight seconds.
    // Once usage actually exceeds the budget the calculus changes: leaving 1.7 GiB of targets
    // unmet (Run 32 tail) risks driver residency stalls and TDR, so accept warmer evictions.
    if (overshoot) {
        return 16;
    }
    return pressure == GcPressure::Critical ? 64 : 256;
}

// CurrentTick is the signal value reserved for the command buffer still being recorded. A
// resource referenced by that tick cannot be physically reclaimed without flushing and waiting
// for all work recorded so far.
[[nodiscard]] constexpr bool IsResourceInFlight(u64 last_use_tick, u64 current_tick) noexcept {
    return last_use_tick >= current_tick;
}

// A non-forced allocation pass must never unlink a resource that would require the command
// processor to wait. Timeline values are monotonic, so a sampled completed tick remains safe even
// if it becomes stale while the collector runs.
[[nodiscard]] constexpr bool CanReclaimWithoutWait(u64 last_use_tick, u64 completed_tick) noexcept {
    return last_use_tick <= completed_tick;
}

// Lifetime ticks are monotonic. Keeping this operation separate from eviction recency lets callers
// record a GPU reference without making a potentially huge set of resources artificially hot in
// the LRU policy.
[[nodiscard]] constexpr u64 RecordResourceUse(u64 last_use_tick, u64 use_tick) noexcept {
    return std::max(last_use_tick, use_tick);
}

// Diagnostic events in allocation paths can occur tens of thousands of times under pressure.
// Keep the beginning of a sequence for context, then sample exponentially so logging cannot become
// a material part of the workload.
[[nodiscard]] constexpr bool ShouldLogDiagnosticSample(u64 event_count) noexcept {
    return event_count != 0 && (event_count <= 8 || (event_count & (event_count - 1)) == 0);
}

// Evicting an object far smaller than the reclaim target is pure churn: it frees almost nothing
// and its recreation costs an allocation (and, for sparse buffers, a queue bind) within seconds.
// Run 28: periodic GC at critical pressure chased a ~500 MiB target by evicting ~24 tiny sparse
// buffers (2-3 MiB total) per pass, which the game recreated immediately (~30 binds/s sustained).
// Automatic collection skips such objects; only an actual allocation failure may take them.
constexpr u64 GcMinimumAutomaticEvictionBytes = 2_MB;

[[nodiscard]] constexpr bool ShouldSkipSmallEviction(u64 allocation_size, bool emergency) noexcept {
    return !emergency && allocation_size < GcMinimumAutomaticEvictionBytes;
}

// Skipping a small object is a size comparison; it must not consume the same budget as a real
// inspection or thousands of tiny objects at the old end of the LRU stop the collector from ever
// reaching the large ones behind them (Run 30: ~2,600 of ~2,640 inspections were small-skips and
// no texture was evicted all session). The cheap-scan cap only bounds worst-case list walking.
constexpr u32 GcMaxCheapScans = 16384;

/// Minimum age, in GC epochs, of a sparse block's last synchronized access before the collector
/// may unbind it. Conservative: BDA direct-memory accesses do not stamp epochs, so the age must
/// comfortably exceed any realistic re-reference distance; a wrong drop self-heals through the
/// fault path but costs a transient zero read.
[[nodiscard]] constexpr u64 SparseBlockReclaimMinAge(GcPressure pressure,
                                                     bool overshoot = false) noexcept {
    if (pressure == GcPressure::Critical || overshoot) {
        return 96;
    }
    return 256;
}

// Keep enough room for the allocation itself plus a small amount of driver bookkeeping. The
// driver's heap budget can move between queries, so targeting its exact edge still races other
// processes and allocations made internally by the driver.
[[nodiscard]] constexpr u64 AllocationHeadroom(u64 total_budget) noexcept {
    return std::clamp(total_budget / 20, 128_MB, 256_MB);
}

// Returns the number of bytes that should be reclaimed before attempting an allocation. A forced
// request follows an actual allocation failure and therefore asks for at least the allocation size
// even when the last usage query appeared to have enough room (fragmentation and stale driver
// accounting can otherwise make the identical retry fail again).
[[nodiscard]] constexpr u64 AllocationReclaimTarget(u64 used_memory, u64 total_budget,
                                                    u64 allocation_size, bool force) noexcept {
    const u64 headroom = AllocationHeadroom(total_budget);
    const u64 safe_limit = total_budget > headroom ? total_budget - headroom : 0;
    const u64 projected_usage = used_memory > std::numeric_limits<u64>::max() - allocation_size
                                    ? std::numeric_limits<u64>::max()
                                    : used_memory + allocation_size;
    u64 target = projected_usage > safe_limit ? projected_usage - safe_limit : 0;
    if (force) {
        target = std::max(target, allocation_size);
    }
    return std::min(target, used_memory);
}

} // namespace VideoCore
