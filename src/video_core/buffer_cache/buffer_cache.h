// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>

#include <array>
#include <span>
#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "common/enum.h"
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/bda_address_policy.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/memory_gc.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/texture_cache/image.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class MemoryTracker;
class PageManager;

enum class ObtainBufferFlags {
    None = 0,
    IsWritten = 1 << 0,
    IsTexelBuffer = 1 << 1,
    IgnoreStreamBuffer = 1 << 2,
    InvalidateTextureCache = 1 << 3,
    AvoidTextureGc = 1 << 4,
};
DECLARE_ENUM_FLAG_OPERATORS(ObtainBufferFlags)

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    /// Sparse buffer block size on every known implementation; sparse cache buffers are aligned
    /// to it so blocks can be re-bound between buffers at identical guest addresses.
    static constexpr u64 SPARSE_ALIGNMENT = 64_KB;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (BdaAddressSpaceBits - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

    struct PageData {
        BufferId buffer_id{};
    };

    struct DebugPageState {
        u32 buffer_id{Common::SlotId::INVALID_INDEX};
        VAddr buffer_begin{};
        VAddr buffer_end{};
        bool registered{};
        bool tracker_exists{};
        bool cpu_modified{};
        bool gpu_modified{};
        bool gpu_pending{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = BdaAddressSpaceBits;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    /// Host vertex input state resolved for one draw. Resolution (PrepareVertexBuffers) may
    /// allocate cache buffers and therefore flush the scheduler; recording (BindVertexBuffers)
    /// never does. Keeping the two apart guarantees the bound state lands in the same command
    /// buffer as the draw that consumes it.
    struct VertexBufferBinding {
        Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
        Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
        Vulkan::VertexInputs<vk::Buffer> host_buffers;
        Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
        Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
        Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    };

    struct IndexBufferBinding {
        vk::Buffer buffer{};
        vk::DeviceSize offset{};
        vk::IndexType index_type{vk::IndexType::eUint16};
    };

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        int stream_score = 0;
        bool has_stream_leap = false;
        bool stream_growth_suppressed = false;
        u64 desired_stream_growth = 0;
        u64 speculative_bytes = 0;
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves GPU modified ranges since last CPU fence that haven't been read protected yet.
    [[nodiscard]] RangeSet& GetPendingGpuModifiedRanges() {
        return gpu_modified_ranges_pending;
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool download);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    /// Flushes GPU modified ranges of the uncovered part of the edge pages of an image.
    void ReadEdgeImagePages(const Image& image);

    /// Resolves host vertex buffers for the current draw. May allocate cache buffers and flush
    /// the scheduler, so it must run before any state of the draw is recorded.
    [[nodiscard]] VertexBufferBinding PrepareVertexBuffers(
        const Vulkan::GraphicsPipeline& pipeline,
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Records the resolved vertex input state into the current command buffer. Never flushes.
    void BindVertexBuffers(const VertexBufferBinding& binding);

    /// Resolves the host index buffer for the current draw. May allocate cache buffers and flush
    /// the scheduler, so it must run before any state of the draw is recorded.
    [[nodiscard]] IndexBufferBinding PrepareIndexBuffer(
        u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Records the resolved index buffer into the current command buffer. Never flushes.
    void BindIndexBuffer(const IndexBufferBinding& binding);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(
        VAddr gpu_addr, u32 size, ObtainBufferFlags flags = ObtainBufferFlags::None,
        BufferId buffer_id = {});

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Returns non-mutating cache/tracker state for repeated-fault diagnostics.
    [[nodiscard]] DebugPageState GetDebugPageState(VAddr addr);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Mark a region as CPU-modified so that subsequent SynchronizeBuffer picks it up.
    /// Backdoor for external paths (e.g. storage image sync) that write guest memory
    /// without going through the buffer cache's own ObtainBuffer/WriteDataBuffer.
    void MarkRegionAsCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Mark region as modified from the GPU
    void MarkRegionAsGpuModified(VAddr addr, size_t size);

    /// Marks a region as CPU-authoritative after its GPU contents were written back to guest
    /// memory (eviction readback). Guest memory becomes the single source of truth again.
    void MarkRegionAsFlushed(VAddr addr, size_t size);

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size, bool allow_texture_gc = true);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Record memory barrier. Used for buffers when accessed via BDA.
    void MemoryBarrier();

    /// Processes ready preemptive downloads not consumed by the guest.
    void ProcessPreemptiveDownloads();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size, bool is_written = false);

    /// Advances the eviction epoch. Called once per guest submission by the rasterizer.
    void AdvanceGcEpoch();

    /// Reclaims CPU-authoritative buffers without waiting for GPU readbacks.
    [[nodiscard]] GcResult RunGarbageCollector(GcBudget& budget);

    /// Unbinds cold, CPU-authoritative 64 KiB blocks of live sparse buffers. Called by the
    /// garbage collector when whole-buffer eviction leaves the budget unmet.
    void SweepColdSparseBlocks(GcBudget& budget, GcResult& result);

    /// Ghost bookkeeping for demand binds: measures reclaim->rebind distances and protects the
    /// rebound blocks. Only the binds newly created by the current call may be passed.
    void NoteDemandBinds(Buffer& buffer, std::span<const vk::SparseMemoryBind> binds);

    /// Installs the rasterizer-owned shared collector used by pressured allocations.
    void SetAllocationReclaimCallback(std::function<void(u64, u64, bool, bool)> callback) {
        allocation_reclaim_callback = std::move(callback);
    }

    /// Notifies memory tracker of GPU modified ranges from the last CPU fence.
    void CommitPendingGpuRanges();

    struct Statistics {
        u64 replacements{};            ///< CreateBuffer calls
        u64 stale_slot_rebinds{};      ///< binding hints that pointed at a reused slot (raced GC)
        u64 vertex_residency_misses{}; ///< vertex bindings whose V# extent was not resident
        u64 takeover_mismatches{};     ///< sparse merges where taken bytes != absorbed bytes
        u64 replacements_new{};        ///< ... with no overlap
        u64 replacements_grow{};       ///< ... with exactly one overlap
        u64 replacements_bridge{};     ///< ... with two or more overlaps
        u64 replaced_bytes{};          ///< bytes of absorbed buffers (copied or aliased)
        u64 demand_bindings{};         ///< sparse: EnsureRangeBound calls that bound something
        u64 demand_bound_bytes{};
        u64 live_buffers{};
        u64 bound_bytes{};    ///< sum of AllocationSizeBytes over live buffers
        u64 block_reclaims{}; ///< sparse: cold blocks unbound by the GC sweep
        u64 block_reclaimed_bytes{};
        u64 block_reclaim_age{}; ///< sparse: current adaptive age threshold, in GC epochs
        u64 reinstatements{};    ///< sparse: victim-stage blocks recalled for free
        u64 reinstated_bytes{};
        u64 limbo_rescued{}; ///< sparse: expired blocks kept because GPU data arrived in limbo
        u64 limbo_rescued_bytes{};
        u64 limbo_bytes{}; ///< sparse: bytes currently in the victim stage (still resident)
        u64 ghost_hits{};  ///< sparse: demand binds that re-bound a reclaimed range
        u64 ghost_hit_bytes{};
        u64 ghost_live{};                ///< sparse: ghost entries currently tracked
        std::array<u64, 6> ghost_dist{}; ///< re-reference distance histogram:
                                         ///< <256, <512, <1024, <2048, <4096, >=4096 epochs
        bool sparse{};
    };

    /// Cumulative counters plus a snapshot of live cache contents (O(live buffers)).
    [[nodiscard]] Statistics GetStatistics();

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    template <bool async>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write);
    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size, u64 used_memory,
                                                u64 total_budget);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                     std::vector<vk::SparseMemoryBind>& sparse_binds);

    /// Writes device addresses for [offset, offset + size) of the buffer into the BDA page table.
    void WriteBdaEntries(const Buffer& buffer, u64 offset, u64 size);

    /// Sparse buffers: binds memory for a range about to be accessed and publishes its BDA.
    void EnsureRangeBound(Buffer& buffer, VAddr device_addr, u64 size);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size, bool allow_texture_gc = true);

    void ReclaimForAllocation(u64 allocation_size, bool force, bool allow_texture_gc);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    vk::Buffer UploadCopies(const Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    bool SynchronizeBufferFromImage(const Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void MarkBufferUsed(Buffer& buffer);

    void TouchBuffer(Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id, GcResult* gc_result = nullptr,
                      BufferRetirementReason reason = BufferRetirementReason::CacheReplacement);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer> slot_buffers;
    u64 gc_tick = 0;
    u64 vertex_residency_log_count = 0;
    // Reclaim -> rebind feedback: Run 43 released 20.9 GB of cold blocks and demanded 20.7 GB
    // straight back (worst bind 1.9 s). The age threshold adapts to the measured ratio instead
    // of guessing a constant: sustained rebinding doubles it, sustained headroom halves it.
    u64 adaptive_block_age = 256;
    u64 adapt_reclaimed_snapshot = 0;
    u64 adapt_demand_snapshot = 0;
    u32 adapt_raise_streak = 0;
    u32 adapt_lower_streak = 0;
    // Ghost accounting of reclaimed ranges (metadata only): 64 KiB page -> reclaim epoch. A
    // demand bind that hits a ghost measures the range's true re-reference distance; the
    // rebound blocks are then individually exempted for twice that distance, so the hot subset
    // stops round-tripping without raising the global threshold for genuinely cold memory.
    tsl::robin_map<u64, u64> reclaim_ghosts;
    bool sparse_buffers{};
    u64 pressure_allocation_log_count{};
    u64 chain_advance_log_count{};
    u64 grow_log_count{};
    u64 replacement_count{};
    Statistics stats{};
    u64 replacement_chain_tick{};
    u64 replacement_chain_deferred_bytes{};
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    RangeSet gpu_modified_ranges;
    RangeSet gpu_modified_ranges_pending;
    struct PreemptiveDownload {
        VAddr device_addr;
        u64 size;
        u8* staging;
        u64 done_tick;

        auto operator<=>(const PreemptiveDownload&) const = default;
    };
    SplitRangeMap<PreemptiveDownload> preemptive_downloads;
    using BufferCopies = boost::container::small_vector<vk::BufferCopy, 8>;
    tsl::robin_map<BufferId, BufferCopies> preemptive_copies;
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
    std::function<void(u64, u64, bool, bool)> allocation_reclaim_callback;
};

} // namespace VideoCore
