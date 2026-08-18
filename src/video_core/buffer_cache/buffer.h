// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/sparse_coverage.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

enum class BufferRetirementReason : u32 {
    Unknown,
    CacheReplacement,
    GarbageCollection,
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

/// One contiguous memory range bound into a sparse buffer. Blocks are moved between buffers by
/// aliasing (bind the same memory into the successor); only one buffer owns the allocation.
struct SparseBlock {
    VmaAllocation allocation{};
    VkDeviceMemory memory{};
    VkDeviceSize memory_offset{};
    VkDeviceSize buffer_offset{};
    VkDeviceSize size{};
    bool owned{};
    u64 last_use_epoch{};    ///< GC epoch of the last synchronized access covering this block
    u64 min_reclaim_epoch{}; ///< Ghost-hit protection: not reclaimable before this epoch
    u64 limbo_epoch{};       ///< Epoch the block entered limbo (victim stage), 0 while bound
};

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other) noexcept
        : device{std::exchange(other.device, {})},
          allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          bda_addr{std::exchange(other.bda_addr, 0)},
          is_sparse{std::exchange(other.is_sparse, false)},
          sparse_block_size{std::exchange(other.sparse_block_size, 0)},
          sparse_memory_type_bits{std::exchange(other.sparse_memory_type_bits, 0)},
          sparse_size{std::exchange(other.sparse_size, 0)},
          sparse_bound_bytes{std::exchange(other.sparse_bound_bytes, 0)},
          sparse_blocks{std::move(other.sparse_blocks)},
          sparse_coverage{std::move(other.sparse_coverage)} {
        other.sparse_blocks.clear();
        other.sparse_coverage.Clear();
    }
    UniqueBuffer& operator=(UniqueBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        Destroy();
        device = std::exchange(other.device, {});
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        bda_addr = std::exchange(other.bda_addr, 0);
        is_sparse = std::exchange(other.is_sparse, false);
        sparse_block_size = std::exchange(other.sparse_block_size, 0);
        sparse_memory_type_bits = std::exchange(other.sparse_memory_type_bits, 0);
        sparse_size = std::exchange(other.sparse_size, 0);
        sparse_bound_bytes = std::exchange(other.sparse_bound_bytes, 0);
        sparse_blocks = std::move(other.sparse_blocks);
        sparse_coverage = std::move(other.sparse_coverage);
        other.sparse_blocks.clear();
        other.sparse_coverage.Clear();
        return *this;
    }

    void Destroy() noexcept;

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info,
                const std::function<void()>& allocation_failure_callback = {});

    /// Creates a sparse-resident buffer with no memory bound. Memory is bound on demand with
    /// BindRange and inherited from replaced buffers with TakeOverBlocks.
    void CreateSparse(const vk::BufferCreateInfo& buffer_ci);

    /// Allocates and records bindings for every part of [offset, offset + size) that has no
    /// memory yet. Appends the Vulkan bind operations to out_binds (the caller submits them) and
    /// returns the number of bytes newly bound.
    u64 BindRange(VkDeviceSize offset, VkDeviceSize size, MemoryUsage usage, u64 epoch,
                  const std::function<void()>& allocation_failure_callback,
                  std::vector<vk::SparseMemoryBind>& out_binds);

    /// Aliases every block of `other` into this buffer at (block offset + delta) and takes over
    /// ownership of the allocations. `other` keeps its bindings, so commands already recorded
    /// against it stay valid until it is retired.
    void TakeOverBlocks(UniqueBuffer& other, s64 delta,
                        std::vector<vk::SparseMemoryBind>& out_binds);

    [[nodiscard]] bool IsRangeBound(VkDeviceSize offset, VkDeviceSize size) const noexcept;

    /// Raises the reclaim floor of every block overlapping [offset, offset + size): a block
    /// whose range round-tripped through the reclaimer is individually exempt for a period
    /// scaled to its measured re-reference distance, instead of dragging a global threshold.
    void ProtectRange(VkDeviceSize offset, VkDeviceSize size, u64 until_epoch) noexcept {
        auto it = std::ranges::upper_bound(sparse_blocks, offset, {}, &SparseBlock::buffer_offset);
        if (it != sparse_blocks.begin()) {
            --it;
        }
        const VkDeviceSize end = offset + size;
        for (; it != sparse_blocks.end() && it->buffer_offset < end; ++it) {
            if (it->buffer_offset + it->size > offset) {
                it->min_reclaim_epoch = std::max(it->min_reclaim_epoch, until_epoch);
            }
        }
    }

    /// Stamps the GC epoch on every block overlapping [offset, offset + size).
    void TouchRange(VkDeviceSize offset, VkDeviceSize size, u64 epoch) noexcept {
        auto it = std::ranges::upper_bound(sparse_blocks, offset, {}, &SparseBlock::buffer_offset);
        if (it != sparse_blocks.begin()) {
            --it;
        }
        const VkDeviceSize end = offset + size;
        for (; it != sparse_blocks.end() && it->buffer_offset < end; ++it) {
            if (it->buffer_offset + it->size > offset) {
                it->last_use_epoch = epoch;
            }
        }
    }

    /// Victim stage (limbo): a reclaimed block keeps its memory and Vulkan binding for a
    /// grace period. Coverage is subtracted, so demand for the range flows through
    /// ReinstateLimboRange, which restores the block for free - no bind operation, no re-upload,
    /// and the BDA entries were never touched. Only blocks that survive limbo unreferenced are
    /// actually unbound, and that unbind is deferred and ordered after the recording batch.
    template <typename CanDrop>
    u64 MoveColdBlocksToLimbo(u64 cutoff_epoch, u64 current_epoch, u64 max_bytes,
                              CanDrop&& can_drop) {
        u64 moved = 0;
        for (auto it = sparse_blocks.begin(); it != sparse_blocks.end() && moved < max_bytes;) {
            if (it->last_use_epoch >= cutoff_epoch || it->min_reclaim_epoch > current_epoch ||
                !it->owned || !can_drop(it->buffer_offset, it->size)) {
                ++it;
                continue;
            }
            sparse_coverage.Subtract(it->buffer_offset, it->size);
            sparse_bound_bytes -= it->size;
            limbo_bytes += it->size;
            moved += it->size;
            it->limbo_epoch = current_epoch;
            limbo_blocks.push_back(*it);
            it = sparse_blocks.erase(it);
        }
        return moved;
    }

    /// Restores limbo blocks intersecting [offset, offset + size) to the bound set. Free: the
    /// memory never left and the BDA entries were never cleared.
    template <typename OnReinstate>
    void ReinstateLimboRange(VkDeviceSize offset, VkDeviceSize size, u64 epoch,
                             OnReinstate&& on_reinstate) {
        if (limbo_blocks.empty()) {
            return;
        }
        const VkDeviceSize end = offset + size;
        for (auto it = limbo_blocks.begin(); it != limbo_blocks.end();) {
            if (it->buffer_offset >= end || it->buffer_offset + it->size <= offset) {
                ++it;
                continue;
            }
            SparseBlock block = *it;
            it = limbo_blocks.erase(it);
            on_reinstate(block.buffer_offset, block.size, block.limbo_epoch);
            block.last_use_epoch = epoch;
            block.limbo_epoch = 0;
            const auto insert_at = std::ranges::upper_bound(sparse_blocks, block.buffer_offset, {},
                                                            &SparseBlock::buffer_offset);
            sparse_blocks.insert(insert_at, block);
            sparse_coverage.Add(block.buffer_offset, block.size);
            sparse_bound_bytes += block.size;
            limbo_bytes -= block.size;
        }
    }

    /// Removes limbo blocks whose grace period ended, emitting the null-memory unbind operations
    /// (the caller defers them behind the recording batch) and the allocations to free once the
    /// unbind has executed. on_expired runs per block before removal.
    template <typename OnExpired>
    u64 CollectExpiredLimbo(u64 expiry_epoch, OnExpired&& on_expired,
                            std::vector<vk::SparseMemoryBind>& out_unbinds,
                            std::vector<VmaAllocation>& out_allocations) {
        u64 released = 0;
        for (auto it = limbo_blocks.begin(); it != limbo_blocks.end();) {
            if (it->limbo_epoch > expiry_epoch) {
                ++it;
                continue;
            }
            out_unbinds.push_back(vk::SparseMemoryBind{
                .resourceOffset = it->buffer_offset,
                .size = it->size,
                .memory = VK_NULL_HANDLE,
                .memoryOffset = 0,
            });
            out_allocations.push_back(it->allocation);
            released += it->size;
            limbo_bytes -= it->size;
            on_expired(it->buffer_offset, it->size);
            it = limbo_blocks.erase(it);
        }
        return released;
    }

    /// Calls func(offset, size) for each maximal contiguous bound range (adjacent blocks merged).
    template <typename Func>
    void ForEachBoundRange(Func&& func) const {
        sparse_coverage.ForEach(std::forward<Func>(func));
    }

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
    bool is_sparse{};
    VkDeviceSize sparse_block_size{};
    u32 sparse_memory_type_bits{};
    VkDeviceSize sparse_size{};
    VkDeviceSize sparse_bound_bytes{};
    std::vector<SparseBlock> sparse_blocks; ///< Sorted by buffer_offset, non-overlapping.
    std::vector<SparseBlock> limbo_blocks;  ///< Victim stage: bound, uncovered, awaiting expiry.
    u64 limbo_bytes{};
    /// Coalesced bound ranges; range queries use this so a buffer that accumulated many small
    /// demand bindings stays cheap to check.
    SparseCoverage sparse_coverage;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_,
                    const std::function<void()>& allocation_failure_callback = {},
                    bool sparse = false);
    ~Buffer();

    /// True when the buffer is sparse-resident: memory is bound per range on demand.
    [[nodiscard]] bool IsSparse() const noexcept {
        return buffer.is_sparse;
    }

    /// Sparse buffers only. Binds memory for [offset, offset + size) where missing; the bind
    /// operations are appended to out_binds for the caller to submit. Returns bytes newly bound.
    u64 EnsureBound(u64 offset, u64 size, u64 epoch,
                    const std::function<void()>& allocation_failure_callback,
                    std::vector<vk::SparseMemoryBind>& out_binds) {
        const u64 bound =
            buffer.BindRange(offset, size, usage, epoch, allocation_failure_callback, out_binds);
        allocation_size = buffer.sparse_bound_bytes + buffer.limbo_bytes;
        return bound;
    }

    /// Sparse buffers only: records that [offset, offset + size) was accessed this GC epoch.
    void TouchSparseRange(u64 offset, u64 size, u64 epoch) noexcept {
        if (buffer.is_sparse) {
            buffer.TouchRange(offset, size, epoch);
        }
    }

    /// Sparse buffers only. See UniqueBuffer::MoveColdBlocksToLimbo.
    template <typename CanDrop>
    u64 MoveColdSparseBlocksToLimbo(u64 cutoff_epoch, u64 current_epoch, u64 max_bytes,
                                    CanDrop&& can_drop) {
        return buffer.MoveColdBlocksToLimbo(cutoff_epoch, current_epoch, max_bytes,
                                            std::forward<CanDrop>(can_drop));
    }

    /// Sparse buffers only. See UniqueBuffer::ReinstateLimboRange.
    template <typename OnReinstate>
    void ReinstateSparseRange(u64 offset, u64 size, u64 epoch, OnReinstate&& on_reinstate) {
        if (buffer.is_sparse) {
            buffer.ReinstateLimboRange(offset, size, epoch,
                                       std::forward<OnReinstate>(on_reinstate));
        }
    }

    /// Sparse buffers only. See UniqueBuffer::CollectExpiredLimbo.
    template <typename OnExpired>
    u64 CollectExpiredSparseLimbo(u64 expiry_epoch, OnExpired&& on_expired,
                                  std::vector<vk::SparseMemoryBind>& out_unbinds,
                                  std::vector<VmaAllocation>& out_allocations) {
        const u64 released = buffer.CollectExpiredLimbo(
            expiry_epoch, std::forward<OnExpired>(on_expired), out_unbinds, out_allocations);
        allocation_size = buffer.sparse_bound_bytes + buffer.limbo_bytes;
        return released;
    }

    /// Sparse buffers only: bytes parked in the victim stage (still resident).
    [[nodiscard]] u64 LimboBytes() const noexcept {
        return buffer.limbo_bytes;
    }

    /// Sparse buffers only: exempts blocks in the range from reclaim until until_epoch.
    void ProtectSparseRange(u64 offset, u64 size, u64 until_epoch) noexcept {
        if (buffer.is_sparse) {
            buffer.ProtectRange(offset, size, until_epoch);
        }
    }

    /// Sparse buffers only. Inherits every memory block of `other` (a buffer this one replaces)
    /// by aliasing it at the matching guest address.
    void TakeOverBlocks(Buffer& other, std::vector<vk::SparseMemoryBind>& out_binds) {
        const s64 delta = static_cast<s64>(other.cpu_addr) - static_cast<s64>(cpu_addr);
        buffer.TakeOverBlocks(other.buffer, delta, out_binds);
        allocation_size = buffer.sparse_bound_bytes;
        other.allocation_size = other.buffer.sparse_bound_bytes;
    }

    [[nodiscard]] bool IsRangeBound(u64 offset, u64 size) const noexcept {
        return !buffer.is_sparse || buffer.IsRangeBound(offset, size);
    }

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = delete;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    [[nodiscard]] u64 AllocationSizeBytes() const noexcept {
        return allocation_size;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    void SetLastUseTick(u64 tick) noexcept {
        last_use_tick = tick;
    }

    [[nodiscard]] u64 LastUseTick() const noexcept {
        return last_use_tick;
    }

    void SetRetirementContext(BufferRetirementReason reason, u64 scheduled_tick) noexcept {
        retirement_reason = reason;
        retirement_scheduled_tick = scheduled_tick;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 allocation_size = 0;
    u64 lru_id = 0;
    u64 last_use_tick = 0;
    u64 address_generation = 0;
    u64 retirement_scheduled_tick = 0;
    BufferRetirementReason retirement_reason{BufferRetirementReason::Unknown};
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

class StreamBufferMapping {
public:
    StreamBufferMapping(StreamBuffer& stream_buffer, u64 size, u64 alignment = 0,
                        bool allow_wait = true);
    ~StreamBufferMapping();

    StreamBufferMapping(const StreamBufferMapping&) = delete;
    StreamBufferMapping& operator=(const StreamBufferMapping&) = delete;

    StreamBufferMapping(StreamBufferMapping&& other)
        : buffer{std::exchange(other.buffer, nullptr)}, data{std::exchange(other.data, nullptr)},
          offset{std::exchange(other.offset, 0)},
          is_temp_buffer{std::exchange(other.is_temp_buffer, false)} {}

    StreamBufferMapping& operator=(StreamBufferMapping&& other) {
        if (this != &other) {
            buffer = std::exchange(other.buffer, nullptr);
            data = std::exchange(other.data, nullptr);
            offset = std::exchange(other.offset, 0);
            is_temp_buffer = std::exchange(other.is_temp_buffer, false);
        }
        return *this;
    }

    VideoCore::Buffer* Buffer() const {
        return buffer;
    }

    u8* Data() const {
        return data;
    }

    u64 Offset() const {
        return offset;
    }

    bool TemporaryBuffer() const {
        return is_temp_buffer;
    }

private:
    VideoCore::Buffer* buffer;
    u8* data{};
    u64 offset{};
    bool is_temp_buffer{};
};

} // namespace VideoCore
