// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <boost/container/small_vector.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/logging/log.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

std::string_view BufferTypeName(MemoryUsage type) {
    switch (type) {
    case MemoryUsage::Upload:
        return "Upload";
    case MemoryUsage::Download:
        return "Download";
    case MemoryUsage::Stream:
        return "Stream";
    case MemoryUsage::DeviceLocal:
        return "DeviceLocal";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
    return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::Upload:
    case MemoryUsage::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryUsage::Download:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryUsage::DeviceLocal:
        return {};
    }
    return {};
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::Upload:
    case MemoryUsage::Download:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    Destroy();
}

void UniqueBuffer::Destroy() noexcept {
    if (is_sparse) {
        if (buffer) {
            device.destroyBuffer(buffer);
        }
        // Blocks that were handed to a successor (aliased) are freed by that successor.
        for (const SparseBlock& block : sparse_blocks) {
            if (block.owned && block.allocation) {
                vmaFreeMemory(allocator, block.allocation);
            }
        }
        sparse_blocks.clear();
        sparse_coverage.Clear();
        sparse_bound_bytes = 0;
        buffer = VK_NULL_HANDLE;
        bda_addr = 0;
        is_sparse = false;
        return;
    }
    if (buffer) {
        vmaDestroyBuffer(allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        bda_addr = 0;
    }
}

void UniqueBuffer::CreateSparse(const vk::BufferCreateInfo& buffer_ci_in) {
    vk::BufferCreateInfo buffer_ci = buffer_ci_in;
    buffer_ci.flags |= vk::BufferCreateFlagBits::eSparseBinding |
                       vk::BufferCreateFlagBits::eSparseResidency |
                       vk::BufferCreateFlagBits::eSparseAliased;
    auto [result, created] = device.createBuffer(buffer_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed creating sparse buffer: {}",
               vk::to_string(result));
    buffer = created;
    is_sparse = true;
    sparse_size = buffer_ci.size;
    allocation = VK_NULL_HANDLE;

    const vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(buffer);
    sparse_block_size = requirements.alignment;
    sparse_memory_type_bits = requirements.memoryTypeBits;
    ASSERT_MSG(sparse_block_size != 0 && (sparse_block_size & (sparse_block_size - 1)) == 0 &&
                   sparse_block_size <= 64_KB,
               "Unexpected sparse block size {}", sparse_block_size);

    if (buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        const vk::BufferDeviceAddressInfo bda_info{.buffer = buffer};
        bda_addr = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_addr != 0, "Failed to get sparse buffer device address");
    }
}

bool UniqueBuffer::IsRangeBound(VkDeviceSize offset, VkDeviceSize size) const noexcept {
    const VkDeviceSize end = std::min<VkDeviceSize>(offset + size, sparse_size);
    return end <= offset || sparse_coverage.Contains(offset, end - offset);
}

u64 UniqueBuffer::BindRange(VkDeviceSize offset, VkDeviceSize size, MemoryUsage usage, u64 epoch,
                            const std::function<void()>& allocation_failure_callback,
                            std::vector<vk::SparseMemoryBind>& out_binds) {
    ASSERT(is_sparse);
    if (size == 0) {
        return 0;
    }
    // Bindings are made in whole sparse blocks inside the buffer.
    const VkDeviceSize begin = Common::AlignDown(offset, sparse_block_size);
    const VkDeviceSize end =
        std::min<VkDeviceSize>(Common::AlignUp(offset + size, sparse_block_size), sparse_size);
    if (begin >= end) {
        return 0;
    }

    // Collect the gaps of [begin, end) that have no memory.
    boost::container::small_vector<std::pair<VkDeviceSize, VkDeviceSize>, 8> gaps;
    sparse_coverage.ForEachGap(
        begin, end, [&](u64 gap_begin, u64 gap_end) { gaps.emplace_back(gap_begin, gap_end); });

    u64 newly_bound = 0;
    for (const auto& [gap_begin, gap_end] : gaps) {
        const VkDeviceSize gap_size = gap_end - gap_begin;
        const VkMemoryRequirements requirements = {
            .size = gap_size,
            .alignment = sparse_block_size,
            .memoryTypeBits = sparse_memory_type_bits,
        };
        VmaAllocationCreateInfo alloc_ci = {
            .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
            .usage = VMA_MEMORY_USAGE_UNKNOWN,
            .requiredFlags = 0,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
        };
        VmaAllocation block_allocation{};
        VmaAllocationInfo info{};
        VkResult result =
            vmaAllocateMemory(allocator, &requirements, &alloc_ci, &block_allocation, &info);
        if ((result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) &&
            allocation_failure_callback) {
            LOG_WARNING(Render_Vulkan,
                        "Sparse block allocation of {} bytes failed within the reported memory "
                        "budget; running emergency garbage collection before retrying",
                        gap_size);
            allocation_failure_callback();
            result =
                vmaAllocateMemory(allocator, &requirements, &alloc_ci, &block_allocation, &info);
        }
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            const bool use_host_fallback =
                Config::getUseHostMemoryFallback() &&
                (usage == MemoryUsage::DeviceLocal || usage == MemoryUsage::Stream);
            LOG_WARNING(Render_Vulkan,
                        "Sparse block allocation of {} bytes failed within the reported memory "
                        "budget; retrying without the budget restriction{}",
                        gap_size, use_host_fallback ? " with host memory fallback" : "");
            alloc_ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            if (use_host_fallback) {
                alloc_ci.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            }
            result =
                vmaAllocateMemory(allocator, &requirements, &alloc_ci, &block_allocation, &info);
        }
        ASSERT_MSG(result == VK_SUCCESS, "Failed allocating sparse block of {} bytes: {}", gap_size,
                   vk::to_string(vk::Result{result}));

        SparseBlock block{
            .allocation = block_allocation,
            .memory = info.deviceMemory,
            .memory_offset = info.offset,
            .buffer_offset = gap_begin,
            .size = gap_size,
            .owned = true,
            .last_use_epoch = epoch,
        };
        const auto insert_at =
            std::ranges::upper_bound(sparse_blocks, gap_begin, {}, &SparseBlock::buffer_offset);
        sparse_blocks.insert(insert_at, block);
        sparse_coverage.Add(gap_begin, gap_size);
        sparse_bound_bytes += gap_size;
        newly_bound += gap_size;
        out_binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = gap_begin,
            .size = gap_size,
            .memory = info.deviceMemory,
            .memoryOffset = info.offset,
        });
    }
    return newly_bound;
}

void UniqueBuffer::TakeOverBlocks(UniqueBuffer& other, s64 delta,
                                  std::vector<vk::SparseMemoryBind>& out_binds) {
    ASSERT(is_sparse && other.is_sparse);
    for (SparseBlock& block : other.sparse_blocks) {
        const s64 new_offset = static_cast<s64>(block.buffer_offset) + delta;
        ASSERT_MSG(new_offset >= 0 &&
                       static_cast<VkDeviceSize>(new_offset) + block.size <= sparse_size,
                   "Sparse block [{:#x},{:#x}) does not fit the successor of size {:#x} (delta {})",
                   block.buffer_offset, block.buffer_offset + block.size, sparse_size, delta);
        SparseBlock taken = block;
        taken.buffer_offset = static_cast<VkDeviceSize>(new_offset);
        taken.owned = block.owned;
        block.owned = false;
        const auto insert_at = std::ranges::upper_bound(sparse_blocks, taken.buffer_offset, {},
                                                        &SparseBlock::buffer_offset);
        sparse_blocks.insert(insert_at, taken);
        sparse_coverage.Add(taken.buffer_offset, taken.size);
        sparse_bound_bytes += taken.size;
        out_binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = taken.buffer_offset,
            .size = taken.size,
            .memory = taken.memory,
            .memoryOffset = taken.memory_offset,
        });
    }
}

void UniqueBuffer::Create(const vk::BufferCreateInfo& buffer_ci, MemoryUsage usage,
                          VmaAllocationInfo* out_alloc_info,
                          const std::function<void()>& allocation_failure_callback) {
    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    VmaAllocationCreateInfo alloc_ci = {
        // VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT makes every backing memory block BDA
        // capable. BDA buffers do not require dedicated allocations, and forcing every cache
        // buffer to be dedicated prevents VMA from suballocating them and increases fragmentation.
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    VkBuffer unsafe_buffer{};
    VkResult result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                      &allocation, out_alloc_info);
    const auto reset_outputs = [&] {
        unsafe_buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        if (out_alloc_info) {
            *out_alloc_info = {};
        }
    };
    if ((result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) &&
        allocation_failure_callback) {
        LOG_WARNING(Render_Vulkan,
                    "Buffer allocation of {} bytes failed within the reported memory budget; "
                    "running emergency garbage collection before retrying",
                    buffer_ci.size);
        reset_outputs();
        allocation_failure_callback();
        result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                 &allocation, out_alloc_info);
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
        const bool use_host_fallback =
            Config::getUseHostMemoryFallback() &&
            (usage == MemoryUsage::DeviceLocal || usage == MemoryUsage::Stream);
        LOG_WARNING(Render_Vulkan,
                    "Buffer allocation of {} bytes failed within the reported memory budget; "
                    "retrying without the budget restriction{}",
                    buffer_ci.size, use_host_fallback ? " with host memory fallback" : "");
        alloc_ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        if (use_host_fallback) {
            alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }
        reset_outputs();
        result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                 &allocation, out_alloc_info);
    }
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer with error {}",
               vk::to_string(vk::Result{result}));
    buffer = vk::Buffer{unsafe_buffer};

    if (with_bda) {
        vk::BufferDeviceAddressInfo bda_info{
            .buffer = buffer,
        };
        auto bda_result = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_result != 0, "Failed to get buffer device address");
        bda_addr = bda_result;
    }
}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
               VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_,
               const std::function<void()>& allocation_failure_callback, bool sparse)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // Create buffer object.
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
    };
    if (sparse) {
        buffer.CreateSparse(buffer_ci);
        allocation_size = 0;
        address_generation = instance->TrackBufferAddress(buffer.bda_addr, size_bytes, cpu_addr,
                                                          allocation_size, static_cast<u32>(usage));
        Vulkan::SetObjectName(instance->GetDevice(), Handle(), "SparseBuffer {:#x}:{:#x}", cpu_addr,
                              size_bytes);
        return;
    }
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, usage, &alloc_info, allocation_failure_callback);
    allocation_size = alloc_info.size;
    address_generation = instance->TrackBufferAddress(buffer.bda_addr, size_bytes, cpu_addr,
                                                      allocation_size, static_cast<u32>(usage));

    const auto device = instance->GetDevice();
    Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);

    // Map it if it is host visible.
    VkMemoryPropertyFlags property_flags{};
    vmaGetAllocationMemoryProperties(instance->GetAllocator(), buffer.allocation, &property_flags);
    if (alloc_info.pMappedData) {
        mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
    }
    is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

Buffer::~Buffer() {
    // A moved-from Buffer no longer owns its Vulkan buffer, so only the live owner retires the
    // diagnostic generation. This preserves exact lifetimes when SlotVector grows and moves its
    // elements.
    if (buffer.bda_addr != 0) {
        instance->RetireBufferAddress(address_generation, last_use_tick, scheduler->CurrentTick(),
                                      retirement_scheduled_tick,
                                      static_cast<u32>(retirement_reason));
    }
}

void Buffer::Fill(u64 offset, u32 num_bytes, u32 value) {
    scheduler->EndRendering();
    ASSERT_MSG(offset % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer size must be a multiple of 4 bytes");
    const auto cmdbuf = scheduler->CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.fillBuffer(buffer, offset, num_bytes, value);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x4000;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x1000;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           MemoryUsage usage, u64 size_bytes)
    : Buffer{instance, scheduler, usage, 0, AllFlags, size_bytes} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, Handle(), "StreamBuffer({}):{:#x}", BufferTypeName(usage),
                          size_bytes);
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (!is_coherent && usage == MemoryUsage::Stream) {
        size = Common::AlignUp(size, instance->NonCoherentAtomSize());
    }

    if (size > this->size_bytes) {
        return {nullptr, 0};
    }

    mapped_size = size;

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the state.
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    const u64 mapped_upper_bound = offset + size;
    if (!WaitPendingOperations(mapped_upper_bound, allow_wait)) {
        return {nullptr, 0};
    }

    return {mapped_data.data() + offset, offset};
}

void StreamBuffer::Commit() {
    if (!is_coherent) {
        if (usage == MemoryUsage::Download) {
            vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, offset,
                                    mapped_size);
        } else {
            vmaFlushAllocation(instance->GetAllocator(), buffer.allocation, offset, mapped_size);
        }
    }

    offset += mapped_size;
    if (current_watch_cursor != 0 &&
        current_watches[current_watch_cursor].tick == scheduler->CurrentTick()) {
        current_watches[current_watch_cursor].upper_bound = offset;
        return;
    }

    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = scheduler->CurrentTick();
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

bool StreamBuffer::WaitPendingOperations(u64 requested_upper_bound, bool allow_wait) {
    if (!invalidation_mark) {
        return true;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        if (!scheduler->IsFree(watch.tick) && !allow_wait) {
            return false;
        }
        scheduler->Wait(watch.tick);
        wait_bound = watch.upper_bound;
        ++wait_cursor;
    }
    return true;
}

StreamBufferMapping::StreamBufferMapping(StreamBuffer& stream_buffer, u64 size, u64 alignment,
                                         bool allow_wait) {
    const auto [data, offset] = stream_buffer.Map(size, alignment, allow_wait);
    if (!data) {
        // This happens if the size is too big or no waiting is allowed when it is required
        is_temp_buffer = true;
        this->buffer = new VideoCore::Buffer(*stream_buffer.instance, *stream_buffer.scheduler,
                                             stream_buffer.usage, 0, AllFlags, size);
        this->data = this->buffer->mapped_data.data();
        this->offset = 0;
        ASSERT_MSG(this->data, "Failed to map temporary buffer");
    } else {
        is_temp_buffer = false;
        buffer = &stream_buffer;
        this->data = data;
        this->offset = offset;
    }
}

StreamBufferMapping::~StreamBufferMapping() {
    if (is_temp_buffer) {
        ASSERT(buffer);
        auto scheduler = buffer->scheduler;
        scheduler->DeferOperation([buffer = this->buffer]() mutable { delete buffer; });
    }
}

} // namespace VideoCore
