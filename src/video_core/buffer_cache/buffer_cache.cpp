// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <vector>
#include "common/alignment.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/memory_patcher.h"
#include "common/scope_exit.h"
#include "core/memory.h"

#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/stream_buffer_policy.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);
    sparse_buffers = instance.IsSparseCacheBufferSupported();

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool download) {
    // Readback-disabled mode never needs a live host buffer to make guest memory authoritative.
    // Always update an existing tracker region so a buffer that was removed from buffer_ranges
    // cannot leave behind an unserviceable write watcher. IteratePages<false> makes this a cheap
    // no-op when the tracker has never seen the range.
    if (Config::readbackSpeed() == Config::ReadbackSpeed::Disable) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        return;
    }
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}
template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const VAddr page_addr = PageManager::GetPageAddr(device_addr);
    auto* memory = Core::Memory::Instance();
    if (preemptive_downloads.Intersects(page_addr, PageManager::PAGE_SIZE)) {
        preemptive_downloads.ForEachInRange(
            page_addr, PageManager::PAGE_SIZE,
            [&](VAddr begin, VAddr end, const PreemptiveDownload& download) {
                scheduler.Wait(download.done_tick);
                memory->TryWriteBacking(std::bit_cast<u8*>(download.device_addr), download.staging,
                                        download.size);
            });
        preemptive_downloads.Subtract(page_addr, PageManager::PAGE_SIZE);
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, is_write);
    } else {
        MarkBufferUsed(buffer);
        const auto [download, offset] = download_buffer.Map(total_size_bytes);
        for (auto& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dstOffset += offset;
        }
        download_buffer.Commit();
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        instance.InsertCheckpoint(cmdbuf, Vulkan::GpuCheckpoint::BufferDownload,
                                  Vulkan::GpuCheckpointContext{
                                      .source_guest_address = device_addr,
                                      .transfer_size = total_size_bytes,
                                      .source_generation = buffer.address_generation,
                                  });
        cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
        const VAddr buffer_addr = buffer.CpuAddr();
        auto write_data = [this, copies = std::move(copies), download, offset, buffer_addr,
                           device_addr, size, is_write] {
            auto* memory = Core::Memory::Instance();
            for (const auto& copy : copies) {
                const VAddr copy_device_addr = buffer_addr + copy.srcOffset;
                const u64 dst_offset = copy.dstOffset - offset;
                memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                        copy.size);
            }
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, is_write);
        };
        if constexpr (async) {
            scheduler.DeferOperation(std::move(write_data));
        } else {
            scheduler.Finish();
            write_data();
        }
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    // if write tick == current_tick -> send flush request

    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    const Buffer& buffer = slot_buffers[buffer_id];
    // ASSERT(buffer.IsInBounds(device_addr, size));

    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        DownloadBufferMemory<false>(buffer, device_addr, size, is_write);
    });
}

void BufferCache::ReadEdgeImagePages(const Image& image) {
    // May happen that after downloading the image and invalidating region,
    // that there were GPU modified ranges that are lost due to CPU reuploading.
    // This doesn't change tracker state and it is spected to call DownloadImageMemory after this.
    const VAddr image_addr = image.info.guest_address;
    const u64 image_size = image.info.guest_size;
    const VAddr image_end = image_addr + image_size;
    const VAddr page_start = PageManager::GetPageAddr(image_addr);
    const VAddr page_end = PageManager::GetNextPageAddr(image_end - 1);
    boost::container::small_vector<vk::BufferCopy, 2> copies;
    u64 total_size_bytes = 0;
    const auto [buffer, base_offset] = ObtainBufferForImage(image_addr, image_size);
    const auto add_download = [&](VAddr start, VAddr end) {
        const u64 new_offset = (start - buffer->CpuAddr()) + base_offset;
        const u64 new_size = end - start;
        copies.push_back(vk::BufferCopy{
            .srcOffset = new_offset,
            .dstOffset = total_size_bytes,
            .size = new_size,
        });
        constexpr u64 align = 64ULL;
        constexpr u64 mask = ~(align - 1ULL);
        total_size_bytes += (new_size + align - 1) & mask;
    };

    gpu_modified_ranges.ForEachInRange(page_start, image_addr - page_start, add_download);
    gpu_modified_ranges.ForEachInRange(image_end, page_end - image_end, add_download);
    gpu_modified_ranges.Subtract(page_start, page_end - page_start);
    if (total_size_bytes == 0) {
        return;
    }
    const auto [download, download_offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += download_offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    instance.InsertCheckpoint(cmdbuf, Vulkan::GpuCheckpoint::BufferDownload,
                              Vulkan::GpuCheckpointContext{
                                  .source_guest_address = buffer->CpuAddr(),
                                  .transfer_size = total_size_bytes,
                                  .source_generation = buffer->address_generation,
                              });
    cmdbuf.copyBuffer(buffer->Handle(), download_buffer.Handle(), copies);
    scheduler.DeferOperation([this, buf_addr = buffer->CpuAddr(), copies = std::move(copies),
                              download, download_offset, image_addr, image_size]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buf_addr + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - download_offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
        }
    });
}

BufferCache::VertexBufferBinding BufferCache::PrepareVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;
    VertexBufferBinding binding{};
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(binding.attributes, binding.bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (binding.bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return binding;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    // Resolve host bindings for every guest vertex buffer.
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            binding.host_buffers.emplace_back(host_buffer_info->vk_buffer);
            binding.host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                           host_buffer_info->base_address);
        } else {
            binding.host_buffers.emplace_back(VK_NULL_HANDLE);
            binding.host_offsets.push_back(0);
        }
        binding.host_sizes.push_back(buffer.GetSize());
        binding.host_strides.push_back(buffer.GetStride());
    }
    return binding;
}

void BufferCache::BindVertexBuffers(const VertexBufferBinding& binding) {
    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        cmdbuf.setVertexInputEXT(binding.bindings, binding.attributes);
    }
    if (binding.bindings.empty()) {
        return;
    }
    const auto num_buffers = static_cast<u32>(binding.host_buffers.size());
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, binding.host_buffers.data(),
                                 binding.host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, binding.host_buffers.data(),
                                  binding.host_offsets.data(), binding.host_sizes.data(),
                                  binding.host_strides.data());
    }
}

BufferCache::IndexBufferBinding BufferCache::PrepareIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Resolve the index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    return IndexBufferBinding{
        .buffer = vk_buffer->Handle(),
        .offset = offset,
        .index_type = index_type,
    };
}

void BufferCache::BindIndexBuffer(const IndexBufferBinding& binding) {
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(binding.buffer, binding.offset, binding.index_type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");

    if (!is_gds) {
        texture_cache.ClearMeta(address);

        const bool use_alt_gpu_check = MemoryPatcher::Quirks().defer_write_protect;

        const bool gpu_modified = use_alt_gpu_check
                                      ? IsRegionGpuModified(address, num_bytes)
                                      : memory_tracker->IsRegionGpuModified(address, num_bytes);

        if (!gpu_modified) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }

    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }

        const auto [buffer, offset] =
            ObtainBuffer(address, num_bytes, ObtainBufferFlags::IsWritten);
        return buffer;
    }();

    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();

    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto [buffer, _] = ObtainBuffer(
            dst, num_bytes, ObtainBufferFlags::IsWritten | ObtainBufferFlags::IsTexelBuffer);
        return *buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    instance.InsertCheckpoint(cmdbuf, Vulkan::GpuCheckpoint::BufferCopy,
                              Vulkan::GpuCheckpointContext{
                                  .source_guest_address = src,
                                  .destination_guest_address = dst,
                                  .transfer_size = num_bytes,
                                  .source_generation = src_buffer.address_generation,
                                  .destination_generation = dst_buffer.address_generation,
                              });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                  ObtainBufferFlags flags, BufferId buffer_id) {

    const bool is_written = (flags & ObtainBufferFlags::IsWritten) != ObtainBufferFlags{};
    const bool is_texel_buffer = (flags & ObtainBufferFlags::IsTexelBuffer) != ObtainBufferFlags{};
    const bool skip_stream_buffer =
        (flags & ObtainBufferFlags::IgnoreStreamBuffer) != ObtainBufferFlags{};
    const bool allow_texture_gc =
        (flags & ObtainBufferFlags::AvoidTextureGc) == ObtainBufferFlags{};

    if (!is_written && !skip_stream_buffer && size <= CACHING_PAGESIZE &&
        !IsRegionGpuModified(device_addr, size) && IsRegionCpuModified(device_addr, size)) {

        const u64 offset = stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
        return {&stream_buffer, offset};
    }

    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size, allow_texture_gc);
    }

    Buffer& buffer = slot_buffers[buffer_id];

    const bool defer_write_protect = MemoryPatcher::Quirks().defer_write_protect;

    SynchronizeBuffer(buffer, device_addr, size, is_written && !defer_write_protect,
                      is_texel_buffer);
    TouchBuffer(buffer);

    if (is_written) {
        if (defer_write_protect) {
            gpu_modified_ranges_pending.Add(device_addr, size);
        } else {
            gpu_modified_ranges.Add(device_addr, size);
        }
    }

    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            TouchBuffer(buffer);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, ObtainBufferFlags::AvoidTextureGc);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

BufferCache::DebugPageState BufferCache::GetDebugPageState(VAddr addr) {
    const auto tracker_state = memory_tracker->GetDebugPageState(addr);
    const VAddr page_addr = PageManager::GetPageAddr(addr);
    DebugPageState result{
        .registered = IsRegionRegistered(addr, 1),
        .tracker_exists = tracker_state.region_exists,
        .cpu_modified = tracker_state.cpu_modified,
        .gpu_modified = tracker_state.gpu_modified,
        .gpu_pending = gpu_modified_ranges_pending.Intersects(page_addr, PageManager::PAGE_SIZE),
    };
    if (result.registered) {
        const BufferId buffer_id = page_table[addr >> CACHING_PAGEBITS].buffer_id;
        if (!IsBufferInvalid(buffer_id)) {
            const Buffer& buffer = slot_buffers[buffer_id];
            result.buffer_id = buffer_id.index;
            result.buffer_begin = buffer.CpuAddr();
            result.buffer_end = buffer.CpuAddr() + buffer.SizeBytes();
        }
    }
    return result;
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (memory_tracker->IsRegionGpuModified(addr, size)) {
        return true;
    }
    const VAddr page_addr = PageManager::GetPageAddr(addr);
    bool modified = false;
    gpu_modified_ranges_pending.ForEachInRange(page_addr, PageManager::PAGE_SIZE,
                                               [&](VAddr, VAddr) { modified = true; });
    return modified;
}

void BufferCache::MarkRegionAsGpuModified(VAddr addr, size_t size) {
    gpu_modified_ranges.Add(addr, size);
    memory_tracker->MarkRegionAsGpuModified(addr, size);
    texture_cache.MarkAsMaybeReused(addr, size);
}

void BufferCache::MarkRegionAsFlushed(VAddr addr, size_t size) {
    gpu_modified_ranges.Subtract(addr, size);
    // is_write=false: transfer eviction authority only. Marking the range CPU-dirty here would
    // make the next buffer sync upload the written-back tiled texel bytes into every covering
    // buffer's device copy — Run 38's static exploding geometry. Device buffer copies keep the
    // content they always had; no data flow exists after this call that did not exist before
    // eviction readback was introduced.
    memory_tracker->UnmarkRegionAsGpuModified(addr, size, false);
}

void BufferCache::MarkRegionAsCpuModified(VAddr addr, size_t size) {
    memory_tracker->MarkRegionAsCpuModified(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size, bool allow_texture_gc) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size, allow_texture_gc);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size, allow_texture_gc);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size,
                                                        u64 used_memory, u64 total_budget) {
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    const VAddr requested_begin = begin;
    const VAddr requested_end = end;
    VAddr overlap_begin = std::numeric_limits<VAddr>::max();
    VAddr overlap_span_end = 0;
    int stream_score = 0;
    bool has_stream_leap = false;
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .stream_score = 0,
            .has_stream_leap = has_stream_leap,
            .stream_growth_suppressed = false,
            .desired_stream_growth = 0,
            .speculative_bytes = 0,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        overlap_begin = std::min(overlap_begin, overlap_device_addr);
        if (overlap_device_addr < begin) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_buffer_end = overlap_device_addr + overlap.SizeBytes();
        if (overlap_buffer_end > end) {
            end = overlap_buffer_end;
        }
        overlap_span_end = std::max(overlap_span_end, overlap_buffer_end);
        stream_score = AccumulateStreamScore(stream_score, overlap.StreamScore());
    }

    // Padding must happen after natural overlap closure. Growing `end` inside the loop makes the
    // overlap scan consume neighboring buffers merely because they occupy reserved capacity,
    // turning a small streaming step into a much larger merge and copy.
    const VAddr natural_begin = begin;
    const VAddr natural_end = end;
    const bool stream_growth_candidate = stream_score > StreamLeapThreshold;
    const auto growth =
        stream_growth_candidate
            ? DetermineStreamGrowth(requested_begin, requested_end, overlap_begin, overlap_span_end)
            : StreamGrowthDirections{};
    const VAddr natural_size = natural_end - natural_begin;
    VAddr right_growth =
        growth.right ? DetermineStreamGrowthDistance(wanted_size, requested_end - overlap_span_end,
                                                     natural_size)
                     : 0;
    VAddr left_growth =
        growth.left ? DetermineStreamGrowthDistance(wanted_size, overlap_begin - requested_begin,
                                                    natural_size)
                    : 0;
    if (sparse_buffers) {
        // Unbound reserve pages cost no memory, so a streaming buffer can reserve a full growth
        // step ahead and is never suppressed by memory pressure.
        right_growth = growth.right ? SparseStreamGrowthDistance(right_growth, natural_size) : 0;
        left_growth = growth.left ? SparseStreamGrowthDistance(left_growth, natural_size) : 0;
    }
    const u64 desired_stream_growth = left_growth + right_growth;
    const bool stream_growth_suppressed =
        !sparse_buffers && stream_growth_candidate && desired_stream_growth != 0 &&
        !ShouldReserveStreamGrowth(used_memory, total_budget, natural_end - natural_begin,
                                   desired_stream_growth);
    has_stream_leap =
        stream_growth_candidate && desired_stream_growth != 0 && !stream_growth_suppressed;
    if (has_stream_leap) {
        static constexpr VAddr MinAddress = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        static constexpr VAddr MaxAddress = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;

        if (growth.right) {
            VAddr padded_end = GrowStreamEnd(end, right_growth, MaxAddress);
            padded_end = Common::AlignDown(padded_end, CACHING_PAGESIZE);
            // Spare capacity may use empty pages but must not proactively absorb another buffer.
            for (VAddr page_addr = end; page_addr < padded_end; page_addr += CACHING_PAGESIZE) {
                if (page_table[page_addr >> CACHING_PAGEBITS].buffer_id) {
                    padded_end = page_addr;
                    break;
                }
            }
            end = padded_end;
        }
        if (growth.left) {
            VAddr padded_begin = GrowStreamBegin(begin, left_growth, MinAddress);
            padded_begin = Common::AlignUp(padded_begin, CACHING_PAGESIZE);
            for (VAddr page_addr = begin; page_addr > padded_begin;) {
                page_addr -= CACHING_PAGESIZE;
                if (page_table[page_addr >> CACHING_PAGEBITS].buffer_id) {
                    padded_begin = page_addr + CACHING_PAGESIZE;
                    break;
                }
            }
            begin = padded_begin;
        }
    }
    const int replacement_stream_score =
        ReplacementStreamScore(stream_score, overlap_ids.size(), has_stream_leap);
    const u64 speculative_bytes = (natural_begin - begin) + (end - natural_end);
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .stream_score = replacement_stream_score,
        .has_stream_leap = has_stream_leap,
        .stream_growth_suppressed = stream_growth_suppressed,
        .desired_stream_growth = desired_stream_growth,
        .speculative_bytes = speculative_bytes,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              std::vector<vk::SparseMemoryBind>& sparse_binds) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (new_buffer.IsSparse()) {
        // No copy: the replacement aliases the memory blocks of the buffer it absorbs at the
        // matching guest addresses and takes over their ownership. The old buffer keeps its own
        // bindings until it is retired at its last-use tick, so commands already recorded against
        // it stay valid; the caller submits the bind operations once for the whole replacement.
        new_buffer.TakeOverBlocks(overlap, sparse_binds);
        MarkBufferUsed(overlap);
        MarkBufferUsed(new_buffer);
        DeleteBuffer(overlap_id);
        return;
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    instance.InsertCheckpoint(
        cmdbuf, Vulkan::GpuCheckpoint::BufferMerge,
        Vulkan::GpuCheckpointContext{
            .source_guest_address = overlap.CpuAddr(),
            .destination_guest_address = new_buffer.CpuAddr() + dst_base_offset,
            .transfer_size = copy.size,
            .source_generation = overlap.address_generation,
            .destination_generation = new_buffer.address_generation,
        });
    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);
    // Both allocations are referenced by this command buffer. The replacement is not registered
    // in the LRU yet, so record lifetime independently of eviction recency.
    MarkBufferUsed(overlap);
    MarkBufferUsed(new_buffer);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size, bool allow_texture_gc) {
    const VAddr requested_addr = device_addr;
    const u32 requested_size = wanted_size;
    // Sparse cache buffers move memory blocks between buffers by rebinding them at the same guest
    // address, so every buffer base and size must be a multiple of the sparse block size.
    const u64 alignment = sparse_buffers ? SPARSE_ALIGNMENT : CACHING_PAGESIZE;
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, alignment);
    device_addr = Common::AlignDown(device_addr, alignment);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    u64 used_memory = instance.GetDeviceMemoryUsage();
    const u64 total_budget = instance.GetTotalMemoryBudget();
    OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size, used_memory, total_budget);
    if (sparse_buffers) {
        // All cache buffers are sparse-aligned, so widening to the alignment cannot reach into a
        // buffer that the overlap scan did not already select.
        overlap.begin = Common::AlignDown(overlap.begin, SPARSE_ALIGNMENT);
        overlap.end = Common::AlignUp(overlap.end, SPARSE_ALIGNMENT);
    }
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);

    u64 overlap_bytes{};
    u64 largest_overlap{};
    for (const BufferId overlap_id : overlap.ids) {
        Buffer& overlap_buffer = slot_buffers[overlap_id];
        // Protect the selected source across both the optional chain-breaking submission and the
        // allocation reclaim which follows it. JoinOverlap records the new tick after a flush.
        TouchBuffer(overlap_buffer);
        overlap_bytes += overlap_buffer.SizeBytes();
        largest_overlap = std::max<u64>(largest_overlap, overlap_buffer.SizeBytes());
    }

    // A replacement's old allocations remain alive until the command buffer containing their
    // copies completes. Advance the timeline before one recording tick consumes the allocator's
    // entire safety margin with old versions of the same streaming range.
    const u64 current_tick = scheduler.CurrentTick();
    if (replacement_chain_tick != current_tick) {
        replacement_chain_tick = current_tick;
        replacement_chain_deferred_bytes = 0;
    }
    // Every replaced allocation in this tick stays resident until the tick completes, whatever
    // range it belonged to. Bound that total by submitting once it would exceed the allocator's
    // headroom; a submission costs far less than the emergency reclaim it prevents.
    const u64 chain_deferred_before = replacement_chain_deferred_bytes;
    // Sparse replacements alias the absorbed memory instead of copying it, so no extra storage
    // is retained until the tick completes; the chain-breaking submission is a copy-model measure.
    const bool chain_advanced =
        !sparse_buffers &&
        ShouldAdvanceReplacementChain(chain_deferred_before, overlap_bytes, total_budget);
    if (chain_advanced) {
        scheduler.Flush();
        replacement_chain_tick = scheduler.CurrentTick();
        replacement_chain_deferred_bytes = 0;
        used_memory = instance.GetDeviceMemoryUsage();
    }

    const bool under_pressure =
        AllocationReclaimTarget(used_memory, total_budget, size, false) != 0;
    const bool sample_pressure =
        under_pressure && ShouldLogDiagnosticSample(++pressure_allocation_log_count);
    // Chain advances are frequent under sustained streaming; sample them like pressure events so
    // a synchronous log cannot become the workload (Run 25: 24,841 lines in 11 minutes).
    const bool sample_chain =
        chain_advanced && ShouldLogDiagnosticSample(++chain_advance_log_count);
    // Grow replacements dominate sustained streaming; sample them individually so the operation
    // that rewrites buffer backing most often is observable without flooding a synchronous log.
    const bool sample_grow = overlap.ids.size() == 1 && ShouldLogDiagnosticSample(++grow_log_count);
    ++replacement_count;
    ++stats.replacements;
    if (overlap.ids.empty()) {
        ++stats.replacements_new;
    } else if (overlap.ids.size() == 1) {
        ++stats.replacements_grow;
    } else {
        ++stats.replacements_bridge;
    }
    stats.replaced_bytes += overlap_bytes;
    // Sparse buffers bind only the requested pages now; the rest of the range is address space.
    const u64 allocation_bytes = sparse_buffers ? wanted_size : size;
    if (allocation_bytes >= 64_MB || sample_pressure || sample_chain || sample_grow) {
        LOG_INFO(Render_Vulkan,
                 "Cache buffer allocation: request=[{:#x},{:#x}) ({} bytes), "
                 "resolved=[{:#x},{:#x}) ({} bytes), overlaps={} ({} bytes, largest {}), "
                 "stream_leap={}, usage={} MiB, budget={} MiB, pressure_sample={}, "
                 "stream_suppressed={}, desired_growth={} bytes, speculative={} bytes, "
                 "chain_advanced={}, tick_deferred={} bytes, replacements={}, sparse={}",
                 requested_addr, requested_addr + requested_size, requested_size, overlap.begin,
                 overlap.end, size, overlap.ids.size(), overlap_bytes, largest_overlap,
                 overlap.has_stream_leap, used_memory / 1_MB, total_budget / 1_MB, sample_pressure,
                 overlap.stream_growth_suppressed, overlap.desired_stream_growth,
                 overlap.speculative_bytes, chain_advanced, chain_deferred_before,
                 replacement_count, sparse_buffers);
    }

    ReclaimForAllocation(allocation_bytes, false, allow_texture_gc);
    std::function<void()> allocation_failure_callback;
    if (allocation_reclaim_callback) {
        allocation_failure_callback = [this, allocation_bytes, allow_texture_gc] {
            ReclaimForAllocation(allocation_bytes, true, allow_texture_gc);
        };
    }
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size,
                            allocation_failure_callback, sparse_buffers);
    auto& new_buffer = slot_buffers[new_buffer_id];
    // The allocation was created to satisfy work being recorded now. Initializing its lifetime
    // here prevents a zero last-use tick from ever being interpreted as completed by a later
    // allocation-driven collection in the same command buffer.
    MarkBufferUsed(new_buffer);
    std::vector<vk::SparseMemoryBind> sparse_binds;
    VAddr first_overlap_base = 0;
    u64 first_overlap_size = 0;
    if (!overlap.ids.empty()) {
        const Buffer& first_overlap = slot_buffers[overlap.ids[0]];
        first_overlap_base = first_overlap.CpuAddr();
        first_overlap_size = first_overlap.SizeBytes();
    }
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, sparse_binds);
    }
    // Every replacement, unsampled: grow rewrites buffer backing under live guest ranges more
    // often than any other operation (2,779 times in Run 39) and has never been individually
    // observable. ~4 compact lines per second; the Run 25 flood was an order of magnitude denser.
    LOG_INFO(Render_Vulkan,
             "Buffer replace: reason={}, req=[{:#x},{:#x}), old=[{:#x},{:#x}), new=[{:#x},{:#x}), "
             "overlaps={}, gpu_mod={}, leap={}",
             overlap.ids.empty()       ? "new"
             : overlap.ids.size() == 1 ? "grow"
                                       : "bridge",
             requested_addr, requested_addr + requested_size, first_overlap_base,
             first_overlap_base + first_overlap_size, overlap.begin, overlap.end,
             overlap.ids.size(), gpu_modified_ranges.Intersects(overlap.begin, size),
             overlap.has_stream_leap);
    if (sparse_buffers) {
        // Inherited blocks cover the absorbed buffers; the pages of the request itself may still be
        // unbound (fresh range or the reserve of an absorbed buffer). Reserve pages beyond the
        // request stay unbound until they are first touched.
        const size_t takeover_bind_count = sparse_binds.size();
        new_buffer.EnsureBound(device_addr - overlap.begin, wanted_size, gc_tick,
                               allocation_failure_callback, sparse_binds);
        NoteDemandBinds(
            new_buffer,
            std::span<const vk::SparseMemoryBind>(sparse_binds).subspan(takeover_bind_count));
        scheduler.BindSparse(new_buffer.Handle(), sparse_binds);
        if (!overlap.ids.empty()) {
            // Writes made through the retired handles must be visible through the aliasing one.
            scheduler.EndRendering();
            const vk::MemoryBarrier2 alias_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask =
                    vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            };
            scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &alias_barrier,
            });
        }
    }
    const u64 replacement_tick = scheduler.CurrentTick();
    if (replacement_chain_tick != replacement_tick) {
        replacement_chain_tick = replacement_tick;
        replacement_chain_deferred_bytes = 0;
    }
    replacement_chain_deferred_bytes =
        overlap_bytes > std::numeric_limits<u64>::max() - replacement_chain_deferred_bytes
            ? std::numeric_limits<u64>::max()
            : replacement_chain_deferred_bytes + overlap_bytes;
    new_buffer.IncreaseStreamScore(overlap.stream_score);
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ReclaimForAllocation(u64 allocation_size, bool force, bool allow_texture_gc) {
    if (!allocation_reclaim_callback) {
        return;
    }
    const u64 used_memory = instance.GetDeviceMemoryUsage();
    const u64 total_budget = instance.GetTotalMemoryBudget();
    const u64 reclaim_target =
        AllocationReclaimTarget(used_memory, total_budget, allocation_size, force);
    if (reclaim_target == 0) {
        return;
    }
    allocation_reclaim_callback(reclaim_target, allocation_size, force, allow_texture_gc);
}

void BufferCache::ProcessPreemptiveDownloads() {
    if (Config::readbackSpeed() == Config::ReadbackSpeed::Low ||
        Config::readbackSpeed() == Config::ReadbackSpeed::Disable) {
        return;
    }
    auto* memory = Core::Memory::Instance();

    std::vector<std::pair<VAddr, u64>> processed_ranges;

    preemptive_downloads.ForEach([this, memory, &processed_ranges](
                                     VAddr begin, VAddr end, const PreemptiveDownload& download) {
        if (!scheduler.IsFree(download.done_tick)) {
            return false;
        }
        memory->TryWriteBacking(std::bit_cast<u8*>(download.device_addr), download.staging,
                                download.size);

        processed_ranges.emplace_back(begin, end - begin);
        return true;
    });

    for (const auto& [addr, size] : processed_ranges) {
        preemptive_downloads.Subtract(addr, size);
    }
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        if (buffer.IsSparse()) {
            // Only bound ranges have a valid device address; unbound reserve pages keep a null
            // entry so direct-memory shaders take the fault path, which binds them.
            buffer.buffer.ForEachBoundRange(
                [&](u64 offset, u64 range_size) { WriteBdaEntries(buffer, offset, range_size); });
        } else {
            WriteBdaEntries(buffer, 0, buffer.SizeBytes());
        }
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}
void BufferCache::WriteBdaEntries(const Buffer& buffer, u64 offset, u64 size) {
    const VAddr range_begin = buffer.CpuAddr() + offset;
    const u64 page_begin = range_begin >> CACHING_PAGEBITS;
    const u64 page_end = Common::DivCeil(range_begin + size, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
    bda_addrs.reserve(size_pages);
    const vk::DeviceAddress base =
        buffer.BufferDeviceAddress() + ((page_begin << CACHING_PAGEBITS) - buffer.CpuAddr());
    for (u64 i = 0; i < size_pages; ++i) {
        bda_addrs.push_back(base + (i << CACHING_PAGEBITS));
    }
    WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress), bda_addrs.data(),
                    bda_addrs.size() * sizeof(vk::DeviceAddress));
}

void BufferCache::NoteDemandBinds(Buffer& buffer, std::span<const vk::SparseMemoryBind> binds) {
    if (reclaim_ghosts.empty()) {
        return;
    }
    for (const vk::SparseMemoryBind& bind : binds) {
        const VAddr addr = buffer.CpuAddr() + bind.resourceOffset;
        u64 max_distance = 0;
        u64 hit_pages = 0;
        for (VAddr page = addr; page < addr + bind.size; page += SPARSE_ALIGNMENT) {
            const auto it = reclaim_ghosts.find(page >> 16);
            if (it == reclaim_ghosts.end()) {
                continue;
            }
            max_distance = std::max(max_distance, gc_tick - it->second);
            ++hit_pages;
            reclaim_ghosts.erase(it);
        }
        if (hit_pages == 0) {
            continue;
        }
        ++stats.ghost_hits;
        stats.ghost_hit_bytes += hit_pages * SPARSE_ALIGNMENT;
        const u32 bucket = max_distance < 256    ? 0
                           : max_distance < 512  ? 1
                           : max_distance < 1024 ? 2
                           : max_distance < 2048 ? 3
                           : max_distance < 4096 ? 4
                                                 : 5;
        ++stats.ghost_dist[bucket];
        // Twice the observed distance, clamped: enough that the next revisit at the same cadence
        // finds the block still bound, without pinning it forever.
        const u64 protection = std::clamp<u64>(2 * max_distance, 512, 8192);
        buffer.ProtectSparseRange(bind.resourceOffset, bind.size, gc_tick + protection);
    }
}

void BufferCache::EnsureRangeBound(Buffer& buffer, VAddr device_addr, u64 size) {
    if (!buffer.IsSparse()) {
        return;
    }
    const u64 offset = buffer.Offset(device_addr);
    // Every synchronized access stamps the covered blocks' GC epoch; the cold-block sweep only
    // unbinds blocks whose stamp is hundreds of epochs old.
    buffer.TouchSparseRange(offset, size, gc_tick);
    if (buffer.IsRangeBound(offset, size)) {
        return;
    }
    // Demand binding of reserve pages is an allocation like any other: keep it inside the
    // budget. Texture collection is avoided because this can run under the texture cache mutex.
    ReclaimForAllocation(size, false, false);
    std::function<void()> allocation_failure_callback;
    if (allocation_reclaim_callback) {
        allocation_failure_callback = [this, size] { ReclaimForAllocation(size, true, false); };
    }
    std::vector<vk::SparseMemoryBind> sparse_binds;
    const u64 bound =
        buffer.EnsureBound(offset, size, gc_tick, allocation_failure_callback, sparse_binds);
    if (sparse_binds.empty()) {
        return;
    }
    ++stats.demand_bindings;
    stats.demand_bound_bytes += bound;
    NoteDemandBinds(buffer, sparse_binds);
    scheduler.BindSparse(buffer.Handle(), sparse_binds);
    // Reads of an unbound sparse range return undefined data on non-strict-residency hardware.
    // A coverage accounting bug here would surface as transient garbage geometry, so verify the
    // invariant at the only place that can restore it.
    if (!buffer.IsRangeBound(offset, size)) {
        LOG_ERROR(Render_Vulkan,
                  "Sparse range still unbound after demand binding: buffer={:#x}, "
                  "offset={:#x}, size={:#x}",
                  buffer.CpuAddr(), offset, size);
    }
    // Only registered buffers reach this path (every caller resolved the buffer through the page
    // table first), so the new pages can be published to direct-memory shaders immediately.
    for (const vk::SparseMemoryBind& bind : sparse_binds) {
        WriteBdaEntries(buffer, bind.resourceOffset, bind.size);
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    // Synchronization is always followed by a GPU access to this cache buffer, even when the
    // memory tracker finds no dirty CPU pages and therefore emits no upload. This is especially
    // important for direct-memory shaders, which dereference the BDA page table at execution time.
    MarkBufferUsed(buffer);
    EnsureRangeBound(buffer, device_addr, size);
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    if (src_buffer) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        instance.InsertCheckpoint(cmdbuf, Vulkan::GpuCheckpoint::BufferUpload,
                                  Vulkan::GpuCheckpointContext{
                                      .destination_guest_address = device_addr,
                                      .transfer_size = total_size_bytes,
                                      .destination_generation = buffer.address_generation,
                                  });
        cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(const Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(const Buffer& buffer, VAddr device_addr, u32 size) {
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size, bool is_written) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, is_written, false);
    });
}

void BufferCache::CommitPendingGpuRanges() {
    size_t total_size_bytes = 0;
    gpu_modified_ranges_pending.ForEach([&](VAddr begin, VAddr end) {
        memory_tracker->ForEachPreemptiveFlushPage(begin, end - begin, [&](VAddr page_addr) {
            const BufferId buffer_id = page_table[page_addr >> CACHING_PAGEBITS].buffer_id;
            const Buffer& buffer = slot_buffers[buffer_id];
            const VAddr start_addr = std::max(page_addr, begin);
            const VAddr end_addr = std::min(page_addr + PageManager::PAGE_SIZE, end);
            const u32 size = end_addr - start_addr;
            preemptive_copies[buffer_id].emplace_back(buffer.Offset(start_addr), total_size_bytes,
                                                      size);
            total_size_bytes += size;
            return false;
        });
        SynchronizeBuffersInRange(begin, end - begin, true);
    });
    gpu_modified_ranges.m_ranges_set += gpu_modified_ranges_pending.m_ranges_set;
    gpu_modified_ranges_pending.Clear();
    if (!preemptive_copies.empty()) {
        const u64 done_tick = scheduler.CurrentTick();
        const auto [staging, offset] = download_buffer.Map(total_size_bytes);
        download_buffer.Commit();
        for (auto it = preemptive_copies.begin(); it != preemptive_copies.end(); ++it) {
            const BufferId buffer_id = it.key();
            auto& copies = it.value();
            const Buffer& buffer = slot_buffers[buffer_id];
            for (auto& copy : copies) {
                const VAddr start_addr = buffer.CpuAddr() + copy.srcOffset;
                preemptive_downloads.Add(start_addr, copy.size,
                                         PreemptiveDownload{
                                             .device_addr = start_addr,
                                             .size = copy.size,
                                             .staging = staging + copy.dstOffset,
                                             .done_tick = done_tick,

                                         });
                copy.dstOffset += offset;
            }
            scheduler.EndRendering();
            const auto cmdbuf = scheduler.CommandBuffer();
            const u64 copy_bytes = std::accumulate(
                copies.begin(), copies.end(), u64{},
                [](u64 total, const vk::BufferCopy& copy) { return total + copy.size; });
            instance.InsertCheckpoint(cmdbuf, Vulkan::GpuCheckpoint::BufferDownload,
                                      Vulkan::GpuCheckpointContext{
                                          .source_guest_address = buffer.CpuAddr(),
                                          .transfer_size = copy_bytes,
                                          .source_generation = buffer.address_generation,
                                      });
            cmdbuf.copyBuffer(buffer.Handle(), download_buffer.Handle(), copies);
        }
        preemptive_copies.clear();
        scheduler.Flush();
    }
}

void BufferCache::MemoryBarrier() {
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    vk::MemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::SweepColdSparseBlocks(GcBudget& budget, GcResult& result) {
    // Whole-buffer eviction cannot reach a merged streaming span that is partially warm: any
    // touch anywhere keeps the buffer's LRU entry fresh while most of its 64 KiB blocks go cold
    // (Run 42: 2.2 GiB bound against a 1.7 GiB unmet reclaim target). Blocks whose last
    // synchronized access is hundreds of epochs old are dropped: their ranges hold no
    // GPU-authored data, so guest memory is authoritative and a later touch simply demand-binds
    // and re-uploads. The unbind travels the batched BindSparse path, so previously submitted
    // work still reads the old binding; the allocation itself is freed only after the tick that
    // carries the unbind completes.
    const u64 reclaimed_delta = stats.block_reclaimed_bytes - adapt_reclaimed_snapshot;
    const u64 demand_delta = stats.demand_bound_bytes - adapt_demand_snapshot;
    if (reclaimed_delta >= 64_MB) {
        // demand_bound_bytes also counts first-touch binds of genuinely new ranges, so the ratio
        // overestimates round-tripping; that only errs toward keeping blocks longer. Run 44's
        // single-window reaction cycled 256<->4096 eleven times each way (a raise zeroes the very
        // rebind signal that justified it), so adaptation now needs two consecutive windows on
        // the same side, and the ceiling is 2048: the 4096 band cost ~478 MiB of average usage
        // for 65-82 MiB of reclaim per window.
        const u64 previous_age = adaptive_block_age;
        if (demand_delta * 2 >= reclaimed_delta) {
            ++adapt_raise_streak;
            adapt_lower_streak = 0;
        } else if (demand_delta * 8 <= reclaimed_delta) {
            ++adapt_lower_streak;
            adapt_raise_streak = 0;
        } else {
            adapt_raise_streak = 0;
            adapt_lower_streak = 0;
        }
        if (adapt_raise_streak >= 2 && adaptive_block_age < 2048) {
            adaptive_block_age *= 2;
            adapt_raise_streak = 0;
        } else if (adapt_lower_streak >= 2 && adaptive_block_age > 256) {
            adaptive_block_age /= 2;
            adapt_lower_streak = 0;
        }
        if (adaptive_block_age != previous_age) {
            LOG_INFO(Render_Vulkan,
                     "Sparse block reclaim age adapted: {} -> {} epochs (reclaimed={} MiB, "
                     "demand_bound={} MiB since last adaptation)",
                     previous_age, adaptive_block_age, reclaimed_delta / 1_MB, demand_delta / 1_MB);
        }
        adapt_reclaimed_snapshot = stats.block_reclaimed_bytes;
        adapt_demand_snapshot = stats.demand_bound_bytes;
    }
    const u64 min_age =
        (budget.pressure == GcPressure::Critical || budget.overshoot)
            ? std::max<u64>(SparseBlockReclaimMinAge(budget.pressure, true), adaptive_block_age / 4)
            : adaptive_block_age;
    if (gc_tick <= min_age) {
        return;
    }
    const u64 cutoff = gc_tick - min_age;
    u64 sweep_cap = std::min<u64>(budget.bytes_remaining, 512_MB);
    ForEachBufferInRange(0, std::numeric_limits<VAddr>::max(), [&](BufferId id, Buffer& buffer) {
        if (sweep_cap == 0 || !buffer.IsSparse() || buffer.is_deleted) {
            return;
        }
        std::vector<vk::SparseMemoryBind> unbinds;
        boost::container::small_vector<VmaAllocation, 16> freed;
        const u64 reclaimed = buffer.ReclaimColdSparseBlocks(
            cutoff, gc_tick, sweep_cap,
            [&](u64 offset, u64 block_size) {
                const VAddr addr = buffer.CpuAddr() + offset;
                return !gpu_modified_ranges_pending.Intersects(addr, block_size) &&
                       !preemptive_downloads.Intersects(addr, block_size) &&
                       !IsRegionGpuModified(addr, block_size);
            },
            [&](u64 offset, u64 block_size, VmaAllocation allocation) {
                const VAddr addr = buffer.CpuAddr() + offset;
                // Metadata ghost per 64 KiB page: a later demand bind of this range reveals the
                // true re-reference distance and pins the rebound blocks accordingly.
                for (VAddr page = addr; page < addr + block_size; page += SPARSE_ALIGNMENT) {
                    reclaim_ghosts[page >> 16] = gc_tick;
                }
                // Guest memory becomes the sole copy: any later use must re-upload.
                memory_tracker->MarkRegionAsCpuModified(addr, block_size);
                // Null the BDA entries so direct-memory shaders take the fault path, which
                // rebinds and synchronizes the pages.
                const u64 page_begin = addr >> CACHING_PAGEBITS;
                const u64 page_end = Common::DivCeil(addr + block_size, CACHING_PAGESIZE);
                const u64 pt_offset =
                    bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
                bda_pagetable_buffer.Fill(pt_offset,
                                          (page_end - page_begin) * sizeof(vk::DeviceAddress), 0);
                freed.push_back(allocation);
            },
            unbinds);
        if (unbinds.empty()) {
            return;
        }
        scheduler.BindSparse(buffer.Handle(), unbinds);
        scheduler.DeferOperation(
            [allocator = instance.GetAllocator(),
             allocations = std::vector<VmaAllocation>(freed.begin(), freed.end())] {
                for (VmaAllocation allocation : allocations) {
                    vmaFreeMemory(allocator, allocation);
                }
            });
        budget.Reclaim(reclaimed);
        result.reclaimed_bytes += reclaimed;
        stats.block_reclaims += unbinds.size();
        stats.block_reclaimed_bytes += reclaimed;
        sweep_cap -= std::min(sweep_cap, reclaimed);
    });
    if (reclaim_ghosts.size() > 131072) {
        // Entries this old are dead history (nothing re-references them); drop them so the map
        // stays a bounded metadata cache rather than a leak.
        const u64 expiry = gc_tick > 8192 ? gc_tick - 8192 : 0;
        for (auto it = reclaim_ghosts.begin(); it != reclaim_ghosts.end();) {
            it = it->second < expiry ? reclaim_ghosts.erase(it) : std::next(it);
        }
    }
}

void BufferCache::AdvanceGcEpoch() {
    ++gc_tick;
}

GcResult BufferCache::RunGarbageCollector(GcBudget& budget) {
    GcResult result;
    if (!budget.Active()) {
        return result;
    }

    // Under critical pressure consider older resources sooner, but never resources touched in the
    // current epochs. GPU-authored buffers are skipped: reading each one back synchronously here
    // can serialize dozens of scheduler.Finish() calls on the command processor.
    const u64 min_age = GcMinimumAge(budget.pressure, budget.overshoot);
    if (gc_tick <= min_age) {
        return result;
    }
    const u64 cutoff = gc_tick - min_age;
    const u32 inspection_multiplier = budget.pressure == GcPressure::Critical ? 8 : 4;
    static constexpr u32 MaxGcInspections = 4096;
    u32 inspections_remaining = budget.objects_remaining > MaxGcInspections / inspection_multiplier
                                    ? MaxGcInspections
                                    : budget.objects_remaining * inspection_multiplier;

    struct Candidate {
        BufferId id;
        u64 size;
        u64 last_use_tick;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(inspections_remaining);
    u64 candidate_bytes{};

    u32 cheap_scans_remaining = GcMaxCheapScans;
    const auto collect = [&](BufferId buffer_id) {
        if (inspections_remaining == 0 || cheap_scans_remaining == 0 ||
            (budget.pressure != GcPressure::Critical &&
             candidate_bytes >= budget.bytes_remaining)) {
            return true;
        }
        --cheap_scans_remaining;
        if (IsBufferInvalid(buffer_id)) {
            return false;
        }

        Buffer& buffer = slot_buffers[buffer_id];
        const VAddr address = buffer.CpuAddr();
        const u64 guest_size = buffer.SizeBytes();
        const u64 allocation_size = buffer.AllocationSizeBytes();
        ASSERT_MSG(allocation_size != 0 || buffer.IsSparse(),
                   "Tracked buffer has no physical allocation");
        // A size test is not an inspection: consuming the inspection budget on it would let the
        // tiny old tail of the LRU hide every large object behind it.
        if (ShouldSkipSmallEviction(allocation_size, budget.emergency)) {
            ++result.skipped_small;
            return false;
        }
        --inspections_remaining;
        ++result.inspected_objects;

        if (IsResourceInFlight(buffer.LastUseTick(), scheduler.CurrentTick())) {
            ++result.skipped_in_flight;
            return false;
        }
        if (budget.require_completed &&
            !CanReclaimWithoutWait(buffer.LastUseTick(), budget.completed_tick)) {
            ++result.skipped_pending;
            return false;
        }

        // Pending or preemptive ranges still reference this buffer and cannot be safely
        // reconstructed from guest memory yet. Reconsider it on a later pass.
        if (gpu_modified_ranges_pending.Intersects(address, guest_size) ||
            preemptive_downloads.Intersects(address, guest_size)) {
            ++result.skipped_gpu_modified;
            return false;
        }

        // Keep GPU-authored data resident. A future batched readback path can make these candidates
        // reclaimable without placing a blocking wait inside the garbage collector.
        if (IsRegionGpuModified(address, guest_size)) {
            ++result.skipped_gpu_modified;
            return false;
        }

        candidates.push_back({buffer_id, allocation_size, buffer.LastUseTick()});
        candidate_bytes += allocation_size;
        return false;
    };

    lru_cache.ForEachItemBelow(cutoff, collect);
    if (budget.pressure == GcPressure::Critical) {
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) { return lhs.size > rhs.size; });
    }

    for (const Candidate& candidate : candidates) {
        if (!budget.Active()) {
            break;
        }
        Buffer& buffer = slot_buffers[candidate.id];
        const VAddr address = buffer.CpuAddr();

        // Candidate discovery and retirement are intentionally separate. Revalidate the lifetime
        // at the destructive boundary so a use recorded in between cannot be retired based on the
        // stale candidate snapshot.
        if (buffer.LastUseTick() != candidate.last_use_tick ||
            IsResourceInFlight(buffer.LastUseTick(), scheduler.CurrentTick())) {
            ++result.skipped_in_flight;
            continue;
        }
        if (budget.require_completed &&
            !CanReclaimWithoutWait(buffer.LastUseTick(), budget.completed_tick)) {
            ++result.skipped_pending;
            continue;
        }

        // With no GPU-authored data left, guest memory becomes the sole authoritative copy after
        // eviction. Release the MemoryTracker write watcher before unregistering the host buffer;
        // otherwise InvalidateMemory can no longer find an owner capable of releasing it.
        memory_tracker->MarkRegionAsCpuModified(address, buffer.SizeBytes());
        DeleteBuffer(candidate.id, budget.collect_retirements ? &result : nullptr,
                     BufferRetirementReason::GarbageCollection);
        budget.Reclaim(candidate.size);
        result.reclaimed_bytes += candidate.size;
        ++result.evicted_objects;
    }
    if (sparse_buffers && budget.Active() && Config::getUseSparseBlockReclaim()) {
        SweepColdSparseBlocks(budget, result);
    }
    return result;
}

BufferCache::Statistics BufferCache::GetStatistics() {
    Statistics snapshot = stats;
    snapshot.sparse = sparse_buffers;
    snapshot.block_reclaim_age = adaptive_block_age;
    snapshot.ghost_live = reclaim_ghosts.size();
    snapshot.live_buffers = 0;
    snapshot.bound_bytes = 0;
    lru_cache.ForEachItemBelow(std::numeric_limits<u64>::max(), [&](BufferId id) {
        if (!IsBufferInvalid(id)) {
            ++snapshot.live_buffers;
            snapshot.bound_bytes += slot_buffers[id].AllocationSizeBytes();
        }
    });
    return snapshot;
}

void BufferCache::MarkBufferUsed(Buffer& buffer) {
    buffer.SetLastUseTick(RecordResourceUse(buffer.LastUseTick(), scheduler.CurrentTick()));
}

void BufferCache::TouchBuffer(Buffer& buffer) {
    MarkBufferUsed(buffer);
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id, GcResult* gc_result,
                               BufferRetirementReason reason) {
    Buffer& buffer = slot_buffers[buffer_id];
    const u64 last_use_tick = buffer.LastUseTick();
    const u64 allocation_size = buffer.AllocationSizeBytes();
    const bool diagnose_retirement = gc_result != nullptr;
    buffer.SetRetirementContext(reason, scheduler.CurrentTick());
    Unregister(buffer_id);

    Common::UniqueFunction<void> retirement = [this, buffer_id, allocation_size, last_use_tick,
                                               diagnose_retirement] {
        const auto start = std::chrono::steady_clock::now();
        slot_buffers.erase(buffer_id);
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (diagnose_retirement && elapsed_us >= 10'000) {
            LOG_WARNING(Render_Vulkan,
                        "Slow GC buffer retirement: id={}, allocation={} bytes, "
                        "last_use_tick={}, buffer_destroy={} us",
                        buffer_id.index, allocation_size, last_use_tick, elapsed_us);
        }
    };
    if (gc_result) {
        gc_result->QueueRetirement(last_use_tick, std::move(retirement));
    } else {
        // General cache invalidation remains tied to the current command buffer. Allocation-driven
        // GC instead returns this operation to the rasterizer, which waits for the exact last use.
        scheduler.DeferOperation(std::move(retirement));
    }

    buffer.is_deleted = true;
}

} // namespace VideoCore
