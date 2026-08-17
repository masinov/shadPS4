// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include <xxhash.h>

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& page_manager_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, page_manager{page_manager_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)},
      readback_linear_images{Config::getReadbackLinearImages()} {}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }

    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const auto image_addr = image.info.guest_address;
    const auto image_size = image.info.guest_size;
    const auto image_mips = image.info.resources.levels;
    u32 copy_size = 0;
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image_mips; ++mip) {
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> mip, 1u) : 1u;

        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[mip];
        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
        copy_size += mip_size;
    }

    if (buffer_copies.empty()) {
        return;
    }

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    const auto [mapping_data, mapping_offset] = download_buffer.Map(copy_size, true);
    tile_manager.TileImage(image, buffer_copies, download_buffer.Handle(), mapping_offset,
                           copy_size);

    scheduler.Finish();

    const u32 write_size = static_cast<u32>(std::min<u64>(copy_size, image_size));

    std::vector<u8> download_data;
    download_data.resize(write_size);
    std::memcpy(download_data.data(), mapping_data, write_size);

    if (sync) {
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image_addr),
                                                  download_data.data(), write_size);
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image_addr, download_data = std::move(download_data), write_size] {
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download_data.data(), write_size);
            });
    }
}

bool TextureCache::IsReadbackViable(ImageId image_id, Image& image) {
    // 96% of Run 37 nominations aborted at writeback because another GPU-modified image aliases
    // the range; every such nomination cost a recorded tiler dispatch for nothing. Decide before
    // recording. An aliased group is still viable when this image dominates it: every other
    // GPU-modified alias lies fully inside this image's range and was accessed no later, so this
    // image's bytes are the authoritative content of the whole range (guest memory holds one copy;
    // on hardware the aliases would read exactly these bytes).
    bool viable = true;
    ForEachImageInRegion(
        image.info.guest_address, image.info.guest_size, [&](ImageId other_id, Image& other) {
            if (other_id == image_id || False(other.flags & ImageFlagBits::GpuModified)) {
                return;
            }
            const bool contained = other.info.guest_address >= image.info.guest_address &&
                                   other.info.guest_address + other.info.guest_size <=
                                       image.info.guest_address + image.info.guest_size;
            if (!contained || other.tick_accessed_last > image.tick_accessed_last ||
                other.binding.is_bound || other.binding.is_target) {
                viable = false;
            }
        });
    return viable;
}

void TextureCache::RecordEvictionReadbacks() {
    std::scoped_lock lock{mutex};
    ApplyReadyReadbacks();
    if (eviction_readbacks.empty()) {
        return;
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    for (const ImageId image_id : eviction_readbacks) {
        Image& image = slot_images[image_id];
        const auto clear_flag = [&] { image.flags &= ~ImageFlagBits::EvictionReadback; };
        if (False(image.flags & ImageFlagBits::Registered) ||
            False(image.flags & ImageFlagBits::GpuModified) || image.binding.is_bound ||
            image.binding.is_target) {
            clear_flag();
            continue;
        }

        const VAddr image_addr = image.info.guest_address;
        const u64 image_size = image.info.guest_size;
        u32 copy_size = 0;
        boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
        for (u32 mip = 0; mip < image.info.resources.levels; ++mip) {
            const u32 width = std::max(image.info.size.width >> mip, 1u);
            const u32 height = std::max(image.info.size.height >> mip, 1u);
            const u32 depth =
                image.info.props.is_volume ? std::max(image.info.size.depth >> mip, 1u) : 1u;
            const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[mip];
            const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
            const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
            buffer_copies.push_back(vk::BufferImageCopy{
                .bufferOffset = mip_offset,
                .bufferRowLength = mip_pitch,
                .bufferImageHeight = mip_height,
                .imageSubresource{
                    .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                    .mipLevel = mip,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {extent_width, extent_height, depth},
            });
            copy_size += mip_size;
        }
        if (copy_size == 0) {
            clear_flag();
            continue;
        }
        // Never wait for staging space here; a full ring simply retries on a later pass.
        const auto [mapping_data, mapping_offset] = download_buffer.Map(copy_size, 1, false);
        if (!mapping_data) {
            clear_flag();
            continue;
        }

        scheduler.EndRendering();
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
        tile_manager.TileImage(image, buffer_copies, download_buffer.Handle(), mapping_offset,
                               copy_size);
        download_buffer.Commit();
        image.tick_accessed_last = scheduler.CurrentTick();

        const u64 record_access_tick = image.tick_accessed_last;
        const u32 write_size = static_cast<u32>(std::min<u64>(copy_size, image_size));
        eviction_readback_inflight.insert(image_id.index);
        // The callback may execute inside a flush performed while the cache mutex is held, so it
        // only snapshots the staged bytes (the download-ring region is reusable from this moment)
        // and defers all cache-state work to the next RecordEvictionReadbacks call.
        scheduler.DeferOperation(
            [this, image_id, image_addr, image_size, mapping_data, write_size, record_access_tick] {
                ReadyReadback ready{
                    .image_id = image_id,
                    .guest_address = image_addr,
                    .guest_size = image_size,
                    .record_access_tick = record_access_tick,
                };
                ready.data.resize(write_size);
                std::memcpy(ready.data.data(), mapping_data, write_size);
                std::unique_lock lk{download_images_mutex};
                ready_readbacks.push_back(std::move(ready));
            });
    }
    eviction_readbacks.clear();
}

void TextureCache::ApplyReadyReadbacks() {
    std::vector<ReadyReadback> ready;
    {
        std::unique_lock lk{download_images_mutex};
        ready.swap(ready_readbacks);
    }
    for (ReadyReadback& readback : ready) {
        if (eviction_readback_inflight.erase(readback.image_id.index) == 0) {
            // Deleted before completion; the slot may already hold a different image.
            continue;
        }
        Image& image = slot_images[readback.image_id];
        const bool valid = True(image.flags & ImageFlagBits::Registered) &&
                           True(image.flags & ImageFlagBits::EvictionReadback) &&
                           image.info.guest_address == readback.guest_address;
        if (!valid) {
            continue;
        }
        image.flags &= ~ImageFlagBits::EvictionReadback;
        if (image.tick_accessed_last != readback.record_access_tick) {
            // Used (and possibly re-written) after the download was recorded; the staged copy may
            // be stale. Abandon; the collector can nominate it again later.
            ++readback_stats.aborted_reused;
            continue;
        }
        // GPU ticks say nothing about CPU writes: if the guest streamed new data into this range
        // after the download was recorded, writing the stale GPU copy back would corrupt it — the
        // exact hazard behind Run 35's exploding geometry and untraceable crash. Dirty flags are
        // set whenever the range was CPU-written (write watcher or untrack), so any dirt aborts.
        if (True(image.flags & ImageFlagBits::Dirty)) {
            ++readback_stats.aborted_dirty;
            continue;
        }
        // Revalidate the conditions that were checked at nomination; both alias classes can have
        // appeared while the download was in flight (Run 36's vertex-buffer clobber came through
        // exactly this window).
        if (buffer_cache.IsRegionGpuModified(readback.guest_address, readback.guest_size)) {
            ++readback_stats.aborted_buffer_alias;
            continue;
        }
        if (!IsReadbackViable(readback.image_id, image)) {
            ++readback_stats.aborted_image_alias;
            continue;
        }
        ++readback_stats.applied;
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(readback.guest_address),
                                                  readback.data.data(), readback.data.size());
        image.flags &= ~ImageFlagBits::GpuModified;
        buffer_cache.MarkRegionAsFlushed(readback.guest_address, readback.guest_size);
        // Dominated aliases (older, fully contained) are superseded byte-for-byte by the written
        // range. Their device copies are stale; make guest memory their source of truth too.
        ForEachImageInRegion(readback.guest_address, readback.guest_size,
                             [&](ImageId other_id, Image& other) {
                                 if (other_id == readback.image_id ||
                                     False(other.flags & ImageFlagBits::GpuModified)) {
                                     return;
                                 }
                                 other.flags &= ~ImageFlagBits::GpuModified;
                                 other.flags |= ImageFlagBits::CpuDirty;
                             });
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size, ImageId exclude_image_id) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        if (exclude_image_id && image_id == exclude_image_id) {
            return;
        }
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

TextureCache::DebugPageState TextureCache::GetDebugPageState(VAddr addr) {
    const VAddr page_begin = PageManager::GetPageAddr(addr);
    const VAddr page_end = page_begin + PageManager::PAGE_SIZE;
    DebugPageState result;

    std::scoped_lock lock{mutex};
    ForEachImageInRegion(page_begin, PageManager::PAGE_SIZE, [&](ImageId image_id, Image& image) {
        ++result.registered_images;
        if (image.IsTracked() && image.track_addr < page_end && page_begin < image.track_addr_end) {
            if (result.tracked_images == 0) {
                result.first_tracked_image = image_id.index;
                result.image_begin = image.info.guest_address;
                result.image_end = image.info.guest_address + image.info.guest_size;
                result.track_begin = image.track_addr;
                result.track_end = image.track_addr_end;
            }
            ++result.tracked_images;
        }
        if (True(image.flags & (ImageFlagBits::CpuDirty | ImageFlagBits::MaybeCpuDirty))) {
            ++result.cpu_dirty_images;
        }
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            ++result.gpu_modified_images;
        }
    });
    return result;
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::MarkAsMaybeReused(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(addr, size, [&](ImageId image_id, Image& image) {
        image.flags |= ImageFlagBits::MaybeReused;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};
    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (!image.info.props.is_tiled && image.info.guest_address != 0 &&
            image.info.props.is_volume) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));
        const u32 size = w * h * image.info.num_bits >> (3 + image.info.props.is_block ? 4 : 0);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    if (auto barrier = in_buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
    }

    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    image.Upload(image_copies, buffer, offset);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    lru_cache.Free(image.lru_id);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        page_manager.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    page_manager.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    page_manager.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        page_manager.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = page_manager.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    page_manager.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = page_manager.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    page_manager.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::AdvanceGcEpoch() {
    ++gc_tick;
}

GcResult TextureCache::RunGarbageCollector(GcBudget& budget) {
    GcResult result;
    if (!budget.Active()) {
        return result;
    }

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

    std::scoped_lock lock{mutex};

    struct Candidate {
        ImageId id;
        u64 size;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(inspections_remaining);
    u64 candidate_bytes{};

    u32 cheap_scans_remaining = GcMaxCheapScans;
    const auto collect = [&](ImageId image_id) {
        if (inspections_remaining == 0 || cheap_scans_remaining == 0 ||
            (budget.pressure != GcPressure::Critical &&
             candidate_bytes >= budget.bytes_remaining)) {
            return true;
        }
        --cheap_scans_remaining;

        auto& image = slot_images[image_id];
        // A size test is not an inspection; see the buffer collector.
        if (image.info.pixel_format != vk::Format::eUndefined &&
            ShouldSkipSmallEviction(image.AllocationSizeBytes(), budget.emergency)) {
            ++result.skipped_small;
            return false;
        }
        --inspections_remaining;
        ++result.inspected_objects;
        // Stencil-only guest ranges are deliberately represented by registered Image entries with
        // an undefined format and no Vulkan backing. They participate in overlap resolution and
        // depth association, but cannot contribute bytes toward a device-memory reclaim target.
        if (image.info.pixel_format == vk::Format::eUndefined) {
            ASSERT_MSG(image.AllocationSizeBytes() == 0,
                       "Undefined-format image unexpectedly owns a physical allocation");
            ++result.skipped_unallocated;
            return false;
        }

        if (image.binding.is_bound || image.binding.is_target) {
            ++result.skipped_bound;
            return false;
        }

        if (IsResourceInFlight(image.tick_accessed_last, scheduler.CurrentTick())) {
            ++result.skipped_in_flight;
            return false;
        }
        if (budget.require_completed &&
            !CanReclaimWithoutWait(image.tick_accessed_last, budget.completed_tick)) {
            ++result.skipped_pending;
            return false;
        }

        if (True(image.flags & ImageFlagBits::GpuModified)) {
            // GPU-authored contents cannot be discarded, but they can be written back to guest
            // memory asynchronously; once that completes the image is CPU-authoritative and a
            // later pass can evict it. Nominate a bounded number of large candidates per pass.
            if (False(image.flags & ImageFlagBits::EvictionReadback) &&
                False(image.flags & ImageFlagBits::Dirty) &&
                image.info.pixel_format != vk::Format::eUndefined &&
                image.AllocationSizeBytes() >= 4_MB && eviction_readbacks.size() < 4 &&
                !buffer_cache.IsRegionGpuModified(image.info.guest_address,
                                                  image.info.guest_size) &&
                IsReadbackViable(image_id, image)) {
                image.flags |= ImageFlagBits::EvictionReadback;
                eviction_readbacks.push_back(image_id);
                ++readback_stats.nominated;
            }
            ++result.skipped_gpu_modified;
            return false;
        }

        const u64 size = image.AllocationSizeBytes();
        ASSERT_MSG(size != 0, "Tracked image has no physical allocation");
        candidates.push_back({image_id, size});
        candidate_bytes += size;
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
        FreeImage(candidate.id, budget.collect_retirements ? &result : nullptr);
        budget.Reclaim(candidate.size);
        result.reclaimed_bytes += candidate.size;
        ++result.evicted_objects;
    }
    return result;
}

void TextureCache::TouchImage(Image& image) {
    image.tick_accessed_last = scheduler.CurrentTick();
    lru_cache.Touch(image.lru_id, gc_tick);

    // Image is still valid
    image.flags &= ~ImageFlagBits::MaybeReused;
}
void TextureCache::DeleteImage(ImageId image_id, GcResult* gc_result) {
    Image& image = slot_images[image_id];
    const u64 last_use_tick = image.tick_accessed_last;
    const u64 allocation_size = image.AllocationSizeBytes();
    const bool diagnose_retirement = gc_result != nullptr;
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        eviction_readback_inflight.erase(image_id.index);
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    Common::UniqueFunction<void> retirement = [this, image_id, allocation_size, last_use_tick,
                                               diagnose_retirement] {
        const auto start = std::chrono::steady_clock::now();
        Image& image = slot_images[image_id];
        u32 view_count{};
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
                ++view_count;
            }
        }
        const auto views_end = std::chrono::steady_clock::now();
        slot_images.erase(image_id);
        const auto image_end = std::chrono::steady_clock::now();

        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(image_end - start).count();
        if (diagnose_retirement && total_us >= 10'000) {
            const auto views_us =
                std::chrono::duration_cast<std::chrono::microseconds>(views_end - start).count();
            const auto image_us =
                std::chrono::duration_cast<std::chrono::microseconds>(image_end - views_end)
                    .count();
            LOG_WARNING(Render_Vulkan,
                        "Slow GC image retirement: id={}, allocation={} bytes, last_use_tick={}, "
                        "views={}, view_destroy={} us, image_destroy={} us, total={} us",
                        image_id.index, allocation_size, last_use_tick, view_count, views_us,
                        image_us, total_us);
        }
    };
    if (gc_result) {
        gc_result->QueueRetirement(last_use_tick, std::move(retirement));
    } else {
        scheduler.DeferOperation(std::move(retirement));
    }
}

} // namespace VideoCore
