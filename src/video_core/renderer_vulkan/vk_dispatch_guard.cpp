// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/host_shaders/dispatch_clamp_comp.h"
#include "video_core/host_shaders/draw_clamp_comp.h"
#include "video_core/renderer_vulkan/vk_dispatch_guard.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
// Up to 16 bytes of arguments per slot, which also aligns storage-descriptor offsets. 4096 slots
// wrap only after 4096 guarded commands, far beyond any realistic in-flight window.
constexpr u64 SlotSize = 16;
constexpr u64 NumSlots = 4096;
// Written into a slot when its pre-pass is recorded; still there if the GPU never ran it.
constexpr u32 NotExecuted = 0xDEADC0DE;
constexpr u64 DispatchCounterOffset = 0;
constexpr u64 DrawCounterOffset = sizeof(u32);
// PS4 dispatch dimension limit; also within every Vulkan implementation's minimum guarantee.
constexpr u32 MaxSaneGroups = 65535;
} // namespace

DispatchGuard::DispatchGuard(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      args_ring{instance_,
                scheduler_,
                VideoCore::MemoryUsage::Download,
                0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                SlotSize * NumSlots},
      clamp_counter{instance_, scheduler_, VideoCore::MemoryUsage::Download, 0,
                    vk::BufferUsageFlagBits::eStorageBuffer, 2 * sizeof(u32)},
      slot_info(NumSlots) {
    const auto device = instance.GetDevice();
    SetObjectName(device, args_ring.Handle(), "Indirect Dispatch Args Ring");
    SetObjectName(device, clamp_counter.Handle(), "Indirect Dispatch Clamp Counter");
    std::memset(clamp_counter.mapped_data.data(), 0, 2 * sizeof(u32));
    std::memset(args_ring.mapped_data.data(), 0, SlotSize * NumSlots);

    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    desc_layout = Check(device.createDescriptorSetLayoutUnique(desc_layout_ci));

    const auto module =
        Compile(HostShaders::DISPATCH_CLAMP_COMP, vk::ShaderStageFlagBits::eCompute, device);
    SetObjectName(device, module, "Indirect Dispatch Clamp");

    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(u32),
    };
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &(*desc_layout),
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_range,
    };
    pipeline_layout = Check(device.createPipelineLayoutUnique(layout_info));

    const vk::ComputePipelineCreateInfo pipeline_info = {
        .stage =
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            },
        .layout = *pipeline_layout,
    };
    pipeline = Check(device.createComputePipelineUnique({}, pipeline_info));
    SetObjectName(device, *pipeline, "Indirect Dispatch Clamp Pipeline");
    device.destroyShaderModule(module);

    const auto draw_module =
        Compile(HostShaders::DRAW_CLAMP_COMP, vk::ShaderStageFlagBits::eCompute, device);
    SetObjectName(device, draw_module, "Indirect Draw Clamp");
    const vk::PipelineLayoutCreateInfo draw_layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &(*desc_layout),
    };
    draw_pipeline_layout = Check(device.createPipelineLayoutUnique(draw_layout_info));
    const vk::ComputePipelineCreateInfo draw_pipeline_info = {
        .stage =
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = draw_module,
                .pName = "main",
            },
        .layout = *draw_pipeline_layout,
    };
    draw_pipeline = Check(device.createComputePipelineUnique({}, draw_pipeline_info));
    SetObjectName(device, *draw_pipeline, "Indirect Draw Clamp Pipeline");
    device.destroyShaderModule(draw_module);
}

DispatchGuard::~DispatchGuard() = default;

DispatchGuard::ClampedArgs DispatchGuard::Clamp(vk::CommandBuffer cmdbuf, vk::Buffer src_buffer,
                                                u64 src_offset, VAddr guest_address) {
    return Record(cmdbuf, *pipeline, *pipeline_layout, src_buffer, src_offset, 3,
                  DispatchCounterOffset, Kind::Dispatch, guest_address);
}

DispatchGuard::ClampedArgs DispatchGuard::ClampDraw(vk::CommandBuffer cmdbuf,
                                                    vk::Buffer src_buffer, u64 src_offset,
                                                    VAddr guest_address) {
    ++guarded_draws;
    return Record(cmdbuf, *draw_pipeline, *draw_pipeline_layout, src_buffer, src_offset, 4,
                  DrawCounterOffset, Kind::Draw, guest_address);
}

DispatchGuard::ClampedArgs DispatchGuard::Record(vk::CommandBuffer cmdbuf,
                                                 vk::Pipeline clamp_pipeline,
                                                 vk::PipelineLayout layout, vk::Buffer src_buffer,
                                                 u64 src_offset, u32 num_words,
                                                 u64 counter_offset, Kind kind,
                                                 VAddr guest_address) {
    const u64 slot = ring_cursor++ % NumSlots;
    const u64 slot_offset = slot * SlotSize;
    const u64 args_size = num_words * sizeof(u32);
    slot_info[slot] = {guest_address, kind};
    const std::array<u32, 4> not_executed{NotExecuted, NotExecuted, NotExecuted, NotExecuted};
    std::memcpy(args_ring.mapped_data.data() + slot_offset, not_executed.data(), SlotSize);

    const vk::DescriptorBufferInfo src_info = {
        .buffer = src_buffer,
        .offset = src_offset,
        .range = args_size,
    };
    const vk::DescriptorBufferInfo dst_info = {
        .buffer = args_ring.Handle(),
        .offset = slot_offset,
        .range = args_size,
    };
    const vk::DescriptorBufferInfo counter_info = {
        .buffer = clamp_counter.Handle(),
        .offset = counter_offset,
        .range = sizeof(u32),
    };
    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &src_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &dst_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &counter_info,
        },
    }};

    // The arguments may have been written by GPU work recorded earlier in this batch.
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
        .buffer = src_buffer,
        .offset = src_offset,
        .size = args_size,
    };
    // The dispatch sources the ring slot; earlier ring writes are ordered by the same barrier
    // on the next use of the slot's cache line, so only this slot needs guarding here.
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
        .buffer = args_ring.Handle(),
        .offset = slot_offset,
        .size = args_size,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, clamp_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, set_writes);
    if (kind == Kind::Dispatch) {
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(u32),
                             &MaxSaneGroups);
    }
    cmdbuf.dispatch(1, 1, 1);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });

    return {args_ring.Handle(), slot_offset};
}

u32 DispatchGuard::ClampCount() const {
    u32 value{};
    std::memcpy(&value, clamp_counter.mapped_data.data() + DispatchCounterOffset, sizeof(u32));
    return value;
}

u32 DispatchGuard::DrawClampCount() const {
    u32 value{};
    std::memcpy(&value, clamp_counter.mapped_data.data() + DrawCounterOffset, sizeof(u32));
    return value;
}

void DispatchGuard::LogRecent() const {
    // Commands recorded after the one that hung never ran, so the newest entries are still
    // marked. The first entry past them belongs to the last pre-pass the GPU completed: the
    // command it guards is the one that hung, or the last one before it.
    constexpr u64 MaxLogged = 12;
    const u64 recorded = std::min<u64>(ring_cursor, NumSlots);
    const auto read_slot = [this](u64 slot) {
        std::array<u32, 4> words{};
        std::memcpy(words.data(), args_ring.mapped_data.data() + slot * SlotSize, SlotSize);
        return words;
    };
    u64 pending = 0;
    while (pending < recorded &&
           read_slot((ring_cursor - 1 - pending) % NumSlots)[0] == NotExecuted) {
        ++pending;
    }
    const u64 count = std::min<u64>(recorded - pending, MaxLogged);
    LOG_CRITICAL(Render_Vulkan,
                 "Indirect argument guard: dispatches_zeroed={}, draws_zeroed={}, "
                 "guarded_draws={}; {} newest guarded commands never executed; the {} before "
                 "them, newest first:",
                 ClampCount(), DrawClampCount(), guarded_draws, pending, count);
    for (u64 i = 0; i < count; ++i) {
        const u64 slot = (ring_cursor - 1 - pending - i) % NumSlots;
        const std::array<u32, 4> words = read_slot(slot);
        const SlotInfo& info = slot_info[slot];
        if (info.kind == Kind::Draw) {
            LOG_CRITICAL(Render_Vulkan,
                         "  -{}: draw args_addr={:#x}, consumed vertices={:#x}, instances={:#x}, "
                         "first_vertex={:#x}, first_instance={:#x}",
                         i, info.guest_address, words[0], words[1], words[2], words[3]);
        } else {
            LOG_CRITICAL(Render_Vulkan,
                         "  -{}: dispatch args_addr={:#x}, consumed groups={:#x} x {:#x} x {:#x}",
                         i, info.guest_address, words[0], words[1], words[2]);
        }
    }
}

} // namespace Vulkan
