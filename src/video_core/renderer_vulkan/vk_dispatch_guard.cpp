// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/host_shaders/dispatch_clamp_comp.h"
#include "video_core/renderer_vulkan/vk_dispatch_guard.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
// 12 bytes of arguments per slot, aligned to 16 for storage-descriptor offsets. 4096 slots wrap
// only after 4096 indirect dispatches, far beyond any realistic in-flight window.
constexpr u64 SlotSize = 16;
constexpr u64 NumSlots = 4096;
// PS4 dispatch dimension limit; also within every Vulkan implementation's minimum guarantee.
constexpr u32 MaxSaneGroups = 65535;
} // namespace

DispatchGuard::DispatchGuard(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      args_ring{instance_,
                scheduler_,
                VideoCore::MemoryUsage::DeviceLocal,
                0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                SlotSize * NumSlots},
      clamp_counter{instance_, scheduler_, VideoCore::MemoryUsage::Download, 0,
                    vk::BufferUsageFlagBits::eStorageBuffer, sizeof(u32)} {
    const auto device = instance.GetDevice();
    SetObjectName(device, args_ring.Handle(), "Indirect Dispatch Args Ring");
    SetObjectName(device, clamp_counter.Handle(), "Indirect Dispatch Clamp Counter");
    std::memset(clamp_counter.mapped_data.data(), 0, sizeof(u32));

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
}

DispatchGuard::~DispatchGuard() = default;

DispatchGuard::ClampedArgs DispatchGuard::Clamp(vk::CommandBuffer cmdbuf, vk::Buffer src_buffer,
                                                u64 src_offset) {
    const u64 slot_offset = (ring_cursor++ % NumSlots) * SlotSize;

    const vk::DescriptorBufferInfo src_info = {
        .buffer = src_buffer,
        .offset = src_offset,
        .range = 3 * sizeof(u32),
    };
    const vk::DescriptorBufferInfo dst_info = {
        .buffer = args_ring.Handle(),
        .offset = slot_offset,
        .range = 3 * sizeof(u32),
    };
    const vk::DescriptorBufferInfo counter_info = {
        .buffer = clamp_counter.Handle(),
        .offset = 0,
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
        .size = 3 * sizeof(u32),
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
        .size = 3 * sizeof(u32),
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, set_writes);
    cmdbuf.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(u32),
                         &MaxSaneGroups);
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
    std::memcpy(&value, clamp_counter.mapped_data.data(), sizeof(u32));
    return value;
}

} // namespace Vulkan
