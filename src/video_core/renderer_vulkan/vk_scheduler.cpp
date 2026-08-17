// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <string_view>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;
std::atomic<SubmitCriticalPhase> Scheduler::submit_critical_phase{SubmitCriticalPhase::None};

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    current_cmdbuf.endRendering();
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        Flush();
    }
    master_semaphore.Wait(tick);
}

u32 Scheduler::PopPendingOperations(bool refresh_timeline) {
    if (refresh_timeline) {
        master_semaphore.Refresh();
    }
    return pending_ops.Drain([this](u64 gpu_tick) { return master_semaphore.IsFree(gpu_tick); });
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));
    instance.InsertCheckpoint(current_cmdbuf, GpuCheckpoint::CommandBufferBegin);

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    const auto submit_start = std::chrono::steady_clock::now();
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    instance.InsertCheckpoint(current_cmdbuf, GpuCheckpoint::CommandBufferEnd);
    Check(current_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);
    const bool has_sparse_binds = !pending_sparse_binds.empty();
    if (has_sparse_binds) {
        if (!sparse_bind_semaphore) {
            const vk::StructureChain semaphore_chain = {
                vk::SemaphoreCreateInfo{},
                vk::SemaphoreTypeCreateInfo{
                    .semaphoreType = vk::SemaphoreType::eTimeline,
                    .initialValue = 0,
                },
            };
            auto [sem_result, semaphore] =
                instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
            ASSERT_MSG(sem_result == vk::Result::eSuccess,
                       "Failed to create sparse bind semaphore: {}", vk::to_string(sem_result));
            sparse_bind_semaphore = std::move(semaphore);
        }
        // Sparse binding operations complete asynchronously with respect to command execution;
        // this submission must not start before the binds it depends on finish.
        info.AddWait(*sparse_bind_semaphore, sparse_bind_value + 1,
                     vk::PipelineStageFlagBits::eAllCommands);
    }

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = info.wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    const auto prepare_end = std::chrono::steady_clock::now();
    const auto lock_start = std::chrono::steady_clock::now();
    vk::Result submit_result;
    std::chrono::steady_clock::time_point lock_end;
    std::chrono::steady_clock::time_point imgui_end;
    std::chrono::steady_clock::time_point queue_end;
    SubmitCriticalPhase contended_phase = SubmitCriticalPhase::None;
    {
        // Vulkan queues require external synchronization. ImGui texture upload also submits to and
        // waits on this queue, so it belongs in the same critical section. Command-buffer
        // preparation, timeline maintenance, and arbitrary deferred callbacks are scheduler-local
        // and must not hold this global lock: callbacks are allowed to submit recursively.
        SubmitLock lk{SubmitCriticalPhase::ImGui};
        contended_phase = lk.ContendedPhase();
        lock_end = std::chrono::steady_clock::now();
        ImGui::Core::TextureManager::Submit();
        imgui_end = std::chrono::steady_clock::now();
        if (has_sparse_binds) {
            SetSubmitCriticalPhase(SubmitCriticalPhase::SparseBind);
            const u64 bind_signal_value = ++sparse_bind_value;
            boost::container::small_vector<vk::SparseBufferMemoryBindInfo, 8> buffer_binds;
            size_t total_ranges = 0;
            for (const PendingSparseBind& pending : pending_sparse_binds) {
                buffer_binds.push_back(vk::SparseBufferMemoryBindInfo{
                    .buffer = pending.buffer,
                    .bindCount = static_cast<u32>(pending.binds.size()),
                    .pBinds = pending.binds.data(),
                });
                total_ranges += pending.binds.size();
            }
            const vk::Semaphore bind_semaphore = *sparse_bind_semaphore;
            const vk::TimelineSemaphoreSubmitInfo bind_timeline_info = {
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &bind_signal_value,
            };
            const vk::BindSparseInfo bind_info = {
                .pNext = &bind_timeline_info,
                .bufferBindCount = static_cast<u32>(buffer_binds.size()),
                .pBufferBinds = buffer_binds.data(),
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &bind_semaphore,
            };
            const auto bind_start = std::chrono::steady_clock::now();
            const vk::Result bind_result =
                instance.GetGraphicsQueue().bindSparse(bind_info, VK_NULL_HANDLE);
            const auto bind_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - bind_start)
                                     .count();
            ++sparse_bind_stats.count;
            sparse_bind_stats.total_ms += static_cast<u64>(bind_ms);
            sparse_bind_stats.max_ms = std::max<u64>(sparse_bind_stats.max_ms, bind_ms);
            if (bind_ms >= 20) {
                LOG_WARNING(Render_Vulkan,
                            "Slow sparse bind: {} ms for {} ranges in {} buffers, value={}",
                            bind_ms, total_ranges, buffer_binds.size(), bind_signal_value);
            }
            if (bind_result == vk::Result::eErrorDeviceLost) {
                instance.ReportDeviceLoss("sparse memory binding");
            }
            ASSERT_MSG(bind_result == vk::Result::eSuccess,
                       "Failed to bind sparse buffer memory: {}", vk::to_string(bind_result));
            pending_sparse_binds.clear();
        }
        SetSubmitCriticalPhase(SubmitCriticalPhase::QueueSubmit);
        submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
        queue_end = std::chrono::steady_clock::now();
    }
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.ReportDeviceLoss("graphics queue submission");
    }
    ASSERT_MSG(submit_result == vk::Result::eSuccess, "Failed to submit command buffer: {}",
               vk::to_string(submit_result));

    master_semaphore.Refresh();
    const auto refresh_end = std::chrono::steady_clock::now();
    AllocateWorkerCommandBuffers();
    const auto allocate_end = std::chrono::steady_clock::now();

    // Apply pending operations using the timeline value sampled immediately above. Refreshing it
    // again here would issue a duplicate driver query on every submission.
    const u32 executed_pending_operations = PopPendingOperations(false);
    const auto pending_end = std::chrono::steady_clock::now();

    const auto elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    };
    const auto total_ms = elapsed_ms(submit_start, pending_end);
    if (total_ms >= 100) {
        LOG_WARNING(Render_Vulkan,
                    "Slow scheduler submission: total={} ms, prepare={} ms, submit_lock={} ms, "
                    "contended_phase={}, imgui_submit={} ms, queue_submit={} ms, "
                    "timeline_refresh={} ms, command_buffer={} ms, pending_ops={} ms "
                    "({} callbacks)",
                    total_ms, elapsed_ms(submit_start, prepare_end),
                    elapsed_ms(lock_start, lock_end), SubmitCriticalPhaseName(contended_phase),
                    elapsed_ms(lock_end, imgui_end), elapsed_ms(imgui_end, queue_end),
                    elapsed_ms(queue_end, refresh_end), elapsed_ms(refresh_end, allocate_end),
                    elapsed_ms(allocate_end, pending_end), executed_pending_operations);
    }
}

void Scheduler::BindSparse(vk::Buffer buffer, std::span<const vk::SparseMemoryBind> binds) {
    if (binds.empty()) {
        return;
    }
    pending_sparse_binds.push_back(
        PendingSparseBind{buffer, std::vector<vk::SparseMemoryBind>{binds.begin(), binds.end()}});
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}

} // namespace Vulkan
