// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/device_address_tracker.h"
#include "video_core/renderer_vulkan/gpu_diagnostic_tracker.h"

TEST(DeviceAddressTracker, CorrelatesLiveAndRecentlyUnboundRanges) {
    Vulkan::DeviceAddressTracker tracker;
    tracker.Bind(0x400000000, 0x800000, false);

    auto match = tracker.Find(0x400002000);
    ASSERT_TRUE(match);
    EXPECT_EQ(match->state, Vulkan::AddressBindingState::Live);
    EXPECT_TRUE(match->contains);
    EXPECT_EQ(match->base, 0x400000000);

    tracker.Unbind(0x400000000, 0, false);
    match = tracker.Find(0x400002000);
    ASSERT_TRUE(match);
    EXPECT_EQ(match->state, Vulkan::AddressBindingState::RecentlyUnbound);
    EXPECT_TRUE(match->contains);
    EXPECT_EQ(match->size, 0x800000);
}

TEST(DeviceAddressTracker, ReportsDistancePastTheEndWithoutOverflow) {
    Vulkan::DeviceAddressTracker tracker;
    tracker.Bind(0x400000000, 0x800000, false);

    const auto match = tracker.Find(0x400802000);
    ASSERT_TRUE(match);
    EXPECT_FALSE(match->contains);
    EXPECT_EQ(match->distance, 0x2000);

    tracker.Bind(std::numeric_limits<u64>::max() - 15, 32, true);
    const auto overflow_match = tracker.Find(std::numeric_limits<u64>::max());
    ASSERT_TRUE(overflow_match);
    EXPECT_TRUE(overflow_match->contains);
    EXPECT_TRUE(overflow_match->internal);
}

TEST(DeviceAddressTracker, RetainsOverlappingBindings) {
    Vulkan::DeviceAddressTracker tracker;
    tracker.Bind(0x1000, 0x10000, false);
    tracker.Bind(0x2000, 0x100, true);

    const auto match = tracker.Find(0x9000);
    ASSERT_TRUE(match);
    EXPECT_TRUE(match->contains);
    EXPECT_EQ(match->base, 0x1000);
    EXPECT_EQ(match->size, 0x10000);
}

TEST(DeviceAddressTracker, ReportsLiveAndRetiredMatchesAfterAddressReuse) {
    Vulkan::DeviceAddressTracker tracker;
    tracker.Bind(0x4000, 0x1000, false);
    tracker.Unbind(0x4000, 0x1000, false);
    tracker.Bind(0x4000, 0x1000, false);

    const auto matches = tracker.FindAll(0x4800);
    ASSERT_GE(matches.size(), 2);
    EXPECT_EQ(matches[0].state, Vulkan::AddressBindingState::Live);
    EXPECT_EQ(matches[1].state, Vulkan::AddressBindingState::RecentlyUnbound);
    EXPECT_TRUE(matches[0].contains);
    EXPECT_TRUE(matches[1].contains);
}

TEST(GpuDiagnosticTracker, ResolvesUniqueCheckpointContext) {
    Vulkan::GpuCheckpointTracker tracker;
    Vulkan::GpuCheckpointContext first_context{
        .pipeline_hash = 0x1234,
        .shader_hashes = {0x1111, 0, 0, 0, 0x5555, 0},
        .item_count = 36,
        .instance_count = 2,
        .writable_buffer_count = 1,
        .color_attachment_count = 2,
        .has_depth_attachment = true,
        .uses_dma = true,
        .indexed = true,
    };
    first_context.writable_buffers[0] = {0x100000, 0x4000, 7};

    const void* first = tracker.Record(Vulkan::GpuCheckpoint::DirectDraw, first_context);
    const void* second = tracker.Record(Vulkan::GpuCheckpoint::DirectDraw, {});
    EXPECT_NE(first, second);

    const auto record = tracker.Resolve(first);
    ASSERT_TRUE(record);
    EXPECT_EQ(record->checkpoint, Vulkan::GpuCheckpoint::DirectDraw);
    EXPECT_EQ(record->context.pipeline_hash, 0x1234);
    EXPECT_EQ(record->context.item_count, 36);
    EXPECT_EQ(record->context.writable_buffers[0].guest_address, 0x100000);
    EXPECT_EQ(record->context.shader_hashes[0], 0x1111);
    EXPECT_EQ(record->context.shader_hashes[4], 0x5555);
    EXPECT_EQ(record->context.color_attachment_count, 2);
    EXPECT_TRUE(record->context.has_depth_attachment);
    EXPECT_TRUE(record->context.uses_dma);
}

TEST(GpuDiagnosticTracker, RetainsBufferGenerationAcrossAddressReuse) {
    Vulkan::BufferAddressTracker tracker;
    const u64 old_generation = tracker.Register(0x800000, 0x10000, 0x1000000, 0x10000, 0);
    tracker.Retire(old_generation, 41, 48, 43, 2);
    const u64 live_generation = tracker.Register(0x800000, 0x10000, 0x2000000, 0x10000, 0);

    const auto matches = tracker.FindAll(0x808000);
    ASSERT_GE(matches.size(), 2);
    EXPECT_EQ(matches[0].generation, live_generation);
    EXPECT_EQ(matches[0].state, Vulkan::BufferAddressState::Live);
    EXPECT_EQ(matches[0].guest_address, 0x2000000);
    EXPECT_EQ(matches[1].generation, old_generation);
    EXPECT_EQ(matches[1].state, Vulkan::BufferAddressState::Retired);
    EXPECT_EQ(matches[1].guest_address, 0x1000000);
    EXPECT_EQ(matches[1].last_use_tick, 41);
    EXPECT_EQ(matches[1].retirement_cpu_tick, 48);
    EXPECT_EQ(matches[1].retirement_scheduled_tick, 43);
    EXPECT_EQ(matches[1].retirement_reason, 2);
}

TEST(GpuDiagnosticTracker, LiveBufferHistoryIsNotOverwrittenByRegistrationChurn) {
    Vulkan::BufferAddressTracker tracker;
    const u64 first_generation = tracker.Register(0x100000, 0x1000, 0x200000, 0x1000, 0);
    for (size_t i = 0; i < Vulkan::BufferAddressTracker::MaxRecentRetiredRecords; ++i) {
        const u64 offset = static_cast<u64>(i + 1) * 0x2000;
        (void)tracker.Register(0x1000000 + offset, 0x1000, 0x2000000 + offset, 0x1000, 0);
    }

    const auto matches = tracker.FindAll(0x100800);
    ASSERT_FALSE(matches.empty());
    EXPECT_EQ(matches[0].generation, first_generation);
    EXPECT_EQ(matches[0].state, Vulkan::BufferAddressState::Live);
    EXPECT_TRUE(matches[0].contains);
    EXPECT_EQ(tracker.LiveCount(), Vulkan::BufferAddressTracker::MaxRecentRetiredRecords + 1);
    EXPECT_EQ(tracker.MissingRetirementCount(), 0u);
}
