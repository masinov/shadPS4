// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>
#include "video_core/memory_gc.h"

TEST(MemoryGc, AllocationHeadroomIsBounded) {
    EXPECT_EQ(VideoCore::AllocationHeadroom(1_GB), 128_MB);
    EXPECT_EQ(VideoCore::AllocationHeadroom(4_GB), 4_GB / 20);
    EXPECT_EQ(VideoCore::AllocationHeadroom(8_GB), 256_MB);
}

TEST(MemoryGc, AllocationWithinSafeLimitNeedsNoCollection) {
    EXPECT_EQ(VideoCore::AllocationReclaimTarget(3_GB, 4_GB, 64_MB, false), 0u);
}

TEST(MemoryGc, AllocationReclaimsProjectedBudgetExcess) {
    constexpr u64 used = 3_GB + 768_MB;
    constexpr u64 allocation = 512_MB;
    constexpr u64 safe_limit = 4_GB - (4_GB / 20);
    EXPECT_EQ(VideoCore::AllocationReclaimTarget(used, 4_GB, allocation, false),
              used + allocation - safe_limit);
}

TEST(MemoryGc, FailedAllocationForcesAtLeastItsOwnSize) {
    EXPECT_EQ(VideoCore::AllocationReclaimTarget(3_GB, 4_GB, 64_MB, true), 64_MB);
}

TEST(MemoryGc, AllocationProjectionSaturatesWithoutWrapping) {
    constexpr u64 used = std::numeric_limits<u64>::max() - 16;
    constexpr u64 safe_limit = 4_GB - (4_GB / 20);
    EXPECT_EQ(VideoCore::AllocationReclaimTarget(used, 4_GB, 64_MB, false),
              std::numeric_limits<u64>::max() - safe_limit);
}

TEST(MemoryGc, DiagnosticSamplingKeepsInitialContextThenUsesPowersOfTwo) {
    EXPECT_FALSE(VideoCore::ShouldLogDiagnosticSample(0));
    for (u64 event_count = 1; event_count <= 8; ++event_count) {
        EXPECT_TRUE(VideoCore::ShouldLogDiagnosticSample(event_count));
    }
    EXPECT_FALSE(VideoCore::ShouldLogDiagnosticSample(9));
    EXPECT_TRUE(VideoCore::ShouldLogDiagnosticSample(16));
    EXPECT_FALSE(VideoCore::ShouldLogDiagnosticSample(17));
    EXPECT_TRUE(VideoCore::ShouldLogDiagnosticSample(1024));
}

TEST(MemoryGc, CurrentCommandBufferResourcesAreInFlight) {
    EXPECT_FALSE(VideoCore::IsResourceInFlight(41, 42));
    EXPECT_TRUE(VideoCore::IsResourceInFlight(42, 42));
    EXPECT_TRUE(VideoCore::IsResourceInFlight(43, 42));
}

TEST(MemoryGc, PreflightOnlyReclaimsCompletedLastUseTicks) {
    EXPECT_TRUE(VideoCore::CanReclaimWithoutWait(41, 41));
    EXPECT_TRUE(VideoCore::CanReclaimWithoutWait(40, 41));
    EXPECT_FALSE(VideoCore::CanReclaimWithoutWait(42, 41));
}

TEST(MemoryGc, ResourceUseTicksNeverMoveBackwards) {
    EXPECT_EQ(VideoCore::RecordResourceUse(41, 47), 47u);
    EXPECT_EQ(VideoCore::RecordResourceUse(47, 41), 47u);
    EXPECT_EQ(VideoCore::RecordResourceUse(47, 47), 47u);
}

TEST(MemoryGc, RetirementsTrackLatestUseAndExecuteOnce) {
    VideoCore::GcResult result;
    u32 executions{};
    result.QueueRetirement(41, [&executions] { ++executions; });
    result.QueueRetirement(47, [&executions] { ++executions; });

    EXPECT_EQ(result.latest_use_tick, 47u);
    EXPECT_EQ(result.retirements.size(), 2u);
    result.ExecuteRetirements();
    EXPECT_EQ(executions, 2u);
    EXPECT_TRUE(result.retirements.empty());

    result.ExecuteRetirements();
    EXPECT_EQ(executions, 2u);
}

TEST(MemoryGc, RetirementsReportExecutedCount) {
    VideoCore::GcResult result;
    result.QueueRetirement(1, [] {});
    result.QueueRetirement(2, [] {});
    EXPECT_EQ(result.ExecuteRetirements(), 2u);
    EXPECT_EQ(result.ExecuteRetirements(), 0u);
}

TEST(MemoryGc, EvictionAgeIsMeasuredInSubmissions) {
    // Run 21: hot ~150 MiB streaming buffers were evicted and rebuilt through ~8 full-copy
    // replacements because the epoch advanced per collector call (once per allocation under
    // pressure). Ages must be several guest submissions even under critical pressure.
    EXPECT_GE(VideoCore::GcMinimumAge(VideoCore::GcPressure::Critical), 8u);
    EXPECT_GT(VideoCore::GcMinimumAge(VideoCore::GcPressure::High),
              VideoCore::GcMinimumAge(VideoCore::GcPressure::Critical));
}
