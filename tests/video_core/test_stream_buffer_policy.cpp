// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>
#include "video_core/buffer_cache/stream_buffer_policy.h"

TEST(StreamBufferPolicy, PadsTheSideExtendedByTheRequest) {
    // Run 12 crossed the stream threshold while visiting a small overlap on the left, even though
    // the request was extending the 436 MiB overlap on the right.
    constexpr VAddr requested_begin = 0x29d390000;
    constexpr VAddr requested_end = 0x29d804000;
    constexpr VAddr overlap_begin = 0x282180000;
    constexpr VAddr overlap_end = 0x29d7e8000;

    constexpr auto growth = VideoCore::DetermineStreamGrowth(requested_begin, requested_end,
                                                             overlap_begin, overlap_end);
    EXPECT_FALSE(growth.left);
    EXPECT_TRUE(growth.right);
}

TEST(StreamBufferPolicy, DetectsBidirectionalGrowth) {
    constexpr auto growth = VideoCore::DetermineStreamGrowth(90, 210, 100, 200);
    EXPECT_TRUE(growth.left);
    EXPECT_TRUE(growth.right);
}

TEST(StreamBufferPolicy, ReservesFourObservedStreamingStepsWithinBounds) {
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB), 16_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(64_KB, 64_KB), 2_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(8_MB, 1_MB), 8_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(128_MB, 128_MB), 16_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(std::numeric_limits<VAddr>::max(),
                                                       std::numeric_limits<VAddr>::max()),
              16_MB);
}

TEST(StreamBufferPolicy, ReservesProportionallyForLargeStreamingRanges) {
    // Small ranges keep the stride-based reserve; large ranges reserve one eighth of their natural
    // size so replacements (allocation + full copy) become rarer as the buffer grows.
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, 8_MB), 16_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, 128_MB), 16_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, 256_MB), 32_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, 640_MB), 64_MB);
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(64_KB, 64_KB, 24_MB), 3_MB);
    // The proportional part is capped independently of the stride part.
    EXPECT_EQ(VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, 4_GB), 64_MB);
    EXPECT_EQ(
        VideoCore::DetermineStreamGrowthDistance(4_MB, 4_MB, std::numeric_limits<VAddr>::max()),
        64_MB);
}

TEST(StreamBufferPolicy, AdvancesReplacementChainRegardlessOfPressure) {
    constexpr u64 budget = 4790_MB;
    const u64 headroom = VideoCore::AllocationHeadroom(budget);
    EXPECT_FALSE(VideoCore::ShouldAdvanceReplacementChain(0, 600_MB, budget));
    EXPECT_FALSE(VideoCore::ShouldAdvanceReplacementChain(headroom / 4, headroom / 4, budget));
    EXPECT_TRUE(VideoCore::ShouldAdvanceReplacementChain(headroom / 2, headroom, budget));
    EXPECT_TRUE(VideoCore::ShouldAdvanceReplacementChain(1177_MB, 4_MB, budget));
}

TEST(StreamBufferPolicy, PreservesAmortizedGrowthForLargeReplacementsUnderPressure) {
    constexpr u64 budget = 4790_MB;

    EXPECT_TRUE(VideoCore::ShouldReserveStreamGrowth(4200_MB, budget, 64_MB, 16_MB));
    EXPECT_TRUE(VideoCore::ShouldReserveStreamGrowth(4484_MB, budget, 568_MB, 16_MB));
    EXPECT_TRUE(VideoCore::ShouldReserveStreamGrowth(4689_MB, budget, 64_MB, 16_MB));
    EXPECT_FALSE(VideoCore::ShouldReserveStreamGrowth(4689_MB, budget, 4_MB, 16_MB));
    EXPECT_FALSE(VideoCore::ShouldReserveStreamGrowth(4200_MB, budget, 64_MB, 0));
}

TEST(StreamBufferPolicy, AdvancesTimelineBeforeDeferredReplacementsConsumeHeadroom) {
    constexpr u64 budget = 4790_MB;

    EXPECT_FALSE(VideoCore::ShouldAdvanceReplacementChain(0, 150_MB, budget));
    EXPECT_FALSE(VideoCore::ShouldAdvanceReplacementChain(100_MB, 100_MB, budget));
    EXPECT_TRUE(VideoCore::ShouldAdvanceReplacementChain(150_MB, 100_MB, budget));
    EXPECT_TRUE(VideoCore::ShouldAdvanceReplacementChain(200_MB, 150_MB, budget));
    EXPECT_FALSE(VideoCore::ShouldAdvanceReplacementChain(200_MB, 0, budget));
}

TEST(StreamBufferPolicy, AddressSpaceBoundsDoNotWrap) {
    EXPECT_EQ(VideoCore::GrowStreamBegin(16_KB, 16_MB, 32_KB), 16_KB);
    EXPECT_EQ(VideoCore::GrowStreamBegin(48_KB, 16_MB, 32_KB), 32_KB);
    EXPECT_EQ(VideoCore::GrowStreamEnd(std::numeric_limits<VAddr>::max() - 1_MB, 16_MB,
                                       std::numeric_limits<VAddr>::max()),
              std::numeric_limits<VAddr>::max());
    EXPECT_EQ(VideoCore::GrowStreamEnd(std::numeric_limits<VAddr>::max(), 16_MB,
                                       std::numeric_limits<VAddr>::max()),
              std::numeric_limits<VAddr>::max());
}

TEST(StreamBufferPolicy, RetainsBoundedStreamingConfidenceAfterLeap) {
    EXPECT_EQ(VideoCore::ReplacementStreamScore(16, 1, false), VideoCore::StreamingScore);
    EXPECT_EQ(VideoCore::ReplacementStreamScore(VideoCore::StreamingScore, 1, true),
              VideoCore::StreamingScore);
    EXPECT_EQ(VideoCore::ReplacementStreamScore(0, 1000, false), VideoCore::StreamingScore);
    EXPECT_EQ(VideoCore::AccumulateStreamScore(-1, 1), 1);
    EXPECT_EQ(VideoCore::AccumulateStreamScore(std::numeric_limits<int>::max(),
                                               std::numeric_limits<int>::max()),
              VideoCore::StreamingScore);
}
