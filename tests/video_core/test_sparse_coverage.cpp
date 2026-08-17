// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "video_core/buffer_cache/sparse_coverage.h"

using VideoCore::SparseCoverage;

TEST(SparseCoverage, CoalescesTouchingAndOverlappingRanges) {
    SparseCoverage coverage;
    coverage.Add(0x20000, 0x10000);
    coverage.Add(0x00000, 0x10000);
    EXPECT_EQ(coverage.Count(), 2u);
    coverage.Add(0x10000, 0x10000); // bridges the two
    EXPECT_EQ(coverage.Count(), 1u);
    EXPECT_TRUE(coverage.Contains(0x0, 0x30000));
    coverage.Add(0x28000, 0x20000); // overlaps the tail
    EXPECT_EQ(coverage.Count(), 1u);
    EXPECT_TRUE(coverage.Contains(0x0, 0x48000));
    EXPECT_FALSE(coverage.Contains(0x0, 0x48001));
}

TEST(SparseCoverage, ContainsIsExactAtBoundaries) {
    SparseCoverage coverage;
    coverage.Add(0x40000, 0x40000);
    EXPECT_TRUE(coverage.Contains(0x40000, 0x40000));
    EXPECT_TRUE(coverage.Contains(0x50000, 0x1000));
    EXPECT_FALSE(coverage.Contains(0x3ffff, 2));
    EXPECT_FALSE(coverage.Contains(0x7ffff, 2));
    EXPECT_TRUE(coverage.Contains(0x123, 0));
}

TEST(SparseCoverage, EnumeratesGapsOfARequest) {
    SparseCoverage coverage;
    coverage.Add(0x10000, 0x10000);
    coverage.Add(0x40000, 0x10000);
    std::vector<std::pair<u64, u64>> gaps;
    coverage.ForEachGap(0x0, 0x60000, [&](u64 begin, u64 end) { gaps.emplace_back(begin, end); });
    ASSERT_EQ(gaps.size(), 3u);
    EXPECT_EQ(gaps[0], (std::pair<u64, u64>{0x0, 0x10000}));
    EXPECT_EQ(gaps[1], (std::pair<u64, u64>{0x20000, 0x40000}));
    EXPECT_EQ(gaps[2], (std::pair<u64, u64>{0x50000, 0x60000}));

    gaps.clear();
    coverage.ForEachGap(0x12000, 0x18000,
                        [&](u64 begin, u64 end) { gaps.emplace_back(begin, end); });
    EXPECT_TRUE(gaps.empty());
}
