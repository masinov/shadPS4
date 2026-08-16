// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>
#include "video_core/buffer_cache/bda_address_policy.h"

TEST(BdaAddressPolicy, AcceptsOnlyTheModeledFortyBitGuestAddressSpace) {
    EXPECT_TRUE(VideoCore::IsBdaAddressInRange(0));
    EXPECT_TRUE(VideoCore::IsBdaAddressInRange(VideoCore::BdaAddressSpaceSize - 1));
    EXPECT_FALSE(VideoCore::IsBdaAddressInRange(VideoCore::BdaAddressSpaceSize));
    EXPECT_FALSE(VideoCore::IsBdaAddressInRange(VideoCore::BdaAddressSpaceSize + 1_GB));
    EXPECT_FALSE(VideoCore::IsBdaAddressInRange(std::numeric_limits<VAddr>::max()));
}

TEST(BdaAddressPolicy, PageIndicesCannotTruncateWhenAddressIsAccepted) {
    constexpr u32 page_bits = 14;
    constexpr u64 page_count = VideoCore::BdaAddressSpaceSize >> page_bits;
    constexpr VAddr last_address = VideoCore::BdaAddressSpaceSize - 1;

    static_assert(page_count <= u64{std::numeric_limits<u32>::max()} + 1);
    EXPECT_EQ(last_address >> page_bits, page_count - 1);
    EXPECT_LE(last_address >> page_bits, std::numeric_limits<u32>::max());
}
