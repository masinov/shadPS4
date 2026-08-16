// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <gtest/gtest.h>
#include "video_core/page_watcher_counter.h"

TEST(PageWatcherCounter, CrossesInlineLimitWithoutWrapping) {
    VideoCore::PageWatcherCounterOverflow overflow;
    u8 inline_count{};

    for (u32 count = 1; count <= 1024; ++count) {
        EXPECT_EQ(overflow.Increment(7, inline_count), count);
        EXPECT_EQ(inline_count, std::min(count, 255u));
        EXPECT_EQ(overflow.Value(7, inline_count), count);
    }

    for (u32 count = 1024; count > 0; --count) {
        EXPECT_EQ(overflow.Decrement(7, inline_count), count - 1);
        EXPECT_EQ(inline_count, std::min(count - 1, 255u));
        EXPECT_EQ(overflow.Value(7, inline_count), count - 1);
    }
}

TEST(PageWatcherCounter, KeepsOverflowKeysIndependent) {
    VideoCore::PageWatcherCounterOverflow overflow;
    u8 first{255};
    u8 second{255};

    EXPECT_EQ(overflow.Increment(1, first), 256u);
    EXPECT_EQ(overflow.Increment(2, second), 256u);
    EXPECT_EQ(overflow.Increment(1, first), 257u);
    EXPECT_EQ(overflow.Value(1, first), 257u);
    EXPECT_EQ(overflow.Value(2, second), 256u);

    EXPECT_EQ(overflow.Decrement(2, second), 255u);
    EXPECT_EQ(overflow.Decrement(1, first), 256u);
    EXPECT_EQ(overflow.Decrement(1, first), 255u);
}
