// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <vector>

#include "common/lru_cache.h"

namespace {

TEST(LeastRecentlyUsedCache, ForEachItemBelowExcludesCutoffEpoch) {
    Common::LeastRecentlyUsedCache<int, u64> cache;
    (void)cache.Insert(10, 0);
    (void)cache.Insert(20, 1);

    std::vector<int> visited;
    cache.ForEachItemBelow(1, [&](int value) { visited.push_back(value); });

    EXPECT_EQ(visited, std::vector<int>{10});
}

TEST(LeastRecentlyUsedCache, CurrentEpochObjectsCannotBeCollected) {
    Common::LeastRecentlyUsedCache<int, u64> cache;
    const size_t old_id = cache.Insert(10, 0);
    (void)cache.Insert(20, 1);
    cache.Touch(old_id, 2);

    std::vector<int> visited;
    cache.ForEachItemBelow(2, [&](int value) { visited.push_back(value); });

    EXPECT_EQ(visited, std::vector<int>{20});
}

TEST(LeastRecentlyUsedCache, EmptyAndZeroCutoffVisitNothing) {
    Common::LeastRecentlyUsedCache<int, u64> cache;
    std::vector<int> visited;
    cache.ForEachItemBelow(0, [&](int value) { visited.push_back(value); });
    EXPECT_TRUE(visited.empty());

    (void)cache.Insert(10, 0);
    cache.ForEachItemBelow(0, [&](int value) { visited.push_back(value); });
    EXPECT_TRUE(visited.empty());
}

} // namespace
