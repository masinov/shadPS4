// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <mutex>
#include <unordered_map>
#include "common/assert.h"
#include "common/types.h"

namespace VideoCore {

// Extends compact per-page u8 counters without increasing every PageState in the 40-bit guest
// address space. Counts above 255 are rare and spill into this shared sparse table.
class PageWatcherCounterOverflow {
public:
    u32 Increment(u32 key, u8& inline_count) {
        if (inline_count != std::numeric_limits<u8>::max()) {
            return ++inline_count;
        }

        std::scoped_lock lock{mutex};
        auto [it, inserted] = overflow_counts.try_emplace(key, 256);
        if (!inserted) {
            ASSERT_MSG(it->second != std::numeric_limits<u32>::max(),
                       "Page watcher overflow count exhausted");
            ++it->second;
        }
        return it->second;
    }

    u32 Decrement(u32 key, u8& inline_count) {
        ASSERT_MSG(inline_count > 0, "Not enough page watchers");
        if (inline_count != std::numeric_limits<u8>::max()) {
            return --inline_count;
        }

        std::scoped_lock lock{mutex};
        const auto it = overflow_counts.find(key);
        if (it == overflow_counts.end()) {
            return --inline_count;
        }
        if (it->second == 256) {
            overflow_counts.erase(it);
            return inline_count;
        }
        ASSERT_MSG(it->second > 256, "Invalid page watcher overflow count");
        return --it->second;
    }

    [[nodiscard]] u32 Value(u32 key, u8 inline_count) const {
        if (inline_count != std::numeric_limits<u8>::max()) {
            return inline_count;
        }

        std::scoped_lock lock{mutex};
        const auto it = overflow_counts.find(key);
        return it == overflow_counts.end() ? inline_count : it->second;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<u32, u32> overflow_counts;
};

} // namespace VideoCore
