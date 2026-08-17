// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <utility>
#include <vector>
#include "common/types.h"

namespace VideoCore {

/// Sorted, coalesced set of [offset, offset + size) ranges. Sparse buffers accumulate one bound
/// block per demand binding; queries against the coalesced list stay cheap regardless of how many
/// blocks a buffer holds.
class SparseCoverage {
public:
    using Range = std::pair<u64, u64>; ///< offset, size

    void Add(u64 offset, u64 size) {
        if (size == 0) {
            return;
        }
        const u64 end = offset + size;
        // First range whose end is at or after offset: it may touch or overlap the new range.
        auto first = std::ranges::lower_bound(ranges, offset, {},
                                              [](const Range& r) { return r.first + r.second; });
        u64 merged_begin = offset;
        u64 merged_end = end;
        auto it = first;
        while (it != ranges.end() && it->first <= merged_end) {
            merged_begin = std::min(merged_begin, it->first);
            merged_end = std::max(merged_end, it->first + it->second);
            ++it;
        }
        it = ranges.erase(first, it);
        ranges.insert(it, Range{merged_begin, merged_end - merged_begin});
    }

    /// True when [offset, offset + size) lies entirely inside one range.
    [[nodiscard]] bool Contains(u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return true;
        }
        auto it =
            std::ranges::upper_bound(ranges, offset, {}, [](const Range& r) { return r.first; });
        if (it == ranges.begin()) {
            return false;
        }
        --it;
        return it->first <= offset && it->first + it->second >= offset + size;
    }

    /// Calls func(gap_begin, gap_end) for each part of [begin, end) not covered by any range.
    template <typename Func>
    void ForEachGap(u64 begin, u64 end, Func&& func) const {
        u64 cursor = begin;
        for (const auto& [range_offset, range_size] : ranges) {
            const u64 range_end = range_offset + range_size;
            if (range_end <= cursor) {
                continue;
            }
            if (range_offset >= end) {
                break;
            }
            if (range_offset > cursor) {
                func(cursor, range_offset);
            }
            cursor = std::max(cursor, range_end);
            if (cursor >= end) {
                return;
            }
        }
        if (cursor < end) {
            func(cursor, end);
        }
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (const auto& [offset, size] : ranges) {
            func(offset, size);
        }
    }

    [[nodiscard]] size_t Count() const noexcept {
        return ranges.size();
    }

    void Clear() noexcept {
        ranges.clear();
    }

private:
    std::vector<Range> ranges;
};

} // namespace VideoCore
