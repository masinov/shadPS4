// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/config.h"
#include "common/div_ceil.h"

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    u16& NumFlushes(u32 page) {
        return flushes[page];
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else if constexpr (type == Type::CPU) {
            // Same protect-before-clear ordering as ForEachModifiedRange: closing the window in
            // which a concurrent game write is neither caught by a fault nor left marked dirty.
            RegionBits cleaned(bits, start_page, end_page);
            for (const auto& [arm_start, arm_end] : cleaned) {
                writeable.UnsetRange(arm_start, arm_end);
            }
            tracker->UpdatePageWatchersForRegion<true, false>(cpu_addr, cleaned);
            bits.UnsetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Fast) {
            UpdateProtection<enable, true>();
        } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Disable) {
            UpdateProtection<!enable, false>();
        } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Unsafe) {
            UpdateProtection<!enable, false>();
        } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Low) {
            UpdateProtection<enable, true>();
        } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Default) {
            UpdateProtection<enable, true>();
        }
        if (Config::readbackSpeed() != Config::ReadbackSpeed::Low) {
            for (size_t page = start_page; page != end_page && !enable; ++page) {
                ++flushes[page];
            }
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            if constexpr (type == Type::CPU) {
                // Arm the write watchers BEFORE clearing the dirty bits. With the old order, a
                // game write landing between the clear and the mprotect neither faulted (page
                // still writable) nor stayed marked (bits already cleared): the freshly streamed
                // asset's bytes silently never uploaded again, and the draw rendered the previous
                // occupant's stale data - the load-driven explosions of the whole series. The
                // mprotect is the linearization point: writes before it are captured by the
                // upload copy that follows this enumeration; writes after it fault, wait on the
                // region lock, and re-mark the pages dirty for the next synchronization.
                for (const auto& [arm_start, arm_end] : mask) {
                    writeable.UnsetRange(arm_start, arm_end);
                }
                tracker->UpdatePageWatchersForRegion<true, false>(cpu_addr, mask);
                bits.UnsetRange(start_page, end_page);
            } else {
                bits.UnsetRange(start_page, end_page);
            }
            if constexpr (type == Type::CPU) {
                UpdateProtection<true, false>(); // reconciliation; no-op after the arming above
            } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Disable) {
                UpdateProtection<true, false>();
            } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Unsafe) {
                UpdateProtection<false, false>();
            } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Fast) {
                UpdateProtection<false, true>();
            } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Low) {
                UpdateProtection<false, true>();
            } else if (Config::readbackSpeed() == Config::ReadbackSpeed::Default) {
                UpdateProtection<false, true>();
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        const RegionBits& bits = GetRegionBits<type>();
        RegionBits test(bits, start_page, end_page);
        return test.Any();
    }

    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
    RegionWords flushes{};
};

} // namespace VideoCore