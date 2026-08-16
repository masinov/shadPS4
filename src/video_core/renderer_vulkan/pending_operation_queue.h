// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <queue>

#include "common/scope_exit.h"
#include "common/types.h"
#include "common/unique_function.h"

namespace Vulkan {

// Timeline-ordered callbacks which may enqueue work or recursively submit from inside a callback.
// Queue state is protected, but user code is never invoked under the queue mutex.
class PendingOperationQueue {
public:
    void Push(Common::UniqueFunction<void>&& callback, u64 gpu_tick) {
        std::scoped_lock lock{mutex};
        entries.emplace(std::move(callback), gpu_tick);
    }

    template <typename IsReady>
    u32 Drain(IsReady&& is_ready) {
        {
            std::scoped_lock lock{mutex};
            // The active drain continues after a recursive submission returns. This preserves
            // callback completion order and prevents two consumers from interleaving callbacks.
            if (draining) {
                return 0;
            }
            draining = true;
        }
        SCOPE_EXIT {
            std::scoped_lock lock{mutex};
            draining = false;
        };

        u32 executed{};
        while (true) {
            Entry entry;
            {
                std::scoped_lock lock{mutex};
                if (entries.empty() || !is_ready(entries.front().gpu_tick)) {
                    break;
                }
                entry = std::move(entries.front());
                entries.pop();
            }

            entry.callback();
            ++executed;
        }
        return executed;
    }

private:
    struct Entry {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick{};
    };

    std::mutex mutex;
    std::queue<Entry> entries;
    bool draining{};
};

} // namespace Vulkan
