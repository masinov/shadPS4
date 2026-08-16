// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <latch>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/pending_operation_queue.h"

TEST(PendingOperationQueue, ExecutesReadyCallbacksInFifoOrder) {
    Vulkan::PendingOperationQueue queue;
    std::vector<int> order;
    queue.Push([&] { order.push_back(1); }, 3);
    queue.Push([&] { order.push_back(2); }, 5);
    queue.Push([&] { order.push_back(3); }, 7);

    EXPECT_EQ(queue.Drain([](u64 tick) { return tick <= 5; }), 2u);
    EXPECT_EQ(order, (std::vector{1, 2}));
    EXPECT_EQ(queue.Drain([](u64) { return true; }), 1u);
    EXPECT_EQ(order, (std::vector{1, 2, 3}));
}

TEST(PendingOperationQueue, CallbackCanEnqueueAndRecursivelyDrainWithoutInterleaving) {
    Vulkan::PendingOperationQueue queue;
    std::vector<int> order;
    u32 nested_count = 1;
    queue.Push(
        [&] {
            order.push_back(1);
            queue.Push([&] { order.push_back(3); }, 0);
            nested_count = queue.Drain([](u64) { return true; });
            order.push_back(2);
        },
        0);

    EXPECT_EQ(queue.Drain([](u64) { return true; }), 2u);
    EXPECT_EQ(nested_count, 0u);
    EXPECT_EQ(order, (std::vector{1, 2, 3}));
}

TEST(PendingOperationQueue, ConcurrentDrainDoesNotConsumeBehindActiveCallback) {
    Vulkan::PendingOperationQueue queue;
    std::latch callback_started{1};
    std::latch release_callback{1};
    std::atomic<u32> first_count{};
    queue.Push(
        [&] {
            callback_started.count_down();
            release_callback.wait();
        },
        0);
    queue.Push([] {}, 0);

    std::thread first{[&] {
        first_count.store(queue.Drain([](u64) { return true; }), std::memory_order_relaxed);
    }};
    callback_started.wait();
    EXPECT_EQ(queue.Drain([](u64) { return true; }), 0u);
    release_callback.count_down();
    first.join();
    EXPECT_EQ(first_count.load(std::memory_order_relaxed), 2u);
}

TEST(PendingOperationQueue, ExceptionReleasesDrainOwnership) {
    Vulkan::PendingOperationQueue queue;
    queue.Push([] { throw std::runtime_error{"expected"}; }, 0);

    EXPECT_THROW(queue.Drain([](u64) { return true; }), std::runtime_error);
    queue.Push([] {}, 0);
    EXPECT_EQ(queue.Drain([](u64) { return true; }), 1u);
}
