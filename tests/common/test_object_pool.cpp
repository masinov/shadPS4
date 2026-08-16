// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "common/object_pool.h"

namespace {

struct TrackedObject {
    explicit TrackedObject(int value_) : value{std::make_unique<int>(value_)} {
        ++constructed;
        ++alive;
    }

    ~TrackedObject() {
        ++destroyed;
        --alive;
    }

    TrackedObject(const TrackedObject&) = delete;
    TrackedObject& operator=(const TrackedObject&) = delete;

    static void ResetCounts() {
        constructed = 0;
        destroyed = 0;
        alive = 0;
    }

    static inline int constructed{};
    static inline int destroyed{};
    static inline int alive{};
    std::unique_ptr<int> value;
};

struct ThrowingObject {
    ThrowingObject() {
        if (throw_on_construction) {
            throw std::runtime_error{"construction failed"};
        }
        ++alive;
    }

    ~ThrowingObject() {
        --alive;
    }

    static inline bool throw_on_construction{};
    static inline int alive{};
};

TEST(ObjectPool, ReleaseContentsDestroysEveryConstructedObject) {
    TrackedObject::ResetCounts();
    Common::ObjectPool<TrackedObject> pool{2};

    for (int i = 0; i < 5; ++i) {
        (void)pool.Create(i);
    }
    ASSERT_EQ(TrackedObject::alive, 5);

    pool.ReleaseContents();

    EXPECT_EQ(TrackedObject::constructed, 5);
    EXPECT_EQ(TrackedObject::destroyed, 5);
    EXPECT_EQ(TrackedObject::alive, 0);
}

TEST(ObjectPool, DestructorReleasesObjectsInPartiallyUsedChunk) {
    TrackedObject::ResetCounts();
    {
        Common::ObjectPool<TrackedObject> pool{8};
        (void)pool.Create(1);
        (void)pool.Create(2);
        ASSERT_EQ(TrackedObject::alive, 2);
    }

    EXPECT_EQ(TrackedObject::constructed, 2);
    EXPECT_EQ(TrackedObject::destroyed, 2);
    EXPECT_EQ(TrackedObject::alive, 0);
}

TEST(ObjectPool, FailedConstructionDoesNotMarkStorageAsOccupied) {
    ThrowingObject::alive = 0;
    ThrowingObject::throw_on_construction = false;
    Common::ObjectPool<ThrowingObject> pool{2};
    (void)pool.Create();

    ThrowingObject::throw_on_construction = true;
    EXPECT_THROW((void)pool.Create(), std::runtime_error);

    ThrowingObject::throw_on_construction = false;
    (void)pool.Create();
    EXPECT_EQ(ThrowingObject::alive, 2);

    pool.ReleaseContents();
    EXPECT_EQ(ThrowingObject::alive, 0);
}

} // namespace
