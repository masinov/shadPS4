// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>

#include <gtest/gtest.h>
#include "video_core/buffer_cache/fault_manager_policy.h"

TEST(FaultManagerPolicy, ReservesCounterSlotAndClampsUntrustedGpuCount) {
    EXPECT_EQ(VideoCore::RetainedFaultCount(0, 1024), 0u);
    EXPECT_EQ(VideoCore::RetainedFaultCount(100, 1024), 100u);
    EXPECT_EQ(VideoCore::RetainedFaultCount(1023, 1024), 1023u);
    EXPECT_EQ(VideoCore::RetainedFaultCount(1024, 1024), 1023u);
    EXPECT_EQ(VideoCore::RetainedFaultCount(std::numeric_limits<u32>::max(), 1024), 1023u);
    EXPECT_EQ(VideoCore::RetainedFaultCount(1, 0), 0u);
}
