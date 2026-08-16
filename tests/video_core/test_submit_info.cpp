// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/vk_scheduler.h"

TEST(SubmitInfo, WaitStageIsStoredWithItsSemaphore) {
    Vulkan::SubmitInfo info{};
    info.AddWait({}, 7, vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddWait({}, 11, vk::PipelineStageFlagBits::eFragmentShader);

    ASSERT_EQ(info.num_wait_semas, 2u);
    EXPECT_EQ(info.wait_ticks[0], 7u);
    EXPECT_EQ(info.wait_stage_masks[0], vk::PipelineStageFlagBits::eColorAttachmentOutput);
    EXPECT_EQ(info.wait_ticks[1], 11u);
    EXPECT_EQ(info.wait_stage_masks[1], vk::PipelineStageFlagBits::eFragmentShader);
}

TEST(SubmitInfo, DefaultWaitStageIsConservativelyAllCommands) {
    Vulkan::SubmitInfo info{};
    info.AddWait({}, 13);

    ASSERT_EQ(info.num_wait_semas, 1u);
    EXPECT_EQ(info.wait_ticks[0], 13u);
    EXPECT_EQ(info.wait_stage_masks[0], vk::PipelineStageFlagBits::eAllCommands);
}
