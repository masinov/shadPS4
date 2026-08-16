// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#ifdef _WIN64
#include <windows.h>
#include "core/veh_stack.h"

namespace {

thread_local uintptr_t handler_stack_address{};

long CaptureHandlerStack(_EXCEPTION_POINTERS*) noexcept {
    const uintptr_t stack_marker = reinterpret_cast<uintptr_t>(&stack_marker);
    handler_stack_address = stack_marker;
    return EXCEPTION_CONTINUE_EXECUTION;
}

TEST(VehStack, RunsHandlerOnDedicatedStack) {
    const uintptr_t caller_stack_address = reinterpret_cast<uintptr_t>(&caller_stack_address);

    ASSERT_TRUE(Core::InitializeVehStackForCurrentThread());
    EXPECT_EQ(Core::RunOnVehStack(CaptureHandlerStack, nullptr), EXCEPTION_CONTINUE_EXECUTION);
    EXPECT_NE(handler_stack_address, 0u);

    // The handler fiber reserves its own stack, so the two local variables must not share a page.
    EXPECT_NE(handler_stack_address >> 12, caller_stack_address >> 12);

    Core::CleanupVehStackForCurrentThread();
}

} // namespace
#endif
