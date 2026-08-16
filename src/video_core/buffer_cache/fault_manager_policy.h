// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace VideoCore {

// Element zero stores the GPU's atomic counter; only the remaining elements contain addresses.
[[nodiscard]] constexpr u32 RetainedFaultCount(u32 reported_count, u32 element_capacity) noexcept {
    const u32 address_capacity = element_capacity > 0 ? element_capacity - 1 : 0;
    return std::min(reported_count, address_capacity);
}

} // namespace VideoCore
