// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

inline constexpr u32 BdaAddressSpaceBits = 40;
inline constexpr VAddr BdaAddressSpaceSize = VAddr{1} << BdaAddressSpaceBits;

[[nodiscard]] constexpr bool IsBdaAddressInRange(VAddr address) noexcept {
    return address < BdaAddressSpaceSize;
}

} // namespace VideoCore
