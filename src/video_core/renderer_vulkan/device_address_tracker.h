// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "common/types.h"

namespace Vulkan {

enum class AddressBindingState : u8 {
    Live,
    RecentlyUnbound,
};

struct AddressBindingMatch {
    AddressBindingState state{};
    u64 base{};
    u64 size{};
    u64 distance{};
    bool contains{};
    bool internal{};
    u64 sequence{};
};

class DeviceAddressTracker {
public:
    static constexpr size_t MaxLiveBindings = 65'536;
    static constexpr size_t MaxRecentUnboundBindings = 256;

    void Bind(u64 base, u64 size, bool internal) {
        std::scoped_lock lock{mutex};
        const auto key = std::pair{base, size};
        if (const auto it = live_bindings.find(key); it != live_bindings.end()) {
            it->second = LiveBinding{internal, ++sequence};
        } else if (live_bindings.size() < MaxLiveBindings) {
            live_bindings.emplace(key, LiveBinding{internal, ++sequence});
        } else {
            ++dropped_bindings;
        }
    }

    void Unbind(u64 base, u64 size, bool internal) {
        std::scoped_lock lock{mutex};
        auto it = live_bindings.find(std::pair{base, size});
        if (it == live_bindings.end()) {
            it = live_bindings.lower_bound(std::pair{base, u64{0}});
        }
        if (it != live_bindings.end() && it->first.first == base) {
            size = it->first.second;
            internal = it->second.internal;
            live_bindings.erase(it);
        }
        recent_unbound[recent_unbound_cursor] = RecentBinding{base, size, internal, ++sequence};
        recent_unbound_cursor = (recent_unbound_cursor + 1) % recent_unbound.size();
    }

    [[nodiscard]] std::optional<AddressBindingMatch> Find(u64 address) const {
        const auto matches = FindAll(address, 1);
        return matches.empty() ? std::nullopt : std::optional{matches.front()};
    }

    [[nodiscard]] std::vector<AddressBindingMatch> FindAll(u64 address,
                                                           size_t max_matches = 8) const {
        std::scoped_lock lock{mutex};
        std::vector<AddressBindingMatch> matches;
        matches.reserve(std::min(max_matches, live_bindings.size() + recent_unbound.size()));
        const auto better = [](const auto& lhs, const auto& rhs) {
            if (lhs.contains != rhs.contains) {
                return lhs.contains > rhs.contains;
            }
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }
            if (lhs.sequence != rhs.sequence) {
                return lhs.sequence > rhs.sequence;
            }
            return lhs.state == AddressBindingState::Live &&
                   rhs.state == AddressBindingState::RecentlyUnbound;
        };
        const auto consider = [&](AddressBindingState state, u64 base, const Binding& binding) {
            const bool contains = address >= base && address - base < binding.size;
            const u64 distance = contains ? 0 : DistanceToRange(address, base, binding.size);
            const AddressBindingMatch candidate{
                state, base, binding.size, distance, contains, binding.internal, binding.sequence};
            if (matches.size() < max_matches) {
                matches.push_back(candidate);
                return;
            }
            if (matches.empty()) {
                return;
            }
            size_t worst = 0;
            for (size_t i = 1; i < matches.size(); ++i) {
                if (better(matches[worst], matches[i])) {
                    worst = i;
                }
            }
            if (better(candidate, matches[worst])) {
                matches[worst] = candidate;
            }
        };

        // Binding ranges can overlap (for example, parent allocations and internal objects), so
        // inspect all retained ranges during the exceptional device-loss path rather than losing
        // a containing range to a nearest-base shortcut.
        for (const auto& [range, binding] : live_bindings) {
            consider(AddressBindingState::Live, range.first,
                     Binding{range.second, binding.internal, binding.sequence});
        }
        for (const auto& binding : recent_unbound) {
            if (binding.sequence != 0) {
                consider(AddressBindingState::RecentlyUnbound, binding.base,
                         Binding{binding.size, binding.internal, binding.sequence});
            }
        }
        std::ranges::sort(matches, better);
        return matches;
    }

    [[nodiscard]] size_t LiveCount() const {
        std::scoped_lock lock{mutex};
        return live_bindings.size();
    }

    [[nodiscard]] u64 DroppedCount() const {
        std::scoped_lock lock{mutex};
        return dropped_bindings;
    }

private:
    struct Binding {
        u64 size{};
        bool internal{};
        u64 sequence{};
    };

    struct LiveBinding {
        bool internal{};
        u64 sequence{};
    };

    struct RecentBinding : Binding {
        u64 base{};

        RecentBinding() = default;
        RecentBinding(u64 base_, u64 size_, bool internal_, u64 sequence_)
            : Binding{size_, internal_, sequence_}, base{base_} {}
    };

    [[nodiscard]] static constexpr u64 SaturatingEnd(u64 base, u64 size) noexcept {
        return size > std::numeric_limits<u64>::max() - base ? std::numeric_limits<u64>::max()
                                                             : base + size;
    }

    [[nodiscard]] static constexpr u64 DistanceToRange(u64 address, u64 base, u64 size) noexcept {
        if (address < base) {
            return base - address;
        }
        const u64 end = SaturatingEnd(base, size);
        return address >= end ? address - end : 0;
    }

    mutable std::mutex mutex;
    std::map<std::pair<u64, u64>, LiveBinding> live_bindings;
    std::array<RecentBinding, MaxRecentUnboundBindings> recent_unbound{};
    size_t recent_unbound_cursor{};
    u64 sequence{};
    u64 dropped_bindings{};
};

} // namespace Vulkan
