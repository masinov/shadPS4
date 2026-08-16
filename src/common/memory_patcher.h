// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

namespace MemoryPatcher {

extern EXPORT uintptr_t g_eboot_address;
extern uint64_t g_eboot_image_size;
extern std::string g_game_serial;
extern std::string patch_file;

enum PatchMask : uint8_t {
    None,
    Mask,
    Mask_Jump32,
};

inline bool IsSpecialCusa() {
    const std::string_view serial{MemoryPatcher::g_game_serial};

    return serial == "CUSA00035" || serial == "CUSA00785" || serial == "CUSA00076" ||
           serial == "CUSA00552" || serial == "CUSA00556" || serial == "CUSA00557" ||
           serial == "CUSA00554";
}

/// Title-specific behavior switches. The game serial is constant for the lifetime of the process,
/// so these are evaluated once by SetGameSerial instead of comparing strings on hot paths (page
/// watcher updates run once per 4 KiB page, buffer lookups once per bound resource).
struct GameQuirks {
    /// Collapses read/write page watchers into one aggregate counter (PageManager fast path).
    bool fast_path_page_watchers{};
    /// Disables render-target/storage synchronization heuristics in the rasterizer.
    bool disable_render_sync{};
    /// Defers write protection of GPU-written ranges until the next CPU fence.
    bool defer_write_protect{};
    /// Skips render-target write recording even when render sync is otherwise enabled.
    bool skip_rt_write_record{};
    /// Texture cache image transition/tiling workarounds for CUSA01968/CUSA01936.
    bool image_transition_workaround{};
};

inline GameQuirks g_game_quirks{};

inline GameQuirks ComputeGameQuirks(std::string_view serial) {
    GameQuirks quirks{};
    quirks.fast_path_page_watchers =
        serial == "CUSA03173" || serial == "CUSA00900" || serial == "CUSA00208" ||
        serial == "CUSA01363" || serial == "CUSA01322" || serial == "CUSA003027" ||
        serial == "CUSA00299" || serial == "CUSA00207" || serial == "CUSA03014" ||
        serial == "CUSA03023" || serial == "CUSA50617" || serial == "CUSA18723" ||
        serial == "CUSA28863";
    quirks.disable_render_sync =
        serial == "CUSA03173" || serial == "CUSA00900" || serial == "CUSA00208" ||
        serial == "CUSA01363" || serial == "CUSA01322" || serial == "CUSA003027" ||
        serial == "CUSA00299" || serial == "CUSA00207" || serial == "CUSA03014" ||
        serial == "CUSA03023" || serial == "CUSA03388" || serial == "CUSA01589" ||
        serial == "CUSA01760" || serial == "CUSA07439" || serial == "CUSA07339" ||
        serial == "CUSA08692" || serial == "CUSA08495" || serial == "CUSA50617" ||
        serial == "CUSA18723" || serial == "CUSA28863" || serial == "CUSA00093" ||
        serial == "CUSA00003";
    quirks.defer_write_protect = serial == "CUSA00093" || serial == "CUSA00003";
    quirks.skip_rt_write_record = serial == "CUSA11227" || serial == "CUSA12982" ||
                                  serial == "CUSA00093" || serial == "CUSA00003" ||
                                  serial == "CUSA01778" || serial == "CUSA01627";
    quirks.image_transition_workaround = serial == "CUSA01968" || serial == "CUSA01936";
    return quirks;
}

/// Sets the active game serial and refreshes the cached quirk table.
inline void SetGameSerial(std::string serial) {
    g_game_serial = std::move(serial);
    g_game_quirks = ComputeGameQuirks(g_game_serial);
}

[[nodiscard]] inline const GameQuirks& Quirks() noexcept {
    return g_game_quirks;
}

struct patchInfo {
    std::string gameSerial;
    std::string modNameStr;
    std::string offsetStr;
    std::string valueStr;
    std::string targetStr;
    std::string sizeStr;
    bool isOffset;
    bool littleEndian;
    PatchMask patchMask;
    int maskOffset;
};

std::string convertValueToHex(const std::string type, const std::string valueStr);

void OnGameLoaded();
void AddPatchToQueue(const patchInfo& patchToAdd);
void ApplyRuntimePatch(const std::string& modNameStr, const std::string& offsetStr,
                       const std::string& valueStr, const std::string& targetStr,
                       const std::string& sizeStr, bool isOffset, bool littleEndian, int patchMask,
                       int maskOffset);
void PatchMemory(const patchInfo& patch);

static std::vector<int32_t> PatternToByte(const std::string& pattern);
uintptr_t PatternScan(const std::string& signature);

struct PendingPatch {
    std::string modName;
    std::string address;
    std::string value;
    std::string target;
    std::string size;
    bool littleEndian = false;
    PatchMask mask = PatchMask::None;
    int maskOffset = 0;
};

} // namespace MemoryPatcher