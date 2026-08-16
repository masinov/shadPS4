// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <deque>
#include <utility>

#include <imgui.h>
#include "common/assert.h"
#include "common/config.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/stb.h"
#include "common/thread.h"
#include "imgui_impl_vulkan.h"
#include "texture_manager.h"

namespace ImGui {

namespace Core::TextureManager {
struct Inner {
    std::atomic_int count = 0;
    ImTextureID texture_id = nullptr;
    u32 width = 0;
    u32 height = 0;

    Vulkan::UploadTextureData upload_data{};

    ~Inner();
};
} // namespace Core::TextureManager

using namespace Core::TextureManager;

RefCountedTexture::RefCountedTexture(Inner* inner) : inner(inner) {
    ++inner->count;
}

RefCountedTexture RefCountedTexture::DecodePngTexture(std::vector<u8> data) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngTexture(std::move(data), core);
    return RefCountedTexture(core);
}

RefCountedTexture RefCountedTexture::DecodePngFile(std::filesystem::path path) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngFile(std::move(path), core);
    return RefCountedTexture(core);
}

RefCountedTexture::RefCountedTexture() : inner(nullptr) {}

RefCountedTexture::RefCountedTexture(const RefCountedTexture& other) : inner(other.inner) {
    if (inner != nullptr) {
        ++inner->count;
    }
}

RefCountedTexture::RefCountedTexture(RefCountedTexture&& other) noexcept : inner(other.inner) {
    other.inner = nullptr;
}

RefCountedTexture& RefCountedTexture::operator=(const RefCountedTexture& other) {
    if (this == &other)
        return *this;
    RefCountedTexture copy{other};
    std::swap(inner, copy.inner);
    return *this;
}

RefCountedTexture& RefCountedTexture::operator=(RefCountedTexture&& other) noexcept {
    if (this == &other)
        return *this;
    std::swap(inner, other.inner);
    return *this;
}

RefCountedTexture::~RefCountedTexture() {
    if (inner != nullptr) {
        if (inner->count.fetch_sub(1) == 1) {
            delete inner;
        }
    }
}

RefCountedTexture::Image RefCountedTexture::GetTexture() const {
    if (inner == nullptr) {
        return {};
    }
    return Image{
        .im_id = inner->texture_id,
        .width = inner->width,
        .height = inner->height,
    };
}

RefCountedTexture::operator bool() const {
    return inner != nullptr && inner->texture_id != nullptr;
}

struct Job {
    Inner* core;
    std::vector<u8> data;
    std::filesystem::path path;
};

struct UploadJob {
    Inner* core = nullptr;
    Vulkan::UploadTextureData data{};
    int tick = 0; // Used to skip the first frame when destroying to await the current frame to draw
};

// Protected by g_job_list_mtx. Keeping the state under the same lock as the queue gives the
// condition variable a reliable predicate and avoids lost wakeups during startup and shutdown.
static bool g_is_worker_running = false;
static std::jthread g_worker_thread;
static std::condition_variable g_worker_cv;

static std::mutex g_job_list_mtx;
static std::deque<Job> g_job_list;

static std::mutex g_upload_mtx;
static std::deque<UploadJob> g_upload_list;

namespace Core::TextureManager {

void ReleaseInner(Inner* core) {
    if (core != nullptr && core->count.fetch_sub(1) == 1) {
        delete core;
    }
}

Inner::~Inner() {
    if (upload_data.im_texture != nullptr) {
        std::unique_lock state_lk{g_job_list_mtx};
        if (g_is_worker_running) {
            std::unique_lock lk{g_upload_mtx};
            g_upload_list.emplace_back(UploadJob{
                .data = this->upload_data,
                .tick = 2,
            });
            return;
        }
        state_lk.unlock();
        upload_data.Destroy();
    }
}

void WorkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ImGuiTextureManager");
    while (true) {
        Job job;
        {
            std::unique_lock lk{g_job_list_mtx};
            g_worker_cv.wait(lk,
                             [] { return !g_is_worker_running || !g_job_list.empty(); });
            if (!g_is_worker_running && g_job_list.empty()) {
                break;
            }
            job = std::move(g_job_list.front());
            g_job_list.pop_front();
        }

        auto [core, png_raw, path] = std::move(job);

        if (Config::getVkCrashDiagnosticEnabled()) {
            // FIXME: Crash diagnostic hangs when building the command buffer here
            ReleaseInner(core);
            continue;
        }

        if (!path.empty()) { // Decode PNG from file
            Common::FS::IOFile file(path, Common::FS::FileAccessMode::Read);
            if (!file.IsOpen()) {
                LOG_ERROR(ImGui, "Failed to open PNG file: {}", path.string());
                ReleaseInner(core);
                continue;
            }
            png_raw.resize(file.GetSize());
            file.Seek(0);
            file.ReadRaw<u8>(png_raw.data(), png_raw.size());
            file.Close();
        }

        int width{};
        int height{};
        stbi_uc* pixels =
            stbi_load_from_memory(png_raw.data(), png_raw.size(), &width, &height, nullptr, 4);
        if (pixels == nullptr || width <= 0 || height <= 0) {
            const char* reason = stbi_failure_reason();
            LOG_ERROR(ImGui, "Failed to decode PNG texture: {}",
                      reason != nullptr ? reason : "unknown error");
            stbi_image_free(pixels);
            ReleaseInner(core);
            continue;
        }

        auto texture = Vulkan::UploadTexture(pixels, vk::Format::eR8G8B8A8Unorm, width, height,
                                             static_cast<size_t>(width) *
                                                 static_cast<size_t>(height) * 4 * sizeof(stbi_uc));
        stbi_image_free(pixels);

        core->upload_data = texture;
        core->width = width;
        core->height = height;

        std::unique_lock upload_lk{g_upload_mtx};
        g_upload_list.emplace_back(UploadJob{
            .core = core,
        });
    }
}

void StartWorker() {
    std::unique_lock lk{g_job_list_mtx};
    ASSERT(!g_is_worker_running && !g_worker_thread.joinable());
    g_is_worker_running = true;
    g_worker_thread = std::jthread(WorkerLoop);
}

void StopWorker() {
    std::deque<Job> canceled_jobs;
    {
        std::unique_lock lk{g_job_list_mtx};
        ASSERT(g_is_worker_running);
        g_is_worker_running = false;
        canceled_jobs.swap(g_job_list);
    }
    g_worker_cv.notify_all();
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }

    for (auto& job : canceled_jobs) {
        ReleaseInner(job.core);
    }

    // No more submissions occur after shutdown starts. Release decoded-but-not-submitted textures
    // and delayed destruction jobs while the Vulkan backend is still alive.
    std::deque<UploadJob> pending_uploads;
    {
        std::unique_lock lk{g_upload_mtx};
        pending_uploads.swap(g_upload_list);
    }
    for (auto& upload : pending_uploads) {
        if (upload.core != nullptr) {
            upload.core->upload_data.Destroy();
            upload.core->texture_id = nullptr;
            ReleaseInner(upload.core);
        } else {
            upload.data.Destroy();
        }
    }
}

void DecodePngTexture(std::vector<u8> data, Inner* core) {
    Job job{
        .core = core,
        .data = std::move(data),
    };
    std::unique_lock lk{g_job_list_mtx};
    if (!g_is_worker_running) {
        return;
    }
    ++core->count;
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

void DecodePngFile(std::filesystem::path path, Inner* core) {
    Job job{
        .core = core,
        .path = std::move(path),
    };
    std::unique_lock lk{g_job_list_mtx};
    if (!g_is_worker_running) {
        return;
    }
    ++core->count;
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

void Submit() {
    UploadJob upload;
    {
        std::unique_lock lk{g_upload_mtx};
        if (g_upload_list.empty()) {
            return;
        }
        // Upload one texture at a time to avoid slow down
        upload = g_upload_list.front();
        g_upload_list.pop_front();
        if (upload.tick > 0) {
            --upload.tick;
            g_upload_list.emplace_back(upload);
            return;
        }
    }
    if (upload.core != nullptr) {
        upload.core->upload_data.Upload();
        upload.core->texture_id = upload.core->upload_data.im_texture;
        ReleaseInner(upload.core);
    } else {
        upload.data.Destroy();
    }
}
} // namespace Core::TextureManager

} // namespace ImGui
