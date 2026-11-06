# Shadlix hardening notes

Review of the 49 commits on `Shadlix` above the fork point `e2f4d441` ("Fixing Rebase",
2026-08-07), written 2026-10-09 against HEAD `eaf8b2d8`. Test title throughout is Bloodborne
(CUSA03173).

| Document | Contents |
| --- | --- |
| [01-commit-review.md](01-commit-review.md) | Every commit: what it does, whether it is live for the tester's setup, and what the review found |
| [02-vertex-explosions.md](02-vertex-explosions.md) | State of the vertex-explosion investigation: what is established, what was ruled out, new findings and next steps |
| [03-cleanup-backlog.md](03-cleanup-backlog.md) | Cleanup and improvement candidates, ordered by value |

## Where things stand

- **Size of the fork:** 99 files, +8,550 / −946 lines. Roughly 60% of that landed in the first
  commit (`c53dde10`, a squash of the Codex-era work); the rest is 48 small commits from
  Aug 17–22.
- **Working well:** VRAM garbage collection, sparse cache buffers, the download-ring alignment
  fix, the fault-service contract fix, crash/terminate dumps, log rotation and timestamps.
- **Still open:** vertex explosions (rarer, not gone), the indirect-dispatch device loss
  (GDS producer/consumer shaders on a single Vulkan queue), and one unexplained CPU-side
  freeze (run 76).
- **Last analysed tester report:** run 77. Run 80 (`async-validator` build, the current HEAD)
  was sent on Aug 22 and no report for it exists in `local-reports/`.

## Tester configuration assumed in these notes

Taken from the run 77 report's `config.toml`. Several conclusions depend on it, so re-check
if it has changed.

| Key | Value | Consequence |
| --- | --- | --- |
| `readbackSpeedMode` | 3 (Default) | Write-fault service is live; eviction readback is switched off by its own gate |
| `useSparseCacheBuffers` | true | Buffer growth aliases memory blocks instead of copying |
| `useSparseBlockReclaim` | absent (false) | The whole reclaim/limbo/ghost subsystem is dormant |
| `directMemoryAccess` | true | |
| `copyGPUBuffers` | false | |
| `app0_read_bandwidth_mibps` | 12 | Asset reads go through the storage scheduler |
| Bloodborne quirks | `fast_path_page_watchers`, `disable_render_sync` | Hard-wired per title in `memory_patcher.h` |

## Sources used

- The repository history and the code at HEAD.
- `local-reports/` (gitignored): per-run report bundles with `ANALYSIS.md` /
  `IMPLEMENTATION.md`, raw logs and telemetry through run 77.
- `Build/RUN71..80-INSTRUCTIONS.txt`: what the tester was asked to do for each build.
- Claude memory notes for this project (`~/.claude/projects/C--Users-Datision-projects-shadPS4/memory/`),
  a running log through run 79.
- `~/.claude/history.jsonl`: the 115 prompts of the earlier Claude session
  (`3c74d891-0e70-456b-b10d-d8bb5eeb0507`, Aug 16–22).
- `~/.codex/sessions/2026/08/13/rollout-2026-08-13T01-07-54-019ff83b-….jsonl` (87 MB): the
  original Codex session, Aug 13–16, still on disk.

The transcript of the earlier Claude session itself is no longer on disk: only its prompts,
its memory notes and its outputs in `local-reports/` and `Build/` survive.

## Scope of this review

Read in full at HEAD: the dirty tracker (`region_manager.h`, `memory_tracker.h`,
`page_manager.cpp`), `buffer_cache.cpp`, the stream buffer, the texture-cache readback and
fault-service paths, the dispatch guard, and the file-read entry points. Read as diffs: the
rasterizer, command processor, shader recompiler, fault manager, memory manager and file
system changes.

Not reviewed line by line: the ImGui texture manager, presenter, `vk_instance.cpp`,
`vk_scheduler.cpp`, HTTP, input handler, and the workflow script. Nothing was built or run.
