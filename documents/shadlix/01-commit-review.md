# Commit review

All 49 commits above `e2f4d441`, oldest first. "Status" is for the tester's setup described in
the [README](README.md): Bloodborne, `readbackSpeedMode = 3`, sparse buffers on, reclaim off.

- **Live**: runs in normal play.
- **Dormant**: behind a toggle that is off.
- **Inert**: compiled in and enabled, but cannot execute for this title or this readback mode.
- **Diag**: logging, counters or crash evidence only.
- **Tool**: build and exchange scripts.

## Summary

| Status | Commits |
| --- | --- |
| Live, sound (two only partly live) | 25 |
| Live, with a finding | 6 |
| Dormant | 7 |
| Inert | 4 |
| Diag / Tool | 7 |

The six with findings: `33eb27b2` (no-op with a misleading comment), `f6a4ae52` (image fault
service lacks guards; per-sync probe cost), `10ad5997`/`0cee66f9`/`eaf8b2d8` (validator blind
spots), `b0c05d6a` (unsampled per-replacement log).

## Aug 16: the squash

### `c53dde10` Production hardening: memory lifetime, VRAM GC, draw ordering, diagnostics
81 files, +4,966 / −873. Everything from the Codex sessions plus the first Claude pass.
**Live.** Main pieces:

- **VRAM GC rewrite.** The upstream asynchronous collectors were replaced with a synchronous
  shared collector driven from `Rasterizer::OnSubmit` (pressure tiers, hysteresis, object
  budgets in `memory_gc.h`) plus allocation-driven reclaim (`ReclaimForAllocation`, preflight
  and emergency) with retirements tied to each resource's exact last-use tick.
- **Two-phase draw binding.** `PrepareVertexBuffers` / `PrepareIndexBuffer` resolve every
  cache buffer before any draw state is recorded, so an allocation-triggered flush cannot
  split a draw from its bindings.
- **Quirk cache.** Per-title serial comparisons moved out of hot paths into
  `MemoryPatcher::Quirks()`.
- **Page-watcher robustness.** `PageWatcherCounterOverflow` extends the u8 per-page counters;
  `CheckAndRepairWriteFault` re-applies a lost protection transition and logs faults that make
  no progress.
- **Direct-memory shaders.** `get_bda_pointer` range-checks the address, uses an atomic OR on
  the fault bitmap; the fault parser clamps its count and synchronizes the buffers it
  materializes.
- **Shader cache keys.** `RuntimeInfo::operator==` compared `num_invocations && other.num_invocations`
  (a truthiness test) and skipped several fields; now compares all of them.
- **`ObjectPool`.** `Release` destroyed `Storage` objects, whose destructor is empty, so `T`
  was never destroyed.
- **File system.** POSIX share semantics for guest files; `rename`/`unlink` report errors
  instead of throwing or silently succeeding; null checks before `strlen`.
- **Diagnostics.** GPU checkpoints (`gpu_diagnostic_tracker.h`, `device_address_tracker.h`),
  present-phase timing, focused gtests under `tests/`.

Reviewed at HEAD only where it touches the tracker, buffer cache and rasterizer. The ImGui,
presenter, instance and HTTP parts were not reviewed.

## Aug 17: sparse cache buffers and GC tuning

| Commit | Status | Notes |
| --- | --- | --- |
| `0bcd629b` Sample chain-advance allocation logs | Diag | |
| `18372add` Sparse-resident cache buffers (opt-in) | Live | Default is off; the tester enables it. Merges alias 64 KiB memory blocks instead of copying. Run 56 showed explosions with it off too, so it is not their cause. |
| `20079464` Sampled statistics line | Diag | The `Buffer cache:` line every report relies on. |
| `83e6adff` GC skips objects under 2 MiB | Live | Fixed the run 28 bind thrash. |
| `35286a2f` Stream reserve capped at 64 MiB | Live | |
| `f7dd6d15` Small-object skips don't consume the inspection budget | Live | |
| `efa7f0e7` Eviction ages in submits; dispatch-arg snapshot | Live | The snapshot `memcpy`s guest memory directly on the GPU thread. Safe under the fast-path quirk (pages are never read-protected); would self-fault with `forceAccuratePageWatchers`. |
| `7098d2f7` Batched sparse binds; overshoot age; `on_gpu` | Live | After this fix every `DispatchIndirect` call site passes `on_gpu = false`, and the marker/no-marker branches in `liverpool.cpp` are now identical. The parameter is vestigial. |
| `74e45f38` Tile-manager checkpoints; smaller OS reserve | Live | |
| `4e1f32de` GC-driven asynchronous eviction readback | **Inert** | Gated to `readbackSpeed == Disable` by `7c8f651e`. Runs only if the tester goes back to mode 0. |
| `58f7fe05` Readback guards; last-resort minidump | Inert / Diag | The minidump filter is live and has been useful. |

## Aug 18: readback refinement, ring alignment, block reclaim

| Commit | Status | Notes |
| --- | --- | --- |
| `c56a4900`, `2b7ffcc9` Workflow: `.env`, Send / FetchReport | Tool | |
| `bbcbdcd6` Abort on buffer-side GPU aliasing | Inert | This guard exists only on the eviction path. See `f6a4ae52`. |
| `03915efa` Viability at nomination; dominated aliases | Partly live | `IsReadbackViable` is shared with the fault service. |
| `34e75a34` Readback transfers authority only; tiler scratch zero-fill | Partly live | The zero-fill in `TileImage` applies to every download. |
| `b0c05d6a` Per-replacement logging; color-only readback | Live, **finding** | `Buffer replace:` is logged unsampled: 9,176 lines in run 77 on a synchronous log. The color-only rule exists only on the eviction path. |
| `4803a0d3` Align download-ring mappings used as storage descriptors | **Live, keep** | A real inherited bug: `Map(copy_size, true)` passed `true` as the alignment. Confirmed by the run 41/42 A/B. Upstream candidate. |
| `8fa394e3` Sparse cold-block reclamation | Dormant | |
| `32fcdb3c` Adaptive reclaim age | Dormant | |
| `4dec248d` Ghost accounting | Dormant | |
| `0240027c` Victim-stage limbo, deferred unbinds | Dormant | |
| `af84205d` Limbo revalidation, grace tiers | Dormant | |
| `44f843ae` Constant emergency tiers | Dormant | |
| `8ff701e5` Default block reclaim off | Live | Six commits and roughly 800 added lines are dormant behind this one default. `EnsureRangeBound` still stamps block epochs on every synchronization for a sweep that never runs. |

## Aug 19: the explosion hunt

| Commit | Status | Notes |
| --- | --- | --- |
| `682a9f9f` Strict age dominance for alias conversion | Live | |
| `513a2937` Revalidate buffer binding hints | Live | `stale_hints` has read 0 in every session. Cheap; keep as a guard. |
| `ba63a4d9` Vertex-input residency validation | Live | 5 hits in a 91-minute session. The "halved stalls" attributed to it in run 55 was never tested against a control. |
| `7c8f651e` Backing-view hashing; gate eviction readback; counters | **Live, keep** | Internal hashing no longer faults through armed protection. |
| `e00970d9` Fault-service contract | **Live, keep** | Fixed 64,140 services that wrote back zero bytes. The preemptive branch it left alone has its own defect; see `f4430c26`. |
| `33eb27b2` Arm write watchers before clearing dirty bits | Live, **finding** | See below. |
| `f158e6d1` Execution-time clamp for indirect dispatch | Live | One extra compute dispatch, pipeline bind and two barriers per indirect dispatch. |
| `81cea8f2` Record the clamp before the guest descriptor push | Live | Fixed a descriptor clobber introduced by the previous commit. |
| `f5c37777` Track the clamp's compute-stage read | Live | |
| `d7474d86` Timestamps, upload gauge, terminate dumps | Diag | The map/unmap logging it added was retired in `f6a4ae52`. |

### Finding: `33eb27b2` changes nothing

The commit message and code comment describe a "lost-write window": a guest write landing
between clearing the dirty bits and `mprotect` would neither fault nor stay marked.

That write was never lost. In both the old and new order the sequence under the region lock is
(a) clear bits and protect, in either order, then (b) enumerate ranges, and only afterwards
(c) `UploadCopies` reads guest memory. A write that lands before the `mprotect` is already in
RAM when (c) copies it. A write after the `mprotect` faults and re-marks the page. The old
order had no window.

This matches the test record: three clean legs in run 63, then the same binary exploded in
run 66. The reorder is harmless and can stay, but the comment in `region_manager.h:152-159`
claims to have closed "the load-driven explosions of the whole series", which is false and
will mislead the next reader.

## Aug 20: device loss, evidence, the image fault service

| Commit | Status | Notes |
| --- | --- | --- |
| `3a470952` Honor the rewind valid bit on compute queues | Live, keep | Matches the graphics queue. Partial: both dispatch pipelines lost the device again afterwards. |
| `8f1f1b31` Terminate dumps written directly; log rotation | Diag | Its message wrongly attributes a restart to the terminate handler; `c877621c` corrects that. |
| `c877621c` Log and flush guest `LoadExec` | Diag | Explained the "silent death" class: the game relaunching itself. |
| `f6a4ae52` Download barriers, image fault service, collision probe | Live, **finding** | See below. |
| `e0945b18` `forceAccuratePageWatchers` / `forceRenderSync` | Dormant | Run 73 measured the cost: GPU utilisation 40% against 99%. |
| `f3af96d8` Precise fault-service gate; two inherited bugs | Live | The `RefreshImage` hash-domain fix and the `StreamBuffer::Commit` coalescing fix are both correct as far as I can tell. I traced the coalescing against the ring's wrap and wait logic and found no hazard. |

### Finding: `f6a4ae52`

Three independent parts.

1. **Download barriers** in `DownloadBufferMemory`: sound, keep.
2. **Image fault service** (`TextureCache::ServiceFaultRange`): writes an image's texels over
   its whole guest range after a buffer-side write fault. It carries fewer guards than the
   eviction readback it was modelled on. Details in
   [02-vertex-explosions.md](02-vertex-explosions.md#1-the-image-fault-service-is-missing-guards).
3. **Collision probe**: `GetCpuGpuOverlapPages` runs on every `SynchronizeBuffer`, which is
   about 5 million times a session, each taking the region lock an extra time. It has fired
   once in the project's history, and in mode 3 the fault service makes the state it looks for
   nearly unreachable. It no longer earns its place on the hottest path.

## Aug 21–22: validator, ring guard, hash baseline

| Commit | Status | Notes |
| --- | --- | --- |
| `c552fa47` Gate the image fault service; fix predicate order | Live | The default was flipped back to `true` in the next commit. |
| `0d266a53` Dispatch guard skips corrupt arguments | Live | The original clamp turned garbage into 65535³ workgroups. |
| `10ad5997` Content validator | Live, **finding** | See below. |
| `f4430c26` Guard parked preemptive-download pointers | **Inert** | See below. |
| `49f95717` Never mint a hash baseline from post-write content | Live, keep | Sound on its own merits, though it was not the elevator bug. |
| `0cee66f9` Validator v2: unbacked and race filters | Live, finding | |
| `eaf8b2d8` Validator never waits on the GPU | Live, finding | Removes the ~21 GPU drains per second that caused the slow motion and elevator clipping. |

### Finding: `f4430c26` guards a path Bloodborne never takes

Preemptive downloads are created only in `CommitPendingGpuRanges`, from
`gpu_modified_ranges_pending`. The only code that adds to that set is
`buffer_cache.cpp:658-659`, under `Quirks().defer_write_protect`, which is set for CUSA00093
and CUSA00003 only. For Bloodborne the set is always empty, no preemptive download ever
exists, and `stale_rescues` can only read 0, which it has.

The memory notes call this stale-pointer mechanism "the strongest mechanism candidate yet" for
the explosions. It is unreachable for the test title. The guard itself is correct and is real
protection for the two titles that do use the path.

While there: the preemptive branch of `DownloadBufferMemory` (lines 134-143) writes back each
intersecting entry **whole** but subtracts only the faulted page from the map. The remaining
pieces keep the original address, pointer and size, so a later fault or sweep rewrites the
whole entry again, including pages the CPU has written since. Inherited, and inert for
Bloodborne, but a real overwrite hazard for the `defer_write_protect` titles.

### Finding: the content validator sees very little

The invariant is the right one. The implementation samples it thinly:

- It compares only the **first 4 KiB** of a synchronized range. A stale page anywhere else in
  a vertex buffer is invisible to it.
- It runs only for synchronizations that upload nothing and are not GPU-modified, one in 4,096.
- Its three "populations" in run 77 have ordinary explanations (see
  [02-vertex-explosions.md](02-vertex-explosions.md#what-the-run-77-validator-hits-actually-are)).

So "0 bad" does not retire the lost-upload family, which is how the memory notes planned to
read it. Also: the comment above `useContentValidator` in `config.cpp` still describes the
original design ("a small download and drain every few seconds").
