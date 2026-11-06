# Cleanup and improvement backlog

Candidates found while reviewing the fork at `eaf8b2d8`. Sections are ordered by value; within
a section, items are independent.

## Done on 2026-10-09 (working tree, on top of `eaf8b2d8`)

| Item | Change |
| --- | --- |
| A1 | `ServiceFaultRange` now abstains for non-color images, for images not tracked over their full range, and for images whose bytes the buffer side GPU-wrote. `ReadMemory` captures the faulted pages' buffer-written byte ranges before the download erases them. Abstaining is the pre-service behaviour, so the change can only make the service write less. |
| A3 | `aio.cpp` `ReadNative` reads into host memory and copies. |
| A4 | The preemptive-download service retires every entry it wrote, whole. |
| A5 | Dead `slot_buffers` read removed from `ReadMemory`. |
| B1 (part) | The validator rotates its 4 KiB window through the range. The image-alias filter is not done. Stale `config.cpp` comment fixed. |
| B2 | `region_manager.h` comments corrected. |
| B3 | `Buffer replace:` is sampled. |
| B4 | Collision probe removed, with its counters and the `overlap=` field of the `Buffer cache:` line. Report tooling that parses that line needs to drop the field. |
| E | `ForEachBackingChunk` doc comment corrected. Neither it nor `TryWriteBacking` copies memory-area structures any more. |
| New | `TryWriteBacking` advances its source pointer across physical areas (inherited bug; see [02-vertex-explosions.md](02-vertex-explosions.md)). |
| E0 | Eviction readback runs in every readback mode again. `GC gpu-modified skip sample` is sampled. |
| Run 81 | Compute-shader loops are bounded (`computeLoopIterationLimit`, build `loop-limit`) after the indirect-dispatch device loss recurred. An incompatible on-disk shader cache is now rebuilt; before, it was ignored and never rewritten, so every later session recompiled everything. |

One side effect to know about for A1: the buffer-written byte set is only cleared by a download
or flush, so memory that was once GPU-written as a buffer and later becomes a render target
with no CPU write in between will make the service abstain for that target. That costs the
service's benefit there (the mode-3 texture flashing it was built to cure), not correctness.

Still open: A2, the rest of B1, and sections C–F.

## A. Correctness

1. **Give `ServiceFaultRange` the eviction path's guards.**
   `texture_cache.cpp:339`. Before writing texels back: require the color aspect, require the
   image to be tracked over its full range, and abstain when the buffer side has GPU-authored
   bytes inside the image's own byte range. The page tracker cannot answer the last question
   after `DownloadBufferMemory` has cleared the faulted page, so capture it before the buffer
   download (the byte-granular `gpu_modified_ranges` set, intersected with the image range).
   Clear tracker bits only for the pages actually written. Background in
   [02-vertex-explosions.md](02-vertex-explosions.md#1-the-image-fault-service-is-missing-guards).

2. **Mark command-processor writes as CPU-modified.**
   `liverpool.cpp:790, 806, 829` write labels and counters to guest backing with
   `TryWriteBacking` and tell nobody (`vk_predication.cpp:485` does the same call; I did not
   read its surroundings). The cached device copy of that page
   goes stale. One `InvalidateMemory`-style mark after each write fixes it and removes the
   validator's single-word hits.

3. **Bounce-buffer `aio.cpp` `ReadNative`.**
   Line 264 reads from the host file directly into guest memory. Read into a host buffer and
   `memcpy`, as `ReadFile` and the storage scheduler already do, so a page re-protected in the
   meantime faults instead of failing the read.

4. **Fix the preemptive-download service** (inert for Bloodborne, live for CUSA00093 and
   CUSA00003). `buffer_cache.cpp:134-143` writes each intersecting entry whole and subtracts
   one page. Either subtract the whole entry it wrote, or write only the faulted page's slice.

5. **Remove the dead read in `BufferCache::ReadMemory`.**
   Lines 235-238 index `slot_buffers` with a possibly null `BufferId` on the faulting guest
   thread, concurrently with the GPU thread, and never use the result.

## B. Make the instruments tell the truth

1. **Content validator coverage.** Sample a random 4 KiB page within the synchronized range
   instead of always the first. Skip ranges covered by a texel-buffer image alias so the
   benign population stops dominating the count. Update the stale comment above
   `useContentValidator` in `config.cpp`.

2. **Correct the comment in `region_manager.h:152-159`.** It states the reorder closed the
   series' explosions. It did not, and the old order had no lost-write window. Keep the code,
   replace the comment with one line saying the order is deliberate but equivalent.

3. **Sample or demote `Buffer replace:`** (`buffer_cache.cpp:1102`). It logs every
   replacement, 9,176 lines in run 77, on a synchronous log on an HDD. It was added for one
   experiment (run 40).

4. **Remove or sample the collision probe** in `SynchronizeBuffer`
   (`buffer_cache.cpp:1423`). An extra locked tracker walk on roughly five million calls per
   session for a counter that has fired once.

## C. Dormant and inert code

Each of these is a decision for the owner, not a defect. Listed with what keeping it costs.

| Code | State | Cost of keeping | Suggestion |
| --- | --- | --- | --- |
| Sparse block reclaim: limbo, ghosts, adaptive age, deferred unbinds (`8fa394e3` … `44f843ae`, ~800 lines across `buffer.h`, `buffer.cpp`, `buffer_cache.cpp`, `vk_scheduler.*`, `sparse_coverage.h`, `memory_gc.h`) | Off by default since `8ff701e5`, judged unsound for the corruption it caused | Per-synchronization epoch stamping still runs; a large surface nobody exercises | Delete. It is in history if ever wanted. Otherwise at least skip `TouchSparseRange` when the toggle is off. |
| Eviction readback (`4e1f32de` and five follow-ups) | Runs only at `readbackSpeedMode = 0` | None at mode 3 | Keep while mode 0 remains a supported configuration; decide when production defaults are chosen. |
| `BufferCache::ReadEdgeImagePages` | No callers; holds a raw download-ring pointer across a deferred operation | A trap for whoever calls it next | Delete. |
| `DownloadBufferMemory<true>` | Never instantiated | Dead template branch | Collapse to the synchronous form. |
| `on_gpu` parameter of `DispatchIndirect` | Always `false`; the two marker branches in `liverpool.cpp:719-728` and `1293-1300` are identical | Noise | Remove the parameter and the duplicate branches. |
| `InvalidateMemory(..., bool download)` | Parameter unused | Noise | Remove. |
| `forceAccuratePageWatchers`, `forceRenderSync` | Off; measured at 40% GPU utilisation | Two config keys | Keep as experiment switches; they are cheap. |
| Dedicated present queue (`EnableDedicatedPresentQueue = false`, `vk_instance.cpp:664`) | Compiled out after run 22 | Dead branch | Delete or leave with its warning comment. |
| `useImageFaultService` | On | See A1 | Default to off until A1 is done, if the config-only test implicates it. |

## D. Comments that record history instead of describing code

Many comments added during the hunt narrate run numbers and retracted theories ("the
load-driven explosions of the whole series", "Run 59 intersected these ranges…", "run 73,
t=830.68"). Some of the narrated conclusions were later withdrawn, and the comments were not.
A reader cannot tell which still hold.

Suggested rule: a code comment states what the code guarantees and why the obvious alternative
is wrong. Run history belongs in commit messages and in these notes. A pass over
`buffer_cache.cpp`, `texture_cache.cpp`, `region_manager.h`, `vk_rasterizer.cpp`,
`vk_dispatch_guard.cpp` and `config.cpp` would shorten them considerably.

## E0. VRAM grows without bound at `readbackSpeedMode = 3` (measured, run 77)

Over the 70-minute run 77 session, device memory usage climbed from 3,883 to 5,618 MiB against
a 5,132 MiB budget, and the collector's unmet target grew from 286 to 1,844 MiB. Almost every
pass reads `evicted=0` with 45–70 `skipped_gpu_modified`: retired render targets hold
GPU-authored content and cannot be evicted without writing it back first. The mechanism built
for exactly that, eviction readback, has reported `nominated=0` since `7c8f651e` restricted it
to `readbackSpeedMode = 0`. The collector meanwhile runs about twice a second finding nothing
it may free.

This is the run 34 wall again (monotonic growth, then multi-second driver stalls), reintroduced
for mode 3. The gate's stated reason was caution, not an observed failure: "two overlapping
writers with live protection is untested interplay". Options: lift the gate (the readback's
guards do not depend on the mode, and it ran through runs 42–57), or leave it and accept the
growth. The gate was lifted on 2026-10-09 (build `review-fixes-2`); if anything looks wrong
with it, `useEvictionReadback = false` turns it off without a build.

## E. Performance observations (not measured)

These come from reading, not profiling. Measure before acting on any of them.

- `MemoryTracker::ForEachUploadRange` with `is_written` holds the region spin lock across
  `UploadCopies`, which copies guest memory and can block in `StreamBuffer::Map` waiting for
  the GPU. Guest threads faulting on that 4 MiB region spin for the duration. Inherited.
- `MemoryManager::ForEachBackingChunk` copies whole `VirtualMemoryArea` objects into a vector
  under the shared lock on each call, and its doc comment says it does not call `func` when it
  returns false, which is not what the loop does for a partially backed range.
- The image fault service and `RedownloadToGuest` each end in a full `scheduler.Finish()`.
  The write-fault service does this ~60,000 times a session already; an asynchronous or
  batched service is the known route to making readback affordable.
- Every indirect dispatch pays for the guard's extra pipeline bind, push descriptor, dispatch
  and two barriers. Its counter has never left zero.

## F. Upstream contribution candidates

Self-contained fixes to inherited bugs, each verified by test or by reading:

- `4803a0d3` download-ring alignment for tiler storage descriptors.
- `e00970d9` tracker-driven, page-granular fault service.
- `f3af96d8` `RefreshImage` hash-domain mismatch and `StreamBuffer::Commit` coalescing index.
- `49f95717` hash baseline recorded only on the upload path.
- `3a470952` rewind valid bit on compute queues.
- From `c53dde10`: `ObjectPool::Release`, `RuntimeInfo::operator==`, the `get_bda_pointer`
  range check and atomic fault write, the fault-parser count clamp, `posix_unlink` /
  `posix_rename` null and error handling.

Not candidates: `33eb27b2` (no behavioural change) and `f4430c26` as an explosion fix (though
it is a valid guard for the titles that use preemptive downloads).
