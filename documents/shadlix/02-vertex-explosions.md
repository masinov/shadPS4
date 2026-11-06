# Vertex explosions: state of the investigation

Written 2026-10-09 from the code at `eaf8b2d8`, the reports through run 77 and the session
notes through run 79. The root cause is **not found**. This document records what is
established, what has been ruled out, what this review adds, and what to do next.

## The defect

Geometry stretches into spikes, sheets or ribbons. What the tester has established:

- It predates the fork: it happens on stock upstream builds and is known to the community.
  The tester's game copy carries a community vertex-explosion patch that reduces it.
- **Once an asset explodes it stays exploded until that asset is reloaded.**
- Animation still applies correctly to the exploded mesh (Hunter's Dream flowers swaying in
  the wind while spiked). Shader, transforms and animation inputs are fine; the values feeding
  vertex position are wrong.
- The victim varies: NPC body, the player's gun, environment, flowers.
- No fault, assert, device loss or counter anomaly has ever accompanied one.

Permanent-until-reload means some copy of the data went wrong once and nothing repaired it.
There are three places that can be: guest RAM itself, the cached device copy, or the address a
draw resolves to.

## Ruled out by test

Each of these was removed or changed and explosions continued.

| # | Hypothesis | Run |
| --- | --- | --- |
| 1 | Eviction-readback tiler writing into the previous ring block | Fixed in 42 (a real bug); explosions returned in 45 |
| 2 | Sparse block reclaim | 50 |
| 3 | Frame-rate-dependent game physics (60 FPS patch) | 51 |
| 4 | Model LOD patch | 52 |
| 5 | Stale buffer slot in two-pass binding | 54 (`stale_hints = 0`) |
| 6 | Vertex-input residency | 55 (`vertex_misses = 0`) |
| 7 | The whole sparse subsystem | 56 |
| 8 | Readback disabled (mode 0 against mode 3) | 58 |
| 9 | Fault service writing back zero bytes | 60 (a real bug, fixed) |
| 10 | Stale direct-memory reads | 61 |
| 11 | Torn command buffers (`copyGPUBuffers`) | 62 |
| 12 | "Lost-write window" in the tracker | 66; and see [the commit review](01-commit-review.md#finding-33eb27b2-changes-nothing) — there was no window |
| 13 | Image-authored content missing from buffers | 73 |
| 14 | CPU reads of GPU memory that cannot fault | 73, all quirks off |
| 15 | CPU-dirty and GPU-modified on the same page at upload | 72 (probe fired once in 5 million uploads) |

Runs 75–77 then went 157 minutes with no explosion while the tester tried to provoke one. No
change in those builds explains that, and the record has a precedent of three clean legs
followed by an explosion on the same binary, so it is not evidence of a fix.

## What this review adds

### 1. The image fault service is missing guards

`TextureCache::ServiceFaultRange` (`texture_cache.cpp:339`), default on since `0d266a53`.

After a guest write faults on a page the buffer tracker has marked GPU-modified,
`BufferCache::ReadMemory` downloads the buffer's bytes for that page into guest RAM, then calls
`ServiceFaultRange`. For every GPU-modified image overlapping the page that passes its checks,
it downloads the image and writes the texels over the image's **entire guest range** with
`TryWriteBacking`, then clears the buffer tracker's GPU-modified bits for that whole range
(`MarkRegionAsFlushed`).

The eviction readback it was copied from learned three lessons the hard way that were not
carried over:

| Guard | Eviction readback | Fault service |
| --- | --- | --- |
| Abort if the buffer side has GPU-authored data in the image's range (run 36: tiled texels written over live vertex ranges) | Yes, at nomination and at apply | **No** |
| Color aspect only (depth round trip unverified; stencil plane is scratch fill) | Yes | **No** |
| Abort on any dirty flag | Yes (composite mask) | Only `CpuDirty` and `GpuDirty` |

The first row matters most. The service only runs because the **buffer** tracker says the
faulted page is GPU-authored. If a stale image object still covers that address range (the game
freed a render target and now uses the memory for compute output), nothing has marked the image
dirty, because no CPU write touched it. The service then overwrites the bytes the buffer
download just wrote with old texels, the page is uploaded from RAM on the next synchronization,
and the device copy of that page is replaced with garbage that stays until the game rewrites
it. That is the run 36 failure reached through a different door.

`MarkRegionAsFlushed` also rounds outward to whole pages, clearing GPU-modified bits on
boundary pages that the image shares with unrelated GPU-authored buffer data, without
downloading them.

A related gap: `TextureCache::InvalidateMemory` can untrack just the head or tail page of an
image without marking it dirty (lines 451-461). Neither readback path checks that the image is
still tracked over its full range before writing texels across all of it.

What this does and does not explain: explosions existed long before this service, so it is
not the original cause. It is a mechanism that is live today, in the default configuration, and
produces exactly the permanent-until-rewritten shape. Run 73 saw explosions continue with the
service running at twenty times its usual rate, which was read as "the image hole is not the
mechanism"; it is equally what a harmful service would produce.

**Cheapest test:** `useImageFaultService = false`, no build needed.

### 2. Two earlier "fixes" could not have fixed anything

- `33eb27b2` (lost-write window): the old ordering already captured the write. No behaviour
  changed.
- `f4430c26` (stale preemptive-download pointer): the path only exists under the
  `defer_write_protect` quirk, which Bloodborne does not have.

Both are harmless. Both should be struck from the list of things believed to be in place.

### What the run 77 validator hits actually are

120 "Silent cache divergence" lines, in three groups.

- **107 at three fixed buffer-start addresses, guest side all zeros.** The device side changes
  between hits and reads as depth values (floats between 0.98 and 1.0) or small integers. These
  look like ranges the game treats as images and the buffer cache fills by tiling the image
  into the buffer (`SynchronizeBufferFromImage`), which writes the device copy without marking
  it GPU-modified. Guest RAM never receives those bytes, by design. This is my reading of the
  bytes, not something I traced to a call site. The session notes attributed them to unbacked
  pages.
- **11 in one 160 MiB buffer**, plausible data on both sides, mismatching mid-page. That buffer
  is rewritten continuously; the v1 validator waited on the GPU between sampling the two sides,
  so the game moved on. Races, as the notes concluded.
- **2 with a single word differing** (guest `1`, device `0`). Consistent with the command
  processor's label and counter writes (`liverpool.cpp:790, 806, 829`,
  `vk_predication.cpp:485`), which go straight to guest backing and never mark the page
  CPU-dirty. Inherited. Harmless unless a shader reads the label from a cached buffer, but it
  is a genuine violation of the invariant, and fixable by marking the range dirty after the
  write.

None of the three is an explosion. Combined with the validator's coverage (first 4 KiB of a
range only), its silence says little either way.

### 3. Kernel-mode writes into guest memory bypass the tracker

The dirty tracker depends on a write to a protected page raising an access violation. A write
made by the Windows kernel on the emulator's behalf (`ReadFile` into a guest buffer) does not
raise one; it fails or is cut short.

- `ReadFile` in `file_system.cpp:344` and the storage scheduler's `Publish` read into a bounce
  buffer and `memcpy`, which faults normally. **Safe.** Bloodborne's `/app0` asset reads take
  the storage scheduler path with the tester's config.
- `aio.cpp:264` `ReadNative` invalidates the range and then reads **directly** into the guest
  buffer. If the GPU thread re-protects a page of that range in between (a draw synchronizing a
  neighbouring allocation on the same 4 KiB page), the read hits a read-only page.

So for this tester the exposure is limited to asynchronous reads that are not scheduled `/app0`
reads. It is a latent bug worth closing with the same bounce buffer, not a leading suspect.

### 4. `MemoryManager::TryWriteBacking` repeated its first bytes (fixed 2026-10-09)

Found after the rest of this document was written. Every GPU-to-guest write-back goes through
this function: the write-fault service, image downloads, eviction readback, command-processor
labels. It walks the physical areas behind the destination and copies a piece into each, but
it never advanced the **source** pointer:

```cpp
memcpy(backing, data, copy_size);   // data is the start of the buffer every time
size -= copy_size;
```

A write-back confined to one physical area is correct. One that crosses into a second area (or
a second mapping) writes the beginning of the source again where the continuation should be.
Guest RAM then holds wrong bytes that nothing marks as wrong, and they stay until the game
rewrites them.

Inherited: the loop arrived with upstream's sparse-physical-memory change (`3e1f5a0b`,
2026-01-21) and is identical at the fork point. `ForEachBackingChunk`, the read-side twin
added in this fork, does not have the problem.

This has the right shape for the explosions (silent, permanent until rewritten, rare, present
upstream, indifferent to every cache-side change that was tested), and how often it fires
depends on how often a write-back straddles a boundary, which I have not measured. It is not
confirmed as the cause. The fix is in the working tree; a boundary-crossing counter would tell
how common the case was.

## What is still unexamined

- **Guest RAM being wrong at the source.** Nothing checks the bytes the game itself produced.
  If decompression or a CPU-side computation yields bad vertex data (an instruction emulation
  fault, for instance), every cache in the emulator faithfully carries it and every
  instrument built so far reads clean.
- **The draw reading the right buffer at the wrong offset**, or with the wrong stride or
  format. The validator cannot see this either.
- **Which of the two the explosion is.** The flower observation leaves "vertex positions are
  garbage" and "one constant is garbage" both open.

## Recommended next steps

Ordered by cost to the tester.

1. **Answer from memory, no session needed.** Which build and which `readbackSpeedMode` is in
   use now? Is `useImageFaultService` still at its default? Roughly how often do explosions
   appear, and in the same places?
2. **Config-only:** play normally with `useImageFaultService = false`. If the residual
   explosions stop, finding 1 is the cause and the service needs the eviction path's guards
   before it is turned back on.
3. **One RenderDoc capture of a persistent explosion.** `Build/RENDERDOC-CAPTURE-GUIDE.txt`
   was sent in the run 75 era; no result is recorded. Because the explosion persists, this
   needs no luck: walk to the exploded asset, press F12. A single capture shows the draw call,
   the vertex buffer contents, the constants and the guest addresses, and separates "bad
   vertex data" from "bad constant" from "wrong offset" outright. After fifteen eliminations
   by inference, this is the highest-value action available.
4. **Code, no tester time:** the fixes in [03-cleanup-backlog.md](03-cleanup-backlog.md),
   sections A and B.
5. **If a capture shows bad vertex bytes:** make the validator useful for that address. Sample
   a random page of the range instead of the first, and on a mismatch also log whether guest
   RAM matches the file the asset was read from. That splits "RAM wrong at source" from
   "device copy diverged".

## Other open defect classes

- **Indirect-dispatch device loss** (pipelines `0x35bd42f41aa8c5bf` and `0xd18f624bd90339c5`,
  twelve of the eighteen losses recorded through run 75). Both shaders are GDS consume loops that spin until a
  producer publishes work. The emulator exposes one Vulkan queue, so producer and consumer are
  serialized and the consumer can spin forever. The rewind fix and the dispatch guard did not
  change it. The real fix is a second queue for async compute; not designed yet.
  Run 81 (`review-fixes-2`, 2026-10-09) died this way after 25 minutes, on `0x35bd…`, the first
  such loss since run 75. Build `loop-limit` bounds every compute-shader loop to
  `computeLoopIterationLimit` iterations per entry (default 65,536, 0 disables), so a consumer
  that spins gives up instead of hanging the device. It is a watchdog, not the fix: the work
  that loop was waiting for is skipped for that dispatch. Nothing reports when the bound ends a
  loop; the only evidence it works is this loss class not recurring.
- **Indirect-draw device loss** (`0x406ddce8249ffb87`, `0x2ffc129cafdb142b`): three occurrences,
  different signature, not investigated.
- **CPU-side freeze** (run 76 session 2): zero CPU, no device loss, log cut mid-line inside the
  garbage collector. One occurrence. `ServiceFaultRange` holds the texture-cache mutex across a
  full `scheduler.Finish()`, which runs deferred operations; that is the first place to look
  if it recurs.
