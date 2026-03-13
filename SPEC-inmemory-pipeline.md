# Spec: In-Memory Frame Pipeline

## Problem

The display selector runs at **~6 FPS** despite a target of **17 FPS** and a buffer of 1,000+ frames. The bottleneck is the disk round-trip:

```
Interpolation Worker                   Display Selector
─────────────────────                  ─────────────────
VAE decode → PIL Image                 Image.open(file_path)  ← BOTTLENECK
  │                                      │
  ├─ img.save(PNG, compress_level=1)     ├─ ~2-5ms per read on NVMe
  │  ~15-30ms per frame                  │  but BLOCKED waiting for batch
  │  20 frames → disk in burst           │  to be written first
  │                                      │
  └─ mark_ready(seq, file_path)          └─ push to H.264 encoder → VPS
```

**What actually happens at runtime:**
1. Interpolation worker VAE-decodes 20 frames in ~800ms burst
2. Saves 20 PNGs to disk in ~300-500ms burst
3. Marks all 20 as READY in FrameBuffer
4. Display selector wakes up, reads and pushes ~10 frames at ~15 FPS (burst)
5. **Stalls 2-3 seconds** waiting for the next interpolation batch to be saved
6. Repeat → effective rate: **~6 FPS**

The frames already exist as PIL Images in memory after VAE decode. Writing to disk and re-reading is pure waste in cloud mode where there's no local display.

## Solution

Add an in-memory image field to `FrameSpec`. The interpolation worker stores the PIL Image directly. The display selector reads from memory instead of disk. In cloud mode, skip disk writes entirely.

## Design

### FrameSpec gets an `image` field

```python
@dataclass
class FrameSpec:
    # ... existing fields ...
    file_path: Optional[Path] = None

    # In-memory image (cloud mode — skip disk round-trip)
    image: Optional[Image.Image] = None
```

When `image` is set, the display selector uses it directly. When `image` is None, it falls back to `Image.open(file_path)` (local/desktop mode still works).

### Interpolation worker: store images in memory

In `_process_interpolation_batch()` Phase 3, instead of saving to disk:

```python
# Current (disk):
save_task = loop.run_in_executor(
    None,
    lambda img=image, path=frame_spec.file_path: img.save(str(path), "PNG", ...)
)
await save_task
self.frame_buffer.mark_ready(sequence_num, frame_spec.file_path)

# New (cloud mode — in-memory):
if self.cloud_mode:
    self.frame_buffer.mark_ready(sequence_num, image=image)
else:
    # Desktop mode: still save to disk (for Rainmeter, local viewer)
    save_task = loop.run_in_executor(...)
    await save_task
    self.frame_buffer.mark_ready(sequence_num, frame_spec.file_path)
```

### FrameBuffer.mark_ready: accept image

```python
def mark_ready(
    self,
    sequence_num: int,
    file_path: Optional[Path] = None,
    image: Optional[Image.Image] = None,
) -> None:
    if sequence_num in self.frames:
        self.frames[sequence_num].state = FrameState.READY
        self.frames[sequence_num].generated_at = time.time()
        if file_path:
            self.frames[sequence_num].file_path = file_path
        if image is not None:
            self.frames[sequence_num].image = image
```

### Display selector: read from memory

In `select_and_display_next_frame()`:

```python
# Current:
image = await loop.run_in_executor(
    None,
    lambda: Image.open(frame_spec.file_path)
)

# New:
if frame_spec.image is not None:
    image = frame_spec.image
elif frame_spec.file_path and frame_spec.file_path.exists():
    image = await loop.run_in_executor(
        None,
        lambda: Image.open(frame_spec.file_path)
    )
else:
    logger.error(f"Frame {frame_spec.sequence_num}: no image and no file")
    return False
```

### Memory management: clear image after display

After the frame is pushed to the cloud callback and marked displayed, clear the PIL image to free memory:

```python
# In select_and_display_next_frame(), after callback + mark_displayed:
if frame_spec.image is not None:
    frame_spec.image = None  # Free PIL image memory
```

With 20 frames at 1024x512 RGB (~1.5MB each), peak memory is ~30MB for one interpolation batch. This is negligible on a B200 node with 192GB RAM.

### Keyframe worker: also in-memory

The keyframe worker currently:
1. Calls `generator.generate_from_image_async()` → returns a file path
2. Moves the file to `keyframe_dir`
3. Marks ready with file path

The `DirectSDBackend.generate_from_image_async()` returns a **file path**, not an image. We need to also return or cache the PIL image.

Looking at `direct_sd_backend.py`, the generation produces a PIL image internally, saves it to disk, and returns the path. For cloud mode, we can skip the save and return the image directly.

```python
# In DirectSDBackend.generate_from_image():
# Current: saves to output_path, returns output_path
# New option: returns (output_path, pil_image) or just pil_image in cloud mode

# Simplest approach: after generation, the image is in self._last_output_image
# Add a property to expose it
```

Actually, the simpler approach: the keyframe worker can just `Image.open()` the generated file immediately (it was JUST written, still in page cache) and store it. This avoids changing the generator interface:

```python
# In keyframe_worker, after moving to target_path:
if self.cloud_mode:
    kf_image = Image.open(target_path)
    self.frame_buffer.mark_ready(kf_sequence_num, image=kf_image)
else:
    self.frame_buffer.mark_ready(kf_sequence_num, file_path=target_path)
```

Since the keyframe was JUST written, the OS page cache guarantees instant read. And keyframes are only 1 per second, so the overhead is negligible.

### cloud_mode flag propagation

The `cloud_mode` flag needs to reach the interpolation worker and keyframe worker. Currently:

- `DreamController` knows `self.cloud_enabled`
- Workers are created in `DreamController.__init__` or via `AsyncGenerationOrchestrator`
- The orchestrator creates workers

Options:
1. Pass `cloud_mode` to worker constructors
2. Set it on the config dict (already has `cloud.enabled`)

The config dict already has `cloud.enabled: true`. Workers already receive the config. So they can check `self.config.get('cloud', {}).get('enabled', False)`.

### Cleanup changes

With in-memory frames in cloud mode:
- `cleanup_displayed_frames` still works — but instead of deleting files, it clears the `image` field
- No files to delete (they were never written)
- The display selector's `_delete_frame_async()` should no-op when there's no file

```python
# In display selector cleanup:
if self.cleanup_enabled:
    if frame_spec.file_path and frame_spec.file_path.exists():
        await self._delete_frame_async(frame_spec.file_path)
    # Always clear in-memory image
    frame_spec.image = None
```

## Files Changed

| File | Change |
|------|--------|
| `core/frame_buffer.py` | Add `image` field to `FrameSpec`; update `mark_ready()` |
| `core/workers/interpolation_worker.py` | Store images in memory instead of disk (cloud mode) |
| `core/workers/keyframe_worker.py` | Load and store keyframe image in memory (cloud mode) |
| `core/display_selector.py` | Read from `frame_spec.image` when available |

## Performance Impact

| Metric | Current (disk) | After (in-memory) |
|--------|---------------|-------------------|
| Interpolation save | ~300-500ms (20 PNGs) | **0ms** (skip disk) |
| Display read | ~2-5ms/frame + stall | **0ms** (already in memory) |
| Effective FPS | ~6 FPS (bursty) | **~17 FPS** (smooth) |
| Memory overhead | ~0 (disk) | ~30MB peak (20 × 1.5MB) |
| Disk I/O | ~40MB/s write + read | **Zero** (cloud mode) |

The burst-then-stall pattern disappears because frames are available in memory the instant VAE decode completes — no disk write delay, no disk read delay.

## Implementation Order

1. Add `image` field to `FrameSpec` + update `mark_ready()`
2. Modify interpolation worker to store images in-memory (cloud mode)
3. Modify display selector to read from memory
4. Modify keyframe worker to store image (cloud mode)
5. Test on B200 — verify 17 FPS sustained
6. Verify desktop mode (disk path) still works

## Risk

- **Memory**: 20 frames × 1.5MB = 30MB per batch. With 60s buffer target and 17 FPS, peak would be ~1,000 frames × 1.5MB = **1.5GB**. This is fine for B200 (192GB RAM) but would need attention on smaller machines. The existing `max_output_frames: 400` config limits this to 600MB.
- **Buffer overflow**: With faster display, the buffer drains faster. If generation can't keep up, we'll see underruns. But with 60s of buffer, there's plenty of headroom.
- **Keyframe latent cache**: The interpolation worker uses keyframe file paths to load latents for slerp. In cloud mode with no disk files, this needs the keyframe image to be available. Since keyframes are stored in-memory in the FrameSpec, the interpolation worker can read from there. But this may need a small interface change to `latent_encoder.encode()`.

Actually — the latent encoder operates on keyframe file paths:
```python
# interpolation_worker.py
start_latent = self.latent_encoder.encode(start_kf_path)
end_latent = self.latent_encoder.encode(end_kf_path)
```

In cloud mode, we'd need `encode()` to accept PIL images too, or keep the keyframe files on disk (they're only 1 per second, ~50KB JPEG each — negligible I/O).

**Recommendation**: Keep keyframe files on disk (they're tiny and needed for latent encoding). Only skip disk for the 20 interpolation frames per cycle, which are the bulk of the I/O.
