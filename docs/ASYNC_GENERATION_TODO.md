# Async/Parallel Generation Architecture - TODO

## 🎯 **CRITICAL**: Current Implementation is Synchronous!

The current hybrid generator **blocks during interpolation** to generate the next keyframe. This defeats the entire purpose of interpolation smoothing!

## 📊 Current Flow (WRONG)

```
Frame 0:  Generate keyframe A (2.1s) ──────────┐
                                              WAIT
Frame 1:  Need to interpolate A→B             ↓
          But B doesn't exist yet!             │
          → Generate B NOW (2.1s) ─────────────┘
          → Then interpolate frame 1 (0.05s)
          
Frame 2:  Interpolate A→B (0.05s) ← fast
Frame 3:  Interpolate A→B (0.05s) ← fast
...
Frame 10: Interpolate A→B (0.05s) ← fast

Frame 11: Need keyframe C
          → Generate C NOW (2.1s) ─────────────┐
                                              WAIT
```

**Problem**: We **pause interpolation** to generate the next keyframe!

**Result**: Stuttering visuals, defeats the purpose of smooth interpolation

## ✅ Intended Flow (CORRECT)

```
Frame 0:  Generate keyframe A (2.1s) ────┐
                                        DONE
                                         ↓
          Queue generation of B in background ─────────┐
                                                      ASYNC
Frame 1:  Interpolate A→? (0.05s) ← smooth            ↓
Frame 2:  Interpolate A→? (0.05s) ← smooth            ↓
Frame 3:  Interpolate A→? (0.05s) ← smooth            ↓
...                                                    ↓
Frame 10: Interpolate A→? (0.05s) ← smooth            ↓
                                                       ↓
Frame 11: Check: Is B ready? ────────────────────────┘
          YES! → Interpolate A→B (0.05s)
          Queue generation of C in background ────────┐
                                                     ASYNC
Frame 12: Interpolate A→B (0.05s) ← smooth            ↓
...
```

**Key insight**: Generation happens **in parallel with interpolation**

**Result**: Smooth continuous output, no stuttering

## 🔧 What Needs to Change

### 1. **Async Generation Queue** (ComfyUI API already supports this!)

ComfyUI has a queue system. We need to:
- Queue prompt without waiting
- Store the `prompt_id` as a "future"
- Continue interpolating
- Poll for completion asynchronously

```python
class AsyncKeyframeGenerator:
    def __init__(self, generator, frame_manager):
        self.generator = generator
        self.frame_manager = frame_manager
        self.pending_keyframes = {}  # frame_num -> (prompt_id, prompt, start_time)
    
    def queue_keyframe(self, frame_num, image_path, prompt, denoise):
        """Queue keyframe generation without waiting"""
        # Submit to ComfyUI
        prompt_id = self.generator.client.queue_prompt(workflow)
        
        # Store as pending
        self.pending_keyframes[frame_num] = {
            'prompt_id': prompt_id,
            'prompt': prompt,
            'queued_at': time.time(),
            'state': 'queued'
        }
        
        logger.info(f"Queued keyframe {frame_num} (prompt_id: {prompt_id})")
    
    def check_keyframe_ready(self, frame_num) -> bool:
        """Check if queued keyframe is ready"""
        if frame_num not in self.pending_keyframes:
            return False
        
        pending = self.pending_keyframes[frame_num]
        prompt_id = pending['prompt_id']
        
        # Check ComfyUI queue
        queue = self.generator.client.get_queue()
        running = any(item[1] == prompt_id for item in queue.get("queue_running", []))
        pending_queue = any(item[1] == prompt_id for item in queue.get("queue_pending", []))
        
        if not running and not pending_queue:
            # Generation complete! Fetch result
            output_files = self.generator.client.get_output_images(prompt_id)
            if output_files:
                # Download and store
                image_data = self.generator.client.get_image_data(output_files[0])
                output_path = self.generator.output_dir / f"frame_{frame_num:05d}.png"
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                # Mark ready in frame manager
                self.frame_manager.mark_ready(frame_num, output_path, pending['prompt'])
                
                # Remove from pending
                del self.pending_keyframes[frame_num]
                
                logger.info(f"Keyframe {frame_num} ready! (took {time.time() - pending['queued_at']:.1f}s)")
                return True
        
        return False
    
    def get_keyframe_or_wait(self, frame_num, timeout=60):
        """Wait for keyframe with timeout"""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_keyframe_ready(frame_num):
                return True
            time.sleep(0.1)  # Poll every 100ms
        return False
```

### 2. **Predictive Keyframe Scheduling**

Need to queue the NEXT keyframe as soon as CURRENT keyframe is ready:

```python
def generate_next_frame(self, current_image, prompt, frame_number, denoise):
    frame_spec = self.frame_manager.get_or_create_frame_spec(frame_number)
    
    if frame_spec.is_keyframe():
        # This is a keyframe - wait for it or generate it
        if not self.async_generator.check_keyframe_ready(frame_number):
            # Not ready yet, wait
            logger.info(f"Waiting for queued keyframe {frame_number}...")
            self.async_generator.get_keyframe_or_wait(frame_number)
        
        # Get the keyframe
        keyframe_path = self.frame_manager.frames[frame_number].output_path
        
        # Encode it
        latent = self.latent_encoder.encode(keyframe_path)
        self.frame_manager.store_keyframe_data(frame_number, latent, keyframe_path)
        
        # IMMEDIATELY queue the NEXT keyframe
        next_keyframe_num = frame_number + (self.interpolation_frames + 1)
        self.async_generator.queue_keyframe(
            frame_num=next_keyframe_num,
            image_path=keyframe_path,
            prompt=prompt,
            denoise=denoise
        )
        logger.info(f"→ Queued next keyframe {next_keyframe_num} in background")
        
        return keyframe_path
    
    else:
        # Interpolated frame - just interpolate
        # The next keyframe should already be generating in the background!
        ...
```

### 3. **Frame Buffer Management**

Need to handle cases where generation is slower than interpolation:

```python
# If we're on frame 10 (last interpolation frame)
# and next keyframe (frame 11) isn't ready yet:

if frame_number == last_interpolation_frame:
    next_keyframe_num = frame_number + 1
    
    if not self.frame_manager.frames[next_keyframe_num].is_ready():
        # Next keyframe not ready - we need to STALL
        logger.warning(f"Buffer underrun! Waiting for keyframe {next_keyframe_num}...")
        
        # Wait for it (this is the "backup" - should rarely happen)
        self.async_generator.get_keyframe_or_wait(next_keyframe_num, timeout=30)
        
        if not self.frame_manager.frames[next_keyframe_num].is_ready():
            # Still not ready - fallback to synchronous generation
            logger.error(f"Keyframe {next_keyframe_num} timed out! Generating synchronously...")
            # ... synchronous fallback ...
```

### 4. **Interpolation Strategy Update**

Current interpolation needs both keyframes. We need to adapt:

**Option A: Optimistic Interpolation** (use temporary endpoint)
```python
# On frame 1, we don't have keyframe B yet
# Interpolate A→A (just use A) until B is ready
# Then switch to A→B once B arrives

if self.frame_manager.has_keyframe(end_keyframe):
    # Normal interpolation
    interpolate(keyframe_A, keyframe_B, t)
else:
    # Temporary: just use keyframe_A (will be replaced once B is ready)
    return keyframe_A  # or interpolate A→A with slight noise
```

**Option B: Wait for First Interpolation** (simpler)
```python
# On frame 1 (first interpolation), wait for next keyframe
if frame_in_sequence == 1:
    # Ensure next keyframe is ready before starting interpolation sequence
    if not self.frame_manager.has_keyframe(end_keyframe):
        logger.info(f"Waiting for keyframe {end_keyframe} before interpolation...")
        self.async_generator.get_keyframe_or_wait(end_keyframe)

# Now interpolate normally
interpolate(keyframe_A, keyframe_B, t)
```

**Recommendation**: Use Option B for simplicity. We only wait once at the start of each interpolation sequence.

## 📋 Implementation Checklist

- [ ] Create `AsyncKeyframeGenerator` class
  - [ ] `queue_keyframe()` - submit to ComfyUI without waiting
  - [ ] `check_keyframe_ready()` - poll queue status
  - [ ] `get_keyframe_or_wait()` - blocking wait with timeout
  - [ ] Track pending keyframes with `prompt_id`

- [ ] Update `HybridGenerator.generate_next_frame()`
  - [ ] On keyframe: check if already queued, wait if needed
  - [ ] On keyframe: immediately queue NEXT keyframe
  - [ ] On interpolation frame 1: ensure next keyframe ready before interpolating
  - [ ] On other interpolation frames: just interpolate (next keyframe generating in background)

- [ ] Update main loop
  - [ ] Initialize `AsyncKeyframeGenerator`
  - [ ] Queue frame 11 when frame 0 completes (bootstrap)
  - [ ] Monitor generation queue in status updates

- [ ] Add buffer monitoring
  - [ ] Track "frames ahead" (how many keyframes are queued)
  - [ ] Warn on buffer underrun
  - [ ] Fallback to synchronous generation if async fails

- [ ] Testing
  - [ ] Verify no stuttering during interpolation
  - [ ] Verify smooth transitions at keyframe boundaries
  - [ ] Test with slow generation (buffer underrun scenarios)
  - [ ] Test with fast generation (multiple queued keyframes)

## 🎯 Success Criteria

After implementation, the system should:

1. ✅ **Never block during interpolation** (except on frame 1 of each sequence, once)
2. ✅ **Always have next keyframe ready** when interpolation sequence ends
3. ✅ **Generate continuously** (ComfyUI always has 1-2 frames queued)
4. ✅ **Smooth visual output** (no stuttering, no pauses)
5. ✅ **Graceful degradation** (falls back to sync if async fails)

## 📊 Performance Comparison

### Current (Synchronous):
```
Keyframe:       ████████████████████ (2.1s) BLOCKS
Interpolation:  █ (0.05s) x 10 = 0.5s
Keyframe:       ████████████████████ (2.1s) BLOCKS
Interpolation:  █ (0.05s) x 10 = 0.5s

Total for 22 frames: 2.1 + 0.5 + 2.1 + 0.5 = 5.2s
Effective FPS: 22 / 5.2 = 4.2 FPS
```

### After (Async):
```
Keyframe A:     ████████████████████ (2.1s) ┐
                                            DONE
                                             ↓
Queue B:        (instant) ───────────────────────────────┐
Interpolation:  █████████████ (0.5s for 10 frames)      │
                                                    (B generating)
Keyframe B:     (already done! 2.1s elapsed)            │
Queue C:        (instant) ─────────────────────────────┐│
Interpolation:  █████████████ (0.5s for 10 frames)    │↓
                                                  (C generating)
...

Total for 22 frames: 2.1 + 0.5 + 0.5 = 3.1s
Effective FPS: 22 / 3.1 = 7.1 FPS  (71% faster!)
```

**The math**: If generation time < interpolation sequence time, we get **perfect pipelining**.

With 10 interpolation frames at ~0.05s each = 0.5s of interpolation
Generation time = 2.1s

Since 2.1s > 0.5s, we need ~4 interpolation sequences to cover one generation.
**Optimal config**: `interpolation_frames = 40` (2.0s of interpolation @ 4ms/frame)

Then generation and interpolation perfectly overlap! 🎯

## 🚨 Current Blocker

Before implementing this, we need to fix the **CUDA timeout error** that occurs when encoding after generation.

**Once that's fixed**, this async architecture can be implemented without worrying about CUDA state issues.

## 📝 Notes

- ComfyUI's queue system already supports this - we just need to use it properly
- The `torch.cuda.synchronize()` fixes won't affect async - they just ensure clean state
- This is a **major** architectural change but will dramatically improve visual smoothness
- Current stuttering is because we block during interpolation to generate next keyframe

---

**Priority**: Fix CUDA errors first → Then implement async generation → Smooth visuals! 🎨✨

