# Dream Window Performance Debug & Optimization Plan

**Goal:** Diagnose why async system is 2x slower than expected (4.5s vs 2.5s per cycle) with low GPU/CPU utilization (20-40% instead of 60-80%).

**Key Symptoms:**
- ComfyUI visually stutters during async operation
- GPU utilization: 20-40% (should be 60-80%)
- CPU utilization: 40% (should be higher)
- Both diffusion AND interpolation taking 4s each (should run in parallel ~2.5s)
- Low utilization in BOTH sync and async systems

---

## Phase 1: Add Comprehensive Profiling

### 1.1 VAE Lock Monitoring

**File:** `backend/core/async_orchestrator.py`

**Location:** In `_update_buffer_status_loop()` method, after existing status updates (~line 450)

```python
async def _update_buffer_status_loop(self) -> None:
    """Periodically update status.json with buffer information"""
    while self.running:
        try:
            # ... existing status code ...
            
            # === VAE LOCK STATS ===
            if self.vae_access:
                lock_stats = self.vae_access.get_lock_stats()
                
                # Log if significant contention
                if lock_stats['avg_wait_time_ms'] > 10:
                    logger.warning(
                        f"VAE Lock Contention: {lock_stats['acquisitions']} ops, "
                        f"avg wait: {lock_stats['avg_wait_time_ms']:.1f}ms, "
                        f"max wait: {lock_stats['max_wait_time_ms']:.1f}ms"
                    )
                else:
                    logger.debug(
                        f"VAE Lock: {lock_stats['acquisitions']} ops, "
                        f"avg wait: {lock_stats['avg_wait_time_ms']:.1f}ms"
                    )
                
                # Add to status.json
                status_data["vae_lock_acquisitions"] = lock_stats["acquisitions"]
                status_data["vae_lock_avg_wait_ms"] = lock_stats["avg_wait_time_ms"]
                status_data["vae_lock_max_wait_ms"] = lock_stats["max_wait_time_ms"]
                
                # Reset stats every 10 seconds for moving average
                if int(time.time()) % 10 == 0:
                    self.vae_access.reset_stats()
            
            self.status_writer.write_status(status_data)
            await asyncio.sleep(1.0)
```

**What to look for:**
- `avg_wait_time_ms < 50ms`: Lock contention is NOT the problem
- `avg_wait_time_ms > 100ms`: Lock contention is significant (see Fix 1)
- `max_wait_time_ms > 500ms`: Severe lock starvation (see Fix 1 + Fix 2)

---

### 1.2 Detailed Interpolation Timing

**File:** `backend/core/workers/interpolation_worker.py`

**Location:** Replace `_generate_interpolations()` method with instrumented version

```python
async def _generate_interpolations(
    self,
    start_kf_num: int,
    end_kf_num: int,
    start_latent: torch.Tensor,
    end_latent: torch.Tensor,
    interp_sequence_nums: List[int]
) -> bool:
    """Generate interpolation frames with detailed timing"""
    import time
    
    count = len(interp_sequence_nums)
    
    # Timing breakdown
    timings = {
        'slerp_precompute': 0,
        'slerp_per_frame': [],
        'decode_per_frame': [],
        'save_per_frame': [],
        'overhead_per_frame': [],
        'total': 0
    }
    
    cycle_start = time.perf_counter()
    
    # Precompute slerp parameters
    from interpolation.spherical_lerp import precompute_slerp_params, spherical_lerp
    
    pair_key = (start_kf_num, end_kf_num)
    if pair_key not in self.slerp_precomputed:
        t_precompute = time.perf_counter()
        loop = asyncio.get_event_loop()
        self.slerp_precomputed[pair_key] = await loop.run_in_executor(
            None,
            precompute_slerp_params,
            start_latent,
            end_latent
        )
        timings['slerp_precompute'] = time.perf_counter() - t_precompute
    
    # Generate each frame
    success_count = 0
    
    for i, sequence_num in enumerate(interp_sequence_nums, start=1):
        try:
            frame_start = time.perf_counter()
            
            self.frame_buffer.mark_generating(sequence_num)
            frame_spec = self.frame_buffer.frames[sequence_num]
            t = frame_spec.interpolation_t
            
            # SLERP
            t_slerp = time.perf_counter()
            loop = asyncio.get_event_loop()
            interpolated_latent = await loop.run_in_executor(
                None,
                spherical_lerp,
                start_latent,
                end_latent,
                t,
                1e-6,
                self.slerp_precomputed[pair_key]
            )
            timings['slerp_per_frame'].append(time.perf_counter() - t_slerp)
            
            # DECODE (VAE with lock)
            t_decode = time.perf_counter()
            interpolated_image = await self.vae_access.decode_async(
                interpolated_latent,
                upscale_to_target=True
            )
            timings['decode_per_frame'].append(time.perf_counter() - t_decode)
            
            # SAVE (sync I/O - BLOCKING!)
            t_save = time.perf_counter()
            output_path = frame_spec.file_path
            interpolated_image.save(output_path, "PNG", optimize=False, compress_level=1)
            timings['save_per_frame'].append(time.perf_counter() - t_save)
            
            # Mark ready
            self.frame_buffer.mark_ready(sequence_num, output_path)
            success_count += 1
            self.frames_generated += 1
            
            # Overhead (everything else)
            frame_total = time.perf_counter() - frame_start
            frame_work = timings['slerp_per_frame'][-1] + timings['decode_per_frame'][-1] + timings['save_per_frame'][-1]
            timings['overhead_per_frame'].append(frame_total - frame_work)
            
        except Exception as e:
            logger.error(f"Failed to generate interpolation {i}: {e}", exc_info=True)
    
    timings['total'] = time.perf_counter() - cycle_start
    
    # Calculate averages
    avg_slerp = sum(timings['slerp_per_frame']) / len(timings['slerp_per_frame']) if timings['slerp_per_frame'] else 0
    avg_decode = sum(timings['decode_per_frame']) / len(timings['decode_per_frame']) if timings['decode_per_frame'] else 0
    avg_save = sum(timings['save_per_frame']) / len(timings['save_per_frame']) if timings['save_per_frame'] else 0
    avg_overhead = sum(timings['overhead_per_frame']) / len(timings['overhead_per_frame']) if timings['overhead_per_frame'] else 0
    
    total_work = sum(timings['slerp_per_frame']) + sum(timings['decode_per_frame']) + sum(timings['save_per_frame'])
    unaccounted = timings['total'] - total_work - timings['slerp_precompute']
    
    # Log detailed breakdown
    logger.info(f"[TIMING] Interpolation {start_kf_num}->{end_kf_num} breakdown:")
    logger.info(f"  Total time:        {timings['total']:.3f}s")
    logger.info(f"  Slerp precompute:  {timings['slerp_precompute']:.3f}s")
    logger.info(f"  Avg per frame:")
    logger.info(f"    - Slerp:         {avg_slerp*1000:.1f}ms")
    logger.info(f"    - Decode (VAE):  {avg_decode*1000:.1f}ms")
    logger.info(f"    - Save (I/O):    {avg_save*1000:.1f}ms")
    logger.info(f"    - Overhead:      {avg_overhead*1000:.1f}ms")
    logger.info(f"  Unaccounted time:  {unaccounted:.3f}s ({unaccounted/timings['total']*100:.1f}%)")
    
    # Cache midpoint (existing code)
    if success_count > 0 and hasattr(self, 'cache_worker'):
        # ... existing cache code ...
        pass
    
    return success_count == count
```

**What to look for:**
- `avg_decode > 150ms`: VAE is slow or contended (see Fix 1, Fix 2)
- `avg_save > 30ms`: I/O blocking is significant (see Fix 3)
- `avg_overhead > 50ms`: Executor/async overhead (see Fix 4)
- `unaccounted > 1s`: Something else is blocking (see Fix 5)

---

### 1.3 Executor Queue Monitoring

**File:** `backend/core/workers/interpolation_worker.py`

**Location:** In `run()` method, after getting a pair from queue (~line 322)

```python
async def run(self) -> None:
    """Main worker loop with executor monitoring"""
    self.running = True
    logger.info("InterpolationWorker started")
    
    while self.running:
        try:
            # Get next pair
            try:
                pair = await asyncio.wait_for(self.pair_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            
            # === MONITOR EXECUTOR ===
            loop = asyncio.get_event_loop()
            executor = loop._default_executor
            
            if executor:
                # Check if executor has work queue
                if hasattr(executor, '_work_queue'):
                    queue_depth = executor._work_queue.qsize()
                    if queue_depth > 5:
                        logger.warning(f"Executor queue depth high: {queue_depth} pending tasks")
                    else:
                        logger.debug(f"Executor queue: {queue_depth} tasks")
            
            # ... rest of existing code ...
```

**What to look for:**
- Queue depth consistently > 10: Executor is bottleneck (see Fix 4)
- Queue depth = 0: Executor is not the problem

---

### 1.4 CUDA Context Detection

**File:** `backend/core/async_orchestrator.py`

**Location:** In `__init__()` method, after initializing workers

```python
def __init__(self, ...):
    # ... existing init code ...
    
    # === CHECK CUDA CONTEXTS ===
    if torch.cuda.is_available():
        logger.info("=== CUDA Context Info ===")
        logger.info(f"  Current device: {torch.cuda.current_device()}")
        logger.info(f"  Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA streams: {torch.cuda.current_stream()}")
        
        # Check if multiple contexts exist (can cause issues)
        try:
            # This will show if contexts are being recreated
            ctx_handle = torch.cuda.current_context()
            logger.info(f"  Context handle: {ctx_handle}")
        except Exception as e:
            logger.debug(f"  Context check: {e}")
        
        logger.info("=========================")
```

---

## Phase 2: Analyze Results & Apply Fixes

### Fix 1: Reduce VAE Lock Contention

**If:** `avg_wait_time_ms > 100ms` OR `avg_decode > 200ms`

**Solution:** Batch all VAE decode operations to acquire lock once

**File:** `backend/core/workers/interpolation_worker.py`

**Replace `_generate_interpolations()` with batched version:**

```python
async def _generate_interpolations_batched(
    self,
    start_kf_num: int,
    end_kf_num: int,
    start_latent: torch.Tensor,
    end_latent: torch.Tensor,
    interp_sequence_nums: List[int]
) -> bool:
    """Generate interpolations with batched VAE operations"""
    from interpolation.spherical_lerp import precompute_slerp_params, spherical_lerp
    
    count = len(interp_sequence_nums)
    
    # Precompute slerp
    pair_key = (start_kf_num, end_kf_num)
    if pair_key not in self.slerp_precomputed:
        loop = asyncio.get_event_loop()
        self.slerp_precomputed[pair_key] = await loop.run_in_executor(
            None, precompute_slerp_params, start_latent, end_latent
        )
    
    # === PHASE 1: Generate all latents (no VAE lock needed) ===
    latents_and_specs = []
    
    for i, sequence_num in enumerate(interp_sequence_nums, start=1):
        frame_spec = self.frame_buffer.frames[sequence_num]
        t = frame_spec.interpolation_t
        
        # Slerp in executor (CPU-bound, no lock)
        loop = asyncio.get_event_loop()
        interpolated_latent = await loop.run_in_executor(
            None,
            spherical_lerp,
            start_latent, end_latent, t, 1e-6,
            self.slerp_precomputed[pair_key]
        )
        
        latents_and_specs.append((interpolated_latent, frame_spec, sequence_num))
    
    # === PHASE 2: Decode ALL frames in one VAE lock acquisition ===
    logger.info(f"  Decoding {count} frames in batch...")
    
    decoded_images = []
    decode_start = time.time()
    
    for latent, frame_spec, sequence_num in latents_and_specs:
        # Each decode acquires lock individually, but they happen back-to-back
        # without ComfyUI interrupting
        image = await self.vae_access.decode_async(latent, upscale_to_target=True)
        decoded_images.append((image, frame_spec, sequence_num))
    
    decode_time = time.time() - decode_start
    logger.info(f"  Batch decode: {decode_time:.2f}s ({decode_time/count:.3f}s per frame)")
    
    # === PHASE 3: Save all frames (async I/O) ===
    save_tasks = []
    loop = asyncio.get_event_loop()
    
    for image, frame_spec, sequence_num in decoded_images:
        # Save async to avoid blocking
        save_task = loop.run_in_executor(
            None,
            image.save,
            str(frame_spec.file_path),
            "PNG"
        )
        save_tasks.append((save_task, frame_spec, sequence_num))
    
    # Wait for all saves to complete
    success_count = 0
    for save_task, frame_spec, sequence_num in save_tasks:
        try:
            await save_task
            self.frame_buffer.mark_ready(sequence_num, frame_spec.file_path)
            success_count += 1
            self.frames_generated += 1
        except Exception as e:
            logger.error(f"Failed to save frame {sequence_num}: {e}")
    
    return success_count == count
```

**Expected improvement:** 30-50% speedup if lock contention was the issue

---

### Fix 2: Use Dedicated Executor for VAE Operations

**If:** Queue depth > 10 consistently OR executor overhead > 100ms

**Solution:** Create separate executor for VAE to prevent queue saturation

**File:** `backend/core/shared_resources.py`

**Add to `SharedVAEAccess.__init__()`:**

```python
class SharedVAEAccess:
    def __init__(self, latent_encoder):
        self.encoder = latent_encoder
        self.lock = asyncio.Lock()
        
        # === DEDICATED EXECUTOR FOR VAE ===
        # Prevents VAE ops from queueing behind other executor work
        import concurrent.futures
        self.vae_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # VAE operations are serial anyway due to lock
            thread_name_prefix="vae_worker"
        )
        
        logger.info("SharedVAEAccess initialized with dedicated executor")
    
    async def encode_async(self, image: Path, for_interpolation: bool = False):
        async with self.lock:
            # Use dedicated executor instead of default
            loop = asyncio.get_event_loop()
            latent = await loop.run_in_executor(
                self.vae_executor,  # Use dedicated executor
                self.encoder.encode,
                image,
                for_interpolation
            )
            return latent
    
    async def decode_async(self, latent, upscale_to_target: bool = False):
        async with self.lock:
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                self.vae_executor,  # Use dedicated executor
                self.encoder.decode,
                latent,
                upscale_to_target
            )
            return image
```

**Expected improvement:** 20-30% speedup if executor queue was saturated

---

### Fix 3: Async Image Saving

**If:** `avg_save > 30ms`

**Solution:** Already implemented in Fix 1 (batched version). If using non-batched, apply separately:

**File:** `backend/core/workers/interpolation_worker.py`

**In `_generate_interpolations()` around line 264:**

```python
# BEFORE:
interpolated_image.save(output_path, "PNG", optimize=False, compress_level=1)

# AFTER:
loop = asyncio.get_event_loop()
await loop.run_in_executor(
    None,
    lambda: interpolated_image.save(
        str(output_path), "PNG", optimize=False, compress_level=1
    )
)
```

**Expected improvement:** 10-20% speedup, prevents event loop blocking

---

### Fix 4: Async Display I/O

**If:** Display is blocking (separate issue but easy win)

**File:** `backend/core/display_selector.py`

**In `select_and_display_next_frame()` around line 139:**

```python
async def select_and_display_next_frame(self) -> bool:
    """Get next frame and display it with async I/O"""
    frame_spec = self.buffer.get_next_display_frame()
    
    if frame_spec is None or not frame_spec.file_path.exists():
        return False
    
    try:
        # Make display write async
        await self._write_current_frame_async(frame_spec.file_path)
        
        self.buffer.mark_displayed(frame_spec.sequence_num)
        self.buffer.advance_display()
        self.frames_displayed += 1
        
        # ... logging ...
        return True
        
    except Exception as e:
        logger.error(f"Error displaying frame: {e}", exc_info=True)
        return False

async def _write_current_frame_async(self, frame_path: Path) -> None:
    """Write frame to current_frame.png using async I/O"""
    loop = asyncio.get_event_loop()
    
    # Run sync I/O in executor
    await loop.run_in_executor(
        None,
        self._write_current_frame_sync,
        frame_path
    )

def _write_current_frame_sync(self, frame_path: Path) -> None:
    """Sync I/O operations (runs in executor)"""
    import tempfile
    import shutil
    
    image = Image.open(frame_path)
    
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=self.output_dir,
        delete=False,
        suffix='.tmp',
        prefix='.current_frame_'
    ) as tmp_file:
        image.save(tmp_file, format='PNG', optimize=False)
        tmp_path = Path(tmp_file.name)
    
    shutil.move(str(tmp_path), str(self.current_frame_path))
```

**Expected improvement:** Display reaches 4 FPS (from 1.3 FPS)

---

### Fix 5: CUDA Stream Optimization

**If:** Unaccounted time > 1s AND context switching suspected

**Solution:** Use CUDA streams to overlap operations

**File:** `backend/interpolation/latent_encoder.py`

**Add stream support:**

```python
class LatentEncoder:
    def __init__(self, ...):
        # ... existing init ...
        
        # Create dedicated CUDA stream for VAE operations
        if self.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream()
            logger.info(f"Created dedicated CUDA stream for VAE")
        else:
            self.cuda_stream = None
    
    def encode(self, image, for_interpolation=False):
        """Encode with dedicated CUDA stream"""
        if self.cuda_stream:
            with torch.cuda.stream(self.cuda_stream):
                return self._encode_impl(image, for_interpolation)
        else:
            return self._encode_impl(image, for_interpolation)
    
    def decode(self, latent, upscale_to_target=False):
        """Decode with dedicated CUDA stream"""
        if self.cuda_stream:
            with torch.cuda.stream(self.cuda_stream):
                return self._decode_impl(latent, upscale_to_target)
        else:
            return self._decode_impl(latent, upscale_to_target)
    
    def _encode_impl(self, image, for_interpolation):
        # Existing encode logic here
        pass
    
    def _decode_impl(self, latent, upscale_to_target):
        # Existing decode logic here
        pass
```

**Expected improvement:** 10-20% if CUDA context switching was the issue

**Warning:** Only do this if ComfyUI and VAE are in same process/CUDA context!

---

## Phase 3: Nuclear Options (If Nothing Works)

### Option A: Reduce Interpolation Resolution

**If:** All fixes yield < 20% improvement AND GPU utilization stays low

**Diagnosis:** Your Maxwell Titan X GPUs may be memory-bandwidth limited

**Solution:** Use lower resolution for interpolation (already supported!)

**File:** `backend/config.yaml`

```yaml
generation:
  hybrid:
    # Reduce interpolation workload
    interpolation_resolution_divisor: 2  # Half resolution (256×128)
    interpolation_upscale_method: "bicubic"  # High quality upscale
    
    # OR go even lower for maximum speed
    interpolation_resolution_divisor: 3  # 1/3 resolution (170×85)
```

**Expected result:**
- 2x-3x faster interpolation
- Slight quality loss (but still good with proper upscaling)
- Reaches target 4 FPS

---

### Option B: Reduce Target FPS

**If:** Quality degradation from Option A is unacceptable

**Solution:** Accept 2-3 FPS instead of 4 FPS

**File:** `backend/config.yaml`

```yaml
generation:
  hybrid:
    target_interpolation_fps: 2.5  # Reduced from 4
    
display:
  buffer_target_seconds: 30.0  # Keep same buffer duration
  min_buffer_seconds: 10.0
```

**Expected result:**
- System can keep up with generation
- Buffer stays full
- Smoother overall experience (no buffer depletion)

---

### Option C: Disable Cache Analysis During Generation

**If:** CPU is the bottleneck (40% usage but pegged on one core)

**Solution:** Run cache analysis in background, not inline

**File:** `backend/core/async_orchestrator.py`

**In `_coordinate()` around line 475:**

```python
# BEFORE:
await self.cache_worker.submit_frame(...)

# AFTER:
# Fire and forget - don't await
asyncio.create_task(self.cache_worker.submit_frame(...))
```

**Expected improvement:** 5-10% reduction in CPU blocking

---

### Option D: Run ComfyUI on Different GPU

**If:** You have dual GPUs AND context switching is proven issue

**Solution:** Dedicate GPU 0 to ComfyUI, GPU 1 to VAE

**File:** `backend/config.yaml`

```yaml
system:
  gpu_id: 1  # VAE runs on GPU 1
  comfyui_url: "http://127.0.0.1:8188"  # ComfyUI runs on GPU 0
```

**ComfyUI startup:**
```bash
# Force ComfyUI to GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py
```

**Expected result:**
- Zero CUDA context switching
- Both GPUs utilized
- Near-linear speedup (2x if GPUs are equal)

---

## Phase 4: LAST CALL - Accept Hardware Limits

### When to Give Up

If after implementing Fixes 1-5 you still see:
- GPU utilization < 50%
- Total cycle time > 4s
- Unaccounted time > 30%
- No improvement from any optimization

**Then it's likely your Maxwell Titan X architecture is the bottleneck:**

1. **Memory bandwidth:** Maxwell has 336 GB/s vs newer cards at 900+ GB/s
2. **Tensor cores:** Maxwell has none (all compute is CUDA cores)
3. **PCIe bandwidth:** Older PCIe 3.0 limits CPU↔GPU transfers
4. **Driver overhead:** Maxwell drivers have more overhead in context switches

### Recommended Fallback Configuration

**File:** `backend/config.yaml`

```yaml
generation:
  # Use async (still better than sync for buffer management)
  use_async_orchestrator: true
  
  hybrid:
    # Reduce interpolation workload significantly
    interpolation_frames: 8  # Down from 10
    interpolation_resolution_divisor: 3  # 1/3 resolution
    interpolation_upscale_method: "bicubic"
    
    # Reduce target FPS to achievable level
    target_interpolation_fps: 2.5  # Down from 4
    
    keyframe_denoise: 0.2  # Keep same
    
display:
  buffer_target_seconds: 20.0  # Smaller buffer = faster startup
  min_buffer_seconds: 8.0
  
performance:
  generation_timeout: 120  # Longer timeout for slower system
```

**Expected performance:**
- Generation: ~3s per cycle (8 frames at 1/3 res + 1 keyframe)
- Display: 2.5 FPS
- Buffer: Stays full, smooth playback
- Quality: Still visually good with bicubic upscaling

### Visual Quality Preservation

Even at 1/3 resolution with 8 frames, you can maintain quality by:

1. **Using bicubic upscaling** (already configured above)
2. **Increasing interpolation weight** for smoother motion:

```yaml
generation:
  hybrid:
    interpolation_blend_bias: 0.6  # Favor interpolation over endpoints
```

3. **Tuning cache for quality frames**:

```yaml
generation:
  cache:
    population_mode: "selective"  # Only cache high-quality diverse frames
    diversity_threshold: 0.15  # Higher = more selective
```

---

## Summary Decision Tree

```
Run Phase 1 profiling
    │
    ├─→ avg_decode > 200ms? 
    │   └─→ Apply Fix 1 (Batched VAE)
    │
    ├─→ executor queue > 10?
    │   └─→ Apply Fix 2 (Dedicated Executor)
    │
    ├─→ avg_save > 30ms?
    │   └─→ Apply Fix 3 (Async I/O)
    │
    ├─→ unaccounted > 1s?
    │   └─→ Apply Fix 5 (CUDA Streams)
    │
    ├─→ Still slow after all fixes?
    │   ├─→ Try Option A (Lower Resolution)
    │   ├─→ Try Option B (Lower FPS)
    │   ├─→ Try Option C (Disable Inline Cache)
    │   └─→ Try Option D (Dual GPU)
    │
    └─→ STILL slow?
        └─→ Accept hardware limits
            └─→ Use fallback config (Phase 4)
```

---

## Testing Protocol

For each fix:

1. **Baseline measurement:**
   - Run for 5 minutes
   - Record avg cycle time
   - Record GPU/CPU utilization (via `nvidia-smi dmon` and Task Manager)
   - Record buffer status (min/max/avg)

2. **Apply fix**

3. **Measure again:**
   - Same duration, same metrics
   - Compare to baseline
   - > 15% improvement = keep
   - < 15% improvement = revert, try next fix

4. **Log results** in a performance journal:
   ```
   Baseline: 4.5s cycle, 30% GPU, 40% CPU
   Fix 1 (Batched VAE): 3.2s cycle, 45% GPU, 40% CPU [KEEP - 29% faster]
   Fix 2 (Executor): 3.1s cycle, 45% GPU, 40% CPU [MARGINAL - only 3% gain]
   ```

---

## Expected Best Case Results

If ALL optimizations work perfectly:
- **Cycle time:** 2.5-3.0s (down from 4.5s)
- **GPU utilization:** 50-60%
- **CPU utilization:** 50-60%
- **Display:** 4 FPS (smooth)
- **Buffer:** Stays at 30s (never depletes)

If hardware is limiting:
- **Cycle time:** 3.5-4.0s (with lower res interpolation)
- **GPU utilization:** 40-50%
- **Display:** 2.5-3 FPS (still smooth)
- **Buffer:** Stays at 20s (stable)

Either way, you'll have a smooth, stable system - just a question of whether you can hit the original 4 FPS target or need to adjust expectations to hardware reality.

---

**Good luck debugging! Start with Phase 1 profiling and report back what you find.**