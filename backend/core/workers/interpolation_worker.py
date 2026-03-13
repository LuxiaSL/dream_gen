"""
Interpolation Worker - Async GPU VAE Operations

Handles VAE-based latent interpolation between keyframes, using SharedVAEAccess
to prevent CUDA conflicts with inline injection logic.

This worker is GPU bound (VAE encode/decode), so operations run in executor
to avoid blocking the event loop during the ~2s batch processing time.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import torch

from utils.vram_profiler import dump_vram_on_oom, is_oom_error
from utils.perf_stats import get_perf_stats
from core.frame_buffer import FrameState

logger = logging.getLogger(__name__)


class InterpolationWorker:
    """
    Async worker for VAE-based frame interpolation
    
    Responsibilities:
    - Maintain queue of keyframe pairs to interpolate
    - Perform VAE encode/decode via SharedVAEAccess (with lock)
    - Execute spherical lerp (slerp) in latent space
    - Output frames directly to FrameBuffer
    - Cache latents to avoid re-encoding keyframes
    
    Queue Flow:
        Coordinator -> submit_pair() -> pair_queue
        ↓
        run() loop processes pairs
        ↓
        Outputs directly to FrameBuffer (no result queue needed)
    
    Latent Caching:
        Keyframes are encoded once and cached for reuse across
        multiple interpolation pairs (e.g., KF2 is end of KF1->KF2
        and start of KF2->KF3).
    
    Usage:
        worker = InterpolationWorker(
            vae_access=shared_vae,
            frame_buffer=buffer,
            config=config
        )
        
        # Start worker loop
        asyncio.create_task(worker.run())
        
        # Submit pair for interpolation
        await worker.submit_pair(
            start_kf_num=1,
            end_kf_num=2,
            start_kf_path=Path("keyframe_001.png"),
            end_kf_path=Path("keyframe_002.png")
        )
    """
    
    def __init__(
        self,
        vae_access,  # SharedVAEAccess instance
        frame_buffer,  # FrameBuffer instance
        config: Dict[str, Any],
        max_queue_size: int = 10
    ):
        """
        Initialize interpolation worker
        
        Args:
            vae_access: SharedVAEAccess instance (thread-safe VAE wrapper)
            frame_buffer: FrameBuffer instance (for output)
            config: Configuration dictionary
            max_queue_size: Maximum pending pairs (backpressure control)
        """
        self.vae_access = vae_access
        self.frame_buffer = frame_buffer
        self.config = config
        
        # Queue
        self.pair_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # State
        self.running = False
        self.processing = False
        
        # Latent cache (avoid re-encoding keyframes)
        self.keyframe_latents: Dict[int, torch.Tensor] = {}
        self.keyframe_paths: Dict[int, Path] = {}
        
        # Slerp precomputed parameters cache
        self.slerp_precomputed: Dict[Tuple[int, int], Dict] = {}
        
        # GPU slerp mode: run spherical lerp on GPU instead of CPU thread pool
        # This reduces context switching overhead on cloud GPUs with headroom
        self.gpu_slerp = config.get('generation', {}).get('hybrid', {}).get('gpu_slerp', False)
        
        # Statistics
        self.pairs_processed = 0
        self.frames_generated = 0
        self.total_interpolation_time = 0.0
        
        logger.info(f"InterpolationWorker initialized (max queue: {max_queue_size}, gpu_slerp: {self.gpu_slerp})")
    
    async def submit_pair(
        self,
        start_kf_num: int,
        end_kf_num: int,
        start_kf_path: Path,
        end_kf_path: Path,
        interp_sequence_nums: List[int]
    ) -> None:
        """
        Submit a keyframe pair for interpolation
        
        Args:
            start_kf_num: Starting keyframe number
            end_kf_num: Ending keyframe number
            start_kf_path: Path to starting keyframe
            end_kf_path: Path to ending keyframe
            interp_sequence_nums: List of sequence numbers for interpolations (pre-registered!)
        """
        pair = {
            'start_kf_num': start_kf_num,
            'end_kf_num': end_kf_num,
            'start_kf_path': start_kf_path,
            'end_kf_path': end_kf_path,
            'interp_sequence_nums': interp_sequence_nums
        }
        
        await self.pair_queue.put(pair)
        
        logger.debug(
            f"Submitted interpolation pair: KF{start_kf_num}->KF{end_kf_num} "
            f"(queue depth: {self.pair_queue.qsize()})"
        )
    
    async def _encode_keyframe(
        self,
        kf_num: int,
        kf_path: Path
    ) -> Optional[torch.Tensor]:
        """
        Encode a keyframe to latent space (with caching)
        
        Args:
            kf_num: Keyframe number
            kf_path: Path to keyframe image
            
        Returns:
            Latent tensor (on GPU), or None on failure
        """
        # Check cache first
        if kf_num in self.keyframe_latents:
            logger.debug(f"Using cached latent for KF{kf_num}")
            return self.keyframe_latents[kf_num]
        
        # Encode via SharedVAEAccess (with lock)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latent = await self.vae_access.encode_async(
                kf_path,
                for_interpolation=True
            )
            
            # Cache for reuse
            self.keyframe_latents[kf_num] = latent
            self.keyframe_paths[kf_num] = kf_path
            
            logger.debug(f"Encoded and cached KF{kf_num}")
            
            return latent
            
        except Exception as e:
            logger.error(f"Failed to encode keyframe {kf_num}: {e}", exc_info=True)
            return None
    
    async def _generate_interpolations(
        self,
        start_kf_num: int,
        end_kf_num: int,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        interp_sequence_nums: List[int]
    ) -> bool:
        """
        Generate interpolation frames with BATCHED VAE operations (optimized)
        
        This version acquires the VAE lock ONCE per pair instead of per-frame,
        dramatically reducing lock contention and improving performance.
        
        Pipeline:
        1. Generate all latents (no lock needed)
        2. Decode ALL latents in one batch (single lock acquisition)
        3. Save all images async (no blocking)
        
        Args:
            start_kf_num: Starting keyframe number
            end_kf_num: Ending keyframe number
            start_latent: Starting latent tensor
            end_latent: Ending latent tensor
            interp_sequence_nums: Sequence numbers for interpolations (pre-registered by orchestrator!)
            
        Returns:
            True if successful
        """
        count = len(interp_sequence_nums)
        
        # === PROFILING: Detailed timing breakdown ===
        timings = {
            'slerp_precompute': 0,
            'slerp_all': 0,
            'decode_all': 0,
            'save_all': 0,
            'total': 0
        }
        
        cycle_start = time.perf_counter()
        
        # Precompute slerp parameters for this pair
        from interpolation.spherical_lerp import precompute_slerp_params, spherical_lerp
        
        pair_key = (start_kf_num, end_kf_num)
        if pair_key not in self.slerp_precomputed:
            t_precompute = time.perf_counter()
            if self.gpu_slerp:
                # GPU mode: precompute directly (tensors already on GPU)
                self.slerp_precomputed[pair_key] = precompute_slerp_params(
                    start_latent,
                    end_latent
                )
            else:
                # CPU mode: run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self.slerp_precomputed[pair_key] = await loop.run_in_executor(
                    None,
                    precompute_slerp_params,
                    start_latent,
                    end_latent
                )
            timings['slerp_precompute'] = time.perf_counter() - t_precompute
        
        # === PHASE 1: Generate ALL latents (no VAE lock needed) ===
        logger.debug(f"  Phase 1: Generating {count} latents{'(GPU)' if self.gpu_slerp else '(CPU)'}...")
        t_slerp_start = time.perf_counter()
        
        latents_and_specs = []
        loop = asyncio.get_event_loop()
        
        for i, sequence_num in enumerate(interp_sequence_nums, start=1):
            try:
                self.frame_buffer.mark_generating(sequence_num)
                frame_spec = self.frame_buffer.frames[sequence_num]
                t = frame_spec.interpolation_t
                
                if self.gpu_slerp:
                    # GPU mode: run slerp directly (tensors on GPU, fast)
                    interpolated_latent = spherical_lerp(
                        start_latent,
                        end_latent,
                        t,
                        1e-6,
                        self.slerp_precomputed[pair_key]
                    )
                else:
                    # CPU mode: run in executor to avoid blocking event loop
                    interpolated_latent = await loop.run_in_executor(
                        None,
                        spherical_lerp,
                        start_latent,
                        end_latent,
                        t,
                        1e-6,
                        self.slerp_precomputed[pair_key]
                    )
                
                latents_and_specs.append((interpolated_latent, frame_spec, sequence_num))
                
            except Exception as e:
                logger.error(f"Failed to generate latent {i}: {e}", exc_info=True)
        
        timings['slerp_all'] = time.perf_counter() - t_slerp_start
        
        # === PHASE 2: Decode ALL latents in TRUE BATCH (single GPU call!) ===
        logger.info(f"  Phase 2: Batch decoding {len(latents_and_specs)} frames...")
        t_decode_start = time.perf_counter()
        
        # Extract just the latents for batch decode
        latent_list = [latent for latent, _, _ in latents_and_specs]
        
        try:
            # Single GPU call to decode all latents at once!
            # In cloud mode, return numpy arrays (skip PIL — saves ~120ms per batch)
            cloud_mode = self.config.get('cloud', {}).get('enabled', False)
            images = await self.vae_access.decode_batch_async(
                latent_list, upscale_to_target=True, return_numpy=cloud_mode
            )
            
            # Pair images back with their frame specs
            decoded_images = []
            for i, (image, (_, frame_spec, sequence_num)) in enumerate(zip(images, latents_and_specs)):
                decoded_images.append((image, frame_spec, sequence_num))
            
            # Verify batch decode returned expected count
            if len(decoded_images) != len(latents_and_specs):
                logger.warning(
                    f"[DECODE MISMATCH] Expected {len(latents_and_specs)} images, "
                    f"got {len(decoded_images)} - batch decode may have failed partially"
                )
            else:
                logger.info(f"  Phase 2 complete: {len(decoded_images)} images decoded")
        except Exception as e:
            logger.error(f"Batch decode failed, falling back to sequential: {e}", exc_info=True)
            
            # Dump VRAM diagnostics if this is an OOM error
            if is_oom_error(e):
                dump_vram_on_oom(
                    context=f"Batch decode {start_kf_num}->{end_kf_num} ({count} frames)",
                    exception=e
                )
                # Try to free memory before sequential fallback
                torch.cuda.empty_cache()
            
            # Fallback to sequential decode with per-frame timing
            decoded_images = []
            decode_times = []
            for i, (latent, frame_spec, sequence_num) in enumerate(latents_and_specs):
                try:
                    t_single = time.perf_counter()
                    image = await self.vae_access.decode_async(latent, upscale_to_target=True)
                    decode_times.append(time.perf_counter() - t_single)
                    decoded_images.append((image, frame_spec, sequence_num))
                except Exception as e2:
                    logger.error(f"Failed to decode latent: {e2}", exc_info=True)
            
            # Log sequential decode performance
            if decode_times:
                avg_time = sum(decode_times) / len(decode_times) * 1000
                logger.info(f"[PERF] Sequential decode: {len(decode_times)} frames, avg {avg_time:.1f}ms/frame")
        
        timings['decode_all'] = time.perf_counter() - t_decode_start
        
        # === PHASE 3: Store frames (in-memory for cloud, disk for desktop) ===
        cloud_mode = self.config.get('cloud', {}).get('enabled', False)
        t_save_start = time.perf_counter()
        success_count = 0

        if cloud_mode:
            # Cloud mode: store PIL images directly in FrameBuffer (zero disk I/O)
            logger.info(f"  Phase 3: Storing {len(decoded_images)} frames in memory...")
            for image, frame_spec, sequence_num in decoded_images:
                try:
                    self.frame_buffer.mark_ready(sequence_num, image=image)
                    success_count += 1
                    self.frames_generated += 1
                except Exception as e:
                    logger.error(f"Failed to store frame {sequence_num}: {e}", exc_info=True)
                    self.frame_buffer.mark_failed(sequence_num)
        else:
            # Desktop mode: save PNGs to disk (for Rainmeter, local viewer)
            logger.info(f"  Phase 3: Saving {len(decoded_images)} frames to disk...")
            save_tasks = []
            for image, frame_spec, sequence_num in decoded_images:
                save_task = loop.run_in_executor(
                    None,
                    lambda img=image, path=frame_spec.file_path: img.save(
                        str(path), "PNG", optimize=False, compress_level=1
                    )
                )
                save_tasks.append((save_task, frame_spec, sequence_num))

            save_timeout = 30.0
            for i, (save_task, frame_spec, sequence_num) in enumerate(save_tasks):
                try:
                    await asyncio.wait_for(save_task, timeout=save_timeout)
                    self.frame_buffer.mark_ready(sequence_num, frame_spec.file_path)
                    success_count += 1
                    self.frames_generated += 1
                    if (i + 1) % 5 == 0 or i == len(save_tasks) - 1:
                        logger.debug(f"  Saved {i + 1}/{len(save_tasks)} frames")
                except asyncio.TimeoutError:
                    logger.error(
                        f"[SAVE TIMEOUT] Frame {sequence_num} ({frame_spec.file_path.name}) "
                        f"took >{save_timeout}s - disk I/O stall?"
                    )
                    self.frame_buffer.mark_failed(sequence_num)
                except Exception as e:
                    logger.error(f"Failed to save frame {sequence_num}: {e}", exc_info=True)
                    self.frame_buffer.mark_failed(sequence_num)

        timings['save_all'] = time.perf_counter() - t_save_start
        timings['total'] = time.perf_counter() - cycle_start
        
        # Report save status
        if success_count < count:
            logger.warning(
                f"[PARTIAL SAVE] Only {success_count}/{count} frames saved for "
                f"KF{start_kf_num}->KF{end_kf_num} - check for disk or timeout issues"
            )
            
            # Safety: Mark any frames still stuck in GENERATING as FAILED
            # This ensures display can skip them instead of freezing
            stuck_frames = 0
            for seq_num in interp_sequence_nums:
                if seq_num in self.frame_buffer.frames:
                    frame = self.frame_buffer.frames[seq_num]
                    if frame.state == FrameState.GENERATING:
                        self.frame_buffer.mark_failed(seq_num)
                        stuck_frames += 1
            
            if stuck_frames > 0:
                logger.warning(
                    f"[CLEANUP] Marked {stuck_frames} stuck GENERATING frames as FAILED "
                    f"for KF{start_kf_num}->KF{end_kf_num}"
                )
        else:
            logger.info(f"  Phase 3 complete: {success_count}/{count} frames saved")
        
        # === PROFILING: Calculate statistics ===
        avg_slerp = timings['slerp_all'] / count if count > 0 else 0
        avg_decode = timings['decode_all'] / count if count > 0 else 0
        avg_save = timings['save_all'] / count if count > 0 else 0
        
        # Log detailed breakdown
        logger.info(f"[TIMING] Interpolation {start_kf_num}->{end_kf_num} breakdown (BATCHED):")
        logger.info(f"  Total time:        {timings['total']:.3f}s")
        logger.info(f"  Slerp precompute:  {timings['slerp_precompute']:.3f}s")
        logger.info(f"  Phase timings:")
        logger.info(f"    - Slerp all:     {timings['slerp_all']:.3f}s ({avg_slerp*1000:.1f}ms per frame)")
        logger.info(f"    - Decode all:    {timings['decode_all']:.3f}s ({avg_decode*1000:.1f}ms per frame)")
        logger.info(f"    - Save all:      {timings['save_all']:.3f}s ({avg_save*1000:.1f}ms per frame)")
        logger.info(f"  [BATCHED] Single VAE lock acquisition for {count} frames")
        
        # Record to perf stats
        perf = get_perf_stats()
        perf.record_interpolation_batch(count, timings['total'])
        # Record per-frame decode time for bottleneck analysis
        if count > 0:
            perf.record_decode_time(timings['decode_all'] / count)
        
        # === CACHE INTERPOLATION MIDPOINT (for diversity) ===
        # Only cache the midpoint (t≈0.5) frame - naturally diverse transitions
        if success_count > 0 and hasattr(self, 'cache_worker'):
            try:
                # Find midpoint frame (t ≈ 0.5)
                midpoint_idx = count // 2
                midpoint_sequence = interp_sequence_nums[midpoint_idx]
                midpoint_frame = self.frame_buffer.frames[midpoint_sequence]
                
                logger.info(
                    f"[INTERP_CACHE] Submitting midpoint for cache analysis "
                    f"({start_kf_num} -> {end_kf_num} @ t={midpoint_frame.interpolation_t:.2f})"
                )
                
                # Submit to cache worker (will handle encoding & selective caching)
                # In cloud mode, midpoint may not have a file_path (in-memory only)
                if self.cache_worker and midpoint_frame.file_path:
                    await self.cache_worker.submit_frame(
                        frame_path=midpoint_frame.file_path,
                        prompt=f"interpolation_{start_kf_num}_{end_kf_num}_t0.5",
                        metadata={
                            'type': 'interpolation_midpoint',
                            'start_kf': start_kf_num,
                            'end_kf': end_kf_num,
                            't': midpoint_frame.interpolation_t
                        }
                    )
            except Exception as e:
                logger.debug(f"Interpolation midpoint cache submission failed: {e}")
        
        return success_count == count
    
    async def run(self) -> None:
        """
        Main worker loop
        
        Processes interpolation pairs from queue and outputs to FrameBuffer.
        Runs until stop() is called.
        """
        self.running = True
        logger.info("InterpolationWorker started")
        
        while self.running:
            try:
                # Get next pair (with timeout to allow checking running flag)
                try:
                    pair = await asyncio.wait_for(
                        self.pair_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No pair available, continue loop
                    continue
                
                self.processing = True
                
                # === PROFILING: Monitor executor queue depth ===
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
                
                # Extract pair data
                start_kf_num = pair['start_kf_num']
                end_kf_num = pair['end_kf_num']
                start_kf_path = pair['start_kf_path']
                end_kf_path = pair['end_kf_path']
                interp_sequence_nums = pair['interp_sequence_nums']
                
                # Guard against empty sequence list (can happen during recovery)
                if not interp_sequence_nums:
                    logger.warning(
                        f"Skipping interpolation pair KF{start_kf_num}->KF{end_kf_num}: "
                        f"no sequence numbers (possibly orphaned during recovery)"
                    )
                    continue
                
                logger.info(
                    f"Processing interpolation pair: KF{start_kf_num}->KF{end_kf_num} "
                    f"(seq {interp_sequence_nums[0]}-{interp_sequence_nums[-1]})"
                )
                
                start_time = time.time()
                
                try:
                    # Encode both keyframes (with caching)
                    start_latent = await self._encode_keyframe(start_kf_num, start_kf_path)
                    end_latent = await self._encode_keyframe(end_kf_num, end_kf_path)
                    
                    if start_latent is None or end_latent is None:
                        logger.error(
                            f"Failed to encode keyframes for pair "
                            f"{start_kf_num}->{end_kf_num}"
                        )
                        continue
                    
                    # Generate interpolations
                    success = await self._generate_interpolations(
                        start_kf_num,
                        end_kf_num,
                        start_latent,
                        end_latent,
                        interp_sequence_nums
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if success:
                        self.pairs_processed += 1
                        self.total_interpolation_time += elapsed
                        
                        count = self.config['generation']['hybrid']['interpolation_frames']
                        avg_per_frame = elapsed / count
                        
                        logger.info(
                            f"[OK] Interpolated {start_kf_num}->{end_kf_num} "
                            f"in {elapsed:.2f}s ({avg_per_frame:.3f}s/frame)"
                        )
                    else:
                        logger.error(
                            f"Interpolation pair {start_kf_num}->{end_kf_num} "
                            f"incomplete or failed"
                        )
                
                except Exception as e:
                    logger.error(
                        f"Error processing interpolation pair "
                        f"{start_kf_num}->{end_kf_num}: {e}",
                        exc_info=True
                    )
                
                finally:
                    # Mark task as done
                    self.pair_queue.task_done()
                    self.processing = False
                    
                    # Clean up old latents (keep last 5 keyframes)
                    self.cleanup_old_keyframes(keep_recent=5)
            
            except asyncio.CancelledError:
                logger.info("InterpolationWorker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in interpolation worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error
        
        logger.info("InterpolationWorker stopped")
    
    def cleanup_old_keyframes(self, keep_recent: int = 5) -> None:
        """
        Clean up old keyframe latents to save VRAM
        
        Args:
            keep_recent: Number of recent keyframes to keep
        """
        if len(self.keyframe_latents) <= keep_recent:
            return
        
        # Sort keyframes by number
        keyframe_numbers = sorted(self.keyframe_latents.keys())
        
        # Delete old ones
        to_delete = keyframe_numbers[:-keep_recent]
        
        for kf_num in to_delete:
            if kf_num in self.keyframe_latents:
                del self.keyframe_latents[kf_num]
            if kf_num in self.keyframe_paths:
                del self.keyframe_paths[kf_num]
            
            # Clean up slerp params involving this keyframe
            keys_to_delete = [
                k for k in self.slerp_precomputed.keys()
                if kf_num in k
            ]
            for k in keys_to_delete:
                del self.slerp_precomputed[k]
        
        if to_delete:
            logger.debug(f"Cleaned up {len(to_delete)} old keyframe latents")
    
    def stop(self) -> None:
        """
        Stop the worker gracefully
        
        The worker will finish processing the current pair and then exit.
        """
        logger.info("Stopping InterpolationWorker...")
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics
        
        Returns:
            Dictionary with worker stats
        """
        avg_time = 0.0
        if self.pairs_processed > 0:
            avg_time = self.total_interpolation_time / self.pairs_processed
        
        return {
            'pairs_processed': self.pairs_processed,
            'frames_generated': self.frames_generated,
            'avg_pair_time': avg_time,
            'queue_depth': self.pair_queue.qsize(),
            'cached_keyframes': len(self.keyframe_latents),
            'is_processing': self.processing,
            'is_running': self.running
        }
    
    def is_keyframe_protected(self, frame_path: Path) -> bool:
        """
        Check if a keyframe should be protected from deletion.
        
        A keyframe is protected if:
        1. It's in the pending interpolation queue (start or end of a pair)
        2. It's currently being processed
        3. It's cached as a latent (we might still need it)
        
        Args:
            frame_path: Path to check
            
        Returns:
            True if keyframe should NOT be deleted
        """
        frame_name = frame_path.name if isinstance(frame_path, Path) else Path(frame_path).name
        
        # Check if it's in the cached keyframe paths (recently used, might be needed)
        for kf_num, cached_path in self.keyframe_paths.items():
            if cached_path.name == frame_name:
                logger.debug(f"Protecting {frame_name}: cached latent for KF{kf_num}")
                return True
        
        # Check pending queue - peek at items without removing them
        # Note: This is thread-safe read of queue internal state
        try:
            # Access queue's internal list (asyncio.Queue uses collections.deque)
            pending_items = list(self.pair_queue._queue)
            for pair in pending_items:
                start_path = pair.get('start_kf_path')
                end_path = pair.get('end_kf_path')
                
                if start_path and Path(start_path).name == frame_name:
                    logger.debug(f"Protecting {frame_name}: pending interpolation start")
                    return True
                if end_path and Path(end_path).name == frame_name:
                    logger.debug(f"Protecting {frame_name}: pending interpolation end")
                    return True
        except Exception as e:
            # If we can't check, err on side of protection
            logger.debug(f"Queue check failed, protecting {frame_name}: {e}")
            return True
        
        return False

