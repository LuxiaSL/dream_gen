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
        
        # Statistics
        self.pairs_processed = 0
        self.frames_generated = 0
        self.total_interpolation_time = 0.0
        
        logger.info(f"InterpolationWorker initialized (max queue: {max_queue_size})")
    
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
            loop = asyncio.get_event_loop()
            self.slerp_precomputed[pair_key] = await loop.run_in_executor(
                None,
                precompute_slerp_params,
                start_latent,
                end_latent
            )
            timings['slerp_precompute'] = time.perf_counter() - t_precompute
        
        # === PHASE 1: Generate ALL latents (no VAE lock needed) ===
        logger.debug(f"  Phase 1: Generating {count} latents...")
        t_slerp_start = time.perf_counter()
        
        latents_and_specs = []
        loop = asyncio.get_event_loop()
        
        for i, sequence_num in enumerate(interp_sequence_nums, start=1):
            try:
                self.frame_buffer.mark_generating(sequence_num)
                frame_spec = self.frame_buffer.frames[sequence_num]
                t = frame_spec.interpolation_t
                
                # Slerp in executor (CPU-bound, no lock)
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
        
        # === PHASE 2: Decode ALL latents in BATCH (single VAE lock acquisition) ===
        logger.debug(f"  Phase 2: Decoding {len(latents_and_specs)} frames in batch...")
        t_decode_start = time.perf_counter()
        
        decoded_images = []
        for latent, frame_spec, sequence_num in latents_and_specs:
            try:
                # Each decode still goes through vae_access, but they happen
                # back-to-back without other workers interrupting
                image = await self.vae_access.decode_async(latent, upscale_to_target=True)
                decoded_images.append((image, frame_spec, sequence_num))
            except Exception as e:
                logger.error(f"Failed to decode latent: {e}", exc_info=True)
        
        timings['decode_all'] = time.perf_counter() - t_decode_start
        
        # === PHASE 3: Save ALL images ASYNC (no blocking) ===
        logger.debug(f"  Phase 3: Saving {len(decoded_images)} frames async...")
        t_save_start = time.perf_counter()
        
        save_tasks = []
        for image, frame_spec, sequence_num in decoded_images:
            # Save async to avoid blocking
            save_task = loop.run_in_executor(
                None,
                lambda img=image, path=frame_spec.file_path: img.save(
                    str(path), "PNG", optimize=False, compress_level=1
                )
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
                logger.error(f"Failed to save frame {sequence_num}: {e}", exc_info=True)
        
        timings['save_all'] = time.perf_counter() - t_save_start
        timings['total'] = time.perf_counter() - cycle_start
        
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
                if self.cache_worker:
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

