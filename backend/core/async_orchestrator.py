"""
AsyncGenerationOrchestrator - Coordinates Async Workers for Parallelized Generation

Coordinates three concurrent workers to eliminate blocking operations and
achieve 2x+ FPS improvement:

- KeyframeWorker: HTTP I/O bound (ComfyUI generation)
- InterpolationWorker: GPU bound (VAE interpolation)
- CacheAnalysisWorker: CPU bound (similarity analysis)

Critical Design:
- Injection decisions stay INLINE in orchestrator (not in workers)
- Uses SharedVAEAccess for thread-safe VAE operations
- Respects VAE lock (prevents CUDA conflicts)
- Maintains keyframe sequence integrity

Usage:
    orchestrator = AsyncGenerationOrchestrator(
        frame_buffer=buffer,
        generator=generator,
        vae_access=vae_access,
        prompt_manager=prompt_manager,
        cache_manager=cache,
        similarity_manager=similarity_manager,
        config=config
    )
    
    await orchestrator.run()
"""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import deque

import torch

from backend.core.workers import KeyframeWorker, InterpolationWorker, CacheAnalysisWorker
from backend.cache.injection_strategy import CacheInjectionStrategy
from backend.cache.collapse_detector import ModeCollapseDetector

logger = logging.getLogger(__name__)


class AsyncGenerationOrchestrator:
    """
    Async orchestrator for coordinating parallel generation workers
    
    Responsibilities:
    1. Start/stop all three workers
    2. Track completed keyframes
    3. Submit interpolation pairs (only when both keyframes ready)
    4. Make injection decisions INLINE (collapse detection + cache/seed injection)
    5. Handle backpressure (queue depth monitoring)
    6. Coordinate graceful shutdown
    
    Architecture:
        Orchestrator (this class)
            │
            ├──> KeyframeWorker (HTTP I/O)
            │    └─> Async wait on ComfyUI
            │
            ├──> InterpolationWorker (GPU compute)
            │    └─> VAE operations via SharedVAEAccess
            │
            └──> CacheAnalysisWorker (CPU analysis)
                 └─> Diversity checks, cache population
        
        INLINE: Injection decisions (collapse detection, VAE blending)
    """
    
    def __init__(
        self,
        frame_buffer,  # FrameBuffer instance
        generator,  # DreamGenerator instance (with async methods)
        vae_access,  # SharedVAEAccess instance
        prompt_manager,  # PromptManager instance
        cache_manager,  # CacheManager instance
        similarity_manager,  # DualMetricSimilarityManager instance
        config: Dict[str, Any],
        seed_image: Optional[Path] = None
    ):
        """
        Initialize async generation orchestrator
        
        Args:
            frame_buffer: FrameBuffer for output frames
            generator: DreamGenerator (with async methods)
            vae_access: SharedVAEAccess (thread-safe VAE wrapper)
            prompt_manager: PromptManager for prompt generation
            cache_manager: CacheManager for frame caching
            similarity_manager: DualMetricSimilarityManager for embeddings
            config: Configuration dictionary
            seed_image: Optional seed image to start generation
        """
        self.buffer = frame_buffer
        self.generator = generator
        self.vae_access = vae_access
        self.prompt_manager = prompt_manager
        self.cache = cache_manager
        self.similarity_manager = similarity_manager
        self.config = config
        
        # === Create Workers ===
        self.keyframe_worker = KeyframeWorker(
            generator=generator,
            frame_buffer=frame_buffer,
            config=config,
            max_queue_size=5
        )
        
        self.interpolation_worker = InterpolationWorker(
            vae_access=vae_access,
            frame_buffer=frame_buffer,
            config=config,
            max_queue_size=10
        )
        
        self.cache_worker = CacheAnalysisWorker(
            cache=cache_manager,
            similarity_manager=similarity_manager,
            config=config,
            max_queue_size=20
        )
        
        # Pass cache_worker to interpolation worker for midpoint caching
        self.interpolation_worker.cache_worker = self.cache_worker
        
        # === Initialize Injection Components ===
        # These stay in orchestrator (inline decisions)
        
        self.collapse_detector = None
        self.injection_strategy = None
        
        # Initialize collapse detector if enabled
        if self.config['generation']['cache'].get('collapse_detection', True):
            # Get dual-metric thresholds if available
            cache_config = self.config['generation']['cache']
            color_config = cache_config.get('color_histogram', {})
            phash_config = cache_config.get('phash', {})
            
            self.collapse_detector = ModeCollapseDetector(
                similarity_manager=similarity_manager,
                history_size=50,
                detection_window=20,
                color_convergence_threshold=color_config.get('convergence_threshold', 0.15),
                color_force_cache_threshold=color_config.get('force_cache_threshold', 0.30),
                struct_convergence_threshold=phash_config.get('convergence_threshold', 0.08),
                struct_force_cache_threshold=phash_config.get('force_cache_threshold', 0.15),
                convergence_mode=cache_config.get('convergence_mode', 'absolute'),
                log_stats=cache_config.get('log_convergence_stats', True)
            )
            logger.info("Collapse detector initialized (inline in orchestrator)")
        
        # Initialize injection strategy (always enabled when cache/similarity available)
        if cache_manager and similarity_manager:
            self.injection_strategy = CacheInjectionStrategy(
                config=self.config,
                cache_manager=cache_manager,
                similarity_manager=similarity_manager,
                vae_access=vae_access,  # Pass SharedVAEAccess (not encoder)
                buffer=frame_buffer
            )
            logger.info("Injection strategy initialized (async with VAE lock)")
        
        # === State Tracking ===
        self.running = False
        self.current_keyframe_num = 0
        self.current_image_path = seed_image
        
        # Track sequence numbers for keyframes (for marking ready)
        # {kf_num: sequence_num}
        self.keyframe_sequences: Dict[int, int] = {}
        
        # Injection tracking
        self.last_seed_injection_kf = 0
        self.last_cache_injection_kf = 0
        self.cache_injections = 0
        self.current_injection_rate = self.config['generation']['cache'].get('injection_probability', 0.15)
        
        # Injection frequency tracking (for seed forcing)
        self.recent_cache_injections = deque(maxlen=10)
        
        # Worker tasks (for graceful shutdown)
        self.keyframe_task: Optional[asyncio.Task] = None
        self.interpolation_task: Optional[asyncio.Task] = None
        self.cache_task: Optional[asyncio.Task] = None
        self.coordination_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.start_time = None
        self.frames_generated = 0
        
        # Note: CUDA profiling removed from __init__ to prevent hanging during initialization
        # CUDA context queries can block if the GPU is busy or torch is still initializing
        
        logger.info("AsyncGenerationOrchestrator initialized")
        logger.info(f"  - Seed image: {seed_image}")
        logger.info(f"  - Collapse detection: {self.collapse_detector is not None}")
        logger.info(f"  - Injection strategy: {self.injection_strategy is not None}")
    
    async def run(self) -> None:
        """
        Main orchestrator run loop
        
        Starts all workers and coordination loop, runs until stopped.
        """
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        logger.info("="*70)
        logger.info("STARTING ASYNC GENERATION ORCHESTRATOR")
        logger.info("="*70)
        
        try:
            # Start all workers
            logger.info("Starting workers...")
            self.keyframe_worker.running = True
            self.interpolation_worker.running = True
            self.cache_worker.running = True
            
            self.keyframe_task = asyncio.create_task(self.keyframe_worker.run())
            self.interpolation_task = asyncio.create_task(self.interpolation_worker.run())
            self.cache_task = asyncio.create_task(self.cache_worker.run())
            
            logger.info("[OK] All workers started")
            
            # Bootstrap: Initialize with seed image as first keyframe
            if self.current_image_path:
                logger.info(f"Bootstrap: Registering seed image as keyframe 1")
                self.current_keyframe_num = 1
                sequence_num = self.buffer.register_keyframe(1)
                self.buffer.mark_ready(sequence_num, self.current_image_path)
                
                # Track sequence number for this keyframe
                self.keyframe_sequences[1] = sequence_num
                
                # Encode seed for interpolation
                try:
                    latent = await self.vae_access.encode_async(
                        self.current_image_path,
                        for_interpolation=True
                    )
                    # Store in interpolation worker cache
                    self.interpolation_worker.keyframe_latents[1] = latent
                    self.interpolation_worker.keyframe_paths[1] = self.current_image_path
                    logger.info("  Seed encoded to latent for interpolation")
                except Exception as e:
                    logger.error(f"Failed to encode seed: {e}")
                
                # Pre-register FIRST CYCLE (keyframe 2 + interpolations 1->2)
                logger.info("Pre-registering first generation cycle...")
                
                # Register interpolations 1->2 FIRST (they come before keyframe 2 in sequence)
                interp_seqs = self.buffer.register_interpolations(1, 2, 
                    self.config['generation']['hybrid']['interpolation_frames'])
                logger.info(f"  Registered interpolations 1->2: seq {interp_seqs[0]}-{interp_seqs[-1]}")
                
                # Then register keyframe 2
                kf2_seq = self.buffer.register_keyframe(2)
                self.keyframe_sequences[2] = kf2_seq
                logger.info(f"  Registered keyframe 2: seq {kf2_seq}")
                
                # Submit keyframe 2 generation (pass sequence number!)
                await self.keyframe_worker.submit_request(
                    current_image=self.current_image_path,
                    keyframe_num=2,
                    sequence_num=kf2_seq,
                    prompt=self.prompt_manager.get_next_prompt()
                )
                logger.info("  Submitted keyframe 2 generation request")
            else:
                logger.error("No seed image provided, cannot start generation")
                self.running = False
                return
            
            # Run coordination loop
            logger.info("Starting coordination loop...")
            await self._coordinate()
            
        except asyncio.CancelledError:
            logger.info("Orchestrator run loop cancelled")
        except Exception as e:
            logger.error(f"Orchestrator run loop error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """
        Stop orchestrator and all workers gracefully
        """
        if not self.running:
            return
        
        logger.info("Stopping AsyncGenerationOrchestrator...")
        self.running = False
        
        # Stop workers
        self.keyframe_worker.running = False
        self.interpolation_worker.running = False
        self.cache_worker.running = False
        
        # Cancel worker tasks
        tasks = [
            ('keyframe', self.keyframe_task),
            ('interpolation', self.interpolation_task),
            ('cache', self.cache_task)
        ]
        
        for name, task in tasks:
            if task and not task.done():
                logger.info(f"  Cancelling {name} worker...")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(f"  {name} worker did not stop cleanly")
                except asyncio.CancelledError:
                    pass
        
        # Print final statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info("="*70)
        logger.info("ORCHESTRATOR STATISTICS")
        logger.info(f"  Total runtime: {elapsed:.1f}s")
        logger.info(f"  Keyframes generated: {self.keyframe_worker.requests_processed}")
        logger.info(f"  Interpolations: {self.interpolation_worker.pairs_processed}")
        logger.info(f"  Cache injections: {self.cache_injections}")
        logger.info(f"  Cache analyses: {self.cache_worker.frames_analyzed}")
        logger.info("="*70)
    
    async def _coordinate(self) -> None:
        """
        Main coordination loop - Smart Pre-Registration Pattern
        
        Key Changes from Original:
        1. Pre-registers entire cycles (interpolations + next keyframe) atomically
        2. Passes sequence numbers to workers (workers don't register)
        3. Uses buffer.needs_interpolations() for gap detection
        4. Checks buffer pacing (don't over-generate)
        5. Removes duplicate state tracking
        
        Flow:
        1. Wait for keyframe completion
        2. Mark keyframe ready in buffer
        3. Check for missing interpolation pairs (gap detection)
        4. Pre-register next cycle (interpolations + keyframe)
        5. Submit work to workers with sequence numbers
        6. Check buffer pacing / backpressure
        """
        logger.info("Coordination loop active (Smart Pre-Registration)")
        
        while self.running:
            try:
                # === 1. Wait for Keyframe Completion OR Buffer Drain (if throttled) ===
                # First check: are we currently throttled?
                buffer_status = self.buffer.get_buffer_status()
                seconds_buffered = buffer_status['seconds_buffered']
                target_seconds = buffer_status['target_seconds']
                is_throttled = seconds_buffered >= target_seconds
                
                if is_throttled:
                    # Buffer is full - don't wait for completions (there may be none!)
                    # Instead, sleep briefly and check buffer status again
                    logger.debug(f"  System throttled ({seconds_buffered:.1f}s / {target_seconds}s), waiting for buffer to drain...")
                    await asyncio.sleep(0.5)
                    continue  # Loop back to check buffer status again
                
                # Not throttled, wait for next keyframe completion
                try:
                    result = await asyncio.wait_for(
                        self.keyframe_worker.result_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No keyframe ready yet, check if we should stop
                    continue
                
                kf_num = result['keyframe_num']
                sequence_num = result.get('sequence_num')
                
                # === CHECK FOR FAILURE ===
                if not result.get('success', True):
                    # Keyframe generation failed! Need to recover.
                    error = result.get('error', 'Unknown error')
                    retries = result.get('retries', 0)
                    logger.error(f"[FAIL] Keyframe {kf_num} failed after {retries} retries: {error}")
                    
                    # Mark task done
                    self.keyframe_worker.result_queue.task_done()
                    
                    # === RECOVERY: Skip this keyframe and its pending interpolations ===
                    await self._handle_keyframe_failure(kf_num, sequence_num)
                    continue
                
                kf_path = result['path']
                prompt = result['prompt']
                gen_time = result.get('generation_time', 0.0)
                
                logger.info(f"[OK] Keyframe {kf_num} completed: {kf_path.name}")
                
                # === 2. Mark Keyframe Ready ===
                if sequence_num is not None:
                    self.buffer.mark_ready(sequence_num, kf_path)
                    logger.debug(f"  Marked keyframe {kf_num} ready (seq {sequence_num})")
                else:
                    # Fallback: look up sequence number
                    sequence_num = self.keyframe_sequences.get(kf_num)
                    if sequence_num is not None:
                        self.buffer.mark_ready(sequence_num, kf_path)
                    else:
                        logger.error(f"  Cannot find sequence number for keyframe {kf_num}!")
                
                # Update current state
                self.current_image_path = kf_path
                self.current_keyframe_num = kf_num
                
                # === Cleanup old keyframes ===
                # Keep last 2 keyframes for retry capability, delete older ones
                # This is safe because keyframe N-1 is no longer needed as source
                # once keyframe N is successfully generated
                await self._cleanup_old_keyframes(kf_num)
                
                # Mark task done
                self.keyframe_worker.result_queue.task_done()
                
                # === 3. Check for Missing Interpolation Pairs (Gap Detection) ===
                # Use buffer's built-in logic to find gaps!
                missing_pair = self.buffer.needs_interpolations()
                
                if missing_pair:
                    start_kf, end_kf = missing_pair
                    logger.info(f"  Gap detected: Missing interpolations {start_kf}->{end_kf}")
                    
                    # Check if we have both keyframe paths
                    start_path = None
                    end_path = None
                    
                    # Get start keyframe path
                    start_seq = self.buffer.get_keyframe_sequence_num(start_kf)
                    if start_seq is not None and start_seq in self.buffer.frames:
                        start_frame = self.buffer.frames[start_seq]
                        if start_frame.is_ready() and start_frame.file_path:
                            start_path = start_frame.file_path
                    
                    # Get end keyframe path
                    end_seq = self.buffer.get_keyframe_sequence_num(end_kf)
                    if end_seq is not None and end_seq in self.buffer.frames:
                        end_frame = self.buffer.frames[end_seq]
                        if end_frame.is_ready() and end_frame.file_path:
                            end_path = end_frame.file_path
                    
                    if start_path and end_path:
                        # Get interpolation sequence numbers (already registered)
                        interp_seqs = []
                        for seq, frame in self.buffer.frames.items():
                            if (frame.is_interpolated() and 
                                frame.keyframe_pair == (start_kf, end_kf)):
                                interp_seqs.append((seq, frame.interpolation_t))
                        
                        # Sort by sequence number
                        interp_seqs.sort(key=lambda x: x[0])
                        sequence_nums = [seq for seq, _ in interp_seqs]
                        
                        logger.info(f"  Submitting gap-fill: KF{start_kf}->KF{end_kf}")
                        await self.interpolation_worker.submit_pair(
                            start_kf_num=start_kf,
                            end_kf_num=end_kf,
                            start_kf_path=start_path,
                            end_kf_path=end_path,
                            interp_sequence_nums=sequence_nums
                        )
                    else:
                        logger.warning(f"  Cannot fill gap {start_kf}->{end_kf}: missing keyframe paths")
                
                # === 4. Check if Adjacent Interpolations Need Submission ===
                # If previous keyframe exists and its interpolations to current are registered
                prev_kf = kf_num - 1
                if prev_kf > 0:
                    # Check if interpolations prev_kf -> kf_num are registered but not submitted
                    prev_seq = self.buffer.get_keyframe_sequence_num(prev_kf)
                    if prev_seq is not None and prev_seq in self.buffer.frames:
                        prev_frame = self.buffer.frames[prev_seq]
                        if prev_frame.is_ready() and prev_frame.file_path:
                            prev_path = prev_frame.file_path
                            
                            # Check if interpolations exist and are pending
                            interp_pending = False
                            interp_seqs = []
                            
                            for seq, frame in self.buffer.frames.items():
                                if (frame.is_interpolated() and 
                                    frame.keyframe_pair == (prev_kf, kf_num)):
                                    interp_seqs.append((seq, frame.interpolation_t))
                                    if frame.state == self.buffer.frames[seq].state.__class__.PENDING:
                                        interp_pending = True
                            
                            if interp_seqs and interp_pending:
                                # Sort by sequence number
                                interp_seqs.sort(key=lambda x: x[0])
                                sequence_nums = [seq for seq, _ in interp_seqs]
                                
                                logger.info(f"  Submitting interpolation: KF{prev_kf}->KF{kf_num}")
                                await self.interpolation_worker.submit_pair(
                                    start_kf_num=prev_kf,
                                    end_kf_num=kf_num,
                                    start_kf_path=prev_path,
                                    end_kf_path=kf_path,
                                    interp_sequence_nums=sequence_nums
                                )
                
                # === 5. Submit Cache Analysis (fire and forget) ===
                await self.cache_worker.submit_frame(
                    frame_path=kf_path,
                    prompt=prompt,
                    metadata={'denoise': gen_time, 'type': 'keyframe'}
                )
                
                # === 6. Decide Next Keyframe (with injection logic) ===
                next_kf = kf_num + 1
                
                # Check if should inject (collapse detection + probability)
                should_inject, injection_type = await self._should_inject_now(kf_path)
                
                if should_inject:
                    logger.info(f"  -> Injection triggered ({injection_type})")
                    
                    # Pre-register cycle for injection
                    # Only register: current->next interpolations + next keyframe
                    # Don't register further ahead!
                    
                    # Check if interpolations kf_num -> next_kf already registered
                    has_interp = any(
                        frame.is_interpolated() and frame.keyframe_pair == (kf_num, next_kf)
                        for frame in self.buffer.frames.values()
                    )
                    
                    if not has_interp:
                        interp_seqs = self.buffer.register_interpolations(
                            kf_num, next_kf,
                            self.config['generation']['hybrid']['interpolation_frames']
                        )
                        logger.info(f"  Pre-registered interpolations {kf_num}->{next_kf}: seq {interp_seqs[0]}-{interp_seqs[-1]}")
                    
                    # Check if next keyframe already registered
                    if next_kf in self.keyframe_sequences:
                        next_seq = self.keyframe_sequences[next_kf]
                        logger.debug(f"  Keyframe {next_kf} already registered at seq {next_seq}")
                    else:
                        next_seq = self.buffer.register_keyframe(next_kf)
                        self.keyframe_sequences[next_kf] = next_seq
                        logger.info(f"  Pre-registered keyframe {next_kf}: seq {next_seq}")
                    
                    # Perform injection inline
                    injected_result = await self._inject_frame_inline(
                        next_kf,
                        next_seq,
                        kf_path,
                        injection_type
                    )
                    
                    if injected_result:
                        # Injection succeeded, keyframe is already marked ready
                        # Feed back into result queue for next iteration
                        await self.keyframe_worker.result_queue.put({
                            'keyframe_num': next_kf,
                            'path': injected_result['path'],
                            'prompt': 'injected',
                            'generation_time': injected_result.get('injection_time', 0.0),
                            'sequence_num': next_seq
                        })
                        
                        # Track injection
                        if injection_type == 'seed':
                            self.last_seed_injection_kf = next_kf
                        else:
                            self.last_cache_injection_kf = next_kf
                            self.cache_injections += 1
                            self.recent_cache_injections.append(True)
                        
                        logger.info(f"  [OK] Injection completed, proceeding to next iteration")
                        continue
                    else:
                        logger.warning(f"  Injection failed, falling back to normal generation")
                        # Fall through to normal generation
                else:
                    # Track no injection
                    self.recent_cache_injections.append(False)
                
                # === 8. Normal Generation - Pre-register Next Cycle ===
                next_prompt = self.prompt_manager.get_next_prompt()
                
                # Pre-register ONLY ONE cycle ahead:
                # 1. Interpolations current -> next (if not already done)
                # 2. Keyframe next
                # Don't register further ahead - causes duplicate registrations!
                
                # Check if interpolations kf_num -> next_kf already registered
                has_interp = any(
                    frame.is_interpolated() and frame.keyframe_pair == (kf_num, next_kf)
                    for frame in self.buffer.frames.values()
                )
                
                if not has_interp:
                    interp_seqs = self.buffer.register_interpolations(
                        kf_num, next_kf,
                        self.config['generation']['hybrid']['interpolation_frames']
                    )
                    logger.info(f"  Pre-registered interpolations {kf_num}->{next_kf}: seq {interp_seqs[0]}-{interp_seqs[-1]}")
                
                # Check if next keyframe already registered
                if next_kf in self.keyframe_sequences:
                    # Already registered in previous iteration
                    next_seq = self.keyframe_sequences[next_kf]
                    logger.debug(f"  Keyframe {next_kf} already registered at seq {next_seq}")
                else:
                    # Register next keyframe
                    next_seq = self.buffer.register_keyframe(next_kf)
                    self.keyframe_sequences[next_kf] = next_seq
                    logger.info(f"  Pre-registered keyframe {next_kf}: seq {next_seq}")
                
                # Mark as generating
                self.buffer.mark_generating(next_seq)
                
                # Submit keyframe generation with sequence number
                logger.info(f"  Submitting keyframe {next_kf} generation request")
                await self.keyframe_worker.submit_request(
                    current_image=kf_path,
                    keyframe_num=next_kf,
                    sequence_num=next_seq,
                    prompt=next_prompt
                )
                
                # === 9. Memory Management ===
                # Clean up old keyframe sequence tracking (keep last 10)
                if len(self.keyframe_sequences) > 10:
                    old_kfs = sorted(self.keyframe_sequences.keys())[:-10]
                    for old_kf in old_kfs:
                        del self.keyframe_sequences[old_kf]
                
                # === 10. Backpressure Check ===
                interp_depth = self.interpolation_worker.pair_queue.qsize()
                if interp_depth > 5:
                    logger.warning(
                        f"Interpolation queue depth high ({interp_depth}), throttling..."
                    )
                    await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                logger.info("Coordination loop cancelled")
                break
            except Exception as e:
                logger.error(f"Coordination loop error: {e}", exc_info=True)
                
                # === CRITICAL: Don't let exceptions starve the system ===
                # Check if workers are still running
                if not self.keyframe_worker.running:
                    logger.error("KeyframeWorker died! Restarting...")
                    self.keyframe_worker.running = True
                    self.keyframe_task = asyncio.create_task(self.keyframe_worker.run())
                
                if not self.interpolation_worker.running:
                    logger.error("InterpolationWorker died! Restarting...")
                    self.interpolation_worker.running = True
                    self.interpolation_task = asyncio.create_task(self.interpolation_worker.run())
                
                # Sleep briefly and continue (don't crash on single error)
                await asyncio.sleep(1.0)
        
        logger.info("Coordination loop exited")
    
    async def _should_inject_now(
        self,
        current_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if should inject cached/seed frame (INLINE decision)
        
        Integrates:
        - Collapse detection (analyze convergence)
        - Cooldown checks (prevent injection loops)
        - Warmup period (skip injections during warmup)
        - Adaptive injection rate (scale based on collapse)
        - Seed forcing (persistent collapse)
        
        Args:
            current_path: Path to current keyframe
            
        Returns:
            Tuple of (should_inject: bool, injection_type: str or None)
            injection_type is 'seed' or 'cache' if should_inject is True
        """
        if not self.injection_strategy or not self.cache or not self.similarity_manager:
            return False, None
        
        kf_num = self.current_keyframe_num + 1
        
        # === WARMUP PERIOD CHECK ===
        warmup_keyframes = self.config['generation']['cache'].get('warmup_keyframes', 0)
        in_warmup = kf_num <= warmup_keyframes
        
        if in_warmup:
            logger.debug(
                f"[WARMUP] Keyframe {kf_num}/{warmup_keyframes} - "
                f"skipping injection (establishing baseline)"
            )
            return False, None
        elif kf_num == warmup_keyframes + 1:
            # First keyframe after warmup
            logger.info(
                f"[WARMUP_COMPLETE] Warmup period finished! "
                f"Collapse detection and adaptive interventions now ACTIVE. "
                f"Cache size: {self.cache.size() if self.cache else 0}"
            )
        
        # === MODE COLLAPSE DETECTION ===
        collapse_result = None
        if self.collapse_detector and self.similarity_manager and current_path:
            try:
                # Encode current frame for collapse detection
                current_embedding = self.similarity_manager.encode_image(current_path)
                if current_embedding is not None:
                    collapse_result = self.collapse_detector.analyze_frame(current_embedding)
                    
                    # Record collapse status for adaptive seed injection
                    is_collapsed = collapse_result['status'] in ['converging', 'collapsed']
                    self.injection_strategy.record_collapse_detection(is_collapsed)
                    
                    # Log collapse metrics
                    if collapse_result['convergence_delta'] > 0:
                        logger.info(
                            f"[COLLAPSE_METRICS] Status: {collapse_result['status']}, "
                            f"Delta: {collapse_result['convergence_delta']:.3f}, "
                            f"Similarity: {collapse_result['avg_similarity']:.3f}"
                        )
                    
                    # Adjust injection rate based on collapse status
                    baseline_prob = self.config['generation']['cache']['injection_probability']
                    
                    if collapse_result['action'] == 'scale_injection':
                        # Gradually scale from baseline to 100%
                        scaled_prob = collapse_result.get('scaled_injection_probability', 0.0)
                        self.current_injection_rate = baseline_prob + (1.0 - baseline_prob) * scaled_prob
                        logger.info(
                            f"[SCALING] Injection probability scaled to {self.current_injection_rate:.0%} "
                            f"(delta: {collapse_result['convergence_delta']:.3f})"
                        )
                    elif collapse_result['action'] == 'force_cache':
                        # Will force injection below
                        pass
                    else:
                        # Normal baseline rate
                        self.current_injection_rate = baseline_prob
            except Exception as e:
                logger.error(f"Collapse detection failed: {e}", exc_info=True)
        
        # === SEED INJECTION (Adaptive or Forced) ===
        seed_cooldown = self.config['generation']['cache'].get('seed_injection_cooldown', 2)
        keyframes_since_seed = kf_num - self.last_seed_injection_kf
        
        if keyframes_since_seed > seed_cooldown:
            # Check if we should force seed based on injection frequency
            recent_injection_count = sum(1 for x in self.recent_cache_injections if x)
            injection_frequency = recent_injection_count / len(self.recent_cache_injections) if self.recent_cache_injections else 0.0
            
            force_seed_threshold = self.config['generation']['cache'].get('force_seed_injection_frequency', 0.5)
            force_seed_from_frequency = injection_frequency > force_seed_threshold
            
            if force_seed_from_frequency:
                logger.warning(
                    f"[EMERGENCY] High injection frequency ({injection_frequency:.0%}) -> forcing seed"
                )
                return True, 'seed'
            
            # Or adaptive seed injection
            should_seed = self.injection_strategy.should_inject_seed()
            if should_seed:
                return True, 'seed'
        
        # === CACHE INJECTION (Dissimilar Strategy) ===
        cache_cooldown = self.config['generation']['cache'].get('injection_cooldown', 3)
        keyframes_since_cache = kf_num - self.last_cache_injection_kf
        on_cooldown = keyframes_since_cache <= cache_cooldown
        
        force_cache = collapse_result and collapse_result['action'] == 'force_cache'
        probability_cache = (
            not force_cache and
            not on_cooldown and
            self.cache.size() > 0 and
            random.random() < self.current_injection_rate
        )
        
        # Force cache ignores cooldown
        if force_cache or probability_cache:
            return True, 'cache'
        
        return False, None
    
    async def _inject_frame_inline(
        self,
        keyframe_num: int,
        sequence_num: int,
        current_path: Path,
        injection_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Perform frame injection inline (BLOCKS orchestrator but workers continue)
        
        Uses SharedVAEAccess for VAE operations to prevent CUDA conflicts.
        
        Args:
            keyframe_num: Target keyframe number
            sequence_num: Sequence number for this keyframe (already registered!)
            current_path: Path to current keyframe
            injection_type: 'seed' or 'cache'
            
        Returns:
            Dictionary with injection result:
            {'path': Path, 'injection_time': float}
            or None if injection failed
        """
        start_time = time.time()
        
        try:
            # Sequence already registered by orchestrator, just mark generating
            self.buffer.mark_generating(sequence_num)
            
            result = None
            
            # === SEED INJECTION ===
            if injection_type == 'seed':
                logger.info(f"  -> Injecting SEED frame (keyframe {keyframe_num})")
                
                result = await self.injection_strategy.inject_seed_frame(
                    target_keyframe_num=keyframe_num,
                    current_image_path=current_path
                )
                
                if result:
                    target_path, metadata = result
                    
                    # Reset embedding history to break convergence signal
                    if self.collapse_detector:
                        reset_mode = self.config['generation']['cache'].get('embedding_history_reset', 'partial')
                        if reset_mode == 'full':
                            self.collapse_detector.reset()
                        elif reset_mode == 'partial':
                            keep_recent = self.config['generation']['cache'].get('embedding_history_keep_recent', 5)
                            self.collapse_detector.partial_reset(keep_recent)
                        logger.info(f"  Embedding history reset ({reset_mode}) after seed injection")
                    
                    # Mark as ready in buffer
                    self.buffer.mark_ready(sequence_num, target_path)
                    
                    # Encode for VAE interpolation
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        latent = await self.vae_access.encode_async(
                            target_path,
                            for_interpolation=True
                        )
                        self.interpolation_worker.keyframe_latents[keyframe_num] = latent
                        self.interpolation_worker.keyframe_paths[keyframe_num] = target_path
                        
                        logger.debug(f"  Encoded seed keyframe {keyframe_num} to latent")
                    except Exception as e:
                        logger.error(f"Failed to encode seed keyframe: {e}")
                    
                    # Submit to cache analysis (seeds are always diverse)
                    await self.cache_worker.submit_frame(
                        frame_path=target_path,
                        prompt='seed_injection',
                        metadata={'denoise': 0.0, 'type': 'seed', 'injection': True}
                    )
                    
                    injection_time = time.time() - start_time
                    logger.info(f"[OK] Keyframe {keyframe_num} from SEED ({metadata['type']}) in {injection_time:.2f}s")
                    logger.info(f"     Saved to: {target_path.name}")
                    
                    return {
                        'path': target_path,
                        'injection_time': injection_time
                    }
            
            # === CACHE INJECTION ===
            elif injection_type == 'cache':
                logger.info(f"  -> Injecting CACHE frame (keyframe {keyframe_num})")
                
                # Extract which metric triggered (for smart selection)
                collapse_trigger = None
                # Note: collapse_result not passed here, would need to store in state if needed
                
                result = await self.injection_strategy.inject_dissimilar_keyframe(
                    current_image_path=current_path,
                    target_keyframe_num=keyframe_num,
                    collapse_trigger=collapse_trigger
                )
                
                if result:
                    target_path, metadata = result
                    
                    # Reset embedding history to break convergence signal
                    if self.collapse_detector:
                        reset_mode = self.config['generation']['cache'].get('embedding_history_reset', 'partial')
                        if reset_mode == 'full':
                            self.collapse_detector.reset()
                        elif reset_mode == 'partial':
                            keep_recent = self.config['generation']['cache'].get('embedding_history_keep_recent', 5)
                            self.collapse_detector.partial_reset(keep_recent)
                        logger.info(f"  Embedding history reset ({reset_mode}) after cache injection")
                    
                    # Mark as ready in buffer
                    self.buffer.mark_ready(sequence_num, target_path)
                    
                    # Encode for VAE interpolation
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        latent = await self.vae_access.encode_async(
                            target_path,
                            for_interpolation=True
                        )
                        self.interpolation_worker.keyframe_latents[keyframe_num] = latent
                        self.interpolation_worker.keyframe_paths[keyframe_num] = target_path
                        
                        logger.debug(f"  Encoded dissimilar keyframe {keyframe_num} to latent")
                    except Exception as e:
                        logger.error(f"Failed to encode dissimilar keyframe: {e}")
                    
                    # Submit to cache analysis (for diversity tracking)
                    await self.cache_worker.submit_frame(
                        frame_path=target_path,
                        prompt='cache_injection',
                        metadata={'denoise': 0.0, 'type': 'cache_injection', 'injection': True}
                    )
                    
                    injection_time = time.time() - start_time
                    logger.info(f"[OK] Keyframe {keyframe_num} from DISSIMILAR CACHE in {injection_time:.2f}s")
                    logger.info(f"     Saved to: {target_path.name}")
                    
                    return {
                        'path': target_path,
                        'injection_time': injection_time
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Injection failed: {e}", exc_info=True)
            return None
    
    async def _cleanup_old_keyframes(self, current_kf_num: int, keep_count: int = 2) -> None:
        """
        Clean up old keyframe files that are no longer needed.
        
        Keeps the last `keep_count` keyframes for retry capability.
        Called after a keyframe is successfully generated.
        
        Args:
            current_kf_num: The keyframe that was just successfully generated
            keep_count: How many old keyframes to keep (default 2)
        """
        try:
            keyframe_dir = self.buffer.keyframe_dir if hasattr(self.buffer, 'keyframe_dir') else None
            if not keyframe_dir or not keyframe_dir.exists():
                return
            
            # Delete keyframes older than (current - keep_count)
            min_to_keep = current_kf_num - keep_count
            if min_to_keep <= 0:
                return  # Nothing to clean up yet
            
            deleted_count = 0
            for kf_file in keyframe_dir.glob("keyframe_*.png"):
                try:
                    # Extract keyframe number from filename
                    kf_num_str = kf_file.stem.replace("keyframe_", "")
                    kf_num = int(kf_num_str)
                    
                    if kf_num < min_to_keep:
                        kf_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Cleaned up old keyframe: {kf_file.name}")
                except (ValueError, OSError) as e:
                    logger.debug(f"Could not cleanup {kf_file.name}: {e}")
            
            if deleted_count > 0:
                logger.debug(f"Cleaned up {deleted_count} old keyframe(s) (keeping last {keep_count})")
                
        except Exception as e:
            # Non-fatal - just log and continue
            logger.debug(f"Keyframe cleanup error: {e}")
    
    async def _handle_keyframe_failure(self, kf_num: int, sequence_num: Optional[int]) -> None:
        """
        Handle a failed keyframe by cleaning up orphaned interpolations and continuing
        
        Recovery strategy:
        1. Mark the failed keyframe as skipped/failed in buffer
        2. Remove any pending interpolations that depend on this keyframe
        3. Use the PREVIOUS keyframe as the current image for next generation
        4. Re-register a new keyframe to replace the failed one
        
        Args:
            kf_num: The failed keyframe number
            sequence_num: The sequence number in buffer (if known)
        """
        logger.warning(f"=== RECOVERING FROM KEYFRAME {kf_num} FAILURE ===")
        
        # === 1. Mark failed keyframe in buffer ===
        if sequence_num is not None:
            try:
                # Mark as failed/skipped
                self.buffer.mark_failed(sequence_num)
                logger.info(f"  Marked keyframe {kf_num} (seq {sequence_num}) as failed")
            except AttributeError:
                # Buffer might not have mark_failed - just remove it
                if sequence_num in self.buffer.frames:
                    del self.buffer.frames[sequence_num]
                    logger.info(f"  Removed failed keyframe {kf_num} from buffer")
        
        # === 2. Remove orphaned interpolations ===
        # Find all interpolations that depend on the failed keyframe
        orphaned_seqs = []
        for seq, frame in list(self.buffer.frames.items()):
            if frame.is_interpolated():
                pair = frame.keyframe_pair
                if pair and kf_num in pair:
                    orphaned_seqs.append(seq)
        
        if orphaned_seqs:
            logger.info(f"  Found {len(orphaned_seqs)} orphaned interpolations to remove")
            for seq in orphaned_seqs:
                try:
                    if seq in self.buffer.frames:
                        del self.buffer.frames[seq]
                except Exception as e:
                    logger.warning(f"  Failed to remove orphaned frame {seq}: {e}")
            logger.info(f"  Cleaned up {len(orphaned_seqs)} orphaned interpolations")
        
        # === 3. Find the last successful keyframe to use as base ===
        # Look for the most recent keyframe with an existing file
        # (keyframes are no longer cleaned up, so files should exist)
        last_good_kf = kf_num - 1
        last_good_path = None
        last_good_seq = None
        
        # Get keyframe directory from buffer
        keyframe_dir = self.buffer.keyframe_dir if hasattr(self.buffer, 'keyframe_dir') else None
        
        while last_good_kf > 0 and keyframe_dir:
            # Check if keyframe file exists on disk
            kf_path = keyframe_dir / f"keyframe_{last_good_kf:03d}.png"
            if kf_path.exists():
                last_good_path = kf_path
                last_good_seq = self.keyframe_sequences.get(last_good_kf)
                break
            last_good_kf -= 1
        
        if last_good_path:
            self.current_image_path = last_good_path
            self.current_keyframe_num = last_good_kf
            logger.info(f"  Falling back to keyframe {last_good_kf}: {last_good_path.name}")
            
            # === 4. Reset display position if we have orphaned display ===
            # If display is waiting on the failed keyframe's interpolations,
            # reset it to the last good keyframe
            if last_good_seq is not None:
                current_display = self.buffer.display_sequence_num
                min_orphan = min(orphaned_seqs) if orphaned_seqs else float('inf')
                
                if orphaned_seqs and current_display in orphaned_seqs:
                    # Display was pointing at an orphaned frame - reset to last good keyframe
                    self.buffer.display_sequence_num = last_good_seq
                    logger.warning(f"  Reset display from seq {current_display} to {last_good_seq} (last good KF)")
                elif current_display >= min_orphan:
                    # Display is past the orphaned range - still reset to be safe
                    self.buffer.display_sequence_num = last_good_seq
                    logger.warning(f"  Reset display to seq {last_good_seq} (was {current_display})")
        else:
            logger.warning(f"  No previous keyframe file found, keeping current state")
        
        # === 5. Trigger immediate re-generation ===
        # Re-register the next keyframe and its interpolations
        next_kf = self.current_keyframe_num + 1
        interp_frames = self.config['generation']['hybrid']['interpolation_frames']
        
        # Calculate new sequence number after the last good keyframe
        if last_good_seq is not None:
            # Pre-register interpolations and keyframe
            self.buffer.register_interpolations(
                self.current_keyframe_num, 
                next_kf, 
                interp_frames
            )
            new_kf_seq = self.buffer.register_keyframe(next_kf)
            self.keyframe_sequences[next_kf] = new_kf_seq
            logger.info(f"  Re-registered KF{next_kf} at seq {new_kf_seq} with {interp_frames} interps")
        
        logger.info(f"  Recovery complete. Next generation: KF{next_kf}")
        logger.warning(f"=== RECOVERY COMPLETE ===")
    
    async def hard_reset(self) -> None:
        """
        Perform a complete system reset - nuclear option when all recovery fails.
        
        This clears all state and restarts generation from scratch using a seed image.
        Called by display selector when multiple consecutive recoveries have failed.
        
        DOES NOT delete existing files - just resets internal state and restarts.
        """
        logger.error("=" * 60)
        logger.error("HARD RESET - Restarting generation pipeline from scratch")
        logger.error("=" * 60)
        
        try:
            # === 1. Clear buffer state ===
            logger.warning("  Clearing buffer...")
            self.buffer.frames.clear()
            self.buffer.next_sequence_num = 0
            self.buffer.display_sequence_num = 0
            self.keyframe_sequences.clear()
            
            # === 2. Reset orchestrator state ===
            logger.warning("  Resetting orchestrator state...")
            self.current_keyframe_num = 0
            self.cache_injections = 0
            
            # === 3. Find a seed image to restart from ===
            # Priority: 1) Recent keyframe on disk, 2) Seed directory, 3) Cache
            restart_image = None
            
            # Try to find any existing keyframe
            keyframe_dir = self.buffer.keyframe_dir if hasattr(self.buffer, 'keyframe_dir') else None
            if keyframe_dir and keyframe_dir.exists():
                keyframes = sorted(keyframe_dir.glob("keyframe_*.png"), reverse=True)
                if keyframes:
                    restart_image = keyframes[0]  # Most recent keyframe
                    logger.warning(f"  Found restart image: {restart_image.name}")
            
            # Fallback to seed directory
            if not restart_image:
                seed_dir = Path(self.config['system'].get('seed_dir', 'seeds'))
                if seed_dir.exists():
                    seeds = list(seed_dir.glob("*.png")) + list(seed_dir.glob("*.jpg"))
                    if seeds:
                        import random
                        restart_image = random.choice(seeds)
                        logger.warning(f"  Using seed image: {restart_image.name}")
            
            if not restart_image:
                logger.error("  No restart image found - cannot perform hard reset!")
                return
            
            # === 4. Reinitialize generation ===
            logger.warning(f"  Restarting from: {restart_image}")
            self.current_image_path = restart_image
            
            # Register first keyframe
            first_kf_seq = self.buffer.register_keyframe(1)
            self.keyframe_sequences[1] = first_kf_seq
            
            # Mark it as ready (using the restart image)
            self.buffer.mark_ready(first_kf_seq, restart_image)
            
            # Copy restart image to keyframe_001 for consistency
            if keyframe_dir:
                target_path = keyframe_dir / "keyframe_001.png"
                if not target_path.exists() or restart_image != target_path:
                    import shutil
                    shutil.copy(str(restart_image), str(target_path))
                    self.current_image_path = target_path
            
            self.current_keyframe_num = 1
            
            # Pre-register second keyframe cycle
            interp_frames = self.config['generation']['hybrid']['interpolation_frames']
            self.buffer.register_interpolations(1, 2, interp_frames)
            second_kf_seq = self.buffer.register_keyframe(2)
            self.keyframe_sequences[2] = second_kf_seq
            
            # Update interpolation worker's keyframe paths
            if self.interpolation_worker:
                self.interpolation_worker.keyframe_paths[1] = self.current_image_path
            
            logger.warning("=" * 60)
            logger.warning("HARD RESET COMPLETE - Generation restarting")
            logger.warning(f"  Restart image: {self.current_image_path.name}")
            logger.warning(f"  First keyframe: seq {first_kf_seq}")
            logger.warning(f"  Registered: {interp_frames} interps + KF2")
            logger.warning("=" * 60)
            
        except Exception as e:
            logger.error(f"Hard reset failed: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics including mode collapse metrics
        
        Returns:
            Dictionary with statistics
        """
        cache_size = self.cache.size() if self.cache else 0
        
        # Get avg generation time from keyframe worker
        avg_gen_time = 0.0
        if self.keyframe_worker:
            worker_stats = self.keyframe_worker.get_stats()
            avg_gen_time = worker_stats.get('avg_generation_time', 0.0)
        
        # Get interpolation count from interpolation worker
        interpolations_generated = 0
        if self.interpolation_worker:
            interp_stats = self.interpolation_worker.get_stats()
            interpolations_generated = interp_stats.get('frames_generated', 0)
        
        stats = {
            "keyframes_generated": self.current_keyframe_num,
            "interpolations_generated": interpolations_generated,  # From worker stats
            "cache_injections": self.cache_injections,
            "cache_size": cache_size,
            "current_keyframe": self.current_keyframe_num,
            "avg_generation_time": avg_gen_time,  # CRITICAL: Required by status updater
            "is_running": self.running
        }
        
        # Add mode collapse detection stats
        if self.collapse_detector:
            try:
                collapse_stats = self.collapse_detector.get_stats()
                stats.update({
                    "collapse_recent_similarity": collapse_stats.get("recent_avg_similarity", 0.0),
                    "collapse_overall_similarity": collapse_stats.get("overall_avg_similarity", 0.0),
                    "collapse_frames_analyzed": collapse_stats.get("frames_analyzed", 0)
                })
            except Exception as e:
                logger.debug(f"Failed to get collapse stats: {e}")
        
        # Add injection strategy stats
        if self.injection_strategy:
            try:
                injection_stats = self.injection_strategy.get_stats()
                stats.update({
                    "total_seed_injections": injection_stats.get("total_seed_injections", 0),
                    "total_cache_injections": injection_stats.get("total_cache_injections", 0),
                    "collapse_frequency": injection_stats.get("recent_collapse_frequency", 0.0)
                })
            except Exception as e:
                logger.debug(f"Failed to get injection stats: {e}")
        
        # Add cache diversity stats
        if self.cache and self.similarity_manager:
            try:
                diversity_stats = self.cache.get_diversity_stats(similarity_manager=self.similarity_manager)
                
                # Handle dual-metric stats
                if 'diversity_score_color' in diversity_stats:
                    stats.update({
                        "cache_diversity_score_color": diversity_stats.get("diversity_score_color", 0.0),
                        "cache_diversity_score_struct": diversity_stats.get("diversity_score_struct", 0.0),
                        "cache_avg_color_similarity": diversity_stats.get("avg_color_similarity", 0.0),
                        "cache_avg_struct_similarity": diversity_stats.get("avg_struct_similarity", 0.0)
                    })
                else:
                    # Fallback for single-metric
                    stats.update({
                        "cache_diversity_score": diversity_stats.get("diversity_score", 0.0),
                        "cache_avg_similarity": diversity_stats.get("avg_similarity", 0.0)
                    })
            except Exception as e:
                logger.debug(f"Failed to get cache diversity stats: {e}")
        
        # Add worker queue depths
        if self.keyframe_worker:
            stats["keyframe_queue_depth"] = self.keyframe_worker.request_queue.qsize()
        if self.interpolation_worker:
            stats["interpolation_queue_depth"] = self.interpolation_worker.pair_queue.qsize()
        if self.cache_worker:
            stats["cache_queue_depth"] = self.cache_worker.analysis_queue.qsize()
        
        return stats

