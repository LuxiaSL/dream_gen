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
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import deque

import torch

from backend.core.workers import KeyframeWorker, InterpolationWorker, CacheAnalysisWorker
from backend.cache.injection_strategy import CacheInjectionStrategy
from backend.fresh import FreshFrameBuffer

logger = logging.getLogger(__name__)


class AsyncGenerationOrchestrator:
    """
    Async orchestrator for coordinating parallel generation workers
    
    Responsibilities:
    1. Start/stop all three workers
    2. Track completed keyframes
    3. Submit interpolation pairs (only when both keyframes ready)
    4. Make injection decisions INLINE (cache/seed injection)
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
        
        INLINE: Injection decisions (cache/seed injection, VAE blending)
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
        seed_image: Optional[Path] = None,
        injection_vae=None  # Dedicated LatentEncoder for injection blending (zero contention)
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
            injection_vae: Optional dedicated LatentEncoder for injection blending.
                          If provided, eliminates lock contention with interpolation.
        """
        self.buffer = frame_buffer
        self.generator = generator
        self.vae_access = vae_access
        self.prompt_manager = prompt_manager
        self.cache = cache_manager
        self.similarity_manager = similarity_manager
        self.config = config
        self.injection_vae = injection_vae  # Dedicated VAE for injection (no contention)
        
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
        
        self.injection_strategy = None

        # Initialize injection strategy (always enabled when cache/similarity available)
        if cache_manager and similarity_manager:
            # Use dedicated injection VAE if available (zero contention with interpolation)
            # Otherwise fall back to shared VAE access (with lock)
            if injection_vae:
                self.injection_strategy = CacheInjectionStrategy(
                    config=self.config,
                    cache_manager=cache_manager,
                    similarity_manager=similarity_manager,
                    latent_encoder=injection_vae,  # Dedicated VAE - no lock needed!
                    buffer=frame_buffer
                )
                logger.info("Injection strategy initialized (dedicated VAE - zero contention)")
            else:
                self.injection_strategy = CacheInjectionStrategy(
                    config=self.config,
                    cache_manager=cache_manager,
                    similarity_manager=similarity_manager,
                    vae_access=vae_access,  # Shared VAE with lock (legacy path)
                    buffer=frame_buffer
                )
                logger.info("Injection strategy initialized (shared VAE with lock)")
        
        # === Denoising State Machine (Phase 2) ===
        # Detect if using CombinatorialPromptSystem (has should_mutate method)
        # NOTE: Must be set early as fresh_buffer checks this
        self.use_combinatorial = hasattr(self.prompt_manager, 'should_mutate')
        
        # === Fresh Frame Buffer (for template-aware seed injection) ===
        # Only initialized when using CombinatorialPromptSystem
        self.fresh_buffer: Optional[FreshFrameBuffer] = None
        if self.use_combinatorial:
            self.fresh_buffer = FreshFrameBuffer(
                generator=generator,
                prompt_system=prompt_manager,
                config=config
            )
            logger.info("Fresh frame buffer initialized (pre-generation enabled)")
        
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
        
        # === INTERVENTION STATS (Phase 4 - for tuning) ===
        # Track different intervention types to understand system behavior
        self.forced_mutation_count = 0      # Collapse-triggered mutations
        self.template_switch_count = 0      # Full template switches (seed injection)
        
        # Injection frequency tracking (for seed forcing)
        self.recent_cache_injections = deque(maxlen=10)

        # === Chronicle (optional observer, attached by DreamController) ===
        # Events are decided for next_kf before that keyframe completes, so
        # they're staged here by keyframe number and drained at completion.
        self.chronicle = None  # ChronicleRecorder | None
        self.chronicle_events: Dict[int, list] = {}
        
        # Denoise values for DRIFT/BEND modes
        fresh_config = self.config.get('fresh_generation', {})
        denoising_config = fresh_config.get('denoising', {})
        
        # Support both nested dict and flat config formats
        if isinstance(denoising_config, dict):
            self.denoise_drift = denoising_config.get('drift', 0.20)
            self.denoise_bend = denoising_config.get('bend', 0.50)
            self.bend_duration = denoising_config.get('bend_frames', 4)
        else:
            # Fall back to hybrid keyframe_denoise for drift
            self.denoise_drift = self.config['generation']['hybrid'].get('keyframe_denoise', 0.20)
            self.denoise_bend = 0.50  # Default bend denoise
            self.bend_duration = 4
        
        # === MANUAL BYPASS MODE (Simple Frame Counting) ===
        # Bypasses all complex metrics/convergence detection with simple modulo triggers
        manual_bypass_config = self.config['generation']['cache'].get('manual_bypass', {})
        self.manual_bypass_enabled = manual_bypass_config.get('enabled', False)
        self.manual_bypass_mutation_interval = manual_bypass_config.get('mutation_interval', 5)
        self.manual_bypass_cache_interval = manual_bypass_config.get('cache_injection_interval', 25)
        self.manual_bypass_template_interval = manual_bypass_config.get('template_swap_interval', 75)
        
        if self.manual_bypass_enabled:
            logger.info("=" * 60)
            logger.info("MANUAL BYPASS MODE ENABLED")
            logger.info("  All adaptive metrics/convergence detection BYPASSED")
            logger.info(f"  Mutation every {self.manual_bypass_mutation_interval} frames (% {self.manual_bypass_mutation_interval})")
            logger.info(f"  Cache injection every {self.manual_bypass_cache_interval} frames (% {self.manual_bypass_cache_interval})")
            logger.info(f"  Template swap every {self.manual_bypass_template_interval} frames (% {self.manual_bypass_template_interval})")
            logger.info("=" * 60)
        
        if self.use_combinatorial:
            logger.info("Denoising state machine ENABLED (CombinatorialPromptSystem detected)")
            logger.info(f"  - DRIFT denoise: {self.denoise_drift}")
            logger.info(f"  - BEND denoise: {self.denoise_bend}")
            logger.info(f"  - BEND duration: {self.bend_duration} frames")
        
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
            
            # === FRESH FRAME BUFFER POPULATION ===
            # Populate the entire fresh frame buffer before starting generation
            # This generates one txt2img frame per template
            if self.fresh_buffer:
                logger.info("=" * 60)
                logger.info("Populating fresh frame buffer (required before generation)...")
                logger.info("=" * 60)
                await self.fresh_buffer.populate_all()
                logger.info("Fresh frame buffer ready!")
            
            # Bootstrap: Get first frame from fresh buffer (txt2img)
            if self.fresh_buffer:
                logger.info("Bootstrap: Getting initial frame from fresh buffer...")
                bootstrap_frame = await self.fresh_buffer.select_and_consume()
                self.current_image_path = bootstrap_frame.path
                
                # Switch prompt system to this template/components
                self.prompt_manager.switch_template(
                    bootstrap_frame.template_id,
                    bootstrap_frame.components
                )
                logger.info(f"  Template: '{bootstrap_frame.template_id}'")
                logger.info(f"  Components: {bootstrap_frame.components}")
            else:
                # No fresh buffer - this shouldn't happen in normal operation
                # Fresh buffer is required when using CombinatorialPromptSystem
                logger.error(
                    "No fresh buffer configured! "
                    "FreshFrameBuffer is required for CombinatorialPromptSystem. "
                    "Ensure use_combinatorial mode is enabled."
                )
                self.running = False
                return
            
            # Register as keyframe 1
            logger.info(f"Bootstrap: Registering as keyframe 1")
            self.current_keyframe_num = 1
            sequence_num = self.buffer.register_keyframe(1)
            self.buffer.mark_ready(sequence_num, self.current_image_path)
            
            # Track sequence number for this keyframe
            self.keyframe_sequences[1] = sequence_num
            
            # Encode for interpolation
            try:
                latent = await self.vae_access.encode_async(
                    self.current_image_path,
                    for_interpolation=True
                )
                # Store in interpolation worker cache
                self.interpolation_worker.keyframe_latents[1] = latent
                self.interpolation_worker.keyframe_paths[1] = self.current_image_path
                logger.info("  Bootstrap frame encoded to latent for interpolation")
            except Exception as e:
                logger.error(f"Failed to encode bootstrap frame: {e}")
                # Continue anyway - we can still generate keyframes even without initial latent
                # Interpolation for 1->2 may fail but subsequent cycles should work
            
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
            
            # Get prompt and negative prompt based on system type
            if self.use_combinatorial:
                bootstrap_prompt = self.prompt_manager.get_next_prompt()
                bootstrap_negative = self.prompt_manager.get_negative_prompt()
            else:
                bootstrap_prompt = self.prompt_manager.get_next_prompt()
                bootstrap_negative = self.prompt_manager.get_negative_prompt() if hasattr(self.prompt_manager, 'get_negative_prompt') else None
            
            # Submit keyframe 2 generation (DRIFT mode for bootstrap)
            await self.keyframe_worker.submit_request(
                current_image=self.current_image_path,
                keyframe_num=2,
                sequence_num=kf2_seq,
                prompt=bootstrap_prompt,
                negative_prompt=bootstrap_negative,
                denoise=self.denoise_drift,
                generation_mode="drift"
            )
            logger.info(f"  Submitted keyframe 2 generation request (DRIFT, denoise={self.denoise_drift:.2f})")
            
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
    
    def _chronicle_note(self, kf_num: int, kind: str, detail: str = "") -> None:
        """
        Stage a chronicle event for a keyframe that hasn't completed yet.

        No-op when no recorder is attached. Never raises. The staged events
        are drained (popped) when keyframe kf_num completes in _coordinate.
        """
        if self.chronicle is None:
            return
        try:
            self.chronicle_events.setdefault(kf_num, []).append(
                {"kind": kind, "detail": detail}
            )
            # Bound the staging dict: keyframes that failed/never completed
            # would otherwise leak their staged events forever
            if len(self.chronicle_events) > 50:
                for old_kf in sorted(self.chronicle_events.keys())[:-25]:
                    del self.chronicle_events[old_kf]
        except Exception:
            pass

    def _chronicle_component_diff(self, before: dict, after: dict) -> str:
        """Human-readable diff of prompt components, e.g. "color_logic: 'a' -> 'b'"."""
        try:
            changes = [
                f"{cat}: '{before.get(cat, '?')}' -> '{word}'"
                for cat, word in after.items()
                if before.get(cat) != word
            ]
            return ", ".join(changes)
        except Exception:
            return ""

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
                    # Fresh frame buffer handles its own regeneration in background
                    if self.fresh_buffer:
                        stats = self.fresh_buffer.get_stats()
                        ready = stats.get('ready_count', 0)
                        total = stats.get('total_templates', 0)
                        logger.debug(
                            f"  System throttled ({seconds_buffered:.1f}s / {target_seconds}s), "
                            f"fresh buffer: {ready}/{total} ready"
                        )
                    else:
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
                    original_prompt = result.get('prompt')  # Get cached prompt for retry
                    logger.error(f"[FAIL] Keyframe {kf_num} failed after {retries} retries: {error}")
                    
                    # Mark task done
                    self.keyframe_worker.result_queue.task_done()
                    
                    # === RECOVERY: Skip this keyframe and its pending interpolations ===
                    await self._handle_keyframe_failure(kf_num, sequence_num, original_prompt)
                    continue
                
                kf_path = result['path']
                prompt = result['prompt']
                gen_time = result.get('generation_time', 0.0)
                
                logger.info(f"[OK] Keyframe {kf_num} completed: {kf_path.name}")
                
                # === 2. Mark Keyframe Ready ===
                if sequence_num is not None:
                    self.buffer.mark_ready(sequence_num, kf_path, prompt=prompt)
                    logger.debug(f"  Marked keyframe {kf_num} ready (seq {sequence_num})")
                else:
                    # Fallback: look up sequence number
                    sequence_num = self.keyframe_sequences.get(kf_num)
                    if sequence_num is not None:
                        self.buffer.mark_ready(sequence_num, kf_path, prompt=prompt)
                    else:
                        logger.error(f"  Cannot find sequence number for keyframe {kf_num}!")
                
                # Update current state
                self.current_image_path = kf_path
                self.current_keyframe_num = kf_num

                # Mark task done
                self.keyframe_worker.result_queue.task_done()

                # === Chronicle: record this keyframe (fire-and-forget) ===
                if self.chronicle is not None:
                    try:
                        pm = self.prompt_manager
                        await self.chronicle.on_keyframe(
                            keyframe_num=kf_num,
                            sequence_num=sequence_num if sequence_num is not None else -1,
                            prompt=prompt,
                            negative=pm.get_current_negative() if hasattr(pm, 'get_current_negative') else "",
                            template_id=pm.get_current_template_id() if hasattr(pm, 'get_current_template_id') else "",
                            components=pm.get_current_components() if hasattr(pm, 'get_current_components') else {},
                            events=self.chronicle_events.pop(kf_num, []),
                            image_path=kf_path,
                        )
                    except Exception:
                        logger.debug("Chronicle hook failed", exc_info=True)

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
                    has_interp = (kf_num, next_kf) in self.buffer.registered_interp_pairs
                    
                    if not has_interp:
                        injection_interp_count = self.config['generation']['hybrid'].get(
                            'injection_interpolation_frames',
                            self.config['generation']['hybrid']['interpolation_frames']
                        )
                        interp_seqs = self.buffer.register_interpolations(
                            kf_num, next_kf,
                            injection_interp_count
                        )
                        logger.info(f"  Pre-registered injection interpolations {kf_num}->{next_kf}: {injection_interp_count} frames, seq {interp_seqs[0]}-{interp_seqs[-1]}")
                    
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
                            self.template_switch_count += 1
                            new_template = (
                                self.prompt_manager.get_current_template_id()
                                if hasattr(self.prompt_manager, 'get_current_template_id') else ""
                            )
                            self._chronicle_note(next_kf, "seed_injection")
                            self._chronicle_note(
                                next_kf, "template_switch", f"-> {new_template}"
                            )
                        else:
                            self.last_cache_injection_kf = next_kf
                            self.cache_injections += 1
                            self.recent_cache_injections.append(True)
                            self._chronicle_note(next_kf, "cache_injection")
                        
                        logger.info(f"  [OK] Injection completed, proceeding to next iteration")
                        continue
                    else:
                        logger.warning(f"  Injection failed, falling back to normal generation")
                        # Fall through to normal generation
                else:
                    # Track no injection
                    self.recent_cache_injections.append(False)
                
                # === 8. Normal Generation - Pre-register Next Cycle ===
                
                # === DENOISING STATE MACHINE (Phase 2) ===
                # Check for mutations and determine denoise based on DRIFT/BEND mode
                generation_mode = "drift"
                denoise = self.denoise_drift
                negative_prompt = None
                
                if self.use_combinatorial:
                    # === MANUAL BYPASS: Force mutation on interval ===
                    should_force_mutation_bypass = (
                        self.manual_bypass_enabled and
                        next_kf > 0 and
                        next_kf % self.manual_bypass_mutation_interval == 0 and
                        # Don't force mutation if we're already doing a bigger intervention
                        next_kf % self.manual_bypass_cache_interval != 0 and
                        next_kf % self.manual_bypass_template_interval != 0
                    )
                    
                    if should_force_mutation_bypass:
                        logger.info(
                            f"  [MANUAL_BYPASS] Frame {next_kf} triggers FORCED MUTATION "
                            f"(% {self.manual_bypass_mutation_interval} == 0)"
                        )
                        components_before = self.prompt_manager.get_current_components()
                        self.prompt_manager.mutate()
                        self.forced_mutation_count += 1
                        self._chronicle_note(
                            next_kf, "forced_mutation",
                            self._chronicle_component_diff(
                                components_before,
                                self.prompt_manager.get_current_components()
                            )
                        )
                        logger.info(f"  [MUTATION] Manual bypass forced mutation, entering BEND mode")
                    # Check if should mutate components (normal adaptive path)
                    elif self.prompt_manager.should_mutate():
                        components_before = self.prompt_manager.get_current_components()
                        self.prompt_manager.mutate()
                        self._chronicle_note(
                            next_kf, "mutation",
                            self._chronicle_component_diff(
                                components_before,
                                self.prompt_manager.get_current_components()
                            )
                        )
                        logger.info(f"  [MUTATION] Component mutated, entering BEND mode")
                    
                    # Get mode and denoise
                    if self.prompt_manager.is_in_bend_mode():
                        generation_mode = "bend"
                        denoise = self.denoise_bend
                        logger.info(f"  [BEND] Using high denoise ({denoise:.2f}) for prompt transition")
                    else:
                        generation_mode = "drift"
                        denoise = self.denoise_drift
                    
                    # Get prompts from combinatorial system
                    next_prompt = self.prompt_manager.get_next_prompt()
                    negative_prompt = self.prompt_manager.get_negative_prompt()
                else:
                    # Legacy path: use old PromptManager
                    next_prompt = self.prompt_manager.get_next_prompt()
                    negative_prompt = self.prompt_manager.get_negative_prompt() if hasattr(self.prompt_manager, 'get_negative_prompt') else None
                
                # Pre-register ONLY ONE cycle ahead:
                # 1. Interpolations current -> next (if not already done)
                # 2. Keyframe next
                # Don't register further ahead - causes duplicate registrations!
                
                # Check if interpolations kf_num -> next_kf already registered
                has_interp = (kf_num, next_kf) in self.buffer.registered_interp_pairs
                
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
                
                # Submit keyframe generation with denoising state
                logger.info(f"  Submitting keyframe {next_kf} ({generation_mode.upper()}, denoise={denoise:.2f})")
                await self.keyframe_worker.submit_request(
                    current_image=kf_path,
                    keyframe_num=next_kf,
                    sequence_num=next_seq,
                    prompt=next_prompt,
                    negative_prompt=negative_prompt,
                    denoise=denoise,
                    generation_mode=generation_mode
                )
                
                # === PERIODIC STATS (every 10 keyframes) ===
                if next_kf % 10 == 0 and self.use_combinatorial:
                    prompt_stats = self.prompt_manager.get_stats() if hasattr(self.prompt_manager, 'get_stats') else {}
                    fresh_stats = self.fresh_buffer.get_stats() if self.fresh_buffer else {}
                    
                    logger.info("=" * 60)
                    logger.info(f"[STATS] Keyframe {next_kf}")
                    logger.info(f"  Prompt System:")
                    logger.info(f"    Template: {prompt_stats.get('current_template', 'N/A')}")
                    logger.info(f"    Total mutations: {prompt_stats.get('total_mutations', 0)}")
                    logger.info(f"    Frames since mutation: {prompt_stats.get('frames_since_mutation', 0)}")
                    logger.info(f"    In BEND mode: {prompt_stats.get('in_bend_mode', False)}")
                    if fresh_stats:
                        logger.info(f"  Fresh Buffer:")
                        logger.info(f"    Ready: {fresh_stats.get('is_ready', False)}")
                        logger.info(f"    Generated: {fresh_stats.get('total_generated', 0)}")
                        logger.info(f"    Consumed: {fresh_stats.get('total_consumed', 0)}")
                        logger.info(f"    Avg gen time: {fresh_stats.get('avg_generation_time', 0):.2f}s")
                    logger.info(f"  Cache injections: {self.cache_injections}")
                    logger.info("=" * 60)
                
                # === INTERVENTION STATS (every 50 keyframes) ===
                # Logs intervention breakdown for tuning collapse prevention parameters
                if next_kf % 50 == 0 and self.use_combinatorial:
                    logger.info("=" * 60)
                    logger.info(f"[INTERVENTION_STATS] Keyframe {next_kf}")
                    logger.info(f"  Forced mutations: {self.forced_mutation_count}")
                    logger.info(f"  Cache injections: {self.cache_injections}")
                    logger.info(f"  Template switches: {self.template_switch_count}")

                    # Calculate ratios for tuning guidance
                    total_interventions = (
                        self.forced_mutation_count + 
                        self.cache_injections + 
                        self.template_switch_count
                    )
                    if total_interventions > 0:
                        mutation_pct = self.forced_mutation_count / total_interventions * 100
                        cache_pct = self.cache_injections / total_interventions * 100
                        switch_pct = self.template_switch_count / total_interventions * 100
                        logger.info(
                            f"  Ratios: mutations={mutation_pct:.0f}%, "
                            f"cache={cache_pct:.0f}%, switches={switch_pct:.0f}%"
                        )
                        logger.info(
                            f"  Target: mostly mutations, some cache, rare switches"
                        )
                    logger.info("=" * 60)
                
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
        - Manual bypass mode (simple frame counting)
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
        
        # === MANUAL BYPASS MODE ===
        # Simple frame counting bypasses all adaptive metrics
        if self.manual_bypass_enabled:
            # Priority order: template swap > cache injection > mutation (handled separately)
            # Check template swap first (most dramatic intervention)
            if kf_num > 0 and kf_num % self.manual_bypass_template_interval == 0:
                logger.info(
                    f"[MANUAL_BYPASS] Frame {kf_num} triggers TEMPLATE SWAP "
                    f"(% {self.manual_bypass_template_interval} == 0)"
                )
                return True, 'seed'
            
            # Check cache injection (medium intervention)
            if kf_num > 0 and kf_num % self.manual_bypass_cache_interval == 0:
                # Only if cache has frames
                if self.cache.size() > 0:
                    logger.info(
                        f"[MANUAL_BYPASS] Frame {kf_num} triggers CACHE INJECTION "
                        f"(% {self.manual_bypass_cache_interval} == 0)"
                    )
                    return True, 'cache'
                else:
                    logger.debug(
                        f"[MANUAL_BYPASS] Frame {kf_num} would trigger cache injection "
                        f"but cache is empty - skipping"
                    )
            
            # Mutation is handled separately in _coordinate() since it's not an injection
            # It just forces a mutation before normal generation
            return False, None
        
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
                f"Adaptive interventions now ACTIVE. "
                f"Cache size: {self.cache.size() if self.cache else 0}"
            )

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
        
        probability_cache = (
            not on_cooldown and
            self.cache.size() > 0 and
            random.random() < self.current_injection_rate
        )

        if probability_cache:
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
            
            # === SEED INJECTION (via Fresh Frame Buffer) ===
            if injection_type == 'seed':
                logger.info(f"  -> Injecting FRESH frame (keyframe {keyframe_num})")
                
                # Get fresh frame from buffer (blocks if needed)
                if not self.fresh_buffer:
                    logger.error("No fresh buffer configured - cannot inject seed frame!")
                    return None
                
                fresh_frame = await self.fresh_buffer.select_and_consume()
                
                # Use pre-generated fresh frame with template switch
                target_path = fresh_frame.path
                new_template_id = fresh_frame.template_id
                new_components = fresh_frame.components
                buffer_age = time.time() - fresh_frame.generated_at
                
                logger.info(f"  [FRESH] Using pre-generated frame:")
                logger.info(f"    Template: '{new_template_id}'")
                logger.info(f"    Components: {new_components}")
                logger.info(f"    Buffer age: {buffer_age:.1f}s")
                logger.info(f"    Prompt: {fresh_frame.prompt[:80]}...")
                
                # === VAE INTERPOLATION: Blend current frame toward fresh frame ===
                # This creates smoother visual transitions than direct copy
                keyframe_path = self.buffer.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
                
                if current_path and current_path.exists():
                    try:
                        # Encode both frames to latent space
                        current_latent = await self.vae_access.encode_async(
                            current_path,
                            for_interpolation=True
                        )
                        fresh_latent = await self.vae_access.encode_async(
                            fresh_frame.path,
                            for_interpolation=True
                        )
                        
                        # Blend heavily toward fresh frame (default 85%)
                        # This preserves most of the fresh aesthetic while smoothing the transition
                        blend_weight = self.config['generation']['cache'].get('seed_blend_weight', 0.85)
                        blended_latent = (
                            fresh_latent * blend_weight +
                            current_latent * (1.0 - blend_weight)
                        )
                        
                        # Decode blended result
                        blended_image = await self.vae_access.decode_async(
                            blended_latent,
                            upscale_to_target=True
                        )
                        
                        # Save to keyframe location
                        blended_image.save(keyframe_path, "PNG", optimize=False, compress_level=1)
                        
                        logger.info(
                            f"  [TEMPLATE_SWITCH] Interpolated blend: "
                            f"{blend_weight*100:.0f}% fresh + {(1-blend_weight)*100:.0f}% current"
                        )
                        
                        target_path = keyframe_path
                        
                    except Exception as e:
                        logger.error(f"Interpolation failed, falling back to direct copy: {e}")
                        # Fallback to direct copy
                        shutil.copy2(fresh_frame.path, keyframe_path)
                        target_path = keyframe_path
                else:
                    # No current frame to blend from (e.g., bootstrap), use direct copy
                    shutil.copy2(fresh_frame.path, keyframe_path)
                    target_path = keyframe_path
                
                # Perform coordinated template switch
                old_template_id = self.prompt_manager.get_current_template_id() if self.use_combinatorial else None
                
                logger.info(f"  [TEMPLATE_SWITCH] '{old_template_id}' → '{new_template_id}'")
                
                # 1. Switch prompt system to new template
                if self.use_combinatorial:
                    self.prompt_manager.switch_template(new_template_id, new_components)
                    logger.info(f"    ✓ Prompt system switched")
                
                # 2. Switch cache manager (archive old, potentially restore if returning)
                if self.cache:
                    self.cache.switch_template(new_template_id)
                    logger.info(f"    ✓ Cache manager switched (old cache archived)")

                # Note: Buffer automatically triggers regeneration for consumed template
                
                metadata = {
                    'type': 'fresh_frame_injection',
                    'template_id': new_template_id,
                    'old_template_id': old_template_id,
                    'prompt': fresh_frame.prompt
                }
                
                if target_path:
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
                        prompt=metadata.get('prompt', 'seed_injection'),
                        metadata={'denoise': 0.0, 'type': metadata['type'], 'injection': True}
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
                
                result = await self.injection_strategy.inject_dissimilar_keyframe(
                    current_image_path=current_path,
                    target_keyframe_num=keyframe_num,
                )

                if result:
                    target_path, metadata = result

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
    
    async def _handle_keyframe_failure(
        self, 
        kf_num: int, 
        sequence_num: Optional[int],
        original_prompt: Optional[str] = None
    ) -> None:
        """
        Handle a failed keyframe by cleaning up orphaned interpolations and continuing
        
        Recovery strategy:
        1. Mark the failed keyframe as skipped/failed in buffer
        2. Remove any pending interpolations that depend on this keyframe
        3. Use the PREVIOUS keyframe as the current image for next generation
        4. Re-register a new keyframe and its interpolations
        5. Submit the new keyframe request to worker (using original prompt if available)
        6. Fix display position if stuck on deleted frames
        
        Args:
            kf_num: The failed keyframe number
            sequence_num: The sequence number in buffer (if known)
            original_prompt: The prompt that was used for the failed keyframe (for retry consistency)
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
        # Look for the most recent ready keyframe
        last_good_kf = kf_num - 1
        last_good_path = None
        last_good_seq = None
        
        while last_good_kf > 0:
            seq = self.keyframe_sequences.get(last_good_kf)
            if seq is not None and seq in self.buffer.frames:
                frame = self.buffer.frames[seq]
                if frame.is_ready() and frame.file_path and frame.file_path.exists():
                    last_good_path = frame.file_path
                    last_good_seq = seq
                    break
            last_good_kf -= 1
        
        if last_good_path:
            self.current_image_path = last_good_path
            self.current_keyframe_num = last_good_kf
            logger.info(f"  Falling back to keyframe {last_good_kf}: {last_good_path.name}")
        else:
            # No previous keyframe - get fresh frame from buffer
            logger.warning(f"  No previous keyframe found!")
            
            if self.fresh_buffer and self.fresh_buffer.is_ready():
                logger.info(f"  Getting fresh frame from buffer for recovery...")
                try:
                    # This is sync context in async handler - need to handle carefully
                    # The fresh buffer should have frames ready from startup
                    fresh_frame = self.fresh_buffer.peek()
                    if fresh_frame:
                        self.current_image_path = fresh_frame.path
                        self.current_keyframe_num = 0
                        logger.info(f"  Using fresh frame: {self.current_image_path.name}")
                    else:
                        logger.error(f"  Fresh buffer empty - cannot recover!")
                        raise RuntimeError("Keyframe failure recovery failed: no frames available")
                except Exception as e:
                    logger.error(f"  Failed to get fresh frame: {e}")
                    raise RuntimeError(f"Keyframe failure recovery failed: {e}")
            else:
                logger.error(f"  No fresh buffer available - cannot recover!")
                raise RuntimeError(
                    "Keyframe failure recovery failed: no previous keyframe and no fresh buffer. "
                    "This should not happen if fresh buffer was populated at startup."
                )
        
        # === 4. Fix display position if stuck on deleted frames ===
        current_display = self.buffer.display_sequence_num
        if current_display in orphaned_seqs or (sequence_num and current_display >= sequence_num):
            # Display is pointing at deleted/orphaned frames
            if last_good_seq is not None:
                logger.warning(f"  Display was at seq {current_display} (deleted) - resetting to {last_good_seq}")
                self.buffer.display_sequence_num = last_good_seq
            else:
                # Reset to beginning
                logger.warning(f"  Display reset to beginning")
                self.buffer.display_sequence_num = 0
        
        # === 5. Register NEW keyframe cycle ===
        next_kf = self.current_keyframe_num + 1
        interp_frames = self.config['generation']['hybrid']['interpolation_frames']
        
        # Register interpolations first (they come before the keyframe in sequence)
        interp_seqs = self.buffer.register_interpolations(
            self.current_keyframe_num, 
            next_kf, 
            interp_frames
        )
        logger.info(f"  Registered {interp_frames} interpolations: seq {interp_seqs[0]}-{interp_seqs[-1]}")
        
        # Register the new keyframe
        new_kf_seq = self.buffer.register_keyframe(next_kf)
        self.keyframe_sequences[next_kf] = new_kf_seq
        logger.info(f"  Registered KF{next_kf} at seq {new_kf_seq}")
        
        # === 6. Submit the new keyframe request ===
        # Use the original prompt if available (for retry consistency)
        # Otherwise get the next prompt from the rotation
        if original_prompt:
            prompt = original_prompt
            logger.info(f"  Using original prompt for retry (preserving consistency)")
        else:
            prompt = self.prompt_manager.get_next_prompt()
            logger.info(f"  Using new prompt from rotation")
        
        # Get negative prompt and denoise for recovery
        negative_prompt = None
        if self.use_combinatorial:
            negative_prompt = self.prompt_manager.get_negative_prompt()
        elif hasattr(self.prompt_manager, 'get_negative_prompt'):
            negative_prompt = self.prompt_manager.get_negative_prompt()
        
        await self.keyframe_worker.submit_request(
            current_image=self.current_image_path,
            keyframe_num=next_kf,
            sequence_num=new_kf_seq,
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=self.denoise_drift,  # Use DRIFT for recovery
            generation_mode="drift"
        )
        logger.info(f"  Submitted new KF{next_kf} request (recovery, DRIFT mode)")
        
        logger.info(f"  Recovery complete. Generating: KF{next_kf}")
        logger.warning(f"=== RECOVERY COMPLETE ===")
    
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
            "is_running": self.running,
            # === INTERVENTION STATS (Phase 4) ===
            "forced_mutation_count": self.forced_mutation_count,
            "template_switch_count": self.template_switch_count
        }
        
        # Add denoising state machine stats (Phase 2)
        if self.use_combinatorial:
            try:
                prompt_stats = self.prompt_manager.get_stats()
                stats.update({
                    "prompt_template": prompt_stats.get("current_template"),
                    "prompt_mutations": prompt_stats.get("total_mutations", 0),
                    "prompt_in_bend_mode": prompt_stats.get("in_bend_mode", False),
                    "prompt_frames_since_mutation": prompt_stats.get("frames_since_mutation", 0)
                })
            except Exception as e:
                logger.debug(f"Failed to get prompt stats: {e}")
        
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
        
        # Add worker queue depths
        if self.keyframe_worker:
            stats["keyframe_queue_depth"] = self.keyframe_worker.request_queue.qsize()
        if self.interpolation_worker:
            stats["interpolation_queue_depth"] = self.interpolation_worker.pair_queue.qsize()
        if self.cache_worker:
            stats["cache_queue_depth"] = self.cache_worker.analysis_queue.qsize()
        
        # Add fresh frame buffer stats
        if self.fresh_buffer:
            fresh_stats = self.fresh_buffer.get_stats()
            stats.update({
                "fresh_buffer_ready": fresh_stats.get("is_ready", False),
                "fresh_buffer_template": fresh_stats.get("buffered_template"),
                "fresh_frames_generated": fresh_stats.get("total_generated", 0),
                "fresh_frames_consumed": fresh_stats.get("total_consumed", 0)
            })
        
        return stats

