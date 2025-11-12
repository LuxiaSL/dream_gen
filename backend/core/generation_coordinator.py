"""
Generation Coordinator - Orchestrates frame generation

This module coordinates the generation of keyframes and interpolations,
ensuring a continuous flow of frames into the buffer while maintaining
proper sequencing and timing.

Key responsibilities:
- Continuously generate keyframes via img2img
- Trigger interpolation generation when new keyframe pairs exist
- Monitor buffer status and adjust generation pace
- Update status for daemon/Rainmeter
- Manage generation lifecycle
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
import asyncio
import torch

from .frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


class GenerationCoordinator:
    """
    Orchestrates frame generation to maintain buffer
    
    Coordinates keyframe generation (img2img) and interpolation generation
    (VAE latent interpolation) to keep the frame buffer filled with ready
    frames for smooth playback.
    
    Usage:
        coordinator = GenerationCoordinator(
            frame_buffer=buffer,
            hybrid_generator=hybrid_gen,
            config=config
        )
        
        # Run generation loop
        await coordinator.run()
    """
    
    def __init__(
        self,
        frame_buffer: FrameBuffer,
        generator,
        latent_encoder,
        prompt_manager,
        config: Dict[str, Any],
        cache_manager=None,
        similarity_manager=None
    ):
        """
        Initialize generation coordinator
        
        Args:
            frame_buffer: FrameBuffer instance
            generator: DreamGenerator instance
            latent_encoder: LatentEncoder instance
            prompt_manager: PromptManager instance
            config: Configuration dictionary
            cache_manager: CacheManager instance (optional, for cache injection)
            similarity_manager: DualMetricSimilarityManager instance (optional, for embeddings)
        """
        self.buffer = frame_buffer
        self.generator = generator
        self.latent_encoder = latent_encoder
        self.prompt_manager = prompt_manager
        self.config = config
        self.cache = cache_manager
        self.similarity_manager = similarity_manager
        
        # Keyframe storage for interpolation
        self.keyframe_latents: Dict[int, Any] = {}  # keyframe_num -> torch.Tensor
        self.keyframe_paths: Dict[int, Path] = {}  # keyframe_num -> Path
        self.slerp_precomputed: Dict = {}  # (start_kf, end_kf) -> precomputed params
        
        # State
        self.running = False
        self.paused = False
        self.current_keyframe_num = 0
        self.current_image_path = None
        
        # Statistics
        self.keyframes_generated = 0
        self.interpolations_generated = 0
        self.cache_injections = 0
        self.generation_times = []
        
        # Mode collapse prevention
        self.collapse_detector = None
        self.injection_strategy = None
        self.current_injection_rate = config['generation']['cache']['injection_probability']
        
        # Track recent cache injections for seed forcing logic
        self.recent_cache_injections = deque(maxlen=20)  # Last 20 keyframes
        
        # Injection cooldown tracking (prevents loops)
        self.last_cache_injection_kf = -999  # Keyframe number of last cache injection
        self.last_seed_injection_kf = -999   # Keyframe number of last seed injection
        
        # Initialize mode collapse detection if enabled
        if cache_manager and similarity_manager and config['generation']['cache'].get('collapse_detection', True):
            try:
                from cache.collapse_detector import ModeCollapseDetector
                from cache.injection_strategy import CacheInjectionStrategy
                
                cache_config = config['generation']['cache']
                
                # Get dual-metric thresholds if available
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
                
                self.injection_strategy = CacheInjectionStrategy(
                    config=config,
                    cache_manager=cache_manager,
                    similarity_manager=similarity_manager,
                    latent_encoder=latent_encoder,
                    buffer=frame_buffer
                )
                
                logger.info("Mode collapse prevention enabled")
                logger.info(f"  Similarity method: {cache_config.get('similarity_method', 'dual_metric')}")
                logger.info(f"  Collapse detection: active")
                logger.info(f"  Injection mode: {config['generation']['cache'].get('injection_mode', 'dissimilar')}")
                
            except Exception as e:
                logger.error(f"Failed to initialize mode collapse prevention: {e}")
                logger.info("Continuing without collapse prevention")
        
        logger.info("GenerationCoordinator initialized")
        if self.cache:
            logger.info(f"  Cache injection enabled (probability: {config['generation']['cache']['injection_probability']})")
    
    def cleanup_old_keyframes(self, keep_recent: int = 5) -> None:
        """
        Clean up old keyframe latents to save memory
        
        Args:
            keep_recent: Number of recent keyframes to keep
        """
        if len(self.keyframe_latents) <= keep_recent:
            return
        
        # Sort keyframes by frame number
        keyframe_numbers = sorted(self.keyframe_latents.keys())
        
        # Delete old ones (keep_recent most recent)
        to_delete = keyframe_numbers[:-keep_recent]
        
        for frame_num in to_delete:
            if frame_num in self.keyframe_latents:
                del self.keyframe_latents[frame_num]
            if frame_num in self.keyframe_paths:
                del self.keyframe_paths[frame_num]
            # Also clean up precomputed slerp params involving this keyframe
            keys_to_delete = [k for k in self.slerp_precomputed.keys() if frame_num in k]
            for k in keys_to_delete:
                del self.slerp_precomputed[k]
        
        logger.debug(f"Cleaned up {len(to_delete)} old keyframes from memory")
    
    def set_seed_image(self, seed_path: Path) -> None:
        """
        Set the initial seed image for generation
        
        Args:
            seed_path: Path to seed image
        """
        self.current_image_path = seed_path
        logger.info(f"Seed image set: {seed_path.name}")
    
    
    def add_keyframe_to_cache(self, keyframe_path: Path, prompt: str, denoise: float) -> bool:
        """
        Add a generated keyframe to the cache with dual-metric embedding
        
        Uses selective caching if mode collapse prevention is enabled.
        
        Args:
            keyframe_path: Path to generated keyframe
            prompt: Generation prompt used
            denoise: Denoise value used
            
        Returns:
            True if successfully added to cache
        """
        if not self.cache or not self.similarity_manager:
            return False
        
        try:
            # Encode image with dual-metric similarity system
            embedding = self.similarity_manager.encode_image(keyframe_path)
            
            if embedding is None:
                logger.warning(f"Failed to encode keyframe for cache: {keyframe_path.name}")
                return False
            
            # Check if should cache (selective caching for diversity)
            population_mode = self.config['generation']['cache'].get('population_mode', 'selective')
            
            should_cache = True
            if population_mode == 'selective' and self.cache.size() > 0:
                should_cache = self.cache.should_cache_frame(
                    embedding, 
                    force=False,
                    similarity_manager=self.similarity_manager
                )
                
                if not should_cache:
                    logger.debug(f"Skipping cache (frame not diverse enough)")
                    return False
            
            # Convert to serializable format if needed
            if isinstance(embedding, dict) and 'color' in embedding:
                # Dual-metric: convert to serializable
                embedding = self.similarity_manager.to_serializable(embedding)
            
            # Add to cache
            generation_params = {
                "denoise": denoise,
                "prompt": prompt,
                "model": self.config["generation"]["model"],
                "resolution": self.config["generation"]["resolution"]
            }
            
            cache_id = self.cache.add(
                image_path=keyframe_path,
                prompt=prompt,
                generation_params=generation_params,
                embedding=embedding
            )
            
            logger.debug(f"Added keyframe to cache: {cache_id} (total: {self.cache.size()})")
            
            # Log diversity stats periodically
            if self.config['generation']['cache'].get('log_diversity_stats', True):
                diversity_interval = self.config['generation']['cache'].get('diversity_check_interval', 10)
                if self.cache.size() % diversity_interval == 0:
                    diversity_stats = self.cache.get_diversity_stats(similarity_manager=self.similarity_manager)
                    
                    # Log dual-metric diversity stats
                    if 'diversity_score_color' in diversity_stats:
                        logger.info(
                            f"[CACHE_DIVERSITY] Color:{diversity_stats['diversity_score_color']:.3f}, "
                            f"Struct:{diversity_stats['diversity_score_struct']:.3f}, "
                            f"Size:{diversity_stats['cache_size']}"
                        )
                    else:
                        logger.info(
                            f"[CACHE_DIVERSITY] Score:{diversity_stats['diversity_score']:.3f}, "
                            f"Avg similarity:{diversity_stats['avg_pairwise_similarity']:.3f}, "
                            f"Size:{diversity_stats['cache_size']}"
                        )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add keyframe to cache: {e}", exc_info=True)
            return False
    
    async def run(self, check_interval: float = 0.5) -> None:
        """
        Main generation loop
        
        Continuously generates keyframes and interpolations to maintain buffer.
        
        Args:
            check_interval: Seconds between generation checks
        """
        self.running = True
        logger.info("Generation coordinator starting...")
        
        while self.running:
            try:
                # Skip if paused
                if self.paused:
                    await asyncio.sleep(check_interval)
                    continue
                
                # Check what needs to be generated
                # PRIORITY: Check interpolations first to keep sequences smooth!
                needs_interpolations = self.buffer.needs_interpolations()
                needs_keyframe = self.should_generate_keyframe()
                
                if needs_interpolations:
                    # PRIORITY: Fill in missing interpolations before generating more keyframes
                    # This ensures smooth, complete sequences
                    start_kf, end_kf = needs_interpolations
                    await self.generate_interpolations(start_kf, end_kf)
                
                elif needs_keyframe:
                    # Generate next keyframe only after interpolations are filled
                    await self.generate_keyframe()
                
                else:
                    # Buffer is adequately filled, wait a bit
                    await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                logger.info("Generation coordinator cancelled")
                break
            except Exception as e:
                logger.error(f"Error in generation loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Back off on error
        
        logger.info("Generation coordinator stopped")
    
    def should_generate_keyframe(self) -> bool:
        """
        Determine if we should generate another keyframe
        
        Returns:
            True if keyframe generation is needed
        """
        # Always need at least 2 keyframes to start interpolations
        if self.current_keyframe_num < 2:
            return True
        
        # Check if buffer needs more content
        status = self.buffer.get_buffer_status()
        
        # Generate keyframes to keep buffer ahead
        # Target: always have at least one complete cycle ahead
        frames_per_cycle = self.buffer.interpolation_frames + 1
        target_ahead = frames_per_cycle * 2  # 2 cycles ahead
        
        frames_ahead = self.buffer.next_sequence_num - self.buffer.display_sequence_num
        
        return frames_ahead < target_ahead
    
    async def generate_keyframe(self) -> bool:
        """
        Generate the next keyframe (or inject from cache/seed)
        
        Integrates mode collapse detection and adaptive injection strategies.
        
        Returns:
            True if successful
        """
        if self.current_image_path is None:
            logger.error("No current image for keyframe generation")
            return False
        
        self.current_keyframe_num += 1
        keyframe_num = self.current_keyframe_num
        
        logger.info(f"{'='*70}")
        logger.info(f"Generating KEYFRAME {keyframe_num}")
        
        # Register keyframe in buffer
        sequence_num = self.buffer.register_keyframe(keyframe_num)
        self.buffer.mark_generating(sequence_num)
        
        # Get prompt
        prompt = self.prompt_manager.get_next_prompt()
        logger.info(f"Prompt: {prompt[:60]}...")
        
        # === MODE COLLAPSE DETECTION & ADAPTIVE INJECTION ===
        collapse_result = None
        if self.collapse_detector and self.similarity_manager and self.current_image_path:
            try:
                # Encode current frame for collapse detection
                current_embedding = self.similarity_manager.encode_image(self.current_image_path)
                if current_embedding is not None:
                    collapse_result = self.collapse_detector.analyze_frame(current_embedding)
                    
                    # Record collapse status for adaptive seed injection
                    if self.injection_strategy:
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
                    # Three modes:
                    # 1. "none": baseline 15%
                    # 2. "scale_injection": gradually scale 15% -> 100% based on delta
                    # 3. "force_cache": force 100% (handled separately below)
                    baseline_prob = self.config['generation']['cache']['injection_probability']
                    
                    if collapse_result['action'] == 'scale_injection':
                        # Gradually scale from baseline to 100% as convergence increases
                        scaled_prob = collapse_result.get('scaled_injection_probability', 0.0)
                        self.current_injection_rate = baseline_prob + (1.0 - baseline_prob) * scaled_prob
                        logger.info(
                            f"[SCALING] Injection probability scaled to {self.current_injection_rate:.0%} "
                            f"(delta: {collapse_result['convergence_delta']:.3f})"
                        )
                    elif collapse_result['action'] == 'force_cache':
                        # Will be handled below with forced injection (bypasses probability)
                        pass
                    else:
                        # Normal baseline rate
                        self.current_injection_rate = baseline_prob
            except Exception as e:
                logger.error(f"Collapse detection failed: {e}", exc_info=True)
        
        # === SEED INJECTION (Adaptive or Forced) ===
        if self.injection_strategy:
            # Check cooldown (prevent seed injection loops)
            seed_cooldown = self.config['generation']['cache'].get('seed_injection_cooldown', 2)
            keyframes_since_seed = keyframe_num - self.last_seed_injection_kf
            
            if keyframes_since_seed <= seed_cooldown:
                logger.debug(f"Seed injection on cooldown ({keyframes_since_seed}/{seed_cooldown} keyframes)")
            else:
                # Check if we should force seed based on injection frequency
                # If we've had many cache injections recently, it means collapse is persistent
                recent_injection_count = sum(1 for x in self.recent_cache_injections if x)
                injection_frequency = recent_injection_count / len(self.recent_cache_injections) if self.recent_cache_injections else 0.0
                
                force_seed_threshold = self.config['generation']['cache'].get('force_seed_injection_frequency', 0.5)
                force_seed_from_frequency = injection_frequency > force_seed_threshold
                
                if force_seed_from_frequency:
                    logger.warning(
                        f"[EMERGENCY] High injection frequency ({injection_frequency:.0%}) -> forcing seed"
                    )
                
                # Or adaptive seed injection
                should_seed = force_seed_from_frequency or self.injection_strategy.should_inject_seed()
                
                if should_seed:
                    logger.info("  -> Attempting seed injection...")
                    result = self.injection_strategy.inject_seed_frame(
                        target_keyframe_num=keyframe_num,
                        current_image_path=self.current_image_path
                    )
                    
                    if result:
                        target_path, metadata = result
                        
                        # Track seed injection
                        self.last_seed_injection_kf = keyframe_num
                        
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
                        
                        # Update current image for next generation
                        self.current_image_path = target_path
                        
                        # Encode for VAE interpolation
                        if self.latent_encoder:
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                
                                latent = self.latent_encoder.encode(
                                    target_path,
                                    for_interpolation=True
                                )
                                self.keyframe_latents[keyframe_num] = latent
                                self.keyframe_paths[keyframe_num] = target_path
                                
                                logger.debug(f"  Encoded seed keyframe {keyframe_num} to latent")
                            except Exception as e:
                                logger.error(f"Failed to encode seed keyframe: {e}")
                        
                        # Add to cache (seeds are always diverse)
                        self.add_keyframe_to_cache(target_path, prompt, 0.0)
                        
                        logger.info(f"[OK] Keyframe {keyframe_num} from SEED ({metadata['type']})")
                        logger.info(f"     Saved to: {target_path.name}")
                        return True
        
        # === CACHE INJECTION (Dissimilar Strategy) ===
        # Three modes:
        # 1. FORCED: severe collapse detected, must inject (100%)
        # 2. SCALED: converging, probability scaled based on delta (15% -> 100%)
        # 3. BASELINE: normal random variance (15%)
        
        # Check cooldown (prevent cache injection loops)
        cache_cooldown = self.config['generation']['cache'].get('injection_cooldown', 3)
        keyframes_since_cache = keyframe_num - self.last_cache_injection_kf
        on_cooldown = keyframes_since_cache <= cache_cooldown
        
        force_cache = collapse_result and collapse_result['action'] == 'force_cache'
        probability_cache = (
            not force_cache and
            not on_cooldown and  # Respect cooldown for probability-based injections
            self.current_keyframe_num > 0 and
            self.cache and 
            self.similarity_manager and
            self.cache.size() > 0 and
            random.random() < self.current_injection_rate
        )
        
        # Force cache ignores cooldown (severe convergence)
        cache_injection_allowed = force_cache or probability_cache
        
        if on_cooldown and not force_cache:
            logger.debug(f"Cache injection on cooldown ({keyframes_since_cache}/{cache_cooldown} keyframes)")
        
        if cache_injection_allowed:
            if force_cache:
                logger.warning(f"  -> FORCING cache injection (severe convergence, 100%)")
            else:
                logger.info(f"  -> Attempting cache injection (probability: {self.current_injection_rate:.0%})")
            
            # Use modern dissimilar injection strategy
            if self.injection_strategy:
                injection_mode = self.config['generation']['cache'].get('injection_mode', 'dissimilar')
                
                if injection_mode == 'dissimilar':
                    # Extract which metric triggered (for smart selection)
                    collapse_trigger = None
                    if collapse_result and 'trigger_reason' in collapse_result:
                        trigger_reason = collapse_result['trigger_reason']
                        # Parse trigger reason to determine which metric
                        if 'COLOR' in trigger_reason and 'STRUCT' in trigger_reason:
                            collapse_trigger = "BOTH"
                        elif 'COLOR' in trigger_reason:
                            collapse_trigger = "COLOR"
                        elif 'STRUCT' in trigger_reason:
                            collapse_trigger = "STRUCTURAL"
                    
                    result = self.injection_strategy.inject_dissimilar_keyframe(
                        current_image_path=self.current_image_path,
                        target_keyframe_num=keyframe_num,
                        collapse_trigger=collapse_trigger
                    )
                    
                    if result:
                        target_path, metadata = result
                        
                        # Track cache injection
                        self.last_cache_injection_kf = keyframe_num
                        
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
                        
                        # Update current image for next generation
                        self.current_image_path = target_path
                        
                        # Encode for VAE interpolation
                        if self.latent_encoder:
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                
                                latent = self.latent_encoder.encode(
                                    target_path,
                                    for_interpolation=True
                                )
                                self.keyframe_latents[keyframe_num] = latent
                                self.keyframe_paths[keyframe_num] = target_path
                                
                                logger.debug(f"  Encoded dissimilar keyframe {keyframe_num} to latent")
                            except Exception as e:
                                logger.error(f"Failed to encode dissimilar keyframe: {e}")
                        
                        self.cache_injections += 1
                        
                        # Track cache injection for seed forcing logic
                        self.recent_cache_injections.append(True)
                        
                        logger.info(f"[OK] Keyframe {keyframe_num} from DISSIMILAR CACHE")
                        logger.info(f"     Saved to: {target_path.name}")
                        return True
                else:
                    logger.debug("Dissimilar injection returned None, continuing to normal generation")
        
        # === NORMAL GENERATION (No injection) ===
        # Get denoise from config
        denoise = self.config['generation']['hybrid']['keyframe_denoise']
        
        start_time = time.time()
        
        try:
            # Generate keyframe using hybrid generator
            keyframe_path = await self._generate_keyframe_sync(
                keyframe_num=keyframe_num,
                current_image=self.current_image_path,
                prompt=prompt,
                denoise=denoise
            )
            
            if keyframe_path and keyframe_path.exists():
                elapsed = time.time() - start_time
                self.generation_times.append(elapsed)
                
                # Mark as ready in buffer
                self.buffer.mark_ready(sequence_num, keyframe_path)
                
                # Add to cache for future injections (with selective caching)
                self.add_keyframe_to_cache(keyframe_path, prompt, denoise)
                
                # Update current image for next generation
                self.current_image_path = keyframe_path
                
                self.keyframes_generated += 1
                
                # Track that no cache injection occurred (for seed forcing logic)
                self.recent_cache_injections.append(False)
                
                logger.info(f"[OK] Keyframe {keyframe_num} generated in {elapsed:.2f}s")
                logger.info(f"     Saved to: {keyframe_path.name}")
                
                return True
            else:
                logger.error(f"Keyframe generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error generating keyframe: {e}", exc_info=True)
            return False
    
    async def _generate_keyframe_sync(
        self,
        keyframe_num: int,
        current_image: Path,
        prompt: str,
        denoise: float
    ) -> Optional[Path]:
        """
        Synchronous keyframe generation (wrapped for async)
        
        Args:
            keyframe_num: Keyframe number
            current_image: Current image path
            prompt: Generation prompt
            denoise: Denoise strength
            
        Returns:
            Path to generated keyframe
        """
        # Generate using generator's img2img
        keyframe_path = self.generator.generate_from_image(
            image_path=current_image,
            prompt=prompt,
            denoise=denoise
        )
        
        if not keyframe_path:
            return None
        
        # Move to keyframe directory with proper naming
        target_path = self.buffer.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
        
        # Move to target location (removes duplicate from output root)
        import shutil
        shutil.move(str(keyframe_path), str(target_path))
        
        # Encode and store keyframe latent if VAE available
        if self.latent_encoder:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latent = self.latent_encoder.encode(
                    target_path,
                    for_interpolation=True
                )
                
                # Store in coordinator's keyframe cache
                self.keyframe_latents[keyframe_num] = latent
                self.keyframe_paths[keyframe_num] = target_path
                
                logger.debug(f"  Encoded keyframe {keyframe_num} to latent: {latent.shape}")
                
            except Exception as e:
                logger.error(f"Failed to encode keyframe: {e}", exc_info=True)
        
        return target_path
    
    async def generate_interpolations(self, start_kf: int, end_kf: int) -> bool:
        """
        Generate interpolations between two keyframes
        
        Args:
            start_kf: Starting keyframe number
            end_kf: Ending keyframe number
            
        Returns:
            True if successful
        """
        count = self.buffer.interpolation_frames
        
        logger.info(f"{'='*70}")
        logger.info(f"Generating INTERPOLATIONS {start_kf} -> {end_kf} ({count} frames)")
        
        # Register interpolations in buffer
        sequence_nums = self.buffer.register_interpolations(start_kf, end_kf, count)
        
        # Check if we have latents for both keyframes
        if start_kf not in self.keyframe_latents:
            logger.error(f"Missing latent for keyframe {start_kf}")
            return False
        
        if end_kf not in self.keyframe_latents:
            logger.error(f"Missing latent for keyframe {end_kf}")
            return False
        
        start_latent = self.keyframe_latents[start_kf]
        end_latent = self.keyframe_latents[end_kf]
        
        # Precompute slerp parameters for this pair
        from interpolation.spherical_lerp import precompute_slerp_params, spherical_lerp
        
        pair_key = (start_kf, end_kf)
        if pair_key not in self.slerp_precomputed:
            self.slerp_precomputed[pair_key] = precompute_slerp_params(
                start_latent,
                end_latent
            )
        
        # Generate each interpolation
        start_time = time.time()
        success_count = 0
        
        for i, sequence_num in enumerate(sequence_nums, start=1):
            try:
                self.buffer.mark_generating(sequence_num)
                
                # Get frame spec for interpolation t value
                frame_spec = self.buffer.frames[sequence_num]
                t = frame_spec.interpolation_t
                
                # Perform spherical lerp
                interpolated_latent = spherical_lerp(
                    start_latent,
                    end_latent,
                    t,
                    precomputed=self.slerp_precomputed[pair_key]
                )
                
                # Decode to image
                interpolated_image = self.latent_encoder.decode(
                    interpolated_latent,
                    upscale_to_target=True
                )
                
                # Save to interpolation directory
                output_path = frame_spec.file_path
                interpolated_image.save(output_path, "PNG", optimize=False, compress_level=1)
                
                # Mark as ready
                self.buffer.mark_ready(sequence_num, output_path)
                
                success_count += 1
                self.interpolations_generated += 1
                
                logger.debug(f"  Generated interpolation {i}/{count} (t={t:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to generate interpolation {i}: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        logger.info(f"[OK] Generated {success_count}/{count} interpolations in {elapsed:.2f}s")
        logger.info(f"     Average: {elapsed/success_count:.3f}s per frame")
        
        # === CACHE INTERPOLATION MIDPOINT (Component 5) ===
        if self.config['generation']['cache'].get('cache_interpolations', True):
            if self.cache and self.similarity_manager and success_count > 0:
                try:
                    # Find midpoint frame (t ≈ 0.5)
                    midpoint_idx = count // 2
                    midpoint_sequence = sequence_nums[midpoint_idx]
                    midpoint_frame = self.buffer.frames[midpoint_sequence]
                    
                    # Encode midpoint
                    midpoint_embedding = self.similarity_manager.encode_image(
                        midpoint_frame.file_path
                    )
                    
                    if midpoint_embedding is not None:
                        # Use selective caching (same diversity eval as keyframes)
                        population_mode = self.config['generation']['cache'].get('population_mode', 'selective')
                        should_cache_interp = True
                        
                        if population_mode == 'selective' and self.cache.size() > 0:
                            should_cache_interp = self.cache.should_cache_frame(
                                midpoint_embedding,
                                similarity_manager=self.similarity_manager
                            )
                        
                        if should_cache_interp:
                            # Convert to serializable format if needed
                            if isinstance(midpoint_embedding, dict) and 'color' in midpoint_embedding:
                                # Dual-metric: convert to serializable
                                midpoint_embedding = self.similarity_manager.to_serializable(midpoint_embedding)
                            else:
                                raise Exception("Midpoint embedding is not a dictionary")
                            
                            self.cache.add(
                                image_path=midpoint_frame.file_path,
                                prompt=f"interpolation_{start_kf}_{end_kf}_t0.5",
                                generation_params={
                                    "type": "interpolation",
                                    "start_kf": start_kf,
                                    "end_kf": end_kf,
                                    "t": 0.5
                                },
                                embedding=midpoint_embedding
                            )
                            
                            logger.info(
                                f"[INTERP_CACHE] Cached midpoint "
                                f"({start_kf} -> {end_kf} @ t=0.5)"
                            )
                        else:
                            logger.debug(f"[INTERP_CACHE] Skipping midpoint (not diverse enough)")
                
                except Exception as e:
                    logger.debug(f"Interpolation caching failed: {e}")
        
        return success_count == count
    
    def pause(self) -> None:
        """Pause generation"""
        self.paused = True
        logger.info("Generation paused")
    
    def resume(self) -> None:
        """Resume generation"""
        self.paused = False
        logger.info("Generation resumed")
    
    def stop(self) -> None:
        """Stop generation"""
        self.running = False
        logger.info("Generation stopping...")
    
    def get_stats(self) -> Dict:
        """
        Get generation statistics including mode collapse metrics
        
        Returns:
            Dictionary with statistics
        """
        avg_time = 0.0
        if self.generation_times:
            avg_time = sum(self.generation_times[-10:]) / min(10, len(self.generation_times))
        
        cache_size = self.cache.size() if self.cache else 0
        
        stats = {
            "keyframes_generated": self.keyframes_generated,
            "interpolations_generated": self.interpolations_generated,
            "cache_injections": self.cache_injections,
            "cache_size": cache_size,
            "current_keyframe": self.current_keyframe_num,
            "avg_generation_time": avg_time,
            "is_paused": self.paused,
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
                    "collapse_frequency": injection_stats.get("recent_collapse_frequency", 0.0)
                })
            except Exception as e:
                logger.debug(f"Failed to get injection stats: {e}")
        
        # Add cache diversity stats
        if self.cache:
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
                    stats.update({
                        "cache_diversity_score": diversity_stats.get("diversity_score", 0.0),
                        "cache_avg_similarity": diversity_stats.get("avg_pairwise_similarity", 0.0)
                    })
            except Exception as e:
                logger.debug(f"Failed to get diversity stats: {e}")
        
        return stats