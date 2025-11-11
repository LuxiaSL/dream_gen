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
import time
from pathlib import Path
from typing import Optional, Dict, Any
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
        aesthetic_matcher=None
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
            aesthetic_matcher: AestheticMatcher instance (optional, for cache injection)
        """
        self.buffer = frame_buffer
        self.generator = generator
        self.latent_encoder = latent_encoder
        self.prompt_manager = prompt_manager
        self.config = config
        self.cache = cache_manager
        self.aesthetic_matcher = aesthetic_matcher
        
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
    
    def should_inject_cache(self) -> bool:
        """
        Decide if should inject cached image for next keyframe
        
        Uses probability from config and checks cache has content.
        
        Returns:
            True if should inject cache
        """
        if self.current_keyframe_num == 0:
            return False  # Never inject on first keyframe
        
        if not self.cache or not self.aesthetic_matcher:
            return False  # Cache system not available
        
        # Check cache has entries
        if self.cache.size() == 0:
            return False
        
        # Random probability check
        import random
        injection_prob = self.config['generation']['cache']['injection_probability']
        return random.random() < injection_prob
    
    def inject_cached_keyframe(self) -> Optional[Path]:
        """
        Inject similar cached image as keyframe based on current aesthetic
        
        Returns:
            Path to cached image, or None if injection fails
        """
        if self.current_image_path is None or not self.current_image_path.exists():
            return None
        
        try:
            # Encode current frame
            current_embedding = self.aesthetic_matcher.encode_image(self.current_image_path)
            
            if current_embedding is None:
                logger.warning("Failed to encode current frame for cache injection")
                return None
            
            # Get all cached entries
            cache_entries = self.cache.get_all()
            if not cache_entries:
                logger.debug("Cache is empty, no injection possible")
                return None
            
            # Prepare candidates (cache_id, embedding) pairs
            candidates = [
                (entry.cache_id, entry.embedding)
                for entry in cache_entries
                if entry.embedding is not None
            ]
            
            if not candidates:
                logger.debug("No cache entries with embeddings")
                return None
            
            # Find similar cached images
            threshold = self.config['generation']['cache']['similarity_threshold']
            similar = self.aesthetic_matcher.find_similar(
                target_embedding=current_embedding,
                candidate_embeddings=candidates,
                threshold=threshold,
                top_k=5
            )
            
            if not similar:
                logger.debug(f"No similar images found (threshold: {threshold})")
                return None
            
            # Weighted random selection from similar images
            selected_cache_id = self.aesthetic_matcher.weighted_random_selection(similar)
            
            if not selected_cache_id:
                return None
            
            # Get the cached image path
            cached_entry = self.cache.get(selected_cache_id)
            if cached_entry:
                self.cache_injections += 1
                # Log with similarity info from similar list
                similarity_score = next((s for cid, s in similar if cid == selected_cache_id), 0.0)
                logger.info(f"[CACHE] INJECTION #{self.cache_injections}: {selected_cache_id} (similarity: {similarity_score:.3f})")
                return cached_entry.image_path
            
        except Exception as e:
            logger.error(f"Cache injection failed: {e}", exc_info=True)
        
        return None
    
    def add_keyframe_to_cache(self, keyframe_path: Path, prompt: str, denoise: float) -> bool:
        """
        Add a generated keyframe to the cache with CLIP embedding
        
        Args:
            keyframe_path: Path to generated keyframe
            prompt: Generation prompt used
            denoise: Denoise value used
            
        Returns:
            True if successfully added to cache
        """
        if not self.cache or not self.aesthetic_matcher:
            return False
        
        try:
            # Encode image to CLIP embedding
            embedding = self.aesthetic_matcher.encode_image(keyframe_path)
            
            if embedding is None:
                logger.warning(f"Failed to encode keyframe for cache: {keyframe_path.name}")
                return False
            
            # Convert numpy array to list for JSON serialization
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
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
        Generate the next keyframe (or inject from cache)
        
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
        
        # Check for cache injection
        if self.should_inject_cache():
            logger.info("  -> Attempting cache injection...")
            cached_path = self.inject_cached_keyframe()
            
            if cached_path and cached_path.exists():
                # Use cached image as keyframe
                import shutil
                target_path = self.buffer.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
                shutil.copy(cached_path, target_path)
                
                # Mark as ready in buffer
                self.buffer.mark_ready(sequence_num, target_path)
                
                # Update current image for next generation
                self.current_image_path = target_path
                
                # Encode cached frame if using VAE
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
                        
                        logger.debug(f"  Encoded cached keyframe {keyframe_num} to latent")
                    except Exception as e:
                        logger.error(f"Failed to encode cached keyframe: {e}")
                
                logger.info(f"[OK] Keyframe {keyframe_num} from CACHE")
                logger.info(f"     Saved to: {target_path.name}")
                return True
        
        # Normal generation (no cache injection)
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
                
                # Add to cache for future injections
                self.add_keyframe_to_cache(keyframe_path, prompt, denoise)
                
                # Update current image for next generation
                self.current_image_path = keyframe_path
                
                self.keyframes_generated += 1
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
        Get generation statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_time = 0.0
        if self.generation_times:
            avg_time = sum(self.generation_times[-10:]) / min(10, len(self.generation_times))
        
        cache_size = self.cache.size() if self.cache else 0
        
        return {
            "keyframes_generated": self.keyframes_generated,
            "interpolations_generated": self.interpolations_generated,
            "cache_injections": self.cache_injections,
            "cache_size": cache_size,
            "current_keyframe": self.current_keyframe_num,
            "avg_generation_time": avg_time,
            "is_paused": self.paused,
            "is_running": self.running
        }