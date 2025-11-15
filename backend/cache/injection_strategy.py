"""
Cache Injection Strategy

Handles all cache and seed injection logic for mode collapse prevention.
This separates the complex injection logic from the generation coordinator,
keeping the codebase modular and maintainable.

Key Features:
- Dissimilar cache injection with VAE blending
- Adaptive seed injection based on collapse frequency
- Emergency seed injection triggers
- Performance: ~150ms for VAE blending operations
"""

import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CacheInjectionStrategy:
    """
    Manages cache and seed injection strategies for mode collapse prevention
    
    This class encapsulates all the complex logic for:
    1. Dissimilar cache injection (Component 3)
    2. Adaptive seed injection (Component 4)
    3. VAE latent blending
    4. Emergency triggers
    
    The generation coordinator delegates to this class to keep itself lean.
    
    Usage:
        strategy = CacheInjectionStrategy(
            config=config,
            cache_manager=cache,
            similarity_manager=similarity_manager,
            latent_encoder=latent_encoder,
            buffer=buffer
        )
        
        # Dissimilar cache injection
        result = strategy.inject_dissimilar_keyframe(
            current_image_path=path,
            target_keyframe_num=42
        )
        
        # Adaptive seed injection
        if strategy.should_inject_seed(collapse_history):
            result = strategy.inject_seed_frame(
                current_image_path=path,
                target_keyframe_num=42
            )
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        cache_manager,
        similarity_manager,
        latent_encoder=None,  # Legacy: for backward compatibility
        buffer=None,
        vae_access=None  # NEW: SharedVAEAccess for async operations
    ):
        """
        Initialize injection strategy
        
        Args:
            config: Configuration dictionary
            cache_manager: CacheManager instance
            similarity_manager: DualMetricSimilarityManager instance
            latent_encoder: (DEPRECATED) LatentEncoder instance - use vae_access instead
            buffer: FrameBuffer instance (optional for direct output_dir)
            vae_access: SharedVAEAccess instance for thread-safe async VAE operations
        """
        self.config = config
        self.cache = cache_manager
        self.similarity_manager = similarity_manager
        
        # Support both legacy (latent_encoder) and new (vae_access) interfaces
        if vae_access is not None:
            self.vae_access = vae_access
            self.latent_encoder = None  # Not used in async mode
            self.is_async = True
        elif latent_encoder is not None:
            self.latent_encoder = latent_encoder
            self.vae_access = None  # Not available in sync mode
            self.is_async = False
        else:
            logger.warning("Neither vae_access nor latent_encoder provided - VAE blending disabled")
            self.vae_access = None
            self.latent_encoder = None
            self.is_async = False
        
        self.buffer = buffer
        
        # Determine output directory
        if buffer:
            self.output_dir = buffer.keyframe_dir
        else:
            self.output_dir = Path(config['system']['output_dir']) / 'frames' / 'keyframes'
        
        # Cache config shortcuts
        self.cache_config = config['generation']['cache']
        
        # Injection tracking
        self.total_cache_injections = 0
        self.total_seed_injections = 0
        self.collapse_detection_history: deque = deque(maxlen=20)
        self.recent_cache_injections: deque = deque(maxlen=5)  # Track last 5 injected cache IDs
        
        logger.info("CacheInjectionStrategy initialized")
        logger.info(f"  Mode: {'ASYNC (thread-safe)' if self.is_async else 'SYNC (legacy)'}")
        logger.info(f"  Similarity method: {self.cache_config.get('similarity_method', 'dual_metric')}")
        logger.info(f"  Injection mode: {self.cache_config.get('injection_mode', 'dissimilar')}")
        logger.info(f"  Blend weight: {self.cache_config.get('blend_weight', 0.6)}")
        logger.info(f"  Anti-loop tracking: {5} recent injections")
    
    async def inject_dissimilar_keyframe(
        self,
        current_image_path: Path,
        target_keyframe_num: int,
        collapse_trigger: Optional[str] = None
    ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Inject DISSIMILAR cached frame with VAE latent blending
        
        This actively breaks mode collapse by introducing different aesthetics.
        Uses dissimilarity-based selection and intelligently prioritizes the
        metric that triggered the collapse.
        
        Args:
            current_image_path: Path to current frame
            target_keyframe_num: Target keyframe number for saving
            collapse_trigger: Which metric(s) triggered (e.g., "COLOR", "STRUCTURAL", "BOTH")
                             Used for smart selection prioritization
            
        Returns:
            Tuple of (blended_frame_path, metadata) or None if injection fails
        """
        if not current_image_path or not current_image_path.exists():
            logger.warning("No current image for dissimilar injection")
            return None
        
        try:
            # Encode current frame
            current_embedding = self.similarity_manager.encode_image(current_image_path)
            
            if current_embedding is None:
                logger.warning("Failed to encode current frame")
                return None
            
            # Get cache candidates
            cache_entries = self.cache.get_all()
            if not cache_entries:
                logger.debug("Cache is empty")
                return None
            
            # Use dual-metric similarity system
            is_dual_metric = isinstance(current_embedding, dict) and 'color' in current_embedding
            
            if is_dual_metric:
                # Dual-metric OR logic: Select frames dissimilar in EITHER color OR structure
                # Smart prioritization: if collapse_trigger specified, prioritize that metric
                color_range = self.cache_config.get('color_histogram', {}).get('dissimilarity_range', [0.90, 1.70])
                struct_range = self.cache_config.get('phash', {}).get('dissimilarity_range', [0.42, 0.62])
                
                # Log smart prioritization if active
                if collapse_trigger:
                    if "COLOR" in collapse_trigger and "STRUCTURAL" not in collapse_trigger:
                        logger.info(f"  Smart selection: Prioritizing COLOR dissimilarity (color collapse detected)")
                    elif "STRUCTURAL" in collapse_trigger and "COLOR" not in collapse_trigger:
                        logger.info(f"  Smart selection: Prioritizing STRUCTURAL dissimilarity (structural collapse detected)")
                    elif "BOTH" in collapse_trigger:
                        logger.info(f"  Smart selection: Using max dissimilarity (both metrics triggered)")
                
                candidates = []
                for entry in cache_entries:
                    if entry.embedding is None:
                        continue
                    
                    # Skip if not dual-metric embedding
                    if not isinstance(entry.embedding, dict) or 'color' not in entry.embedding:
                        continue
                    
                    # Calculate similarities for BOTH metrics
                    color_sim = self.similarity_manager.get_color_similarity(current_embedding, entry.embedding)
                    struct_sim = self.similarity_manager.get_struct_similarity(current_embedding, entry.embedding)
                    
                    # OR logic: candidate if dissimilar in EITHER metric
                    color_dissimilar = color_range[0] < color_sim < color_range[1]
                    struct_dissimilar = struct_range[0] < struct_sim < struct_range[1]
                    
                    if color_dissimilar or struct_dissimilar:
                        # Calculate combined dissimilarity score for ranking
                        # Normalize each metric to 0-1 scale where higher = more dissimilar
                        # For ColorHist: lower sim in range = more dissimilar
                        color_dissim = 1.0 - ((color_sim - color_range[0]) / (color_range[1] - color_range[0]))
                        # For pHash: lower sim in range = more dissimilar
                        struct_dissim = 1.0 - ((struct_sim - struct_range[0]) / (struct_range[1] - struct_range[0]))
                        
                        # Smart selection: prioritize metric that triggered collapse
                        if collapse_trigger and "COLOR" in collapse_trigger and "STRUCTURAL" not in collapse_trigger:
                            # Color collapse detected → prioritize COLOR dissimilarity
                            combined_dissimilarity = color_dissim
                        elif collapse_trigger and "STRUCTURAL" in collapse_trigger and "COLOR" not in collapse_trigger:
                            # Structural collapse detected → prioritize STRUCTURAL dissimilarity
                            combined_dissimilarity = struct_dissim
                        else:
                            # Both triggered or baseline injection → use max (most dissimilar dimension)
                            combined_dissimilarity = max(color_dissim, struct_dissim)
                        
                        candidates.append((entry, combined_dissimilarity, color_sim, struct_sim, color_dissim, struct_dissim))
                
                if not candidates:
                    logger.debug(
                        f"No dissimilar frames (color:{color_range}, struct:{struct_range})"
                    )
                    return None
                
                # Weighted random selection FAVORING dissimilarity
                dissimilarities = np.array([c[1] for c in candidates])
                
                # Amplify differences with exponential weighting
                weights = np.exp(dissimilarities * 2)
                
                # Anti-loop: Penalize recently injected frames
                for i, (entry, _, _, _, _, _) in enumerate(candidates):
                    if entry.cache_id in self.recent_cache_injections:
                        weights[i] *= 0.1  # 90% penalty for recently used
                        logger.debug(f"  Penalizing recently used {entry.cache_id} (weight *= 0.1)")
                
                # Normalize weights
                weights = weights / weights.sum()
                
                selected_idx = np.random.choice(len(candidates), p=weights)
                selected_entry, selected_dissimilarity, selected_color_sim, selected_struct_sim, color_dissim, struct_dissim = candidates[selected_idx]
                
                # Track this injection
                self.recent_cache_injections.append(selected_entry.cache_id)
                
                # Log selection with smart prioritization info
                priority_info = ""
                if collapse_trigger:
                    if "COLOR" in collapse_trigger and "STRUCTURAL" not in collapse_trigger:
                        priority_info = f", prioritized COLOR (dissim:{color_dissim:.3f})"
                    elif "STRUCTURAL" in collapse_trigger and "COLOR" not in collapse_trigger:
                        priority_info = f", prioritized STRUCT (dissim:{struct_dissim:.3f})"
                
                logger.info(
                    f"[DISSIMILAR] Selected {selected_entry.cache_id} "
                    f"(color:{selected_color_sim:.3f}, struct:{selected_struct_sim:.3f}, "
                    f"dissim:{selected_dissimilarity:.3f}{priority_info})"
                )
            
            # VAE latent blending (not direct copy!)
            if not self.vae_access and not self.latent_encoder:
                logger.warning("No VAE access available, falling back to direct copy")
                return self._direct_copy_fallback(selected_entry, target_keyframe_num)
            
            try:
                # Use async VAE access if available (thread-safe)
                if self.vae_access:
                    current_latent = await self.vae_access.encode_async(
                        current_image_path,
                        for_interpolation=True
                    )
                    cached_latent = await self.vae_access.encode_async(
                        selected_entry.image_path,
                        for_interpolation=True
                    )
                else:
                    # Legacy sync mode (no lock - only safe in single-threaded context)
                    current_latent = self.latent_encoder.encode(
                        current_image_path,
                        for_interpolation=True
                    )
                    cached_latent = self.latent_encoder.encode(
                        selected_entry.image_path,
                        for_interpolation=True
                    )
                
                # Blend: weighted towards cached (breaking collapse)
                blend_weight = self.cache_config.get('blend_weight', 0.6)
                blended_latent = (
                    cached_latent * blend_weight +
                    current_latent * (1.0 - blend_weight)
                )
                
                # Decode to image
                if self.vae_access:
                    blended_image = await self.vae_access.decode_async(
                        blended_latent,
                        upscale_to_target=True
                    )
                else:
                    blended_image = self.latent_encoder.decode(
                        blended_latent,
                        upscale_to_target=True
                    )
                
                # Save blended frame
                target_path = self.output_dir / f"keyframe_{target_keyframe_num:03d}.png"
                blended_image.save(target_path, "PNG", optimize=False, compress_level=1)
                
                self.total_cache_injections += 1
                
                logger.info(
                    f"[BLEND] Created blended keyframe "
                    f"({blend_weight*100:.0f}% cached, {(1-blend_weight)*100:.0f}% current)"
                )
                
                # Create metadata based on which metric was used
                if is_dual_metric:
                    metadata = {
                        "type": "dissimilar_cache_injection",
                        "cache_id": selected_entry.cache_id,
                        "color_similarity": selected_color_sim,
                        "struct_similarity": selected_struct_sim,
                        "dissimilarity": selected_dissimilarity,
                        "blend_weight": blend_weight
                    }
                else:
                    metadata = {
                        "type": "dissimilar_cache_injection",
                        "cache_id": selected_entry.cache_id,
                        "color_similarity": selected_color_sim,
                        "struct_similarity": selected_struct_sim,
                        "dissimilarity": selected_dissimilarity,
                        "blend_weight": blend_weight
                    }
                
                return target_path, metadata
                
            except Exception as e:
                logger.error(f"VAE blending failed: {e}", exc_info=True)
                logger.info("Falling back to direct copy")
                return self._direct_copy_fallback(selected_entry, target_keyframe_num)
            
        except Exception as e:
            logger.error(f"Dissimilar injection failed: {e}", exc_info=True)
            return None
    
    def _direct_copy_fallback(
        self,
        cache_entry,
        target_keyframe_num: int
    ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Fallback: Direct copy of cached frame (no blending)
        
        Used when VAE blending fails or latent encoder unavailable.
        """
        import shutil
        
        try:
            target_path = self.output_dir / f"keyframe_{target_keyframe_num:03d}.png"
            shutil.copy(cache_entry.image_path, target_path)
            
            self.total_cache_injections += 1
            
            logger.info(f"[DIRECT_COPY] Copied cached frame {cache_entry.cache_id}")
            
            metadata = {
                "type": "direct_cache_copy",
                "cache_id": cache_entry.cache_id
            }
            
            return target_path, metadata
            
        except Exception as e:
            logger.error(f"Direct copy fallback failed: {e}")
            return None
    
    def should_inject_seed(self, collapse_history: Optional[deque] = None) -> bool:
        """
        Adaptive seed injection based on cache injection frequency
        
        Logic:
        - BOOST probability when cache is small (bootstrap phase)
        - Probability increases with cache injection count (floor -> max)
        - Forced injection if frequent collapse detected
        
        Args:
            collapse_history: Deque of recent collapse detections (True/False)
                             If None, uses internal tracking
        
        Returns:
            True if should inject seed
        """
        # Use provided history or internal tracking
        if collapse_history is None:
            collapse_history = self.collapse_detection_history
        
        # Check for forced seed injection (emergency)
        if len(collapse_history) > 0:
            recent_collapses = sum(1 for x in collapse_history if x)
            collapse_frequency = recent_collapses / len(collapse_history)
            
            force_seed_threshold = self.cache_config.get('force_seed_collapse_frequency', 0.3)
            
            if collapse_frequency > force_seed_threshold:
                logger.warning(
                    f"[EMERGENCY] High collapse frequency "
                    f"({collapse_frequency:.1%}) - forcing seed injection"
                )
                return True
        
        # Check for early seed injection boost (bootstrap phase)
        cache_size = self.cache.size()
        boost_threshold = self.cache_config.get('seed_injection_boost_threshold', 10)
        
        if cache_size < boost_threshold:
            # Higher seed probability during cache bootstrap
            boost_probability = self.cache_config.get('seed_injection_boost_probability', 0.20)
            should_inject = random.random() < boost_probability
            
            if should_inject:
                logger.info(
                    f"[SEED] Bootstrap seed injection "
                    f"(cache size: {cache_size}/{boost_threshold}, "
                    f"probability: {boost_probability:.1%})"
                )
            
            return should_inject
        
        # Adaptive probability based on cache injection count
        floor_probability = self.cache_config.get('seed_injection_floor', 0.02)
        max_probability = self.cache_config.get('seed_injection_max', 0.15)
        ramp_injections = self.cache_config.get('seed_injection_ramp', 50)
        
        # Linear ramp from floor to max over ramp_injections
        progress = min(self.total_cache_injections / ramp_injections, 1.0)
        current_probability = (
            floor_probability + 
            (max_probability - floor_probability) * progress
        )
        
        should_inject = random.random() < current_probability
        
        if should_inject:
            logger.info(
                f"[SEED] Adaptive seed injection "
                f"(probability: {current_probability:.1%}, "
                f"cache injections: {self.total_cache_injections})"
            )
        
        return should_inject
    
    async def inject_seed_frame(
        self,
        target_keyframe_num: int,
        current_image_path: Optional[Path] = None
    ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """
        Inject seed image with optional VAE blending
        
        Args:
            target_keyframe_num: Target keyframe number for saving
            current_image_path: Current image path for blending (optional)
        
        Returns:
            Tuple of (seed_frame_path, metadata) or None if injection fails
        """
        seed_dir = Path(self.config['system']['seed_dir'])
        seed_images = list(seed_dir.glob("*.png")) + list(seed_dir.glob("*.jpg"))
        
        if not seed_images:
            logger.error("No seed images found")
            return None
        
        seed_path = random.choice(seed_images)
        logger.info(f"[SEED] Selected seed: {seed_path.name}")
        
        # Check if should blend with current
        blend_seeds = self.cache_config.get('blend_seed_injection', True)
        
        if blend_seeds and current_image_path and (self.vae_access or self.latent_encoder):
            try:
                # VAE blend: 50/50 with current for smoother transition
                if self.vae_access:
                    # Async mode (thread-safe)
                    current_latent = await self.vae_access.encode_async(
                        current_image_path,
                        for_interpolation=True
                    )
                    seed_latent = await self.vae_access.encode_async(
                        seed_path,
                        for_interpolation=True
                    )
                else:
                    # Legacy sync mode
                    current_latent = self.latent_encoder.encode(
                        current_image_path,
                        for_interpolation=True
                    )
                    seed_latent = self.latent_encoder.encode(
                        seed_path,
                        for_interpolation=True
                    )
                
                blend_weight = self.cache_config.get('seed_blend_weight', 0.5)
                
                blended_latent = (
                    seed_latent * blend_weight +
                    current_latent * (1.0 - blend_weight)
                )
                
                if self.vae_access:
                    blended_image = await self.vae_access.decode_async(
                        blended_latent,
                        upscale_to_target=True
                    )
                else:
                    blended_image = self.latent_encoder.decode(
                        blended_latent,
                        upscale_to_target=True
                    )
                
                # Save blended seed
                target_path = self.output_dir / f"keyframe_{target_keyframe_num:03d}.png"
                blended_image.save(target_path, "PNG", optimize=False, compress_level=1)
                
                self.total_seed_injections += 1
                
                logger.info(f"[SEED] Blended seed with current frame ({blend_weight*100:.0f}% seed)")
                
                metadata = {
                    "type": "blended_seed_injection",
                    "seed_name": seed_path.name,
                    "blend_weight": blend_weight
                }
                
                return target_path, metadata
                
            except Exception as e:
                logger.error(f"Seed blending failed, using direct seed: {e}")
                # Fall through to direct seed injection
        
        # Direct seed injection (no blend)
        import shutil
        
        try:
            target_path = self.output_dir / f"keyframe_{target_keyframe_num:03d}.png"
            shutil.copy(seed_path, target_path)
            
            self.total_seed_injections += 1
            
            logger.info(f"[SEED] Direct seed injection: {seed_path.name}")
            
            metadata = {
                "type": "direct_seed_injection",
                "seed_name": seed_path.name
            }
            
            return target_path, metadata
            
        except Exception as e:
            logger.error(f"Direct seed injection failed: {e}")
            return None
    
    def record_collapse_detection(self, is_collapsed: bool) -> None:
        """
        Record collapse detection result for adaptive seed injection
        
        Args:
            is_collapsed: True if collapse detected
        """
        self.collapse_detection_history.append(is_collapsed)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get injection statistics
        
        Returns:
            Dictionary with statistics
        """
        collapse_count = sum(1 for x in self.collapse_detection_history if x)
        collapse_frequency = collapse_count / len(self.collapse_detection_history) if self.collapse_detection_history else 0.0
        
        return {
            "total_cache_injections": self.total_cache_injections,
            "total_seed_injections": self.total_seed_injections,
            "recent_collapse_frequency": collapse_frequency,
            "recent_collapse_count": collapse_count,
            "history_size": len(self.collapse_detection_history)
        }

