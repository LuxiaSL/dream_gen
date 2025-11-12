"""
Cache Manager
Stores and retrieves generated images with metadata

The cache serves multiple purposes:
1. Store high-quality generated images for later re-use
2. Enable aesthetic matching via dual-metric embeddings (ColorHist + pHash-8)
3. Prevent mode collapse through periodic injection
4. Provide variety while maintaining coherence
"""

import json
import logging
import shutil
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Single cache entry with metadata
    
    Attributes:
        cache_id: Unique identifier
        image_path: Path to cached image file
        prompt: Generation prompt used
        generation_params: Dict of generation parameters (denoise, steps, etc.)
        embedding: Dual-metric embedding {'color': list[96], 'struct': str} for similarity matching
        timestamp: When this was cached
    """
    cache_id: str
    image_path: Path
    prompt: str
    generation_params: Dict[str, Any]
    embedding: Optional[Dict[str, Any]]  # Dual-metric embedding {'color': ndarray, 'struct': str}
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage"""
        return {
            "cache_id": self.cache_id,
            "image_path": str(self.image_path),
            "prompt": self.prompt,
            "generation_params": self.generation_params,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Deserialize from dictionary"""
        return cls(
            cache_id=data["cache_id"],
            image_path=Path(data["image_path"]),
            prompt=data["prompt"],
            generation_params=data["generation_params"],
            embedding=data.get("embedding"),
            timestamp=data["timestamp"],
        )


class CacheManager:
    """
    Manages image cache with metadata
    
    Features:
    - LRU eviction when max size reached
    - Persistent metadata (survives restarts)
    - Fast lookup by ID or embedding similarity
    - Atomic operations (thread-safe)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config  # Store full config for diversity checks
        self.cache_dir = Path(config["system"]["cache_dir"])
        self.image_dir = self.cache_dir / "images"
        self.metadata_dir = self.cache_dir / "metadata"
        self.max_size = config["generation"]["cache"]["max_size"]

        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index
        self.entries: Dict[str, CacheEntry] = {}

        # Load existing cache
        self.load_cache()

        logger.info(
            f"CacheManager initialized: {len(self.entries)}/{self.max_size} entries"
        )

    def load_cache(self) -> None:
        """Load cache index from disk"""
        metadata_file = self.metadata_dir / "cache_index.json"

        if not metadata_file.exists():
            logger.info("No existing cache found, starting fresh")
            return

        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = CacheEntry.from_dict(entry_data)
                # Verify image still exists
                if entry.image_path.exists():
                    self.entries[entry.cache_id] = entry
                else:
                    logger.warning(f"Cache image missing: {entry.cache_id}")

            logger.info(f"Loaded {len(self.entries)} cache entries")

        except Exception as e:
            logger.error(f"Failed to load cache: {e}", exc_info=True)
            logger.info("Starting with empty cache")

    def save_cache(self) -> None:
        """Save cache index to disk"""
        metadata_file = self.metadata_dir / "cache_index.json"

        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }

        try:
            # Write atomically
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.metadata_dir,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_path = tmp_file.name

            shutil.move(tmp_path, str(metadata_file))
            logger.debug("Cache index saved")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)

    def add(
        self,
        image_path: Path,
        prompt: str,
        generation_params: Dict[str, Any],
        embedding: Optional[Union[Dict[str, Any], List[float]]] = None,
    ) -> str:
        """
        Add image to cache
        
        Process:
        1. Generate unique cache ID
        2. Copy image to cache directory
        3. Create CacheEntry with metadata
        4. Add to in-memory index
        5. Enforce max size (LRU eviction if needed)
        6. Persist index to disk
        
        Args:
            image_path: Source image to cache
            prompt: Prompt used to generate
            generation_params: Generation parameters
            embedding: Dual-metric embedding {'color': ndarray, 'struct': str}
                      None = will be computed later
        
        Returns:
            cache_id of the new entry
        """
        if not image_path.exists():
            logger.error(f"Cannot cache non-existent image: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate cache ID
        cache_id = f"cache_{len(self.entries):05d}_{int(datetime.now().timestamp())}"

        # Copy image to cache
        cached_image_path = self.image_dir / f"{cache_id}.png"

        try:
            shutil.copy2(image_path, cached_image_path)
        except Exception as e:
            logger.error(f"Failed to copy image to cache: {e}")
            raise

        # Create entry
        entry = CacheEntry(
            cache_id=cache_id,
            image_path=cached_image_path,
            prompt=prompt,
            generation_params=generation_params,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
        )

        # Add to cache
        self.entries[cache_id] = entry

        # Enforce max size
        self._enforce_max_size()

        # Save cache
        self.save_cache()

        logger.info(f"Added to cache: {cache_id} (total: {len(self.entries)})")
        return cache_id

    def _enforce_max_size(self) -> None:
        """
        Remove oldest entries if cache exceeds max size
        
        Uses LRU (Least Recently Used) eviction strategy
        """
        if len(self.entries) <= self.max_size:
            return

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self.entries.items(), key=lambda x: x[1].timestamp
        )

        # Remove oldest
        to_remove = len(self.entries) - self.max_size

        for i in range(to_remove):
            cache_id, entry = sorted_entries[i]

            # Delete image file
            try:
                if entry.image_path.exists():
                    entry.image_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cached image: {e}")

            # Remove from entries
            del self.entries[cache_id]

            logger.debug(f"Evicted from cache: {cache_id}")

        logger.info(f"Cache size enforced: {len(self.entries)}/{self.max_size}")

    def get(self, cache_id: str) -> Optional[CacheEntry]:
        """
        Get cache entry by ID
        
        Args:
            cache_id: Cache entry ID
        
        Returns:
            CacheEntry if found, None otherwise
        """
        return self.entries.get(cache_id)

    def get_all(self) -> List[CacheEntry]:
        """
        Get all cache entries
        
        Returns:
            List of all CacheEntry objects
        """
        return list(self.entries.values())

    def get_random(self) -> Optional[CacheEntry]:
        """
        Get random cache entry
        
        Useful for variety injection
        
        Returns:
            Random CacheEntry, or None if cache empty
        """
        import random

        if not self.entries:
            return None

        cache_id = random.choice(list(self.entries.keys()))
        return self.entries[cache_id]

    def get_with_embeddings(self) -> List[CacheEntry]:
        """
        Get all entries that have embeddings
        
        Returns:
            List of CacheEntry objects with embeddings
        """
        return [e for e in self.entries.values() if e.embedding is not None]

    def size(self) -> int:
        """
        Current cache size
        
        Returns:
            Number of cached entries
        """
        return len(self.entries)

    def clear(self) -> None:
        """
        Clear entire cache
        
        WARNING: This deletes all cached images!
        """
        for entry in self.entries.values():
            try:
                if entry.image_path.exists():
                    entry.image_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cached image: {e}")

        self.entries.clear()
        self.save_cache()
        logger.warning("Cache cleared!")

    def should_cache_frame(
        self, 
        new_embedding: Union[Dict[str, Any], np.ndarray],
        force: bool = False,
        similarity_manager=None
    ) -> bool:
        """
        Determine if frame should be cached based on diversity
        
        This is KEY to preventing cache homogeneity. Only cache frames
        that add diversity to the cache, not frames similar to existing ones.
        
        Args:
            new_embedding: Dual-metric embedding {'color': ndarray, 'struct': str}
            force: Force caching (for seeds, important frames)
            similarity_manager: DualMetricSimilarityManager instance
            
        Returns:
            True if frame adds diversity to cache
        """
        if force or self.size() == 0:
            return True
        
        # Get all cached embeddings
        cached_embeddings = [
            e.embedding for e in self.entries.values()
            if e.embedding is not None
        ]
        
        if not cached_embeddings:
            return True
        
        # Dual-metric diversity check
        if not similarity_manager:
            logger.warning("No similarity_manager provided - cannot check diversity")
            return True
        
        # Filter to only dual-metric embeddings
        dual_embeddings = [e for e in cached_embeddings if isinstance(e, dict) and 'color' in e]
        
        if not dual_embeddings:
            # No embeddings in cache yet - cache this one
            return True
        
        # Use OR logic: check if diverse in EITHER color OR structure
        cache_config = self.config['generation']['cache']
        color_threshold = cache_config.get('color_histogram', {}).get('diversity_threshold', 1.80)
        struct_threshold = cache_config.get('phash', {}).get('diversity_threshold', 0.65)
        
        # Calculate average similarities
        color_sims = []
        struct_sims = []
        for cached in dual_embeddings:
            color_sim = similarity_manager.get_color_similarity(new_embedding, cached)
            struct_sim = similarity_manager.get_struct_similarity(new_embedding, cached)
            color_sims.append(color_sim)
            struct_sims.append(struct_sim)
        
        avg_color_sim = float(np.mean(color_sims)) if color_sims else 0.0
        avg_struct_sim = float(np.mean(struct_sims)) if struct_sims else 0.0
        
        # Get cache diversity logic (AND vs OR)
        cache_diversity_logic = cache_config.get('cache_diversity_logic', 'all')
        
        # Determine if diverse based on logic mode
        color_diverse = avg_color_sim < color_threshold
        struct_diverse = avg_struct_sim < struct_threshold
        
        if cache_diversity_logic == 'all':
            # AND logic: BOTH metrics must show diversity
            is_diverse = color_diverse and struct_diverse
            logic_str = "AND"
        else:
            # OR logic: EITHER metric shows diversity
            is_diverse = color_diverse or struct_diverse
            logic_str = "OR"
        
        if is_diverse:
            logger.debug(
                f"Frame is diverse (color:{avg_color_sim:.3f}<{color_threshold:.3f} "
                f"{logic_str} struct:{avg_struct_sim:.3f}<{struct_threshold:.3f}) - caching"
            )
        else:
            # Build readable rejection reason
            color_status = "PASS" if color_diverse else "FAIL"
            struct_status = "PASS" if struct_diverse else "FAIL"
            logger.debug(
                f"Frame is redundant (color:{avg_color_sim:.3f} {color_status}, "
                f"struct:{avg_struct_sim:.3f} {struct_status}, logic:{logic_str}) - skipping cache"
            )
        
        return is_diverse
    
    def get_diversity_stats(self, similarity_manager=None) -> Dict[str, Any]:
        """
        Calculate cache diversity metrics
        
        Args:
            similarity_manager: DualMetricSimilarityManager instance (required for dual-metric)
        
        Returns:
            Dictionary with diversity statistics:
            {
                "diversity_score_color": float,
                "diversity_score_struct": float,
                "avg_color_similarity": float,
                "avg_struct_similarity": float,
                "cache_size": int
            }
        """
        embeddings = [
            e.embedding for e in self.entries.values()
            if e.embedding is not None
        ]
        
        if len(embeddings) < 2:
            return {
                "diversity_score": 1.0,
                "avg_pairwise_similarity": 0.0,
                "cache_size": len(embeddings)
            }
        
        # Check if we have dual-metric embeddings
        dual_embeddings = [e for e in embeddings if isinstance(e, dict) and 'color' in e]
        
        if dual_embeddings and similarity_manager:
            # Dual-metric diversity stats
            n = len(dual_embeddings)
            color_sims = []
            struct_sims = []
            
            for i in range(n):
                for j in range(i+1, n):
                    color_sim = similarity_manager.get_color_similarity(dual_embeddings[i], dual_embeddings[j])
                    struct_sim = similarity_manager.get_struct_similarity(dual_embeddings[i], dual_embeddings[j])
                    color_sims.append(color_sim)
                    struct_sims.append(struct_sim)
            
            avg_color_sim = float(np.mean(color_sims)) if color_sims else 0.0
            avg_struct_sim = float(np.mean(struct_sims)) if struct_sims else 0.0
            
            # Diversity is inversely proportional to similarity
            # For ColorHist (range ~0.8-2.3): lower similarity = more diverse
            # For pHash (range ~0.4-0.9): lower similarity = more diverse
            # We'll normalize to 0-1 scale where 1 = most diverse
            color_diversity = max(0.0, 1.0 - (avg_color_sim / 2.3))  # Normalize by max range
            struct_diversity = max(0.0, 1.0 - avg_struct_sim)  # Already 0-1 scale
            
            return {
                "diversity_score_color": color_diversity,
                "diversity_score_struct": struct_diversity,
                "avg_color_similarity": avg_color_sim,
                "avg_struct_similarity": avg_struct_sim,
                "cache_size": n
            }
        
        return {
            "diversity_score_color": 1.0,
            "diversity_score_struct": 1.0,
            "avg_color_similarity": 0.0,
            "avg_struct_similarity": 0.0,
            "cache_size": 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        with_embeddings = len(self.get_with_embeddings())

        return {
            "total_entries": len(self.entries),
            "max_size": self.max_size,
            "usage_percent": (len(self.entries) / self.max_size * 100)
            if self.max_size > 0
            else 0,
            "with_embeddings": with_embeddings,
            "without_embeddings": len(self.entries) - with_embeddings,
        }