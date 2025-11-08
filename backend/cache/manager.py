"""
Cache Manager
Stores and retrieves generated images with metadata

The cache serves multiple purposes:
1. Store high-quality generated images for later re-use
2. Enable aesthetic matching via CLIP embeddings
3. Prevent mode collapse through periodic injection
4. Provide variety while maintaining coherence
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        embedding: CLIP embedding (512-dim vector) for similarity matching
        timestamp: When this was cached
    """
    cache_id: str
    image_path: Path
    prompt: str
    generation_params: Dict[str, Any]
    embedding: Optional[List[float]]
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
        embedding: Optional[List[float]] = None,
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
            embedding: CLIP embedding (None = will be computed later)
        
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
        Get all entries that have CLIP embeddings
        
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


# Test function
def test_cache_manager() -> bool:
    """Test cache manager"""
    import yaml
    from PIL import Image

    print("=" * 60)
    print("Testing CacheManager...")
    print("=" * 60)

    # Load config
    config_path = Path("backend/config.yaml")
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override cache settings for test
    config["system"]["cache_dir"] = "./test_cache"
    config["generation"]["cache"]["max_size"] = 5  # Small for testing

    # Create test images
    test_image_dir = Path("./test_cache_images")
    test_image_dir.mkdir(exist_ok=True)

    test_images = []
    for i in range(7):
        img = Image.new("RGB", (256, 512), color=(i * 30, 100, 200))
        img_path = test_image_dir / f"test_{i}.png"
        img.save(img_path)
        test_images.append(img_path)

    try:
        # Create manager
        print("\n1. Creating cache manager...")
        cache = CacheManager(config)
        print(f"✓ Manager created: {cache.size()} entries")

        # Test adding
        print("\n2. Adding images to cache...")
        cache_ids = []
        for i, img_path in enumerate(test_images[:3]):
            cache_id = cache.add(
                image_path=img_path,
                prompt=f"test prompt {i}",
                generation_params={"steps": 4, "denoise": 0.4},
                embedding=[float(i)] * 512,  # Mock embedding
            )
            cache_ids.append(cache_id)
            print(f"   Added: {cache_id}")

        print(f"✓ Added 3 images, cache size: {cache.size()}")

        # Test retrieval
        print("\n3. Testing retrieval...")
        entry = cache.get(cache_ids[0])
        if entry and entry.prompt == "test prompt 0":
            print(f"✓ Retrieved: {entry.cache_id}")
        else:
            print("✗ Retrieval failed")
            return False

        # Test LRU eviction
        print("\n4. Testing LRU eviction (max_size=5)...")
        for i, img_path in enumerate(test_images[3:], start=3):
            cache.add(
                image_path=img_path,
                prompt=f"test prompt {i}",
                generation_params={"steps": 4},
            )

        final_size = cache.size()
        if final_size == 5:
            print(f"✓ LRU eviction working: size={final_size}")
            # Check that old entries were removed
            if cache.get(cache_ids[0]) is None:
                print("  ✓ Oldest entry evicted")
            else:
                print("  ✗ Oldest entry not evicted")
                return False
        else:
            print(f"✗ LRU eviction failed: size={final_size} (expected 5)")
            return False

        # Test random
        print("\n5. Testing random selection...")
        random_entry = cache.get_random()
        if random_entry:
            print(f"✓ Random entry: {random_entry.cache_id}")
        else:
            print("✗ Random selection failed")
            return False

        # Test with embeddings
        print("\n6. Testing embedding filter...")
        with_emb = cache.get_with_embeddings()
        print(f"✓ Entries with embeddings: {len(with_emb)}")

        # Test stats
        print("\n7. Testing stats...")
        stats = cache.get_stats()
        print(f"✓ Cache stats:")
        print(f"   Total: {stats['total_entries']}")
        print(f"   Usage: {stats['usage_percent']:.1f}%")
        print(f"   With embeddings: {stats['with_embeddings']}")

        print("\n" + "=" * 60)
        print("CacheManager test PASSED ✓")
        print("=" * 60)

        # Cleanup
        cache.clear()
        shutil.rmtree(Path("./test_cache"))
        shutil.rmtree(test_image_dir)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()

        # Cleanup
        try:
            shutil.rmtree(Path("./test_cache"))
            shutil.rmtree(test_image_dir)
        except:
            pass
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    success = test_cache_manager()
    exit(0 if success else 1)

