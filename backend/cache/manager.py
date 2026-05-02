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
from dataclasses import dataclass
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
        template_id: Optional identifier for the template that generated this frame.
                    Used for per-template cache isolation during template switches.
        components: Optional dict of component values used (e.g. {'color': 'cyan', 'mood': 'ethereal'}).
                   Preserved for debugging and potential similarity-guided selection.
    """
    cache_id: str
    image_path: Path
    prompt: str
    generation_params: Dict[str, Any]
    embedding: Optional[Dict[str, Any]]  # Dual-metric embedding {'color': ndarray, 'struct': str}
    timestamp: str
    template_id: Optional[str] = None
    components: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage"""
        result = {
            "cache_id": self.cache_id,
            "image_path": str(self.image_path),
            "prompt": self.prompt,
            "generation_params": self.generation_params,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
        }
        # Only include template metadata if present (backwards compatible)
        if self.template_id is not None:
            result["template_id"] = self.template_id
        if self.components is not None:
            result["components"] = self.components
        return result

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
            template_id=data.get("template_id"),
            components=data.get("components"),
        )


class CacheManager:
    """
    Manages image cache with metadata
    
    Features:
    - LRU eviction when max size reached
    - Persistent metadata (survives restarts)
    - Fast lookup by ID or embedding similarity
    - Atomic operations (thread-safe)
    - Per-template cache isolation with archive/restore
    
    Archive Structure:
        cache/
        ├── active/           # Current working cache (images + metadata)
        │   ├── images/
        │   └── metadata/
        ├── archived/         # Previous templates' caches
        │   ├── template_{id}_{timestamp}/
        │   │   ├── images/
        │   │   └── metadata/
        │   └── ...
        └── archive_index.json  # Maps template IDs to archive paths
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config  # Store full config for diversity checks
        self.cache_dir = Path(config["system"]["cache_dir"])
        self.max_size = config["generation"]["cache"]["max_size"]
        
        # Active cache directories (where current template's frames live)
        self.active_dir = self.cache_dir / "active"
        self.image_dir = self.active_dir / "images"
        self.metadata_dir = self.active_dir / "metadata"
        
        # Archive directory for per-template cache isolation
        self.archive_dir = self.cache_dir / "archived"
        self.archive_index_path = self.cache_dir / "archive_index.json"
        
        # Track current template for cache tagging
        self._current_template_id: Optional[str] = None

        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index
        self.entries: Dict[str, CacheEntry] = {}
        
        # Archive index: maps template_id -> archive directory path
        self._archive_index: Dict[str, str] = {}

        # Maximum archived sessions to keep (oldest pruned first)
        self.max_archived_sessions = config.get("generation", {}).get("cache", {}).get(
            "max_archived_sessions", 10
        )

        # Load existing cache and archive index
        self.load_cache()
        self._load_archive_index()
        self._prune_stale_archives()

        logger.info(
            f"CacheManager initialized: {len(self.entries)}/{self.max_size} entries, "
            f"{len(self._archive_index)} archived templates"
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
            "version": "1.1",  # Bumped for template_id support
            "last_updated": datetime.now().isoformat(),
            "entry_count": len(self.entries),
            "current_template_id": self._current_template_id,
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
    
    def _load_archive_index(self) -> None:
        """Load archive index from disk"""
        if not self.archive_index_path.exists():
            logger.debug("No archive index found, starting fresh")
            return
        
        try:
            with open(self.archive_index_path, "r") as f:
                data = json.load(f)
            
            self._archive_index = data.get("templates", {})
            
            # Validate that archive directories still exist
            valid_templates = {}
            for template_id, archive_path in self._archive_index.items():
                if Path(archive_path).exists():
                    valid_templates[template_id] = archive_path
                else:
                    logger.warning(f"Archive path missing for template '{template_id}': {archive_path}")
            
            self._archive_index = valid_templates
            logger.info(f"Loaded archive index: {len(self._archive_index)} templates")
            
        except Exception as e:
            logger.error(f"Failed to load archive index: {e}", exc_info=True)
            self._archive_index = {}
    
    def _save_archive_index(self) -> None:
        """Save archive index to disk"""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "templates": self._archive_index,
        }
        
        try:
            import tempfile
            
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.cache_dir,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_path = tmp_file.name
            
            shutil.move(tmp_path, str(self.archive_index_path))
            logger.debug("Archive index saved")
            
        except Exception as e:
            logger.error(f"Failed to save archive index: {e}", exc_info=True)

    def _prune_stale_archives(self) -> None:
        """
        Remove oldest archived sessions when total exceeds max_archived_sessions.

        Also cleans up orphaned archive directories that exist on disk but are
        not tracked in the archive index (e.g. from interrupted archival).
        """
        try:
            # Clean orphaned dirs not in the index
            if self.archive_dir.exists():
                indexed_paths = set(Path(p) for p in self._archive_index.values())
                for child in self.archive_dir.iterdir():
                    if child.is_dir() and child not in indexed_paths:
                        logger.info(f"Removing orphaned archive: {child.name}")
                        shutil.rmtree(child)

            # Prune excess indexed archives (keep most recent by timestamp in dir name)
            if len(self._archive_index) <= self.max_archived_sessions:
                return

            # Sort by the timestamp suffix in archive path names (template_{id}_{timestamp})
            def _archive_timestamp(item: tuple) -> int:
                path_str = item[1]
                name = Path(path_str).name
                # Extract trailing timestamp: template_{id}_{timestamp}
                parts = name.rsplit("_", 1)
                try:
                    return int(parts[-1])
                except (ValueError, IndexError):
                    return 0

            sorted_archives = sorted(self._archive_index.items(), key=_archive_timestamp)
            n_to_remove = len(sorted_archives) - self.max_archived_sessions
            removed = 0

            for template_id, archive_path in sorted_archives[:n_to_remove]:
                archive_dir = Path(archive_path)
                if archive_dir.exists():
                    shutil.rmtree(archive_dir)
                del self._archive_index[template_id]
                removed += 1

            if removed > 0:
                self._save_archive_index()
                logger.info(
                    f"Archive pruning: removed {removed} old session(s), "
                    f"{len(self._archive_index)} remaining"
                )

        except Exception as e:
            logger.warning(f"Archive pruning failed (non-fatal): {e}")

    def add(
        self,
        image_path: Path,
        prompt: str,
        generation_params: Dict[str, Any],
        embedding: Optional[Union[Dict[str, Any], List[float]]] = None,
        template_id: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
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
            template_id: Optional identifier for the template that generated this frame.
                        Used for per-template cache isolation.
            components: Optional dict of component values used for this frame.
        
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

        # Create entry with template metadata
        entry = CacheEntry(
            cache_id=cache_id,
            image_path=cached_image_path,
            prompt=prompt,
            generation_params=generation_params,
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            template_id=template_id,
            components=components,
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
    
    # =========================================================================
    # Per-Template Cache Management
    # =========================================================================
    
    def set_current_template(self, template_id: str) -> None:
        """
        Set the current template ID for cache tagging
        
        New cache entries will be tagged with this template ID.
        Does not affect existing entries or trigger archive operations.
        
        Args:
            template_id: Identifier for the current template
        """
        old_template = self._current_template_id
        self._current_template_id = template_id
        
        if old_template != template_id:
            logger.info(f"Cache template context changed: '{old_template}' -> '{template_id}'")
    
    def get_current_template(self) -> Optional[str]:
        """
        Get the current template ID
        
        Returns:
            Current template ID or None if not set
        """
        return self._current_template_id
    
    def archive_current(self, template_id: Optional[str] = None) -> Optional[str]:
        """
        Archive the current active cache for a template
        
        Moves the entire active cache (images + metadata) to an archive directory.
        This preserves the cache state for potential restoration if the template
        is revisited later.
        
        If template_id is not provided, uses the current template ID.
        If no entries exist, does nothing.
        
        Args:
            template_id: Template ID to associate with this archive.
                        Defaults to current template ID if not specified.
        
        Returns:
            Path to the archive directory, or None if nothing to archive
        """
        # Determine template ID
        archive_template_id = template_id or self._current_template_id
        
        if not self.entries:
            logger.debug("No cache entries to archive")
            return None
        
        if not archive_template_id:
            logger.warning("Cannot archive cache: no template ID specified or set")
            return None
        
        # Create archive directory with timestamp
        timestamp = int(datetime.now().timestamp())
        archive_name = f"template_{archive_template_id}_{timestamp}"
        archive_path = self.archive_dir / archive_name
        
        try:
            # Move entire active directory to archive
            logger.info(f"Archiving cache for template '{archive_template_id}' ({len(self.entries)} entries)")
            
            # Create archive structure
            archive_images = archive_path / "images"
            archive_metadata = archive_path / "metadata"
            archive_images.mkdir(parents=True, exist_ok=True)
            archive_metadata.mkdir(parents=True, exist_ok=True)
            
            # Move images
            for entry in self.entries.values():
                if entry.image_path.exists():
                    dest = archive_images / entry.image_path.name
                    shutil.move(str(entry.image_path), str(dest))
            
            # Update entry paths in memory before saving to archive
            for entry in self.entries.values():
                entry.image_path = archive_images / entry.image_path.name
            
            # Save metadata to archive
            archive_metadata_file = archive_metadata / "cache_index.json"
            archive_data = {
                "version": "1.1",
                "archived_at": datetime.now().isoformat(),
                "template_id": archive_template_id,
                "entry_count": len(self.entries),
                "entries": [entry.to_dict() for entry in self.entries.values()],
            }
            with open(archive_metadata_file, "w") as f:
                json.dump(archive_data, f, indent=2)
            
            # Update archive index (may overwrite previous archive for same template)
            old_archive = self._archive_index.get(archive_template_id)
            if old_archive and Path(old_archive).exists():
                # Remove old archive for this template (keep only most recent)
                logger.info(f"  Removing previous archive for '{archive_template_id}'")
                shutil.rmtree(old_archive)
            
            self._archive_index[archive_template_id] = str(archive_path)
            self._save_archive_index()
            
            # Clear in-memory cache (active directory is now empty)
            self.entries.clear()
            
            # Clean up active metadata file
            active_metadata = self.metadata_dir / "cache_index.json"
            if active_metadata.exists():
                active_metadata.unlink()
            
            logger.info(f"  Archived to: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Failed to archive cache: {e}", exc_info=True)
            return None
    
    def restore_archived(self, template_id: str) -> bool:
        """
        Restore a previously archived cache for a template
        
        Moves the archived cache back to the active directory. If there's
        currently an active cache, it should be archived first (caller's
        responsibility).
        
        Args:
            template_id: Template ID whose archive to restore
        
        Returns:
            True if restoration successful, False otherwise
        """
        archive_path_str = self._archive_index.get(template_id)
        
        if not archive_path_str:
            logger.debug(f"No archive found for template '{template_id}'")
            return False
        
        archive_path = Path(archive_path_str)
        
        if not archive_path.exists():
            logger.warning(f"Archive path no longer exists: {archive_path}")
            # Clean up stale index entry
            del self._archive_index[template_id]
            self._save_archive_index()
            return False
        
        try:
            logger.info(f"Restoring cache for template '{template_id}' from {archive_path}")
            
            archive_images = archive_path / "images"
            archive_metadata = archive_path / "metadata"
            
            # Ensure active directories exist and are empty
            if self.entries:
                logger.warning("Active cache not empty during restore - entries will be overwritten")
                self.clear()
            
            # Move images back to active
            if archive_images.exists():
                for img_file in archive_images.iterdir():
                    dest = self.image_dir / img_file.name
                    shutil.move(str(img_file), str(dest))
            
            # Load metadata from archive
            archive_metadata_file = archive_metadata / "cache_index.json"
            if archive_metadata_file.exists():
                with open(archive_metadata_file, "r") as f:
                    data = json.load(f)
                
                # Restore entries with updated paths
                for entry_data in data.get("entries", []):
                    # Update image path to active directory
                    old_path = Path(entry_data["image_path"])
                    entry_data["image_path"] = str(self.image_dir / old_path.name)
                    
                    entry = CacheEntry.from_dict(entry_data)
                    if entry.image_path.exists():
                        self.entries[entry.cache_id] = entry
                    else:
                        logger.warning(f"Restored entry image missing: {entry.cache_id}")
            
            # Remove archive directory (now empty)
            shutil.rmtree(archive_path)
            
            # Remove from archive index
            del self._archive_index[template_id]
            self._save_archive_index()
            
            # Save active cache metadata
            self.save_cache()
            
            # Update current template
            self._current_template_id = template_id
            
            logger.info(f"  Restored {len(self.entries)} entries for template '{template_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore archive: {e}", exc_info=True)
            return False
    
    def has_archived(self, template_id: str) -> bool:
        """
        Check if an archive exists for a template
        
        Args:
            template_id: Template ID to check
        
        Returns:
            True if archive exists
        """
        archive_path = self._archive_index.get(template_id)
        return archive_path is not None and Path(archive_path).exists()
    
    def get_archived_templates(self) -> List[str]:
        """
        Get list of template IDs with existing archives
        
        Returns:
            List of template IDs
        """
        return list(self._archive_index.keys())
    
    def reload(self) -> None:
        """
        Reload cache from disk
        
        Clears in-memory state and reloads from the active cache directory.
        Useful after archive/restore operations or external modifications.
        """
        self.entries.clear()
        self.load_cache()
        self._load_archive_index()
        logger.info(f"Cache reloaded: {len(self.entries)} entries")
    
    def switch_template(
        self,
        new_template_id: str,
        archive_current_template: bool = True,
        restore_if_archived: bool = True
    ) -> Dict[str, Any]:
        """
        Handle template switch with automatic archive/restore
        
        This is the primary method for template-aware cache management.
        On template switch:
        1. Archives the current cache (if archive_current_template=True)
        2. Checks if new template has an existing archive
        3. Restores the archive if available (if restore_if_archived=True)
        4. Otherwise starts with empty cache for new template
        
        Args:
            new_template_id: Template ID to switch to
            archive_current_template: Whether to archive current cache
            restore_if_archived: Whether to restore existing archive for new template
        
        Returns:
            Dictionary with operation results:
            {
                "archived": bool,  # Whether current was archived
                "archive_path": str or None,
                "restored": bool,  # Whether previous archive was restored
                "cache_size": int,  # Current cache size after switch
                "old_template": str or None,
                "new_template": str
            }
        """
        result = {
            "archived": False,
            "archive_path": None,
            "restored": False,
            "cache_size": 0,
            "old_template": self._current_template_id,
            "new_template": new_template_id
        }
        
        # Skip if already on this template
        if self._current_template_id == new_template_id:
            logger.debug(f"Already on template '{new_template_id}', no switch needed")
            result["cache_size"] = len(self.entries)
            return result
        
        logger.info(f"Switching template: '{self._current_template_id}' -> '{new_template_id}'")
        
        # Archive current cache if requested and there are entries
        if archive_current_template and self.entries and self._current_template_id:
            archive_path = self.archive_current(self._current_template_id)
            if archive_path:
                result["archived"] = True
                result["archive_path"] = archive_path
        
        # Check if new template has an archived cache
        if restore_if_archived and self.has_archived(new_template_id):
            restored = self.restore_archived(new_template_id)
            result["restored"] = restored
        else:
            # Start fresh for new template
            self.entries.clear()
            # Save empty cache state
            self.save_cache()
        
        # Update current template
        self._current_template_id = new_template_id
        result["cache_size"] = len(self.entries)
        
        logger.info(
            f"Template switch complete: "
            f"archived={result['archived']}, restored={result['restored']}, "
            f"cache_size={result['cache_size']}"
        )
        
        return result