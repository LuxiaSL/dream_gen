# ⏱️ Sunday Session 1: Cache System

**Goal**: Store and manage generated images

**Duration**: 2 hours

---

## Overview

Create `backend/cache/manager.py` - cache management with metadata.

### Key Components

1. **CacheManager class**
2. **CacheEntry dataclass**
3. LRU eviction
4. Metadata persistence (JSON)

### What It Does

- Stores generated images in cache/images/
- Tracks metadata (prompt, params, timestamp)
- Enforces max cache size
- Persists index to disk

---

## Core Features

### Cache Operations

- `add()` - Add image to cache
- `get()` - Retrieve by ID
- `get_all()` - List all entries
- `get_random()` - Random selection

### Metadata Structure

```json
{
  "cache_id": "cache_00001",
  "image_path": "cache/images/cache_00001.png",
  "prompt": "ethereal angel...",
  "generation_params": {"denoise": 0.4, "steps": 4},
  "timestamp": "2025-11-08T10:30:00"
}
```

---

## Implementation

**Full code**: See original WEEKEND_SPRINT.md lines 1050-1350

Key aspects:
- JSON persistence to `cache/metadata/cache_index.json`
- Atomic file operations
- LRU eviction when max_size reached

---

## Integration with Generator

Modify `generator.py` to add images to cache after generation:

```python
# After generating image
cache_manager.add(
    image_path=output_path,
    prompt=prompt,
    generation_params={"denoise": denoise, "steps": steps}
)
```

---

## Validation

```powershell
python -c "from backend.cache.manager import CacheManager; cm = CacheManager({'system': {'cache_dir': 'cache'}, 'generation': {'cache': {'max_size': 75}}}); print('✓ Cache manager ready')"
```

---

## Success Criteria

- [ ] Cache manager created
- [ ] Images being cached
- [ ] Metadata persisted
- [ ] LRU eviction working

---

**Next**: SUN_02_AESTHETIC_MATCHING.md

---

**Session 5 of 8** | ~2 hours

