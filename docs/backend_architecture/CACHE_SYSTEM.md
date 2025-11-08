# 🏗️ Cache System

**Image storage and aesthetic matching**

---

## 💾 CacheManager (cache/manager.py)

### Purpose
Store and retrieve generated images with metadata. Prevents mode collapse by enabling cache injection.

### Key Components

#### CacheEntry Dataclass

```python
@dataclass
class CacheEntry:
    cache_id: str
    image_path: Path
    prompt: str
    generation_params: Dict
    embedding: Optional[List[float]]  # CLIP embedding
    timestamp: str
```

### Key Methods

#### `add(image_path, prompt, generation_params, embedding)`
Adds image to cache:
1. Generate unique cache_id
2. Copy image to cache/images/
3. Create CacheEntry
4. Add to in-memory index
5. Enforce max_size (LRU eviction)
6. Persist to disk

#### `get_all()`
Returns list of all CacheEntry objects

#### `get_random()`
Returns random cached image

#### `_enforce_max_size()`
LRU Eviction:
- Sort by timestamp (oldest first)
- Remove oldest entries
- Delete both image file and metadata

### Metadata Persistence

Stored in `cache/metadata/cache_index.json`:

```json
{
  "version": "1.0",
  "last_updated": "2025-11-08T10:30:00",
  "entry_count": 75,
  "entries": [
    {
      "cache_id": "cache_00001",
      "image_path": "cache/images/cache_00001.png",
      "prompt": "ethereal angel...",
      "generation_params": {"denoise": 0.4, "steps": 4},
      "embedding": [0.234, -0.123, ... ],  // 512 values
      "timestamp": "2025-11-08T09:15:00"
    }
  ]
}
```

### Design Pattern
**Repository Pattern** - Abstracts data storage, provides collection-like interface

---

## 🎨 AestheticMatcher (cache/aesthetic_matcher.py)

### Purpose
Find visually similar images using CLIP embeddings for intelligent cache injection.

### How CLIP Works

CLIP (Contrastive Language-Image Pre-training):
- Vision encoder: Image → 512-dim embedding
- Text encoder: Text → 512-dim embedding
- Embeddings in same latent space
- Cosine similarity measures visual similarity

### Key Methods

#### `encode_image(image_path)`
Encodes image to CLIP embedding:
1. Load and preprocess image
2. Pass through CLIP vision encoder
3. Normalize to unit vector
4. Return 512-dimensional numpy array

#### `find_similar(target_embedding, candidate_embeddings, threshold, top_k)`
Finds similar cached images:
1. Compute cosine similarity with all candidates
2. Filter by threshold (e.g., 0.7)
3. Sort by similarity (high to low)
4. Return top K matches

#### `cosine_similarity(embedding1, embedding2)`
For normalized vectors:
```python
similarity = np.dot(embedding1, embedding2)
```
Returns value from -1 (opposite) to 1 (identical)

#### `weighted_random_selection(candidates)`
Selects from candidates with probability proportional to similarity:
- Higher similarity = higher probability
- Still allows variation

### Similarity Interpretation

```
0.9 - 1.0:  Nearly identical
0.8 - 0.9:  Very similar (same subject, pose)
0.7 - 0.8:  Similar aesthetic (good for injection)
0.6 - 0.7:  Related style
< 0.6:      Different aesthetic
```

### Usage in Generation Loop

```python
# Every N frames
if frame_count % 15 == 0:
    # Encode current frame
    current_emb = matcher.encode_image(current_frame)
    
    # Find similar in cache
    cache_entries = cache.get_all()
    similar = matcher.find_similar(
        target_embedding=current_emb,
        candidate_embeddings=[(e.cache_id, e.embedding) for e in cache_entries],
        threshold=0.7,
        top_k=5
    )
    
    # Inject if found
    if similar:
        selected = matcher.weighted_random_selection(similar)
        injection_image = cache.get(selected)
        # Use injection_image as next keyframe
```

### Design Pattern
**Strategy Pattern** - Encapsulates matching algorithm, could swap for other models

---

## 🔄 Cache Injection Flow

```
Generation Loop
    │
    ├──> Generate frame N
    │
    ├──> Encode to CLIP embedding
    │
    ├──> Add to cache (CacheManager)
    │
    ├──> Check: frame_count % injection_interval == 0?
    │        │
    │        ├──[YES]──> Find similar (AestheticMatcher)
    │        │              │
    │        │              ├──> Get all cache entries
    │        │              ├──> Compute similarities
    │        │              ├──> Filter by threshold
    │        │              └──> Weighted random selection
    │        │
    │        └──> Use injection as next keyframe
    │
    └──> Continue to frame N+1
```

---

## 📊 Performance Notes

**CLIP Encoding**:
- Time: 100-200ms per image
- VRAM: ~600MB model size
- First run: Downloads model (~600MB)

**Similarity Computation**:
- Time: < 1ms for 75 images
- Pure numpy operations
- Very fast with normalized vectors

**Cache Operations**:
- Add: ~50ms (includes file copy)
- Get: < 1ms (in-memory lookup)
- Persist: ~10-50ms (JSON write)

---

## 🧪 Testing

### Unit Tests
- Test cache add/get operations
- Test LRU eviction
- Test CLIP encoding
- Test similarity computation

### Integration Tests
- Test cache injection in main loop
- Verify aesthetic coherence maintained
- Test with full cache (75+ images)

---

**Next**: [UTILITIES.md](UTILITIES.md) for file operations and system monitoring

