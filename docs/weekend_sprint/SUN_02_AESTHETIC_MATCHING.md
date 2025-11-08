# ⏱️ Sunday Session 2: Aesthetic Matching

**Goal**: CLIP-based similarity for cache injection

**Duration**: 2 hours

---

## Overview

Create `backend/cache/aesthetic_matcher.py` - finds visually similar images using CLIP embeddings.

### Key Components

1. **AestheticMatcher class**
2. CLIP model loading
3. Image encoding
4. Similarity computation
5. Weighted random selection

### What It Does

- Encodes images to 512-dim vectors
- Computes cosine similarity
- Finds similar cached images
- Enables aesthetic cache injection

---

## Core Methods

### `encode_image()`

Converts image to CLIP embedding:
1. Load image
2. Preprocess for CLIP
3. Pass through vision encoder
4. Normalize to unit vector
5. Return 512-dim array

### `find_similar()`

Finds cached images matching current aesthetic:
1. Compare target embedding with all cached
2. Compute cosine similarities
3. Filter by threshold (0.7+)
4. Sort by similarity
5. Return top K matches

---

## Implementation

**Full code**: See original WEEKEND_SPRINT.md lines 1400-1650

Key libraries:
- `transformers` - CLIP model
- `torch` - GPU operations
- `numpy` - Vector math

---

## Integration with Main Loop

```python
# Every 15 frames, inject similar cached image
if frame_count % 15 == 0:
    current_embedding = matcher.encode_image(current_frame)
    similar = matcher.find_similar(current_embedding, cache.get_all())
    if similar:
        injection_image = similar[0]  # Most similar
        current_image = injection_image
```

---

## Validation

```powershell
python -c "from backend.cache.aesthetic_matcher import AestheticMatcher; am = AestheticMatcher(); print('✓ CLIP model loaded')"
```

First run will download ~600MB CLIP model.

---

## Success Criteria

- [ ] Aesthetic matcher created
- [ ] CLIP model loaded
- [ ] Can encode images
- [ ] Similarity computation working
- [ ] Cache injection functional

---

## 🎉 Sunday Morning Milestone!

**Milestone 3**: Intelligent Generation

You now have:
- ✅ Cache system storing images
- ✅ CLIP embeddings for similarity
- ✅ Aesthetic injection preventing mode collapse

---

**Next**: SUN_03_RAINMETER_WIDGET.md

---

**Session 6 of 8** | ~2 hours

