# Async Cache Monitoring System Design
## Future Enhancement Proposal

### Overview

Current cache system uses synchronous, average-based similarity checks and LRU eviction. The proposed async system would enable continuous cache quality monitoring and intelligent eviction/acceptance strategies.

---

## Current System Limitations

### 1. Average Similarity Problem
```python
# Current logic in should_cache_frame():
avg_color_sim = np.mean(color_sims)  # Average across all cache
avg_struct_sim = np.mean(struct_sims)

is_diverse = (avg_color_sim < threshold) and (avg_struct_sim < threshold)
```

**Issues:**
- Average hides local clusters of similar frames
- A frame can be "diverse on average" but duplicate existing frames
- No visibility into cache internal diversity
- Can't detect when multiple near-duplicates accumulate

### 2. LRU Eviction Problem
```python
# Current logic in _enforce_max_size():
sorted_entries = sorted(entries, key=lambda x: x.timestamp)  # Oldest first
# Remove oldest frames
```

**Issues:**
- Age != redundancy
- Might evict unique old frames while keeping recent duplicates
- No consideration of cache diversity when evicting
- Passive rather than proactive quality control

### 3. Synchronous Bottleneck
- All similarity calculations block generation
- Can't run continuous background monitoring
- Limited computational budget per frame

---

## Proposed Async Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                 Generation Pipeline                       │
│                    (Main Thread)                         │
└────────────┬────────────────────────────────────────────┘
             │
             ├─► Keyframe Generated
             │   ├─ Quick embedding calculation
             │   ├─ Async cache quality check (non-blocking)
             │   └─ Continue to next frame
             │
             v
┌─────────────────────────────────────────────────────────┐
│              Cache Health Monitor                        │
│                (Background Thread)                       │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Continuous Tasks (async/await):                │   │
│  │  • Calculate pairwise cache diversity           │   │
│  │  • Identify least diverse frames                │   │
│  │  • Monitor cache quality metrics                │   │
│  │  • Update eviction candidates                   │   │
│  │  • Recommend acceptance thresholds              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  Metrics:                                               │
│  • Internal cache diversity score                       │
│  • Cluster detection (similar frame groups)            │
│  • Per-frame redundancy scores                          │
│  • Acceptance rate trends                               │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Continuous Cache Quality Monitoring

**Pairwise Diversity Matrix:**
```python
class AsyncCacheMonitor:
    async def calculate_diversity_matrix(self):
        """
        Build NxN matrix of all cache frame similarities.
        Update incrementally as cache changes.
        """
        cache_frames = self.cache.get_all()
        n = len(cache_frames)
        
        # Parallel similarity calculations
        tasks = []
        for i in range(n):
            for j in range(i+1, n):
                task = self._async_similarity(cache_frames[i], cache_frames[j])
                tasks.append(task)
        
        # Await all in parallel
        similarities = await asyncio.gather(*tasks)
        
        # Build symmetric matrix
        self.diversity_matrix = self._build_matrix(similarities, n)
        
    async def _async_similarity(self, frame_a, frame_b):
        """Non-blocking similarity calculation"""
        color_sim = self.similarity_manager.get_color_similarity(
            frame_a.embedding, frame_b.embedding
        )
        struct_sim = self.similarity_manager.get_struct_similarity(
            frame_a.embedding, frame_b.embedding
        )
        return (color_sim, struct_sim)
```

**Per-Frame Redundancy Score:**
```python
async def calculate_redundancy_scores(self):
    """
    For each cache frame, calculate how redundant it is.
    
    Redundancy = max similarity to any other frame
    (or avg of top-3 most similar frames)
    """
    scores = {}
    
    for frame_id in self.cache.entries.keys():
        # Get similarities to all other frames
        sims = self.diversity_matrix[frame_id]
        
        # Redundancy options:
        # Option 1: Maximum similarity (strictest)
        max_color = np.max(sims['color'])
        max_struct = np.max(sims['struct'])
        
        # Option 2: Average of top-3 (balanced)
        top3_color = np.mean(sorted(sims['color'])[-3:])
        top3_struct = np.mean(sorted(sims['struct'])[-3:])
        
        # Combined redundancy score (higher = more redundant)
        redundancy = (max_color / 2.5 + max_struct) / 2
        scores[frame_id] = redundancy
    
    return scores
```

### 2. Intelligent Eviction Strategy

**Smart Eviction (instead of LRU):**
```python
def get_eviction_candidate(self) -> str:
    """
    Select frame to evict based on redundancy, not age.
    
    Priority order:
    1. Most redundant frame (highest similarity to others)
    2. If tie, prefer newer frames (preserve old unique ones)
    3. If still tie, use random selection
    """
    redundancy_scores = await self.monitor.get_redundancy_scores()
    
    # Find most redundant frame
    max_redundancy = max(redundancy_scores.values())
    candidates = [
        frame_id for frame_id, score in redundancy_scores.items()
        if score >= max_redundancy - 0.05  # Within 5% of max
    ]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Tie-breaker: prefer newer frames (keep old unique ones)
    entries = [self.cache.get(cid) for cid in candidates]
    entries_sorted = sorted(entries, key=lambda x: x.timestamp, reverse=True)
    
    logger.info(
        f"Evicting {entries_sorted[0].cache_id} "
        f"(redundancy: {max_redundancy:.3f})"
    )
    return entries_sorted[0].cache_id
```

### 3. Enhanced Acceptance Logic

**Maximum Similarity Check (instead of Average):**
```python
async def should_cache_frame_smart(self, new_embedding) -> tuple[bool, dict]:
    """
    Improved acceptance logic using maximum similarity.
    
    Returns:
        (should_cache: bool, stats: dict)
    """
    cache_frames = self.cache.get_all()
    
    if len(cache_frames) == 0:
        return True, {"reason": "empty_cache"}
    
    # Calculate similarity to ALL cache frames (parallel)
    tasks = [
        self._async_similarity_to_cache(new_embedding, frame)
        for frame in cache_frames
    ]
    similarities = await asyncio.gather(*tasks)
    
    color_sims = [s[0] for s in similarities]
    struct_sims = [s[1] for s in similarities]
    
    # Use MAXIMUM similarity (strictest check)
    max_color_sim = np.max(color_sims)
    max_struct_sim = np.max(struct_sims)
    
    # Also track average for comparison
    avg_color_sim = np.mean(color_sims)
    avg_struct_sim = np.mean(struct_sims)
    
    # AND logic with maximum similarity
    color_diverse = max_color_sim < self.color_threshold
    struct_diverse = max_struct_sim < self.struct_threshold
    
    is_diverse = color_diverse and struct_diverse
    
    stats = {
        "max_color_sim": max_color_sim,
        "max_struct_sim": max_struct_sim,
        "avg_color_sim": avg_color_sim,
        "avg_struct_sim": avg_struct_sim,
        "most_similar_frame": cache_frames[np.argmax(color_sims)].cache_id,
        "is_diverse": is_diverse,
        "reason": "max_similarity_check"
    }
    
    if is_diverse:
        logger.debug(
            f"Frame diverse (max color:{max_color_sim:.3f}, "
            f"struct:{max_struct_sim:.3f}) - caching"
        )
    else:
        logger.debug(
            f"Frame too similar to {stats['most_similar_frame']} "
            f"(color:{max_color_sim:.3f}, struct:{max_struct_sim:.3f}) - rejecting"
        )
    
    return is_diverse, stats
```

### 4. Cluster Detection

**Identify Groups of Similar Frames:**
```python
async def detect_clusters(self, similarity_threshold: float = 0.10):
    """
    Find clusters of similar frames in cache.
    
    Uses hierarchical clustering on diversity matrix.
    Returns groups of frames that are too similar to each other.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    
    # Build condensed distance matrix
    distances = []
    n = len(self.cache.entries)
    
    for i in range(n):
        for j in range(i+1, n):
            # Convert similarity to distance
            color_dist = self.diversity_matrix[i][j]['color']
            struct_dist = self.diversity_matrix[i][j]['struct']
            
            # Combined distance (lower = more similar)
            dist = (color_dist / 2.5 + struct_dist) / 2
            distances.append(dist)
    
    # Hierarchical clustering
    linkage_matrix = linkage(distances, method='average')
    clusters = fcluster(linkage_matrix, similarity_threshold, criterion='distance')
    
    # Group frames by cluster
    frame_ids = list(self.cache.entries.keys())
    cluster_groups = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(frame_ids[idx])
    
    # Find problematic clusters (>1 frame)
    problematic = {
        cid: frames for cid, frames in cluster_groups.items()
        if len(frames) > 1
    }
    
    if problematic:
        logger.warning(
            f"Detected {len(problematic)} clusters of similar frames: "
            f"{[(cid, len(frames)) for cid, frames in problematic.items()]}"
        )
    
    return problematic
```

### 5. Adaptive Threshold Tuning

**Auto-adjust Thresholds Based on Cache Quality:**
```python
async def recommend_thresholds(self) -> dict:
    """
    Analyze cache quality and recommend threshold adjustments.
    
    Goals:
    - Maintain 25-30% acceptance rate
    - Keep internal diversity high (avg redundancy < 0.4)
    - Prevent cluster formation
    """
    stats = await self.get_cache_stats()
    
    recommendations = {
        "color_threshold": self.color_threshold,
        "struct_threshold": self.struct_threshold,
        "adjustments": []
    }
    
    # Check 1: Acceptance rate too high
    if stats['acceptance_rate'] > 0.30:
        recommendations['color_threshold'] -= 0.02
        recommendations['struct_threshold'] -= 0.01
        recommendations['adjustments'].append(
            "Acceptance rate high (>30%) - tightening thresholds"
        )
    
    # Check 2: Internal redundancy too high
    if stats['avg_redundancy'] > 0.40:
        recommendations['color_threshold'] -= 0.03
        recommendations['struct_threshold'] -= 0.02
        recommendations['adjustments'].append(
            "Cache redundancy high - tightening thresholds"
        )
    
    # Check 3: Clusters detected
    if stats['num_clusters'] > 3:
        recommendations['adjustments'].append(
            f"Detected {stats['num_clusters']} clusters - consider evicting cluster members"
        )
    
    # Check 4: Acceptance rate too low
    if stats['acceptance_rate'] < 0.20:
        recommendations['color_threshold'] += 0.02
        recommendations['struct_threshold'] += 0.01
        recommendations['adjustments'].append(
            "Acceptance rate low (<20%) - loosening thresholds"
        )
    
    return recommendations
```

---

## Performance Considerations

### Computational Complexity

**Current System:**
- O(N) similarity calculations per new frame (N = cache size)
- Blocking, runs during generation

**Async System:**
- O(N²) diversity matrix calculation (background, periodic)
- O(N) per-frame check (non-blocking)
- Eviction: O(1) lookup (pre-computed scores)

**Optimization Strategies:**
1. **Incremental matrix updates**: Only recalculate when cache changes
2. **Lazy evaluation**: Calculate redundancy scores on-demand
3. **Caching**: Store computed similarities between refreshes
4. **Parallel processing**: Use asyncio.gather() for batch operations
5. **Sampling**: For very large caches (>100), sample subset

### Resource Budget

Assuming 50-frame cache:
- Diversity matrix: 50×50 = 2,500 pairwise comparisons
- At ~1ms per comparison: 2.5 seconds total
- Spread over 10 keyframes: 250ms per keyframe (background)
- **Impact on generation: 0ms (fully async)**

---

## Migration Path

### Phase 1: Add Async Monitor (Non-Breaking)
```python
# Add alongside existing system
cache_monitor = AsyncCacheMonitor(cache, similarity_manager)
asyncio.create_task(cache_monitor.start_monitoring())

# Log recommendations but don't act on them yet
recommendations = await cache_monitor.recommend_thresholds()
logger.info(f"Threshold recommendations: {recommendations}")
```

### Phase 2: Switch to Smart Eviction
```python
# Replace LRU with redundancy-based eviction
if cache.size() >= cache.max_size:
    evict_id = await cache_monitor.get_eviction_candidate()
    cache.remove(evict_id)
```

### Phase 3: Enhanced Acceptance Logic
```python
# Use maximum similarity instead of average
is_diverse, stats = await cache_monitor.should_cache_frame_smart(embedding)
if is_diverse:
    cache.add(frame, embedding)
```

### Phase 4: Full Adaptive System
```python
# Auto-tune thresholds based on cache health
if keyframe_num % 50 == 0:  # Every 50 keyframes
    recommendations = await cache_monitor.recommend_thresholds()
    if recommendations['adjustments']:
        apply_threshold_adjustments(recommendations)
```

---

## Monitoring Metrics

### Real-Time Dashboard Data

```python
{
    # Cache quality
    "cache_size": 50,
    "avg_internal_diversity": 0.35,  # Lower = more similar frames
    "num_clusters": 2,
    "cluster_sizes": [3, 2],
    
    # Acceptance tracking
    "recent_acceptance_rate": 0.27,
    "frames_cached_last_50": 14,
    "frames_rejected_last_50": 36,
    
    # Redundancy distribution
    "min_redundancy": 0.15,
    "max_redundancy": 0.68,
    "avg_redundancy": 0.35,
    "frames_with_redundancy_>0.5": 8,
    
    # Eviction stats
    "eviction_method": "redundancy_based",
    "last_evicted_frame": "cache_00023",
    "eviction_redundancy_score": 0.68,
    
    # Threshold health
    "color_threshold": 1.92,
    "struct_threshold": 0.80,
    "threshold_adjustment_needed": False,
    
    # Performance
    "diversity_matrix_age_seconds": 12.5,
    "last_full_scan_duration_ms": 2234,
}
```

---

## Benefits Summary

### Cache Quality
- ✅ Eliminates near-duplicates (maximum similarity check)
- ✅ Prevents cluster formation (continuous monitoring)
- ✅ Maintains high internal diversity (smart eviction)

### Performance
- ✅ Non-blocking cache checks (async)
- ✅ Parallel similarity calculations (faster)
- ✅ Computational headroom for deeper analysis

### Adaptability
- ✅ Auto-tuning thresholds (responsive to aesthetic shifts)
- ✅ Early warning system (cluster detection)
- ✅ Real-time quality metrics (visibility)

### Operational
- ✅ Better cache utilization (evict redundant, keep unique)
- ✅ Predictable quality (fewer surprises)
- ✅ Data-driven tuning (metrics, not guessing)

---

## Recommended First Steps

1. **Implement async diversity matrix calculation**
   - Background task that updates every N keyframes
   - Start with simple pairwise calculations
   - Log metrics without acting on them

2. **Add redundancy score tracking**
   - Per-frame scores based on diversity matrix
   - Expose via cache.get_stats()
   - Monitor in logs to validate approach

3. **Switch eviction strategy**
   - Replace LRU with redundancy-based
   - Compare results visually
   - Validate improvement in cache quality

4. **Test max similarity acceptance**
   - Run parallel to average-based
   - Compare acceptance rates
   - Adjust thresholds if needed

5. **Deploy cluster detection**
   - Run periodically (every 50 keyframes)
   - Log warnings when clusters form
   - Manual inspection of clustered frames

---

**Status**: Design specification  
**Priority**: Medium-High (significant quality improvement)  
**Complexity**: Moderate (async architecture, numpy matrix ops)  
**Risk**: Low (can run alongside existing system for validation)


