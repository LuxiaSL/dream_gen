# Auto-Cleanup Feature for Long-Running Sessions
**Date:** 2025-11-17  
**Feature:** Automatic deletion of displayed frames to prevent disk space exhaustion

## Problem Statement

During long-running sessions (hours/days), Dream Window generates frames continuously:
- **Keyframes**: ~1 every 2-3 seconds (depending on interpolation count)
- **Interpolation frames**: 10-20 frames between keyframes
- **Total rate**: ~5-10 frames/minute

**Storage impact:**
- Each frame: ~50-200 KB (PNG, 512×256)
- **1 hour**: ~300-600 frames = 15-120 MB
- **24 hours**: ~7,200-14,400 frames = 360 MB - 2.9 GB
- **1 week**: ~50,000-100,000 frames = **2.5-20 GB**

For 24/7 operation, this is **infeasible** without cleanup!

---

## Solution: Configurable Auto-Cleanup

### How It Works

**Simple & Safe: Delete Immediately After Display**

1. **Copy to display**: Frame is copied to `current_frame.png`
2. **Mark as displayed**: Frame marked in buffer
3. **Delete source**: Original frame file is deleted immediately
4. **Async operation**: Deletion runs in executor (doesn't block display)

**Why this is safe:**
- Frame is already copied to `current_frame.png` (user sees it)
- Frame is already displayed (no longer needed in buffer)
- Only deletes the specific frame just displayed
- Never touches buffered frames (they haven't been displayed yet!)

### Safety Features

✅ **Never deletes buffered frames** - Only the frame just displayed  
✅ **Copy before delete** - Frame is copied to current_frame.png first  
✅ **Immediate deletion** - No bulk operations that could delete wrong frames  
✅ **Graceful failure** - If deletion fails, logs debug message and continues  
✅ **Async operation** - Runs in executor, doesn't block playback  
✅ **Cache untouched** - Cache has its own size limit and LRU eviction  
✅ **Path validation** - Only deletes files in output directory

---

## Configuration

### File: `backend/config.yaml`

```yaml
display:
  # === STORAGE MANAGEMENT ===
  # Auto-cleanup of displayed frames to prevent disk space issues on long runs
  
  cleanup_displayed_frames: true  # Delete frames immediately after display (recommended)
```

### Configuration Guide

**`cleanup_displayed_frames`** (default: `true`)
- `true`: **Recommended** - Delete each frame immediately after it's displayed
- `false`: Keep all frames (useful for debugging, short sessions, or creating animations)

### Storage with Cleanup Enabled

**What's kept on disk:**
- **Buffer frames**: Waiting to be displayed (~30s @ 4 FPS = 120 frames)
- **current_frame.png**: Currently displayed frame
- **Total**: ~120-150 frames max

**Example (30s buffer):**
```yaml
buffer_target_seconds: 30.0  # 120 frames buffered
cleanup_displayed_frames: true
```
- **Disk usage**: ~6-30 MB (stable, regardless of runtime!)

**Example (10s buffer):**
```yaml
buffer_target_seconds: 10.0  # 40 frames buffered  
cleanup_displayed_frames: true
```
- **Disk usage**: ~2-10 MB (stable)

**Example (disabled cleanup):**
```yaml
cleanup_displayed_frames: false
```
- **Disk usage**: Grows indefinitely (use only for short sessions)

---

## Implementation Details

### Code Flow

```
DisplayFrameSelector.select_and_display_next_frame()
  ↓
1. Copy frame to current_frame.png
  ↓
2. Mark as DISPLAYED in buffer
  ↓
3. If cleanup_enabled:
     - Add frame path to displayed_frames_history (deque)
     - Call _cleanup_old_frames_async()
  ↓
4. Advance display pointer
```

### Cleanup Algorithm

```python
# After displaying frame:
1. Copy frame to current_frame.png
2. Mark as DISPLAYED in buffer  
3. Delete source frame immediately

# Pseudocode:
async def select_and_display_next_frame():
    frame = buffer.get_next_display_frame()
    
    # Copy to display
    await copy_to_current_frame(frame.path)
    
    # Delete immediately (frame is now in current_frame.png)
    if cleanup_enabled:
        await delete_frame(frame.path)
```

**Why this works:**
- Frame is safely copied before deletion
- Only deletes the ONE frame just displayed
- Never touches buffered frames (they're still needed!)
- No complex tracking or bulk operations

### Performance Characteristics

**Cleanup frequency**: After every frame display (but only when history full)  
**Operation time**: ~1-5ms (runs in executor, async)  
**I/O pattern**: Sequential deletes (minimal disk seeking)  
**Memory overhead**: ~8 bytes × keep_count (negligible)

---

## Monitoring & Debugging

### Stats Available

```python
display_stats = display_selector.get_stats()
# Returns:
{
    "frames_displayed": 1234,
    "cleanup_enabled": True,
    "frames_deleted": 1184,        # Total frames deleted
    "frames_kept": 50,              # Current history size
    "keep_count": 50                # Configured keep window
}
```

### Log Messages

**Initialization:**
```
DisplayFrameSelector initialized
  Auto-cleanup: ENABLED (delete after display)
```

**Every 10th frame (if cleanup active):**
```
Displayed frame: Seq #140 (keyframe_014.png)
  Buffer: 25.5s (102 frames)
  Cleanup: 140 frames deleted total
```

**Debug logs (per frame displayed):**
```
Deleted displayed frame: keyframe_001.png
Deleted displayed frame: 001-002_005.png
```

---

## Storage Impact Analysis

### Before Auto-Cleanup

| Duration | Frames Generated | Disk Usage (Est.) | Status |
|----------|------------------|-------------------|--------|
| 1 hour   | 600-1,200        | 30-240 MB         | ✅ OK  |
| 6 hours  | 3,600-7,200      | 180 MB - 1.4 GB   | ⚠️ Concern |
| 24 hours | 14,400-28,800    | 720 MB - 5.8 GB   | 🔴 Problem |
| 1 week   | 100,000-200,000  | **5-40 GB**       | 🔴 Critical |

### After Auto-Cleanup (keep_displayed_frames: 50)

| Duration | Frames Kept | Disk Usage (Est.) | Status |
|----------|-------------|-------------------|--------|
| 1 hour   | 170 max     | 8.5-34 MB         | ✅ Stable |
| 6 hours  | 170 max     | 8.5-34 MB         | ✅ Stable |
| 24 hours | 170 max     | 8.5-34 MB         | ✅ Stable |
| 1 week   | 170 max     | **8.5-34 MB**     | ✅ Stable |

**Savings**: 99%+ reduction for long-running sessions!

---

## Edge Cases & Safety

### What Happens If...

**Display is paused?**
- ✅ Cleanup stops (no frames being displayed)
- ✅ Existing frames remain until playback resumes

**Buffer refills during playback?**
- ✅ Buffered frames are safe (not in displayed history yet)
- ✅ Only frames already shown are eligible for cleanup

**File deletion fails?**
- ✅ Logs warning but continues
- ✅ Doesn't crash or stop playback
- ✅ Will retry on next cleanup pass (file might be locked temporarily)

**User wants to create animation later?**
- ✅ Disable cleanup (`cleanup_displayed_frames: false`)
- ✅ Export animation before cleanup fills history
- ✅ Or increase `keep_displayed_frames` to retain more

**Cache directory?**
- ✅ **Not touched!** Cache has separate LRU eviction (max 50 frames)
- ✅ Cache diversity is preserved

---

## Migration Guide

### Upgrading from Previous Versions

**Old behavior** (before this feature):
- Frames accumulated indefinitely
- Required manual cleanup or `max_output_frames` limit
- Disk space issues on long runs

**New behavior** (with auto-cleanup):
1. Add config to `backend/config.yaml`:
   ```yaml
   display:
     cleanup_displayed_frames: true
     keep_displayed_frames: 50
   ```

2. Restart daemon/controller

3. Monitor logs to verify:
   ```
   Auto-cleanup: ENABLED (keep 50 recent frames)
   ```

4. After ~15 minutes, check disk usage stabilizes

### For Animation Export

If you want to export full animations from frame sequences:

**Option 1: Disable cleanup temporarily**
```yaml
cleanup_displayed_frames: false  # Frames accumulate for export
```
Then run `generate_animation.py` to create video.

**Option 2: Increase keep window**
```yaml
keep_displayed_frames: 500  # Keep 500 frames (~2 minutes @ 4 FPS)
```
Export periodically before history fills.

**Option 3: Use separate output directory**
Modify config to output to different directory when exporting.

---

## Comparison with Cache

| Feature | Cache Directory | Output Directory (with cleanup) |
|---------|----------------|----------------------------------|
| **Purpose** | Store diverse frames for injection | Display frames for Rainmeter |
| **Size limit** | Fixed (max 50 frames) | Rolling window (configurable) |
| **Eviction** | LRU (least recently used) | FIFO (oldest displayed) |
| **Selection** | By diversity/dissimilarity | Sequential (playback order) |
| **Cleanup** | Built-in (automatic) | Optional (configurable) |
| **Disk usage** | ~2.5-10 MB (stable) | ~8-34 MB (stable with cleanup) |

Both systems work **independently** and don't interfere!

---

## Testing Checklist

### Before Deployment
- [ ] Set `cleanup_displayed_frames: true` in config
- [ ] Set appropriate `keep_displayed_frames` value
- [ ] Verify logs show "Auto-cleanup: ENABLED"

### During Operation
- [ ] Monitor disk usage (should stabilize after ~5 min)
- [ ] Check logs for cleanup stats every 10th frame
- [ ] Verify playback remains smooth (no stuttering)
- [ ] Check buffer status (should stay full)

### Long-Term Verification
- [ ] Run for 1 hour → Disk usage should be stable
- [ ] Run for 24 hours → Disk usage should remain constant
- [ ] Check cleanup logs for any repeated errors

---

## Troubleshooting

### Issue: Disk space still growing

**Possible causes:**
1. Cleanup disabled in config
2. `keep_displayed_frames` set too high
3. Other processes writing to output directory

**Solutions:**
```bash
# Check config
grep "cleanup_displayed_frames" backend/config.yaml

# Check current frame count
ls output/keyframes/ | wc -l
ls output/interpolations/ | wc -l

# Check disk usage
du -sh output/

# Enable debug logging
# Set log_level: DEBUG in config.yaml
```

### Issue: Frames deleted too aggressively

**Symptoms:**
- Playback stutters
- Buffer empties unexpectedly

**Solutions:**
1. Increase `keep_displayed_frames`:
   ```yaml
   keep_displayed_frames: 100  # Was 50
   ```

2. Increase `buffer_target_seconds`:
   ```yaml
   buffer_target_seconds: 45.0  # Was 30.0
   ```

### Issue: File deletion errors in logs

**Example:**
```
WARNING - Failed to delete keyframe_042.png: Permission denied
```

**Possible causes:**
- File locked by antivirus
- File opened in viewer/editor
- Permission issues

**Solutions:**
- Usually transient, will retry next pass
- Check antivirus exclusions for `output/` directory
- Ensure no other processes have files open

---

## Performance Impact

### CPU Usage
- **Cleanup overhead**: <0.1% CPU (runs in executor)
- **Display loop**: No change (cleanup is async)

### Memory Usage
- **Deque overhead**: ~400 bytes (50 frames × 8 bytes/pointer)
- **Negligible impact**

### Disk I/O
- **Delete rate**: ~1-10 files/second (depending on frame rate)
- **I/O pattern**: Sequential (minimal seeking)
- **Impact**: Minimal (modern SSDs handle this easily)

---

## Future Enhancements

### Potential Improvements

1. **Batch cleanup** (reduce I/O):
   ```python
   # Instead of deleting on every frame, batch every N frames
   if len(to_delete) > 10:
       delete_batch(to_delete)
   ```

2. **Configurable cleanup interval**:
   ```yaml
   cleanup_interval_frames: 10  # Only cleanup every 10 frames
   ```

3. **Disk space monitoring**:
   ```python
   # Only cleanup if disk space < threshold
   if get_free_space() < min_free_space:
       cleanup_old_frames()
   ```

4. **Selective cleanup** (keep keyframes longer):
   ```yaml
   keep_keyframes: 100
   keep_interpolations: 20
   ```

---

## Summary

✅ **Enabled by default** (`cleanup_displayed_frames: true`)  
✅ **Safe** (keeps configurable safety buffer)  
✅ **Async** (doesn't block playback)  
✅ **Effective** (99%+ disk space savings on long runs)  
✅ **Configurable** (adjust keep window to your needs)

This feature makes Dream Window suitable for **true 24/7 operation** without disk space concerns!

---

**End of Documentation**

