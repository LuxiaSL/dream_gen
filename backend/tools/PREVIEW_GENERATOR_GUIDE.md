# Looped Preview Generator - Quick Start Guide

## Overview

The `generate_looped_preview.py` tool creates a compact, looping GIF animation perfect for GitHub README previews or project showcases. It interpolates between two selected keyframes using your existing interpolation infrastructure, creating a smooth, continuous loop under 1MB.

## What It Does

```
Pattern: KeyframeA → [InterpolationFrames] → KeyframeB → [InterpolationFrames Reversed] → Loop
```

### Example Run
- **Input**: Keyframes 024 and 010
- **Output**: `preview.gif` (0.77MB, 22 frames, 4.4 seconds at 5 FPS)
- **Process**: 
  1. Encode both keyframes to VAE latent space
  2. Generate 10 smooth interpolation frames from A→B using spherical interpolation
  3. Add keyframe B
  4. Generate 10 reverse interpolation frames from B→A
  5. Loop seamlessly back to keyframe A
  6. Compress as GIF with automatic optimization

## Quick Start

### Default (Keyframes 024 & 010)
```bash
uv run backend/tools/generate_looped_preview.py
# Creates: output/preview.gif (0.77MB)
```

### Custom Keyframes
```bash
uv run backend/tools/generate_looped_preview.py --keyframe-a 30 --keyframe-b 15
```

### Smoother Animation (More Frames)
```bash
uv run backend/tools/generate_looped_preview.py --interpolation-frames 20
# More frames = smoother but slightly larger file
```

### Faster Animation
```bash
uv run backend/tools/generate_looped_preview.py --fps 8
# Plays faster (5 FPS is default)
```

### Custom Output
```bash
uv run backend/tools/generate_looped_preview.py --output my_preview.gif
```

### More Aggressive Size Optimization
```bash
uv run backend/tools/generate_looped_preview.py --max-size 0.8
# Target 0.8MB instead of 1.0MB
```

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--keyframe-a` | 24 | First keyframe number |
| `--keyframe-b` | 10 | Second keyframe number |
| `--interpolation-frames` | 10 | Frames between keyframes (more = smoother) |
| `--fps` | 5 | Playback speed in frames per second |
| `--output` | `preview.gif` | Output filename |
| `--output-dir` | `output` | Output directory |
| `--input-dir` | `output` | Input keyframes directory |
| `--max-size` | 1.0 | Target max file size in MB |
| `--resolution` | Auto | Override resolution (e.g., `256x128`) |

## How It Works - Deep Dive

### Phase 1: Setup
- Locates keyframes by number (e.g., `keyframe_024.png`, `keyframe_010.png`)
- Determines resolution from first keyframe
- Initializes VAE encoder (SD 1.5 on CUDA if available)

### Phase 2: Encoding
- Encodes both keyframes to latent space using VAE
- Precomputes spherical interpolation (slerp) parameters for efficiency

### Phase 3: Interpolation
- Generates interpolation frames A→B using spherical lerp
- Reverses the process for B→A to create seamless loop
- Each frame is decoded back to image space

### Phase 4: Compression
- **Attempt 1**: Standard GIF optimization
- **Attempt 2**: If oversized, downsamples frames 2x (256x128)
- **Attempt 3**: If still oversized, decimates frames (keeps every 2nd frame)
- Automatically picks smallest version that meets size target

## Size Optimization Tips

### If file is too large:
1. **Reduce interpolation frames**: `--interpolation-frames 5` (default: 10)
2. **Lower FPS**: `--fps 3` (default: 5)
3. **Stricter size limit**: `--max-size 0.5` (default: 1.0)
4. **Lower resolution**: `--resolution 256x128` (default: 512x256)

### If animation looks jittery:
1. **Increase interpolation frames**: `--interpolation-frames 15` (smoother transitions)
2. **Increase FPS**: `--fps 8` (smoother playback)

### Typical Results
- **22 frames, 512x256, 5 FPS**: ~0.77MB ✓
- **22 frames, 256x128, 5 FPS**: ~0.20MB ✓✓
- **32 frames, 512x256, 8 FPS**: ~1.2MB (requires optimization)

## Real-World Example

Create a GitHub preview for a design showcase:

```bash
# High quality, smooth loop
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 50 \
  --keyframe-b 30 \
  --interpolation-frames 15 \
  --fps 6 \
  --output design_preview.gif
```

Then add to README.md:
```markdown
## Preview

![Design Animation](output/design_preview.gif)
```

## Troubleshooting

### "Keyframe not found"
- Check keyframe number exists in `output/keyframes/`
- Use zero-padded format: `--keyframe-a 24` (not `--keyframe-a 024`)

### "File size exceeds limit"
- Tool automatically downsamples, but if you want control:
  - Reduce `--interpolation-frames`
  - Reduce `--fps`
  - Add `--resolution 256x128`

### VAE Loading Issues
- Requires `diffusers` package (included in project)
- Uses SD 1.5 VAE from HuggingFace (auto-downloaded)
- First run takes longer to download model

### GPU Memory Issues
- Tool uses 165MB VRAM (fp16 VAE)
- Falls back to CPU if CUDA unavailable (much slower)

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| VAE Init | ~30s | First run only (downloads model) |
| Encode keyframes | ~4s | Cached by default |
| Slerp precompute | <1s | One-time per pair |
| Generate frames | ~5s | Depends on interpolation count |
| GIF encoding | ~4s | Includes compression |
| **Total** | **~13-45s** | Subsequent runs faster |

## Integration Ideas

### 1. GitHub README Preview
```bash
uv run backend/tools/generate_looped_preview.py --output github_preview.gif
# Add to README.md: ![Preview](output/github_preview.gif)
```

### 2. Social Media Clip
```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 60 --keyframe-b 40 \
  --fps 8 --interpolation-frames 15 \
  --output social_clip.gif
```

### 3. Project Showcase Loop
```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 75 --keyframe-b 25 \
  --interpolation-frames 20 --fps 4 \
  --output showcase_loop.gif
```

### 4. Batch Generate Multiple Previews
```bash
for kf_a in 20 30 40; do
  for kf_b in 10 15 25; do
    uv run backend/tools/generate_looped_preview.py \
      --keyframe-a $kf_a --keyframe-b $kf_b \
      --output preview_${kf_a}_${kf_b}.gif
  done
done
```

## File Format Details

- **Format**: GIF (Universal compatibility)
- **Max Size**: 1.0MB (GitHub compatible)
- **Compression**: Pillow optimize + Potential downsampling
- **Color Space**: RGB (24-bit)
- **Loop**: Infinite (0 loop count = continuous)

## Related Tools

- **`generate_animation.py`**: Full sequence animation (all keyframes)
- **`profile_interpolation.py`**: Detailed performance analysis
- **`test_quality_comparison.py`**: Quality metrics and comparisons

---

**Version**: 1.0  
**Created**: November 15, 2025  
**Tested**: ✓ Keyframes 024 & 010 → 0.77MB under 1MB target

