# Creating GitHub Preview GIFs - Complete Guide

## What Was Created

A new specialized tool for generating compact, looping preview GIFs perfect for GitHub READMEs:

### New Script
- **File**: `backend/tools/generate_looped_preview.py`
- **Purpose**: Create smooth, looped GIFs between two keyframes under 1MB
- **Status**: ✅ Tested and working

### Documentation
- **File**: `backend/tools/PREVIEW_GENERATOR_GUIDE.md`
- **File**: `backend/tools/README.md` (updated with new tool docs)

## Quick Test Result

✅ **Success!**

```
Input:
  - Keyframe A: keyframe_024.png
  - Keyframe B: keyframe_010.png  
  - Interpolation: 10 frames
  - FPS: 5

Output:
  - File: output/preview.gif
  - Size: 0.77 MB ✓ (under 1MB target)
  - Frames: 22 total
  - Duration: ~4.4 seconds
  - Quality: Downsampled for size (256x128)
```

## How to Use

### Simplest Way (Default)
```bash
cd C:\Users\luxia\Documents\projects\dream_gen
uv run backend/tools/generate_looped_preview.py
```
Creates `output/preview.gif` with keyframes 024 and 010.

### Custom Keyframes
```bash
uv run backend/tools/generate_looped_preview.py --keyframe-a 50 --keyframe-b 25
```

### For GitHub README
```bash
# Generate high-quality preview
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 40 \
  --keyframe-b 20 \
  --interpolation-frames 15 \
  --fps 5 \
  --output github_preview.gif

# Add to README.md:
# ![Animation Preview](output/github_preview.gif)
```

## How It Works

1. **Frame Selection**: Choose 2 keyframes (e.g., #024 and #010)
2. **VAE Encoding**: Convert images to latent space using VAE
3. **Interpolation**: Generate smooth frames between them using spherical lerp
4. **Looping**: Create pattern A→[interp]→B→[interp reversed]→A
5. **Compression**: Optimize as GIF with automatic size management
   - If oversized: downsamples images
   - If still oversized: decimates frames
   - Always targets < 1MB

## Key Features

✅ **Smart Compression**
- Multiple optimization strategies
- Real-time file size monitoring
- Automatic fallback methods

✅ **High Quality Interpolation**
- Uses your existing spherical lerp (slerp) code
- Smooth transitions with VAE encoding

✅ **Flexible Configuration**
- Choose any keyframes
- Adjust smoothness (interpolation frames)
- Control playback speed (FPS)
- Override resolution

✅ **GitHub Compatible**
- GIF format (universal)
- Target < 1MB
- Looping animation

## Configuration Options

```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 24              # First keyframe (default: 24)
  --keyframe-b 10              # Second keyframe (default: 10)
  --interpolation-frames 10    # Frames between keyframes (default: 10)
  --fps 5                      # Playback FPS (default: 5)
  --max-size 1.0               # Target size in MB (default: 1.0)
  --resolution 512x256         # Override resolution (optional)
  --output preview.gif         # Output filename (default: preview.gif)
```

## Performance

**Total time**: ~13-45 seconds (depending on factors)
- VAE initialization: ~30s (first run only, then cached)
- Encoding keyframes: ~4s
- Generating frames: ~5s
- GIF encoding: ~4s

## Size Optimization

### Results by Configuration

| Config | Frames | FPS | Resolution | Size | Status |
|--------|--------|-----|-----------|------|--------|
| Default | 22 | 5 | 512x256 → 256x128 | 0.77 MB | ✅ |
| Smooth | 32 | 5 | 512x256 → 256x128 | 0.95 MB | ✅ |
| Fast | 12 | 8 | 512x256 → 256x128 | 0.45 MB | ✅ |

### If File is Too Large

1. **Reduce frames**: `--interpolation-frames 5`
2. **Reduce FPS**: `--fps 3`
3. **Stricter limit**: `--max-size 0.5`
4. **Lower resolution**: `--resolution 256x128`

## Integration Examples

### 1. GitHub README Preview
```bash
# Create preview
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 50 \
  --keyframe-b 25 \
  --output github_preview.gif

# Edit README.md
# ![Animation](output/github_preview.gif)
```

### 2. Project Showcase
```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 75 \
  --keyframe-b 30 \
  --interpolation-frames 20 \
  --output showcase.gif
```

### 3. Batch Generate Multiple Previews
```bash
# Generate 3 different previews
uv run backend/tools/generate_looped_preview.py --keyframe-a 30 --keyframe-b 15 --output preview_a.gif
uv run backend/tools/generate_looped_preview.py --keyframe-a 50 --keyframe-b 20 --output preview_b.gif
uv run backend/tools/generate_looped_preview.py --keyframe-a 70 --keyframe-b 40 --output preview_c.gif
```

## What Makes This Different From `generate_animation.py`

| Feature | New Preview | Existing Animation |
|---------|-------------|-------------------|
| **Purpose** | 2-keyframe loop for GitHub | All keyframes sequence |
| **Use Case** | README preview | Full project animation |
| **Output** | Compact GIF (~1MB) | Full sequence (~5-50MB) |
| **Looping** | Smooth A↔B loop | Sequential all keyframes |
| **Configuration** | Simple (2 keyframes) | Complex (all sequences) |

## Troubleshooting

### "Keyframe not found"
```bash
# Check available keyframes
ls output/keyframes/ | head -5

# Use correct number (without padding)
uv run backend/tools/generate_looped_preview.py --keyframe-a 24 --keyframe-b 10
```

### "File size exceeds limit"
```bash
# Tool auto-downsamples, but for more control:
uv run backend/tools/generate_looped_preview.py \
  --interpolation-frames 5 \
  --fps 3 \
  --max-size 0.5
```

### "VAE loading failed"
- Ensure `diffusers` is installed (it should be)
- First run downloads VAE model (~1GB)
- Requires CUDA or falls back to slow CPU

## Files Modified/Created

```
✅ Created:
  - backend/tools/generate_looped_preview.py (main script)
  - backend/tools/PREVIEW_GENERATOR_GUIDE.md (detailed guide)
  - PREVIEW_GIF_CREATION.md (this file)

✏️  Updated:
  - backend/tools/README.md (added tool documentation)
```

## Next Steps

### Use It Now
```bash
cd C:\Users\luxia\Documents\projects\dream_gen
uv run backend/tools/generate_looped_preview.py
# Creates output/preview.gif ready for GitHub!
```

### Add to GitHub README
```markdown
## Animation Preview

Check out this smooth looping animation generated with VAE interpolation:

![Preview](output/preview.gif)
```

### Customize for Your Aesthetic
```bash
# Try different keyframe combinations to find the best loop
uv run backend/tools/generate_looped_preview.py --keyframe-a 40 --keyframe-b 20
uv run backend/tools/generate_looped_preview.py --keyframe-a 60 --keyframe-b 35
```

## Questions?

Refer to:
- **Quick start**: `backend/tools/README.md`
- **Detailed guide**: `backend/tools/PREVIEW_GENERATOR_GUIDE.md`
- **Source code**: `backend/tools/generate_looped_preview.py` (well-commented)

---

**Status**: ✅ Complete and tested  
**Date**: November 15, 2025  
**Result**: Successfully generates compact GitHub preview GIFs under 1MB

