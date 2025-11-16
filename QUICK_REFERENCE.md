# Preview GIF Generator - Quick Reference Card

## 🚀 One Command (Default)
```bash
uv run backend/tools/generate_looped_preview.py
```
Creates `output/preview.gif` (0.77 MB, 4.4 sec, keyframes 024 & 010)

---

## 📋 Common Commands

### GitHub Preview
```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 40 --keyframe-b 20 \
  --output github_preview.gif
```

### Smooth & Slow
```bash
uv run backend/tools/generate_looped_preview.py \
  --interpolation-frames 15 --fps 4
```

### Fast & Compact
```bash
uv run backend/tools/generate_looped_preview.py \
  --interpolation-frames 5 --fps 3 --max-size 0.5
```

### Different Keyframes
```bash
uv run backend/tools/generate_looped_preview.py \
  --keyframe-a 50 --keyframe-b 30
```

### Half Resolution (Smaller)
```bash
uv run backend/tools/generate_looped_preview.py \
  --resolution 256x128
```

---

## 🎛️ All Options

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--keyframe-a` | 24 | 1-82 | First keyframe number |
| `--keyframe-b` | 10 | 1-82 | Second keyframe number |
| `--interpolation-frames` | 10 | 1-30 | Frames between keyframes |
| `--fps` | 5 | 1-30 | Playback frames per second |
| `--max-size` | 1.0 | 0.1-10 | Target size in MB |
| `--resolution` | auto | WIDTHxHEIGHT | Override resolution |
| `--output` | preview.gif | any | Output filename |
| `--output-dir` | output | any | Output directory |
| `--input-dir` | output | any | Input directory |

---

## 📊 Size Optimization Strategies

**Too Large?** Try in order:

1. **Reduce interpolation frames**
   ```bash
   --interpolation-frames 5  # (default: 10)
   ```

2. **Reduce FPS**
   ```bash
   --fps 3  # (default: 5)
   ```

3. **Stricter size limit**
   ```bash
   --max-size 0.5  # (default: 1.0)
   ```

4. **Lower resolution**
   ```bash
   --resolution 256x128  # (default: 512x256)
   ```

---

## ⏱️ Typical Results

| Command | Frames | Size | Duration |
|---------|--------|------|----------|
| Default | 22 | 0.77 MB | 4.4s |
| + 5 interp | 12 | 0.45 MB | 2.4s |
| + 15 interp | 32 | 1.2 MB* | 6.4s |
| Half res | 22 | 0.20 MB | 4.4s |

*Auto-compressed, may downsize

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| File too large | Use size optimization strategies above |
| Jittery animation | Increase `--interpolation-frames` or `--fps` |
| Slow generation | Normal (~45s first run, ~13s after) |
| Keyframe not found | Check number exists in `output/keyframes/` |
| VAE error | First run downloads model (~1GB) |

---

## 📖 Documentation

- **Quick start**: This card
- **Full guide**: `backend/tools/PREVIEW_GENERATOR_GUIDE.md`
- **Overview**: `PREVIEW_GIF_CREATION.md`
- **Tech details**: `IMPLEMENTATION_SUMMARY.md`
- **Tool readme**: `backend/tools/README.md`

---

## 💾 For GitHub README

Add to your README.md:

```markdown
## Preview

![Animation Preview](output/preview.gif)
```

---

## 🎯 Recommended Configurations

**Professional**: Smooth, slow
```bash
--keyframe-a 50 --keyframe-b 25 \
--interpolation-frames 15 --fps 5
```

**Social Media**: Compact, fast
```bash
--interpolation-frames 5 --fps 8 \
--max-size 0.5 --resolution 256x128
```

**Showcase**: High quality
```bash
--interpolation-frames 20 --fps 6 \
--keyframe-a 75 --keyframe-b 30
```

---

**Version**: 1.0 | **Date**: Nov 15, 2025 | **Status**: ✅ Production Ready

