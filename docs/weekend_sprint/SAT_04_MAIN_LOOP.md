# ⏱️ Saturday Session 4: Main Generation Loop

**Goal**: Continuous morphing image generation

**Duration**: 2 hours

---

## Overview

Create `backend/main.py` - the entry point that runs the continuous generation loop.

### Key Components

1. **Configuration loading**
2. **Main async loop**
3. **img2img feedback**
4. **Frame management**
5. **Status updates**

### What It Does

- Loads config.yaml
- Initializes generator
- Runs continuous loop
- Uses previous frame as input for next generation
- Writes to output/current_frame.png

---

## Generation Flow

```
1. Start with seed image
2. Generate new image (img2img with denoise=0.4)
3. Save as current_frame.png
4. Wait refresh_interval seconds
5. Use current_frame.png as input
6. Go to step 2
```

---

## Core Loop Structure

```python
async def main_loop():
    current_image = load_seed()
    
    while True:
        # Generate next frame
        new_image = generator.generate_from_image(
            image_path=current_image,
            prompt=get_prompt(),
            denoise=0.4
        )
        
        # Update current
        copy_to_output(new_image, "current_frame.png")
        current_image = new_image
        
        # Wait
        await asyncio.sleep(refresh_interval)
```

**Full code**: See original WEEKEND_SPRINT.md lines 702-950

---

## Validation

```powershell
# Run the loop
python backend\main.py
```

**Expected behavior**:
- Console shows generation times
- output/ directory fills with frames
- Each frame ~2 seconds apart
- Images slowly morph

---

## Success Criteria

- [ ] Main loop running
- [ ] Continuous generation working
- [ ] Images morphing smoothly
- [ ] Can run for 50+ frames
- [ ] No crashes or errors

---

## 🎉 Saturday Milestone Reached!

**Milestone 2**: Morphing Loop Working

You now have:
- ✅ Continuous generation
- ✅ img2img feedback loop
- ✅ Folder filling with evolving images

---

**Next**: Sunday morning - Cache system

---

**Session 4 of 8** | ~2 hours
**Saturday Complete!** 🌙

