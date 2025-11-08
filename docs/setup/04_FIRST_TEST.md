# 🔧 First Generation Test (Part 4 of 6)

**Saturday Morning Sprint - Step 4**

This guide covers creating a workflow and generating your first test image.

---

## STEP 4: First Generation Test

### 4.1 Create Minimal Flux Workflow

In ComfyUI web interface (http://localhost:8188):

1. Clear default workflow (click "Clear" button)

**Manually create nodes:**

**Node 1: Load Checkpoint**
- Right-click → "Add Node" → "loaders" → "Load Checkpoint"
- Set `ckpt_name`: `flux1-schnell.safetensors`

**Node 2: Positive Prompt**
- Right-click → "Add Node" → "conditioning" → "CLIP Text Encode (Prompt)"
- Set `text`: "ethereal digital angel, technical wireframe, monochrome with cyan accents"
- Connect to checkpoint's CLIP output

**Node 3: Empty Latent**
- Right-click → "Add Node" → "latent" → "Empty Latent Image"
- Set `width`: 256
- Set `height`: 512

**Node 4: KSampler**
- Right-click → "Add Node" → "sampling" → "KSampler"
- Set `steps`: 4
- Set `cfg`: 1.0
- Set `sampler_name`: euler
- Set `scheduler`: simple
- Connect model from checkpoint
- Connect positive conditioning
- Connect latent

**Node 5: VAE Decode**
- Connect KSampler output
- Connect VAE from checkpoint

**Node 6: Save Image**
- Connect VAE Decode output
- Set `filename_prefix`: "test"

### 4.2 Generate First Image

1. Click "Queue Prompt" (top-right)
2. Watch console for progress
3. **Time it**: Should take 1-3 seconds

**Expected output in console:**
```
got prompt
model_type EPS
...
Prompt executed in 1.84 seconds
```

**Find output:**
- `C:\AI\ComfyUI\ComfyUI\output\`
- File: `test_00001_.png`

### 4.3 Verify Quality

Open the generated image:
- Should be 256×512 pixels
- Should match prompt vaguely
- Should be clear, not blurry

**If generation took > 5 seconds**: 
- Check Task Manager → GPU #2 should show activity
- May need optimization (see TROUBLESHOOTING)

**If image is black or error**:
- Check console for error messages
- Verify VAE loaded correctly
- Try different sampler

✅ **Checkpoint**: Successfully generated first image in < 3 seconds

---

## Next Steps

→ Continue to **05_PYTHON_ENVIRONMENT.md** to set up the Python controller

---

**Part 4 of 6 Complete** | Estimated time: 15 minutes

