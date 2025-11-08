# 🔧 Flux Model Download (Part 3 of 6)

**Saturday Morning Sprint - Step 3**

This guide covers downloading and installing the Flux.1-schnell model.

---

## STEP 3: Download Flux.1-schnell Model

### 3.1 Model Requirements

**Flux.1-schnell**:
- Size: ~24GB download
- VRAM: 6-8GB usage
- Speed: 1-2 seconds at 256×512

**Storage location**: `ComfyUI/models/checkpoints/`

### 3.2 Download from HuggingFace

**Option A: Direct Download (Easy)**

1. Go to: https://huggingface.co/black-forest-labs/FLUX.1-schnell
2. Click "Files and versions"
3. Download: `flux1-schnell.safetensors` (~24GB)
4. Move to: `C:\AI\ComfyUI\ComfyUI\models\checkpoints\`

**Option B: Using huggingface-cli**

```powershell
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='black-forest-labs/FLUX.1-schnell', filename='flux1-schnell.safetensors', local_dir='C:/AI/ComfyUI/ComfyUI/models/checkpoints')"
```

**This will take 30-60 minutes**. Get coffee. ☕

### 3.3 Download VAE (Required)

1. Go to: https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main/ae.safetensors
2. Download: `ae.safetensors`
3. Move to: `C:\AI\ComfyUI\ComfyUI\models\vae\`
4. Rename to: `flux_vae.safetensors`

### 3.4 Verify Files

Check that you have:
```
C:\AI\ComfyUI\ComfyUI\models\
├── checkpoints\
│   └── flux1-schnell.safetensors    (~24GB)
└── vae\
    └── flux_vae.safetensors         (~335MB)
```

✅ **Checkpoint**: Flux model downloaded and in correct location

---

## Next Steps

→ Continue to **04_FIRST_TEST.md** to generate your first image

---

**Part 3 of 6 Complete** | Estimated time: 30-60 minutes (mostly downloading)

