# 🔧 Project Structure Finalization (Part 6 of 6)

**Saturday Morning Sprint - Step 6 (Final)**

This guide covers creating the config files and verifying everything is ready.

---

## STEP 6: Create Initial Project Structure

### 6.1 Copy Seed Images

Copy your angel images to the seeds directory:

```powershell
# Copy your images
copy "path\to\bg_5.png" "C:\AI\DreamWindow\seeds\angels\"
copy "path\to\num_1.png" "C:\AI\DreamWindow\seeds\angels\"
copy "path\to\num_3.png" "C:\AI\DreamWindow\seeds\angels\"
```

### 6.2 Create Basic Config File

Create `backend/config.yaml`:

```yaml
system:
  gpu_id: 1
  comfyui_url: "http://127.0.0.1:8188"
  output_dir: "../output"
  cache_dir: "../cache"
  seed_dir: "../seeds/angels"
  log_level: "INFO"
  
generation:
  model: "flux.1-schnell"
  resolution: [256, 512]
  mode: "img2img"
  
  flux:
    steps: 4
    cfg_scale: 1.0
    sampler: "euler"
    scheduler: "simple"
  
  img2img:
    denoise: 0.4
    
display:
  refresh_interval: 4.0
  
prompts:
  base_themes:
    - "ethereal digital angel, dissolving particles, technical wireframe, monochrome with cyan accents"
  
  negative: "photorealistic, photo, 3d render, blurry, low quality"
```

### 6.3 Verify Final Structure

Your directory should look like:

```
C:\AI\DreamWindow\
├── venv\
├── backend\
│   └── config.yaml
├── seeds\
│   └── angels\
│       ├── bg_5.png
│       ├── num_1.png
│       └── num_3.png
├── cache\
│   ├── images\
│   └── metadata\
├── output\
├── logs\
└── comfyui_workflows\
```

### 6.4 Validation Tests

**Test 1: GPU Isolation**
```powershell
python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']='1'; import torch; print(f'Using GPU: {torch.cuda.get_device_name(0)}')"
```

**Test 2: ComfyUI API**
```powershell
python -c "import requests; resp = requests.get('http://localhost:8188/system_stats'); print('API Status:', resp.status_code)"
```
Expected: `API Status: 200`

**Test 3: Config Loading**
```powershell
python -c "import yaml; config = yaml.safe_load(open('backend/config.yaml')); print(f'GPU: {config[\"system\"][\"gpu_id\"]}')"
```

All tests passing? **YOU'RE READY TO BUILD!** 🚀

---

## 🎉 SETUP COMPLETE!

You now have:
- ✅ Python 3.13 environment
- ✅ ComfyUI running on GPU #2
- ✅ Flux.1-schnell model loaded
- ✅ First test generation working (< 3 seconds)
- ✅ Project structure created
- ✅ Dependencies installed
- ✅ Seed images ready

---

## ⏭️ Next Steps

Your setup is complete! Choose your path:

**SPRINT MODE (Recommended):**
→ Go to `weekend_sprint/` directory for implementation guides

**DEEP DIVE MODE:**
→ Check `backend_architecture/` for code structure details

---

**Part 6 of 6 Complete** | Estimated time: 15 minutes

**Total Setup Time: ~2-3 hours**

