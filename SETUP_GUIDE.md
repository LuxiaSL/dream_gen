# 🔧 SETUP GUIDE - Dream Window Installation

**Saturday Morning Sprint (Est. 2-3 hours)**

This guide will walk you through every single step of setting up your development environment. By the end, you'll have ComfyUI running, Flux loaded, and your first test generation working.

---

## ✅ Pre-Flight Checklist

Before starting, verify you have:
- [ ] Windows 10 installed
- [ ] Python 3.13 installed (or 3.10+)
- [ ] 50-70GB free disk space (HDD is fine for now)
- [ ] Administrator access
- [ ] Internet connection
- [ ] Your angel images ready (bg_5.png, num_1.png, num_3.png, etc.)

---

## 📋 Installation Overview

```
Step 1: Verify Python & GPU [15 min]
Step 2: Install ComfyUI [30 min]
Step 3: Download Flux.1-schnell [30 min]
Step 4: First Generation Test [15 min]
Step 5: Setup Python Environment [30 min]
Step 6: Create Project Structure [15 min]
```

**Total Time**: ~2-3 hours (mostly downloads)

---

## STEP 1: Verify Python & GPU

### 1.1 Check Python Installation

Open PowerShell (Right-click Start → Windows PowerShell):

```powershell
python --version
```

**Expected output**: `Python 3.13.x` or `3.10+`

If Python is not installed:
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ **CHECK**: "Add Python to PATH"
4. Choose "Install Now"

### 1.2 Verify GPU is Detected

```powershell
# Install nvidia-smi access
nvidia-smi
```

**Expected output**: Should list both Titan X GPUs

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 12.x  |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  TITAN X (Pascal)   WDDM  | 00000000:01:00.0  On |                  N/A |
|   1  TITAN X (Pascal)   WDDM  | 00000000:02:00.0 Off |                  N/A |
...
```

**Note GPU IDs**: GPU 0 and GPU 1. We'll use GPU 1 for Dream Window.

If `nvidia-smi` doesn't work:
- Update GPU drivers: https://www.nvidia.com/Download/index.aspx
- Reboot after install

### 1.3 Test CUDA Access

```powershell
python -c "import sys; print(sys.version)"
python -c "import torch; print(torch.cuda.is_available())"
```

**If torch is not installed** (likely):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**This will take 5-10 minutes** - PyTorch is large.

Verify again:
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```

**Expected output**:
```
CUDA Available: True
GPU Count: 2
GPU Name: GeForce GTX TITAN X
```

✅ **Checkpoint**: Python working, both GPUs detected, CUDA available

---

## STEP 2: Install ComfyUI

### 2.1 Choose Installation Directory

Pick a location with 50GB+ free space. Example:
```powershell
C:\AI\DreamWindow\
```

Create directories:
```powershell
mkdir C:\AI
mkdir C:\AI\DreamWindow
cd C:\AI\DreamWindow
```

### 2.2 Download ComfyUI

**Option A: Portable Package (RECOMMENDED for beginners)**

1. Go to: https://github.com/comfyanonymous/ComfyUI/releases
2. Download: `ComfyUI_windows_portable_nvidia_cu118_or_cu121.7z`
3. Extract to `C:\AI\ComfyUI`

**Option B: Git Clone (if you have git)**

```powershell
cd C:\AI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

Then install dependencies:
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2.3 Verify ComfyUI Structure

Your directory should look like:
```
C:\AI\ComfyUI\
├── ComfyUI_windows_portable\   (if using portable)
│   ├── python_embeded\
│   ├── ComfyUI\
│   └── run_nvidia_gpu.bat
│
OR
│
├── main.py                      (if git cloned)
├── nodes.py
├── requirements.txt
└── ...
```

### 2.4 First Launch (GPU Selection Test)

**If using portable:**
```powershell
cd C:\AI\ComfyUI\ComfyUI_windows_portable
.\run_nvidia_gpu.bat
```

**If git cloned:**
```powershell
cd C:\AI\ComfyUI
python main.py --cuda-device 1
```

**What to expect:**
- Console window opens
- Lots of text scrolling
- Eventually: `To see the GUI go to: http://127.0.0.1:8188`

Open browser to: http://localhost:8188

You should see the ComfyUI node editor (empty graph).

✅ **Checkpoint**: ComfyUI launches, web interface loads

### 2.5 Force GPU #2 (Critical Step)

**If using portable**, edit `run_nvidia_gpu.bat`:

Right-click → Edit → Add this line at the top:
```batch
set CUDA_VISIBLE_DEVICES=1
```

Full file should look like:
```batch
@echo off
set CUDA_VISIBLE_DEVICES=1
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
pause
```

**If git cloned**, always launch with:
```powershell
set CUDA_VISIBLE_DEVICES=1
python main.py
```

**Verify GPU Selection:**

In the ComfyUI console, look for:
```
Total VRAM 12288 MB, total RAM 32768 MB
pytorch version: 2.x.x
Set vram state to: NORMAL_VRAM
Device: cuda:0 NVIDIA GeForce GTX TITAN X : cudaMallocAsync
```

If it says `cuda:1`, you need to set `CUDA_VISIBLE_DEVICES=1` before launching.

✅ **Checkpoint**: ComfyUI using GPU #2 (secondary GPU)

---

## STEP 3: Download Flux.1-schnell Model

### 3.1 Understand Model Requirements

**Flux.1-schnell** is a distilled fast model:
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

**Option B: Using Git LFS (if you have git-lfs)**

```powershell
cd C:\AI\ComfyUI\ComfyUI\models\checkpoints
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-schnell
```

**Option C: Using huggingface-cli**

```powershell
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='black-forest-labs/FLUX.1-schnell', filename='flux1-schnell.safetensors', local_dir='C:/AI/ComfyUI/ComfyUI/models/checkpoints')"
```

**This will take 30-60 minutes** depending on internet speed. Get coffee. ☕

### 3.3 Download VAE (Required)

Flux needs a VAE for encoding/decoding:

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

## STEP 4: First Generation Test

### 4.1 Load Basic Workflow

In ComfyUI web interface (http://localhost:8188):

1. **Clear default workflow**: Click "Clear" button
2. **Load example**: Go to "Load" → Look for "default" workflows

If no Flux workflow exists, **manually create nodes:**

**Minimal Flux Workflow:**

1. Right-click → "Add Node" → "loaders" → "Load Checkpoint"
   - Set `ckpt_name`: `flux1-schnell.safetensors`

2. Right-click → "Add Node" → "conditioning" → "CLIP Text Encode (Prompt)"
   - Set `text`: "ethereal digital angel, technical wireframe, monochrome with cyan accents"
   - Connect to checkpoint's CLIP output

3. Right-click → "Add Node" → "sampling" → "KSampler"
   - Set `steps`: 4
   - Set `cfg`: 1.0
   - Set `sampler_name`: euler
   - Set `scheduler`: simple
   - Connect model from checkpoint
   - Connect conditioning from CLIP text

4. Right-click → "Add Node" → "latent" → "Empty Latent Image"
   - Set `width`: 256
   - Set `height`: 512
   - Set `batch_size`: 1
   - Connect to KSampler latent input

5. Right-click → "Add Node" → "latent" → "VAE Decode"
   - Connect KSampler output to samples input
   - Connect VAE from checkpoint to vae input

6. Right-click → "Add Node" → "image" → "Save Image"
   - Connect VAE Decode output
   - Set `filename_prefix`: "test"

### 4.2 Generate First Image

1. Click "Queue Prompt" button (top-right)
2. Watch console for progress
3. **Time it**: Should take 1-3 seconds on Titan X

**What to expect:**
```
got prompt
model_type EPS
model sampling discrete
Using pytorch attention in VAE
...
Prompt executed in 1.84 seconds
```

**Where to find output:**
- `C:\AI\ComfyUI\ComfyUI\output\`
- File named like: `test_00001_.png`

### 4.3 Verify Output Quality

Open the generated image:
- Should be 256×512 pixels
- Should show something vaguely matching your prompt
- Quality should be clear, not blurry

**If generation took > 5 seconds**: 
- Check Task Manager → GPU #2 should show activity
- Might need to optimize settings (see TROUBLESHOOTING.md)

**If image is black or error**:
- Check console for error messages
- Verify VAE is loaded correctly
- Try different sampler (dpmpp_2m instead of euler)

✅ **Checkpoint**: Successfully generated first image in < 3 seconds

---

## STEP 5: Setup Python Development Environment

### 5.1 Create Project Structure

```powershell
cd C:\AI\DreamWindow
mkdir backend
mkdir seeds
mkdir seeds\angels
mkdir cache
mkdir cache\images
mkdir cache\metadata
mkdir output
mkdir logs
mkdir comfyui_workflows
mkdir rainmeter
```

### 5.2 Create Virtual Environment

```powershell
cd C:\AI\DreamWindow
python -m venv venv
```

### 5.3 Activate Virtual Environment

```powershell
.\venv\Scripts\activate
```

Your prompt should change to show `(venv)`.

### 5.4 Install Python Dependencies

Create `requirements.txt`:

```powershell
# Create requirements file
@"
# Core dependencies
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pillow>=10.1.0
opencv-python>=4.8.1

# API and networking
requests>=2.31.0
websockets>=12.0

# AI/ML
transformers>=4.35.0
huggingface-hub>=0.19.0

# Configuration and utilities
pyyaml>=6.0.1
python-dotenv>=1.0.0

# System monitoring
psutil>=5.9.6
py-cpuinfo>=9.0.0

# Windows-specific
pywin32>=306

# Logging and debugging
coloredlogs>=15.0.1
tqdm>=4.66.0

# Optional but recommended
watchdog>=3.0.0
"@ | Out-File -FilePath requirements.txt -Encoding utf8
```

Install all dependencies:

```powershell
pip install -r requirements.txt
```

**This will take 5-10 minutes.**

### 5.5 Test Imports

```powershell
python -c "import torch, numpy, PIL, cv2, transformers, requests, yaml; print('All imports successful')"
```

**Expected output**: `All imports successful`

✅ **Checkpoint**: Python environment setup with all dependencies

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
  mode: "img2img"  # Start with simple mode
  
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

### 6.3 Create Directory Structure Script

Create `backend/setup_directories.py`:

```python
"""
Setup script to create all necessary directories
"""
import os
from pathlib import Path

DIRECTORIES = [
    "backend/core",
    "backend/cache", 
    "backend/interpolation",
    "backend/utils",
    "cache/images",
    "cache/metadata",
    "output",
    "logs",
    "seeds/angels",
    "comfyui_workflows",
]

def create_directories():
    base_path = Path(__file__).parent.parent
    
    for directory in DIRECTORIES:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith("backend/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Auto-generated __init__.py\n")
        
        print(f"✓ Created: {directory}")
    
    print("\n✅ All directories created successfully!")

if __name__ == "__main__":
    create_directories()
```

Run it:

```powershell
cd C:\AI\DreamWindow
python backend\setup_directories.py
```

### 6.4 Verify Project Structure

Your directory should now look like:

```
C:\AI\DreamWindow\
├── venv\                          # Virtual environment
├── backend\
│   ├── __init__.py
│   ├── config.yaml
│   ├── setup_directories.py
│   ├── core\
│   │   └── __init__.py
│   ├── cache\
│   │   └── __init__.py
│   ├── interpolation\
│   │   └── __init__.py
│   └── utils\
│       └── __init__.py
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

✅ **Checkpoint**: Project structure created and organized

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

## 🧪 Quick Validation Test

Let's verify everything is working together:

### Test 1: GPU Isolation

```powershell
# Should show GPU #2 (index 1)
python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']='1'; import torch; print(f'Using GPU: {torch.cuda.get_device_name(0)}')"
```

### Test 2: ComfyUI API

With ComfyUI running, test the API:

```powershell
python -c "import requests; resp = requests.get('http://localhost:8188/system_stats'); print('API Status:', resp.status_code)"
```

**Expected**: `API Status: 200`

### Test 3: Image I/O

```python
# test_io.py
from PIL import Image
import os

# Test loading seed
seed_path = "seeds/angels/bg_5.png"
if os.path.exists(seed_path):
    img = Image.open(seed_path)
    print(f"✓ Loaded seed image: {img.size}")
    
    # Test saving to output
    output_path = "output/test_output.png"
    img.save(output_path)
    print(f"✓ Saved to output: {output_path}")
else:
    print("✗ Seed image not found!")
```

Run:
```powershell
python test_io.py
```

### Test 4: Config Loading

```python
# test_config.py
import yaml

with open("backend/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Config loaded successfully:")
print(f"  GPU ID: {config['system']['gpu_id']}")
print(f"  Resolution: {config['generation']['resolution']}")
print(f"  Model: {config['generation']['model']}")
```

Run:
```powershell
python test_config.py
```

All tests passing? **YOU'RE READY TO BUILD!** 🚀

---

## 📍 Where You Are Now

```
[X] Setup Complete
[ ] Backend Implementation
[ ] Hybrid Mode
[ ] Cache System  
[ ] Rainmeter Widget
[ ] Final Polish
```

---

## ⏭️ Next Steps

Your setup is complete! Choose your path:

**SPRINT MODE (Recommended):**
→ Go to **WEEKEND_SPRINT.md** and start Saturday Afternoon tasks

**DEEP DIVE MODE:**
→ Go to **BACKEND_ARCHITECTURE.md** to understand the code structure before building

**VISUAL MODE:**
→ Go to **AESTHETIC_SPEC.md** to see design mockups and plan the look

---

## 🔧 Troubleshooting Quick Reference

### Issue: "CUDA not available"

**Fix:**
```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "ComfyUI won't launch"

**Fix:**
1. Check if another instance is running (Task Manager)
2. Try port 8189 instead: `python main.py --port 8189`
3. Check logs in `ComfyUI/comfyui.log`

### Issue: "Flux model not loading"

**Fix:**
1. Verify file size: `flux1-schnell.safetensors` should be ~24GB
2. Check path: Must be in `models/checkpoints/`
3. Check console for error messages
4. Try renaming: `flux1-schnell.safetensors` → `flux_schnell.safetensors`

### Issue: "Generation too slow (> 5 seconds)"

**Check:**
1. Task Manager → Performance → GPU 1 should be active
2. Try fewer steps: 4 → 2
3. Verify CUDA_VISIBLE_DEVICES is set
4. Check VRAM isn't full: `nvidia-smi`

Full troubleshooting guide: **TROUBLESHOOTING.md**

---

## ✅ Pre-Development Checklist

Before moving to coding, confirm:

- [ ] ComfyUI launches without errors
- [ ] Generated first test image successfully
- [ ] Generation time < 3 seconds
- [ ] GPU #2 is being used (check nvidia-smi)
- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] Project structure created
- [ ] Seed images in place
- [ ] Config file exists

**All checked?** You're ready to build the backend! 🎯

---

## 💾 Backup Recommendation

Before modifying anything:

```powershell
# Create backup of working ComfyUI
cd C:\AI
tar -czf ComfyUI_backup.tar.gz ComfyUI\

# Create backup of project
tar -czf DreamWindow_backup.tar.gz DreamWindow\
```

Now you can experiment without fear!

---

**Ready for the next phase?**

→ **WEEKEND_SPRINT.md** - Start building the backend
→ **BACKEND_ARCHITECTURE.md** - Understand the code first
→ **Take a break** - You've been at this for 2-3 hours, grab food! 🍕

The hard part (setup) is done. Now comes the fun part (making it work)! 🌀✨
