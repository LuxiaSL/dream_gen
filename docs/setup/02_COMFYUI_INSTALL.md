# 🔧 ComfyUI Installation (Part 2 of 6)

**Saturday Morning Sprint - Step 2**

This guide covers downloading, installing, and configuring ComfyUI for GPU #2.

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

**Option A: Portable Package (RECOMMENDED)**

1. Go to: https://github.com/comfyanonymous/ComfyUI/releases
2. Download: `ComfyUI_windows_portable_nvidia_cu118_or_cu121.7z`
3. Extract to `C:\AI\ComfyUI`

**Option B: Git Clone**

```powershell
cd C:\AI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 Verify Structure

Your directory should look like:
```
C:\AI\ComfyUI\
├── ComfyUI_windows_portable\   (if using portable)
│   ├── python_embeded\
│   ├── ComfyUI\
│   └── run_nvidia_gpu.bat
```

### 2.4 First Launch Test

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

**Expected:**
- Console window opens
- Eventually: `To see the GUI go to: http://127.0.0.1:8188`

Open browser to: http://localhost:8188

You should see the ComfyUI node editor.

✅ **Checkpoint**: ComfyUI launches, web interface loads

### 2.5 Force GPU #2 (CRITICAL)

**If using portable**, edit `run_nvidia_gpu.bat`:

Right-click → Edit → Add at top:
```batch
set CUDA_VISIBLE_DEVICES=1
```

Full file:
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

In ComfyUI console, look for:
```
Device: cuda:0 NVIDIA GeForce GTX TITAN X : cudaMallocAsync
```

✅ **Checkpoint**: ComfyUI using GPU #2

---

## Troubleshooting

### Port Already in Use
```powershell
python main.py --port 8189
```

### Another Instance Running
```powershell
taskkill /F /IM python.exe
```

---

## Next Steps

→ Continue to **03_FLUX_MODEL.md** to download the Flux model

---

**Part 2 of 6 Complete** | Estimated time: 30 minutes

