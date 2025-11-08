# 🔧 Environment Setup (Part 1 of 6)

**Saturday Morning Sprint - Step 1**

This guide covers Python, GPU, and initial environment verification.

---

## ✅ Pre-Flight Checklist

Before starting, verify you have:
- [ ] Windows 10 installed
- [ ] Python 3.13 installed (or 3.10+)
- [ ] 50-70GB free disk space
- [ ] Administrator access
- [ ] Internet connection

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

**If torch is not installed**:
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

## Next Steps

→ Continue to **02_COMFYUI_INSTALL.md** to install ComfyUI

---

**Part 1 of 6 Complete** | Estimated time: 15 minutes

