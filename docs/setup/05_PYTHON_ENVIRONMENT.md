# 🔧 Python Environment Setup (Part 5 of 6)

**Saturday Morning Sprint - Step 5**

This guide covers creating the Python virtual environment and installing dependencies.

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

Create `requirements.txt` in project root:

```txt
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

# Configuration
pyyaml>=6.0.1
python-dotenv>=1.0.0

# System monitoring
psutil>=5.9.6

# Windows-specific
pywin32>=306

# Logging
coloredlogs>=15.0.1
tqdm>=4.66.0
```

Install dependencies:

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

## Next Steps

→ Continue to **06_PROJECT_STRUCTURE.md** to finalize the setup

---

**Part 5 of 6 Complete** | Estimated time: 30 minutes

