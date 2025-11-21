# Dream Window - Complete Installation Guide

**From Zero to Running: A Beginner-Friendly Guide**

This guide assumes little-to-no technical experience. Follow each step carefully, and you'll have Dream Window generating beautiful AI art on your desktop!

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Step 1: Install Python](#step-1-install-python)
4. [Step 2: Install UV Package Manager](#step-2-install-uv-package-manager)
5. [Step 3: Download Dream Window](#step-3-download-dream-window)
6. [Step 4: Install ComfyUI](#step-4-install-comfyui)
7. [Step 5: Download Stable Diffusion 1.5 Model](#step-5-download-stable-diffusion-15-model)
8. [Step 6: Install Dream Window Dependencies](#step-6-install-dream-window-dependencies)
9. [Step 7: Configure Dream Window](#step-7-configure-dream-window)
10. [Step 8: Install Rainmeter (Optional but Recommended)](#step-8-install-rainmeter-optional-but-recommended)
11. [Step 9: First Run](#step-9-first-run)
12. [Performance Settings by GPU Tier](#performance-settings-by-gpu-tier)
13. [Troubleshooting](#troubleshooting)
14. [Next Steps](#next-steps)

---

## Prerequisites

Before we begin, you'll need:

- **Windows 10 or Windows 11** (64-bit)
- **NVIDIA GPU** with at least 6GB VRAM
  - Tested on: Maxwell Titan X (12GB), GTX 10-series, RTX 20/30/40-series
  - Note: Newer architectures (RTX cards) may perform better than older "faster" cards due to improved CUDA support and drivers
- **20GB free disk space** (for software, models, and cache)
- **Internet connection** (for downloading software and models)
- **Administrator access** on your PC

---

## System Requirements

### Minimum Requirements (Baseline: Maxwell Titan X)
- **GPU:** NVIDIA GPU with 6GB+ VRAM (Maxwell architecture or newer)
- **RAM:** 8GB system RAM
- **CPU:** Any modern quad-core processor
- **Storage:** 20GB free space

### Recommended Requirements
- **GPU:** NVIDIA RTX 2060 or better (8GB+ VRAM)
- **RAM:** 16GB system RAM
- **CPU:** Modern 6+ core processor
- **Storage:** SSD with 30GB+ free space

### Performance Notes
- **Architecture Matters:** Even if benchmarks show older cards as "faster," newer RTX cards (20/30/40-series) often perform better in practice due to:
  - Updated CUDA cores and Tensor cores
  - Better driver optimization for AI workloads
  - Improved FP16 performance
- **Example:** An RTX 2060 (6GB) may outperform a Maxwell Titan X (12GB) despite lower VRAM

---

## Step 1: Install Python

Dream Window requires **Python 3.11 or 3.12**. Do NOT use Python 3.13+ as it's not yet fully compatible.

### Download Python

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download **Python 3.12.x** (latest 3.12 version, not 3.13)
3. Run the installer

### Installation Steps

1. **IMPORTANT:** Check the box that says **"Add Python to PATH"** at the bottom of the installer
2. Click **"Customize installation"**
3. On the "Optional Features" page, make sure these are checked:
   - pip
   - py launcher
   - for all users (requires admin)
4. Click **Next**
5. On "Advanced Options", check:
   - Install for all users
   - Add Python to environment variables
   - Precompile standard library
6. Click **Install**
7. Wait for installation to complete

### Verify Installation

1. Press `Windows Key + R`
2. Type `cmd` and press Enter
3. In the black window (Command Prompt), type:
   ```
   python --version
   ```
4. You should see: `Python 3.12.x`
5. If you see an error, restart your computer and try again

---

## Step 2: Install UV Package Manager

UV is a fast, modern Python package manager that makes installation much easier.

### Installation Steps

1. Press `Windows Key + X` and select **"Windows PowerShell (Admin)"** or **"Terminal (Admin)"**
2. Copy and paste this command, then press Enter:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. Wait for installation to complete
4. Close PowerShell and open a new one

### Verify Installation

1. Open PowerShell (doesn't need admin this time)
2. Type:
   ```
   uv --version
   ```
3. You should see a version number like `uv 0.x.x`

---

## Step 3: Download Dream Window

### Option A: Using Git (Recommended)

If you have Git installed:

1. Open PowerShell
2. Navigate to where you want to install Dream Window (e.g., Documents):
   ```powershell
   cd $HOME\Documents\projects
   ```
3. Clone the repository:
   ```powershell
   git clone https://github.com/LuxiaSL/dream_gen.git
   cd dream_gen
   ```

### Option B: Download ZIP

If you don't have Git:

1. Go to the Dream Window GitHub page
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to a permanent location:
   - **Recommended location:** `C:\Users\YourName\Documents\projects\dream_gen`
   - **NOT recommended:** Desktop, Downloads (these are temporary)

### Important Directory Notes

- **Choose a permanent location** - Don't move the folder after setup
- **Avoid special characters** in the path (no emoji, non-English characters)
- **Short paths are better** - Avoid very deep folder structures
- **Example good path:** `C:\Users\John\Documents\projects\dream_gen`
- **Example bad path:** `C:\Users\John\Desktop\New Folder (1)\dream_gen-main`

From now on, we'll call this location your **"Dream Window folder"**.

---

## Step 4: Install ComfyUI

ComfyUI is the AI image generation backend that Dream Window uses.

### Download ComfyUI

1. Go to [https://github.com/comfyanonymous/ComfyUI/releases](https://github.com/comfyanonymous/ComfyUI/releases)
2. Scroll to the latest release
3. Under "Assets", download **"ComfyUI_windows_portable_nvidia.7z"** or **".zip"**
4. You'll need 7-Zip to extract it. If you don't have it:
   - Download from [https://www.7-zip.org/](https://www.7-zip.org/)
   - Install 7-Zip
   - Right-click the ComfyUI file → 7-Zip → Extract Here

### Place ComfyUI in Dream Window

**CRITICAL:** ComfyUI must go in the correct location!

1. After extracting, you'll have a folder called `ComfyUI_windows_portable` or similar
2. **Rename this folder to just `ComfyUI`**
3. Move/copy this `ComfyUI` folder into your Dream Window folder
4. The final path should be:
   ```
   C:\Users\YourName\Documents\projects\dream_gen\diffusion\ComfyUI\
   ```

### Verify ComfyUI Structure

After placement, your Dream Window folder should look like this:

```
dream_gen\
├── backend\
├── diffusion\
│   └── ComfyUI\          ← ComfyUI goes here!
│       ├── ComfyUI\      ← Main ComfyUI code
│       ├── python_embeded\
│       ├── run_nvidia_gpu.bat
│       └── ...
├── rainmeter_skin\
└── ...
```

### Test ComfyUI

1. Navigate to the ComfyUI folder:
   ```powershell
   cd diffusion\ComfyUI
   ```
2. Double-click `run_nvidia_gpu.bat` (or run from command line)
3. Wait for it to start (30-60 seconds)
4. You should see text ending with something like: `To see the GUI go to: http://127.0.0.1:8188`
5. Open your web browser and go to [http://127.0.0.1:8188](http://127.0.0.1:8188)
6. If you see the ComfyUI interface, success! ✓
7. Close the command window to stop ComfyUI (we'll run it properly later)

**Note:** You might see a red error about missing checkpoints - that's normal, we'll add the model next!

---

## Step 5: Download Stable Diffusion 1.5 Model

Dream Window uses Stable Diffusion 1.5 as its default model.

### Download the Model

1. Go to [https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive)
2. Click on **"Files and versions"**
3. Download **"v1-5-pruned-emaonly-fp16.safetensors"** (1.7GB)
   - The FP16 version is smaller and faster with minimal quality loss
   - If you want maximum quality, download "v1-5-pruned-emaonly.safetensors" instead (4GB)

### Place the Model

1. After download completes, move the `.safetensors` file to:
   ```
   dream_gen\diffusion\ComfyUI\ComfyUI\models\checkpoints\
   ```
2. The final path should be:
   ```
   C:\Users\YourName\Documents\projects\dream_gen\diffusion\ComfyUI\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors
   ```

### Verify Model Installation

1. The `checkpoints` folder should now contain the `.safetensors` file
2. File size should be ~1.7GB (fp16) or ~4GB (fp32)

---

## Step 6: Install Dream Window Dependencies

Now we'll install all the Python libraries Dream Window needs.

### Navigate to Dream Window Folder

1. Open PowerShell
2. Navigate to your Dream Window folder:
   ```powershell
   cd C:\Users\YourName\Documents\projects\dream_gen
   ```
   (Replace with your actual path)

### Create Virtual Environment

This creates an isolated Python environment for Dream Window:

```powershell
uv venv
```

Wait for it to complete (10-30 seconds).

### Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your command line.

### Install PyTorch with CUDA Support

PyTorch is the AI framework. We need the CUDA version for GPU acceleration.

**For Modern GPUs (RTX 20/30/40 series, GTX 16 series, newer):**

```powershell
uv sync
```

That's it! UV will automatically install PyTorch with CUDA 12.4 support using the configuration in `pyproject.toml`.

**For Older GPUs (Maxwell/Pascal - GTX 10 series, Titan X, etc.):**

If you have an older GPU and run into issues, you may need CUDA 12.1 or 11.8 instead:

```powershell
# For CUDA 12.1 (Pascal/GTX 10 series)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (Maxwell/older cards)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then run `uv sync` to install the other dependencies.

**Note:** The download is ~2-3GB. Wait patiently (5-10 minutes depending on your internet).

### Install Other Dependencies

If you didn't already run `uv sync` in the previous step:

```powershell
uv sync
```

This installs all required packages (2-5 minutes).

### Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

You should see:
- PyTorch version (e.g., `2.5.0+cu124`, `2.5.0+cu121`, or `2.5.0+cu118` depending on what you installed)
- `CUDA Available: True`
- CUDA Version (e.g., `12.4`, `12.1`, or `11.8`)

**CUDA Version Guide:**
- `cu124` = CUDA 12.4 - **Modern default**, best for RTX 20/30/40, GTX 16 series
- `cu121` = CUDA 12.1 - Good for GTX 10 series, Pascal, some Maxwell
- `cu118` = CUDA 11.8 - Fallback for older Maxwell GPUs

**If CUDA Available shows False:**
- Your GPU might not be supported
- NVIDIA drivers might be outdated
- Update drivers from [https://www.nvidia.com/download/index.aspx](https://www.nvidia.com/download/index.aspx)

---

## Step 7: Configure Dream Window

Now we'll configure Dream Window for your system.

### Open Configuration File

1. Navigate to your Dream Window folder
2. Open `backend\config.yaml` in a text editor
   - Notepad works, but [Notepad++](https://notepad-plus-plus.org/) or [VS Code](https://code.visualstudio.com/) are better
3. We'll edit several settings

### Key Settings to Configure

#### A. GPU Configuration

Find the `system` section and set your GPU:

```yaml
system:
  comfyui_url: "http://127.0.0.1:8188"
  gpu_id: 0  # Change this based on your setup
```

**GPU ID Guide:**
- **Single GPU:** Set to `0`
- **Dual GPU (Dream Window on second GPU):** Set to `1`
- To check your GPU setup, run in PowerShell:
  ```powershell
  nvidia-smi -L
  ```
  This lists all GPUs. Use the index number (0, 1, etc.)

#### B. ComfyUI Startup Script

Find the `daemon.comfyui` section:

```yaml
daemon:
  comfyui:
    startup_script: "diffusion/ComfyUI/run_nvidia_gpu.bat"
```

**Change this based on your GPU setup:**

- **Single GPU:** `"diffusion/ComfyUI/run_nvidia_gpu.bat"`
- **Dual GPU (using GPU #2):** `"diffusion/ComfyUI/run_nvidia_gpu_gpu1_fast.bat"`

**Note about the _gpu1_fast script:**
- The included `run_nvidia_gpu_gpu1_fast.bat` is configured for GPU index 1 (second GPU)
- It includes `--cuda-device 1` which locks ComfyUI to GPU #1
- **For single GPU setups:** Use `run_nvidia_gpu.bat` instead (no GPU locking)
- **For dual GPU setups:** The _gpu1_fast script is perfect as-is

#### C. Performance Settings (See "Performance Settings by GPU Tier" section below)

Leave default for now. We'll optimize after first run.

#### D. Model Name

Verify the model name matches what you downloaded:

```yaml
generation:
  model: "sd15"  # This is correct for SD 1.5
```

### Save Configuration

1. Save the `config.yaml` file
2. Close the text editor

---

## Step 8: Install Rainmeter (Optional but Recommended)

Rainmeter displays the AI art on your desktop as a widget.

### Download Rainmeter

1. Go to [https://www.rainmeter.net/](https://www.rainmeter.net/)
2. Click **"Download"**
3. Run the installer
4. Follow the installation wizard (default settings are fine)

### Install Dream Window Skin

1. Open PowerShell in your Dream Window folder
2. Run:
   ```powershell
   .\rainmeter_skin\install.ps1
   ```
3. This automatically installs the skin to Rainmeter

### Configure Rainmeter Widget

1. The installer should have set the project path automatically
2. If the widget doesn't show your image:
   - Right-click the Rainmeter tray icon
   - Select **"Manage"**
   - Find **"DreamWindow"** in the left panel
   - Open `@Resources\Variables.inc`
   - Set `ProjectPath` to your Dream Window folder:
     ```ini
     ProjectPath=C:\Users\YourName\Documents\projects\dream_gen\
     ```
   - Save and refresh the skin

### Load the Skin

1. Right-click Rainmeter tray icon → **Manage**
2. In the left panel, expand **"DreamWindow"**
3. Click **"DreamWindow.ini"**
4. Click **"Load"** in the top-right

The widget should appear on your desktop (might be blank until Dream Window starts generating).

---

## Step 9: First Run

Everything is installed! Time to run Dream Window.

### Method 1: Using Daemon (Recommended)

The daemon automatically manages both ComfyUI and Dream Window:

1. Open PowerShell in your Dream Window folder
2. Activate virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Run the daemon:
   ```powershell
   uv run daemon.py
   ```

You should see:
- ComfyUI starting up (takes 30-60 seconds)
- Health checks passing
- Dream Controller starting
- Frame generation beginning

### Method 2: Manual (Two Terminals)

If daemon doesn't work, run components separately:

**Terminal 1 - ComfyUI:**
```powershell
cd diffusion\ComfyUI
.\run_nvidia_gpu.bat
```
Wait for "To see the GUI go to: http://127.0.0.1:8188"

**Terminal 2 - Dream Window:**
```powershell
cd C:\Users\YourName\Documents\projects\dream_gen
.venv\Scripts\activate
uv run backend\main.py
```

### What to Expect

1. **First 30-60 seconds:** ComfyUI loads models into GPU memory
2. **Next 2-5 seconds:** First keyframe generates
3. **Next 2-3 seconds:** Interpolation frames generate
4. **Then:** Continuous smooth playback begins!

**Watch the logs for:**
- `[INFO] Keyframe generated` - A new AI image was created
- `[INFO] Interpolation complete` - Smooth frames generated
- `[INFO] Buffer status: X.Xs` - How much ahead we are

**Check the output:**
- `output\current_frame.png` - Currently displayed frame (updates 3-4x per second)
- `output\keyframes\` - Full AI-generated keyframes
- `output\interpolations\` - Smooth in-between frames
- `output\status.json` - Live stats (Rainmeter reads this)

### First Generation Timeline

- **0-60s:** Initialization, model loading
- **60-65s:** First keyframe generation (~2.1s on Titan X)
- **65-68s:** First interpolation batch (~2.5s for 10 frames)
- **68s+:** Playback begins! Smooth 3.5 FPS animation

### Monitoring Performance

Watch PowerShell for these key messages:

```
[INFO] Keyframe 1 generated in 2.1s
[INFO] Interpolated 10 frames in 2.5s (avg: 0.25s/frame)
[INFO] Buffer: 30.5s (target: 30s) ✓
[INFO] FPS: 3.5 (target: 3.5) ✓
```

---

## Performance Settings by GPU Tier

Optimize Dream Window for your hardware. Edit `backend\config.yaml`:

### Tier 1: High-End (RTX 3080+, RTX 4070+, 12GB+ VRAM)

**Best quality, smooth playback:**

```yaml
generation:
  hybrid:
    interpolation_frames: 15                    # More frames = smoother
    target_interpolation_fps: 4                 # 4 FPS playback
    keyframe_denoise: 0.25                      # Moderate variation
    interpolation_resolution_divisor: 1         # Full resolution
    interpolation_upscale_method: "bilinear"
    interpolation_downsample_method: "bicubic"

  sd:
    steps: 15                                    # Higher quality
    cfg_scale: 7.0
```

**Expected performance:**
- Keyframe: ~1-1.5s
- Interpolation: ~0.15s per frame
- Smooth 4 FPS playback

### Tier 2: Mid-Range (RTX 2060-3070, GTX 1080, 8GB VRAM)

**Balanced quality and performance:**

```yaml
generation:
  hybrid:
    interpolation_frames: 10                    # Standard
    target_interpolation_fps: 3.5               # 3.5 FPS playback
    keyframe_denoise: 0.2                       # Gentle variation
    interpolation_resolution_divisor: 1         # Full resolution
    interpolation_upscale_method: "bilinear"
    interpolation_downsample_method: "bicubic"

  sd:
    steps: 10                                    # Faster generation
    cfg_scale: 6.0
```

**Expected performance:**
- Keyframe: ~1.5-2s
- Interpolation: ~0.2s per frame
- Smooth 3.5 FPS playback

### Tier 3: Baseline (Maxwell Titan X, GTX 1060/1070, 6GB VRAM)

**Current default config - optimized for reliability:**

```yaml
generation:
  hybrid:
    interpolation_frames: 10                    # Standard
    target_interpolation_fps: 3.5               # 3.5 FPS playback
    keyframe_denoise: 0.2                       # Gentle evolution
    interpolation_resolution_divisor: 1         # Full resolution
    interpolation_upscale_method: "bilinear"
    interpolation_downsample_method: "bicubic"

  sd:
    steps: 10                                    # Quick generation
    cfg_scale: 6.0
```

**Expected performance:**
- Keyframe: ~2-2.5s
- Interpolation: ~0.25s per frame
- Smooth 3.5 FPS playback

**Note:** The current config is already optimized for this tier!

### Tier 4: Entry-Level (GTX 1050/1060, 4-6GB VRAM)

**Performance mode - lower resolution for speed:**

```yaml
generation:
  resolution: [448, 224]                        # Slightly smaller (was 512x256)
  
  hybrid:
    interpolation_frames: 8                     # Fewer frames
    target_interpolation_fps: 3                 # 3 FPS playback
    keyframe_denoise: 0.2                       
    interpolation_resolution_divisor: 1.5       # Reduce to 3/4 resolution
    interpolation_upscale_method: "bilinear"
    interpolation_downsample_method: "bilinear" # Faster

  sd:
    steps: 8                                     # Minimum acceptable
    cfg_scale: 5.5
```

**Expected performance:**
- Keyframe: ~2.5-3s
- Interpolation: ~0.15s per frame (lower res)
- Smooth 3 FPS playback

### Advanced: Maximum Speed Mode (Any GPU)

**For maximum FPS, sacrifice some quality:**

```yaml
generation:
  resolution: [384, 192]                        # Smaller base
  
  hybrid:
    interpolation_frames: 5                     # Minimal
    target_interpolation_fps: 6                 # Higher FPS!
    keyframe_denoise: 0.25                      
    interpolation_resolution_divisor: 2         # Half resolution
    interpolation_upscale_method: "bilinear"
    interpolation_downsample_method: "bilinear"

  sd:
    steps: 8
    cfg_scale: 5.5
```

**Expected performance:**
- 6-10 FPS playback
- Lower visual quality but very smooth
- Good for testing/development

### Architecture Notes

**Why newer "slower" GPUs may perform better:**

1. **Tensor Cores (RTX series):** Hardware acceleration for AI operations
2. **Updated CUDA:** Better optimizations for PyTorch/Diffusers
3. **FP16 Performance:** RTX cards handle half-precision much better
4. **Memory Bandwidth:** Newer cards have faster VRAM access

**Example Comparison:**
- Maxwell Titan X (2015): 12GB VRAM, no Tensor cores, CUDA 5.2
  - Keyframe: ~2.1s
  - Interpolation: ~0.25s/frame
- RTX 2060 (2019): 6GB VRAM, Tensor cores, CUDA 7.5
  - Keyframe: ~1.5s (30% faster despite half the VRAM!)
  - Interpolation: ~0.15s/frame (40% faster)

**Bottom line:** Don't be discouraged by lower VRAM on newer cards - architecture improvements often outweigh raw specs!

---

## Troubleshooting

### ComfyUI Issues

**Problem: "ComfyUI won't start" or "Red error in UI"**

Solutions:
1. Check that the model file is in `diffusion\ComfyUI\ComfyUI\models\checkpoints\`
2. Verify GPU drivers are up to date: [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx)
3. Try running ComfyUI standalone:
   ```powershell
   cd diffusion\ComfyUI
   .\run_nvidia_gpu.bat
   ```
4. Check GPU compatibility:
   ```powershell
   nvidia-smi
   ```

**Problem: "Address already in use" or "Port 8188 busy"**

Solution:
1. Another ComfyUI instance is running
2. Press `Ctrl+Alt+Delete` → Task Manager
3. Find and end `python.exe` processes related to ComfyUI
4. Try again

**Problem: "Out of memory" errors**

Solutions:
1. Lower the resolution in `config.yaml`:
   ```yaml
   resolution: [384, 192]  # Instead of [512, 256]
   ```
2. Use lower resolution interpolation:
   ```yaml
   interpolation_resolution_divisor: 2  # Half res
   ```
3. Close other GPU-using programs
4. Check VRAM usage:
   ```powershell
   nvidia-smi
   ```

### Dream Window Issues

**Problem: "No module named 'torch'"**

Solution:
1. Make sure virtual environment is activated: `(.venv)` should show in PowerShell
2. If not, run:
   ```powershell
   .venv\Scripts\activate
   ```
3. Reinstall PyTorch (modern GPU):
   ```powershell
   uv sync
   ```
4. For older GPUs, see "Install PyTorch with CUDA Support" section above

**Problem: "CUDA not available" or "CUDA Available: False"**

Solutions:
1. Update NVIDIA drivers
2. Verify GPU is CUDA-capable:
   ```powershell
   nvidia-smi
   ```
3. Try different CUDA version:
   ```powershell
   # Modern GPUs - CUDA 12.4 (default via uv sync)
   uv sync
   
   # Older GPUs - CUDA 12.1
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   
   # Very old GPUs - CUDA 11.8
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
   ```
4. Check which CUDA version your drivers support:
   ```powershell
   nvidia-smi
   ```
   Look for "CUDA Version" in the top-right. Use a PyTorch CUDA version <= that number.

**Problem: "Cannot connect to ComfyUI"**

Solutions:
1. Verify ComfyUI is running (visit [http://127.0.0.1:8188](http://127.0.0.1:8188))
2. Check `config.yaml` has correct URL:
   ```yaml
   comfyui_url: "http://127.0.0.1:8188"
   ```
3. Check firewall isn't blocking local connections
4. Try restarting ComfyUI

**Problem: "Frames generating too slowly" or "Buffer depleting"**

Solutions:
1. Lower settings (see "Performance Settings by GPU Tier" above)
2. Reduce interpolation frames:
   ```yaml
   interpolation_frames: 5  # Instead of 10
   ```
3. Use lower resolution divisor:
   ```yaml
   interpolation_resolution_divisor: 2  # Half res interpolation
   ```
4. Check GPU isn't being used by other programs
5. Monitor GPU usage:
   ```powershell
   nvidia-smi -l 1  # Updates every second
   ```

### Rainmeter Issues

**Problem: "Widget shows nothing" or "Blank widget"**

Solutions:
1. Verify `output\current_frame.png` exists
2. Check ProjectPath in `rainmeter_skin\@Resources\Variables.inc`
3. Make sure Dream Window is running (generating frames)
4. Refresh the skin: Right-click widget → Refresh Skin

**Problem: "Widget shows old/frozen image"**

Solutions:
1. The backend might have stopped - check if `daemon.py` is still running
2. Refresh skin: Right-click → Refresh Skin
3. Check logs in `logs\dream_controller.log` for errors

**Problem: "Status not updating"**

Solution:
- Backend must be running to generate `output\status.json`
- If backend crashed, restart it
- Check `logs\` folder for error messages

### Performance Issues

**Problem: "Slideshow instead of smooth playback"**

Solutions:
1. Lower `target_interpolation_fps` to match your GPU capability:
   ```yaml
   target_interpolation_fps: 2.5  # Slower but won't stutter
   ```
2. Increase buffer to give more breathing room:
   ```yaml
   buffer_target_seconds: 60.0  # Double the buffer
   ```
3. Use performance mode settings (see Tier 4 above)

**Problem: "System laggy while Dream Window runs"**

Solutions:
1. If dual GPU: Verify Dream Window is using the correct GPU:
   ```yaml
   gpu_id: 1  # Use second GPU
   ```
2. Lower priority in Task Manager:
   - Find `python.exe` (Dream Window)
   - Right-click → Set Priority → Below Normal
3. Enable game detection to auto-pause:
   ```yaml
   game_detection:
     enabled: true
   ```

### General Debug Steps

1. **Check Logs:**
   - `logs\dream_controller.log` - Main application log
   - `logs\daemon.log` - Daemon/startup log
   - Look for ERROR or WARNING messages

2. **Check GPU:**
   ```powershell
   nvidia-smi
   ```
   - Verify GPU is detected
   - Check memory usage (shouldn't be at 100%)
   - Check GPU utilization (should spike during generation)

3. **Check Disk Space:**
   - Cache can grow large over time
   - Clean old cache: Delete contents of `cache\images\`
   - Clean old outputs: Delete old files in `output\keyframes\` and `output\interpolations\`

4. **Fresh Start:**
   ```powershell
   # Stop everything
   # Press Ctrl+C in all PowerShell windows
   
   # Clear cache
   Remove-Item cache\images\* -Recurse -Force
   Remove-Item cache\metadata\* -Recurse -Force
   
   # Restart
   uv run daemon.py
   ```

5. **Still stuck?**
   - Check GitHub Issues: [https://github.com/LuxiaSL/dream_gen/issues](https://github.com/LuxiaSL/dream_gen/issues)
   - Join Discord: Contact @luxia
   - Provide logs when asking for help!

---

## Next Steps

### Customize Your Aesthetic

Edit `backend\config.yaml` to change the visual style:

```yaml
prompts:
  theme_pairs:
    - positive: "your custom prompt here"
      negative: "things to avoid"
```

**Tips:**
- The default aesthetic is "ethereal technical angels" - monochrome with cyan/red accents
- Experiment with different themes: cyberpunk, nature, abstract, architecture
- Keep prompts consistent for cohesive evolution
- Use negative prompts to avoid unwanted elements

### Add Custom Seed Images

1. Place your own images in the `seeds\` folder
2. Dream Window will occasionally inject these for variety
3. Good seed images:
   - Match your desired resolution (512x256 or similar aspect ratio)
   - Align with your prompt aesthetic
   - High quality, clear subjects

### Fine-Tune Cache Behavior

The cache prevents visual "mode collapse" (getting stuck on similar images):

```yaml
cache:
  injection_probability: 0.15      # How often to inject cached frames (15%)
  diversity_threshold: 1.92        # How different cached frames must be
  seed_injection_probability: 0.20 # How often to inject seed images
```

**Experiment with:**
- Higher injection = more variety, less cohesion
- Lower injection = more cohesion, risk of mode collapse
- Adjust thresholds to control diversity

### Monitor Performance

Watch the logs to understand your system's performance:

```bash
[INFO] Keyframe generated in 2.1s
[INFO] Interpolated 10 frames in 2.5s (avg: 0.25s/frame)
[INFO] Buffer: 30.5s (target: 30s) ✓
```

**Good signs:**
- Buffer stays above target
- No "Buffer depleted!" warnings
- Smooth visual playback

**Bad signs:**
- "Buffer depleted!" messages
- Stuttering/freezing playback
- "Generation timeout" errors

If you see bad signs, lower your performance settings.

### Advanced: Dual GPU Setup

If you have two GPUs and want Dream Window on the secondary GPU:

1. **In `config.yaml`:**
   ```yaml
   system:
     gpu_id: 1  # Use second GPU (0-indexed)
   
   daemon:
     comfyui:
       startup_script: "diffusion/ComfyUI/run_nvidia_gpu_gpu1_fast.bat"
   ```

2. **Verify GPU assignment:**
   ```powershell
   nvidia-smi
   ```
   - Watch GPU1 memory increase when Dream Window starts
   - GPU0 should stay free for gaming/other work

3. **Benefits:**
   - Zero impact on gaming GPU
   - Run Dream Window 24/7 without performance hit
   - Games won't fight for VRAM

### Enable Game Detection

Auto-pause when games launch:

```yaml
game_detection:
  enabled: true
  known_games:
    - "eldenring.exe"
    - "cyberpunk2077.exe"
    - "your_game.exe"  # Add your games here
```

Dream Window will:
- Monitor process list every 5 seconds
- Pause generation when game detected
- Free VRAM for the game
- Resume when game closes

---

## Getting Help

### Resources

- **Documentation:** Check the `docs\` folder in Dream Window
- **GitHub Issues:** [https://github.com/LuxiaSL/dream_gen/issues](https://github.com/LuxiaSL/dream_gen/issues)
- **Discord:** Contact @luxia

### When Asking for Help

Please provide:

1. **Your GPU:** Model and VRAM (check with `nvidia-smi`)
2. **Your settings:** Paste relevant sections from `config.yaml`
3. **Error messages:** Copy from PowerShell or log files
4. **Logs:** Attach `logs\dream_controller.log` and `logs\daemon.log`
5. **What you tried:** List troubleshooting steps you've done

**Good question format:**
```
GPU: RTX 2060 (6GB)
Issue: "Out of memory" error during interpolation
Settings: resolution: [512, 256], interpolation_resolution_divisor: 1
Tried: Lowering resolution to [384, 192] - same error
Logs: [paste error from log]
```

---

## Conclusion

You now have Dream Window fully installed! 🎉

**Quick command reference for daily use:**

```powershell
# Navigate to Dream Window
cd C:\Users\YourName\Documents\projects\dream_gen

# Activate virtual environment
.venv\Scripts\activate

# Run Dream Window (daemon mode)
uv run daemon.py

# Or run manually (two terminals)
# Terminal 1:
cd diffusion\ComfyUI
.\run_nvidia_gpu.bat

# Terminal 2:
uv run backend\main.py
```

**To stop:**
- Press `Ctrl+C` in PowerShell
- Or close the terminal windows

**Daily workflow:**
1. Start Dream Window (daemon or manual)
2. Load Rainmeter skin if not auto-loaded
3. Watch the magic! ✨
4. Customize prompts/settings as desired
5. Stop when done (Ctrl+C)

Enjoy your living AI dream window! 🌟

---

**Version:** 1.0.0  
**Last Updated:** November 2024  
**For:** Dream Window by Luxia

