# ⚡ QUICK REFERENCE GUIDE

**Common Commands and Tasks for Dream Window**

> **Related Documentation**: See [README.md](README.md) for complete documentation navigation

---

## 🚀 Starting the System

```powershell
# 1. Start ComfyUI (GPU #2)
cd C:\AI\ComfyUI\ComfyUI_windows_portable
set CUDA_VISIBLE_DEVICES=1
.\run_nvidia_gpu.bat

# 2. Start Python Controller
cd C:\AI\DreamWindow
.\venv\Scripts\activate
python backend\main.py

# 3. Load Rainmeter Widget
# Right-click Rainmeter tray → Manage → DreamWindow → Load
```

---

## 🛑 Stopping the System

```powershell
# 1. Stop Python Controller
# Press Ctrl+C in console

# 2. Stop ComfyUI
# Close the ComfyUI window or Ctrl+C

# 3. Unload Rainmeter Widget (optional)
# Right-click widget → Unload skin
```

---

## 🔄 Restarting After Changes

```powershell
# After changing config.yaml:
# Just Ctrl+C the Python controller and restart it

# After changing Rainmeter.ini:
# Right-click widget → Refresh Skin

# After changing Python code:
# Ctrl+C controller, restart it
```

---

## ⚙️ Configuration Quick Edits

### Change Generation Speed

```yaml
# backend/config.yaml
generation:
  flux:
    steps: 2  # Faster (was 4)
    # or
    steps: 6  # Slower, higher quality
```

### Change Refresh Rate

```yaml
# backend/config.yaml
display:
  refresh_interval: 3.0  # Faster (was 4.0)
  # or
  refresh_interval: 6.0  # Slower
```

### Change Cache Injection

```yaml
# backend/config.yaml
generation:
  cache:
    injection_probability: 0.25  # More variety (was 0.15)
    # or
    injection_probability: 0.05  # Less variation
```

### Add Custom Prompts

```yaml
# backend/config.yaml
prompts:
  base_themes:
    - "ethereal digital angel..."
    - "YOUR NEW PROMPT HERE"
    - "cyberpunk technical..."
```

---

## 🎨 Rainmeter Customization

### Change Colors

```ini
# @Resources/Variables.inc

; Cyan (default)
ColorCyanPrimary=0,200,255,255

; Make it green
ColorCyanPrimary=0,255,65,255

; Make it red
ColorCyanPrimary=255,0,64,255

; Make it white
ColorCyanPrimary=255,255,255,255
```

### Move Widget

```ini
# @Resources/Variables.inc
WindowX=50    # Pixels from left
WindowY=300   # Pixels from top
```

### Disable Effects

```ini
# @Resources/Variables.inc
ScanlinesEnabled=0  # Turn off scanlines
```

---

## 🔍 Checking Status

```powershell
# View logs
Get-Content logs\dream_controller.log -Tail 50

# Watch logs live
Get-Content logs\dream_controller.log -Wait

# Check GPU usage
nvidia-smi

# Check generated frames
dir output\frame_*.png | measure

# Check cache size
dir cache\images\*.png | measure
```

---

## 🧹 Cleanup Tasks

```powershell
# Clear old frames (keep last 100)
cd output
dir frame_*.png | sort LastWriteTime -Descending | select -Skip 100 | remove-item

# Clear cache completely
remove-item cache\images\*.png
remove-item cache\metadata\cache_index.json

# Clear logs
remove-item logs\*.log
```

---

## 🐛 Quick Fixes

### Generation is slow (> 5s)

```powershell
# Check GPU being used
nvidia-smi
# Should show GPU 1 active

# If not, restart with explicit GPU:
set CUDA_VISIBLE_DEVICES=1
python backend\main.py
```

### Widget not updating

```powershell
# 1. Check Python is running
# 2. Check output\current_frame.png exists
# 3. Refresh Rainmeter
Right-click widget → Refresh Skin
```

### Out of VRAM

```powershell
# Restart ComfyUI
taskkill /F /IM python.exe
# Then restart ComfyUI
```

### Mode collapse (repetitive images)

```yaml
# backend/config.yaml
generation:
  img2img:
    denoise: 0.5  # Increase (was 0.4)
  cache:
    injection_probability: 0.3  # Increase (was 0.15)
```

---

## 📊 Performance Tuning

### For Speed

```yaml
# backend/config.yaml
generation:
  flux:
    steps: 2  # Minimum steps
  cache:
    max_size: 50  # Smaller cache
display:
  refresh_interval: 3.0  # Faster display
```

```ini
# Rainmeter Variables.inc
UpdateRate=100  # Less frequent updates
ScanlinesEnabled=0  # Disable effects
```

### For Quality

```yaml
# backend/config.yaml
generation:
  flux:
    steps: 6  # More steps
  cache:
    max_size: 100  # Larger cache
display:
  refresh_interval: 5.0  # More time per frame
```

```ini
# Rainmeter Variables.inc
UpdateRate=16  # Smoother animations (60 FPS)
ScanlinesEnabled=1  # Enable all effects
```

---

## 🎮 Game Detection

```yaml
# backend/config.yaml
game_detection:
  enabled: true
  known_games:
    - "yourgame.exe"  # Add your games here
    - "anothergame.exe"
```

**Manual pause**:
```powershell
# Create PAUSE file to pause generation
echo. > PAUSE
# Delete to resume
remove-item PAUSE
```

---

## 💾 Backup

```powershell
# Backup project
cd C:\AI
tar -czf DreamWindow_backup_$(Get-Date -Format 'yyyyMMdd').tar.gz DreamWindow\

# Backup just config and cache
cd DreamWindow
tar -czf config_backup.tar.gz backend\config.yaml cache\metadata\cache_index.json
```

---

## 📈 Monitoring

### Real-time Status

```powershell
# Watch status.json
while($true) {
    clear;
    Get-Content output\status.json | ConvertFrom-Json | Format-List;
    sleep 2
}
```

### Performance Stats

```powershell
# Generation speed over last 100 frames
Get-Content logs\dream_controller.log |
    Select-String "Generated in" |
    Select-Object -Last 100
```

### Cache Statistics

```powershell
# Cache size in MB
(Get-ChildItem cache\images\*.png | Measure-Object -Sum Length).Sum / 1MB
```

---

## 🔧 Common File Paths

```
Project Root:         C:\AI\DreamWindow
Config:               backend\config.yaml
Logs:                 logs\dream_controller.log
Output:               output\current_frame.png
Cache:                cache\images\
Seeds:                seeds\angels\

Rainmeter Skin:       ~\Documents\Rainmeter\Skins\DreamWindow\
Widget Config:        @Resources\Variables.inc

ComfyUI:              C:\AI\ComfyUI
ComfyUI Output:       C:\AI\ComfyUI\ComfyUI\output
```

---

## 📞 Getting Help

```powershell
# Collect diagnostic info
python -c "import torch, sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

nvidia-smi > diagnostics.txt
Get-Content logs\dream_controller.log -Tail 100 >> diagnostics.txt
Get-Content backend\config.yaml >> diagnostics.txt
```

**Then share diagnostics.txt when asking for help**

---

## 🎯 One-Liners

```powershell
# Quick start (all in one)
cd C:\AI\DreamWindow; .\venv\Scripts\activate; python backend\main.py

# Quick status check
Get-Content output\status.json | ConvertFrom-Json | Select-Object frame_number, generation_time, status

# Clear everything and restart fresh
remove-item output\*.png; remove-item cache\images\*.png; python backend\main.py

# Export last 100 frames as GIF (requires ImageMagick)
magick convert -delay 10 -loop 0 output\frame_{9900..10000}.png timelapse.gif
```

---

## 🚨 Emergency Stops

```powershell
# Kill all Python processes
taskkill /F /IM python.exe

# Force restart GPU
# (may require reboot if GPU is hung)

# Reset Rainmeter completely
# Right-click tray → Exit
# Then restart Rainmeter
```

---

## ✅ Health Check Checklist

```
[ ] ComfyUI responding (http://localhost:8188)
[ ] GPU #2 has activity (nvidia-smi)
[ ] Python controller running (no errors in console)
[ ] output\current_frame.png updating (check timestamp)
[ ] Rainmeter widget visible
[ ] Generation time < 3 seconds
[ ] VRAM usage < 11GB
[ ] Cache size appropriate (< 100 entries)
[ ] No errors in logs\dream_controller.log
```

---

## 📚 Documentation Navigation

- **Return to**: [README.md](README.md) for complete documentation
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for problem solving
- **Project Vision**: [DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md)
- **Setup Guide**: [setup/01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md)

---

**Keep this reference handy!** These commands cover 90% of daily operations. ⚡✨
