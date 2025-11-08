# 🔧 TROUBLESHOOTING GUIDE

**Common Issues and Solutions for Dream Window**

> **Related Documentation**: See [README.md](README.md) for complete documentation navigation

---

## 🚨 Setup Issues

### Issue: "CUDA not available" or "torch.cuda.is_available() = False"

**Symptoms**: PyTorch can't see GPU

**Solutions**:
```powershell
# 1. Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Verify NVIDIA drivers
nvidia-smi

# 3. Check CUDA environment
python -c "import torch; print(torch.version.cuda)"
```

**If still failing**: Update GPU drivers from nvidia.com

---

### Issue: ComfyUI won't launch

**Symptoms**: Error when running `run_nvidia_gpu.bat` or `python main.py`

**Solutions**:

1. **Port already in use**:
```powershell
# Try different port
python main.py --port 8189
```

2. **Another instance running**:
- Check Task Manager → End "python.exe" processes
- Or: `taskkill /F /IM python.exe`

3. **Missing dependencies**:
```powershell
cd C:\AI\ComfyUI
pip install -r requirements.txt
```

4. **Check logs**:
```powershell
type ComfyUI\comfyui.log
```

---

### Issue: Flux model not loading

**Symptoms**: "Model not found" or "Failed to load checkpoint"

**Solutions**:

1. **Verify file location**:
```powershell
dir "C:\AI\ComfyUI\ComfyUI\models\checkpoints\flux1-schnell.safetensors"
```
Should show ~24GB file.

2. **Check file integrity**:
- Re-download if < 24GB
- Verify not corrupted

3. **Try different model name**:
Rename to `flux_schnell.safetensors` and update config.

4. **Check console output** for specific error message.

---

## ⚡ Performance Issues

### Issue: Generation too slow (> 5 seconds per frame)

**Symptoms**: Each frame takes 6-10+ seconds

**Diagnostic**:
```powershell
# Check which GPU is active
nvidia-smi
# GPU 1 should show activity during generation
```

**Solutions**:

1. **GPU not being used**:
```powershell
# Set environment variable
set CUDA_VISIBLE_DEVICES=1
python backend\main.py
```

2. **Wrong GPU selected**:
Edit `backend/config.yaml`:
```yaml
system:
  gpu_id: 1  # Make sure this is 1, not 0
```

3. **VRAM full**:
```python
# Check VRAM usage
python -c "import torch; print(f'{torch.cuda.memory_allocated()/1e9:.1f} GB')"
```
If > 11GB, restart ComfyUI.

4. **Too many steps**:
Reduce steps in config:
```yaml
flux:
  steps: 2  # Try 2 instead of 4
```

5. **HDD bottleneck**:
- Move `output/` and `cache/` to RAM disk (temporary)
- Or wait for SSD

---

### Issue: Memory leak / Crashes after hours

**Symptoms**: VRAM slowly increases, eventual crash

**Solutions**:

1. **Add explicit cleanup**:
```python
# In generator.py, after each generation
torch.cuda.empty_cache()
gc.collect()
```

2. **Restart loop periodically**:
```python
# In main.py loop
if frame_count % 500 == 0:
    logger.info("Periodic restart for memory cleanup")
    break  # Will auto-restart if in service
```

3. **Monitor VRAM**:
```python
def check_vram():
    used = torch.cuda.memory_allocated() / 1e9
    if used > 11:
        logger.warning(f"High VRAM: {used:.1f}GB")
        torch.cuda.empty_cache()
```

---

## 🖼️ Generation Quality Issues

### Issue: Images don't match aesthetic

**Symptoms**: Wrong style, colors, or theme

**Solutions**:

1. **Strengthen prompts**:
```yaml
prompts:
  base_themes:
    - "(ethereal digital angel:1.3), (technical wireframe:1.2), monochrome with cyan accents"
```
Numbers add weight (1.0-1.5 recommended).

2. **Reduce denoise**:
```yaml
img2img:
  denoise: 0.3  # Lower = closer to seed
```

3. **Add more negative prompts**:
```yaml
negative: "photorealistic, photo, 3d, realistic, color photograph, warm tones, brown, yellow, orange, blurry, low quality, text"
```

4. **Inject seeds more often**:
```yaml
cache:
  injection_probability: 0.25  # 25% instead of 15%
```

---

### Issue: Mode collapse (repetitive images)

**Symptoms**: Same structure/composition every frame

**Solutions**:

1. **Increase denoise**:
```yaml
img2img:
  denoise: 0.5  # Higher = more variation
```

2. **Rotate prompts faster**:
```yaml
prompts:
  rotation_interval: 10  # Every 10 frames instead of 20
```

3. **Inject cache more**:
```yaml
cache:
  injection_probability: 0.3  # 30%
```

4. **Random seeds**:
Make sure seeds are randomizing, not fixed.

---

### Issue: Images are blurry or low quality

**Symptoms**: Soft focus, lack of detail

**Solutions**:

1. **Increase steps**:
```yaml
flux:
  steps: 6  # Up from 4
```

2. **Adjust CFG**:
```yaml
flux:
  cfg_scale: 1.5  # Up from 1.0
```

3. **Better sampler**:
```yaml
flux:
  sampler: "dpmpp_2m"  # Instead of euler
```

4. **Check resolution**:
Make sure generating at 256×512, not being upscaled.

---

## 🔄 File System Issues

### Issue: "Permission denied" when writing files

**Symptoms**: Can't write to output/, crashes on file save

**Solutions**:

1. **Close file viewers**:
- Windows Photos might lock files
- Close any image viewers

2. **Use atomic writes** (already in code):
```python
# Should already be using this pattern
with tempfile.NamedTemporaryFile(...) as tmp:
    shutil.move(tmp.name, output_path)
```

3. **Check directory permissions**:
```powershell
# Run as Administrator
icacls "C:\AI\DreamWindow\output" /grant Users:F
```

---

### Issue: Rainmeter not updating display

**Symptoms**: Widget shows old/no image

**Solutions**:

1. **Refresh Rainmeter skin**:
- Right-click widget → Refresh Skin
- Or: Press Ctrl+Alt+R (refresh all)

2. **Check file path**:
In `DreamWindow.ini`:
```ini
[Variables]
ImagePath=C:\AI\DreamWindow\output\current_frame.png
```
Use absolute path, not relative.

3. **Verify file updates**:
```powershell
# Watch file modification time
while($true) { ls "output\current_frame.png" | select LastWriteTime; sleep 2 }
```

4. **Check Rainmeter log**:
- Right-click widget → Manage → Log
- Look for errors

---

## 🎮 Game Detection Issues

### Issue: Generation doesn't pause during games

**Symptoms**: Performance impact while gaming

**Solutions**:

1. **Add your game to config**:
```yaml
game_detection:
  known_games:
    - "yourgame.exe"
    - "anothergame.exe"
```

2. **Manual pause**:
Create `PAUSE` file in project root:
```powershell
echo. > PAUSE
```
Delete to resume.

3. **GPU monitoring**:
If game uses GPU #1, Python should detect:
```python
# In system_monitor.py, adjust threshold
gpu_threshold: 60  # Lower = more sensitive
```

---

## 🌐 ComfyUI API Issues

### Issue: "Failed to queue prompt" errors

**Symptoms**: API calls failing

**Solutions**:

1. **Check ComfyUI is running**:
```powershell
curl http://localhost:8188/system_stats
```
Should return JSON.

2. **Check queue**:
```powershell
curl http://localhost:8188/queue
```
If full, clear it via ComfyUI web UI.

3. **Increase timeout**:
```python
# In comfyui_api.py
async def wait_for_completion(self, prompt_id: str, timeout: float = 120.0):
    # Increased from 60
```

4. **Check workflow JSON**:
Invalid workflow will fail silently. Validate structure.

---

## 🖥️ Rainmeter Widget Issues

### Issue: Widget not showing up

**Symptoms**: Installed but invisible

**Solutions**:

1. **Check skin is loaded**:
- Right-click Rainmeter tray icon
- Check if DreamWindow is listed

2. **Reload skin**:
- Manage → Load skin → DreamWindow

3. **Check position**:
Might be off-screen. Reset position in `DreamWindow.ini`:
```ini
[Rainmeter]
WindowX=100
WindowY=100
```

4. **Check transparency**:
Make sure `AlphaValue` isn't 0.

---

### Issue: Glitchy animations or flickering

**Symptoms**: Crossfade stutters, flickers

**Solutions**:

1. **Increase update rate**:
```ini
[Rainmeter]
Update=50  # Faster updates (was 100)
```

2. **Simplify animations**:
Disable glitch overlays temporarily.

3. **Check system load**:
High CPU might cause Rainmeter lag.

4. **GPU acceleration**:
```ini
[Rainmeter]
HardwareAcceleration=1
```

---

## 🐛 Python Code Issues

### Issue: Import errors

**Symptoms**: "ModuleNotFoundError"

**Solutions**:

1. **Activate venv**:
```powershell
cd C:\AI\DreamWindow
.\venv\Scripts\activate
```

2. **Reinstall requirements**:
```powershell
pip install -r backend\requirements.txt
```

3. **Check Python path**:
```python
import sys
print('\n'.join(sys.path))
```

---

### Issue: "asyncio" errors

**Symptoms**: Event loop errors

**Solutions**:

1. **Windows asyncio fix**:
```python
# At top of main.py
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

2. **Avoid nested loops**:
Don't call `asyncio.run()` inside another async function.

---

## 📊 Logging & Debugging

### Enable detailed logging

```python
# In main.py
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dream_controller.log'),
        logging.StreamHandler()
    ]
)
```

### Check logs

```powershell
# View last 50 lines
Get-Content logs\dream_controller.log -Tail 50

# Watch live
Get-Content logs\dream_controller.log -Wait
```

---

## 🔄 Nuclear Options

### When all else fails:

**1. Clean restart ComfyUI**:
```powershell
taskkill /F /IM python.exe
# Restart ComfyUI
```

**2. Clear cache**:
```powershell
rm -r cache\images\*
rm -r cache\metadata\*
```

**3. Regenerate output**:
```powershell
rm output\*.png
```

**4. Fresh Python environment**:
```powershell
rm -r venv
python -m venv venv
.\venv\Scripts\activate
pip install -r backend\requirements.txt
```

**5. Reinstall ComfyUI**:
Backup your workflows, delete ComfyUI folder, re-download.

---

## 📞 Getting Help

**When asking for help, provide**:
1. Error message (full traceback)
2. Last 20 lines of `logs/dream_controller.log`
3. ComfyUI console output
4. `nvidia-smi` output
5. Config file contents
6. What you were doing when error occurred

**Useful diagnostic command**:
```powershell
python -c "import torch, sys; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## ✅ Prevention Checklist

**Before starting long runs**:
- [ ] ComfyUI running and responsive
- [ ] GPU #2 being used (check nvidia-smi)
- [ ] Test generation works (< 3 seconds)
- [ ] Logs directory writable
- [ ] Output directory writable
- [ ] Enough disk space (check cache size)
- [ ] Config file syntax valid (test with `yaml.safe_load()`)

---

## 📚 Documentation Navigation

- **Return to**: [README.md](README.md) for complete documentation
- **Command Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for daily operations
- **Project Vision**: [DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md)
- **Setup Guide**: [setup/01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md)

---

**Most issues are fixable!** Check logs first, then try solutions above. 🔧✨
