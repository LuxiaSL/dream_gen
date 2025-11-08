# 📋 WINDOWS INTEGRATION CHECKLIST
## Quick reference for testing on Windows machine

**Print this out or keep it open while testing!**

---

## ✅ PRE-FLIGHT (Before Starting)

- [ ] Windows 10/11 machine available
- [ ] Dual Maxwell Titan X GPUs installed
- [ ] Python 3.11+ installed
- [ ] Git installed (for cloning repo)
- [ ] ~30GB free space (for ComfyUI + models)

---

## 📦 INSTALLATION PHASE

### Step 1: Python Environment
```powershell
cd C:\AI\DreamWindow
python -m pip install uv
uv sync
```
- [ ] uv installed successfully
- [ ] All dependencies installed
- [ ] No error messages

### Step 2: ComfyUI Setup
- [ ] Download ComfyUI portable (Windows)
- [ ] Extract to `C:\AI\ComfyUI\`
- [ ] Download Flux.1-schnell.safetensors (~24GB)
- [ ] Place in `C:\AI\ComfyUI\ComfyUI\models\checkpoints\`
- [ ] Download ae.safetensors (Flux VAE, ~335MB)
- [ ] Place in `C:\AI\ComfyUI\ComfyUI\models\vae\`

### Step 3: Start ComfyUI
```powershell
cd C:\AI\ComfyUI\ComfyUI_windows_portable
set CUDA_VISIBLE_DEVICES=1
.\run_nvidia_gpu.bat
```
- [ ] ComfyUI starts without errors
- [ ] Open http://127.0.0.1:8188 in browser
- [ ] UI loads successfully
- [ ] Check VRAM usage on GPU #2 (should be <10GB)

---

## 🧪 UNIT TEST VALIDATION

Run each test on Windows to verify cross-platform:

```powershell
cd C:\AI\DreamWindow
python -m backend.utils.file_ops
```
- [ ] File ops test PASSED

```powershell
python -m backend.utils.prompt_manager
```
- [ ] Prompt manager test PASSED

```powershell
python -m backend.utils.status_writer
```
- [ ] Status writer test PASSED

```powershell
python -m backend.cache.manager
```
- [ ] Cache manager test PASSED

```powershell
python -m backend.core.workflow_builder
```
- [ ] Workflow builder test PASSED

---

## 🔗 API INTEGRATION TESTS

### Test 1: ComfyUI API Connection
```powershell
python -m backend.core.comfyui_api
```
**Expected**:
```
✓ System stats: OS=nt
  Devices: 2 GPU(s)
✓ Queue: 0 running, 0 pending
```
- [ ] API connection test PASSED
- [ ] Shows correct GPU info
- [ ] No connection errors

### Test 2: Workflow Execution
Load `comfyui_workflows/flux_txt2img.json` in ComfyUI UI
- [ ] Workflow loads without errors
- [ ] All nodes are green (valid)
- [ ] Click "Queue Prompt"
- [ ] Image generates successfully
- [ ] Check output/ directory for image
- [ ] Generation time <3 seconds
- [ ] Image is 256×512 pixels

### Test 3: Generator Integration
```powershell
python -m backend.core.generator
```
- [ ] Generator test runs
- [ ] Queues workflow to ComfyUI
- [ ] Waits for completion
- [ ] Retrieves image
- [ ] Copies to output/ directory
- [ ] No errors in logs

---

## 🎨 CLIP INTEGRATION

### Install CLIP
```powershell
pip install transformers
```
- [ ] Transformers installed

### Test CLIP Encoding
```powershell
python -m backend.cache.aesthetic_matcher
```
**First run will download model (~600MB)**
- [ ] CLIP model downloads successfully
- [ ] Model loads on GPU
- [ ] Encodes seed images
- [ ] Shows similarity scores
- [ ] Self-similarity ≈ 1.0
- [ ] Cross-similarity realistic (0.6-0.9)

**Record similarity scores here**:
- background.png × img_1.png: ______
- background.png × img_2.png: ______  
- background.png × img_3.png: ______
- img_1.png × img_2.png: ______

(Should be 0.7-0.8 for similar aesthetic)

---

## 🔄 GENERATION SEQUENCE TEST

### Manual Generation Loop
Create a simple test script or use main.py (once implemented):

1. Start with seed image: `seeds/background.png`
2. Generate frame 1 (img2img from seed, denoise=0.4)
3. Generate frame 2 (img2img from frame 1, denoise=0.4)
4. Generate frame 3 (img2img from frame 2, denoise=0.4)
5. Continue for 10 frames

**Check**:
- [ ] Each frame is generated successfully
- [ ] Images morph smoothly (no jarring changes)
- [ ] Aesthetic is maintained (monochrome + cyan/red)
- [ ] Technical wireframes still visible
- [ ] Generation time consistent (~2s per frame)
- [ ] Cache fills up as frames are generated

---

## 🎮 GAME DETECTION TEST

### Test Process Detection
1. Add your favorite game to `config.yaml`:
```yaml
game_detection:
  known_games:
    - "yourgame.exe"
```
2. Start main controller (once implemented)
3. Start the game
4. **Check**: Generation should pause
5. Close the game
6. **Check**: Generation should resume

- [ ] Game process detected correctly
- [ ] Generation pauses when game starts
- [ ] Generation resumes when game exits
- [ ] No frames generated during gaming
- [ ] Status.json shows "paused" state

---

## 🪟 RAINMETER INTEGRATION

### Install Rainmeter
- [ ] Download Rainmeter from rainmeter.net
- [ ] Install to default location
- [ ] Rainmeter starts on boot

### Create Widget
- [ ] Copy widget files to Rainmeter\Skins\DreamWindow\
- [ ] Edit paths in Variables.inc
- [ ] Load widget in Rainmeter
- [ ] Widget appears on desktop

### Test Display
- [ ] current_frame.png displays correctly
- [ ] Image updates when new frame generated
- [ ] Crossfade is smooth (no flicker)
- [ ] Status info displays correctly
- [ ] Frame design looks beautiful
- [ ] Corner brackets are cyan
- [ ] Widget doesn't interfere with games

---

## ⏱️ PERFORMANCE VALIDATION

### Benchmark Generation
Run 50 frames and measure:

**Generation Times**:
- Min: ______s
- Max: ______s
- Avg: ______s
- Target: <2s ✓

**System Performance**:
- Python CPU: ______%
- Rainmeter CPU: ______%
- GPU #1 usage: ______% (should be 0%)
- GPU #2 usage: ______% (will spike during gen)
- GPU #2 VRAM: ______GB (should be <10GB)

**Frame Quality**:
- [ ] All frames are 256×512 pixels
- [ ] All frames maintain aesthetic
- [ ] No corrupted/black frames
- [ ] Smooth morphing progression

---

## 🏃 STABILITY TEST

### Overnight Run
Start generation before bed:
```powershell
python backend/main.py
```

**Next morning check**:
- [ ] Still running (no crash)
- [ ] output/ directory has hundreds of frames
- [ ] Cache size is at max (75 images)
- [ ] status.json is recent
- [ ] Logs show no errors
- [ ] Memory usage is stable
- [ ] Aesthetic still maintained

**Stats to record**:
- Total runtime: ______h
- Total frames: ______
- Cache hits: ______
- Errors: ______
- Average gen time: ______s

---

## 🎯 FINAL VALIDATION

### The "Holy Shit" Test
- [ ] Invite a friend over
- [ ] Show them the morphing window
- [ ] They say "holy shit" (or equivalent) 😄

### Aesthetic Check
- [ ] Monochrome base maintained
- [ ] Cyan accents present
- [ ] Red accents (sparse)
- [ ] Technical wireframes visible
- [ ] Particle effects present
- [ ] High contrast maintained
- [ ] Matches seed aesthetic

### User Experience
- [ ] Easy to start (one command)
- [ ] Easy to stop (Ctrl+C)
- [ ] Config is self-explanatory
- [ ] Status is always visible
- [ ] Doesn't interfere with desktop use
- [ ] Doesn't interfere with gaming

---

## 🐛 TROUBLESHOOTING QUICK FIXES

### ComfyUI won't start
```powershell
# Check CUDA:
nvidia-smi
# Should show both GPUs

# Try without GPU restriction:
.\run_nvidia_gpu.bat
```

### Generation times are slow (>5s)
- Check GPU #2 is being used (not #1)
- Check VRAM isn't full
- Try reducing to 3 steps in config.yaml
- Check HDD isn't the bottleneck

### Images don't morph smoothly
- Reduce denoise (try 0.3 instead of 0.4)
- Increase prompt weight
- Check negative prompt is working
- Verify aesthetic cache is working

### Cache not filling up
- Check CLIP encoding is working
- Verify cache_manager.add() is being called
- Check cache_index.json is being written
- Look for errors in logs

### Rainmeter not updating
- Check current_frame.png is being written
- Verify atomic writes are completing
- Check file permissions
- Try increasing refresh_interval in config

---

## ✅ COMPLETION CHECKLIST

Mark these when fully working:

### Core Functionality
- [ ] Generates continuously without crash
- [ ] Morphs smoothly between frames
- [ ] Maintains aesthetic coherence
- [ ] Cache injection adds variety
- [ ] Prompt rotation works
- [ ] Game detection works

### Performance
- [ ] Generation <3s per frame
- [ ] Display refresh 3-5s
- [ ] CPU <5%
- [ ] VRAM stable <10GB
- [ ] Memory doesn't leak

### Stability
- [ ] Runs 8+ hours without crash
- [ ] Survives game start/stop
- [ ] Handles errors gracefully
- [ ] Logs are clean

### Polish
- [ ] Frame design is beautiful
- [ ] Crossfades are smooth
- [ ] Status info accurate
- [ ] Easy to configure
- [ ] Documentation complete

---

## 🎉 YOU'RE DONE WHEN...

✅ Widget morphs smoothly on desktop  
✅ Runs overnight without issues  
✅ Aesthetic is maintained  
✅ Gaming is unaffected  
✅ Friends are impressed  

**Congratulations! You built a living dream window! 🌀✨**

---

*Print this checklist and check items off as you go!*
*Keep INTEGRATION_TEST_TRACKER.md open for detailed instructions*

