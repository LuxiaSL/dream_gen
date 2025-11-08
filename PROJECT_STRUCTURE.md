# 📂 COMPLETE PROJECT STRUCTURE

**Visual guide to the fully implemented Dream Window**

---

## 🗂️ Final Directory Tree

```
C:\AI\
│
├── ComfyUI/                           # Stable Diffusion backend
│   ├── ComfyUI_windows_portable/
│   │   ├── python_embeded/
│   │   ├── ComfyUI/
│   │   │   ├── models/
│   │   │   │   ├── checkpoints/
│   │   │   │   │   └── flux1-schnell.safetensors   (24GB)
│   │   │   │   └── vae/
│   │   │   │       └── flux_vae.safetensors        (335MB)
│   │   │   ├── input/                              (temp images)
│   │   │   └── output/                             (generated images)
│   │   └── run_nvidia_gpu.bat                      (startup script)
│   └── ...
│
└── DreamWindow/                       # Main project
    │
    ├── README.md                      # Quick project overview
    │
    ├── backend/                       # Python controller
    │   ├── __init__.py
    │   ├── main.py                    # Entry point
    │   ├── config.yaml                # User configuration
    │   ├── requirements.txt           # Python dependencies
    │   │
    │   ├── core/                      # Generation logic
    │   │   ├── __init__.py
    │   │   ├── controller.py          # Main orchestration
    │   │   ├── comfyui_api.py         # API client
    │   │   ├── workflow_builder.py    # Workflow generation
    │   │   └── generator.py           # Generation interface
    │   │
    │   ├── cache/                     # Image caching
    │   │   ├── __init__.py
    │   │   ├── manager.py             # Cache CRUD
    │   │   ├── aesthetic_matcher.py   # CLIP similarity
    │   │   └── database.py            # (future)
    │   │
    │   ├── interpolation/             # Latent operations
    │   │   ├── __init__.py
    │   │   ├── spherical_lerp.py      # Interpolation
    │   │   ├── latent_encoder.py      # VAE encode/decode
    │   │   └── hybrid_generator.py    # Combined mode
    │   │
    │   └── utils/                     # Utilities
    │       ├── __init__.py
    │       ├── file_ops.py            # Atomic writes
    │       ├── logging_config.py      # Logging setup
    │       ├── system_monitor.py      # Game detection
    │       ├── prompt_manager.py      # Prompt rotation
    │       ├── status_writer.py       # Status JSON
    │       └── frame_buffer.py        # Pre-generation
    │
    ├── comfyui_workflows/             # Workflow JSONs
    │   ├── flux_txt2img.json
    │   ├── flux_img2img.json
    │   └── flux_hybrid.json
    │
    ├── seeds/                         # Base images
    │   ├── angels/
    │   │   ├── bg_5.png
    │   │   ├── num_1.png
    │   │   ├── num_3.png
    │   │   └── ...
    │   └── metadata.json
    │
    ├── cache/                         # Generated cache
    │   ├── images/                    # (~75 images, 3MB each)
    │   │   ├── cache_00001.png
    │   │   ├── cache_00002.png
    │   │   └── ...
    │   └── metadata/
    │       └── cache_index.json       # Cache metadata
    │
    ├── output/                        # Live output
    │   ├── current_frame.png          # Active display
    │   ├── previous_frame.png         # For crossfade
    │   ├── next_frame.png             # Pre-generated
    │   ├── status.json                # Status info
    │   └── frame_*.png                # Historical frames
    │
    ├── logs/                          # Application logs
    │   ├── dream_controller.log
    │   └── errors.log
    │
    ├── docs/                          # Documentation (THIS!)
    │   ├── README.md
    │   ├── DREAM_WINDOW_MASTER.md
    │   ├── SETUP_GUIDE.md
    │   ├── WEEKEND_SPRINT.md
    │   ├── BACKEND_ARCHITECTURE.md
    │   ├── AESTHETIC_SPEC.md
    │   ├── RAINMETER_WIDGET.md
    │   ├── TROUBLESHOOTING.md
    │   └── QUICK_REFERENCE.md
    │
    ├── rainmeter/                     # Widget (copy to Rainmeter skins)
    │   └── DreamWindow/
    │       ├── DreamWindow.ini        # Main widget
    │       ├── Settings.ini           # Config panel
    │       └── @Resources/
    │           ├── Variables.inc      # User settings
    │           ├── Images/
    │           │   ├── border_frame.png
    │           │   ├── scanlines.png
    │           │   ├── glow_overlay.png
    │           │   └── glitch_overlay.png
    │           ├── Fonts/
    │           │   └── (optional)
    │           └── Scripts/
    │               └── Crossfade.lua  # (optional)
    │
    ├── venv/                          # Python virtual environment
    │   ├── Scripts/
    │   ├── Lib/
    │   └── ...
    │
    └── tests/                         # Unit tests (post-MVP)
        ├── test_generator.py
        ├── test_cache.py
        └── test_interpolation.py
```

---

## 📊 Directory Size Estimates

| Directory | Size | Content |
|-----------|------|---------|
| `ComfyUI/` | ~30GB | Flux model + dependencies |
| `backend/` | ~50MB | Python code + venv |
| `seeds/` | ~20MB | Your angel images |
| `cache/images/` | ~225MB | 75 cached frames |
| `output/` | ~500MB | Historical frames (grows) |
| `logs/` | ~10MB | Log files (grows) |
| `docs/` | ~1MB | This documentation |
| `rainmeter/` | ~5MB | Widget + assets |
| **Total (initial)** | **~30.5GB** | After setup |
| **Total (running)** | **~31-32GB** | With cache full |

---

## 🔑 Key Files Explained

### Configuration
- **`backend/config.yaml`** - Main user configuration (prompts, settings, etc.)
- **`rainmeter/@Resources/Variables.inc`** - Widget visual customization

### Runtime
- **`output/current_frame.png`** - What Rainmeter displays right now
- **`output/status.json`** - Real-time status info
- **`cache/metadata/cache_index.json`** - Cache metadata

### Core Logic
- **`backend/main.py`** - Entry point, main loop
- **`backend/core/generator.py`** - High-level generation interface
- **`backend/cache/manager.py`** - Cache operations

### Display
- **`rainmeter/DreamWindow/DreamWindow.ini`** - Main widget code

---

## 🎯 File Count Summary

```
Python files:         ~20 files
Config files:         ~5 files
Documentation:        ~9 files
Rainmeter files:      ~5 files
Image assets:         ~10+ images (seeds)
Generated cache:      ~75 images (runtime)
Output frames:        ~500+ images (grows over time)

Total codebase:       ~3000 lines of Python
                      ~1000 lines of Rainmeter INI
                      ~8000 lines of documentation
```

---

## 📝 File Creation Order (Weekend Sprint)

**Saturday Morning** (Setup):
```
✓ ComfyUI/ (download)
✓ backend/requirements.txt
✓ backend/config.yaml
✓ backend/core/ (empty structure)
```

**Saturday Afternoon** (Backend):
```
✓ backend/core/comfyui_api.py
✓ backend/core/workflow_builder.py
✓ backend/core/generator.py
✓ backend/utils/prompt_manager.py
✓ backend/utils/status_writer.py
```

**Saturday Evening** (Main Loop):
```
✓ backend/main.py
✓ output/ (first frames generated)
```

**Sunday Morning** (Cache):
```
✓ backend/cache/manager.py
✓ backend/cache/aesthetic_matcher.py
✓ cache/images/ (starts filling)
```

**Sunday Afternoon** (Integration):
```
✓ backend/interpolation/spherical_lerp.py (if using)
✓ cache/ (fully operational)
```

**Sunday Evening** (Frontend):
```
✓ rainmeter/DreamWindow/DreamWindow.ini
✓ rainmeter/@Resources/Variables.inc
✓ rainmeter/@Resources/Images/ (assets)
```

---

## 💾 Backup Recommendations

**Essential files to backup**:
```
backend/config.yaml              # Your settings
cache/metadata/cache_index.json  # Cache metadata
seeds/angels/*.png               # Your source images
rainmeter/@Resources/Variables.inc  # Widget config
comfyui_workflows/*.json         # Workflows
```

**Quick backup command**:
```powershell
cd C:\AI\DreamWindow
tar -czf dreamwindow_backup_$(Get-Date -Format 'yyyyMMdd').tar.gz `
    backend/config.yaml `
    cache/metadata/ `
    seeds/ `
    rainmeter/@Resources/Variables.inc `
    comfyui_workflows/
```

---

## 🚀 Startup Files

**ComfyUI Launch**:
```
C:\AI\ComfyUI\ComfyUI_windows_portable\run_nvidia_gpu.bat
```

**Python Controller Launch**:
```
C:\AI\DreamWindow\venv\Scripts\activate
python C:\AI\DreamWindow\backend\main.py
```

**Optional: Create startup batch file**:
```batch
@echo off
REM start_dreamwindow.bat

REM Start ComfyUI
cd C:\AI\ComfyUI\ComfyUI_windows_portable
set CUDA_VISIBLE_DEVICES=1
start "ComfyUI" run_nvidia_gpu.bat

REM Wait for ComfyUI to start
timeout /t 10

REM Start Python Controller
cd C:\AI\DreamWindow
call venv\Scripts\activate
python backend\main.py
```

---

## 📦 Clean Install Checklist

**Starting from scratch:**

1. **Create root directory**:
   ```
   mkdir C:\AI
   ```

2. **Extract ComfyUI** (from portable download)

3. **Clone/create DreamWindow** (empty structure)

4. **Run setup scripts**:
   ```powershell
   python backend/setup_directories.py
   ```

5. **Download Flux model** → `ComfyUI/models/checkpoints/`

6. **Copy seed images** → `seeds/angels/`

7. **Configure paths** in `config.yaml` and `Variables.inc`

8. **Install Python deps**:
   ```powershell
   pip install -r backend/requirements.txt
   ```

9. **Test generation**:
   ```powershell
   python backend/core/generator.py
   ```

10. **Load Rainmeter widget**

---

## 🎨 Asset Generation Order

**Create these assets** (optional but recommended):

1. **scanlines.png** (for CRT effect)
   ```powershell
   python scripts/generate_scanlines.py
   ```

2. **glow_overlay.png** (for pulsing glow)
   ```powershell
   python scripts/generate_glow.py
   ```

3. **border_frame.png** (pre-rendered frame)
   ```powershell
   python scripts/generate_border.py
   ```

---

## ✅ Verification Checklist

**After complete setup, verify:**

```
File System:
[ ] All directories exist
[ ] Flux model downloaded (~24GB)
[ ] Seed images copied
[ ] Config files created
[ ] Virtual environment setup

Backend:
[ ] ComfyUI launches
[ ] Python can import all modules
[ ] Test generation works
[ ] Cache directory writable
[ ] Logs being created

Frontend:
[ ] Rainmeter skin loads
[ ] Variables.inc configured
[ ] Assets present (if using)
[ ] Widget visible on desktop

Integration:
[ ] Status JSON updating
[ ] current_frame.png updating
[ ] Rainmeter displaying images
[ ] Crossfade working
[ ] No errors in logs
```

---

## 🗺️ Navigation Map

```
Starting Point: README.md (this file)
    │
    ├─> New user? → SETUP_GUIDE.md
    │
    ├─> Ready to build? → WEEKEND_SPRINT.md
    │
    ├─> Need reference? → QUICK_REFERENCE.md
    │
    ├─> Something broken? → TROUBLESHOOTING.md
    │
    ├─> Want deep dive? → BACKEND_ARCHITECTURE.md
    │
    └─> Visual design? → AESTHETIC_SPEC.md
```

---

**Project structure complete!** Everything has its place, ready to build. 📂✨
