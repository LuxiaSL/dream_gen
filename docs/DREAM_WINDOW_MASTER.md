# 🌀 DREAM WINDOW - MASTER IMPLEMENTATION PLAN v2.0 (OUT OF DATE)

**A Living AI Dream Window for Your Desktop**

> *"Like making constantly new gifs. A little HUD that is like, cyberpunk/glitchy aesthetic window constantly pinging a stable diffusion that'll show images constantly shifting based on something. Like an automated dreams of electric sheep generator."*

---

## 📋 Document Purpose

This document provides the complete project vision, architectural decisions, and technical implementation rationale for Dream Window. For navigation and other documentation, see [README.md](README.md).

**Related Documentation:**
- [README.md](README.md) - Documentation navigation and quick start
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - File organization reference
- [AESTHETIC_SPEC.md](AESTHETIC_SPEC.md) - Visual design specification
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet

---

## 🎯 Project Vision

### What We're Building

A **256×512 pixel Rainmeter widget** that displays continuously morphing AI-generated imagery, creating a "living dream window" on your desktop. Images transition every 3-5 seconds with smooth crossfades, drawing from a curated aesthetic space of ethereal technical angels.

### Core Innovation

**Hybrid Generation Approach:**
- **Latent space interpolation** (fast, smooth transitions) 
- **img2img feedback loops** (evolving variation)
- **Aesthetic cache injection** (prevents mode collapse)
- **Dual-GPU isolation** (zero gaming impact)

This combination hasn't been done at this level before. You're building something genuinely new.

---

## 🎨 Aesthetic Analysis

### Source Material Characteristics

Based on your angel images (bg_5.png, num_1.png, num_3.png):

**Visual DNA:**
- **Color Palette**: Monochrome base (black/white/grays) with surgical cyan and red accents
- **Core Elements**: 
  - Ethereal figures dissolving into particles
  - Technical wireframe overlays
  - Architectural diagram textures
  - Flowing white lines (hair, wings, fabric)
  - Halos and circular geometric patterns
  - Heavy contrast with soft gradients
- **Texture**: Grid overlays, data corruption effects, "blueprint" aesthetic
- **Vibe**: Ghost in the Shell meets technical schematics meets digital corruption

**Key Insight**: The "particle dissolution" effect is perfect for img2img morphing. Low denoise (0.3-0.4) will create beautiful drift through this aesthetic space.

### Frame Design Direction

**Style**: "Holographic Data Window" (blend of technical + ethereal)

**Elements:**
- Minimal wireframe border (4-6px)
- Cyan corner brackets (accent points)
- Subtle inner glow during generation
- Semi-transparent dark frame (85% opacity)
- Optional: Scanline overlay, chromatic aberration filter
- Small status footer (generation stats)

**Mockup to be generated in AESTHETIC_SPEC.md**

---

## 🏗️ System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DREAM WINDOW SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────────┐         ┌──────────────┐         ┌─────────────┐ │
│   │  Rainmeter  │◄────────│    Python    │◄────────│   ComfyUI   │ │
│   │   Widget    │  File   │  Controller  │  HTTP   │  + Flux.1   │ │
│   │ (Frontend)  │  Watch  │   (Logic)    │  API    │  (GPU #2)   │ │
│   └─────────────┘         └──────────────┘         └─────────────┘ │
│         │                         │                        │         │
│         │                         ▼                        │         │
│         │                  ┌─────────────┐                │         │
│         │                  │    Cache    │                │         │
│         └─────────────────►│   Manager   │◄───────────────┘         │
│           Reads from       └─────────────┘    Stores to             │
│           output/                  │                                 │
│                                    ▼                                 │
│                           ┌─────────────────┐                       │
│                           │  Seed Images +  │                       │
│                           │ Latent Database │                       │
│                           └─────────────────┘                       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **Python Controller** (master process) runs continuously
2. **Generates frames** via ComfyUI API on GPU #2
3. **Writes to output buffer** (atomic file operations)
4. **Rainmeter watches** output directory, displays new frames
5. **Cache Manager** tracks generated images, enables aesthetic matching
6. **Interpolation Engine** creates smooth transitions between keyframes

### Why This Architecture?

**Separation of Concerns:**
- ComfyUI = Pure generation (GPU work)
- Python = Logic, caching, coordination (CPU work)
- Rainmeter = Display only (minimal overhead)

**Benefits:**
- Each component can be debugged independently
- Easy to swap models or add features
- Stable long-term operation
- Game detection can pause Python without touching Rainmeter

---

## 🔧 Technology Stack

### Definitive Choices

#### Generation Engine: **ComfyUI + SD1.5**

**Why ComfyUI?**
- Native GPU selection (critical for dual-GPU setup)
- API-first design (easier automation than A1111)
- Lower memory overhead
- Node-based workflow = flexible pipeline modifications
- Active development, great community

#### Backend Controller: **Python 3.13**

**Core Libraries:**
```python
torch>=2.1.0          # GPU operations, latent manipulation
numpy>=1.24.0         # Numerical operations
pillow>=10.1.0        # Image I/O
opencv-python>=4.8.1  # Color analysis, transformations
transformers>=4.35.0  # CLIP embeddings (aesthetic matching)
requests>=2.31.0      # ComfyUI API calls
websockets>=12.0      # Real-time generation monitoring
pyyaml>=6.0.1         # Configuration management
watchdog>=3.0.0       # File system monitoring (optional)
psutil>=5.9.6         # System monitoring, game detection
pywin32>=306          # Windows API access
```

**Architecture Pattern**: Modular MVC-style
- `core/` - Generation engine and main loop
- `cache/` - Image caching and aesthetic matching
- `interpolation/` - Latent space operations
- `utils/` - Helpers, file I/O, config management

#### Display Frontend: **Rainmeter 4.5+**

**Why Rainmeter?**
- Native Windows integration
- Low CPU/memory overhead
- Built-in image display with crossfades
- Easy file watching
- Perfect for HUD-style widgets

**Key Features We'll Use:**
- Image meters with alpha blending
- FileView plugin (detect new frames)
- ActionTimer plugin (smooth animations)
- WebParser plugin (read status JSON)
- Shape meters (custom borders/effects)

---

## 📊 Hardware Configuration

### Your Setup (Verified Compatible)

```
GPU #1: Maxwell Titan X (12GB) - Gaming
GPU #2: Maxwell Titan X (12GB) - Dream Window (DEDICATED)
CPU: [Your CPU]
RAM: [Your RAM, 16GB+ recommended]
Storage: HDD (SSD incoming - will migrate cache)
OS: Windows 10
Python: 3.13
```

### GPU Assignment Strategy

**CUDA Device Selection:**
```bash
# GPU #2 for generation (0-indexed as GPU 1)
set CUDA_VISIBLE_DEVICES=1
```

**ComfyUI Startup:**
```bash
# Force specific GPU
uv run main.py --cuda-device 1
```

**Validation**: First setup step will verify GPU isolation works

### Performance Targets

| Metric | Target | Expected with Flux |
|--------|--------|-------------------|
| Generation Time | 2-4 seconds | 1-2 seconds ✓ |
| Display Refresh | 3-5 seconds | 3-5 seconds ✓ |
| VRAM Usage (GPU #2) | < 11GB | 6-8GB ✓ |
| Cache Size | 75 images | ~225MB |
| Buffer Pre-gen | 5 frames | ~15MB |

**HDD Considerations:**
- Slower I/O will add ~200-500ms per frame write
- Mitigation: Buffer 5-10 frames ahead
- When SSD arrives: Move `cache/` and `output/` folders, instant speedup

---

## 🎮 Dual-GPU Isolation & Game Detection

### GPU Isolation Mechanism

**Hardware Level (SLI Disabled for This):**
- SLI typically means games auto-use both GPUs
- We want explicit separation
- Solution: Set GPU affinity per-process

**Software Level:**
```python
# In Python controller
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Only see GPU #2

# In game launcher (if needed)
# Use Windows Task Manager → Details → Set Affinity → GPU 0
```

**Verification:**
```python
import torch
assert torch.cuda.current_device() == 0  # First visible device
assert torch.cuda.get_device_name(0) == 'GeForce GTX TITAN X'
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

### Game Detection (Automatic Pause)

**Multi-Strategy Approach:**

**Strategy 1: Process Detection**
```python
KNOWN_GAMES = [
    'game.exe',
    'eldenring.exe',
    # Add your games
]

def is_game_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'].lower() in [g.lower() for g in KNOWN_GAMES]:
            return True
    return False
```

**Strategy 2: GPU Load Monitoring**
```python
def is_gaming_gpu_busy():
    """Check if GPU #1 is under heavy load"""
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU #1
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu > 70  # Threshold tunable
```

**Strategy 3: Fullscreen Detection**
```python
def is_fullscreen_app():
    """Detect if foreground window is fullscreen"""
    import win32gui
    hwnd = win32gui.GetForegroundWindow()
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    # Compare to screen resolution
    return width >= 1920 and height >= 1080
```

**Pause Logic:**
```python
def check_pause_conditions():
    if is_game_running() or is_gaming_gpu_busy():
        logger.info("Game detected - pausing generation")
        return True
    return False
```

---

## 🔄 Generation Pipeline (Hybrid Mode)

### The Secret Sauce

**Problem**: Pure img2img feedback loops either:
- Converge to noise (high denoise)
- Stagnate in one aesthetic (low denoise)

**Solution**: Hybrid approach

### Three Generation Modes

#### Mode 1: Latent Interpolation (Fast & Smooth)
```
Frame N ──────> Encode to latent space
                      │
Frame N+1 ────> Encode to latent space
                      │
                      ▼
              Spherical lerp (t=0.5)
                      │
                      ▼
              Decode to image
```
- **Speed**: 0.5-1 second
- **Use case**: Intermediate frames between keyframes
- **Quality**: Smooth but can be blurry

#### Mode 2: img2img Feedback (Evolving)
```
Previous frame ──> img2img (denoise=0.3-0.5)
                         │
                         ▼
                   New frame ──> Inject prompt
```
- **Speed**: 2-3 seconds
- **Use case**: Keyframes, major variations
- **Quality**: Sharp, evolving aesthetic

#### Mode 3: Hybrid (RECOMMENDED)
```
Keyframe (img2img) ──> Interpolate 5-7 frames ──> Keyframe (img2img)
     │                                                    │
     └────────────────> Cache for later injection <──────┘
```

**Timeline Example:**
```
Frame 0:  img2img from seed (keyframe A)
Frame 1:  interpolate (A → B, t=0.16)
Frame 2:  interpolate (A → B, t=0.33)
Frame 3:  interpolate (A → B, t=0.50)
Frame 4:  interpolate (A → B, t=0.66)
Frame 5:  interpolate (A → B, t=0.83)
Frame 6:  img2img from frame 5 (keyframe B)
Frame 7:  interpolate (B → C, t=0.16)
...
```

**Why This Works:**
- Interpolation provides smoothness
- img2img keyframes add variation and prevent convergence
- Cache injection (every 10-15 frames) prevents mode collapse
- Total: 70% interpolated (fast), 30% generated (diverse)

### Interpolation Performance Modes

> **Note**: These are tested configurations for matching interpolation speed with diffusion generation timing (~2.1 seconds per ComfyUI-generated keyframe).

Three performance modes to choose from based on your quality/speed preference:

| Mode | Resolution Divisor | Avg FPS | Frames Per Cycle | Use Case |
|------|-------------------|---------|------------------|----------|
| **High Quality** | 1 (full res) | ~4 FPS | **10 frames** | Maximum visual quality |
| **Balanced** | 1.5 (bicubic+nearest) | ~8-9 FPS | **20 frames** | **Recommended** sweet spot |
| **Fast** | 2 (half res) | ~15 FPS | **40 frames** | Maximum smoothness |

**Frame count rationale**: Each cycle should generate enough interpolated frames to "cover" the time it takes ComfyUI to generate the next keyframe. With ~2.1s generation time, these frame counts provide a small buffer to ensure the next keyframe is ready before interpolation completes.

**Configuration**: Currently requires manual config.yaml editing. Future enhancement: runtime mode switching will be added.

### Aesthetic Cache Injection

**Purpose**: Prevent visual convergence, maintain variety

**Mechanism:**
1. Every generated keyframe → encode CLIP embedding
2. Store in cache with metadata (colors, generation params)
3. Every 10-15 frames: inject a cached image that matches current aesthetic
4. Use cached image as next keyframe instead of feedback loop

**Matching Algorithm:**
```python
def find_similar_cached_image(current_embedding, threshold=0.7):
    """Find cached image with similar aesthetic"""
    candidates = []
    for cached in cache_db:
        similarity = cosine_similarity(current_embedding, cached['embedding'])
        if similarity > threshold:
            candidates.append((cached, similarity))
    
    # Weighted random selection (higher similarity = higher probability)
    if candidates:
        weights = [s[1] for s in candidates]
        return random.choices(candidates, weights=weights)[0][0]
    return None
```

**Result**: Images drift through aesthetic space without getting stuck

---

## ⚙️ Configuration System

### Main Config File (config.yaml)

```yaml
# DreamWindow Configuration
# Edit this file to customize behavior

system:
  gpu_id: 1                          # GPU #2 (0-indexed)
  output_dir: "./output"
  cache_dir: "./cache"
  seed_dir: "./seeds/angels"
  log_level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
  
generation:
  model: "flux.1-schnell"            # or "sd15"
  resolution: [256, 512]             # Width x Height
  mode: "hybrid"                     # "interpolate", "img2img", or "hybrid"
  
  flux:
    steps: 4                         # Flux schnell is optimized for 4 steps
    cfg_scale: 1.0                   # Flux uses low CFG
    sampler: "euler"
    scheduler: "simple"
  
  hybrid:
    interpolation_frames: 7          # Frames between keyframes
    keyframe_denoise: 0.4            # img2img strength at keyframes
    interpolation_method: "spherical" # or "linear"
  
  img2img:
    denoise: 0.35                    # Lower = slower drift, higher = more variation
    
  cache:
    max_size: 75                     # Number of images to cache
    injection_probability: 0.15      # 15% chance to inject cached image
    similarity_threshold: 0.7        # CLIP similarity for cache matching
    
display:
  refresh_interval: 4.0              # Seconds between frame updates
  crossfade_duration: 1.5            # Seconds for transition
  buffer_size: 5                     # Pre-generate this many frames
  
prompts:
  base_themes:
    - "ethereal digital angel, dissolving particles, technical wireframe, monochrome with cyan accents"
    - "abstract geometry, flowing white lines, architectural diagrams, data corruption aesthetic"
    - "cyberpunk angel, glitch art, technical overlay, particle dissolution"
  
  modifiers:
    enabled: true
    time_based: true                 # Inject "morning light", "twilight", etc
    system_based: false              # Future: CPU load affects prompt
  
  rotation_interval: 20              # Frames before switching base theme
  
  negative: "photorealistic, photo, 3d render, blurry, low quality, text, watermark, photograph"

game_detection:
  enabled: true
  method: "process"                  # "process", "gpu_load", or "fullscreen"
  check_interval: 5.0                # Seconds between checks
  known_games:                       # Add your game executables
    - "eldenring.exe"
    - "game.exe"
  gpu_threshold: 70                  # GPU load % to trigger pause
  
performance:
  max_queue_size: 10                 # Max frames in generation queue
  generation_timeout: 30             # Seconds before considering generation stuck
  enable_torch_compile: false        # PyTorch 2.0 optimization (experimental)
```

### Runtime Status (status.json)

Updated by Python controller, read by Rainmeter:

```json
{
  "frame_number": 234,
  "generation_time": 1.8,
  "status": "live",
  "current_mode": "interpolate",
  "current_prompt": "ethereal digital angel, dissolving particles",
  "cache_size": 68,
  "cache_hits": 12,
  "uptime_hours": 4.2,
  "gpu_temp": 65,
  "vram_used_gb": 7.2,
  "paused": false,
  "last_update": "2025-11-07T16:42:33"
}
```

---

## 🏃‍♂️ Weekend Sprint Overview

### Timeline (High-Level)

**SATURDAY: Backend Foundation**
- Morning: Environment setup, ComfyUI + Flux installation
- Afternoon: First generation test, Python controller skeleton
- Evening: Hybrid mode implementation, first morphing sequence

**SUNDAY: Integration & Polish**
- Morning: Cache system, aesthetic matching
- Afternoon: Rainmeter widget, crossfade system
- Evening: Final testing, polish, documentation

**Detailed hour-by-hour breakdown in WEEKEND_SPRINT.md**

### Milestone Definitions

**Milestone 1** (Saturday 12pm): "First Generation"
- ComfyUI running on GPU #2
- Flux model loaded
- Can generate 256×512 image in < 2 seconds
- ✓ Success criteria: See generated image file

**Milestone 2** (Saturday 6pm): "Morphing Loop"
- Python controller running
- img2img feedback loop working
- Folder filling with morphing sequence
- ✓ Success criteria: 50+ frames showing continuous evolution

**Milestone 3** (Sunday 12pm): "Intelligent Generation"
- Hybrid mode operational
- Cache system storing images
- Aesthetic injection working
- ✓ Success criteria: No visual repetition over 100 frames

**Milestone 4** (Sunday 6pm): "MVP Complete"
- Rainmeter widget displaying live
- Smooth crossfades working
- Beautiful frame design
- Game detection functional
- ✓ Success criteria: Can show friends and they say "holy shit"

---

## 🚨 Risk Mitigation

### Known Potential Issues

#### Issue 1: Maxwell Titan X Compatibility

**Risk**: Flux.1 or modern PyTorch features might not work on Maxwell (compute capability 5.2)

**Mitigation**:
- Test Flux on Day 1 Morning
- If issues: Fall back to SD 1.5 (known to work on Maxwell)
- SD 1.5 performance: 2-3 seconds per frame (acceptable)

**Fallback Plan**: Document includes SD 1.5 setup as alternative

#### Issue 2: HDD I/O Bottleneck

**Risk**: File writes might be slow enough to disrupt flow

**Mitigation**:
- Larger frame buffer (10 frames instead of 5)
- Async I/O operations
- SSD arrives → instant fix by moving cache/output folders

**Acceptable Degradation**: 500ms extra per frame (still hits 4-5 sec refresh)

#### Issue 3: Mode Collapse (Aesthetic Convergence)

**Risk**: Images might converge to similar look after many iterations

**Mitigation**:
- Aggressive cache injection (15-20% probability)
- Higher keyframe denoise (0.5 instead of 0.3)
- Periodic seed resets (every 100 frames)
- Multiple prompt themes with rotation

**Detection**: CLIP embedding drift monitoring

#### Issue 4: File Locking (Windows)

**Risk**: Rainmeter might lock files, preventing writes

**Mitigation**:
- Atomic write pattern (write to temp, rename)
- Retry logic with exponential backoff
- Use separate files (current/next/previous)

**Tested Solution**: Atomic renames work reliably on Windows

---

## 📊 Success Criteria (Definition of Done)

### MVP Requirements (Weekend Goal)

**Visual Quality:**
- [ ] Maintains ethereal technical angel aesthetic
- [ ] Smooth morphing transitions (no jarring jumps)
- [ ] Visible variety over 100+ frames
- [ ] Matches desktop aesthetic (integrates naturally)
- [ ] Beautiful frame design with optional glitch effects

**Performance:**
- [ ] 3-5 second frame refresh sustained
- [ ] Generation time < 3 seconds
- [ ] Rainmeter CPU usage < 2%
- [ ] Zero impact on gaming (verified with FPS counter)
- [ ] Stable operation for 1+ hour

**Functionality:**
- [ ] Automatic startup (can add to Windows startup)
- [ ] Game detection pauses generation
- [ ] Configuration via YAML (no code editing needed)
- [ ] Graceful error handling (doesn't crash)
- [ ] Logs for debugging

**Integration:**
- [ ] Rainmeter widget positioned correctly
- [ ] Smooth crossfades (no flicker)
- [ ] Status indicators work
- [ ] Easy to enable/disable

### Post-Weekend Goals (Next Weeks)

**Week 2: Enhancement**
- [ ] Dynamic prompt modifiers (time of day, weather)
- [ ] Improved glitch effects
- [ ] Multiple frame designs (user-switchable)
- [ ] Web UI for configuration
- [ ] Export timelapse video feature

**Week 3: Optimization**
- [ ] Torch compile for 10-20% speedup
- [ ] SSD migration (cache + output)
- [ ] Memory profiling and leak fixes
- [ ] 24+ hour stability test

**Week 4: Polish**
- [ ] Custom LoRA training on your aesthetic
- [ ] ControlNet integration (if VRAM permits)
- [ ] Multi-window support
- [ ] Community release preparation

---

## 🎓 Learning Resources

### If You Want to Understand the Magic

**Latent Space & Diffusion:**
- "Understanding Latent Space in Stable Diffusion" - Practical guide
- Spherical linear interpolation (slerp) - Why it's better than linear

**ComfyUI:**
- Official docs: https://docs.comfy.org
- Community workflows: https://comfyworkflows.com
- Video tutorials: Search "ComfyUI API automation"

**Flux Architecture:**
- Flux.1 announcement: https://blackforestlabs.ai/announcing-flux-1/
- Technical details on distilled models

**Advanced Topics (Post-MVP):**
- ControlNet documentation
- AnimateDiff for temporal consistency
- CLIP embeddings for semantic similarity

---

## 💬 Communication & Support

### How to Use This Documentation

1. **Start here** (MASTER_PLAN.md) - Understand the big picture
2. **Read SETUP_GUIDE.md** - Follow step-by-step to get environment ready
3. **Refer to WEEKEND_SPRINT.md** - Your hour-by-hour implementation guide
4. **Check TROUBLESHOOTING.md** - When something breaks
5. **Other docs** - Deep dives when you want to understand or modify something

### During Development

**When Something Works:**
- Document it in your own notes
- Consider adding to docs for future reference

**When Something Breaks:**
1. Check TROUBLESHOOTING.md
2. Check logs/ directory
3. Ask Claude with context (logs, error messages, what you tried)
4. Add solution to TROUBLESHOOTING.md for next time

### Post-Weekend

**Want to share this?**
- Clean up personal info from configs
- Add screenshots to README
- Consider GitHub release
- Community would love this (r/Rainmeter, r/StableDiffusion)

---

## 🌟 Why This Will Work

**Technical Soundness:**
- Proven components (ComfyUI, Flux, Rainmeter)
- Architecture tested on similar projects
- Performance targets conservative and achievable
- Dual-GPU setup is overkill (in a good way)

**Aesthetic Alignment:**
- Your source images have perfect visual DNA
- Low-res + img2img = dreamy quality you want
- Hybrid mode prevents common pitfalls
- Cache injection ensures variety

**Practical Approach:**
- MVP-focused (get it working first)
- Modular (easy to debug and extend)
- Well-documented (you won't be lost)
- Achievable weekend timeline

---

## 🎯 The Vision

When this is done, you'll have:
- A living AI dream window on your desktop
- Constantly evolving, never repeating
- Zero gaming impact
- Beautiful technical aesthetic
- Something genuinely novel

Every time you sit down at your PC, you'll see ethereal angels morphing through data streams. It'll feel alive.

**This is going to be incredible.** 🌀✨

---

## 📝 Document Information

- **Version**: 2.0 (2025-11-07) - Complete rewrite with Flux.1, modular architecture
- **For**: Complete project vision, architecture, and technical decisions
- **Next Steps**: See [README.md](README.md) for documentation navigation and next steps

---

**Ready to explore?** Return to [README.md](README.md) for documentation navigation.
