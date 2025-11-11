# Dream Window

**A living AI dream on your desktop - continuously morphing ethereal imagery that never repeats.**

> *"Like an automated dreams of electric sheep generator. A little HUD that is constantly running diffusion, showing images constantly shifting."*

Dream Window is a desktop widget that displays endlessly evolving AI-generated art. Using a hybrid generation pipeline combining diffusion models with latent space interpolation, it creates smooth, dreamlike transitions between AI-generated keyframes while maintaining a distinctive ethereal technical aesthetic.

<p align="center">
  <img src="example_gens.webp" alt="Dream Window in action - ethereal AI-generated art continuously morphing" width="100%">
  <br>
  <em>Example generation showing the ethereal technical aesthetic</em>
</p>

## ✨ Key Features

- **Hybrid Generation Architecture**: Keyframes generated via img2img diffusion, smoothly interpolated using VAE latent space with spherical linear interpolation (slerp)
- **Buffered Playback System**: 30-second rolling buffer ensures uninterrupted, smooth visual flow
- **CLIP-Based Cache Injection**: Prevents visual mode collapse by intelligently reintroducing aesthetically similar past frames
- **Zero Gaming Impact**: Runs on dedicated GPU with automatic game detection and VRAM management
- **Desktop Integration**: Lightweight Rainmeter widget with configurable styling and live status indicators
- **Production-Ready Daemon**: Autonomous process management with auto-restart, health monitoring, and graceful shutdown

## 🎯 What Makes This Different

Most AI art generators create individual images. Dream Window creates a *continuous stream* - think of it as a window into an algorithm's dreams that morphs through aesthetic space without ever truly repeating.

The secret is in the architecture:
- **Keyframes**: Full diffusion generation provides diversity and detail
- **Interpolations**: VAE latent interpolation between each keyframe provides buttery-smooth transitions
- **Buffer and Queueing**: Allows frames to build up, coordinates between them, makes sure the "current frame" is always available and sequential
- **Cache System**: CLIP embeddings match and reinject past frames to prevent the aesthetic from converging

This hybrid approach gives you both visual quality and real-time performance that pure diffusion could never achieve.

## 🚀 Quick Start

### Prerequisites

- Windows 10/11
- NVIDIA GPU (tested on Maxwell Titan X, works on 10xx and newer)
- Python 3.11 or 3.12
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed
- [Rainmeter](https://www.rainmeter.net/) (for the desktop widget)

### Installation

```bash
# Clone the repository
git clone https://github.com/LuxiaSL/dream_gen.git
cd dream_gen

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv sync

# Configure paths in backend/config.yaml
# Set your ComfyUI path, output directories, etc.
notepad backend\config.yaml

# Install Rainmeter widget
.\rainmeter_skin\install.ps1
```

### Running

```bash
# Option 1: Run everything via daemon (recommended)
uv run daemon.py

# Option 2: Run components separately
# Terminal 1: Start ComfyUI
cd diffusion\ComfyUI
.\run_nvidia_gpu.bat

# Terminal 2: Start Dream Controller
uv run backend\main.py
```

Load the Dream Window skin in Rainmeter and watch the magic happen!

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DREAM WINDOW                            │
│                                                              │
│  ┌──────────────┐      ┌───────────────┐      ┌──────────┐ │
│  │  Rainmeter   │◄─────│  Controller   │◄─────│ ComfyUI  │ │
│  │   Widget     │ File │  (Python)     │ HTTP │ Backend  │ │
│  │              │ Watch│               │ API  │ (GPU #2) │ │
│  └──────────────┘      └───────┬───────┘      └──────────┘ │
│                                 │                            │
│                        ┌────────▼────────┐                  │
│                        │  Frame Buffer   │                  │
│                        │  - Keyframes    │                  │
│                        │  - Interpolated │                  │
│                        │  - 30s buffer   │                  │
│                        └────────┬────────┘                  │
│                                 │                            │
│                        ┌────────▼────────┐                  │
│                        │  Cache Manager  │                  │
│                        │  - CLIP embeds  │                  │
│                        │  - LRU storage  │                  │
│                        │  - Injection    │                  │
│                        └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Generation Flow

1. **Keyframe Generation**: Controller requests diffusion generation via ComfyUI API
2. **Latent Encoding**: Keyframe is encoded to VAE latent space
3. **Interpolation**: Spherical lerp between keyframe latents creates smooth in-betweens
4. **Frame Buffer**: All frames stored in sequence, maintaining 30s rolling buffer
5. **Display Selection**: Buffer provides frames at target FPS (default 4fps) to Rainmeter
6. **Cache Injection**: Periodically injects past frames with similar CLIP embeddings

## 📁 Project Structure

```
dream-gen/
├── backend/
│   ├── core/                    # Core generation logic
│   │   ├── dream_controller.py  # Main orchestrator
│   │   ├── generation_coordinator.py  # Keyframe + interpolation coordination
│   │   ├── frame_buffer.py      # Buffered frame sequencing
│   │   ├── display_selector.py  # Frame selection for display
│   │   ├── generator.py         # ComfyUI API wrapper
│   │   └── workflow_builder.py  # Dynamic workflow construction
│   ├── cache/                   # Aesthetic caching system
│   │   ├── manager.py           # LRU cache manager
│   │   └── aesthetic_matcher.py # CLIP-based similarity matching
│   ├── interpolation/           # Latent space interpolation
│   │   ├── latent_encoder.py    # VAE encoding/decoding
│   │   └── spherical_lerp.py    # Slerp implementation
│   ├── utils/                   # Utilities
│   └── config.yaml              # Main configuration
├── daemon.py                    # Production daemon manager
├── daemon_control.py            # Daemon control interface
├── rainmeter_skin/              # Desktop widget
├── comfyui_workflows/           # Workflow JSON templates
├── seeds/                       # Initial seed images
└── docs/                        # Documentation
```

## ⚙️ Configuration

Key settings in `backend/config.yaml`:

```yaml
system:
  comfyui_url: "http://127.0.0.1:8188"  # ComfyUI API endpoint
  gpu_id: 1                               # Dedicated GPU for generation

generation:
  model: "sd15"                           # or whatever model you want to run inside comfyui
  resolution: [512, 256]                  # Width x Height
  mode: "hybrid"                          # Recommended
  
  hybrid:
    interpolation_frames: 10              # Frames between keyframes
    target_interpolation_fps: 4           # Display framerate
    keyframe_denoise: 0.3                 # Img2img strength
    interpolation_resolution_divisor: 1   # 1=full, 2=half (faster)

  cache:
    max_size: 50                          # Cached frame limit
    injection_probability: 0.15           # 15% chance per keyframe
    similarity_threshold: 0.8             # CLIP similarity for injection

prompts:
  base_themes:
    - "ethereal digital angel, dissolving particles, technical wireframe..."
  rotation_interval: 20                   # Keyframes before theme rotation
```

## 🎨 Aesthetic Customization

The visual style is controlled by:

1. **Prompts** (`config.yaml` → `prompts.base_themes`): Define the aesthetic space
2. **Seed Images** (`seeds/`): Starting points that influence evolution
3. **Denoise Strength** (`config.yaml` → `generation.hybrid.keyframe_denoise`): Controls how much each keyframe drifts

The default aesthetic is "ethereal technical angels" - monochrome with cyan/red accents, particle dissolution, architectural wireframes. Change the prompts and seeds to explore different aesthetic spaces!

## 🎮 Dual-GPU Setup & Game Detection

Dream Window is designed to coexist peacefully with gaming:

- **Dedicated GPU**: Runs on GPU #2, completely isolated from gaming GPU
- **Game Detection**: Monitors process list for known games
- **Auto-Pause**: Automatically pauses generation and frees VRAM when games detected
- **Auto-Resume**: Restarts generation when game closes

Configure in `config.yaml` → `game_detection.known_games`.

## 📊 Performance

**On Maxwell Titan X (12GB) w/ default config:**
- Keyframe generation: ~2.1s (SD 1.5)
- Interpolation: ~0.25s per frame (full res) or ~0.07s (half res)
- Memory usage: ~6-8GB VRAM
- CPU overhead: Negligible (<2%)

**Framerate modes:**
- Full resolution (512x256): ~4 FPS
- 3/4 resolution (384x192): ~8 FPS
- Half resolution (256x128): ~15 FPS
- Configure via `interpolation_resolution_divisor`

## 🔧 Troubleshooting

**ComfyUI not starting?**
- Check `daemon.comfyui.startup_script` path in config.yaml
- Verify ComfyUI runs standalone first
- Check `logs/daemon.log` for errors

**No frames generating?**
- Ensure ComfyUI API is accessible: `curl http://127.0.0.1:8188/system_stats`
- Check GPU availability: `nvidia-smi`
- Review `logs/dream_controller.log`

**Rainmeter widget blank?**
- Verify `output/current_frame.png` exists
- Check ProjectPath in `rainmeter_skin/@Resources/Variables.inc`
- Ensure backend is running

**Frames stuttering?**
- Increase buffer target: `display.buffer_target_seconds: 60`
- Lower resolution: `interpolation_resolution_divisor: 2`
- Reduce interpolation frames: `hybrid.interpolation_frames: 5`

Any other issues, contact @luxia on discord or open an issue.

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **Additional Diffusion Suites**: ComfyUI is strong; could stand to support many others for flexibility/choice, as well as other models
- **Improved Rainmeter Control/Display**: More buttons/knobs to tune the diffusion on the fly from the rainmeter widget itself.
- **Refactoring + Cross-System Capabilities**: System and software agnostic, separating away from Rainmeter explicitly and moving towards independent pieces with the core logic
- **Single GPU Support**: Self explanatory. Hardcoded to try and offload to secondary GPU, can be altered or made to support single ones. Pairs well with system agnostic development.
## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **ComfyUI** by comfyanonymous - The backbone of the generation pipeline
- **Stable Diffusion** - Making this level of AI art accessible
- **Rainmeter** - Elegant desktop customization platform

## 🌟 Gallery

<p align="center">
  <img src="example_gens.webp" alt="Example generations from Dream Window" width="100%">
  <br>
  <em>The system in action - endless variations that never repeat</em>
</p>