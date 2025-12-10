# Dream Window

**A living AI dream on your desktop - continuously morphing ethereal imagery that never repeats.**

> *"Like an automated dreams of electric sheep generator. A little HUD that is constantly running diffusion, showing images constantly shifting."*

Dream Window is a desktop widget that displays endlessly evolving AI-generated art. Using a hybrid generation pipeline combining diffusion models with latent space interpolation, it creates smooth, dreamlike transitions between AI-generated keyframes while maintaining a distinctive ethereal technical aesthetic.

<p align="center">
  <img src="examples/gen_1.webp" alt="Dream Window in action - ethereal AI-generated art continuously morphing" width="45%">
  <img src="examples/gen_3.webp" alt="Dream Window in action - ethereal AI-generated art continuously morphing" width="45%">
  <br>
  <em>Example generations showing the ethereal technical aesthetic</em>
</p>

## 📖 Installation

**New to Dream Window?** Check out the **[Complete Installation Guide](docs/INSTALLATION_GUIDE.md)** for step-by-step instructions from zero to running, including Python setup, ComfyUI installation, and performance optimization for your GPU!

## ✨ Key Features

- **Hybrid Generation Architecture**: Keyframes generated via img2img diffusion, smoothly interpolated using VAE latent space with spherical linear interpolation (slerp)
- **Buffered Playback System**: 30-second rolling buffer ensures uninterrupted, smooth visual flow
- **Dual-Metric Cache Injection**: Prevents visual mode collapse using ColorHist + pHash-8 similarity detection with OR logic for comprehensive collapse prevention
- **Zero Gaming Impact**: Runs on dedicated GPU with automatic game detection and VRAM management
- **Desktop Integration**: Lightweight Rainmeter widget with configurable styling and live status indicators
- **Production-Ready Daemon**: Autonomous process management with auto-restart, health monitoring, and graceful shutdown
- **Cloud Mode**: Stream frames to a web server for browser-based viewing (optional)

## 🎯 What Makes This Different

Most AI art generators create individual images. Dream Window creates a *continuous stream* - think of it as a window into an algorithm's dreams that morphs through aesthetic space without ever truly repeating.

The secret is in the architecture:
- **Keyframes**: Full diffusion generation provides diversity and detail
- **Interpolations**: VAE latent interpolation between each keyframe provides buttery-smooth transitions
- **Buffer and Queueing**: Allows frames to build up, coordinates between them, makes sure the "current frame" is always available and sequential
- **Cache System**: Dual-metric similarity (ColorHist + pHash-8) detects and prevents mode collapse by intelligently reinjecting diverse past frames

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

# Modern GPUs (RTX 20+, GTX 16 series): Just use uv sync!
uv sync

# Older GPUs (GTX 10 series, Pascal): Use CUDA 12.1
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# uv sync

# Very old GPUs (Maxwell Titan X, etc): Use CUDA 11.8
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# uv sync

# See docs/PYTORCH_CUDA_COMPATIBILITY.md for detailed GPU compatibility info

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
│                        │  - Dual-metric  │                  │
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
6. **Cache Injection**: Dual-metric similarity detection (ColorHist + pHash-8) prevents mode collapse

## ☁️ Cloud Mode

Dream Window can run on cloud GPUs and stream frames to a web server for browser-based viewing. This enables:

- **Cross-Platform Access**: Watch Dream Window on any device with a browser
- **Cost-Effective GPU Usage**: Pay only when viewers are present (per-second billing)
- **State Persistence**: Generation resumes seamlessly across GPU restarts

### Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐     WebSocket      ┌─────────────────┐
│  Cloud GPU      │ ──────────────────► │  VPS (æthera)   │ ──────────────────► │  Browsers       │
│  (RunPod)       │   Binary frames    │  Frame Hub      │   Binary frames    │  /dreams page   │
│                 │   + State sync     │                 │   + Status msgs    │                 │
└─────────────────┘                    └─────────────────┘                    └─────────────────┘
```

### Enabling Cloud Mode

1. Set `cloud.enabled: true` in `backend/config.yaml`
2. Configure the VPS WebSocket URL and auth token
3. Deploy to RunPod using the provided Docker image

```yaml
# backend/config.yaml
cloud:
  enabled: true
  vps_websocket_url: "wss://your-domain.com/ws/gpu"
  auth_token: "your_secure_shared_secret"
  
  frame_push:
    format: "webp"
    quality: 85
  
  state_sync:
    interval_keyframes: 10  # Save state every 10 keyframes
    push_on_shutdown: true
  
  resolution_override: [1024, 512]  # Higher res for cloud
```

### Cloud Components

| Component | File | Purpose |
|-----------|------|---------|
| `VPSWebSocketClient` | `cloud/websocket_client.py` | Maintains connection to VPS |
| `CloudFramePusher` | `cloud/frame_pusher.py` | Encodes and transmits frames |
| `CloudStateSync` | `cloud/state_sync.py` | Periodic state snapshots |
| `runpod_handler` | `cloud/runpod_handler.py` | Serverless entry point |

### Backward Compatibility

Cloud mode is **fully optional**. With `cloud.enabled: false` (default), Dream Window runs exactly as before — Rainmeter widget, local output, no external connections.

When cloud mode is enabled, both outputs run simultaneously:
- Local `output/current_frame.png` for Rainmeter
- WebSocket stream to VPS for browsers

### Deployment

See `docs/CLOUD_DEPLOYMENT_PLAN.md` for the full deployment guide, including:
- RunPod setup with Flashboot
- Docker image building
- VPS configuration
- Cost projections

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
│   │   ├── injection_strategy.py # Cache injection strategies
│   │   └── collapse_detector.py # Mode collapse detection
│   ├── cloud/                   # Cloud mode (optional)
│   │   ├── websocket_client.py  # VPS WebSocket connection
│   │   ├── frame_pusher.py      # Frame encoding + transmission
│   │   ├── state_sync.py        # State persistence
│   │   └── runpod_handler.py    # Serverless entry point
│   ├── interpolation/           # Latent space interpolation
│   │   ├── latent_encoder.py    # VAE encoding/decoding
│   │   └── spherical_lerp.py    # Slerp implementation
│   ├── utils/                   # Utility modules
│   │   ├── color_encoder.py     # ColorHist similarity encoder
│   │   ├── phash_encoder.py     # Perceptual hash encoder
│   │   ├── prompt_manager.py    # Prompt rotation
│   │   └── ...
│   └── config.yaml              # Main configuration
├── daemon.py                    # Production daemon manager
├── daemon_control.py            # Daemon control interface
├── docker/                      # Cloud deployment
│   ├── Dockerfile.cloud         # GPU container image
│   └── docker-compose.cloud.yml # Local testing
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
  resolution: [512, 256]                  # Width x Height (update Rainmeter dims too!)
  mode: "hybrid"                          # Recommended
  
  hybrid:
    interpolation_frames: 10              # Frames between keyframes
    target_interpolation_fps: 4           # Display framerate
    keyframe_denoise: 0.3                 # Img2img strength
    interpolation_resolution_divisor: 1   # 1=full, 2=half (faster)

  cache:
    max_size: 50                          # Cached frame limit
    injection_probability: 0.15           # 15% chance per keyframe
    similarity_method: "dual_metric"      # ColorHist + pHash-8 OR logic
    
    color_histogram:
      diversity_threshold: 1.95           # Color diversity threshold
    phash:
      diversity_threshold: 0.82           # Structural diversity threshold

prompts:
  theme_pairs:
    - positive: "ethereal digital angel, dissolving particles, technical wireframe..."
      negative: "colors, warm tones, low quality..."
  rotation_interval: 20                   # Keyframes before theme rotation
```

### 📐 Changing Generation Resolution

To change the output resolution (e.g., to 512x512 or 1024x512):

1. **Update backend config** (`backend/config.yaml`):
   ```yaml
   generation:
     resolution: [1024, 512]  # [width, height]
   ```

2. **Update Rainmeter widget** (`rainmeter_skin/@Resources/Variables.inc`):
   ```ini
   ViewportWidth=1024      # Match width
   ViewportHeight=512      # Match height
   WidgetWidth=1040        # ViewportWidth + 16
   WidgetHeight=608        # ViewportHeight + 96
   ```

3. **Restart both services** for changes to take effect

**Note:** Larger resolutions will reduce FPS. Adjust `interpolation_resolution_divisor` for performance tuning.

## 🎨 Aesthetic Customization

The visual style is controlled by:

1. **Prompts** (`config.yaml` → `prompts.theme_pairs`): Define the aesthetic space with paired positive/negative prompts
2. **Seed Images** (`seeds/`): Starting points that influence evolution
3. **Denoise Strength** (`config.yaml` → `generation.hybrid.keyframe_denoise`): Controls how much each keyframe drifts

The default aesthetic is "ethereal technical angels" - monochrome with cyan/red accents, particle dissolution, architectural wireframes. Each theme now has a tailored negative prompt for better control. Change the prompt pairs and seeds to explore different aesthetic spaces!

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

  <img src="examples/gen_1.webp" alt="Example generations from Dream Window" width="100%">
  <br>

  ---

  <br>
  <img src="examples/gen_2.webp" alt="Example generations from Dream Window" width="100%">

  <br>

  ---

  <br>
  <img src="examples/gen_3.webp" alt="Example generations from Dream Window" width="100%">
  
  <br>

  ---

  <br>
  <img src="examples/gen_4.webp" alt="Example generations from Dream Window" width="100%">
  
  <br>

  ---

  <em>The system in action - endless variations that never repeat</em>
</p>