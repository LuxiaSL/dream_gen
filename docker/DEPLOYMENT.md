# DreamGen Deployment Guide

## Docker Architecture

```
┌──────────────────────┐      ┌──────────────────────┐
│  ComfyUI Image       │      │  DreamGen Image      │
│  (stable, ~4GB)      │←────→│  (changes often)     │
│                      │      │                      │
│  - CUDA 12.1         │      │  - Backend code      │
│  - SD 1.5 model      │      │  - Prompts/YAMLs     │
│  - FSampler node     │      │  - Config            │
│  - PyTorch           │      │  - VAE for interp    │
└──────────────────────┘      └──────────────────────┘
    Dockerfile.comfyui           Dockerfile.dreamgen
```

## Building Images

### ComfyUI (rarely changes)
```bash
# Build locally
docker build -f docker/Dockerfile.comfyui -t luxiasl/dreamgen-comfyui:latest .

# Push to registry
docker push luxiasl/dreamgen-comfyui:latest
```

### DreamGen Backend (changes often)
```bash
# Build with prompts + embeddings
docker build -f docker/Dockerfile.dreamgen -t luxiasl/dreamgen-backend:latest .

# Push to registry
docker push luxiasl/dreamgen-backend:latest
```

### Combined Image (legacy/simple)
```bash
# Uses Dockerfile.base + Dockerfile.cloud for single-container deployment
docker build -f docker/Dockerfile.base -t luxiasl/dreamgen-base:latest .
docker build -f docker/Dockerfile.cloud -t luxiasl/dreamgen:latest .
```

## RunPod Deployment

### Option 1: Serverless (Production)

1. Push combined image to Docker Hub
2. Create RunPod Serverless endpoint
3. Configure Flashboot for fast cold starts

### Option 2: Pod (Development/Testing)

```bash
# SSH to pod and setup
cd /workspace
git clone https://github.com/LuxiaSL/dream_gen.git
cd dream_gen

# ComfyUI setup (one-time)
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
cd /workspace/ComfyUI
pip install -r requirements.txt
git clone https://github.com/obisin/comfyui-FSampler custom_nodes/comfyui-FSampler
wget -O models/checkpoints/v1-5-pruned-emaonly.safetensors \
  "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly.safetensors"

# Start ComfyUI
nohup python main.py --listen 0.0.0.0 --port 8188 --highvram --cuda-malloc &

# Run DreamGen
cd /workspace/dream_gen/backend
PYTHONPATH=/workspace/dream_gen/backend python main.py --config config.pod.yaml
```

## Flashboot Configuration (RunPod)

Flashboot pre-loads Docker image layers for faster cold starts.

### Setup Steps:

1. **Build optimized image:**
   ```bash
   docker build -f docker/Dockerfile.comfyui -t luxiasl/dreamgen-comfyui:latest .
   docker push luxiasl/dreamgen-comfyui:latest
   ```

2. **Enable Flashboot on RunPod:**
   - Go to Serverless > Endpoints
   - Edit endpoint settings
   - Enable "Flashboot" option
   - Select your image

3. **Layer optimization:**
   - Keep model download layer separate (cached)
   - Code layer should be thin (~50MB)
   - Models layer is ~4GB (only downloaded once)

### Expected Cold Start Times:
- Without Flashboot: ~90s (download + extract)
- With Flashboot: ~10-15s (pre-loaded layers)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COMFYUI_URL` | ComfyUI API endpoint | `http://127.0.0.1:8188` |
| `VPS_WEBSOCKET_URL` | VPS for frame streaming | `wss://aetherawi.red/ws/gpu` |
| `DREAM_GEN_AUTH_TOKEN` | Auth token for VPS | None |
| `CONFIG_PATH` | Config file path | `/app/backend/config.cloud.yaml` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## Seedless Operation

DreamGen can now start without seed images:

1. If `seeds/` directory has images → uses random seed
2. If `seeds/` is empty → generates first frame via txt2img

To run fully seedless, ensure ComfyUI is running before starting DreamGen.

## Local Testing

```bash
# Start everything locally
docker-compose -f docker/docker-compose.cloud.yml up

# With VPS connection
VPS_WEBSOCKET_URL=wss://aetherawi.red/ws/gpu docker-compose -f docker/docker-compose.cloud.yml up
```

