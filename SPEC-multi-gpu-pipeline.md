# Spec: Multi-GPU Pipeline Overlap

## Problem

Keyframe generation (UNet) and interpolation (VAE decode) are serialized on a single GPU despite being independent operations. With 8x B200 GPUs available, we can pipeline them on separate devices.

## Current Timing (Single GPU)

```
GPU 0: [KF gen 0.4s] → [Interp 0.5s] → [KF gen 0.4s] → [Interp 0.5s]
                                                                    ↑
                                                          Each 20 frames: 0.9s serial
                                                          = ~22 FPS theoretical, ~15 FPS actual
```

## Proposed: UNet on GPU 0, VAE on GPU 1

```
GPU 0 (UNet):  [KF gen 0.4s] ─────────── [KF gen 0.4s] ─────────── [KF gen]
GPU 1 (VAE):                [Interp 0.5s] ──────────── [Interp 0.5s] ───────
                            ↑
                  Each 20 frames: max(0.4, 0.5) = 0.5s
                  = ~40 FPS theoretical, ~25-30 FPS realistic
```

## Why This Works

1. **UNet and VAE are independent models** — they don't share weights or intermediate state
2. **The async orchestrator already separates them as tasks** — KeyframeWorker and InterpolationWorker
3. **The VAE lock prevents concurrent access to ONE VAE** — with VAE on a separate GPU, the lock still works but no longer blocks UNet
4. **Latent tensors are small** (~256KB each) — transferring between devices is negligible
5. **The `injection_vae` parameter already exists** in the orchestrator for a dedicated second VAE

## Implementation

### 1. Config: Add device mapping

```yaml
system:
  backend: "direct"
  # Multi-GPU: assign components to different devices
  # Default: all on gpu_id (backward compatible)
  gpu_id: 0                    # Default device (UNet)
  gpu_devices:
    unet: 0                    # Keyframe generation (UNet + scheduler)
    vae: 1                     # VAE encode/decode (interpolation + injection)
```

### 2. DirectSDBackend: Separate device for UNet

**File**: `core/direct_sd_backend.py`

```python
# In __init__:
gpu_devices = config.get("system", {}).get("gpu_devices", {})
default_gpu = config.get("system", {}).get("gpu_id", 0)
self.unet_device = f"cuda:{gpu_devices.get('unet', default_gpu)}"
self.vae_device = f"cuda:{gpu_devices.get('vae', default_gpu)}"
self.device = self.unet_device  # Default device for backward compat

# In _setup_pipeline:
# Load full pipeline to UNet device first
self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(...).to(self.unet_device)

# If VAE device differs, move VAE to its own GPU
if self.vae_device != self.unet_device:
    logger.info(f"Multi-GPU: Moving VAE to {self.vae_device}")
    self._txt2img_pipe.vae.to(self.vae_device)
```

**Note**: The img2img pipeline shares the UNet with txt2img. The VAE lives separately. When generating a keyframe, the UNet runs on GPU 0 and the final VAE decode (producing the output image) runs on GPU 1. The pipeline handles device mismatches via `latents.to(vae_device)` before VAE decode.

### 3. LatentEncoder: Accept VAE device

**File**: `interpolation/latent_encoder.py`

```python
def __init__(self, ..., device: str = "cuda", ...):
    # device is now the VAE device (may differ from UNet device)
    self.device = device
```

When `DreamController._init_cloud_mode()` creates the LatentEncoder, it passes the VAE device:

```python
vae_device = f"cuda:{config.get('system', {}).get('gpu_devices', {}).get('vae', gpu_id)}"
self.latent_encoder = LatentEncoder(
    device=vae_device,
    ...
)
```

### 4. Latent Transfer Between Devices

When the interpolation worker encodes a keyframe (which lives on GPU 0 as an image), it needs to:
1. Load the keyframe image (CPU operation, device-agnostic)
2. Preprocess to tensor → move to VAE device (GPU 1)
3. Encode to latent on GPU 1
4. Slerp on GPU 1 (latents already there)
5. Decode batch on GPU 1

The only cross-device transfer is the keyframe image tensor, which happens during VAE encode. This is already handled by `LatentEncoder._preprocess_image()` which moves to `self.device`.

### 5. Pipeline Overlap in Orchestrator

**File**: `core/async_orchestrator.py`

Currently, the orchestrator waits for keyframe completion before submitting interpolation. With multi-GPU, we can submit the interpolation immediately after the keyframe is done — the VAE work happens on GPU 1 while the next keyframe starts generating on GPU 0.

The critical change: **don't wait for interpolation to complete before starting the next keyframe**.

Currently (simplified):
```python
# Wait for keyframe to complete
keyframe_result = await keyframe_worker.get_result()

# Submit interpolation (blocks GPU via lock)
await interp_worker.submit_pair(...)

# Wait for interpolation (implicit — next loop iteration)
# Only THEN start next keyframe
```

With multi-GPU, the orchestrator can:
```python
# Wait for keyframe to complete
keyframe_result = await keyframe_worker.get_result()

# Submit interpolation (runs on GPU 1, non-blocking for GPU 0)
await interp_worker.submit_pair(...)

# Immediately start next keyframe (runs on GPU 0)
await keyframe_worker.submit(next_prompt, ...)
# Both run in parallel!
```

This overlap is actually already close to how it works — the orchestrator loop submits keyframes and interpolations as separate async operations. The bottleneck is that they compete for the same GPU. With separate GPUs, the existing async structure naturally overlaps them.

### 6. share_vae() Compatibility

`DirectSDBackend.share_vae()` lets the LatentEncoder reuse the pipeline's VAE instead of loading a duplicate. With multi-GPU:
- If UNet and VAE are on the same device → `share_vae()` works as before
- If they're on different devices → `share_vae()` should move the shared VAE to the VAE device, or LatentEncoder loads its own copy on GPU 1

Since B200 has 183GB VRAM per GPU, loading a separate 500MB VAE copy on GPU 1 is negligible. Simpler than sharing across devices.

```python
# In dream_controller.py, when multi-GPU:
if unet_device != vae_device:
    # Don't share VAE across devices — load separate copy on VAE GPU
    self.latent_encoder = LatentEncoder(device=vae_device, auto_load=True, ...)
else:
    # Single GPU: share VAE to save VRAM
    self.generator.share_vae(self.latent_encoder)
```

## Files Changed

| File | Change | Complexity |
|------|--------|------------|
| `config.b200.yaml` | Add `gpu_devices` mapping | Trivial |
| `core/direct_sd_backend.py` | Split `self.device` into `unet_device`/`vae_device` | Medium |
| `core/dream_controller.py` | Pass separate devices to components, skip `share_vae` on multi-GPU | Medium |
| `interpolation/latent_encoder.py` | Already accepts `device` param — no change needed | None |
| `core/shared_resources.py` | No change — lock still works, just on different GPU | None |
| `core/async_orchestrator.py` | No change — overlap is natural with separate GPUs | None |
| `core/workers/interpolation_worker.py` | No change — uses SharedVAEAccess | None |
| `core/workers/keyframe_worker.py` | No change — uses DirectSDBackend | None |

## Memory Impact

| GPU | Current | Multi-GPU |
|-----|---------|-----------|
| GPU 0 | UNet (~1.6GB) + VAE (~500MB) + latents | UNet (~1.6GB) only |
| GPU 1 | Unused | VAE (~500MB) + latents + TAESD (~20MB) |
| Total | ~2.1GB of 183GB | ~2.1GB across 2 × 183GB |

Negligible. Each B200 has 183GB — we're using <2% of one GPU.

## Expected Performance

| Metric | Current (1 GPU) | Multi-GPU (2 GPU) |
|--------|-----------------|-------------------|
| Keyframe gen | 0.1-0.5s (serial) | 0.1-0.5s (parallel with interp) |
| Interpolation | 0.5s (serial) | 0.5s (parallel with keyframe) |
| Cycle time | ~0.9s (serial sum) | ~0.5s (parallel max) |
| Frames/cycle | 20 | 20 |
| Theoretical FPS | ~22 | ~40 |
| Expected actual FPS | ~15 | ~25-30 |

## Implementation Order

1. Add `gpu_devices` config option with backward-compatible fallback to `gpu_id`
2. Update `DirectSDBackend` to use separate UNet/VAE devices
3. Update `DreamController` to pass correct devices and skip `share_vae` when multi-GPU
4. Test on B200 with `gpu_devices: {unet: 0, vae: 1}`
5. Measure FPS improvement
