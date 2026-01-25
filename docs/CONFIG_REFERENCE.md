# Dream Window Configuration Reference

> Complete documentation for `backend/config.yaml`

This document explains every configuration section in depth, covering the underlying mechanics, how values interact with each other, and the reasoning behind defaults. For quick tuning guidance while editing the config, see the inline comments in `config.yaml` itself.

---

## Table of Contents

1. [System](#1-system)
2. [Generation](#2-generation)
   - [Model Parameters](#21-model-parameters-flux-and-sd)
   - [Hybrid Mode](#22-hybrid-mode)
   - [Cache System](#23-cache-system)
3. [Display](#3-display)
4. [Prompts](#4-prompts)
5. [Game Detection](#5-game-detection)
6. [Performance](#6-performance)
7. [Daemon](#7-daemon)
8. [Cloud](#8-cloud)

---

## 1. System

Basic infrastructure configuration that other components depend on.

### How It Works

The system section defines paths and connection details that are loaded once at startup and used throughout the application lifecycle. These values are accessed via `config['system']['key']` in virtually every major component.

### Parameters

#### `comfyui_url`
**Default:** `"http://127.0.0.1:8188"`

The HTTP endpoint where ComfyUI's API is accessible. The `ComfyUIClient` class uses this to:
- Upload images for img2img operations
- Submit workflow JSON for generation
- Poll for completion status
- Fetch generated images

If ComfyUI runs on a different machine (e.g., a dedicated GPU server), change the IP. If you've configured ComfyUI to use a different port, change that here.

#### `output_dir`
**Default:** `"./output"`

Where the system writes:
- Generated frames (in `frames/keyframes/` and `frames/interpolations/` subdirectories)
- `current_frame.png` (the frame currently displayed by Rainmeter)
- `status.json` (current generation state for external monitoring)
- Daemon control files

Performance consideration: This should be on a fast drive (SSD preferred) since frames are written continuously during operation.

#### `cache_dir`
**Default:** `"./cache"`

Persistent storage for the LRU image cache. Contains:
- `images/` - Cached PNG files
- `metadata/cache_index.json` - Cache state that survives restarts

The cache is separate from output because it should persist across runs while output can be cleaned up. If disk space is limited, you might want this on a different drive than `output_dir`.

#### `seed_dir`
**Default:** `"./seeds"`

Directory containing seed images (PNG/JPG) used for:
1. **Bootstrap**: The first generation uses a random seed image
2. **Emergency injection**: When severe mode collapse is detected
3. **Periodic refresh**: Adaptive seed injection based on collapse frequency

More diverse seeds = more variety when the system needs to break out of a loop. The images don't need to match your prompts exactly—they serve as starting points that the diffusion process will transform.

#### `gpu_id`
**Default:** `0`

Which CUDA device to use for VAE operations (encoding/decoding latents for interpolation). This is passed to PyTorch as `cuda:{gpu_id}`.

In multi-GPU setups, you might dedicate one GPU to Dream Window while keeping another for gaming or other work. Note that ComfyUI has its own GPU selection (configured in its startup script), so both need to align if you want them on the same or different GPUs.

#### `log_dir` and `log_level`
**Defaults:** `"./logs"`, `"INFO"`

Standard logging configuration. Log levels:
- `DEBUG`: Very verbose, includes per-frame timing, cache decisions, WebSocket messages
- `INFO`: Normal operation, major events, periodic statistics
- `WARNING`: Potential issues that don't stop operation
- `ERROR`: Failures that may require attention

Logs accumulate indefinitely, so for long-running deployments, consider external log rotation.

---

## 2. Generation

The core of Dream Window—controls how images are created and evolved.

### Architecture Overview

Dream Window generates a continuous stream of images through a hybrid pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID GENERATION LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Keyframe N] ──VAE Encode──> [Latent N]                        │
│                                    │                             │
│                              SLERP Interpolation                 │
│                                    │                             │
│                               [Latents...]                       │
│                                    │                             │
│  [Keyframe N+1] ─VAE Encode─> [Latent N+1]                      │
│       │                            │                             │
│       │                       VAE Decode                         │
│       │                            │                             │
│       │                    [Interpolated Frames]                 │
│       │                            │                             │
│       └────── img2img ────────────┘                              │
│              (via ComfyUI)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Keyframes** are generated via diffusion (ComfyUI), taking ~2-3 seconds each. **Interpolation frames** are created by blending latent representations, taking ~50-200ms each. This hybrid approach provides smooth animation while allowing the aesthetic to evolve.

### `use_async_orchestrator`
**Default:** `true`

The async orchestrator runs three workers in parallel:
- **KeyframeWorker**: Submits generation requests to ComfyUI (HTTP I/O bound)
- **InterpolationWorker**: Performs VAE encode/decode (GPU bound)
- **CacheAnalysisWorker**: Computes embeddings and diversity checks (CPU bound)

When `true`, these overlap: while one keyframe generates, the previous keyframe's interpolations are being computed, and frames before that are being analyzed for caching. This roughly doubles effective FPS.

When `false`, operations happen sequentially (legacy mode). Use this for debugging or if you experience stability issues with the parallel system.

### `model`
**Default:** `"sd15"`

Which Stable Diffusion model ComfyUI should use:
- `"sd15"`: Stable Diffusion 1.5 - Most compatible, works on older GPUs (Maxwell Titan X, etc.), well-understood behavior
- `"flux.1-schnell"`: Fast distilled model - Requires modern GPU (RTX 20+), 4-step generation, different aesthetic
- `"sd21-unclip"`: SD 2.1 with unCLIP - Alternative architecture, different training data

The model choice affects which parameter block (`flux` or `sd`) is used for steps, CFG, sampler, and scheduler.

### `resolution`
**Default:** `[512, 256]`

Output dimensions as `[width, height]`. Must be divisible by 8 (VAE requirement).

Resolution has cascading effects:
- **Generation time**: Roughly quadratic with pixel count
- **VRAM usage**: Both ComfyUI and VAE operations scale with resolution
- **Interpolation speed**: VAE encode/decode time scales linearly
- **Disk usage**: Each frame is larger
- **Visual quality**: Higher resolution = more detail

Common choices:
- `[512, 256]`: Fast, cinematic 2:1 aspect ratio (~4 FPS on mid-range GPU)
- `[512, 512]`: Square, balanced
- `[1024, 512]`: High quality, slower (~1-2 FPS)

If you change this, also update the Rainmeter skin dimensions (see comments in config).

### `mode`
**Default:** `"hybrid"`

- `"hybrid"`: Keyframe diffusion + latent interpolation (recommended)
- `"img2img"`: Pure diffusion feedback loop (slower, every frame is a full generation)
- `"interpolate"`: Pure interpolation (no diffusion, needs external keyframe source)

Hybrid mode is the core innovation—it gets the smooth animation of interpolation while allowing the diffusion model to "steer" the evolution through periodic keyframes.

---

### 2.1 Model Parameters (`flux` and `sd`)

These blocks configure diffusion parameters based on the selected model.

#### `steps`
**SD Default:** `10` | **Flux Default:** `4`

Number of denoising iterations per generation. More steps = more refined output but slower.

For Dream Window's drifting aesthetic, fewer steps work well because:
1. We're doing img2img with low denoise, so the model doesn't need to create from scratch
2. Slight imperfections contribute to the dreamy quality
3. Speed matters for maintaining smooth animation

SD 1.5 can go as low as 8 steps with minimal quality loss in this context. Flux is optimized for 4 steps due to its distillation training.

#### `cfg_scale` (Classifier-Free Guidance)
**SD Default:** `6.0` | **Flux Default:** `1.0`

How strongly the model follows the prompt vs. exploring freely.

- Higher CFG (7-12): Stronger prompt adherence, more "on-topic" but can look artificial
- Lower CFG (3-6): More natural, dreamier, may drift from prompt
- Very low CFG (1-2): Almost pure generation, prompt is a gentle suggestion

For evolving dreamscapes, moderate CFG (5-7) works well—enough guidance to maintain theme, enough freedom to evolve organically.

Flux uses low CFG by design due to its training methodology.

#### `sampler`
**SD Default:** `"euler"` | **Flux Default:** `"euler"`

The algorithm for stepping through the denoising process.

- `euler`: Fast, deterministic, good baseline
- `euler_ancestral` (`euler_a`): Adds noise at each step, more variety but less predictable
- `dpm++_2m`: Higher quality, slower
- `ddim`: Deterministic, good for reproducibility

For continuous generation, `euler` or `euler_a` are good choices. Ancestral samplers add natural variation.

#### `scheduler`
**SD Default:** `"karras"` | **Flux Default:** `"simple"`

How noise levels decrease across steps.

- `normal`: Linear decrease
- `karras`: Emphasizes early denoising steps, often better detail
- `simple`: Minimal scheduling (Flux default)

Karras typically produces better results with SD 1.5 at low step counts.

---

### 2.2 Hybrid Mode

The heart of Dream Window's animation system.

#### `interpolation_frames`
**Default:** `10`

How many frames to generate between each keyframe pair using VAE latent interpolation.

The math: If you have 10 interpolation frames and a keyframe takes 2.5 seconds to generate, you get 10 frames over 2.5 seconds ≈ 4 FPS of content produced.

Higher values:
- Smoother transitions between keyframes
- More frames per keyframe cycle
- Better "coverage" of the latent space between points

Lower values:
- Faster aesthetic evolution (keyframes happen more often)
- Less smooth, more "jumpy" between keyframes
- Less computational overhead

#### `target_interpolation_fps`
**Default:** `3.5`

The display frame rate for interpolated frames.

This controls playback speed, not generation speed. If generation produces frames faster than this, they buffer. If slower, the buffer drains.

The ideal value depends on:
- Your GPU's VAE decode speed
- Resolution (lower res = faster decode)
- `interpolation_resolution_divisor` setting

If playback outpaces generation, you'll see the buffer slowly drain and eventually stutter. If generation outpaces playback, the buffer grows (good, up to a point).

#### `keyframe_denoise`
**Default:** `0.2`

The img2img denoise strength when generating new keyframes.

This is the "drift rate" knob:
- `0.1-0.2`: Very slow evolution, strong preservation of structure
- `0.3-0.4`: Balanced, noticeable change per keyframe
- `0.5-0.6`: Rapid mutation, significant changes
- `0.7+`: Major changes, may lose coherence

Lower denoise means each keyframe closely resembles its source (the previous keyframe), creating smooth aesthetic evolution. Higher denoise allows more dramatic changes but risks "breaking" the visual style.

#### `interpolation_decoder`
**Default:** `"vae"`

Which decoder to use for interpolation frames:

- `"vae"`: Full SD 1.5 VAE decoder (~230ms/frame at 1024x512, maximum quality)
- `"taesd"`: Tiny AutoEncoder (~25ms/frame, ~9x faster, slight softness)

**TAESD** (Tiny AutoEncoder for Stable Diffusion) is a distilled ~10MB decoder that provides dramatic speedup at the cost of slightly reduced fine detail. For interpolation frames—which are transitional and displayed briefly (~0.3s each)—the quality difference is essentially imperceptible at playback speed.

**When to use TAESD:**
- Cloud/RunPod deployment (reduces costs, faster iteration)
- Streaming to VPS (more frames per second)
- Lower-end GPUs (reduces decode bottleneck)
- When buffer drains faster than generation fills it

**When to use full VAE:**
- Quality-critical local display (Rainmeter wallpaper)
- When you have GPU headroom and quality matters most
- If you notice artifacts in interpolation frames

TAESD is loaded from HuggingFace (`madebyollin/taesd`) on first use and coexists with the full VAE (only ~20MB additional VRAM).

#### `interpolation_resolution_divisor`
**Default:** `1`

Divide the resolution by this factor for interpolation, then upscale the result.

- `1`: Full resolution interpolation (highest quality, slowest)
- `1.5`: 2/3 resolution (good balance)
- `2`: Half resolution (significant speedup, slight softness)

This is a powerful performance optimization. VAE operations scale with pixel count, so half resolution is roughly 4x faster. The upscaling (bilinear/bicubic) is very fast and the quality loss is often imperceptible in motion.

**Note:** `interpolation_decoder: "taesd"` and `interpolation_resolution_divisor: 2` can be combined for maximum performance (~36x speedup total), though this may introduce noticeable softness.

#### `interpolation_upscale_method`
**Default:** `"bilinear"`

How to upscale after low-resolution interpolation.

- `bilinear`: Fast, smooth, slight softening
- `bicubic`: Slightly sharper, slightly slower
- `nearest`: Pixelated (not recommended unless you want that aesthetic)

#### `interpolation_downsample_method`
**Default:** `"bicubic"`

How to downsample before encoding (when using resolution divisor > 1).

- `bilinear`: Fastest
- `bicubic`: Better preservation of detail
- `lanczos`: Highest quality, ~10ms overhead

The downsample method matters more than upscale because it determines what information is preserved in the latent space.

---

### 2.3 Cache System

The cache system prevents **mode collapse**—the tendency of img2img feedback loops to converge toward a single aesthetic (often a specific color palette or structural pattern).

#### How Mode Collapse Happens

```
Frame 1 → slight blue bias → Frame 2 → more blue → Frame 3 → very blue → ...
                                                                          │
                     Eventually: All frames are nearly identical ←────────┘
```

Each generation slightly amplifies tendencies in the input. Without intervention, the system converges to a fixed point.

#### The Prevention Strategy

Dream Window uses a **dual-metric watchdog** system:

1. **ColorHist**: Tracks color palette distribution (96-dimensional histogram)
2. **pHash-8**: Tracks structural/compositional similarity (64-bit perceptual hash)

These run independently with **OR logic** for detection—if EITHER metric shows convergence, intervention triggers. This catches both color collapse (everything turning magenta) and structural collapse (same composition repeating).

#### Cache Parameters

##### `max_size`
**Default:** `50`

Maximum cached images. When exceeded, oldest entries are evicted (LRU).

Larger cache = more variety available for injection. But diminishing returns past ~100, and more disk usage.

##### `similarity_method`
**Default:** `"dual_metric"`

Documents that we're using ColorHist + pHash. This is informational; the actual method is hardcoded.

##### `population_mode`
**Default:** `"selective"`

How frames enter the cache:
- `"selective"`: Only cache frames that add diversity (recommended)
- `"all"`: Cache everything (cache fills with similar frames)

Selective caching is crucial—if you cache everything during a collapse, you fill the cache with collapsed frames, making injection useless.

##### `cache_interpolations`
**Default:** `true`

Whether to consider interpolation midpoints for caching.

Interpolations can capture transitional states not present in keyframes, adding diversity. But they may also be blurry or transitional.

##### `cache_diversity_logic`
**Default:** `"all"`

For cache acceptance:
- `"all"` (AND): Must be diverse in BOTH color AND structure
- `"any"` (OR): Diverse in EITHER metric is enough

Using `"all"` ensures the cache contains only truly diverse frames, not frames that are structurally identical but different colors (or vice versa).

---

#### Dual-Metric Settings

##### Color Histogram (`color_histogram.*`)

###### `bins_per_channel`
**Default:** `32`

Histogram granularity. 32 bins × 3 channels (HSV) = 96-dimensional vector.

More bins = more sensitive to subtle color shifts. 32 is a good balance between sensitivity and noise tolerance.

###### `diversity_threshold`
**Default:** `1.92`

A frame is cached only if its average similarity to existing cache entries is BELOW this threshold.

The ColorHist similarity metric ranges roughly 0.8-2.3 where:
- ~0.8-1.2: Very similar colors
- ~1.2-1.8: Related palettes
- ~1.8-2.3: Different color schemes

At 1.92, we only cache frames that are meaningfully different in color from what's already cached.

###### `dissimilarity_range`
**Default:** `[1.18, 2.05]`

The "Goldilocks zone" for injection candidates. When injecting to break collapse, we want frames that are:
- Different enough to break the pattern (above 1.18)
- Not so different they're jarring (below 2.05)

###### `convergence_threshold`
**Default:** `0.15`

The similarity delta that triggers scaled injection probability.

The collapse detector compares recent frames (last 20) to earlier frames (first 20 in history). If the average similarity has INCREASED by more than 0.15 (i.e., frames are becoming more alike), it triggers "converging" status.

###### `force_cache_threshold`
**Default:** `0.30`

The delta that triggers mandatory cache injection. If similarity has increased by 0.30+, the system is in severe collapse and must intervene.

---

##### Perceptual Hash (`phash.*`)

###### `hash_size`
**Default:** `8`

DCT hash size (8×8 = 64 bits). Larger = more structural sensitivity.

pHash works by:
1. Resize image to small square
2. Apply DCT (like JPEG compression)
3. Compare DCT coefficients to median

This captures "structure" independent of color—composition, shapes, contrast patterns.

###### `diversity_threshold`
**Default:** `0.80`

Similar to color, but for structural similarity. The pHash metric is already 0-1 normalized:
- ~0.4-0.6: Very different structure
- ~0.6-0.8: Some similarity
- ~0.8-1.0: Structurally similar/identical

At 0.80, we cache frames with meaningfully different composition.

###### `dissimilarity_range`
**Default:** `[0.68, 0.92]`

Injection candidate range for structure. We want frames that break the current structural pattern without being completely unrelated.

###### `convergence_threshold` and `force_cache_threshold`
**Defaults:** `0.08`, `0.15`

Structural deltas for convergence detection. These are smaller than color thresholds because pHash operates on a 0-1 scale.

---

#### Injection Logic

##### `injection_logic`
**Default:** `"any"`

For collapse DETECTION:
- `"any"` (OR): Trigger if EITHER color OR structure shows convergence
- `"all"` (AND): Only trigger if BOTH show convergence

Using `"any"` makes the watchdog more sensitive—it catches color-only collapse (common with certain prompts) and structure-only collapse (less common but possible).

**Important**: This is the OPPOSITE of `cache_diversity_logic` by design:
- Cache acceptance: Strict (`"all"`) → Only truly diverse frames enter cache
- Collapse detection: Sensitive (`"any"`) → Catch any type of collapse early

---

#### Injection Behavior

##### `injection_mode`
**Default:** `"dissimilar"`

What to inject when collapse is detected:
- `"dissimilar"`: Inject frames that are DIFFERENT from current state (breaks the pattern)
- `"similar"`: Would inject similar frames (not useful for collapse prevention)

##### `injection_probability`
**Default:** `0.15`

Base probability of random cache injection per keyframe, independent of collapse detection. This adds variety even when not actively collapsing.

##### `blend_weight`
**Default:** `0.75`

When injecting, how much of the cached frame vs. current frame to use in the VAE latent blend:
- `0.75`: Strong influence from cached frame (recommended for breaking collapse)
- `0.50`: Equal blend
- `0.25`: Subtle influence

Higher blend = more dramatic intervention. Too low and the injection doesn't break the pattern.

##### `injection_cooldown`
**Default:** `2`

Minimum keyframes between cache injections. Prevents rapid-fire injections that could destabilize the aesthetic.

##### `seed_injection_cooldown`
**Default:** `2`

Same, but for seed image injections (more dramatic intervention).

##### `embedding_history_reset`
**Default:** `"partial"`

What to do with the collapse detector's history after injection:
- `"none"`: Keep all history (injection might immediately re-trigger)
- `"partial"`: Keep recent frames, discard old (recommended)
- `"full"`: Clear everything (fresh start)

Partial reset keeps context while breaking the convergence signal.

##### `embedding_history_keep_recent`
**Default:** `10`

If using partial reset, how many recent frames to keep. More = more context preserved, but might preserve collapse signal.

---

#### Collapse Detection

##### `collapse_detection`
**Default:** `true`

Master toggle for the watchdog system.

##### `warmup_keyframes`
**Default:** `10`

No injections until this many keyframes have been generated. Allows the aesthetic to establish naturally before intervention.

Set this to at least match the collapse detection window size (which needs ~40 frames of history to compare early vs. recent).

##### `convergence_mode`
**Default:** `"absolute"`

How to measure convergence:
- `"absolute"`: Compare raw delta values against thresholds
- `"percentage"`: Compare percentage increase (not commonly used)

##### `log_convergence_stats`
**Default:** `true`

Log detailed convergence metrics for tuning. The logs show delta values so you can calibrate thresholds for your specific prompts and aesthetic.

---

#### Adaptive Seed Injection

Seeds are "emergency interventions"—when cache injection isn't enough, inject a completely fresh image from the seed directory.

##### `seed_injection_floor` and `seed_injection_max`
**Defaults:** `0.02`, `0.15`

Seed injection probability scales with collapse frequency:
- Starts at 2% (floor)
- Increases as cache injections accumulate
- Caps at 15% (max)

##### `seed_injection_ramp`
**Default:** `50`

After 50 cache injections, seed probability reaches maximum. This means if the system is frequently cache-injecting (sign of persistent issues), it escalates to seeds.

##### `seed_injection_boost_threshold` and `seed_injection_boost_probability`
**Defaults:** `10`, `0.20`

During bootstrap (cache has fewer than 10 frames), seed injection is boosted to 20%. This helps populate the cache with diverse frames early.

##### `force_seed_injection_frequency`
**Default:** `0.30`

If more than 30% of recent keyframes required cache injection, force a seed injection. This catches situations where cache injection alone isn't breaking the collapse.

##### `blend_seed_injection`
**Default:** `true`

Whether to blend seeds with the current frame or use them directly. Blending creates smoother transitions.

##### `seed_blend_weight`
**Default:** `0.85`

How much of the seed to use. Higher = more dramatic intervention.

---

#### Advanced Monitoring (Future)

The `advanced_monitoring` block contains settings for Phase 2 enhancements (not yet implemented):
- Redundancy-based cache eviction
- Continuous diversity matrix
- Cluster detection
- Adaptive threshold tuning

These are documented for future reference but currently have no effect.

---

## 3. Display

Controls frame output and buffer management.

### Buffer System

Dream Window uses a buffer to ensure smooth playback:

```
Generation → Buffer → Display
   (async)    (FIFO)   (fixed FPS)
```

The buffer absorbs variations in generation time, preventing stutters when a keyframe takes longer than usual.

### `buffer_target_seconds`
**Default:** `30.0`

Target buffer size (the "crumple zone"). The system tries to maintain this much content ahead of display.

Higher values:
- More resilience to generation hiccups
- Longer initial startup wait
- More frames stored in memory/disk

Lower values:
- Faster startup
- Less resilience to slowdowns
- Risk of buffer underrun during heavy keyframes

### `min_buffer_seconds`
**Default:** `10.0`

Minimum buffer before starting playback. Can be lower than target for faster startup while still having some cushion.

### `cleanup_displayed_frames`
**Default:** `true`

Delete frames immediately after display. **Highly recommended for 24/7 operation.**

Without cleanup, frames accumulate indefinitely:
- 10 FPS × 3600 seconds/hour = 36,000 frames/hour
- At ~200KB each = 7.2 GB/hour

With cleanup, only buffered frames + current_frame.png exist at any time.

### `max_output_frames`
**Default:** `100`

Legacy cleanup mechanism—keep only last N frames. Deprecated if `cleanup_displayed_frames` is true.

---

## 4. Prompts

Controls the text guidance for generation.

### Theme Pairs

Each theme is a pair of positive and negative prompts that work together:

```yaml
- positive: "ethereal digital angel, flowing lines, monochrome..."
  negative: "colors, red, blue, green, low quality..."
```

The negative prompt is just as important—it steers the model away from unwanted elements.

### `rotation_interval`
**Default:** `20`

Keyframes before switching to the next theme. This creates variety over time while maintaining coherence within each "phase."

At 20 keyframes with 10 interpolations each and 3.5 FPS, one theme lasts roughly:
20 × 10 / 3.5 ≈ 57 seconds

### Modifiers

##### `modifiers.enabled`
**Default:** `true`

Allow dynamic prompt modifications.

##### `modifiers.time_based`
**Default:** `true`

Append time-of-day atmosphere modifiers:
- 5-8 AM: "dawn light, morning atmosphere"
- 8-12 PM: "bright daylight, clear atmosphere"
- 12-5 PM: "afternoon light, warm tones"
- 5-8 PM: "twilight, golden hour lighting"
- 8-11 PM: "evening atmosphere, deep shadows"
- 11 PM-5 AM: "midnight atmosphere, deep darkness"

This creates natural variation as the day progresses.

##### `modifiers.system_based`
**Default:** `false`

Future feature: Modify prompts based on system state (CPU load, etc.).

---

## 5. Game Detection

Automatically pause generation when games are running to prevent VRAM conflicts.

### How It Works

The detector periodically scans running processes for known game executables. When detected, generation can pause and release VRAM, resuming when the game closes.

### `enabled`
**Default:** `true`

Master toggle.

### `method`
**Default:** `"process"`

Detection method:
- `"process"`: Check process names against known list (reliable)
- `"fullscreen"`: Detect fullscreen applications (Windows only, less reliable)
- `"gpu_load"`: Monitor GPU utilization (not implemented)

### `check_interval`
**Default:** `5.0`

Seconds between checks. Lower = more responsive, higher = less CPU overhead.

### `known_games`
List of game executable names (case-insensitive):

```yaml
known_games:
  - "eldenring.exe"
  - "cyberpunk2077.exe"
```

Add your games here. The detector matches substrings, so "elden" would match "eldenring.exe".

### `gpu_threshold`
**Default:** `80`

For future GPU load method: pause if GPU utilization exceeds this percentage.

---

## 6. Performance

System-wide performance tuning.

### `max_queue_size`
**Default:** `60`

Maximum frames in generation queue. Provides backpressure—if the queue fills, new submissions wait.

Higher values use more memory but provide more buffering capacity during generation bursts.

### `generation_timeout`
**Default:** `180`

Seconds before considering a generation stuck. If ComfyUI doesn't respond within this time, the system assumes failure and may retry.

For older/slower GPUs (Maxwell Titan X, etc.), high-resolution generations can legitimately take 2+ minutes. The default of 180 seconds accommodates this.

For modern GPUs, you can lower this to ~60 seconds for faster failure recovery.

### `enable_torch_compile`
**Default:** `false`

PyTorch 2.0's torch.compile() optimization for VAE operations.

**Requirements:**
- CUDA Capability 7.0+ (Volta architecture or newer)
- Triton (Linux only)

Maxwell (5.x) and Pascal (6.x) GPUs cannot use this. Attempting to enable it will fail gracefully and disable itself.

When available, provides ~20-30% speedup for VAE encode/decode.

### `async_file_operations`
**Default:** `true`

Use non-blocking file writes. Recommended for all cases.

---

## 7. Daemon

Process management for running as a persistent service.

### Overview

The daemon manages the full stack:
1. Starts ComfyUI backend
2. Waits for health check
3. Starts Dream Window controller
4. Monitors both processes
5. Auto-restarts on crashes
6. Handles graceful shutdown

### ComfyUI Settings

##### `comfyui.startup_script`
Path to the script that launches ComfyUI. This is typically a `.bat` file (Windows) or shell script that activates ComfyUI's virtual environment and runs the server.

##### `comfyui.startup_timeout`
**Default:** `300`

Maximum seconds to wait for ComfyUI to become responsive. Large models can take several minutes to load.

##### `comfyui.health_check_url`
**Default:** `"http://127.0.0.1:8188/system_stats"`

URL to poll to verify ComfyUI is ready.

##### `comfyui.health_check_interval`
**Default:** `2`

Seconds between health check attempts during startup.

### Controller Settings

##### `controller.python_executable`
Which Python to use for running the controller. Options:
- `"auto"`: Same Python as daemon
- `".venv/Scripts/python.exe"`: Explicit venv (Windows)
- `".venv/bin/python"`: Explicit venv (Linux/Mac)

##### `controller.main_script`
**Default:** `"backend/main.py"`

Entry point for the Dream Window controller.

### Auto-Restart

##### `auto_restart.comfyui` and `auto_restart.controller`
**Defaults:** `true`

Whether to automatically restart crashed processes.

##### `auto_restart.max_restarts`
**Default:** `5`

Maximum restarts per hour before giving up. Prevents infinite restart loops from persistent failures.

##### `auto_restart.restart_delay`
**Default:** `10`

Seconds to wait before restart attempt. Allows resources to be released.

### Control Interface

##### `control_file`
**Default:** `"output/daemon_control.txt"`

File for external control commands. Write "pause", "resume", or "shutdown" to this file.

##### `control_check_interval`
**Default:** `2`

How often to check the control file.

### Shutdown

##### `shutdown.comfyui_grace_period`
**Default:** `30`

Seconds to wait for ComfyUI to exit cleanly before force-killing.

##### `shutdown.controller_grace_period`
**Default:** `10`

Same for the controller.

##### `shutdown.force_kill_after_timeout`
**Default:** `true`

Force-kill processes that don't exit within grace period.

### Logging

##### `daemon.log_file`
**Default:** `"logs/daemon.log"`

Daemon-specific log file (separate from controller logs).

##### `daemon.log_level`
**Default:** `"INFO"`

Daemon log verbosity.

---

## 8. Cloud

Optional web deployment—push frames to a VPS for browser viewing.

### Overview

Cloud mode is **additive**—local Rainmeter output continues to work. Additionally, frames are pushed via WebSocket to a VPS running the aethera dreams module.

### `enabled`
**Default:** `false`

Master toggle. When true, the system connects to the VPS and pushes frames.

### `vps_websocket_url`
**Default:** `"wss://aetherawi.red/ws/gpu"`

WebSocket endpoint for frame pushing. This should be the `/ws/gpu` endpoint on your aethera deployment.

### `auth_token`
Authentication token. Can be set here or via `DREAM_GEN_AUTH_TOKEN` environment variable.

### Frame Push Settings

##### `frame_push.enabled`
**Default:** `true`

Whether to push frames (can disable while keeping connection for control).

##### `frame_push.format`
**Default:** `"webp"`

Image format:
- `"webp"`: Recommended, ~40-70KB per 1024×512 frame at quality 85
- `"png"`: Lossless but larger (~200-400KB)

##### `frame_push.quality`
**Default:** `85`

WebP quality (1-100). 85 is a good balance of size and quality for AI art.

##### `frame_push.include_interpolations`
**Default:** `true`

Push all frames or just keyframes. Pushing only keyframes reduces bandwidth but creates choppy playback.

### State Sync

##### `state_sync.enabled`
**Default:** `true`

Push state snapshots to VPS for resume capability after GPU restart.

##### `state_sync.interval_keyframes`
**Default:** `10`

Push state every N keyframes.

##### `state_sync.push_on_shutdown`
**Default:** `true`

Always push final state on graceful shutdown.

### `resolution_override`
**Default:** `[1024, 512]`

Override resolution when cloud mode is enabled. Typically higher than local since web viewing benefits from larger images and cloud GPUs are often more powerful.

### Connection Settings

##### `connection.reconnect_delay`
**Default:** `1.0`

Initial reconnect delay after disconnection.

##### `connection.max_reconnect_delay`
**Default:** `60.0`

Maximum delay (exponential backoff cap).

##### `connection.heartbeat_interval`
**Default:** `30.0`

How often to send keepalive messages.

---

## Appendix: Quick Reference Card

| Want to... | Adjust... |
|------------|-----------|
| Faster animation | ↓ `interpolation_frames`, ↑ `keyframe_denoise` |
| Smoother animation | ↑ `interpolation_frames`, ↓ `keyframe_denoise` |
| More variety | ↑ `injection_probability`, ↓ diversity thresholds |
| More stability | ↓ `injection_probability`, ↑ `warmup_keyframes` |
| Faster startup | ↓ `min_buffer_seconds` |
| More resilient buffer | ↑ `buffer_target_seconds` |
| Better collapse detection | ↓ convergence thresholds |
| Less aggressive intervention | ↑ cooldowns, ↑ convergence thresholds |
| Higher quality | ↑ `resolution`, ↓ `interpolation_resolution_divisor`, `interpolation_decoder: "vae"` |
| Better performance | ↓ `resolution`, ↑ `interpolation_resolution_divisor`, `interpolation_decoder: "taesd"` |
| Faster interpolation decode | `interpolation_decoder: "taesd"` (~9x speedup) |

