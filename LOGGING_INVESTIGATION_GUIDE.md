# Dream Gen Logging Investigation Guide

> **Purpose:** This document maps the logging infrastructure across dream_gen to enable efficient debugging and log analysis. Each section covers files with logging, organized by functional category, with detailed log tags, meanings, and debugging strategies.

---

## Quick Debug Cheatsheet

**Jump to common scenarios:**

| Symptom | Where to Look | Key Log Patterns |
|---------|---------------|------------------|
| **Nothing generating** | `daemon.log` | `ComfyUI process died`, `Controller died immediately` |
| **Frames look the same** | `dream_controller.log` | `[COLLAPSE_METRICS]` with high similarity, low mutation count in `[INTERVENTION_STATS]` |
| **Generation stalls** | `dream_controller.log` | `Buffer: 0.0s`, no new `[OK] Keyframe N completed` |
| **Visual glitches/artifacts** | `dream_controller.log` | `[FAIL] Keyframe`, VAE errors, interpolation failures |
| **Performance drops** | `dream_controller.log` | `VAE Lock Contention`, `Interpolation queue depth high` |
| **Memory issues** | `daemon.log` | Restart loops, `exit code` patterns |
| **Cloud disconnects** | `dream_controller.log` | `Failed to connect to VPS`, `Reconnection attempt` |
| **Cache not helping** | `dream_controller.log` | `No dissimilar frames`, `Cache is empty`, no `[BLEND]` logs |
| **Collapse not detected** | `dream_controller.log` | No `[CONVERGING]` or `[COLLAPSE]`, check `[CALIBRATION]` deltas |
| **Keyframe retries** | `dream_controller.log` | `attempt 2/3`, `Waiting Xs before retry`, `ComfyUI recovery` |
| **Interpolation slow** | `dream_controller.log` | `[TIMING]` with high `Decode all` time, `Batch decode failed` |
| **Template switch slow** | `dream_controller.log` | `[FRESH] Waiting for frame regeneration`, `Timeout waiting` |
| **Same template repeating** | `dream_controller.log` | Same template in `[FRESH] Consumed`, pool size always small |
| **VAE slow/broken** | `dream_controller.log` | `[PERF] VAE decode:` with high ms, `Compiled VAE decoder failed` |
| **Startup hangs** | `daemon.log` | Stuck at `[FRESH] Generating N/M`, no `Buffer populated` |
| **Cloud: watchdog restart** | `dream_controller.log` | `WATCHDOG: No activity`, `WATCHDOG RESTART` |
| **Cloud: ComfyUI discovery** | `dream_controller.log` | `ComfyUI not registered`, `Discovery failed` |
| **Cloud: slow frame push** | `dream_controller.log` | `[PERF] Slow frame push`, `[PERF] Slow WS send` |
| **Cloud: state not saving** | `dream_controller.log` | `Cannot push state`, `Failed to push state` |
| **Prompts: no mutations** | `dream_controller.log` | No `[MUTATE]` logs, high `frames_since_mutation` |
| **Prompts: too jarring** | `dream_controller.log` | Frequent `[MUTATE]`, `random fallback` in `[SIMILARITY]` |
| **Prompts: always BEND** | `dream_controller.log` | `BEND mode active` persists, no `BEND → DRIFT` |

**Log file locations:**
- Daemon: `logs/daemon.log`
- Controller: `logs/dream_controller.log` (rotated, up to 5MB × 4 files)

**Search commands:**
```bash
# Find collapse-related logs
grep -E "\[COLLAPSE|INTERVENTION_STATS\]" logs/dream_controller.log

# Find failures and errors
grep -E "\[FAIL\]|ERROR|error:|Failed" logs/dream_controller.log

# Find injection events
grep -E "Injection triggered|TEMPLATE_SWITCH|EMERGENCY" logs/dream_controller.log

# Find daemon restarts
grep -E "Restarting|died|exit code" logs/daemon.log

# Tail live generation
tail -f logs/dream_controller.log | grep -E "\[OK\]|Buffer:|COLLAPSE"

# === CACHE SYSTEM ===
# Find cache injection decisions
grep -E "\[DISSIMILAR\]|\[BLEND\]|\[DIRECT_COPY\]" logs/dream_controller.log

# Find cache diversity gating
grep -E "Frame is diverse|Frame is redundant" logs/dream_controller.log

# Find collapse detection
grep -E "\[CONVERGING\]|\[COLLAPSE\]|\[WARMUP" logs/dream_controller.log

# Find template switches
grep -E "Switching template|Template switch complete|TEMPLATE_SWITCH" logs/dream_controller.log

# === WORKER SYSTEM ===
# Find keyframe generation timing
grep -E "\[OK\] Keyframe|attempt [0-9]/[0-9]" logs/dream_controller.log

# Find interpolation performance
grep -E "\[TIMING\] Interpolation|Decode all:" logs/dream_controller.log

# Find worker queue issues
grep -E "queue depth|queue near capacity" logs/dream_controller.log

# Find retry patterns
grep -E "retry|Waiting.*before retry|ComfyUI recovery" logs/dream_controller.log

# === FRESH FRAME BUFFER ===
# Find fresh buffer population (startup)
grep -E "\[FRESH\] Populating|\[FRESH\] Generated|\[FRESH\] Buffer populated" logs/dream_controller.log

# Find template consumption and pool state
grep -E "\[FRESH\] Consumed|pool:.*remaining|resetting selection pool" logs/dream_controller.log

# Find fresh buffer issues
grep -E "\[FRESH\].*failed|\[FRESH\].*Waiting|\[FRESH\].*Timeout" logs/dream_controller.log

# Find regeneration activity
grep -E "\[FRESH\] Regenerat" logs/dream_controller.log

# === VAE/INTERPOLATION SYSTEM ===
# Find VAE performance issues
grep -E "\[PERF\] VAE decode|\[PERF\] Batch decode" logs/dream_controller.log

# Find VAE compilation/fallback events
grep -E "torch.compile|Compiled VAE|Triton|Reloading VAE" logs/dream_controller.log

# Find resolution and format issues
grep -E "Force resizing|not divisible by 8|GPU upscaled" logs/dream_controller.log

# Find VRAM monitoring
grep -E "VRAM.*allocated|VRAM.*reserved|empty_cache" logs/dream_controller.log

# === CLOUD INFRASTRUCTURE ===
# Find watchdog activity
grep -E "WATCHDOG|No activity for|restart" logs/dream_controller.log

# Find VPS connection events
grep -E "Connected to VPS|Reconnection|Disconnected|Failed to connect" logs/dream_controller.log

# Find frame push performance
grep -E "\[PERF\] Slow frame push|\[PERF\] Slow WS send|Pushed frame" logs/dream_controller.log

# Find state sync events
grep -E "Pushed state snapshot|Failed to push state|State restored" logs/dream_controller.log

# Find ComfyUI discovery (pod mode)
grep -E "Discovering ComfyUI|ComfyUI discovered|ComfyUI not registered" logs/dream_controller.log

# === PROMPT SYSTEM ===
# Find mutation events
grep -E "\[MUTATE\]|\[FORCE_MUTATE\]" logs/dream_controller.log

# Find DRIFT/BEND state transitions
grep -E "DRIFT.*BEND|BEND.*DRIFT|\[STATE\]" logs/dream_controller.log

# Find similarity-guided selection
grep -E "\[SIMILARITY\]|random fallback" logs/dream_controller.log

# Find mutation probability checks (DEBUG level)
grep -E "\[MUTATION_CHECK\]" logs/dream_controller.log

# Find template switch events
grep -E "\[TEMPLATE_SWITCH\]" logs/dream_controller.log
```

---

## Log Format & Correlation

### Log Formats

**Daemon (`daemon.log`):**
```
2026-01-24 12:34:56 - __main__ - INFO - Message here
│                     │           │      └── The log message
│                     │           └── Level (DEBUG/INFO/WARNING/ERROR)
│                     └── Logger name (__main__ for daemon.py)
└── Full timestamp (YYYY-MM-DD HH:MM:SS)
```

**Controller (`dream_controller.log`) - Console:**
```
12:34:56 - INFO - Message here
│          │      └── The log message  
│          └── Level
└── Short timestamp (HH:MM:SS only)
```

**Controller (`dream_controller.log`) - File:**
```
2026-01-24 12:34:56 - backend.core.async_orchestrator - INFO - Message here
│                     │                                  │      └── Message
│                     │                                  └── Level
│                     └── Full module path (logger name)
└── Full timestamp
```

### Correlating Daemon ↔ Controller Logs

The daemon and controller are **separate processes** with separate log files. Use timestamps to correlate:

```bash
# See both logs side by side (sorted by time)
cat logs/daemon.log logs/dream_controller.log | sort | less

# Find what happened around a specific time
grep "2026-01-24 12:34" logs/*.log
```

**Lifecycle correlation:**
1. Daemon starts → `DREAM WINDOW DAEMON STARTING`
2. ComfyUI starts → `STARTING COMFYUI BACKEND` (daemon log)
3. ComfyUI ready → `ComfyUI FULLY READY` (daemon log)
4. Controller starts → `STARTING DREAMCONTROLLER` (daemon log)
5. Controller initializes → `DREAM WINDOW CONTROLLER INITIALIZING` (controller log)
6. Generation begins → `STARTING ASYNC GENERATION ORCHESTRATOR` (controller log)

---

## Overview

Dream Gen uses Python's standard `logging` module with `__name__`-based loggers throughout. The daemon (`daemon.py`) configures the root logger, and all modules inherit this configuration.

**Total files with logging:** 39  
**Heaviest logging files:** `daemon.py` (158), `async_orchestrator.py` (157), `dream_controller.py` (147)

---

## 1. Entry Points & Daemon Control

The top-level orchestration and lifecycle management.

| File | Log Count | Description |
|------|-----------|-------------|
| `daemon.py` | 158 | Main daemon entry point, ComfyUI lifecycle, signal handling |

**Key logging areas:**
- Daemon startup/shutdown sequences
- ComfyUI process management
- PID file handling
- Orphaned process cleanup

---

## 2. Core Generation Pipeline

The heart of frame generation and orchestration.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/core/async_orchestrator.py` | 157 | Main async generation loop, worker coordination |
| `backend/core/dream_controller.py` | 147 | High-level dream generation control, logging configuration |
| `backend/core/comfyui_api.py` | 71 | ComfyUI REST API interactions, workflow submission |
| `backend/core/generation_coordinator.py` | 70 | Sequence management, frame ordering |
| `backend/core/generator.py` | 62 | Core image generation logic |
| `backend/core/display_selector.py` | 56 | Frame selection for display output |
| `backend/core/frame_buffer.py` | 26 | Frame buffering and queue management |
| `backend/core/shared_resources.py` | 8 | Shared state and resource management |
| `backend/core/workflow_builder.py` | 6 | ComfyUI workflow JSON construction |

**Key logging areas:**
- Generation cycle progression
- Worker task submission/completion
- Frame sequence tracking
- ComfyUI API calls and responses
- Display frame selection logic

---

## 3. Worker Subsystem

Background workers handling specific generation tasks.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/core/workers/interpolation_worker.py` | 42 | Latent space interpolation between keyframes |
| `backend/core/workers/cache_worker.py` | 26 | Cache injection and management |
| `backend/core/workers/keyframe_worker.py` | 21 | Keyframe generation requests |

**Key logging areas:**
- Worker task queuing
- Interpolation progress
- Cache injection decisions
- Keyframe generation timing

---

## 4. Cache System

Frame caching, similarity detection, and injection strategies.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/cache/manager.py` | 48 | Cache storage, LRU eviction, persistence |
| `backend/cache/injection_strategy.py` | 26 | When/how to inject cached frames |
| `backend/cache/collapse_detector.py` | 15 | Detects visual collapse/degradation |
| `backend/cache/dual_similarity.py` | 11 | Perceptual + latent similarity scoring |

**Key logging areas:**
- Cache hits/misses
- Eviction decisions
- Collapse detection triggers
- Similarity threshold evaluations
- Archive operations

---

## 5. Cloud Infrastructure

Remote deployment and streaming to the blog.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/cloud/runpod_handler.py` | 117 | RunPod serverless handler, request processing |
| `backend/cloud/websocket_client.py` | 19 | WebSocket connection to aetherawi.red |
| `backend/cloud/state_sync.py` | 10 | State synchronization with remote |
| `backend/cloud/frame_pusher.py` | 9 | Frame upload/streaming |

**Key logging areas:**
- RunPod request handling
- WebSocket connection state
- Frame streaming progress
- Remote state synchronization

---

## 6. Interpolation System

Smooth transitions between keyframes using VAE latent space operations.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/interpolation/latent_encoder.py` | 44 | VAE encoding/decoding, latent operations |

**Key logging areas:**
- VAE model loading and torch.compile optimization
- Encode/decode timing with `[PERF]` tags
- Resolution handling (force resize, divisibility checks)
- VRAM management and GPU upscaling
- Triton fallback handling for compiled decoder

→ **See [Priority 5: VAE/Interpolation System](#priority-5-vaeinterpolation-system-deep-investigation) for detailed log patterns**

---

## 7. Fresh Frame Buffer

Pre-generation buffer for immediate frame availability.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/fresh/buffer.py` | 33 | Fresh frame pre-generation and buffering |

**Key logging areas:**
- Buffer population at startup (`[FRESH] Populating buffer`)
- Per-template frame generation and consumption
- Selection pool management (random-excluding-recent)
- Background regeneration after consumption

→ **See [Priority 4: Fresh Frame Buffer](#priority-4-fresh-frame-buffer-deep-investigation) for detailed log patterns**

---

## 8. Prompt System

Prompt generation and template management.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/prompts/combinatorial.py` | 35 | Combinatorial prompt generation, component selection |

**Key logging areas:**
- Template selection
- Component combination
- Prompt construction
- Denoising state machine (DRIFT/BEND)

---

## 9. Utility Modules

Supporting functionality used across the system.

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/utils/vram_profiler.py` | 43 | GPU VRAM monitoring and profiling |
| `backend/utils/perf_stats.py` | 36 | Performance statistics collection |
| `backend/utils/file_ops.py` | 14 | File I/O operations |
| `backend/utils/game_detector.py` | 11 | Game/fullscreen detection for pausing |
| `backend/utils/status_writer.py` | 9 | Status file writing for external consumers |
| `backend/utils/prompt_manager.py` | 6 | Prompt file loading |
| `backend/utils/phash_encoder.py` | 5 | Perceptual hash computation |
| `backend/utils/color_encoder.py` | 5 | Color analysis encoding |

**Key logging areas:**
- VRAM usage warnings
- Performance timing
- File operation errors
- Game detection state changes

---

## 10. Development Tools

Tools for testing, profiling, and component generation (not part of runtime).

| File | Log Count | Description |
|------|-----------|-------------|
| `backend/tools/profile_interpolation.py` | 67 | Interpolation performance profiling |
| `backend/tools/generate_components.py` | 64 | Prompt component generation |
| `backend/tools/generate_looped_preview.py` | 43 | Preview animation generation |
| `backend/tools/select_diverse_components.py` | 38 | Component diversity selection |
| `backend/tools/compute_embeddings.py` | 15 | Embedding computation |
| `backend/tools/find_diverse_components.py` | 14 | Diversity analysis |
| `backend/tools/analyze_component_quality.py` | 7 | Quality analysis |

**Note:** These tools have their own logging for development/debugging purposes but are not part of the production runtime.

---

## Logger Hierarchy

All loggers use `logging.getLogger(__name__)`, creating a hierarchy:

```
root
├── __main__ (daemon.py when run directly)
├── backend.core.async_orchestrator
├── backend.core.dream_controller
├── backend.core.comfyui_api
├── backend.core.generation_coordinator
├── backend.core.generator
├── backend.core.display_selector
├── backend.core.frame_buffer
├── backend.core.shared_resources
├── backend.core.workflow_builder
├── backend.core.workers.interpolation_worker
├── backend.core.workers.cache_worker
├── backend.core.workers.keyframe_worker
├── backend.cache.manager
├── backend.cache.injection_strategy
├── backend.cache.collapse_detector
├── backend.cache.dual_similarity
├── backend.cloud.runpod_handler
├── backend.cloud.websocket_client
├── backend.cloud.state_sync
├── backend.cloud.frame_pusher
├── backend.interpolation.latent_encoder
├── backend.fresh.buffer
├── backend.prompts.combinatorial
├── backend.utils.vram_profiler
├── backend.utils.perf_stats
├── backend.utils.file_ops
├── backend.utils.game_detector
├── backend.utils.status_writer
├── backend.utils.prompt_manager
├── backend.utils.phash_encoder
└── backend.utils.color_encoder
```

---

## Log Level Configuration

Logging levels are configured in `backend/core/dream_controller.py`:

- **Root logger:** Set based on config (typically INFO)
- **Suppressed loggers (WARNING):** PIL, httpx, httpcore, websockets, asyncio
- **Verbose loggers (INFO):** comfyui_api, workflow_builder

---

## Priority 1: Core Control Flow (Deep Investigation)

The following sections document the three heaviest logging files that form the core control flow of Dream Gen.

---

### daemon.py (158 log statements)

**Logger name:** `__main__` (when run directly) or `daemon` (when imported)

**Purpose:** Top-level process orchestrator. Manages ComfyUI and DreamController lifecycles, handles signals, auto-restart on crashes, and control commands.

#### Log Banners (Major Phase Transitions)

The daemon uses `="*70` separator lines to mark major phases. These are excellent anchors when scrolling through logs:

| Banner | Meaning |
|--------|---------|
| `DREAM WINDOW DAEMON STARTING` | Daemon initialization beginning |
| `STARTING COMFYUI BACKEND` | About to launch ComfyUI subprocess |
| `STARTING DREAMCONTROLLER` | ComfyUI ready, launching controller |
| `MONITORING LOOP STARTED` | All systems go, entering main loop |
| `DAEMON SHUTDOWN INITIATED` | Graceful shutdown starting |
| `DAEMON SHUTDOWN COMPLETE` | All processes terminated cleanly |

#### Status Tags

| Tag | Meaning |
|-----|---------|
| `[OK]` | Success confirmation (e.g., `[OK] Daemon initialization complete`) |
| `[CONTROL]` | Control command received from file (PAUSE/RESUME/SHUTDOWN) |
| `[1/2]`, `[2/2]` | ComfyUI startup phases: 1=web server up, 2=queue endpoint ready |
| `⚠️` | Warning - often orphaned/existing process detection |

#### ComfyUI Lifecycle Logs

**Startup sequence:**
```
Checking for orphaned ComfyUI processes...
No orphaned processes found
Script: /path/to/comfyui.bat
Waiting for ComfyUI to be FULLY ready...
[1/2] Web server up in X.Xs
[2/2] Queue endpoint ready in X.Xs
ComfyUI FULLY READY in X.Xs
```

**Already running (potential issue):**
```
⚠️  ComfyUI ALREADY RUNNING!
ComfyUI appears to be running from a previous session.
CONTINUING with existing ComfyUI instance.
```
→ **Action:** If frames don't generate, run `uv run kill_comfyui.py`

**Orphaned process cleanup:**
```
Found orphaned ComfyUI process (PID: XXXXX)
Killed orphaned process tree (PID: XXXXX)
Cleaned up N orphaned process(es)
```

#### Controller Lifecycle Logs

**Normal startup:**
```
Using daemon Python: /path/to/python
Launching: /path/to/python backend/main.py
Controller process started (PID: XXXXX)
[OK] Controller running
```

**Immediate death (bad):**
```
Controller died immediately (exit code: X)
Common causes:
  - Wrong Python version/venv
  - Missing dependencies
  - Config errors in backend/config.yaml
```
→ **Action:** Check controller's own log file (`logs/dream_controller.log`)

#### Crash/Restart Logs

**Auto-restart pattern:**
```
ComfyUI process died (exit code: X)  # or Controller
Restarting ComfyUI in Xs...
[OK] ComfyUI restarted successfully
```

**Rate limit hit:**
```
comfyui has restarted N times in the last hour. Max restarts (5) reached. Giving up.
```
→ **Action:** Manual investigation needed. Check underlying cause before restarting daemon.

#### Key Debug Points

| Line Pattern | What to Look For |
|--------------|------------------|
| `exit code: X` | Non-zero = crash. Check ComfyUI logs for CUDA/VRAM errors |
| `health check timeout` | ComfyUI taking too long to start. Check model loading, disk I/O |
| `queue endpoint timeout` | Web server up but ComfyUI still loading. Wait or check GPU |
| `Failed to start` | Hard failure - check the `exc_info=True` traceback |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| ComfyUI won't start | `startup script not found` | Fix `daemon.comfyui.startup_script` in config |
| Port conflict | `already running` + frames don't generate | Run `kill_comfyui.py` |
| VRAM exhaustion | ComfyUI crash loop | Check GPU memory, reduce model size |
| Controller crash loop | `died immediately` + restart loop | Check `dream_controller.log` for Python errors |

---

### async_orchestrator.py (157 log statements)

**Logger name:** `backend.core.async_orchestrator`

**Purpose:** Main async generation loop. Coordinates KeyframeWorker (HTTP I/O), InterpolationWorker (GPU compute), and CacheAnalysisWorker (CPU analysis). Makes inline injection decisions for collapse prevention.

#### Log Banners

| Banner | Meaning |
|--------|---------|
| `STARTING ASYNC GENERATION ORCHESTRATOR` | Orchestrator initializing |
| `Populating fresh frame buffer...` | Pre-generating txt2img frames for each template |
| `ORCHESTRATOR STATISTICS` | Final stats on shutdown |
| `=== RECOVERING FROM KEYFRAME X FAILURE ===` | Recovery mode triggered |

#### Worker Status Tags

| Tag | Meaning |
|-----|---------|
| `[OK]` | Success - workers started, keyframe completed, injection succeeded |
| `[FAIL]` | Keyframe generation failed after retries |

**Normal keyframe completion:**
```
[OK] Keyframe N completed: keyframe_XXX.png
  Marked keyframe N ready (seq XXX)
  Pre-registered interpolations N->N+1: seq X-Y
  Pre-registered keyframe N+1: seq Z
  Submitting keyframe N+1 (DRIFT, denoise=0.20)
```

**Keyframe failure and recovery:**
```
[FAIL] Keyframe N failed after X retries: error message
=== RECOVERING FROM KEYFRAME N FAILURE ===
  Marked keyframe N (seq X) as failed
  Found X orphaned interpolations to remove
  Falling back to keyframe N-1: keyframe_XXX.png
  Display was at seq X (deleted) - resetting to Y
  Registered X interpolations: seq A-B
  Registered KFN at seq C
  Submitted new KFN request (recovery, DRIFT mode)
=== RECOVERY COMPLETE ===
```

#### Injection System Tags

**Warmup period (no interventions):**
```
[WARMUP] Keyframe X/Y - skipping injection (establishing baseline)
[WARMUP_COMPLETE] Warmup period finished! Collapse detection and adaptive interventions now ACTIVE.
```

**Collapse detection:**
```
[COLLAPSE_METRICS] Status: ok|converging|collapsed, Delta: X.XXX, Similarity: X.XXX
```
- `ok` - No issues, normal generation
- `converging` - Early warning, may trigger mutation/injection
- `collapsed` - Forced intervention needed

**Interventions (in escalation order):**

| Tag | Meaning | Severity |
|-----|---------|----------|
| `[COLLAPSE_RESPONSE]` | Forced mutation of components | Soft (preferred) |
| `[SCALING]` | Injection probability increased | Medium |
| `-> Injection triggered (cache)` | Cache frame injected | Medium |
| `[EMERGENCY]` | High injection frequency forcing seed | Hard |
| `-> Injection triggered (seed)` | Template switch via fresh buffer | Hard reset |

#### Denoising State Machine Tags

| Tag | Meaning |
|-----|---------|
| `[MUTATION]` | Component mutated, entering BEND mode |
| `[BEND]` | Using high denoise for prompt transition |
| `[FRESH]` | Using pre-generated frame from fresh buffer |
| `[TEMPLATE_SWITCH]` | Full template change (old → new) |

**Template switch sequence:**
```
  -> Injecting FRESH frame (keyframe N)
  [FRESH] Using pre-generated frame:
    Template: 'template_name'
    Components: {...}
    Buffer age: X.Xs
    Prompt: first 80 chars...
  [TEMPLATE_SWITCH] Interpolated blend: 85% fresh + 15% current
  [TEMPLATE_SWITCH] 'old_template' → 'new_template'
    ✓ Prompt system switched
    ✓ Cache manager switched (old cache archived)
    ✓ Collapse detector reset (warmup: 50 frames)
```

#### Statistics Logs

**Every 10 keyframes:**
```
[STATS] Keyframe N
  Prompt System:
    Template: 'template_name'
    Total mutations: X
    Frames since mutation: Y
    In BEND mode: True/False
  Fresh Buffer:
    Ready: True/False
    Generated: X
    Consumed: Y
    Avg gen time: X.XXs
  Cache injections: N
```

**Every 50 keyframes (tuning guidance):**
```
[INTERVENTION_STATS] Keyframe N
  Forced mutations: X
  Cache injections: Y
  Template switches: Z
  Collapse detections: W
  Ratios: mutations=X%, cache=Y%, switches=Z%
  Target: mostly mutations, some cache, rare switches
```
→ **Action:** If switches >> mutations, thresholds may be too aggressive. If mutations >> switches but still collapsing, thresholds may be too lenient.

#### Buffer/Backpressure Logs

**Throttling (normal when buffer is full):**
```
  System throttled (30.0s / 30s), fresh buffer: X/Y ready
```

**Backpressure warning:**
```
Interpolation queue depth high (X), throttling...
```
→ **Action:** VAE may be bottleneck. Check GPU utilization.

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `Cannot find sequence number for keyframe` | Registration bug - should not happen |
| `Injection failed, falling back to normal generation` | Cache/fresh buffer issue |
| `KeyframeWorker died! Restarting...` | Worker crash - check ComfyUI connection |
| `InterpolationWorker died! Restarting...` | GPU/CUDA error in VAE |
| Gap detected: Missing interpolations | Interpolations weren't submitted - race condition |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Visual collapse | High `Similarity` in COLLAPSE_METRICS | Lower convergence thresholds |
| Too many interventions | High switch count in INTERVENTION_STATS | Raise thresholds, longer warmup |
| Generation stalls | No new keyframes, buffer draining | Check worker logs, ComfyUI queue |
| Memory growth | No obvious log | Check keyframe_sequences cleanup |

---

### dream_controller.py (147 log statements)

**Logger name:** `backend.core.dream_controller`

**Purpose:** Entry point and initialization. Sets up logging, initializes all subsystems, runs main loop, handles cloud mode.

#### Logging Configuration (Critical!)

The `setup_logging()` function configures the entire logging hierarchy:

| Logger | Level | Notes |
|--------|-------|-------|
| Root | DEBUG (file), INFO (console) | File gets everything, console filtered |
| `urllib3`, `websockets`, `PIL`, etc. | WARNING | Suppressed - too noisy |
| `utils.status_writer` | INFO | Suppressed DEBUG - very chatty |

**Log file location:** `logs/dream_controller.log`
- Rotates at 5MB
- Keeps 3 backups (`.log.1`, `.log.2`, `.log.3`)

#### Initialization Banners

```
======================================================================
DREAM WINDOW CONTROLLER INITIALIZING
======================================================================
```

**Subsystem initialization sequence:**
```
Initializing subsystems...
[INFINITE GEN] CombinatorialPromptSystem loaded
  Templates: N
  Categories: [list of categories]
Initializing Dual-Metric Similarity Manager...
  Using ColorHist + pHash-8 with OR logic for collapse detection
Initializing hybrid mode with VAE interpolation...
Loading VAE for interpolation...
Using device: cuda:0
Target resolution: (W, H)
[OK] VAE interpolation enabled
Initializing buffered frame system...
Using AsyncGenerationOrchestrator (parallelized)
Loading secondary VAE for injection blending (zero contention)...
[OK] Secondary VAE loaded for injection (dual-VAE architecture)
  Workers: KeyframeWorker, InterpolationWorker, CacheAnalysisWorker
  Expected FPS improvement: 2x+ (from ~2.7 to ~5+ fps)
[OK] Buffered frame system initialized
Cloud mode: disabled (standalone Rainmeter mode)
[OK] Initialization complete
Mode: hybrid
Resolution: [W, H]
Model: model_name
======================================================================
```

#### Status Tags

| Tag | Meaning |
|-----|---------|
| `[OK]` | Subsystem initialized successfully |
| `[INFINITE GEN]` | Using CombinatorialPromptSystem (good) |
| `[BOOTSTRAP]` | Generating initial txt2img frame (legacy mode only) |
| `[GAME]` | Game detection triggered pause/resume |

#### Game Detection Logs

```
[GAME] DETECTED: game_name
Pausing generation and freeing VRAM...
[OK] VRAM freed - safe for gaming!
```

```
[GAME] Game closed - resuming generation
(Models will reload on next generation - ~15s delay expected)
```

#### Buffer Status Loop (Every 10s)

```
Buffer: 25.3s / 30s (84.3%) | KF: 150 | INT: 1200 | Displayed: 5000
```
- `Buffer: Xs / Ys (Z%)` - Buffered content vs target
- `KF: N` - Keyframes generated
- `INT: N` - Interpolations generated
- `Displayed: N` - Frames shown to user

#### VAE Lock Contention Warning

```
VAE Lock Contention: N ops, avg wait: X.Xms, max wait: Y.Yms
```
→ **Action:** If avg_wait > 10ms consistently, consider dual-VAE architecture (already enabled by default)

#### Cloud Mode Logs

```
Initializing cloud mode...
[OK] Cloud mode initialized
  VPS URL: wss://...
  Frame format: webp
  State sync interval: 10 keyframes
Connecting to VPS...
[OK] Connected to VPS
```

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `Failed to initialize hybrid mode with VAE` | VAE loading failed - check VRAM/model path |
| `Failed to load CombinatorialPromptSystem` | templates.yaml or components.yaml issue |
| `Cloud mode dependencies not available` | Missing websockets/msgpack packages |
| `Could not free VRAM` | ComfyUI /free endpoint not available |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| VAE won't load | `Failed to initialize hybrid mode` | Check CUDA, VRAM, model paths |
| Wrong prompt system | No `[INFINITE GEN]` tag | Ensure templates.yaml exists |
| Game detection fails | No `[GAME]` logs when gaming | Check game_detector config |
| Cloud won't connect | `Failed to connect to VPS` | Check network, VPS URL, auth |

---

## Priority 2: Cache System (Deep Investigation)

The cache system is responsible for preventing visual collapse (mode collapse) by storing diverse frames and injecting them when generation starts converging. Understanding these logs is **critical** for debugging "frames look the same" issues.

### Architecture Overview

```
Frame Generation → CacheAnalysisWorker → should_cache_frame() → CacheManager
                           ↓
             DualMetricSimilarityManager (ColorHist + pHash)
                           ↓
                  ModeCollapseDetector
                           ↓
                  CacheInjectionStrategy → Injection decision
```

---

### manager.py (48 log statements)

**Logger name:** `backend.cache.manager`

**Purpose:** Stores generated images with dual-metric embeddings. Handles LRU eviction, persistence, and per-template cache isolation with archive/restore.

#### Initialization Logs

```
CacheManager initialized: 45/100 entries, 2 archived templates
```
- Shows loaded cache size vs max, plus number of template archives

**Cache loading:**
```
No existing cache found, starting fresh
Loaded 45 cache entries
```

**Archive index loading:**
```
No archive index found, starting fresh
Loaded archive index: 2 templates
Archive path missing for template 'old_template': /path/to/archive
```
→ **Action:** Missing archive path = stale index entry, auto-cleaned

#### Cache Operations

| Log Pattern | Meaning |
|-------------|---------|
| `Added to cache: cache_00045_1706123456 (total: 46)` | New frame cached |
| `Cache size enforced: 100/100` | Max size reached, oldest evicted |
| `Evicted from cache: cache_00001_...` | LRU eviction occurred |
| `Cache image missing: cache_00010_...` | Integrity issue - file deleted externally |
| `Cache cleared!` | Full cache wipe (manual or template switch) |

#### Diversity Gating Logs

**Frame accepted (diverse enough):**
```
Frame is diverse (color:1.45<1.80 OR struct:0.52<0.65) - caching
```

**Frame rejected (too similar):**
```
Frame is redundant (color:1.92 FAIL, struct:0.71 FAIL, logic:OR) - skipping cache
```
→ **Key:** `PASS`/`FAIL` shows which metric(s) blocked caching

#### Template-Aware Cache Logs

**Template context change:**
```
Cache template context changed: 'dissolution' -> 'architectural'
```

**Archive operations:**
```
Archiving cache for template 'dissolution' (45 entries)
  Removing previous archive for 'dissolution'
  Archived to: /path/to/cache/archived/template_dissolution_1706123456
```

**Restore operations:**
```
Restoring cache for template 'architectural' from /path/to/archive
Active cache not empty during restore - entries will be overwritten
  Restored 32 entries for template 'architectural'
```

**Template switch summary:**
```
Switching template: 'dissolution' -> 'architectural'
Template switch complete: archived=True, restored=True, cache_size=32
```

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Cache not populating | No `Added to cache` logs | Check diversity thresholds (too strict) |
| Cache fills with similar frames | Many `Frame is diverse` but output looks same | Thresholds too lenient, OR logic may be wrong |
| Archive corruption | `Failed to restore archive` | Delete archive dir, let cache rebuild |
| Memory issues | Many `Evicted from cache` rapidly | Max size too small for diversity |

---

### injection_strategy.py (26 log statements)

**Logger name:** `backend.cache.injection_strategy`

**Purpose:** Decides **when** and **how** to inject cached frames to break mode collapse. Uses VAE latent blending for smooth transitions.

#### Initialization Logs

```
CacheInjectionStrategy initialized
  Mode: ASYNC (thread-safe)
  Similarity method: dual_metric
  Injection mode: dissimilar
  Blend weight: 0.6
  Anti-loop tracking: 5 recent injections
```

#### Dissimilar Selection Logs

**Smart prioritization (based on which metric triggered):**
```
  Smart selection: Prioritizing COLOR dissimilarity (color collapse detected)
  Smart selection: Prioritizing STRUCTURAL dissimilarity (structural collapse detected)
  Smart selection: Using max dissimilarity (both metrics triggered)
```

**Candidate selection:**
```
[DISSIMILAR] Selected cache_00023_1706123456 (color:1.25, struct:0.58, dissim:0.72, prioritized COLOR (dissim:0.85))
```
- `color`/`struct`: Similarity scores to current frame
- `dissim`: Combined dissimilarity score used for ranking
- `prioritized`: Which metric was weighted for selection

**Anti-loop penalty (avoiding recent injections):**
```
  Penalizing recently used cache_00015_... (weight *= 0.1)
```

#### VAE Blending Logs

```
[BLEND] Created blended keyframe (60% cached, 40% current)
```
- Shows blend ratio between cached frame and current frame

**Fallback to direct copy:**
```
No VAE access available, falling back to direct copy
VAE blending failed: CUDA out of memory
Falling back to direct copy
[DIRECT_COPY] Copied cached frame cache_00023_...
```
→ **Action:** Direct copy = no smooth transition, may be jarring

#### Emergency Injection Logs

```
[EMERGENCY] High collapse frequency (35.0%) - forcing seed injection
```
→ **Action:** System repeatedly collapsing - check thresholds

#### Adaptive Seed Injection Logs

```
[SEED] Adaptive seed injection (probability: 8.5%, cache injections: 25)
```
- Shows probability ramp based on cumulative cache injections

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `No current image for dissimilar injection` | Missing reference frame |
| `Cache is empty` | No frames available to inject |
| `No dissimilar frames (color:..., struct:...)` | All cached frames too similar |
| `Failed to encode current frame` | Similarity system error |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Injections jarring | `[DIRECT_COPY]` instead of `[BLEND]` | Fix VAE access, check VRAM |
| Same frame re-injected | No `Penalizing recently used` | Anti-loop broken |
| No candidates found | `No dissimilar frames` | Widen dissimilarity_range |
| Emergency spam | Frequent `[EMERGENCY]` | Thresholds too aggressive |

---

### collapse_detector.py (15 log statements)

**Logger name:** `backend.cache.collapse_detector`

**Purpose:** Real-time detection of mode collapse using dual-metric embedding analysis. Tracks convergence trends and recommends actions.

#### Initialization Logs

```
ModeCollapseDetector initialized (dual-metric mode, history_size=100, detection_window=50, warmup_frames=50, baseline_comparison=True)
  Thresholds: color=0.15/0.30, struct=0.08/0.15
```
- `color=0.15/0.30`: convergence_threshold / force_cache_threshold
- `struct=0.08/0.15`: Same for structural metric

#### Warmup Period Logs

**During warmup (no triggers fire):**
```
[WARMUP] Frame 25/50 - recording baseline
```

**Warmup complete:**
```
[WARMUP_COMPLETE] Baseline established after 50 frames. Color baseline: 1.2345 ± 0.0567, Struct baseline: 0.5678 ± 0.0234
```
→ **Key:** Baseline values become comparison reference for collapse detection

#### Convergence Detection Logs

**Calibration stats (DEBUG level):**
```
[CALIBRATION] (baseline) COLOR: baseline=1.23, recent=1.45, delta=0.22 | STRUCT: baseline=0.56, recent=0.62, delta=0.06
```
- `delta` = recent - baseline (positive = converging)

**Convergence detected (medium severity):**
```
[CONVERGING] COLOR=0.18 OR STRUCT=0.10 -> scaling to 65% + mutation
```
- Shows which metric(s) triggered
- `scaling to N%` = injection probability scaled

**Collapse detected (severe):**
```
[COLLAPSE] Severe convergence! COLOR=0.35>0.30 AND STRUCT=0.18>0.15 -> forcing cache (100%) + mutation
```
- Forces immediate injection + mutation

#### Template Switch Logs

```
ModeCollapseDetector reset for template switch: 'dissolution' -> 'architectural' (entering warmup for 50 frames)
ModeCollapseDetector initialized for template 'architectural' (entering warmup for 50 frames)
```

#### Reset Logs

```
ModeCollapseDetector reset (all histories and baseline cleared, re-entering warmup)
ModeCollapseDetector partial reset (kept 5 recent frames, baseline preserved)
```
- Full reset = injection happened, breaking convergence signal
- Partial reset = softer break, keeps some context

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| High positive `delta` | Visual convergence happening |
| `baseline` very high | Initial generation already repetitive |
| `OR` in trigger | One metric triggered (targeted intervention) |
| `AND` in trigger | Both metrics triggered (severe collapse) |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Too many triggers | Constant `[CONVERGING]` | Raise thresholds |
| Never triggers | Never see convergence logs | Lower thresholds |
| Triggers during warmup | `[WARMUP]` but still collapses | Warmup too short |
| Wrong baseline | `baseline=2.0+` (very high) | Initial generation already collapsed |

---

### dual_similarity.py (11 log statements)

**Logger name:** `backend.cache.dual_similarity`

**Purpose:** Coordinates ColorHist (color palette drift) and pHash-8 (structural drift) with OR logic.

#### Initialization Logs

```
DualMetricSimilarityManager initialized
  Color threshold: 1.80
  Struct threshold: 0.65
  Injection logic: any (OR logic)
```

#### Collapse Check Results

**Both metrics triggered:**
```
BOTH color (1.92>1.80) and structural (0.71>0.65) collapse
```

**Single metric triggered:**
```
COLOR collapse detected (1.92>1.80)
STRUCTURAL collapse detected (0.71>0.65)
```

**No collapse:**
```
No collapse (color:1.45, struct:0.58)
```

#### Encoding Errors

```
Color encoding failed for /path/to/image.png
Structural encoding failed for /path/to/image.png
Failed to encode image /path/to/image.png: [error details]
```
→ **Action:** Check image file exists and is valid PNG

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Always triggers | Most frames show `collapse` | Raise thresholds |
| Never triggers | No `collapse` logs | Lower thresholds |
| Encoding failures | `encoding failed` | Check PIL installation, image paths |

---

## Priority 3: Worker Subsystem (Deep Investigation)

Workers handle the concurrent generation pipeline. Understanding their logs is key for debugging performance issues, stalls, and generation failures.

### Architecture Overview

```
AsyncGenerationOrchestrator
    ├── KeyframeWorker (HTTP I/O → ComfyUI)
    ├── InterpolationWorker (GPU → VAE encode/decode)
    └── CacheAnalysisWorker (CPU → similarity calculations)
```

---

### keyframe_worker.py (21 log statements)

**Logger name:** `backend.core.workers.keyframe_worker`

**Purpose:** Async HTTP I/O for ComfyUI generation. Handles retries with exponential backoff.

#### Initialization Logs

```
KeyframeWorker initialized (max queue: 5, max retries: 3)
KeyframeWorker started
```

#### Request Processing Logs

```
Submitted keyframe request: KF5 (queue depth: 1, mode: drift)
Processing keyframe request: KF5 (seq 100, mode: drift)
```
- `mode: drift` = low denoise (normal)
- `mode: bend` = high denoise (after mutation)

#### Success Logs

```
Moved keyframe: temp_image.png -> keyframe_005.png
[OK] Keyframe 5 generated in 2.34s (total: 150)
[OK] Keyframe 5 generated in 8.50s (after 2 retries, total: 151)
```

#### Retry Logs

```
Keyframe 5 attempt 1/3 failed: Connection timeout
Attempting ComfyUI recovery before retry 2...
Waiting 5s before retry...
```

**Protection during retry:**
```
Retry mode: protecting source image keyframe_004.png
```
→ **Key:** Source image won't be deleted during retry

#### Failure Logs

```
[FAIL] Keyframe 5 failed after 3 attempts: ComfyUI unresponsive
Keyframe 5 generation timed out after 180s (attempt 1/3)
```

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `queue depth: 5` (max) | Backpressure - generation can't keep up |
| `timed out after 180s` | ComfyUI hung - check GPU/queue |
| `ComfyUI unresponsive` | ComfyUI crashed - restart needed |
| Many retries | Network instability or resource contention |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Generation stalls | No new `[OK]` logs | Check ComfyUI status |
| High retry rate | Frequent `attempt 2/3` | Check network, VRAM |
| Queue backed up | `queue depth: 5` sustained | Interpolation bottleneck |

---

### interpolation_worker.py (42 log statements)

**Logger name:** `backend.core.workers.interpolation_worker`

**Purpose:** VAE-based frame interpolation between keyframes. GPU-bound operations with batched processing.

#### Initialization Logs

```
InterpolationWorker initialized (max queue: 10, gpu_slerp: False)
InterpolationWorker started
```
- `gpu_slerp: True` = spherical lerp on GPU (faster on cloud)

#### Pair Processing Logs

```
Submitted interpolation pair: KF5->KF6 (queue depth: 2)
Processing interpolation pair: KF5->KF6 (seq 101-108)
```

#### Timing Breakdown (Key Performance Logs)

```
[TIMING] Interpolation 5->6 breakdown (BATCHED):
  Total time:        1.850s
  Slerp precompute:  0.015s
  Phase timings:
    - Slerp all:     0.120s (15.0ms per frame)
    - Decode all:    1.450s (181.2ms per frame)
    - Save all:      0.285s (35.6ms per frame)
  [BATCHED] Single VAE lock acquisition for 8 frames
```
→ **Key:** `Decode all` is usually the bottleneck (VAE operations)

#### Success/Failure Logs

```
[OK] Interpolated 5->6 in 1.85s (0.231s/frame)
Interpolation pair 5->6 incomplete or failed
```

#### Latent Caching Logs

```
Using cached latent for KF5
Encoded and cached KF6
Cleaned up 3 old keyframe latents
```

#### Batch Decode Fallback

```
Batch decode failed, falling back to sequential: CUDA out of memory
[PERF] Sequential decode: 8 frames, avg 245.3ms/frame
```
→ **Action:** Batch failed = VRAM pressure, sequential is slower

#### Executor Queue Warning

```
Executor queue depth high: 12 pending tasks
```
→ **Action:** Event loop backing up - check for blocking operations

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `queue depth: 10` (max) | Backpressure - interpolation behind |
| `Decode all: 3.0s+` | VAE bottleneck - check GPU load |
| `Batch decode failed` | VRAM exhausted - reduce batch size |
| `Executor queue depth high` | Event loop blocked somewhere |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Slow interpolation | High `Decode all` time | Check GPU utilization |
| Stuttery playback | `queue depth` fluctuating | Buffer underrun |
| OOM errors | `CUDA out of memory` | Reduce batch size, check VRAM |
| Missing frames | `incomplete or failed` | Check VAE initialization |

---

### cache_worker.py (26 log statements)

**Logger name:** `backend.core.workers.cache_worker`

**Purpose:** Async diversity analysis for cache population. CPU-bound similarity calculations.

#### Initialization Logs

```
Using Phase 1 cache analysis (basic diversity)
CacheAnalysisWorker initialized (max queue: 20)
CacheAnalysisWorker started
```

**Phase 2 (advanced monitoring, not yet implemented):**
```
Advanced cache monitoring ENABLED (Phase 2)
Phase 2 hooks are present but not yet implemented!
Background monitoring task started (Phase 2 hook)
```

#### Queue Management Logs

```
Submitted frame for analysis: keyframe_005.png (queue depth: 3)
Cache analysis queue near capacity (16/20) - skipping frame to prevent backlog
```
→ **Key:** Queue near capacity = analysis can't keep up, frames skipped

#### Analysis Logs

```
Analyzing frame: keyframe_005.png
Frame cached: keyframe_005.png
Skipping cache (frame not diverse enough)
```

#### Diversity Stats Logs (Periodic)

```
[CACHE_DIVERSITY] Color:0.72, Struct:0.65, Size:45
```
- Higher scores = more diverse cache
- Color/Struct are 0-1 normalized diversity scores

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `queue depth: 20` (max) | Analysis backed up |
| `skipping frame to prevent backlog` | Dropping frames - not critical |
| Low diversity scores | Cache filling with similar frames |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Queue overflow | `queue near capacity` | Normal if occasional |
| Low diversity | `Color:0.3, Struct:0.3` | Diversity thresholds too lenient |
| Analysis errors | `Error analyzing frame` | Check similarity manager |

---

## Worker Correlation Table

| Symptom | KeyframeWorker | InterpolationWorker | CacheAnalysisWorker |
|---------|----------------|---------------------|---------------------|
| **Generation stalls** | Check for `[FAIL]`, timeouts | Check queue depth | Not relevant |
| **Stuttery playback** | Check avg time | Check decode timing | Not relevant |
| **Visual collapse** | Not relevant | Not relevant | Check diversity stats |
| **Memory issues** | Check retry patterns | Check batch failures | Check queue overflow |

---

## Priority 4: Fresh Frame Buffer (Deep Investigation)

The Fresh Frame Buffer is crucial for preventing visual stagnation during template switches. It pre-generates txt2img frames for each template so switches are instant rather than requiring a generation cycle.

---

### buffer.py (33 log statements)

**Logger name:** `backend.fresh.buffer`

**Purpose:** Maintains one pre-generated "fresh" frame per template. At startup, populates entire buffer with txt2img generations. When a frame is consumed (during seed injection), regenerates that template's frame in the background with fresh components.

#### Architecture Overview

```
Startup: populate_all()
    → For each template: _generate_for_template()
    → Store in _buffer[template_id] = BufferedFrame

Seed Injection: select_and_consume()
    → Pick random template from _available_pool (excluding recently used)
    → Return frame, mark slot as None
    → Background task: _regenerate_template()

Pool Management:
    → _available_pool: templates not yet used this cycle
    → _used_this_cycle: templates already consumed
    → When pool empty: reset (all templates available again)
```

#### Initialization Logs

```
FreshFrameBuffer initialized (per-template mode)
  Output directory: /path/to/frames/fresh/
```

#### Startup Population (Critical Sequence)

**Success pattern:**
```
============================================================
[FRESH] Populating buffer for 5 templates...
============================================================
[FRESH] Generating 1/5: 'dissolution'
[FRESH] Generated 'dissolution' in 3.45s
[FRESH] Generating 2/5: 'architectural'
[FRESH] Generated 'architectural' in 3.21s
... (continues for all templates)
============================================================
[FRESH] Buffer populated: 5 frames in 17.2s
[FRESH] Avg generation time: 3.44s per frame
============================================================
```

**Failure patterns:**

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `Template 'X' not found` | Template missing from prompt system | Check templates.yaml |
| `Generation returned no path for 'X', retry 1/3` | ComfyUI didn't return result | Check ComfyUI queue |
| `Generation error for 'X': [error]` | Exception during generation | Check traceback |
| `Failed to generate 'X' after 3 retries` | Hard failure | Check ComfyUI, models |

→ **Critical:** If `populate_all()` fails, the entire system cannot start. Fix the underlying issue.

#### Consumption & Selection Logs

**Normal consumption:**
```
[FRESH] Consumed 'dissolution' (age: 45.2s, pool: 4/5 remaining)
```
- `age`: How long the frame sat in buffer (older = okay, freshness doesn't degrade)
- `pool: 4/5`: 4 templates still available in current cycle, 5 total templates

**Pool reset (all templates used):**
```
[FRESH] All templates used once, resetting selection pool
```
→ Normal behavior - ensures all templates get equal rotation

**Waiting for regeneration (buffer depleted):**
```
[FRESH] No ready frames, will wait for regeneration...
[FRESH] Waiting for frame regeneration...
[FRESH] Frame ready after wait: 'architectural'
```
→ **Warning:** This means consumption outpaced regeneration. Usually brief, but indicates heavy template switching.

**Race condition recovery:**
```
[FRESH] Ready templates emptied during selection, retrying...
```
→ Normal - handled automatically via recursion

**Timeout (bad):**
```
[FRESH] Timeout waiting for frame (120s)
```
→ **Critical:** ComfyUI likely frozen or crashed. Check daemon logs.

#### Background Regeneration Logs

```
[FRESH] Regenerating 'dissolution' with fresh components...
[FRESH] Regenerated 'dissolution'
```
→ Normal background activity after each consumption

**Regeneration failure:**
```
[FRESH] Failed to regenerate 'dissolution'
```
→ Check ComfyUI status - the template won't be available until next successful generation

**Skipped regeneration:**
```
[FRESH] 'dissolution' already regenerating, skipping
```
→ Normal - prevents duplicate work

#### Debug-Level Logs (Verbose)

Enable `DEBUG` level to see:
```
[FRESH] Template 'dissolution' prompt: ethereal cyan crystalline structures dissolving into...
```
→ Shows first 80 chars of generated prompt

#### Status & Lifecycle Logs

**Deprecation warnings (if using old API):**
```
[FRESH] ensure_ready() is deprecated. Use populate_all() at startup instead.
[FRESH] consume() is deprecated. Use await select_and_consume() instead.
```

**Buffer clear:**
```
[FRESH] Clearing buffer
```
→ Happens on shutdown or config change

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| `age: 300s+` | Buffer not being consumed - check injection triggers |
| `pool: 1/5 remaining` repeatedly | Same template being selected - check selection logic |
| Frequent `Waiting for frame regeneration` | Heavy template switching or slow ComfyUI |
| No `[FRESH]` logs during runtime | Buffer consumed but seed injection not happening |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Startup hangs | Stuck at `Generating N/M` | Check ComfyUI connection |
| No template variety | Same template in logs | Check `_available_pool` management |
| Slow template switches | `Waiting for frame regeneration` | Pre-generate more aggressively |
| Buffer always empty | No `Regenerated` logs | Background tasks not running |
| OOM during populate | `Generation error` + CUDA OOM | Reduce concurrent generations |

#### Correlation with Other Logs

| Fresh Buffer Event | async_orchestrator.py Log |
|--------------------|---------------------------|
| `Consumed 'X'` | `[FRESH] Using pre-generated frame` |
| `Timeout waiting` | `Injection failed, falling back to normal generation` |
| `pool: 0/N remaining` | Potential `[TEMPLATE_SWITCH]` |

---

## Priority 5: VAE/Interpolation System (Deep Investigation)

The VAE (Variational Autoencoder) system handles smooth frame transitions through latent space interpolation. Understanding these logs is critical for debugging visual glitches, performance issues, and VRAM problems.

---

### latent_encoder.py (44 log statements)

**Logger name:** `backend.interpolation.latent_encoder`

**Purpose:** Converts images to/from latent space using the VAE (Variational Autoencoder). Enables smooth interpolation between keyframes by blending in latent space rather than pixel space.

#### Architecture Overview

```
Image → encode() → Latent Tensor (1, 4, H/8, W/8)
                        ↓
               Interpolation (slerp)
                        ↓
Latent Tensor → decode() → Image

Batch Operations:
    encode_batch() - Multiple images in one GPU call
    decode_batch() - Multiple latents in one GPU call (~3x faster)

Resolution Modes:
    Full resolution: Higher quality, slower
    Lower resolution: interpolation_resolution_divisor > 1, faster with upscaling
```

#### Initialization Logs

**Standard initialization:**
```
Loading SD 1.5 VAE for interpolation...
  GPU compute capability: 8.6 (torch.compile supported)
  Attempting to compile VAE decoder with torch.compile...
  [OK] VAE decoder compiled (will verify on first use)
[OK] VAE loaded successfully
  Target device: cuda:0
  Actual device: cuda:0
  Model dtype: torch.float16
  VRAM usage: 334.5 MB
  Scale factor: 0.18215
```

**Lower-resolution interpolation (performance mode):**
```
Lower-res interpolation enabled: 2x downscale
Upscale method: bilinear
```

**torch.compile scenarios:**

| Log Pattern | Meaning |
|-------------|---------|
| `GPU compute capability: 8.6 (torch.compile supported)` | Modern GPU, compilation enabled |
| `GPU compute capability: 6.1 (requires >= 7.0 for torch.compile)` | Pascal/older GPU, no compilation |
| `Skipping torch.compile (GPU too old)` | Falling back to standard decoder |
| `Could not compile VAE decoder: [error]` | Compilation failed, continuing uncompiled |
| `torch.compile disabled via config` | User disabled in config.yaml |

**Warnings during load:**
```
⚠ VAE not on expected device! Expected cuda:0, got cpu
⚠ VAE not in fp16! Got torch.float32
```
→ Performance will suffer - check CUDA availability

#### Encoding Logs

**Force resize (common):**
```
Force resizing from 520x264 to 512x256 (configured resolution)
```
→ Normal - ensures dimensions match VAE expectations

**Auto-resize fallback:**
```
Image dimensions 520x264 not divisible by 8!
Auto-resizing to 520x264 for VAE compatibility
```
→ Automatic fix for incompatible dimensions

**Lower-res encoding (DEBUG level):**
```
Downsampled for interpolation (bilinear): (512, 256) → (256, 128)
```

**Mock latent (VAE not loaded):**
```
VAE not loaded - returning mock latent
```
→ **Warning:** Interpolation won't work properly

#### Decoding Logs

**Performance logging (slow decodes):**
```
[PERF] VAE decode: 125.3ms, latent: cuda:0, shape: torch.Size([1, 4, 32, 64])
```
→ Appears when decode takes > 50ms (expected for first few, concerning if persistent)

**GPU upscale (lower-res mode):**
```
GPU upscaled: (256, 128) → (512, 256) (bilinear)
```

**Triton fallback (important!):**
```
Compiled VAE decoder failed (likely missing Triton): RuntimeError
Falling back to uncompiled decoder...
Reloading VAE without torch.compile...
  [OK] VAE reloaded without compilation
```
→ One-time fallback, will be slower but functional. Common on Windows or when Triton not installed.

**Fallback failure (critical):**
```
Fallback decode also failed: [error]
```
→ VAE completely broken - check CUDA, VRAM

#### Batch Operations Logs (Most Common)

**VRAM monitoring (every 10th batch):**
```
VRAM before batch decode: 4.25GB allocated, 6.00GB reserved
```
→ DEBUG level, shows memory state before heavy operation

**Slow batch decode (performance alert):**
```
[PERF] Batch decode 8 frames: VAE=850ms, postproc=45ms, PIL=12ms, total=907ms (113ms/frame)
```
→ Appears when total > 200ms. Breakdown helps identify bottleneck:
- `VAE=` - GPU decode time (usually dominant)
- `postproc=` - GPU normalization and format conversion
- `PIL=` - CPU image creation

**Performance targets:**
| Metric | Good | Concerning | Bad |
|--------|------|------------|-----|
| VAE per frame | < 80ms | 80-150ms | > 150ms |
| Total per frame | < 120ms | 120-200ms | > 200ms |
| Batch of 8 | < 1s | 1-1.5s | > 1.5s |

#### Mock Operation Warnings

These indicate VAE isn't loaded (fallback mode):
```
VAE not loaded - returning mock latent
VAE not loaded - returning mock image
VAE not loaded - returning mock images
VAE not loaded - returning mock latents
```
→ Interpolation will use placeholder data - investigate initialization

#### ComfyUILatentEncoder Logs

If using ComfyUI-based encoding (alternative implementation):
```
Using ComfyUI API for VAE operations
ComfyUI VAE encoding not yet implemented
Consider using img2img workflows directly instead of manual VAE ops
```
→ This mode delegates VAE to ComfyUI rather than local operations

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| High `VAE=` time | GPU under load or VRAM pressure |
| `torch.compile` failed | Running without optimization - expect ~20% slower |
| Frequent `Force resizing` | Input images inconsistent - check generation settings |
| `⚠ VAE not on expected device` | CUDA issue - models on wrong device |
| No `[PERF]` logs | All decodes fast (good!) or logging disabled |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Very slow interpolation | `VAE=500ms+` per frame | Check GPU utilization, other processes |
| Interpolation artifacts | No errors but visual glitches | Resolution mismatch, check forced_resolution |
| VRAM growth | `reserved` increasing in logs | Missing `torch.cuda.empty_cache()` calls |
| Triton errors on first decode | `Compiled VAE decoder failed` | Normal on Windows - auto-recovers |
| OOM during batch decode | CUDA out of memory | Reduce batch size in config |
| Wrong colors | No specific log | Check VAE scale factor (should be 0.18215) |

#### Correlation with interpolation_worker.py

| latent_encoder.py Event | interpolation_worker.py Log |
|-------------------------|----------------------------|
| `[PERF] Batch decode 8 frames` | `[TIMING] Interpolation N->N+1 breakdown` |
| `Compiled VAE decoder failed` | Potentially slower `Decode all:` time |
| `VAE not loaded` | `Interpolation pair incomplete or failed` |

#### VRAM Management Notes

The encoder includes critical VRAM management:
```python
# At end of decode/decode_batch:
torch.cuda.empty_cache()
```

Without this, PyTorch reserves VRAM that ComfyUI needs. Watch for:
- Logs showing `reserved` much higher than `allocated`
- ComfyUI generation slowing down after many interpolations
- VRAM reported as full but `nvidia-smi` shows otherwise

---

## Priority 6: Cloud Infrastructure (Deep Investigation)

The cloud infrastructure enables Dream Gen to run on RunPod GPUs and stream frames to the blog via WebSocket. Understanding these logs is critical for debugging remote deployment issues, connection problems, and frame delivery failures.

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  RunPod GPU Pod                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ runpod_handler  │───▶│ DreamController  │───▶│ CloudFramePusher│ │
│  │ (entry point)   │    │ (generation)     │    │ (WebP encode)  │ │
│  └─────────────────┘    └──────────────────┘    └───────┬────────┘ │
│           │                      │                       │          │
│           │                      ▼                       │          │
│           │             ┌──────────────────┐             │          │
│           │             │ CloudStateSync   │             │          │
│           │             │ (periodic save)  │             │          │
│           │             └────────┬─────────┘             │          │
│           │                      │                       │          │
│           └──────────────────────┼───────────────────────┘          │
│                                  │                                  │
│                      ┌───────────▼─────────────┐                   │
│                      │ VPSWebSocketClient      │                   │
│                      │ (binary WS protocol)    │                   │
│                      └───────────┬─────────────┘                   │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │ wss://
                                   ▼
                    ┌─────────────────────────────┐
                    │ aetherawi.red/ws/gpu        │
                    │ (VPS Dreams Hub)            │
                    └─────────────────────────────┘
```

**Modes:**
- **Serverless** (legacy): ComfyUI + DreamGen in same container, job-based
- **Pod** (preferred): ComfyUI in separate pod, long-running process

---

### runpod_handler.py (117 log statements)

**Logger name:** `backend.cloud.runpod_handler`

**Purpose:** Entry point for RunPod execution. Handles ComfyUI lifecycle, VPS connection, and the main generation loop. Includes an activity watchdog for automatic restart on stalls.

#### Major Banners

| Banner | Meaning |
|--------|---------|
| `DREAM WINDOW RUNPOD HANDLER STARTING` | Handler initialization |
| `MODE: SERVERLESS\|POD` | Which architecture mode is active |
| `DREAMGEN POD MODE - Long Running Process` | Pod mode startup |

#### Startup Sequence Logs

**GPU info (important for debugging):**
```
GPU: NVIDIA A40
  VRAM: 48.0GB total, 47.5GB free
  Compute capability: 8.6
  CUDA version: 12.1
  cuDNN: 8902
```
→ **Action:** If CUDA not available, check RunPod template

**Environment logging:**
```
VPS WebSocket URL: wss://aetherawi.red/ws/gpu
Auth token: set
```

**Bootstrap mode (pod only):**
```
Bootstrap mode: Fetching secrets from admin panel (https://admin.aetherawi.red)...
Successfully fetched secrets from admin panel
Applied DREAM_GEN_AUTH_TOKEN from admin
Applied VPS_WEBSOCKET_URL: wss://aetherawi.red/ws/gpu
```
→ **Key:** Bootstrap allows minimal env vars at pod creation

#### ComfyUI Discovery Logs (Pod Mode)

```
Step 1: Discovering ComfyUI from VPS (pod mode)...
Discovering ComfyUI from VPS: https://aetherawi.red/api/dreams/comfyui
ComfyUI not registered yet, waiting... (15s)
ComfyUI discovered: http://10.0.1.23:8188
ComfyUI auth: comfyuser:****
Waiting for ComfyUI to be healthy: http://10.0.1.23:8188/system_stats
ComfyUI is healthy after 2.3s
```
→ **Key:** Pod mode waits for ComfyUI pod to register with VPS

**Discovery failures:**
```
VPS auth failed - check DREAM_GEN_AUTH_TOKEN
ComfyUI not registered with VPS after 300s
```

#### ComfyUI Local Start Logs (Serverless Mode)

```
Step 1: Starting ComfyUI locally (serverless mode)...
Starting ComfyUI from /app/ComfyUI...
ComfyUI args: python main.py --listen 127.0.0.1 --port 8188 --highvram ...
ComfyUI logs will be written to: /app/comfyui.log
ComfyUI process started (PID: 12345)
Waiting for ComfyUI... (20s)
ComfyUI ready after 45.2s
```

**Failure:**
```
ComfyUI failed to start within 120s
Failed to start ComfyUI: [error]
```
→ **Action:** Check `/app/comfyui.log` for ComfyUI-specific errors

#### VPS Connection Logs

```
Connecting to VPS...
[OK] Connected to VPS
[OK] Sent target FPS to VPS: 5.0
```

**Failure:**
```
Failed to connect to VPS
VPS connection failed
```

#### Activity Watchdog Logs

**Initialization:**
```
ActivityWatchdog initialized: timeout=90s, max_restarts=3
Watchdog heartbeat registered via frame pusher callback
Watchdog monitoring started
```

**Inactivity warnings:**
```
WATCHDOG: No activity for 45s (timeout: 90s)
WATCHDOG: No activity for 75s (timeout: 90s)
```
→ **Warning:** System may be stalling

**Restart triggered:**
```
WATCHDOG: No activity for 95s! Triggering restart (1/3)...
WATCHDOG RESTART: System stalled! Initiating full restart...
ComfyUI process killed
CUDA cache cleared
WATCHDOG RESTART: Restarting ComfyUI...
Cleared worker queues for fresh start
WATCHDOG RESTART: ComfyUI restarted, attempting to resume...
WATCHDOG: Restart completed, resuming monitoring
```

**Max restarts exceeded (critical):**
```
WATCHDOG: Max restarts (3) exceeded. System is unrecoverable. Stopping watchdog.
```
→ **Action:** Manual investigation required. Check underlying cause.

#### State Restoration Logs

```
Restoring state (245678 bytes)...
[OK] State restored: frame 15000
```

**Failure:**
```
Failed to restore state: [error]
```
→ Continues without state restoration

#### Shutdown Logs

```
Generation loop completed normally
Handler cleanup complete
```

**Cancelled:**
```
Generation cancelled
Handler cleanup complete
```

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| ComfyUI discovery timeout | `not registered with VPS after 300s` | Check ComfyUI pod status, VPS registration endpoint |
| Auth failures | `VPS auth failed`, `Bootstrap auth failed` | Check `DREAM_GEN_AUTH_TOKEN` or `POD_BOOTSTRAP_TOKEN` |
| Repeated watchdog restarts | Multiple `WATCHDOG: Triggering restart` | ComfyUI or network unstable |
| No GPU | `CUDA not available!` | Wrong RunPod template selected |
| Stuck at health check | `Waiting for ComfyUI health... (120s)` | ComfyUI not responding, check pod logs |

---

### websocket_client.py (19 log statements)

**Logger name:** `backend.cloud.websocket_client`

**Purpose:** Maintains persistent WebSocket connection to VPS. Handles binary protocol for frames, state, and control messages. Auto-reconnects on disconnection.

#### Binary Message Protocol

**GPU → VPS (MessageType):**
| Byte | Type | Payload |
|------|------|---------|
| `0x01` | FRAME | WebP image bytes |
| `0x02` | STATE | msgpack state bundle |
| `0x03` | HEARTBEAT | 8-byte timestamp |
| `0x04` | STATUS | JSON bytes |

**VPS → GPU (ControlType):**
| Byte | Command | Payload |
|------|---------|---------|
| `0x10` | PAUSE | none |
| `0x11` | RESUME | none |
| `0x12` | SAVE_STATE | none |
| `0x13` | SHUTDOWN | none |
| `0x14` | LOAD_STATE | state bytes |

#### Connection Logs

**Success:**
```
VPS WebSocket client initialized: wss://aetherawi.red/ws/gpu
Connecting to VPS: wss://aetherawi.red/ws/gpu
Connected to VPS successfully
```

**Failure:**
```
Failed to connect to VPS: Connection refused
```

#### Control Message Logs

```
Received PAUSE command from VPS
Received RESUME command from VPS
Received SAVE_STATE command from VPS
Received SHUTDOWN command from VPS
Received LOAD_STATE command from VPS (245678 bytes)
Unknown control message type: 255
```

#### Performance Logs

**Slow send warning:**
```
[PERF] Slow WS send: 125.3ms for 45.2KB (type=1)
```
→ **Key:** type=1 is FRAME. Slow sends indicate network congestion.

#### Reconnection Logs

```
Reconnection attempt 1 in 1.0s...
Reconnection attempt 2 in 2.0s...
Reconnection attempt 3 in 4.0s...
```
→ Exponential backoff up to 60s max

#### Error Logs

```
Failed to send message: WebSocket connection closed
Receive error: Connection closed
Error closing WebSocket: [error]
Error handling control message: [error]
Heartbeat error: [error]
```

#### Disconnection Logs

```
Disconnected from VPS
```

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| Connection drops | Frequent `Reconnection attempt` | Check network, VPS status |
| Slow sends | `[PERF] Slow WS send: >100ms` | Network congestion, reduce frame rate |
| Auth failure | `Failed to connect` immediately | Check auth token |
| No heartbeat | Long gaps between sends | Connection may be dead |

---

### frame_pusher.py (9 log statements)

**Logger name:** `backend.cloud.frame_pusher`

**Purpose:** Encodes PIL Images to WebP and pushes via WebSocket. Tracks push statistics and timing.

#### Initialization Logs

```
CloudFramePusher initialized: format=webp, quality=85
Push callback registered
```

#### Push Success Logs (DEBUG level)

```
Pushed frame 1234: 42.3KB (encode: 8.5ms, push: 15.2ms)
```

#### Performance Warning Logs (> 100ms)

```
[PERF] Slow frame push 1234: 156.3ms total (encode=12.5ms, network=143.8ms, size=65.2KB)
```
→ **Key:** High `network=` indicates WS congestion

#### Failure Logs

```
Cannot push frame: not connected to VPS
Failed to push frame: [error]
Push callback failed: [error]
```

#### Statistics Methods

Call `frame_pusher.get_stats()` for:
```python
{
    "frames_pushed": 15000,
    "keyframes_pushed": 250,
    "interpolations_pushed": 14750,
    "bytes_pushed": 645000000,
    "bytes_pushed_mb": 615.23,
    "average_push_time_ms": 23.5,
    "average_frame_size_kb": 43.0,
    "format": "webp",
    "quality": 85
}
```

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| No frames pushing | No push logs | Check WS connection |
| Large frames | `size=100KB+` | Reduce quality setting |
| Slow pushes | `[PERF] Slow frame push` | Network or VPS issue |

---

### state_sync.py (10 log statements)

**Logger name:** `backend.cloud.state_sync`

**Purpose:** Periodic state snapshots for resume capability. Pushes every N keyframes and on shutdown.

#### Initialization Logs

```
CloudStateSync initialized: interval=10 keyframes
msgpack not available, using JSON for state serialization (less efficient)
```
→ **Warning:** JSON fallback is ~3x larger than msgpack

#### State Push Logs

**Success:**
```
Pushed state snapshot: 125.3KB (keyframe 150, 45.2ms)
```

**Shutdown push:**
```
Pushing final state on shutdown...
Pushed state snapshot: 256.7KB (keyframe 250, 89.3ms)
```

#### Warning Logs

```
Cannot push state: not connected to VPS
No state to push
Could not get cache metadata: [error]
Could not serialize embeddings: [error]
```

#### Failure Logs

```
Failed to push state: [error]
```

#### State Bundle Contents

```python
{
    "latent": bytes,           # VAE latent for interpolation continuity
    "latent_shape": [1, 4, 64, 64],
    "latent_dtype": "float16",
    "state": {                  # Generation counters
        "frame_count": 15000,
        "keyframe_count": 250,
        "theme_index": 3
    },
    "timestamp": 1706123456.789,
    "keyframe_count": 250,
    "cache_meta": {...},        # Only on shutdown
    "embeddings": bytes         # Only on shutdown
}
```

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| State too large | `500KB+` snapshots | Reduce embeddings, check cache |
| Push failures | `Failed to push state` | Check WS connection |
| No msgpack | `using JSON` warning | Install msgpack package |

---

## Priority 7: Prompt System (Deep Investigation)

The prompt system generates billions of unique visual prompts through template + component combination. Understanding these logs is critical for debugging visual variety issues, collapse prevention (via mutations), and aesthetic transitions.

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ CombinatorialPromptSystem                                           │
│                                                                     │
│  ┌──────────────────┐   ┌────────────────────┐                     │
│  │ templates.yaml   │   │ components.yaml    │                     │
│  │ - id             │   │ - word             │                     │
│  │ - structure      │   │ - embedding        │                     │
│  │ - slots          │   │ - opposite         │                     │
│  └────────┬─────────┘   └─────────┬──────────┘                     │
│           │                       │                                 │
│           ▼                       ▼                                 │
│  ┌────────────────────────────────────────────┐                    │
│  │ Prompt Generation                          │                    │
│  │  "{color} {mood} {setting}" + components   │                    │
│  │  → "cyan ethereal crystalline void"        │                    │
│  └────────────────────────────────────────────┘                    │
│           │                                                         │
│           ▼                                                         │
│  ┌────────────────────────────────────────────┐                    │
│  │ State Machine: DRIFT ↔ BEND                │                    │
│  │  DRIFT: Low denoise (0.20), stable         │                    │
│  │  BEND:  High denoise (0.45), after mutation│                    │
│  └────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key concepts:**
- **Template**: Structure with `{category}` slots (e.g., `"{color} {mood} landscape"`)
- **Component**: Word + embedding + semantic opposite (e.g., `warm → cold`)
- **Mutation**: Change one component to introduce visual variety
- **DRIFT mode**: Normal low-denoise generation
- **BEND mode**: High-denoise after mutation (helps prompt changes "stick")

---

### combinatorial.py (35 log statements)

**Logger name:** `backend.prompts.combinatorial`

**Purpose:** Manages combinatorial prompt generation with templates, component pools, mutations, and DRIFT/BEND state machine.

#### Initialization Logs

```
CombinatorialPromptSystem initialized
  Templates: 12
  Component categories: ['color', 'mood', 'adjective', 'style', 'setting']
  Mutation probability: 12.0%
Initialized with template 'dissolution'
```

**Component loading (DEBUG):**
```
Loaded components: color=25, mood=20, adjective=30, style=15, setting=10
Loaded 100 embeddings from components_embeddings.npz
No embeddings loaded (using random selection for mutations)
```

**Missing file errors:**
```
Templates file not found: prompts/templates.yaml
Components file not found: prompts/components.yaml
```
→ **Critical:** System cannot start without these files

#### DRIFT/BEND State Machine Logs

**Entering BEND (after mutation):**
```
[MUTATE] color: 'cyan' → 'amber' | mutation #5 | DRIFT → BEND (4f)
```
→ Format: `category: 'old' → 'new' | count | state transition (duration)`

**BEND mode active (DEBUG):**
```
[STATE] BEND mode active, 3 frames remaining
```

**Exiting BEND:**
```
[STATE] BEND → DRIFT (frame=150, mutations=5)
```

#### Mutation Logs

**Mutation check (DEBUG):**
```
[MUTATION_CHECK] roll=0.087 vs prob=0.120 → MUTATE (frames_since=12)
[MUTATION_CHECK] roll=0.543 vs prob=0.120 → SKIP (frames_since=8)
```

**Forced mutation (staleness):**
```
[MUTATION_CHECK] STALE - forcing mutation (frames_since=25, threshold=25)
```

**Mutation result:**
```
[MUTATE] mood: 'ethereal' → 'vibrant' | mutation #6 | DRIFT → BEND (4f)
[MUTATE] New prompt: cyan vibrant crystalline structures dissolving...
```

#### Similarity-Guided Selection Logs

**Successful similarity selection:**
```
[SIMILARITY] 'ethereal' → 'delicate' (sim=0.583, target=0.55, candidates=8/20)
```
→ `sim`: Actual similarity to selected component
→ `target`: Desired similarity (configured)
→ `candidates`: Components in acceptable range / total pool

**Top candidates (DEBUG):**
```
[SIMILARITY] Top candidates: [('delicate', '0.583'), ('subtle', '0.567'), ('gentle', '0.542')]
```

**No candidates in range (fallback to random):**
```
[SIMILARITY] No candidates in range [0.30, 0.80] - random fallback: 'bold'
```

#### Force Mutation (Collapse Response) Logs

Called by `ModeCollapseDetector` for soft intervention:

```
[FORCE_MUTATE] Found exact opposite: 'warm' → 'cold'
[FORCE_MUTATE] color: 'warm' → 'cold' (opposite: warm)
[FORCE_MUTATE] mood: 'bright' → 'shadowy' (opposite: bright)
[FORCE_MUTATE] Replaced 2 components, entering BEND mode (4f)
[FORCE_MUTATE] New prompt: cold shadowy crystalline structures...
```

**No categories to mutate:**
```
[FORCE_MUTATE] No categories available to mutate
```

#### Template Switch Logs

```
[TEMPLATE_SWITCH] 'dissolution' → 'architectural'
```
→ Resets mutation tracking and exits BEND mode

#### Statistics (via `get_stats()`)

```python
{
    "current_template": "dissolution",
    "current_components": {"color": "cyan", "mood": "ethereal", ...},
    "total_frames": 1500,
    "total_mutations": 45,
    "frames_since_mutation": 8,
    "in_bend_mode": False,
    "bend_frames_remaining": 0,
    "templates_available": 12,
    "mutation_probability": 0.12
}
```

#### Key Debug Points

| Log Pattern | What It Indicates |
|-------------|-------------------|
| Low `candidates` count | Pool too small or similarity thresholds too narrow |
| High `frames_since_mutation` | Mutations not triggering (check probability) |
| `random fallback` frequent | Embeddings not loaded or thresholds wrong |
| No `[MUTATE]` logs | Check `should_mutate()` calls in orchestrator |
| Always in BEND | `bend_duration` too long or mutations too frequent |

#### Common Issues

| Issue | Log Pattern | Resolution |
|-------|-------------|------------|
| No visual variety | No `[MUTATE]` logs | Increase mutation probability |
| Too jarring changes | Frequent mutations | Decrease probability, use similarity |
| No similarity selection | `random fallback` always | Load embeddings, check npz file |
| Wrong template | Unexpected `current_template` | Check switch logic |
| Stuck in BEND | `BEND mode active` persists | Check frame counter, bend_duration |

---

## Cloud + Prompt System Correlation Table

| Scenario | Cloud Logs | Prompt Logs |
|----------|------------|-------------|
| **Template switch** | `[TEMPLATE_SWITCH]` in orchestrator | `[TEMPLATE_SWITCH]` in combinatorial |
| **Collapse intervention** | `[COLLAPSE_RESPONSE]` | `[FORCE_MUTATE]` |
| **Fresh frame injection** | `[FRESH] Consumed 'X'` | New template + components |
| **State restore** | `State restored` | May reset prompt state |
| **Watchdog restart** | `WATCHDOG RESTART` | Prompt system re-initialized |

---

## Appendix: Quick Reference Cards

### A. Error Severity Guide

| Severity | Example Patterns | Action |
|----------|------------------|--------|
| **Critical** | `CUDA not available`, `Max restarts exceeded`, `file not found` | Stop, investigate immediately |
| **Error** | `Failed to connect`, `Generation error`, `[FAIL]` | Check logs, may auto-recover |
| **Warning** | `[PERF] Slow`, `queue near capacity`, `Reconnection attempt` | Monitor, tune if persistent |
| **Info** | `[OK]`, `[MUTATE]`, `Pushed frame` | Normal operation |

### B. Common grep Patterns

```bash
# === CRITICAL ISSUES ===
grep -E "CRITICAL|FATAL|Max restarts|unrecoverable" logs/*.log

# === ALL ERRORS ===
grep -E "\[FAIL\]|ERROR|Failed|error:" logs/*.log

# === WATCHDOG ACTIVITY ===
grep -E "WATCHDOG" logs/*.log

# === TEMPLATE SWITCHES ===
grep -E "TEMPLATE_SWITCH|\[FRESH\] Consumed" logs/*.log

# === MUTATION ACTIVITY ===
grep -E "\[MUTATE\]|\[FORCE_MUTATE\]|DRIFT.*BEND" logs/*.log

# === CLOUD CONNECTION ===
grep -E "Connected to VPS|Reconnection|Disconnected" logs/*.log

# === PERFORMANCE ===
grep -E "\[PERF\]|Slow|queue depth high" logs/*.log

# === COLLAPSE DETECTION ===
grep -E "\[COLLAPSE\]|\[CONVERGING\]|\[WARMUP" logs/*.log
```

### C. Log File Locations Summary

| Component | File | Rotation |
|-----------|------|----------|
| Daemon | `logs/daemon.log` | None |
| Controller | `logs/dream_controller.log` | 5MB × 4 |
| ComfyUI (cloud) | `/app/comfyui.log` | None |

---

## Document Revision History

- **v1.0** (2026-01-24): Initial document structure and file categorization
- **v2.0** (2026-01-24): Completed all priority investigations (1-7)
- **v2.1** (2026-01-24): Cleanup - removed scaffolding, finalized document

