# Dream Gen Calibration Suite

Tools for establishing similarity baselines and optimizing intervention thresholds.

## Quick Start

### On an Existing RunPod

```bash
# SSH into your pod
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519

# Clone/update the repo
cd /workspace
git clone https://github.com/LuxiaSL/dream_gen.git  # or git pull

# Run calibration
cd dream_gen/calibration
chmod +x run_calibration.sh
./run_calibration.sh broad 500
```

### Analyze Existing Keyframes

```bash
# Without ComfyUI - analyze local keyframes
./run_calibration.sh analyze /path/to/keyframes
```

## Calibration Modes

| Mode | Frames | Purpose | Output |
|------|--------|---------|--------|
| `broad` | 500 | Survey all templates | Per-template similarity ranges |
| `deep` | 500 | Single template drift | Convergence rate, mutation effect |
| `intervention` | 100 | Effect sizes | ΔColorHist, ΔpHash per intervention |
| `full` | 1500 | Complete suite | All of the above |
| `analyze` | - | Existing frames | Threshold recommendations |

## Optimization Targets

For healthy visual variety in a 500-keyframe sprint (~25 min playback):

| Intervention | Target Interval | Expected Per Sprint |
|--------------|-----------------|---------------------|
| Template switch | 100 keyframes | ~5 switches |
| Cache injection | 33-50 keyframes | ~10-15 injections |
| Mutation | 7-18 keyframes | ~28-42 mutations |

## Output Files

```
/workspace/calibration/
├── calibration_broad_YYYYMMDD_HHMMSS.json    # Template survey results
├── calibration_deep_YYYYMMDD_HHMMSS.json     # Drift analysis results
├── calibration_intervention_YYYYMMDD_HHMMSS.json  # Effect sizes
├── output/                                    # Generated keyframes
│   ├── broad_0001_material_study.png
│   ├── deep_0001.png
│   └── ...
└── logs/                                      # Debug logs
```

## Using Results

The calibration produces recommended config values:

```yaml
cache:
  color_histogram:
    diversity_threshold: 1.84
    dissimilarity_range: [0.97, 2.06]
    convergence_threshold: 0.75
    force_cache_threshold: 1.5
  phash:
    diversity_threshold: 0.61
    dissimilarity_range: [0.47, 0.62]
    convergence_threshold: 0.11
    force_cache_threshold: 0.23
```

Copy these to your `config.pod.yaml` or `config.cloud.yaml`.

## Key Findings from Initial Calibration

Based on analysis of 378 keyframes (71,253 pairwise comparisons):

### pHash Range Was Completely Wrong

```
Default config:      [0.68, 0.92]
Actual distribution: P25=0.47, P75=0.56, P90=0.62
Recommended:         [0.47, 0.62]
```

The default range excluded >90% of frame pairs, causing single-frame injection looping.

### Consecutive Frame Signature

| Metric | All Pairs | Consecutive |
|--------|-----------|-------------|
| ColorHist | 1.37 ± 0.50 | 2.46 ± 0.48 |
| pHash | 0.53 ± 0.09 | 0.84 ± 0.16 |

Consecutive frames are naturally much more similar. Collapse detection thresholds
should be tuned against consecutive similarity, not all-pairs.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATION_OUTPUT_DIR` | `/workspace/calibration` | Output directory |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |

## Manual Python Usage

```python
from calibration_benchmark import CalibrationBenchmark

benchmark = CalibrationBenchmark(
    config_path="calibration/config.calibration.yaml",
    output_dir="/workspace/calibration"
)

# Analyze existing frames
results = benchmark.analyze_existing(Path("/path/to/keyframes"))
benchmark.save_results(results)

# Or run generation (requires ComfyUI)
results = await benchmark.run_broad(num_frames=500)
```

