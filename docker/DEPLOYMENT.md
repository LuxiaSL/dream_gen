# DreamGen Deployment Guide

## Current Deployment: Heimdall on B200

DreamGen runs on dedicated B200 GPU hardware via Heimdall job scheduler.
Entry point: `backend/cloud/heimdall_entry.py`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VPS_WEBSOCKET_URL` | VPS for frame streaming | `wss://aetherawi.red/ws/gpu` |
| `DREAM_GEN_AUTH_TOKEN` | Auth token for VPS | None |
| `CONFIG_PATH` | Config file path | `backend/config.b200.yaml` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### Seedless Operation

DreamGen can start without seed images:

1. If `seeds/` directory has images, uses random seed
2. If `seeds/` is empty, generates first frame via txt2img

## Docker (Calibration Only)

The remaining `Dockerfile.calibration` builds a standalone image for
running calibration benchmarks.

```bash
docker build -f docker/Dockerfile.calibration -t luxiasl/dreamgen-calibration:latest .
```
