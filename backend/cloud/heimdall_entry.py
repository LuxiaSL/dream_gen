"""
Heimdall Entry Point — Direct Diffusers Deployment

Simplified entry point for running Dream Window on dedicated GPU hardware
(e.g., B200 via Heimdall job scheduler). No ComfyUI.

Usage:
    python -m cloud.heimdall_entry [--config CONFIG] [--vps-url URL] [--auth-token TOKEN]

Environment variables:
    VPS_WEBSOCKET_URL   — WebSocket URL for VPS streaming (default: wss://aetherawi.red/ws/gpu)
    DREAM_GEN_AUTH_TOKEN — Auth token for VPS connection
    CONFIG_PATH          — Path to config YAML
    CUDA_VISIBLE_DEVICES — GPU selection (set by Heimdall)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import yaml

# Add paths for imports — the codebase uses mixed conventions:
# some files: `from core.X` / `from utils.X` (relative to backend/)
# some files: `from backend.core.X` (relative to repo root)
_backend_dir = str(Path(__file__).resolve().parent.parent)
_repo_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from core.dream_controller import DreamController

logger = logging.getLogger("heimdall_entry")


async def run_direct_generation(
    config_path: str,
    vps_websocket_url: str,
    auth_token: str | None = None,
) -> dict:
    """
    Main generation loop for direct diffusers deployment.

    Loads config with backend: "direct", creates DreamController,
    connects to VPS, and runs the hybrid generation loop.
    """
    controller = None
    vps_client = None

    try:
        # Load and override config
        logger.info(f"Loading config: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Force direct backend
        config.setdefault("system", {})["backend"] = "direct"

        # Enable cloud streaming
        config.setdefault("cloud", {})
        config["cloud"]["enabled"] = True
        config["cloud"]["vps_websocket_url"] = vps_websocket_url
        if auth_token:
            config["cloud"]["auth_token"] = auth_token

        # Disable game detection (headless server)
        config.setdefault("game_detection", {})["enabled"] = False

        # Write overridden config to temp file for DreamController
        temp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        )
        yaml.dump(config, temp_config)
        temp_config.close()
        temp_config_path = temp_config.name

        try:
            controller = DreamController(config_path=temp_config_path)
        finally:
            try:
                os.unlink(temp_config_path)
            except OSError:
                pass

        # Connect to VPS
        vps_client = controller.vps_client
        if not vps_client:
            logger.error("VPS client not initialized — check cloud config")
            return {"status": "error", "error": "VPS client not initialized"}

        # Set up reconnection callbacks
        async def on_vps_reconnected():
            logger.info("VPS reconnected — re-sending FPS config")
            try:
                target_fps = config.get("generation", {}).get("hybrid", {}).get(
                    "target_interpolation_fps", 5.0,
                )
                fps_msg = json.dumps({"target_fps": target_fps}).encode("utf-8")
                await vps_client.send_status(fps_msg)
            except Exception as e:
                logger.warning(f"Failed to re-send FPS config: {e}")

        async def on_vps_disconnected():
            logger.warning("VPS disconnected — frames will queue")

        async def on_vps_reconnecting(attempt: int, url: str):
            logger.info(f"VPS reconnecting (attempt {attempt})")

        vps_client.set_lifecycle_callbacks(
            on_disconnected=on_vps_disconnected,
            on_reconnecting=on_vps_reconnecting,
            on_reconnected=on_vps_reconnected,
        )

        logger.info("Connecting to VPS...")
        connected = await vps_client.connect()
        if not connected:
            logger.error("Failed to connect to VPS")
            return {"status": "error", "error": "VPS connection failed"}
        logger.info("[OK] Connected to VPS")

        # Send target FPS
        target_fps = config.get("generation", {}).get("hybrid", {}).get(
            "target_interpolation_fps", 5.0,
        )
        fps_msg = json.dumps({"target_fps": target_fps}).encode("utf-8")
        await vps_client.send_status(fps_msg)
        logger.info(f"[OK] Target FPS sent: {target_fps}")

        # Run generation loop
        logger.info("Starting generation loop...")
        controller.running = True

        # Hook watchdog heartbeat if frame pusher exists
        if hasattr(controller, "frame_pusher") and controller.frame_pusher:
            logger.info("Frame pusher available for streaming")

        await controller.run_buffered_hybrid_loop()

        return {"status": "completed", "frames": controller.generator.frame_count}

    except asyncio.CancelledError:
        logger.info("Generation cancelled (signal received)")
        return {"status": "cancelled"}
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        if controller:
            try:
                controller.stop()
            except Exception as e:
                logger.warning(f"Controller stop error: {e}")
        if vps_client:
            try:
                await vps_client.disconnect()
            except Exception as e:
                logger.warning(f"VPS disconnect error: {e}")


async def main():
    """CLI entry point for Heimdall / manual execution."""
    parser = argparse.ArgumentParser(description="Dream Window — Direct Backend")
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", "backend/config.cloud.yaml"),
        help="Config YAML path",
    )
    parser.add_argument(
        "--vps-url",
        default=os.environ.get("VPS_WEBSOCKET_URL", "wss://aetherawi.red/ws/gpu"),
        help="VPS WebSocket URL",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("DREAM_GEN_AUTH_TOKEN"),
        help="VPS auth token",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Quiet noisy loggers
    for name in ("urllib3", "websockets", "PIL", "diffusers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("Dream Window — Direct Diffusers Backend")
    logger.info("=" * 60)
    logger.info(f"Config:    {args.config}")
    logger.info(f"VPS:       {args.vps_url}")
    logger.info(f"Auth:      {'set' if args.auth_token else 'not set'}")
    logger.info(f"GPU:       {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # Signal handling for graceful shutdown (Heimdall sends SIGTERM on preemption)
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig} — shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run generation with shutdown monitoring
    gen_task = asyncio.create_task(
        run_direct_generation(args.config, args.vps_url, args.auth_token)
    )

    # Wait for either completion or shutdown signal
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    done, pending = await asyncio.wait(
        {gen_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if gen_task in done:
        result = gen_task.result()
    else:
        result = {"status": "shutdown"}

    logger.info(f"Exit: {result}")


if __name__ == "__main__":
    asyncio.run(main())
