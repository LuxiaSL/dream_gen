"""
RunPod Serverless Handler

This module wraps the DreamController for RunPod serverless execution.
It handles:
- Job initialization with state restoration
- WebSocket connection to VPS
- Streaming frames during generation
- Graceful shutdown with state persistence

Usage on RunPod:
    The handler is automatically invoked when a job is submitted.
    It connects to the VPS, restores state if provided, and begins
    streaming frames until shutdown is requested.

Local Testing:
    python -m backend.cloud.runpod_handler --local
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for RunPod environment"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )


async def run_dream_generation(
    vps_websocket_url: str,
    auth_token: Optional[str] = None,
    initial_state: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Main generation loop for RunPod
    
    Connects to VPS, restores state if provided, and streams frames
    until shutdown is requested.
    
    Args:
        vps_websocket_url: WebSocket URL to connect to VPS
        auth_token: Optional authentication token
        initial_state: Optional state bytes to restore from
    
    Returns:
        Final status dictionary
    """
    # Import here to avoid loading heavy dependencies before needed
    # Use absolute imports from /app (PYTHONPATH root)
    from backend.core.dream_controller import DreamController
    from backend.cloud import VPSWebSocketClient, CloudFramePusher, CloudStateSync
    from backend.cloud.state_sync import deserialize_state
    
    logger.info("=" * 60)
    logger.info("DREAM WINDOW RUNPOD HANDLER STARTING")
    logger.info("=" * 60)
    logger.info(f"VPS WebSocket URL: {vps_websocket_url}")
    
    # Override config for cloud mode
    config_overrides = {
        'cloud': {
            'enabled': True,
            'vps_websocket_url': vps_websocket_url,
            'auth_token': auth_token,
        }
    }
    
    controller = None
    vps_client = None
    
    try:
        # Initialize controller with cloud mode
        logger.info("Initializing DreamController...")
        controller = DreamController(config_path="backend/config.yaml")
        
        # Cloud mode should already be enabled via config, but ensure it
        if not controller.cloud_enabled:
            logger.warning("Cloud mode not enabled in config, enabling now...")
            controller.cloud_enabled = True
            controller._init_cloud_mode()
        
        # Connect to VPS
        vps_client = controller.vps_client
        if vps_client:
            logger.info("Connecting to VPS...")
            connected = await vps_client.connect()
            if not connected:
                logger.error("Failed to connect to VPS")
                return {"status": "error", "error": "VPS connection failed"}
            logger.info("[OK] Connected to VPS")
        else:
            logger.error("VPS client not initialized")
            return {"status": "error", "error": "VPS client not initialized"}
        
        # Restore state if provided
        if initial_state:
            logger.info(f"Restoring state ({len(initial_state)} bytes)...")
            try:
                state_bundle = deserialize_state(initial_state)
                await controller._on_cloud_load_state(initial_state)
                logger.info(f"[OK] State restored: frame {state_bundle.get('state', {}).get('frame_count', '?')}")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
                # Continue without state restoration
        
        # Run generation loop
        logger.info("Starting generation loop...")
        controller.running = True
        
        # Start buffered hybrid loop (async)
        await controller.run_buffered_hybrid_loop()
        
        logger.info("Generation loop completed")
        
        return {
            "status": "completed",
            "frames_generated": controller.frame_count,
        }
    
    except asyncio.CancelledError:
        logger.info("Generation cancelled")
        return {"status": "cancelled"}
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    
    finally:
        # Cleanup
        if controller:
            controller.stop()
        
        if vps_client:
            await vps_client.disconnect()
        
        logger.info("Handler cleanup complete")


# ==================== RunPod Integration ====================

try:
    import runpod
    HAS_RUNPOD = True
    print(f"[DEBUG] runpod module loaded successfully: {runpod.__version__}")
except ImportError as e:
    HAS_RUNPOD = False
    print(f"[DEBUG] runpod import failed: {e}")
except Exception as e:
    HAS_RUNPOD = False
    print(f"[DEBUG] runpod import error: {type(e).__name__}: {e}")


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler entry point (ASYNC)
    
    Called by RunPod when a job is submitted.
    RunPod already runs an event loop, so this must be async.
    
    Args:
        job: Job dictionary with 'id' and 'input' keys
    
    Returns:
        Result dictionary
    """
    job_input = job.get("input", {})
    job_type = job_input.get("type", "stream")
    
    logger.info(f"RunPod job received: type={job_type}, id={job.get('id')}")
    
    if job_type == "start" or job_type == "stream":
        # Main streaming job
        vps_url = job_input.get("vps_websocket_url")
        auth_token = job_input.get("auth_token")
        state = job_input.get("state")
        
        if not vps_url:
            return {"status": "error", "error": "vps_websocket_url required"}
        
        # Already in async context - just await directly
        result = await run_dream_generation(
            vps_websocket_url=vps_url,
            auth_token=auth_token,
            initial_state=state,
        )
        
        return result
    
    elif job_type == "health":
        # Health check
        return {"status": "healthy", "message": "Dream Window handler ready"}
    
    else:
        return {"status": "error", "error": f"Unknown job type: {job_type}"}


async def async_generator_handler(job: Dict[str, Any]):
    """
    Async generator handler for streaming results
    
    RunPod can use this for streaming responses.
    """
    job_input = job.get("input", {})
    vps_url = job_input.get("vps_websocket_url")
    
    if not vps_url:
        yield {"status": "error", "error": "vps_websocket_url required"}
        return
    
    # For now, delegate to regular handler
    # Future: Could stream intermediate status updates
    yield {"status": "starting", "message": "Initializing..."}
    
    result = await run_dream_generation(
        vps_websocket_url=vps_url,
        auth_token=job_input.get("auth_token"),
        initial_state=job_input.get("state"),
    )
    
    yield result


# ==================== Local Testing ====================

async def local_test_mode():
    """Run handler in local test mode (no RunPod)"""
    logger.info("Running in local test mode")
    
    # Get VPS URL from environment or use default
    vps_url = os.environ.get("VPS_WEBSOCKET_URL", "ws://localhost:8000/ws/gpu")
    auth_token = os.environ.get("DREAM_GEN_AUTH_TOKEN")
    
    logger.info(f"VPS URL: {vps_url}")
    
    # Simulate a job
    result = await run_dream_generation(
        vps_websocket_url=vps_url,
        auth_token=auth_token,
        initial_state=None,
    )
    
    logger.info(f"Result: {result}")


def main():
    """Entry point for both RunPod and local testing"""
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    
    # Debug info
    print(f"[DEBUG] sys.argv: {sys.argv}")
    print(f"[DEBUG] HAS_RUNPOD: {HAS_RUNPOD}")
    print(f"[DEBUG] '--local' in argv: {'--local' in sys.argv}")
    
    # Check for local test mode
    if "--local" in sys.argv or not HAS_RUNPOD:
        logger.info("Starting in local test mode...")
        asyncio.run(local_test_mode())
    else:
        # Start RunPod serverless handler
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({
            "handler": handler,
            # Optional: Use async generator for streaming
            # "handler": async_generator_handler,
        })


if __name__ == "__main__":
    main()

