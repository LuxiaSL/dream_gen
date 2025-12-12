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


async def start_comfyui() -> bool:
    """
    Start ComfyUI server and wait for it to be ready.
    
    Returns:
        True if ComfyUI started successfully
    """
    import subprocess
    import aiohttp
    
    comfyui_path = os.environ.get("COMFYUI_PATH", "/app/ComfyUI")
    comfyui_port = 8188
    startup_timeout = 120  # seconds
    
    logger.info(f"Starting ComfyUI from {comfyui_path}...")
    
    # Start ComfyUI in background
    try:
        process = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", str(comfyui_port)],
            cwd=comfyui_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(f"ComfyUI process started (PID: {process.pid})")
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        return False
    
    # Wait for ComfyUI to be ready
    health_url = f"http://127.0.0.1:{comfyui_port}/system_stats"
    start_time = asyncio.get_event_loop().time()
    
    async with aiohttp.ClientSession() as session:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > startup_timeout:
                logger.error(f"ComfyUI failed to start within {startup_timeout}s")
                process.terminate()
                return False
            
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info(f"ComfyUI ready after {elapsed:.1f}s")
                        return True
            except Exception:
                pass  # Still starting up
            
            await asyncio.sleep(2)
            if int(elapsed) % 10 == 0:
                logger.info(f"Waiting for ComfyUI... ({elapsed:.0f}s)")


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
    
    # Log build info for debugging deployment issues
    build_info_path = Path("/app/BUILD_INFO")
    if build_info_path.exists():
        build_info = build_info_path.read_text().strip()
        for line in build_info.split('\n'):
            logger.info(f"  {line}")
    else:
        logger.warning("BUILD_INFO not found - using development build")
    
    # Log GPU info for performance debugging
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            compute_cap = torch.cuda.get_device_capability(0)
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"  VRAM: {gpu_mem_total:.1f}GB total, {gpu_mem_free:.1f}GB free")
            logger.info(f"  Compute capability: {compute_cap[0]}.{compute_cap[1]}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        else:
            logger.warning("CUDA not available!")
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    logger.info(f"VPS WebSocket URL: {vps_websocket_url}")
    
    # Use auth token from param, or fall back to environment variable
    if not auth_token:
        auth_token = os.environ.get("DREAM_GEN_AUTH_TOKEN")
        if auth_token:
            logger.info("Using auth token from environment variable")
    
    logger.info(f"Auth token: {'set' if auth_token else 'NOT SET'}")
    
    # Step 1: Start ComfyUI server
    logger.info("Step 1: Starting ComfyUI...")
    if not await start_comfyui():
        return {"status": "error", "error": "Failed to start ComfyUI"}
    logger.info("ComfyUI is running!")
    
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
        # Use cloud config if available, otherwise fall back to default
        config_path = os.environ.get("CONFIG_PATH", "backend/config.cloud.yaml")
        if not Path(config_path).exists():
            config_path = "backend/config.yaml"
            logger.info(f"Cloud config not found, using default: {config_path}")
        else:
            logger.info(f"Using cloud config: {config_path}")
        
        controller = DreamController(config_path=config_path)
        
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
        # Get from job input, or fall back to environment variables
        vps_url = job_input.get("vps_websocket_url") or os.environ.get("VPS_WEBSOCKET_URL")
        auth_token = job_input.get("auth_token") or os.environ.get("DREAM_GEN_AUTH_TOKEN")
        state = job_input.get("state")
        
        logger.info(f"VPS URL: {vps_url}")
        logger.info(f"Auth token set: {bool(auth_token)}")
        
        if not vps_url:
            return {"status": "error", "error": "vps_websocket_url required (set VPS_WEBSOCKET_URL env var or provide in job input)"}
        
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

