"""
RunPod Handler - Supports both Serverless and Pod architectures

This module is the entry point for DreamGen on RunPod.

Architectures supported:
1. SERVERLESS (legacy): Single container with ComfyUI + DreamGen
   - Starts ComfyUI locally
   - Uses runpod.serverless.start() for job handling
   - ComfyUI at localhost:8188

2. POD (two-pod): Separate ComfyUI and DreamGen pods
   - Does NOT start ComfyUI (it's in another pod)
   - Discovers ComfyUI endpoint from VPS
   - Runs as long-running process (not job-based)
   - Supports basic auth for ComfyUI nginx proxy

Mode detection:
- Set DREAMGEN_MODE=pod to use pod mode
- Default is serverless mode for backwards compatibility

Usage:
    # Serverless (default)
    python -m backend.cloud.runpod_handler

    # Pod mode
    DREAMGEN_MODE=pod python -m backend.cloud.runpod_handler

    # Local testing
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

# Architecture mode: 'serverless' or 'pod'
DREAMGEN_MODE = os.environ.get("DREAMGEN_MODE", "serverless").lower()

# Global handle for ComfyUI log file (prevents garbage collection)
_comfyui_log_file = None

# Global watchdog instance
_watchdog = None


class ActivityWatchdog:
    """
    Monitors system activity and triggers restart on prolonged inactivity.
    
    The watchdog tracks the last time any meaningful activity occurred
    (frame pushed, keyframe generated, etc.). If no activity happens for
    the timeout period, it triggers a full system restart.
    
    Usage:
        watchdog = ActivityWatchdog(timeout_seconds=90, max_restarts=3)
        watchdog.heartbeat()  # Call whenever activity occurs
        await watchdog.run(restart_callback)  # Background monitoring task
    """
    
    def __init__(self, timeout_seconds: float = 90.0, max_restarts: int = 3):
        """
        Initialize the watchdog.
        
        Args:
            timeout_seconds: Seconds of inactivity before triggering restart
            max_restarts: Maximum restart attempts before giving up
        """
        import time
        self.timeout = timeout_seconds
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.last_activity = time.time()
        self._running = False
        self._check_interval = 10.0  # Check every 10 seconds
        
        logger.info(f"ActivityWatchdog initialized: timeout={timeout_seconds}s, max_restarts={max_restarts}")
    
    def heartbeat(self):
        """
        Signal that activity occurred. Call this from frame_pusher, 
        keyframe_worker, or interpolation_worker whenever work completes.
        """
        import time
        self.last_activity = time.time()
    
    def time_since_activity(self) -> float:
        """Get seconds since last activity."""
        import time
        return time.time() - self.last_activity
    
    def is_stalled(self) -> bool:
        """Check if system has been inactive for too long."""
        return self.time_since_activity() > self.timeout
    
    def can_restart(self) -> bool:
        """Check if we haven't exceeded max restart attempts."""
        return self.restart_count < self.max_restarts
    
    async def run(self, restart_callback) -> None:
        """
        Background monitoring task. Checks for inactivity and triggers restart.
        
        Args:
            restart_callback: Async function to call for restart
        """
        self._running = True
        logger.info("Watchdog monitoring started")
        
        while self._running:
            await asyncio.sleep(self._check_interval)
            
            idle_time = self.time_since_activity()
            
            # Log periodic status
            if idle_time > 30:
                logger.warning(f"WATCHDOG: No activity for {idle_time:.0f}s (timeout: {self.timeout}s)")
            
            if self.is_stalled():
                if self.can_restart():
                    self.restart_count += 1
                    logger.error(
                        f"WATCHDOG: No activity for {idle_time:.0f}s! "
                        f"Triggering restart ({self.restart_count}/{self.max_restarts})..."
                    )
                    try:
                        await restart_callback()
                        self.heartbeat()  # Reset after restart attempt
                        logger.info("WATCHDOG: Restart completed, resuming monitoring")
                    except Exception as e:
                        logger.error(f"WATCHDOG: Restart failed: {e}")
                else:
                    logger.error(
                        f"WATCHDOG: Max restarts ({self.max_restarts}) exceeded. "
                        f"System is unrecoverable. Stopping watchdog."
                    )
                    self._running = False
                    break
        
        logger.info("Watchdog monitoring stopped")
    
    def stop(self):
        """Stop the watchdog monitoring loop."""
        self._running = False
        logger.info("Watchdog stop requested")


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
    
    # ComfyUI optimized options for cloud 24GB GPU:
    # --highvram: Keep models in VRAM (we have 24GB, no need to swap)
    # --cuda-malloc: Use CUDA's native allocator (more stable)
    # --preview-method none: No previews needed for headless generation
    # --disable-metadata: Don't embed metadata (faster saves)
    # --gpu-only: Force GPU mode, no CPU fallback
    comfyui_args = [
        "python", "main.py",
        "--listen", "127.0.0.1",
        "--port", str(comfyui_port),
        "--highvram",
        "--cuda-malloc",
        "--preview-method", "none",
        "--disable-metadata",
    ]
    
    logger.info(f"ComfyUI args: {' '.join(comfyui_args)}")
    
    # Create log file for ComfyUI output (critical for debugging hangs!)
    # Use unbuffered mode so logs appear immediately
    comfyui_log_path = "/app/comfyui.log"
    logger.info(f"ComfyUI logs will be written to: {comfyui_log_path}")
    
    # Start ComfyUI in background with output logged to file
    try:
        comfyui_log_file = open(comfyui_log_path, 'w', buffering=1)  # Line-buffered
        
        # Write header with startup info
        import datetime
        comfyui_log_file.write(f"=" * 80 + "\n")
        comfyui_log_file.write(f"ComfyUI Started: {datetime.datetime.now().isoformat()}\n")
        comfyui_log_file.write(f"Args: {' '.join(comfyui_args)}\n")
        comfyui_log_file.write(f"Working Dir: {comfyui_path}\n")
        comfyui_log_file.write(f"=" * 80 + "\n")
        comfyui_log_file.flush()
        
        # Store globally to prevent garbage collection closing the file
        global _comfyui_log_file
        _comfyui_log_file = comfyui_log_file
        
        process = subprocess.Popen(
            comfyui_args,
            cwd=comfyui_path,
            stdout=comfyui_log_file,
            stderr=subprocess.STDOUT,
            text=True,
            # Force Python to run unbuffered for immediate log output
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
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


async def discover_comfyui(vps_api_url: str, auth_token: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Discover ComfyUI endpoint from VPS registry (Pod mode)
    
    In two-pod architecture, ComfyUI runs in a separate pod and registers
    its IP with the VPS. DreamGen queries the VPS to discover where ComfyUI is.
    
    Args:
        vps_api_url: Base URL for VPS API (e.g., https://aetherawi.red)
        auth_token: Authentication token for VPS API
        timeout: Max seconds to wait for ComfyUI to be registered
    
    Returns:
        Dict with 'url', 'auth_user', 'auth_pass' keys
    
    Raises:
        RuntimeError: If ComfyUI not registered within timeout
    """
    import aiohttp
    
    endpoint_url = f"{vps_api_url}/api/dreams/comfyui"
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    logger.info(f"Discovering ComfyUI from VPS: {endpoint_url}")
    
    start_time = asyncio.get_event_loop().time()
    poll_interval = 5  # seconds
    
    async with aiohttp.ClientSession() as session:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed > timeout:
                raise RuntimeError(
                    f"ComfyUI not registered with VPS after {timeout}s. "
                    "Make sure ComfyUI pod is running and registered."
                )
            
            try:
                async with session.get(endpoint_url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"ComfyUI discovered: {data.get('url')}")
                        return data
                    elif resp.status == 503:
                        # Not registered yet, keep waiting
                        logger.info(f"ComfyUI not registered yet, waiting... ({elapsed:.0f}s)")
                    elif resp.status == 401:
                        raise RuntimeError("VPS auth failed - check DREAM_GEN_AUTH_TOKEN")
                    else:
                        logger.warning(f"VPS returned unexpected status: {resp.status}")
            
            except aiohttp.ClientError as e:
                logger.warning(f"VPS request failed: {e}")
            
            await asyncio.sleep(poll_interval)


async def wait_for_comfyui_healthy(comfyui_url: str, auth_user: str = "", auth_pass: str = "", timeout: int = 120) -> bool:
    """
    Wait for ComfyUI to be healthy and responding
    
    Args:
        comfyui_url: ComfyUI base URL
        auth_user: Basic auth username (optional)
        auth_pass: Basic auth password (optional)
        timeout: Max seconds to wait
    
    Returns:
        True if ComfyUI is healthy
    """
    import aiohttp
    
    health_url = f"{comfyui_url}/system_stats"
    
    # Setup basic auth if credentials provided
    auth = None
    if auth_user and auth_pass:
        auth = aiohttp.BasicAuth(auth_user, auth_pass)
    
    logger.info(f"Waiting for ComfyUI to be healthy: {health_url}")
    start_time = asyncio.get_event_loop().time()
    
    async with aiohttp.ClientSession() as session:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed > timeout:
                logger.error(f"ComfyUI not healthy after {timeout}s")
                return False
            
            try:
                async with session.get(health_url, auth=auth, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(f"ComfyUI is healthy after {elapsed:.1f}s")
                        return True
            except Exception as e:
                pass  # Still starting up
            
            await asyncio.sleep(2)
            if int(elapsed) % 10 == 0:
                logger.info(f"Waiting for ComfyUI health... ({elapsed:.0f}s)")


async def run_dream_generation(
    vps_websocket_url: str,
    auth_token: Optional[str] = None,
    initial_state: Optional[bytes] = None,
    vps_api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main generation loop for RunPod
    
    Connects to VPS, restores state if provided, and streams frames
    until shutdown is requested.
    
    Supports two modes:
    - SERVERLESS: Starts ComfyUI locally (default)
    - POD: Discovers ComfyUI from VPS registry
    
    Args:
        vps_websocket_url: WebSocket URL to connect to VPS
        auth_token: Optional authentication token
        initial_state: Optional state bytes to restore from
        vps_api_url: VPS API URL for ComfyUI discovery (pod mode)
    
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
    logger.info(f"MODE: {DREAMGEN_MODE.upper()}")
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
    
    # ComfyUI URL and auth (will be set based on mode)
    comfyui_url = None
    comfyui_auth_user = ""
    comfyui_auth_pass = ""
    
    # Step 1: Get ComfyUI endpoint (mode-dependent)
    if DREAMGEN_MODE == "pod":
        # POD MODE: Discover ComfyUI from VPS registry
        logger.info("Step 1: Discovering ComfyUI from VPS (pod mode)...")
        
        # Get VPS API URL
        if not vps_api_url:
            vps_api_url = os.environ.get("VPS_API_URL", "https://aetherawi.red")
        
        if not auth_token:
            return {"status": "error", "error": "DREAM_GEN_AUTH_TOKEN required for pod mode"}
        
        try:
            comfyui_info = await discover_comfyui(vps_api_url, auth_token)
            comfyui_url = comfyui_info.get("url")
            comfyui_auth_user = comfyui_info.get("auth_user", "")
            comfyui_auth_pass = comfyui_info.get("auth_pass", "")
            
            logger.info(f"ComfyUI discovered: {comfyui_url}")
            if comfyui_auth_user:
                logger.info(f"ComfyUI auth: {comfyui_auth_user}:****")
            
            # Wait for ComfyUI to be healthy
            healthy = await wait_for_comfyui_healthy(
                comfyui_url, comfyui_auth_user, comfyui_auth_pass
            )
            if not healthy:
                return {"status": "error", "error": "ComfyUI not healthy"}
            
        except Exception as e:
            logger.error(f"Failed to discover ComfyUI: {e}")
            return {"status": "error", "error": f"ComfyUI discovery failed: {e}"}
    
    else:
        # SERVERLESS MODE: Start ComfyUI locally
        logger.info("Step 1: Starting ComfyUI locally (serverless mode)...")
        if not await start_comfyui():
            return {"status": "error", "error": "Failed to start ComfyUI"}
        comfyui_url = "http://127.0.0.1:8188"
    
    logger.info(f"ComfyUI ready at: {comfyui_url}")
    
    # Log VRAM (for serverless mode where ComfyUI just started)
    if DREAMGEN_MODE != "pod":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"VRAM after ComfyUI: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except Exception as e:
            logger.warning(f"Failed to get VRAM: {e}")
    
    # Override config for cloud mode + ComfyUI URL
    config_overrides = {
        'system': {
            'comfyui_url': comfyui_url,
        },
        'cloud': {
            'enabled': True,
            'vps_websocket_url': vps_websocket_url,
            'auth_token': auth_token,
        },
    }
    
    # Store ComfyUI auth in environment for ComfyUIClient to use
    if comfyui_auth_user and comfyui_auth_pass:
        os.environ["COMFYUI_AUTH_USER"] = comfyui_auth_user
        os.environ["COMFYUI_AUTH_PASS"] = comfyui_auth_pass
        logger.info("ComfyUI auth credentials set in environment")
    
    controller = None
    vps_client = None
    
    try:
        # Initialize controller with cloud mode
        logger.info("Initializing DreamController...")
        
        # Select config file based on mode
        if DREAMGEN_MODE == "pod":
            config_path = os.environ.get("CONFIG_PATH", "backend/config.cloud.yaml")
        else:
            config_path = os.environ.get("CONFIG_PATH", "backend/config.cloud.yaml")
        
        if not Path(config_path).exists():
            config_path = "backend/config.yaml"
            logger.info(f"Config not found, using default: {config_path}")
        else:
            logger.info(f"Using config: {config_path}")
        
        # Load config and apply overrides BEFORE creating controller
        # This is necessary because DreamController creates DreamGenerator in __init__
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply ComfyUI URL override
        original_url = config.get('system', {}).get('comfyui_url', 'http://127.0.0.1:8188')
        if comfyui_url and comfyui_url != original_url:
            logger.info(f"Overriding ComfyUI URL: {original_url} -> {comfyui_url}")
            config['system']['comfyui_url'] = comfyui_url
        
        # Apply cloud mode override
        if 'cloud' not in config:
            config['cloud'] = {}
        config['cloud']['enabled'] = True
        config['cloud']['vps_websocket_url'] = vps_websocket_url
        if auth_token:
            config['cloud']['auth_token'] = auth_token
        
        # Write modified config to temp file for DreamController
        import tempfile
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config)
        temp_config.close()
        temp_config_path = temp_config.name
        logger.info(f"Created temp config with overrides: {temp_config_path}")
        
        try:
            controller = DreamController(config_path=temp_config_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_config_path)
            except:
                pass
        
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
            
            # Send target FPS configuration to VPS for smooth playback pacing
            target_fps = controller.config.get('generation', {}).get('hybrid', {}).get('target_interpolation_fps', 5.0)
            import json
            fps_config = json.dumps({"target_fps": target_fps}).encode('utf-8')
            fps_sent = await vps_client.send_status(fps_config)
            if fps_sent:
                logger.info(f"[OK] Sent target FPS to VPS: {target_fps}")
            else:
                logger.warning(f"Failed to send target FPS to VPS (will use default)")
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
        
        # Run generation loop with watchdog monitoring
        logger.info("Starting generation loop with watchdog...")
        controller.running = True
        
        # Create watchdog with 90s timeout, max 3 restarts
        global _watchdog
        _watchdog = ActivityWatchdog(timeout_seconds=90.0, max_restarts=3)
        
        # Hook watchdog heartbeat to frame pusher via callback (clean, no monkey-patching)
        if hasattr(controller, 'frame_pusher') and controller.frame_pusher:
            controller.frame_pusher.set_push_callback(_watchdog.heartbeat)
            logger.info("Watchdog heartbeat registered via frame pusher callback")
        else:
            logger.warning("No frame_pusher available - watchdog may not receive heartbeats")
        
        # Restart callback for watchdog
        # This is a "nuclear" restart - we stop everything and let the outer
        # exception handler clean up. The job will be marked as failed and
        # can be resubmitted (or RunPod may auto-retry depending on settings).
        async def watchdog_restart():
            logger.error("WATCHDOG RESTART: System stalled! Initiating full restart...")
            
            # Stop the controller gracefully
            if controller:
                try:
                    controller.stop()
                except Exception as e:
                    logger.warning(f"Controller stop failed: {e}")
            
            # Kill ComfyUI to ensure clean state
            import subprocess
            try:
                subprocess.run(["pkill", "-f", "ComfyUI.*main.py"], timeout=5)
                logger.info("ComfyUI process killed")
            except Exception as e:
                logger.warning(f"ComfyUI kill failed: {e}")
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"CUDA cache clear failed: {e}")
            
            # Wait a moment for cleanup
            await asyncio.sleep(2)
            
            # Restart ComfyUI
            logger.info("WATCHDOG RESTART: Restarting ComfyUI...")
            if not await start_comfyui():
                logger.error("Failed to restart ComfyUI!")
                raise RuntimeError("Watchdog restart failed: ComfyUI won't start")
            
            # Restart the controller's generation loop
            # This is done by simply returning - the generation_task will be
            # cancelled and restarted in the outer loop
            logger.info("WATCHDOG RESTART: ComfyUI restarted, attempting to resume...")
            
            # Reset controller state for fresh start
            if controller:
                controller.running = True
                # Clear any stuck queues
                if hasattr(controller, 'generation_coordinator') and controller.generation_coordinator:
                    coord = controller.generation_coordinator
                    if hasattr(coord, 'keyframe_worker') and coord.keyframe_worker:
                        # Clear the request queue
                        while not coord.keyframe_worker.request_queue.empty():
                            try:
                                coord.keyframe_worker.request_queue.get_nowait()
                            except:
                                break
                    if hasattr(coord, 'interpolation_worker') and coord.interpolation_worker:
                        # Clear the pair queue
                        while not coord.interpolation_worker.pair_queue.empty():
                            try:
                                coord.interpolation_worker.pair_queue.get_nowait()
                            except:
                                break
                logger.info("Cleared worker queues for fresh start")
        
        # Track if we need to restart and current task
        restart_requested = False
        current_generation_task = None
        
        # Modified restart callback that signals for restart
        async def watchdog_restart_signal():
            nonlocal restart_requested, current_generation_task
            await watchdog_restart()
            restart_requested = True
            # Cancel the current generation task to trigger restart
            if current_generation_task and not current_generation_task.done():
                current_generation_task.cancel()
        
        # Run with restart loop
        while _watchdog.can_restart() or _watchdog.restart_count == 0:
            restart_requested = False
            
            # Run generation and watchdog in parallel
            current_generation_task = asyncio.create_task(controller.run_buffered_hybrid_loop())
            watchdog_task = asyncio.create_task(_watchdog.run(watchdog_restart_signal))
            
            try:
                # Wait for generation to complete
                await current_generation_task
                # If we get here normally, generation completed successfully
                logger.info("Generation loop completed normally")
                break
                
            except asyncio.CancelledError:
                if restart_requested:
                    logger.info("Generation cancelled for watchdog restart, restarting loop...")
                    # Give the system a moment before restarting
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.info("Generation cancelled externally")
                    raise
                    
            finally:
                _watchdog.stop()
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
        
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


async def pod_mode():
    """
    Run handler in pod mode (long-running process, not serverless)
    
    In pod mode:
    - ComfyUI is in a separate pod, discovered via VPS registry
    - Generation runs until shutdown signal received
    - No job-based execution (runs continuously)
    """
    logger.info("=" * 60)
    logger.info("DREAMGEN POD MODE - Long Running Process")
    logger.info("=" * 60)
    
    # Get config from environment
    vps_url = os.environ.get("VPS_WEBSOCKET_URL", "wss://aetherawi.red/ws/gpu")
    vps_api_url = os.environ.get("VPS_API_URL", "https://aetherawi.red")
    auth_token = os.environ.get("DREAM_GEN_AUTH_TOKEN")
    
    logger.info(f"VPS WebSocket URL: {vps_url}")
    logger.info(f"VPS API URL: {vps_api_url}")
    logger.info(f"Auth token: {'set' if auth_token else 'NOT SET'}")
    
    if not auth_token:
        logger.error("DREAM_GEN_AUTH_TOKEN is required for pod mode")
        return {"status": "error", "error": "DREAM_GEN_AUTH_TOKEN required"}
    
    # Setup signal handlers for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        shutdown_requested = True
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run generation loop
    result = await run_dream_generation(
        vps_websocket_url=vps_url,
        vps_api_url=vps_api_url,
        auth_token=auth_token,
        initial_state=None,
    )
    
    logger.info(f"Pod mode result: {result}")
    return result


def main():
    """Entry point for serverless, pod, or local testing modes"""
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    
    # Debug info
    print(f"[DEBUG] sys.argv: {sys.argv}")
    print(f"[DEBUG] DREAMGEN_MODE: {DREAMGEN_MODE}")
    print(f"[DEBUG] HAS_RUNPOD: {HAS_RUNPOD}")
    print(f"[DEBUG] '--local' in argv: {'--local' in sys.argv}")
    
    # Check mode
    if "--local" in sys.argv:
        # Local test mode (for development)
        logger.info("Starting in local test mode...")
        asyncio.run(local_test_mode())
    
    elif DREAMGEN_MODE == "pod":
        # Pod mode - long running process
        logger.info("Starting in POD mode (long-running process)...")
        asyncio.run(pod_mode())
    
    elif HAS_RUNPOD:
        # Serverless mode (legacy)
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({
            "handler": handler,
            # Optional: Use async generator for streaming
            # "handler": async_generator_handler,
        })
    
    else:
        # Fallback to local mode if runpod not available
        logger.info("RunPod module not available, falling back to local test mode...")
        asyncio.run(local_test_mode())


if __name__ == "__main__":
    main()

