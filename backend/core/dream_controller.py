"""
Dream Window Main Controller

Entry point for the Dream Window system. Orchestrates:
- Image generation (img2img or hybrid mode)
- Cache management with dual-metric similarity (ColorHist + pHash-8)
- Prompt rotation
- Status monitoring
- Display output

Run with:
    uv run backend/main.py
"""

import asyncio
import time
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
import yaml
import random
from PIL import Image
import torch

from core.generator import DreamGenerator
from core.frame_buffer import FrameBuffer
from core.generation_coordinator import GenerationCoordinator
from core.async_orchestrator import AsyncGenerationOrchestrator
from core.shared_resources import SharedVAEAccess
from core.display_selector import DisplayFrameSelector
from utils.prompt_manager import PromptManager
from prompts.combinatorial import CombinatorialPromptSystem
from utils.status_writer import StatusWriter
from utils.file_ops import atomic_write_image_with_retry
from utils.game_detector import GameDetector
from cache.manager import CacheManager
from cache.dual_similarity import DualMetricSimilarityManager
from utils.perf_stats import get_perf_stats

# Setup logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """
    Configure logging system with rotation
    
    Console: Shows INFO+ by default (important events, warnings, errors)
    File: Captures everything (DEBUG+) for post-mortem analysis
    
    Noisy loggers (urllib3, websockets, etc.) are quieted to WARNING
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "dream_controller.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Shorter format for console (no timestamp, shorter name)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Rotating file handler (max 5MB per file, keep 3 backups)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3  # Keep 3 backup files (dream_controller.log.1, .2, .3)
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - always INFO+ regardless of config (file gets DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Quiet noisy third-party loggers (only show WARNING+)
    noisy_loggers = [
        'urllib3',
        'websockets', 
        'websockets.client',
        'filelock',
        'PIL',
        'httpcore',
        'httpx',
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Quiet internal spammy loggers (INFO only, not DEBUG)
    spammy_loggers = [
        'utils.status_writer',  # Very chatty with status updates
    ]
    for logger_name in spammy_loggers:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    return logging.getLogger(__name__)


class DreamController:
    """
    Main controller for Dream Window
    
    Responsibilities:
    - Initialize all subsystems
    - Run main generation loop
    - Handle lifecycle (start/stop/pause)
    - Coordinate between components
    - Manage hybrid mode logic
    - Cache injection integration
    """
    
    def __init__(self, config_path: str = "backend/config.yaml"):
        """
        Initialize Dream Window controller
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        log_dir = Path(self.config['system']['log_dir'])
        self.logger = setup_logging(log_dir, self.config['system']['log_level'])
        
        self.logger.info("=" * 70)
        self.logger.info("DREAM WINDOW CONTROLLER INITIALIZING")
        self.logger.info("=" * 70)
        
        # Initialize paths
        self.output_dir = Path(self.config['system']['output_dir'])
        self.seed_dir = Path(self.config['system']['seed_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self.logger.info("Initializing subsystems...")
        self.generator = DreamGenerator(self.config)
        
        # Use CombinatorialPromptSystem if templates are available, else legacy PromptManager
        self.prompt_manager = self._init_prompt_system()
        
        self.status_writer = StatusWriter(self.output_dir)
        self.game_detector = GameDetector(self.config)
        self.cache = CacheManager(self.config)
        
        # Initialize dual-metric similarity manager for cache
        self.logger.info("Initializing Dual-Metric Similarity Manager...")
        self.similarity_manager = DualMetricSimilarityManager(self.config)
        self.logger.info("  Using ColorHist + pHash-8 with OR logic for collapse detection")
        
        # Initialize hybrid mode if enabled
        self.latent_encoder = None
        if self.config['generation']['mode'] == 'hybrid':
            self.logger.info("Initializing hybrid mode with VAE interpolation...")
            
            try:
                # Load VAE for true interpolation
                from interpolation.latent_encoder import LatentEncoder
                
                self.logger.info("Loading VAE for interpolation...")
                # Get interpolation resolution settings from config
                resolution_divisor = self.config['generation']['hybrid'].get('interpolation_resolution_divisor', 1)
                upscale_method = self.config['generation']['hybrid'].get('interpolation_upscale_method', 'bilinear')
                
                gpu_id = self.config.get('system', {}).get('gpu_id', 0)
                device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cuda"
                self.logger.info(f"Using device: {device}")

                # Get target resolution from config to force resize
                target_resolution = tuple(self.config['generation']['resolution'])  # [width, height]
                self.logger.info(f"Target resolution: {target_resolution}")
                # Get torch.compile setting
                enable_compile = self.config.get('system', {}).get('enable_torch_compile', False)
                self.logger.info(f"Enable torch compile: {enable_compile}")
                self.latent_encoder = LatentEncoder(
                    device=device,
                    auto_load=True,
                    interpolation_resolution_divisor=resolution_divisor,
                    upscale_method=upscale_method,
                    target_resolution=target_resolution,
                    enable_torch_compile=enable_compile
                )
                
                # Synchronize CUDA after loading models to ensure context is fully initialized
                # This prevents CUDA context errors during the first encode operation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self.logger.debug("CUDA synchronized after model loading")
                
                self.logger.info("[OK] VAE interpolation enabled")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize hybrid mode with VAE: {e}", exc_info=True)
                self.logger.error("Hybrid mode requires VAE interpolation. Please check your configuration.")
                raise
        
        # Initialize buffered frame system
        if self.config['generation']['mode'] == 'hybrid' and self.latent_encoder:
            self.logger.info("Initializing buffered frame system...")
            
            # Get buffer parameters
            interpolation_frames = self.config['generation']['hybrid']['interpolation_frames']
            target_fps = self.config['generation']['hybrid']['target_interpolation_fps']
            buffer_target_seconds = self.config['display'].get('buffer_target_seconds', 30.0)
            min_buffer_seconds = self.config['display'].get('min_buffer_seconds', 10.0)
            
            # Create frame buffer
            self.frame_buffer = FrameBuffer(
                interpolation_frames=interpolation_frames,
                target_fps=target_fps,
                output_dir=self.output_dir,
                buffer_target_seconds=buffer_target_seconds
            )
            
            # Check if we should use async orchestrator (new parallelized system)
            use_async = self.config['generation'].get('use_async_orchestrator', False)
            
            if use_async:
                self.logger.info("Using AsyncGenerationOrchestrator (parallelized pipeline)")
                
                # Create SharedVAEAccess wrapper for thread-safe VAE operations
                self.vae_access = SharedVAEAccess(self.latent_encoder)
                
                # Create DEDICATED second VAE for injection blending
                # This eliminates lock contention between interpolation and injection
                # Cost: ~160MB VRAM (trivial vs 15+ GB headroom)
                self.logger.info("Loading secondary VAE for injection blending (zero contention)...")
                self.injection_vae = LatentEncoder(
                    device=device,
                    auto_load=True,
                    interpolation_resolution_divisor=resolution_divisor,
                    upscale_method=upscale_method,
                    target_resolution=target_resolution,
                    enable_torch_compile=False  # Simpler for occasional use
                )
                self.logger.info("[OK] Secondary VAE loaded for injection (dual-VAE architecture)")
                
                # Get seed image for bootstrap
                seed_image = self.get_random_seed_image()
                
                # Create async orchestrator
                self.generation_coordinator = AsyncGenerationOrchestrator(
                    frame_buffer=self.frame_buffer,
                    generator=self.generator,
                    vae_access=self.vae_access,
                    prompt_manager=self.prompt_manager,
                    cache_manager=self.cache,
                    similarity_manager=self.similarity_manager,
                    config=self.config,
                    seed_image=seed_image,
                    injection_vae=self.injection_vae  # Dedicated VAE for injection blending
                )
                
                self.logger.info("  Workers: KeyframeWorker, InterpolationWorker, CacheAnalysisWorker")
                self.logger.info("  Expected FPS improvement: 2x+ (from ~2.7 to ~5+ fps)")
            else:
                self.logger.info("Using GenerationCoordinator (legacy sequential pipeline)")
                
                # Create legacy generation coordinator
                self.vae_access = None
                self.generation_coordinator = GenerationCoordinator(
                    frame_buffer=self.frame_buffer,
                    generator=self.generator,
                    latent_encoder=self.latent_encoder,
                    prompt_manager=self.prompt_manager,
                    config=self.config,
                    cache_manager=self.cache,
                    similarity_manager=self.similarity_manager
                )
            
            # Get cleanup config
            cleanup_config = self.config.get('display', {})
            cleanup_enabled = cleanup_config.get('cleanup_displayed_frames', False)
            
            # Cloud mode optimization: skip writing current_frame.png to disk
            # (frames are pushed directly via WebSocket, disk write is unnecessary)
            cloud_mode = self.config.get('cloud', {}).get('enabled', False)
            
            # Get keyframe_worker reference for source image protection during retries
            # (only available when using async orchestrator)
            keyframe_worker = None
            if hasattr(self.generation_coordinator, 'keyframe_worker'):
                keyframe_worker = self.generation_coordinator.keyframe_worker
            
            # Get interpolation_worker reference for protecting keyframes during pending interpolations
            interpolation_worker = None
            if hasattr(self.generation_coordinator, 'interpolation_worker'):
                interpolation_worker = self.generation_coordinator.interpolation_worker
            
            # Create display selector
            self.display_selector = DisplayFrameSelector(
                frame_buffer=self.frame_buffer,
                output_dir=self.output_dir,
                target_fps=target_fps,
                min_buffer_seconds=min_buffer_seconds,
                cleanup_displayed_frames=cleanup_enabled,
                skip_disk_write=cloud_mode,  # Skip disk I/O in cloud mode
                keyframe_worker=keyframe_worker,  # Protect source images during retry
                interpolation_worker=interpolation_worker  # Protect keyframes for pending interpolations
            )
            
            self.logger.info("[OK] Buffered frame system initialized")
        else:
            self.frame_buffer = None
            self.generation_coordinator = None
            self.display_selector = None
        
        # State
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.current_image = None
        self.start_time = None
        self.vram_freed = False  # Track if we've freed VRAM for gaming
        self.last_game_check = 0  # Timestamp of last game detection check
        
        # Task management for shutdown
        self.running_tasks = []
        self.asyncio_loop = None
        
        # Statistics (moved to GenerationCoordinator)
        self.generation_times = []
        
        # Frame management
        self.max_output_frames = self.config.get('display', {}).get('max_output_frames', 100)
        
        # Cloud mode initialization (optional)
        self.cloud_enabled = self.config.get('cloud', {}).get('enabled', False)
        self.vps_client = None
        self.frame_pusher = None
        self.state_sync = None
        
        if self.cloud_enabled:
            self._init_cloud_mode()
            # Set cloud callback on display selector if available
            if self.display_selector and self.frame_pusher:
                self.display_selector.on_frame_callback = self._on_frame_displayed
                self.logger.info("Cloud callback attached to display selector")
        else:
            self.logger.info("Cloud mode: disabled (standalone Rainmeter mode)")
        
        self.logger.info("[OK] Initialization complete")
        self.logger.info(f"Mode: {self.config['generation']['mode']}")
        self.logger.info(f"Resolution: {self.config['generation']['resolution']}")
        self.logger.info(f"Model: {self.config['generation']['model']}")
        self.logger.info("=" * 70)
    
    def cleanup_old_frames(self):
        """
        Clean up old numbered frames to prevent unbounded storage growth
        
        Keeps only the most recent N frames (configured in display.max_output_frames).
        Preserves special files: current_frame.png, previous_frame.png, next_frame.png, status.json
        
        This runs periodically during generation to maintain a rolling window of frames.
        """
        try:
            # Get all numbered frames
            frame_files = sorted(self.output_dir.glob("frame_*.png"))
            
            if len(frame_files) <= self.max_output_frames:
                return  # Nothing to clean
            
            # Calculate how many to delete
            num_to_delete = len(frame_files) - self.max_output_frames
            files_to_delete = frame_files[:num_to_delete]
            
            # Delete oldest frames
            for frame_file in files_to_delete:
                try:
                    frame_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete {frame_file.name}: {e}")
            
            self.logger.info(f"Cleaned up {num_to_delete} old frames (keeping last {self.max_output_frames})")
            
        except Exception as e:
            self.logger.error(f"Error during frame cleanup: {e}")
    
    def _init_prompt_system(self):
        """
        Initialize the prompt system - uses CombinatorialPromptSystem if templates available
        
        Priority:
        1. CombinatorialPromptSystem if prompts/templates.yaml exists (infinite gen mode)
        2. Legacy PromptManager otherwise (uses theme_pairs from config)
        
        Returns:
            Prompt system instance (CombinatorialPromptSystem or PromptManager)
        """
        from pathlib import Path
        
        # Check for templates file (indicates infinite gen mode)
        project_root = Path(__file__).parent.parent.parent
        templates_path = project_root / "prompts" / "templates.yaml"
        components_path = project_root / "prompts" / "components.yaml"
        
        if templates_path.exists() and components_path.exists():
            try:
                prompt_system = CombinatorialPromptSystem(
                    templates_path=str(templates_path),
                    components_path=str(components_path),
                    config=self.config
                )
                self.logger.info("[INFINITE GEN] CombinatorialPromptSystem loaded")
                self.logger.info(f"  Templates: {len(prompt_system.templates)}")
                self.logger.info(f"  Categories: {list(prompt_system.components.keys())}")
                return prompt_system
            except Exception as e:
                self.logger.warning(f"Failed to load CombinatorialPromptSystem: {e}")
                self.logger.warning("Falling back to legacy PromptManager")
        
        # Fall back to legacy PromptManager
        return PromptManager(self.config)
    
    def _init_cloud_mode(self) -> None:
        """
        Initialize cloud mode components
        
        Sets up WebSocket client, frame pusher, and state sync for
        pushing frames to VPS. This is only called when cloud.enabled is True.
        """
        self.logger.info("Initializing cloud mode...")
        
        try:
            from cloud import VPSWebSocketClient, CloudFramePusher, CloudStateSync
            
            cloud_config = self.config['cloud']
            
            # Apply resolution override if specified
            resolution_override = cloud_config.get('resolution_override')
            if resolution_override:
                self.logger.info(f"Cloud resolution override: {resolution_override}")
                self.config['generation']['resolution'] = resolution_override
            
            # Create WebSocket client
            self.vps_client = VPSWebSocketClient(cloud_config)
            
            # Set control callbacks
            self.vps_client.set_callbacks(
                on_pause=self._on_cloud_pause,
                on_resume=self._on_cloud_resume,
                on_save_state=self._on_cloud_save_state,
                on_shutdown=self._on_cloud_shutdown,
                on_load_state=self._on_cloud_load_state,
            )
            
            # Create frame pusher
            self.frame_pusher = CloudFramePusher(self.vps_client, cloud_config)
            
            # Create state sync
            self.state_sync = CloudStateSync(self.vps_client, cloud_config)
            
            # Attach callback to display selector for pushing frames
            if self.display_selector and self.frame_pusher:
                self.display_selector.on_frame_callback = self._on_frame_displayed
                self.logger.info("  Cloud callback attached to display selector")
            
            self.logger.info("[OK] Cloud mode initialized")
            self.logger.info(f"  VPS URL: {cloud_config.get('vps_websocket_url')}")
            self.logger.info(f"  Frame format: {cloud_config.get('frame_push', {}).get('format', 'webp')}")
            self.logger.info(f"  State sync interval: {cloud_config.get('state_sync', {}).get('interval_keyframes', 10)} keyframes")
        
        except ImportError as e:
            self.logger.error(f"Cloud mode dependencies not available: {e}")
            self.logger.error("Install websockets and msgpack: pip install websockets msgpack")
            self.cloud_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud mode: {e}", exc_info=True)
            self.cloud_enabled = False
    
    async def connect_to_vps(self) -> bool:
        """
        Connect to VPS WebSocket endpoint
        
        Called at start of generation when cloud mode is enabled.
        
        Returns:
            True if connected successfully
        """
        if not self.cloud_enabled or not self.vps_client:
            return False
        
        self.logger.info("Connecting to VPS...")
        connected = await self.vps_client.connect()
        
        if connected:
            self.logger.info("[OK] Connected to VPS")
        else:
            self.logger.warning("Failed to connect to VPS, will retry in background")
        
        return connected
    
    async def disconnect_from_vps(self) -> None:
        """
        Disconnect from VPS gracefully
        
        Pushes final state before disconnecting.
        """
        if not self.cloud_enabled or not self.vps_client:
            return
        
        # Push final state
        if self.state_sync:
            try:
                await self.state_sync.on_shutdown(
                    cache_manager=self.cache,
                    similarity_manager=self.similarity_manager
                )
            except Exception as e:
                self.logger.error(f"Error pushing final state: {e}")
        
        # Disconnect
        await self.vps_client.disconnect()
        self.logger.info("Disconnected from VPS")
    
    async def push_frame_to_cloud(
        self,
        image: Image.Image,
        is_keyframe: bool = False,
        frame_number: int = 0,
        keyframe_number: int = 0,
    ) -> bool:
        """
        Push a frame to VPS if cloud mode is enabled
        
        Args:
            image: PIL Image to push
            is_keyframe: Whether this is a keyframe
            frame_number: Sequential frame number
            keyframe_number: Current keyframe number
        
        Returns:
            True if pushed successfully (or cloud mode disabled)
        """
        if not self.cloud_enabled or not self.frame_pusher:
            return True  # Not an error
        
        return await self.frame_pusher.push_frame(
            image=image,
            is_keyframe=is_keyframe,
            frame_number=frame_number,
            keyframe_number=keyframe_number,
        )
    
    async def sync_state_to_cloud(
        self,
        keyframe_latent,
        generation_state: dict,
    ) -> bool:
        """
        Sync state to VPS if cloud mode is enabled
        
        Args:
            keyframe_latent: Latent tensor from VAE encoding
            generation_state: Dict with frame_count, keyframe_count, etc.
        
        Returns:
            True if synced (or skipped based on interval)
        """
        if not self.cloud_enabled or not self.state_sync:
            return True  # Not an error
        
        return await self.state_sync.on_keyframe_complete(
            keyframe_latent=keyframe_latent,
            generation_state=generation_state,
            cache_manager=self.cache,
            similarity_manager=self.similarity_manager,
        )
    
    # Cloud control callbacks
    async def _on_cloud_pause(self) -> None:
        """Handle pause command from VPS"""
        self.logger.info("Received pause command from VPS")
        self.paused = True
    
    async def _on_cloud_resume(self) -> None:
        """Handle resume command from VPS"""
        self.logger.info("Received resume command from VPS")
        self.paused = False
    
    async def _on_cloud_save_state(self) -> None:
        """Handle save state command from VPS"""
        self.logger.info("Received save state command from VPS")
        if self.state_sync:
            # Force immediate state push
            await self.state_sync._push_state(
                include_cache=True,
                cache_manager=self.cache,
                similarity_manager=self.similarity_manager,
            )
    
    async def _on_cloud_shutdown(self) -> None:
        """Handle shutdown command from VPS"""
        self.logger.info("Received shutdown command from VPS")
        self.running = False
    
    async def _on_cloud_load_state(self, state_bytes: bytes) -> None:
        """
        Handle load state command from VPS
        
        Restores generation state from a previously saved snapshot, including:
        - Last keyframe latent (for interpolation continuity)
        - Frame/keyframe counters
        - Theme index
        - Cache metadata (if included)
        
        Args:
            state_bytes: Serialized state bundle from VPS
        """
        self.logger.info(f"Received load state command from VPS ({len(state_bytes)} bytes)")
        
        try:
            # Import deserializer
            from cloud.state_sync import deserialize_state
            import torch
            
            # Deserialize the state bundle
            bundle = deserialize_state(state_bytes)
            
            self.logger.info(f"State bundle contains: {list(bundle.keys())}")
            
            # Restore latent tensor for interpolation
            if "latent" in bundle:
                latent_np = bundle["latent"]
                latent_tensor = torch.from_numpy(latent_np)
                
                # Determine device
                if torch.cuda.is_available():
                    device_id = self.config.get('system', {}).get('gpu_id', 0)
                    device = f"cuda:{device_id}"
                else:
                    device = "cpu"
                
                latent_tensor = latent_tensor.to(device)
                
                self.logger.info(f"Restored latent: shape={latent_tensor.shape}, device={device}")
                
                # Inject into generation coordinator if available
                if hasattr(self, 'generation_coordinator') and self.generation_coordinator:
                    # Get the current keyframe number from restored state
                    state = bundle.get("state", {})
                    keyframe_count = state.get("keyframe_count", 1)
                    
                    # Store as the last keyframe latent
                    self.generation_coordinator.keyframe_latents[keyframe_count] = latent_tensor
                    self.logger.info(f"Injected latent as keyframe {keyframe_count}")
                
                # Store for async orchestrator if using that instead
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    # The orchestrator handles this differently
                    self.logger.info("Async orchestrator detected - latent restoration handled at startup")
            
            # Restore generation state counters
            if "state" in bundle:
                state = bundle["state"]
                
                if "frame_count" in state:
                    self.frame_count = state["frame_count"]
                    self.logger.info(f"Restored frame_count: {self.frame_count}")
                
                if "keyframe_count" in state and hasattr(self, 'generation_coordinator'):
                    if self.generation_coordinator:
                        self.generation_coordinator.keyframes_generated = state["keyframe_count"]
                        self.logger.info(f"Restored keyframe_count: {state['keyframe_count']}")
                
                if "theme_index" in state:
                    self.prompt_manager.current_theme_index = state["theme_index"]
                    self.logger.info(f"Restored theme_index: {state['theme_index']}")
                
                if "last_seed" in state and hasattr(self, 'generator'):
                    self.generator.last_seed = state["last_seed"]
                    self.logger.info(f"Restored last_seed: {state['last_seed']}")
            
            # Restore cache metadata if included
            if "cache_meta" in bundle:
                try:
                    self.cache.restore_metadata(bundle["cache_meta"])
                    self.logger.info("Restored cache metadata")
                except Exception as e:
                    self.logger.warning(f"Could not restore cache metadata: {e}")
            
            # Restore similarity embeddings if included
            if "embeddings" in bundle:
                try:
                    self.similarity_manager.deserialize(bundle["embeddings"])
                    self.logger.info("Restored similarity embeddings")
                except Exception as e:
                    self.logger.warning(f"Could not restore embeddings: {e}")
            
            self.logger.info("[OK] State restoration complete")
            
        except Exception as e:
            self.logger.error(f"Failed to restore state: {e}", exc_info=True)
            # Continue without restored state - will start fresh
    
    async def _on_frame_displayed(self, image: Image, frame_number: int, is_keyframe: bool) -> None:
        """
        Callback invoked when a frame is displayed
        
        Pushes the frame to cloud if enabled.
        
        Args:
            image: PIL Image being displayed
            frame_number: Sequential frame number
            is_keyframe: Whether this is a keyframe
        """
        if not self.cloud_enabled or not self.frame_pusher:
            return
        
        await self.frame_pusher.push_frame(
            image=image,
            is_keyframe=is_keyframe,
            frame_number=frame_number,
        )

    def get_random_seed_image(self) -> Path:
        """
        Get random seed image from seed directory
        
        Returns:
            Path to seed image
        
        Raises:
            ValueError: If no seed images found
        """
        seed_images = list(self.seed_dir.glob("*.png")) + list(self.seed_dir.glob("*.jpg"))
        
        if not seed_images:
            raise ValueError(f"No seed images found in {self.seed_dir}")
        
        return random.choice(seed_images)
    
    def check_game_state(self) -> bool:
        """
        Check if game is running and manage VRAM accordingly
        
        This is THE KEY to preventing VRAM conflicts!
        
        When game detected:
        1. Pause generation
        2. Free VRAM (unload models)
        3. Wait for game to close
        
        When game closes:
        4. Resume generation
        5. Model reloads automatically on next generation (~15s penalty)
        
        Returns:
            True if should pause generation (game running)
        """
        # Throttle checks to avoid overhead
        current_time = time.time()
        if current_time - self.last_game_check < self.game_detector.check_interval:
            return self.paused
        
        self.last_game_check = current_time
        
        # Check for running games
        game_detected = self.game_detector.is_game_running()
        
        if game_detected and not self.paused:
            # Game just started - pause and free VRAM!
            self.logger.warning(f"[GAME] DETECTED: {game_detected}")
            self.logger.info("Pausing generation and freeing VRAM...")
            self.paused = True
            
            # Free VRAM (unload models)
            try:
                success = self.generator.client.free_memory(
                    unload_models=True,
                    free_memory=True
                )
                if success:
                    self.vram_freed = True
                    self.logger.info("[OK] VRAM freed - safe for gaming!")
                else:
                    self.logger.warning("Could not free VRAM (ComfyUI might not support /free endpoint)")
                    self.logger.info("Generation paused anyway for safety")
            except Exception as e:
                self.logger.error(f"Error freeing VRAM: {e}")
            
            return True
        
        elif not game_detected and self.paused:
            # Game closed - resume!
            self.logger.info("[GAME] Game closed - resuming generation")
            self.logger.info("(Models will reload on next generation - ~15s delay expected)")
            self.paused = False
            self.vram_freed = False
            return False
        
        return self.paused
    
    def write_current_frame(self, frame_path: Path):
        """
        Write frame to current_frame.png for display
        
        Uses atomic writes to prevent corruption/tearing.
        
        Args:
            frame_path: Path to frame to display
        """
        output_file = self.output_dir / "current_frame.png"
        
        try:
            # Use atomic write with retry
            image = Image.open(frame_path)
            success = atomic_write_image_with_retry(
                image,
                output_file,
                max_retries=3
            )
            
            if success:
                self.logger.debug(f"Updated current_frame.png")
            else:
                self.logger.warning("Failed to update current_frame.png")
                
        except Exception as e:
            self.logger.error(f"Error writing current frame: {e}")
    
    def update_status(self, generation_time: float, mode: str, prompt: str):
        """
        Update status.json for display/monitoring
        
        Args:
            generation_time: Time taken to generate frame
            mode: Generation mode used
            prompt: Current prompt
        """
        try:
            # Calculate buffer status (for widget loading indicator)
            buffer_target = self.config.get('display', {}).get('buffer_size', 5)
            buffer_filled = min(self.frame_count, buffer_target)
            
            status_data = {
                "frame_number": self.frame_count,
                "generation_time": round(generation_time, 2),
                "status": "paused" if self.paused else "live",
                "current_mode": mode,
                "current_prompt": prompt[:100],  # Truncate long prompts
                "cache_size": self.cache.size(),
                "uptime_minutes": round((time.time() - self.start_time) / 60, 1) if self.start_time else 0,
                # Buffer status for widget
                "buffer_filled": buffer_filled,
                "buffer_target": buffer_target,
                "is_buffering": buffer_filled < buffer_target,
            }
            
            self.status_writer.write_status(status_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update status: {e}")
    
    async def run_buffered_hybrid_loop(self) -> None:
        """
        Run buffered hybrid generation loop (NEW ARCHITECTURE)
        
        Uses FrameBuffer, GenerationCoordinator (or AsyncGenerationOrchestrator),
        and DisplayFrameSelector to maintain a 30s buffer of frames for smooth playback.
        """
        if not self.frame_buffer or not self.generation_coordinator or not self.display_selector:
            self.logger.error("Buffered frame system not initialized!")
            return
        
        self.logger.info("=" * 70)
        self.logger.info("STARTING BUFFERED HYBRID MODE")
        self.logger.info("=" * 70)
        
        # Connect to VPS if cloud mode enabled (skip if already connected)
        if self.cloud_enabled and self.vps_client and not self.vps_client.connected:
            await self.connect_to_vps()
        
        # Check if using async orchestrator
        use_async = self.config['generation'].get('use_async_orchestrator', False)
        
        if use_async:
            # AsyncOrchestrator handles seed bootstrap internally
            self.logger.info("Using AsyncGenerationOrchestrator (parallelized)")
            
            # Clear ComfyUI queue
            self.logger.info("Clearing ComfyUI queue...")
            queue_status = self.generator.client.get_queue()
            if queue_status:
                running_count = len(queue_status.get("queue_running", []))
                pending_count = len(queue_status.get("queue_pending", []))
                if running_count > 0 or pending_count > 0:
                    self.logger.warning(f"Found stale jobs: {running_count} running, {pending_count} pending")
                    self.generator.client.interrupt_execution()
                    self.generator.client.clear_queue()
                    self.logger.info("Queue cleared")
            
            # Start orchestrator and display tasks concurrently
            self.logger.info("Starting orchestrator and display tasks...")
            
            generation_task = asyncio.create_task(self.generation_coordinator.run())
            display_task = asyncio.create_task(self.display_selector.run())
            status_task = asyncio.create_task(self._update_buffer_status_loop())
            
            # Store task references for signal handler
            self.running_tasks = [generation_task, display_task, status_task]
            self.asyncio_loop = asyncio.get_event_loop()
            
            try:
                # Run all tasks concurrently
                await asyncio.gather(generation_task, display_task, status_task)
            except asyncio.CancelledError:
                self.logger.info("Buffered hybrid loop cancelled")
            except KeyboardInterrupt:
                self.logger.info("Buffered hybrid loop interrupted")
            except Exception as e:
                self.logger.error(f"Error in buffered hybrid loop: {e}", exc_info=True)
            finally:
                # Clean up orchestrator (handles worker shutdown)
                await self.generation_coordinator.stop()
                self.display_selector.stop()
                
                # Cancel tasks
                for task in [generation_task, display_task, status_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                # Clear task references
                self.running_tasks = []
                self.asyncio_loop = None
        
        else:
            # Legacy GenerationCoordinator (original flow)
            self.logger.info("Using GenerationCoordinator (legacy sequential)")
            
            # Get seed image
            seed_image = self.get_random_seed_image()
            self.logger.info(f"Starting from seed: {seed_image.name}")
            
            # Register and prepare seed as keyframe 1
            self.logger.info("Preparing seed frame as keyframe 1...")
            sequence_num = self.frame_buffer.register_keyframe(1)
            
            # Copy seed to keyframe directory
            target_path = self.frame_buffer.keyframe_dir / "keyframe_001.png"
            Image.open(seed_image).save(target_path)
            self.frame_buffer.mark_ready(sequence_num, target_path)
            
            # Encode seed frame if using VAE
            if self.latent_encoder:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    latent = self.latent_encoder.encode(target_path, for_interpolation=True)
                    self.generation_coordinator.keyframe_latents[1] = latent
                    self.generation_coordinator.keyframe_paths[1] = target_path
                    
                    self.logger.info("  [OK] Seed frame encoded as keyframe 1")
                except Exception as e:
                    self.logger.error(f"  [FAIL] Could not encode seed frame: {e}")
            
            # Set seed in generation coordinator
            self.generation_coordinator.set_seed_image(target_path)
            
            # Mark keyframe 1 as already generated (the seed)
            # This prevents the coordinator from regenerating keyframe 1
            self.generation_coordinator.current_keyframe_num = 1
            self.generation_coordinator.keyframes_generated = 1
            self.logger.info("  [OK] Keyframe 1 registered (seed frame preserved)")
            
            # Clear ComfyUI queue
            self.logger.info("Clearing ComfyUI queue...")
            queue_status = self.generator.client.get_queue()
            if queue_status:
                running_count = len(queue_status.get("queue_running", []))
                pending_count = len(queue_status.get("queue_pending", []))
                if running_count > 0 or pending_count > 0:
                    self.logger.warning(f"Found stale jobs: {running_count} running, {pending_count} pending")
                    self.generator.client.interrupt_execution()
                    self.generator.client.clear_queue()
                    self.logger.info("Queue cleared")
            
            # Start generation and display tasks concurrently
            self.logger.info("Starting generation and display tasks...")
            
            generation_task = asyncio.create_task(self.generation_coordinator.run())
            display_task = asyncio.create_task(self.display_selector.run())
            status_task = asyncio.create_task(self._update_buffer_status_loop())
            
            # Store task references for signal handler
            self.running_tasks = [generation_task, display_task, status_task]
            self.asyncio_loop = asyncio.get_event_loop()
            
            try:
                # Run both tasks concurrently
                await asyncio.gather(generation_task, display_task, status_task)
            except asyncio.CancelledError:
                self.logger.info("Buffered hybrid loop cancelled")
            except KeyboardInterrupt:
                self.logger.info("Buffered hybrid loop interrupted")
            except Exception as e:
                self.logger.error(f"Error in buffered hybrid loop: {e}", exc_info=True)
            finally:
                # Clean up
                self.generation_coordinator.stop()
                self.display_selector.stop()
                
                # Cancel tasks
                for task in [generation_task, display_task, status_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                
                # Clear task references
                self.running_tasks = []
                self.asyncio_loop = None
        
        self.logger.info("=" * 70)
        self.logger.info("BUFFERED HYBRID MODE STOPPED")
        self.logger.info("=" * 70)
    
    async def _update_buffer_status_loop(self) -> None:
        """
        Periodically update status.json with buffer information
        """
        while self.running:
            try:
                # Get buffer status
                buffer_status = self.frame_buffer.get_buffer_status()
                gen_stats = self.generation_coordinator.get_stats()
                display_stats = self.display_selector.get_stats()
                
                # Record buffer status to perf stats (for trend analysis)
                get_perf_stats().record_buffer_status(buffer_status['seconds_buffered'])
                
                # Update status.json with comprehensive stats from new system
                status_data = {
                    "frame_number": display_stats['frames_displayed'],
                    "generation_time": gen_stats['avg_generation_time'],
                    "status": "paused" if self.paused else "live",
                    "current_mode": "hybrid_buffered",
                    "current_prompt": "generating...",
                    "cache_size": gen_stats.get('cache_size', self.cache.size()),
                    "cache_injections": gen_stats.get('cache_injections', 0),
                    "uptime_minutes": round((time.time() - self.start_time) / 60, 1) if self.start_time else 0,
                    # Buffer status
                    "buffer_filled": int(buffer_status['frames_ready']),
                    "buffer_target": int(buffer_status['target_seconds'] * buffer_status.get('target_fps', 4)),
                    "is_buffering": not buffer_status['is_buffer_ready'],
                    "buffer_seconds": buffer_status['seconds_buffered'],
                    "buffer_percentage": buffer_status['buffer_percentage'],
                    # Generation stats
                    "keyframes_generated": gen_stats['keyframes_generated'],
                    "interpolations_generated": gen_stats['interpolations_generated'],
                    # Mode collapse prevention stats (new!)
                    "collapse_recent_similarity": gen_stats.get('collapse_recent_similarity', 0.0),
                    "collapse_overall_similarity": gen_stats.get('collapse_overall_similarity', 0.0),
                    "collapse_frames_analyzed": gen_stats.get('collapse_frames_analyzed', 0),
                    "total_seed_injections": gen_stats.get('total_seed_injections', 0),
                    "collapse_frequency": gen_stats.get('collapse_frequency', 0.0),
                    "cache_diversity_score": gen_stats.get('cache_diversity_score', 0.0),
                    "cache_avg_similarity": gen_stats.get('cache_avg_similarity', 0.0)
                }
                
                # === PROFILING: VAE Lock Stats ===
                if hasattr(self, 'vae_access') and self.vae_access is not None:
                    lock_stats = self.vae_access.get_lock_stats()
                    
                    # Log if significant contention
                    if lock_stats['avg_wait_time_ms'] > 10:
                        self.logger.warning(
                            f"VAE Lock Contention: {lock_stats['acquisitions']} ops, "
                            f"avg wait: {lock_stats['avg_wait_time_ms']:.1f}ms, "
                            f"max wait: {lock_stats['max_wait_time_ms']:.1f}ms"
                        )
                    else:
                        self.logger.debug(
                            f"VAE Lock: {lock_stats['acquisitions']} ops, "
                            f"avg wait: {lock_stats['avg_wait_time_ms']:.1f}ms"
                        )
                    
                    # Add to status.json
                    status_data["vae_lock_acquisitions"] = lock_stats["acquisitions"]
                    status_data["vae_lock_avg_wait_ms"] = lock_stats["avg_wait_time_ms"]
                    status_data["vae_lock_max_wait_ms"] = lock_stats["max_wait_time_ms"]
                    
                    # Reset stats every 10 seconds for moving average
                    if int(time.time()) % 10 == 0:
                        self.vae_access.reset_stats()
                
                self.status_writer.write_status(status_data)
                
                # Log buffer status every 10 seconds
                if int(time.time()) % 10 == 0:
                    self.logger.info(f"Buffer: {buffer_status['seconds_buffered']:.1f}s / {buffer_status['target_seconds']}s "
                                   f"({buffer_status['buffer_percentage']:.1f}%) | "
                                   f"KF: {gen_stats['keyframes_generated']} | "
                                   f"INT: {gen_stats['interpolations_generated']} | "
                                   f"Displayed: {display_stats['frames_displayed']}")
                
                # Log detailed perf summary every 60 seconds (for headroom analysis)
                if int(time.time()) % 60 == 0:
                    get_perf_stats().log_summary()
                
                # Update interval:
                # - Cloud mode: 1s (only for logs, no Rainmeter to update)
                # - Desktop mode: 100ms (10 FPS for smooth Rainmeter updates)
                if self.cloud_enabled:
                    await asyncio.sleep(1.0)  # Less CPU churn in cloud
                else:
                    await asyncio.sleep(0.1)  # Fast updates for Rainmeter
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating buffer status: {e}")
                await asyncio.sleep(1.0)
    
    def run(self, max_frames: Optional[int] = None):
        """
        Main entry point - start the Dream Window!
        
        Args:
            max_frames: Maximum frames to generate (None = infinite)
        """
        self.running = True
        self.start_time = time.time()
        
        # Setup signal handlers for IMMEDIATE shutdown
        def signal_handler(sig, frame):
            self.logger.info("\n[!]  Shutdown signal received - cleaning up immediately...")
            self.running = False
            
            # Set generator shutdown flag to interrupt polling loops
            self.generator._shutdown_requested = True
            
            # Interrupt any running ComfyUI generation
            try:
                self.logger.info("Interrupting ComfyUI execution...")
                self.generator.client.interrupt_execution()
                self.generator.client.clear_queue()
            except Exception as e:
                self.logger.warning(f"Could not interrupt ComfyUI: {e}")
            
            # Cancel all running asyncio tasks if we have a loop
            if self.asyncio_loop and self.running_tasks:
                self.logger.info(f"Cancelling {len(self.running_tasks)} running tasks...")
                for task in self.running_tasks:
                    if not task.done():
                        task.cancel()
            
            # Force exit after a short grace period
            import threading
            def force_exit():
                time.sleep(2.0)  # Give 2 seconds for cleanup
                self.logger.warning("Force exit - cleanup timeout")
                sys.exit(0)
            
            exit_thread = threading.Thread(target=force_exit, daemon=True)
            exit_thread.start()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.logger.info("\n[*] DREAM WINDOW STARTING...")
        self.logger.info(f"Mode: {self.config['generation']['mode']}")
        self.logger.info(f"Press Ctrl+C to stop\n")
        
        try:
            # Choose loop based on mode
            mode = self.config['generation']['mode']
            
            if mode == 'hybrid':
                # Use buffered hybrid loop (modern architecture)
                asyncio.run(self.run_buffered_hybrid_loop())
            else:
                raise ValueError(f"Invalid mode: {mode}")
                
        except KeyboardInterrupt:
            self.logger.info("\n[!]  Stopped by user")
        except Exception as e:
            self.logger.error(f"\nâŒ Fatal error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Clean shutdown"""
        self.logger.info("\n" + "="*70)
        self.logger.info("SHUTTING DOWN")
        self.logger.info("="*70)
        
        # Disconnect from VPS if cloud mode enabled
        if self.cloud_enabled and self.vps_client:
            try:
                # Run async disconnect in event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task if loop is running
                    asyncio.create_task(self.disconnect_from_vps())
                else:
                    # Run directly if loop is not running
                    loop.run_until_complete(self.disconnect_from_vps())
            except Exception as e:
                self.logger.error(f"Error disconnecting from VPS: {e}")
        
        self.logger.info("Shutdown complete")
        
