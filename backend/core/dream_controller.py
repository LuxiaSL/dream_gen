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
from core.display_selector import DisplayFrameSelector
from utils.prompt_manager import PromptManager
from utils.status_writer import StatusWriter
from utils.file_ops import atomic_write_image_with_retry
from utils.game_detector import GameDetector
from cache.manager import CacheManager
from cache.dual_similarity import DualMetricSimilarityManager

# Setup logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Configure logging system with rotation"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "dream_controller.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Rotating file handler (max 5MB per file, keep 3 backups)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3  # Keep 3 backup files (dream_controller.log.1, .2, .3)
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
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
        self.prompt_manager = PromptManager(self.config)
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
                
                # Use same GPU as ComfyUI for consistency
                gpu_id = self.config.get('system', {}).get('gpu_id', 0)
                device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cuda"
                
                # Get target resolution from config to force resize
                target_resolution = tuple(self.config['generation']['resolution'])  # [width, height]
                
                self.latent_encoder = LatentEncoder(
                    device=device,
                    auto_load=True,
                    interpolation_resolution_divisor=resolution_divisor,
                    upscale_method=upscale_method,
                    target_resolution=target_resolution
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
            
            # Create generation coordinator
            self.generation_coordinator = GenerationCoordinator(
                frame_buffer=self.frame_buffer,
                generator=self.generator,
                latent_encoder=self.latent_encoder,
                prompt_manager=self.prompt_manager,
                config=self.config,
                cache_manager=self.cache,
                similarity_manager=self.similarity_manager
            )
            
            # Create display selector
            self.display_selector = DisplayFrameSelector(
                frame_buffer=self.frame_buffer,
                output_dir=self.output_dir,
                target_fps=target_fps,
                min_buffer_seconds=min_buffer_seconds
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
        
        Uses FrameBuffer, GenerationCoordinator, and DisplayFrameSelector
        to maintain a 30s buffer of frames for smooth, uninterrupted playback.
        """
        if not self.frame_buffer or not self.generation_coordinator or not self.display_selector:
            self.logger.error("Buffered frame system not initialized!")
            return
        
        self.logger.info("=" * 70)
        self.logger.info("STARTING BUFFERED HYBRID MODE")
        self.logger.info("=" * 70)
        
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
                
                self.status_writer.write_status(status_data)
                
                # Log buffer status every 10 seconds
                if int(time.time()) % 10 == 0:
                    self.logger.info(f"Buffer: {buffer_status['seconds_buffered']:.1f}s / {buffer_status['target_seconds']}s "
                                   f"({buffer_status['buffer_percentage']:.1f}%) | "
                                   f"KF: {gen_stats['keyframes_generated']} | "
                                   f"INT: {gen_stats['interpolations_generated']} | "
                                   f"Displayed: {display_stats['frames_displayed']}")
                
                await asyncio.sleep(1.0)
                
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
        
