"""
Dream Window Main Controller

Entry point for the Dream Window system. Orchestrates:
- Image generation (img2img or hybrid mode)
- Cache management with aesthetic matching
- Prompt rotation
- Status monitoring
- Display output

Run with:
    python backend/main.py
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

from core.generator import DreamGenerator
from utils.prompt_manager import PromptManager
from utils.status_writer import StatusWriter
from utils.file_ops import atomic_write_image_with_retry
from utils.game_detector import GameDetector
from interpolation.hybrid_generator import SimpleHybridGenerator
from cache.manager import CacheManager
from cache.aesthetic_matcher import AestheticMatcher

# Setup logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Configure logging system"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "dream_controller.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
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
        
        # Initialize aesthetic matcher for cache similarity
        self.logger.info("Initializing aesthetic matcher...")
        self.aesthetic_matcher = AestheticMatcher()
        
        # Initialize hybrid mode if enabled
        self.hybrid_generator = None
        if self.config['generation']['mode'] == 'hybrid':
            self.logger.info("Initializing hybrid mode...")
            self.hybrid_generator = SimpleHybridGenerator(
                generator=self.generator,
                keyframe_interval=self.config['generation']['hybrid']['interpolation_frames'],
                keyframe_denoise=self.config['generation']['hybrid']['keyframe_denoise'],
                fill_denoise=self.config['generation']['img2img']['denoise']
            )
        
        # State
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.current_image = None
        self.start_time = None
        self.vram_freed = False  # Track if we've freed VRAM for gaming
        self.last_game_check = 0  # Timestamp of last game detection check
        
        # Statistics
        self.generation_times = []
        self.cache_injections = 0
        
        self.logger.info("[OK] Initialization complete")
        self.logger.info(f"Mode: {self.config['generation']['mode']}")
        self.logger.info(f"Resolution: {self.config['generation']['resolution']}")
        self.logger.info(f"Model: {self.config['generation']['model']}")
        self.logger.info("=" * 70)
    
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
    
    def add_frame_to_cache(self, frame_path: Path, prompt: str, denoise: float) -> bool:
        """
        Add a generated frame to the cache with CLIP embedding
        
        Args:
            frame_path: Path to generated frame
            prompt: Generation prompt used
            denoise: Denoise value used
            
        Returns:
            True if successfully added to cache
        """
        try:
            # Encode image to CLIP embedding
            embedding = self.aesthetic_matcher.encode_image(frame_path)
            
            if embedding is None:
                self.logger.warning(f"Failed to encode image for cache: {frame_path.name}")
                return False
            
            # Add to cache
            generation_params = {
                "denoise": denoise,
                "prompt": prompt,
                "model": self.config["generation"]["model"],
                "resolution": self.config["generation"]["resolution"]
            }
            
            cache_id = self.cache.add(
                image_path=frame_path,
                prompt=prompt,
                generation_params=generation_params,
                embedding=embedding
            )
            
            self.logger.debug(f"Added to cache: {cache_id} (total: {self.cache.size()})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add frame to cache: {e}", exc_info=True)
            return False
    
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
    
    def should_inject_cache(self) -> bool:
        """
        Decide if should inject cached image this frame
        
        Uses probability from config and checks cache has content.
        
        Returns:
            True if should inject cache
        """
        if self.frame_count == 0:
            return False  # Never inject on first frame
        
        # Check cache has entries
        if self.cache.size() == 0:
            return False
        
        # Random probability check
        injection_prob = self.config['generation']['cache']['injection_probability']
        return random.random() < injection_prob
    
    def inject_cached_frame(self) -> Optional[Path]:
        """
        Inject similar cached image based on current aesthetic
        
        Returns:
            Path to cached image, or None if injection fails
        """
        if self.current_image is None:
            return None
        
        try:
            # Encode current frame
            current_embedding = self.aesthetic_matcher.encode_image(self.current_image)
            
            if current_embedding is None:
                self.logger.warning("Failed to encode current frame for cache injection")
                return None
            
            # Get all cached entries
            cache_entries = self.cache.get_all()
            if not cache_entries:
                self.logger.debug("Cache is empty, no injection possible")
                return None
            
            # Prepare candidates (cache_id, embedding) pairs
            candidates = [
                (entry.cache_id, entry.embedding)
                for entry in cache_entries
                if entry.embedding is not None
            ]
            
            if not candidates:
                self.logger.debug("No cache entries with embeddings")
                return None
            
            # Find similar cached images
            threshold = self.config['generation']['cache']['similarity_threshold']
            similar = self.aesthetic_matcher.find_similar(
                target_embedding=current_embedding,
                candidate_embeddings=candidates,
                threshold=threshold,
                top_k=5
            )
            
            if not similar:
                self.logger.debug(f"No similar images found (threshold: {threshold})")
                return None
            
            # Weighted random selection from similar images
            selected_cache_id, similarity = self.aesthetic_matcher.weighted_random_selection(similar)
            
            # Get the cached image path
            cached_entry = self.cache.get(selected_cache_id)
            if cached_entry:
                self.cache_injections += 1
                self.logger.info(f"[CACHE] INJECTION #{self.cache_injections}: {selected_cache_id} (similarity: {similarity:.3f})")
                return cached_entry.image_path
            
        except Exception as e:
            self.logger.error(f"Cache injection failed: {e}", exc_info=True)
        
        return None
    
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
            status_data = {
                "frame_number": self.frame_count,
                "generation_time": round(generation_time, 2),
                "status": "paused" if self.paused else "live",
                "current_mode": mode,
                "current_prompt": prompt[:100],  # Truncate long prompts
                "cache_size": self.cache.size(),
                "cache_injections": self.cache_injections,
                "uptime_minutes": round((time.time() - self.start_time) / 60, 1) if self.start_time else 0
            }
            
            self.status_writer.write_status(status_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update status: {e}")
    
    async def run_img2img_loop(self, max_frames: Optional[int] = None):
        """
        Run continuous img2img feedback loop
        
        Each frame is generated from the previous frame with cache injection.
        
        Args:
            max_frames: Maximum frames to generate (None = infinite)
        """
        self.logger.info("Starting img2img feedback loop")
        self.logger.info(f"Target: {'infinite' if max_frames is None else max_frames} frames")
        
        # Start with random seed
        self.current_image = self.get_random_seed_image()
        self.logger.info(f"Starting from seed: {self.current_image.name}")
        
        # Copy seed to output as frame 0
        dest = self.output_dir / f"frame_{self.frame_count:05d}.png"
        Image.open(self.current_image).save(dest)
        self.write_current_frame(dest)
        self.frame_count += 1
        
        # Main generation loop
        while self.running:
            if max_frames and self.frame_count >= max_frames:
                break
            
            try:
                # Check for game detection (pause + free VRAM if needed)
                if self.check_game_state():
                    # Game is running - skip generation
                    await asyncio.sleep(2.0)
                    continue
                
                # Check for cache injection
                if self.should_inject_cache():
                    cached_image = self.inject_cached_frame()
                    if cached_image and cached_image.exists():
                        # Use cached image
                        dest = self.output_dir / f"frame_{self.frame_count:05d}.png"
                        Image.open(cached_image).copy().save(dest)
                        self.write_current_frame(dest)
                        self.current_image = dest
                        self.frame_count += 1
                        
                        # Update status
                        self.update_status(0.0, "cache_injection", "cached image")
                        
                        await asyncio.sleep(1.0)
                        continue
                
                # Normal generation
                prompt = self.prompt_manager.get_next_prompt()
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"Frame {self.frame_count}")
                self.logger.info(f"Prompt: {prompt[:60]}...")
                
                start_time = time.time()
                
                # Generate frame
                new_frame = self.generator.generate_from_image(
                    image_path=self.current_image,
                    prompt=prompt,
                    denoise=self.config['generation']['img2img']['denoise']
                )
                
                elapsed = time.time() - start_time
                
                if new_frame and new_frame.exists():
                    # Success
                    self.generation_times.append(elapsed)
                    self.current_image = new_frame
                    self.frame_count += 1
                    
                    # Write to display
                    self.write_current_frame(new_frame)
                    
                    # Update status
                    self.update_status(elapsed, "img2img", prompt)
                    
                    # Log stats
                    avg_time = sum(self.generation_times[-10:]) / min(10, len(self.generation_times))
                    self.logger.info(f"[OK] Generated in {elapsed:.2f}s (avg: {avg_time:.2f}s)")
                    self.logger.info(f"Cache: {self.cache.size()}/{self.config['generation']['cache']['max_size']}")
                    
                    # Small pause
                    await asyncio.sleep(1.0)
                    
                else:
                    self.logger.error("Generation failed, retrying...")
                    await asyncio.sleep(5.0)
                    
            except KeyboardInterrupt:
                self.logger.info("\n[!]  Interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Loop complete: {self.frame_count} frames generated")
        self.logger.info(f"Cache injections: {self.cache_injections}")
        if self.generation_times:
            avg_time = sum(self.generation_times) / len(self.generation_times)
            self.logger.info(f"Average generation time: {avg_time:.2f}s")
    
    async def run_hybrid_loop(self, max_frames: Optional[int] = None):
        """
        Run hybrid generation loop
        
        Alternates between keyframes and fill frames for smooth morphing.
        
        Args:
            max_frames: Maximum frames to generate (None = infinite)
        """
        if self.hybrid_generator is None:
            self.logger.error("Hybrid generator not initialized!")
            return
        
        self.logger.info("Starting hybrid generation loop")
        self.logger.info(f"Keyframe interval: {self.hybrid_generator.keyframe_interval}")
        self.logger.info(f"Target: {'infinite' if max_frames is None else max_frames} frames")
        
        # Clear any stale jobs from queue before starting
        self.logger.info("Clearing ComfyUI queue...")
        queue_status = self.generator.client.get_queue()
        if queue_status:
            running_count = len(queue_status.get("queue_running", []))
            pending_count = len(queue_status.get("queue_pending", []))
            if running_count > 0 or pending_count > 0:
                self.logger.warning(f"Found stale jobs: {running_count} running, {pending_count} pending")
                self.generator.client.interrupt_execution()
                self.generator.client.clear_queue()
                self.logger.info("Queue cleared successfully")
            else:
                self.logger.info("Queue is empty - ready to start")
        
        # Start with random seed
        self.current_image = self.get_random_seed_image()
        self.logger.info(f"Starting from seed: {self.current_image.name}")
        
        # Copy seed as frame 0
        dest = self.output_dir / f"frame_{self.frame_count:05d}.png"
        Image.open(self.current_image).save(dest)
        self.write_current_frame(dest)
        self.frame_count += 1
        
        # Main generation loop
        while self.running:
            if max_frames and self.frame_count >= max_frames:
                break
            
            try:
                # Check for game detection (pause + free VRAM if needed)
                if self.check_game_state():
                    # Game is running - skip generation
                    await asyncio.sleep(2.0)
                    continue
                
                # Check for cache injection
                if self.should_inject_cache():
                    cached_image = self.inject_cached_frame()
                    if cached_image and cached_image.exists():
                        dest = self.output_dir / f"frame_{self.frame_count:05d}.png"
                        Image.open(cached_image).copy().save(dest)
                        self.write_current_frame(dest)
                        self.current_image = dest
                        self.frame_count += 1
                        
                        self.update_status(0.0, "cache_injection", "cached image")
                        await asyncio.sleep(1.0)
                        continue
                
                # Generate next frame via hybrid mode
                prompt = self.prompt_manager.get_next_prompt()
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"Frame {self.frame_count}")
                
                start_time = time.time()
                
                new_frame = self.hybrid_generator.generate_next_frame(
                    current_image=self.current_image,
                    prompt=prompt,
                    frame_number=self.frame_count
                )
                
                elapsed = time.time() - start_time
                
                if new_frame and new_frame.exists():
                    self.generation_times.append(elapsed)
                    self.current_image = new_frame
                    self.frame_count += 1
                    
                    self.write_current_frame(new_frame)
                    
                    is_keyframe = ((self.frame_count - 1) % self.hybrid_generator.keyframe_interval == 0)
                    mode = "hybrid_keyframe" if is_keyframe else "hybrid_fill"
                    self.update_status(elapsed, mode, prompt)
                    
                    # Add keyframes to cache (not fill frames to save processing time)
                    if is_keyframe:
                        denoise = self.hybrid_generator.keyframe_denoise
                        self.add_frame_to_cache(new_frame, prompt, denoise)
                    
                    avg_time = sum(self.generation_times[-10:]) / min(10, len(self.generation_times))
                    self.logger.info(f"[OK] Generated in {elapsed:.2f}s (avg: {avg_time:.2f}s)")
                    self.logger.info(f"Cache: {self.cache.size()}/{self.config['generation']['cache']['max_size']}")
                    
                    await asyncio.sleep(1.0)
                else:
                    self.logger.error("Generation failed, retrying...")
                    await asyncio.sleep(5.0)
                    
            except KeyboardInterrupt:
                self.logger.info("\n[!]  Interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Loop complete: {self.frame_count} frames generated")
        self.logger.info(f"Cache injections: {self.cache_injections}")
        if self.generation_times:
            avg_time = sum(self.generation_times) / len(self.generation_times)
            self.logger.info(f"Average generation time: {avg_time:.2f}s")
    
    def run(self, max_frames: Optional[int] = None):
        """
        Main entry point - start the Dream Window!
        
        Args:
            max_frames: Maximum frames to generate (None = infinite)
        """
        self.running = True
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self.logger.info("\n[!]  Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.logger.info("\n[*] DREAM WINDOW STARTING...")
        self.logger.info(f"Mode: {self.config['generation']['mode']}")
        self.logger.info(f"Press Ctrl+C to stop\n")
        
        try:
            # Choose loop based on mode
            mode = self.config['generation']['mode']
            
            if mode == 'hybrid':
                asyncio.run(self.run_hybrid_loop(max_frames))
            else:  # 'img2img' or fallback
                asyncio.run(self.run_img2img_loop(max_frames))
                
        except KeyboardInterrupt:
            self.logger.info("\n[!]  Stopped by user")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Clean shutdown"""
        self.logger.info("\n" + "="*70)
        self.logger.info("SHUTTING DOWN")
        self.logger.info("="*70)
        
        self.running = False
        
        # Final statistics
        if self.generation_times:
            total_time = sum(self.generation_times)
            avg_time = total_time / len(self.generation_times)
            self.logger.info(f"Total frames: {self.frame_count}")
            self.logger.info(f"Total generation time: {total_time/60:.1f} minutes")
            self.logger.info(f"Average per frame: {avg_time:.2f}s")
            self.logger.info(f"Cache injections: {self.cache_injections}")
            self.logger.info(f"Final cache size: {self.cache.size()}")
        
        self.logger.info("[OK] Shutdown complete")
        self.logger.info("="*70)


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dream Window - AI Desktop Art Generator")
    parser.add_argument(
        "--config",
        default="backend/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to generate (default: infinite)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run short test (10 frames)"
    )
    
    args = parser.parse_args()
    
    # Override for test mode
    if args.test:
        args.max_frames = 10
        print("[TEST MODE] Generating 10 frames")
    
    # Create and run controller
    controller = DreamController(config_path=args.config)
    controller.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()

