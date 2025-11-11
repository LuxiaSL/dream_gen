"""
Image Generator
High-level interface for image generation

This module provides a clean API for generating images through ComfyUI,
abstracting away the complexity of workflow building and API communication.
"""

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

from .comfyui_api import ComfyUIClient
from .workflow_builder import WorkflowBuilder

logger = logging.getLogger(__name__)


class DreamGenerator:
    """
    High-level image generation controller
    
    Responsibilities:
    - Abstract ComfyUI workflow details
    - Manage generation modes (txt2img, img2img)
    - Handle file operations (atomic writes, copying)
    - Track performance metrics
    
    This is the main interface that the controller will use.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generator with configuration
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        
        # Initialize subsystems
        self.client = ComfyUIClient(
            base_url=config["system"]["comfyui_url"]
        )
        self.workflow_builder = WorkflowBuilder(config)
        
        # Paths
        self.output_dir = Path(config["system"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for resized images
        self.temp_dir = self.output_dir / ".temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Get target resolution from config
        self.target_width, self.target_height = config["generation"]["resolution"]
        
        # State
        self.frame_count = 0
        self.generation_times: list[float] = []
        self._shutdown_requested = False  # Flag to interrupt long-running operations
        
        logger.info("DreamGenerator initialized")

    def _resize_image_for_generation(self, image_path: Path) -> Path:
        """
        Resize image to target resolution before uploading to ComfyUI.
        
        This prevents processing oversized images that can cause:
        - Excessive VRAM usage
        - Very slow generation times
        - System crashes
        
        Args:
            image_path: Path to original image
            
        Returns:
            Path to resized temp image (ready for upload)
        """
        # Open and check current size
        img = Image.open(image_path)
        current_w, current_h = img.size
        
        # If already correct size, just return original
        if current_w == self.target_width and current_h == self.target_height:
            logger.debug(f"Image already correct size: {current_w}x{current_h}")
            return image_path
        
        # Resize to target resolution
        logger.info(f"Resizing image from {current_w}x{current_h} to {self.target_width}x{self.target_height}")
        img_resized = img.resize(
            (self.target_width, self.target_height),
            Image.Resampling.LANCZOS  # High-quality downsampling
        )
        
        # Save to temp directory with unique name
        temp_path = self.temp_dir / f"resized_{image_path.name}"
        img_resized.save(temp_path, format='PNG', optimize=True)
        
        logger.debug(f"Resized image saved to: {temp_path}")
        return temp_path

    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Generate image from text prompt (txt2img)
        
        This is used for the initial seed frame or when starting fresh.
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt (uses config default if None)
            seed: Random seed (None = random)
        
        Returns:
            Path to generated image, or None on failure
        """
        start_time = time.time()
        
        if negative_prompt is None:
            negative_prompt = self.config["prompts"]["negative"]
        
        logger.info(f"Generating from prompt: {prompt[:60]}...")
        
        # Get model-specific config
        model_type = self.config.get("generation", {}).get("model", "sd15")
        if model_type == "flux.1-schnell":
            model_config = self.config["generation"]["flux"]
        else:  # sd15 or sd21-unclip
            model_config = self.config["generation"]["sd"]
        
        # Build workflow
        workflow = self.workflow_builder.build_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.config["generation"]["resolution"][0],
            height=self.config["generation"]["resolution"][1],
            steps=model_config["steps"],
            cfg=model_config["cfg_scale"],
            seed=seed,
        )
        
        # Execute workflow
        result_path = self._execute_workflow(workflow, start_time)
        
        if result_path:
            logger.info(f"[OK] txt2img generation complete: {result_path.name}")
        else:
            logger.error("[FAIL] txt2img generation failed")
        
        return result_path

    def generate_from_image(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        denoise: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Generate image from existing image (img2img)
        
        This is the CORE of the morphing effect! Each frame is generated
        from the previous frame with a denoise value that controls how
        much the image changes.
        
        Args:
            image_path: Source image path
            prompt: Generation prompt
            negative_prompt: Negative prompt (uses config default if None)
            denoise: Denoise strength 0.0-1.0 (uses config default if None)
            seed: Random seed (None = random)
        
        Returns:
            Path to generated image, or None on failure
        """
        start_time = time.time()
        
        if not image_path.exists():
            logger.error(f"Source image not found: {image_path}")
            return None
        
        if negative_prompt is None:
            negative_prompt = self.config["prompts"]["negative"]
        
        if denoise is None:
            denoise = self.config["generation"]["img2img"]["denoise"]
        
        logger.info(f"Generating from image: {image_path.name} (denoise={denoise})")
        logger.debug(f"Prompt: {prompt[:60]}...")
        
        # Resize image to target resolution before uploading
        # This prevents processing huge images that cause crashes/slowdowns
        resized_image_path = self._resize_image_for_generation(image_path)
        
        # Upload image to ComfyUI via API
        upload_result = self.client.upload_image(
            image_path=resized_image_path,
            subfolder="",
            image_type="input",
            overwrite=False,  # Allow duplicate detection
        )
        
        if not upload_result:
            logger.error("Failed to upload image to ComfyUI")
            return None
        
        # Get the uploaded filename (may be different if deduplicated)
        input_filename = upload_result["name"]
        logger.debug(f"Uploaded image as: {input_filename}")
        
        # Get model-specific config
        model_type = self.config.get("generation", {}).get("model", "sd15")
        if model_type == "flux.1-schnell":
            model_config = self.config["generation"]["flux"]
        else:  # sd15 or sd21-unclip
            model_config = self.config["generation"]["sd"]
        
        # Build workflow
        # Pass the uploaded filename to the workflow builder
        workflow = self.workflow_builder.build_img2img(
            image_path=input_filename,  # Use the uploaded filename from ComfyUI
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            steps=model_config["steps"],
            cfg=model_config["cfg_scale"],
            seed=seed,
        )
        
        # Execute workflow
        result_path = self._execute_workflow(workflow, start_time)
        
        if result_path:
            logger.info(f"[OK] img2img generation complete: {result_path.name}")
        else:
            logger.error("[FAIL] img2img generation failed")
        
        return result_path

    def _execute_workflow(
        self,
        workflow: Dict[str, Any],
        start_time: float,
    ) -> Optional[Path]:
        """
        Execute a workflow and retrieve the result
        
        This handles the full workflow execution cycle:
        1. Queue the workflow
        2. Wait for completion (via WebSocket)
        3. Retrieve output images
        4. Copy to our output directory
        5. Track performance
        
        Args:
            workflow: ComfyUI workflow JSON
            start_time: When generation started (for timing)
        
        Returns:
            Path to generated image in our output directory
        """
        # Queue prompt
        logger.debug("Queueing workflow to ComfyUI...")
        prompt_id = self.client.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue workflow")
            return None
        
        logger.info(f"Workflow queued with prompt_id: {prompt_id}")
        
        # Wait for completion (use polling instead of async)
        try:
            import time
            timeout = self.config["performance"]["generation_timeout"]
            poll_start = time.time()
            poll_count = 0
            last_log_time = poll_start
            
            logger.debug(f"Polling for completion (timeout: {timeout}s)...")
            
            while True:
                poll_count += 1
                
                # Check for shutdown request
                if self._shutdown_requested:
                    logger.warning(f"Shutdown requested - aborting generation (prompt_id: {prompt_id})")
                    # Try to interrupt the running generation
                    try:
                        self.client.interrupt_execution()
                    except:
                        pass
                    return None
                
                # Check queue status
                queue = self.client.get_queue()
                if not queue:
                    logger.error("Failed to get queue status - ComfyUI may be unresponsive")
                    return None
                
                # Check if our prompt is still in queue
                running = [item for item in queue.get("queue_running", []) if item[1] == prompt_id]
                pending = [item for item in queue.get("queue_pending", []) if item[1] == prompt_id]
                
                # Log progress every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    elapsed = current_time - poll_start
                    total_running = len(queue.get("queue_running", []))
                    total_pending = len(queue.get("queue_pending", []))
                    logger.info(f"Still waiting... ({elapsed:.1f}s) - Queue: {total_running} running, {total_pending} pending")
                    if running or pending:
                        logger.debug(f"Our prompt {prompt_id}: {'in running' if running else ''} {'in pending' if pending else ''}")
                    else:
                        logger.debug(f"Our prompt {prompt_id} not found in queue - may have completed")
                    last_log_time = current_time
                
                if not running and not pending:
                    # Prompt completed (or was never in queue)
                    elapsed = current_time - poll_start
                    logger.info(f"Prompt no longer in queue after {elapsed:.1f}s ({poll_count} polls)")
                    break
                
                # Check timeout
                if current_time - poll_start > timeout:
                    logger.error(f"Generation timeout ({timeout}s) - prompt appears stuck")
                    logger.error(f"Last queue state: {len(running)} running, {len(pending)} pending")
                    logger.error(f"Prompt ID: {prompt_id}")
                    return None
                
                # Wait a bit before checking again
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error waiting for completion: {e}", exc_info=True)
            return None
        
        # Get output images
        output_files = self.client.get_output_images(prompt_id)
        if not output_files:
            logger.error("No output images found")
            return None
        
        # Get first output (we only generate one image at a time)
        output_filename = output_files[0]
        
        # Get image data directly from ComfyUI API (no file system access needed!)
        image_data = self.client.get_image_data(output_filename)
        if not image_data:
            logger.error(f"Failed to retrieve image data for: {output_filename}")
            return None
        
        # Save to our output directory
        self.frame_count += 1
        dest_filename = f"frame_{self.frame_count:05d}.png"
        dest_path = self.output_dir / dest_filename
        
        try:
            # Write image bytes directly
            with open(dest_path, 'wb') as f:
                f.write(image_data)
            logger.debug(f"Saved output to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return None
        
        # Track performance
        elapsed = time.time() - start_time
        self.generation_times.append(elapsed)
        if len(self.generation_times) > 100:
            self.generation_times.pop(0)
        
        logger.info(f"Generation time: {elapsed:.2f}s")
        
        return dest_path

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with avg/min/max generation times
        """
        if not self.generation_times:
            return {}
        
        return {
            "avg_time": sum(self.generation_times) / len(self.generation_times),
            "min_time": min(self.generation_times),
            "max_time": max(self.generation_times),
            "total_frames": self.frame_count,
        }

    def close(self) -> None:
        """Clean up resources"""
        self.client.close()
        logger.info("DreamGenerator closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Test function (will work once ComfyUI is running)
async def test_generator() -> bool:
    """
    Test generator interface
    
    Note: This requires ComfyUI to be running with the configured model loaded.
    By default uses SD1.5, but respects config.yaml model setting.
    For development without ComfyUI, this will fail gracefully.
    """
    import yaml
    
    print("=" * 60)
    print("Testing DreamGenerator...")
    print("=" * 60)
    
    # Load config
    config_path = Path("backend/config.yaml")
    if not config_path.exists():
        print("[FAIL] config.yaml not found")
        return False
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create generator
    print("\n1. Creating generator...")
    gen = DreamGenerator(config)
    print("[OK] Generator created")
    
    # Test txt2img (will fail without ComfyUI)
    print("\n2. Testing txt2img generation...")
    print("   (This will fail without ComfyUI running - expected)")
    result = gen.generate_from_prompt(
        prompt="ethereal digital angel, dissolving particles, technical wireframe",
    )
    
    if result:
        print(f"[OK] Generated: {result}")
    else:
        print("[FAIL] Generation failed (ComfyUI not running)")
        print("   This is expected in development without ComfyUI")
    
    # Print stats
    stats = gen.get_performance_stats()
    if stats:
        print(f"\n3. Performance stats:")
        print(f"   Avg time: {stats['avg_time']:.2f}s")
        print(f"   Frames: {stats['total_frames']}")
    
    gen.close()
    
    print("\n" + "=" * 60)
    print("Generator test completed")
    print("(Full functionality requires ComfyUI)")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    success = asyncio.run(test_generator())
    exit(0 if success else 1)

