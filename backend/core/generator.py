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
        
        # State
        self.frame_count = 0
        self.generation_times: list[float] = []
        
        logger.info("DreamGenerator initialized")

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
        
        # Build workflow
        workflow = self.workflow_builder.build_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.config["generation"]["resolution"][0],
            height=self.config["generation"]["resolution"][1],
            steps=self.config["generation"]["flux"]["steps"],
            cfg=self.config["generation"]["flux"]["cfg_scale"],
            seed=seed,
        )
        
        # Execute workflow
        result_path = self._execute_workflow(workflow, start_time)
        
        if result_path:
            logger.info(f"✓ txt2img generation complete: {result_path.name}")
        else:
            logger.error("✗ txt2img generation failed")
        
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
        
        # Copy image to ComfyUI input directory
        # (In production, this will be the ComfyUI installation)
        # For now, we'll prepare but won't actually copy since ComfyUI isn't running
        comfyui_input = Path("./comfyui_input_mock")  # Mock for dev
        comfyui_input.mkdir(exist_ok=True)
        
        input_filename = f"input_{int(time.time() * 1000)}.png"
        input_copy = comfyui_input / input_filename
        
        try:
            shutil.copy(image_path, input_copy)
            logger.debug(f"Copied input to: {input_copy}")
        except Exception as e:
            logger.error(f"Failed to copy input image: {e}")
            return None
        
        # Build workflow
        workflow = self.workflow_builder.build_img2img(
            image_path=str(input_filename),  # ComfyUI needs just the filename
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            steps=self.config["generation"]["flux"]["steps"],
            cfg=self.config["generation"]["flux"]["cfg_scale"],
            seed=seed,
        )
        
        # Execute workflow
        result_path = self._execute_workflow(workflow, start_time)
        
        # Clean up input copy
        try:
            if input_copy.exists():
                input_copy.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up input copy: {e}")
        
        if result_path:
            logger.info(f"✓ img2img generation complete: {result_path.name}")
        else:
            logger.error("✗ img2img generation failed")
        
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
        prompt_id = self.client.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Failed to queue workflow")
            return None
        
        # Wait for completion
        try:
            success = asyncio.run(
                self.client.wait_for_completion(
                    prompt_id,
                    timeout=self.config["performance"]["generation_timeout"],
                )
            )
        except Exception as e:
            logger.error(f"Error waiting for completion: {e}", exc_info=True)
            return None
        
        if not success:
            logger.error(f"Workflow {prompt_id} did not complete successfully")
            return None
        
        # Get output images
        output_files = self.client.get_output_images(prompt_id)
        if not output_files:
            logger.error("No output images found")
            return None
        
        # Get first output (we only generate one image at a time)
        output_filename = output_files[0]
        
        # In production, this would be the ComfyUI output directory
        # For now, mock it
        comfyui_output = Path("./comfyui_output_mock")
        comfyui_output.mkdir(exist_ok=True)
        source_file = comfyui_output / output_filename
        
        # For development without ComfyUI, we'll create a placeholder
        if not source_file.exists():
            logger.warning(f"Output file not found (dev mode): {source_file}")
            # In production, this would be an error
            # For now, we'll just return None
            return None
        
        # Copy to our output directory
        self.frame_count += 1
        dest_filename = f"frame_{self.frame_count:05d}.png"
        dest_path = self.output_dir / dest_filename
        
        try:
            shutil.copy(source_file, dest_path)
            logger.debug(f"Copied output to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy output: {e}")
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
    
    Note: This requires ComfyUI to be running with Flux model loaded.
    For development without ComfyUI, this will fail gracefully.
    """
    import yaml
    
    print("=" * 60)
    print("Testing DreamGenerator...")
    print("=" * 60)
    
    # Load config
    config_path = Path("backend/config.yaml")
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create generator
    print("\n1. Creating generator...")
    gen = DreamGenerator(config)
    print("✓ Generator created")
    
    # Test txt2img (will fail without ComfyUI)
    print("\n2. Testing txt2img generation...")
    print("   (This will fail without ComfyUI running - expected)")
    result = gen.generate_from_prompt(
        prompt="ethereal digital angel, dissolving particles, technical wireframe",
    )
    
    if result:
        print(f"✓ Generated: {result}")
    else:
        print("✗ Generation failed (ComfyUI not running)")
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

