"""
Workflow Builder
Constructs ComfyUI workflow JSON from configuration

ComfyUI workflows are directed graphs represented as JSON:
- Nodes: Operations (load model, sample, encode, etc.)
- Connections: [node_id, output_index] format
- Inputs: Parameters for each node

This module builds workflows programmatically for different generation modes.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """
    Build ComfyUI workflows for multiple models
    
    Supports:
    - SD 1.5 (stable-diffusion-v1-5) - RECOMMENDED for Maxwell GPUs
    - Flux.1-schnell (fast distilled model) - Requires modern GPU
    
    SD1.5 is the primary focus due to VRAM constraints and Maxwell compatibility.
    
    SD1.5 characteristics:
    - 4 inference steps
    - Low CFG (guidance scale ~1.0)
    - Euler sampler with simple scheduler
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize workflow builder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Model selection based on config
        model_type = config.get("generation", {}).get("model", "sd15")
        
        if model_type == "sd15":
            self.model_name = "v1-5-pruned-emaonly.safetensors"
            self.default_steps = 20
            self.default_cfg = 7.0
            self.default_sampler = "euler_a"
        elif model_type == "sd21-unclip":
            self.model_name = "sd21-unclip-h.ckpt"
            self.default_steps = 20
            self.default_cfg = 7.0
            self.default_sampler = "euler_a"
        else:  # flux.1-schnell
            self.model_name = "flux1-schnell.safetensors"
            self.default_steps = 4
            self.default_cfg = 1.0
            self.default_sampler = "euler"
        self.vae_name = "ae.safetensors"  # Flux VAE

    def build_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 256,
        height: int = 512,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build text-to-image workflow
        
        Node flow:
        1. CheckpointLoaderSimple - Load Flux model
        2. CLIPTextEncode (positive) - Encode prompt
        3. CLIPTextEncode (negative) - Encode negative prompt
        4. EmptyLatentImage - Create latent canvas
        5. KSampler - Generate in latent space
        6. VAEDecode - Decode to image
        7. SaveImage - Save output
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt (things to avoid)
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps (4 for Flux schnell)
            cfg: Classifier-free guidance scale (1.0 for Flux)
            seed: Random seed (None = random)
        
        Returns:
            Complete workflow JSON dictionary
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Use model-specific defaults if not provided
        if steps is None:
            steps = self.default_steps
        if cfg is None:
            cfg = self.default_cfg

        workflow = {
            "1": {  # Load Checkpoint
                "inputs": {"ckpt_name": self.model_name},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {  # Positive Prompt
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],  # CLIP from checkpoint
                },
                "class_type": "CLIPTextEncode",
            },
            "3": {  # Negative Prompt
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],  # CLIP from checkpoint
                },
                "class_type": "CLIPTextEncode",
            },
            "4": {  # Empty Latent
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
                "class_type": "EmptyLatentImage",
            },
            "5": {  # KSampler (main generation)
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": self.default_sampler,
                    "scheduler": "simple" if "flux" in self.model_name else "normal",
                    "denoise": 1.0,  # Full generation
                    "model": ["1", 0],  # Model from checkpoint
                    "positive": ["2", 0],  # Positive conditioning
                    "negative": ["3", 0],  # Negative conditioning
                    "latent_image": ["4", 0],  # Empty latent
                },
                "class_type": "KSampler",
            },
            "6": {  # VAE Decode
                "inputs": {
                    "samples": ["5", 0],  # Latent from KSampler
                    "vae": ["1", 2],  # VAE from checkpoint
                },
                "class_type": "VAEDecode",
            },
            "7": {  # Save Image
                "inputs": {
                    "filename_prefix": "dream",
                    "images": ["6", 0],  # Decoded image
                },
                "class_type": "SaveImage",
            },
        }

        logger.debug(f"Built txt2img workflow: {steps} steps, seed={seed}")
        return workflow

    def build_img2img(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float = 0.4,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build image-to-image workflow
        
        This is the KEY workflow for morphing! The denoise parameter controls
        how much the image changes:
        - Low denoise (0.3): Preserves structure, slow evolution
        - Medium denoise (0.4): Balanced morphing (RECOMMENDED)
        - High denoise (0.6+): Rapid changes, might break aesthetic
        
        Node flow:
        1. CheckpointLoaderSimple - Load Flux model
        2. LoadImage - Load input image
        3. VAEEncode - Encode image to latent space
        4. CLIPTextEncode (positive) - Encode prompt
        5. CLIPTextEncode (negative) - Encode negative prompt
        6. KSampler - Generate in latent space (with denoise)
        7. VAEDecode - Decode to image
        8. SaveImage - Save output
        
        Args:
            image_path: Path to input image (filename only, in ComfyUI input/)
            prompt: Generation prompt
            negative_prompt: Negative prompt
            denoise: How much to change (0.0-1.0)
            steps: Number of inference steps
            cfg: Classifier-free guidance scale
            seed: Random seed (None = random)
        
        Returns:
            Complete workflow JSON dictionary
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Use model-specific defaults if not provided
        if steps is None:
            steps = self.default_steps
        if cfg is None:
            cfg = self.default_cfg

        # Extract just the filename for ComfyUI
        image_filename = Path(image_path).name

        workflow = {
            "1": {  # Load Checkpoint
                "inputs": {"ckpt_name": self.model_name},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {  # Load Image
                "inputs": {"image": image_filename},
                "class_type": "LoadImage",
            },
            "3": {  # VAE Encode (image → latent)
                "inputs": {
                    "pixels": ["2", 0],  # Image pixels from LoadImage
                    "vae": ["1", 2],  # VAE from checkpoint
                },
                "class_type": "VAEEncode",
            },
            "4": {  # Positive Prompt
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "5": {  # Negative Prompt
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            "6": {  # KSampler
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": self.default_sampler,
                    "scheduler": "simple" if "flux" in self.model_name else "normal",
                    "denoise": denoise,
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],  # Encoded latent from input
                },
                "class_type": "KSampler",
            },
            "7": {  # VAE Decode
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2],
                },
                "class_type": "VAEDecode",
            },
            "8": {  # Save Image
                "inputs": {
                    "filename_prefix": "dream",
                    "images": ["7", 0],
                },
                "class_type": "SaveImage",
            },
        }

        logger.debug(
            f"Built img2img workflow: denoise={denoise}, steps={steps}, seed={seed}"
        )
        return workflow


def save_workflow(workflow: Dict[str, Any], filename: str) -> None:
    """
    Save workflow to JSON file for inspection/debugging
    
    Args:
        workflow: Workflow dictionary
        filename: Output filename (will be saved in comfyui_workflows/)
    """
    output_dir = Path("comfyui_workflows")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)
    
    logger.info(f"Saved workflow: {output_path}")


# Test function
def test_workflow_builder() -> bool:
    """
    Test workflow generation
    
    Creates example workflows and saves them to JSON files
    """
    print("=" * 60)
    print("Testing Workflow Builder...")
    print("=" * 60)
    
    config = {}  # Empty config for testing
    builder = FluxWorkflowBuilder(config)
    
    # Test 1: txt2img workflow
    print("\nTest 1: txt2img workflow")
    txt2img = builder.build_txt2img(
        prompt="ethereal digital angel, dissolving particles, technical wireframe",
        negative_prompt="photorealistic, blurry, low quality",
        width=256,
        height=512,
    )
    
    if txt2img and "1" in txt2img and "7" in txt2img:
        print(f"✓ txt2img workflow created: {len(txt2img)} nodes")
        save_workflow(txt2img, "workflow_txt2img.json")
    else:
        print("✗ txt2img workflow failed")
        return False
    
    # Test 2: img2img workflow
    print("\nTest 2: img2img workflow")
    img2img = builder.build_img2img(
        image_path="../seeds/background.png",
        prompt="ethereal digital angel, technical wireframe overlay",
        negative_prompt="photorealistic, blurry, low quality",
        denoise=0.4,
    )
    
    if img2img and "1" in img2img and "8" in img2img:
        print(f"✓ img2img workflow created: {len(img2img)} nodes")
        save_workflow(img2img, "workflow_img2img.json")
    else:
        print("✗ img2img workflow failed")
        return False
    
    # Test 3: Different denoise values
    print("\nTest 3: Denoise variations")
    for denoise in [0.3, 0.4, 0.5]:
        workflow = builder.build_img2img(
            image_path="test.png",
            prompt="test",
            denoise=denoise,
        )
        actual_denoise = workflow["6"]["inputs"]["denoise"]
        if actual_denoise == denoise:
            print(f"✓ Denoise {denoise}: correct")
        else:
            print(f"✗ Denoise {denoise}: got {actual_denoise}")
            return False
    
    print("\n" + "=" * 60)
    print("Workflow builder test PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_workflow_builder()
    exit(0 if success else 1)

