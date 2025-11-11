"""
Profiling Script for VAE Interpolation Performance

This script profiles each component of the interpolation pipeline to identify
bottlenecks and measure optimization improvements.

Usage:
    uv run backend/tools/profile_interpolation.py
    uv run backend/tools/profile_interpolation.py --iterations 10
"""

import torch
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import numpy as np
from PIL import Image
import argparse

# Add backend directory to path for imports (we're in backend/tools/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpolation.latent_encoder import LatentEncoder
from interpolation.spherical_lerp import spherical_lerp
from utils.file_ops import atomic_write_image_with_retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUTimer:
    """Context manager for accurate GPU timing with synchronization"""
    
    def __init__(self, name: str, sync: bool = True):
        self.name = name
        self.sync = sync
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated
    }


def profile_encode(encoder: LatentEncoder, image: Image.Image, iterations: int = 10) -> Dict[str, float]:
    """Profile VAE encoding: image → latent"""
    logger.info(f"\n{'='*60}")
    logger.info("Profiling VAE Encode (image → latent)")
    logger.info(f"{'='*60}")
    
    times = {
        "preprocess": [],
        "vae_encode": [],
        "total": []
    }
    
    for i in range(iterations):
        # Total encode time
        with GPUTimer("total_encode") as timer:
            latent = encoder.encode(image)
        times["total"].append(timer.elapsed)
        
        # Now profile sub-components by calling them directly
        with GPUTimer("preprocess") as timer:
            img_tensor = encoder._preprocess_image(image)
        times["preprocess"].append(timer.elapsed)
        
        with GPUTimer("vae_encode") as timer:
            with torch.no_grad():
                latent_dist = encoder.vae.encode(img_tensor).latent_dist
                latent = latent_dist.sample()
                latent = latent * encoder.vae_scale_factor
        times["vae_encode"].append(timer.elapsed)
        
        if i == 0:
            logger.info(f"Latent shape: {latent.shape}")
            logger.info(f"Latent device: {latent.device}")
            logger.info(f"Latent dtype: {latent.dtype}")
    
    # Calculate statistics
    results = {}
    for key, values in times.items():
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        results[key] = {
            "avg": avg,
            "std": std,
            "min": min_val,
            "max": max_val
        }
        logger.info(f"{key:15s}: {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms]")
    
    return results


def profile_decode(encoder: LatentEncoder, latent: torch.Tensor, iterations: int = 10) -> Dict[str, float]:
    """Profile VAE decoding: latent → image"""
    logger.info(f"\n{'='*60}")
    logger.info("Profiling VAE Decode (latent → image)")
    logger.info(f"{'='*60}")
    
    times = {
        "vae_decode": [],
        "postprocess": [],
        "total": []
    }
    
    for i in range(iterations):
        # Total decode time
        with GPUTimer("total_decode") as timer:
            image = encoder.decode(latent)
        times["total"].append(timer.elapsed)
        
        # Profile sub-components
        with GPUTimer("vae_decode") as timer:
            with torch.no_grad():
                unscaled = latent / encoder.vae_scale_factor
                image_tensor = encoder.vae.decode(unscaled).sample
        times["vae_decode"].append(timer.elapsed)
        
        with GPUTimer("postprocess") as timer:
            _ = encoder._postprocess_image(image_tensor)
        times["postprocess"].append(timer.elapsed)
        
        if i == 0:
            logger.info(f"Image size: {image.size}")
            logger.info(f"Image mode: {image.mode}")
    
    # Calculate statistics
    results = {}
    for key, values in times.items():
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        results[key] = {
            "avg": avg,
            "std": std,
            "min": min_val,
            "max": max_val
        }
        logger.info(f"{key:15s}: {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms]")
    
    return results


def profile_slerp(latent_a: torch.Tensor, latent_b: torch.Tensor, iterations: int = 10) -> Dict[str, float]:
    """Profile spherical interpolation"""
    logger.info(f"\n{'='*60}")
    logger.info("Profiling Spherical Lerp (slerp)")
    logger.info(f"{'='*60}")
    
    times = []
    
    # Test at t=0.5
    for i in range(iterations):
        with GPUTimer("slerp") as timer:
            result = spherical_lerp(latent_a, latent_b, t=0.5)
        times.append(timer.elapsed)
        
        if i == 0:
            logger.info(f"Input latent A device: {latent_a.device}, dtype: {latent_a.dtype}")
            logger.info(f"Input latent B device: {latent_b.device}, dtype: {latent_b.dtype}")
            logger.info(f"Output latent device: {result.device}, dtype: {result.dtype}")
    
    avg = np.mean(times)
    std = np.std(times)
    min_val = np.min(times)
    max_val = np.max(times)
    
    logger.info(f"slerp          : {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms]")
    
    return {
        "avg": avg,
        "std": std,
        "min": min_val,
        "max": max_val
    }


def profile_image_save(image: Image.Image, output_path: Path, iterations: int = 10) -> Dict[str, float]:
    """Profile image save operations"""
    logger.info(f"\n{'='*60}")
    logger.info("Profiling Image Save (PIL → disk)")
    logger.info(f"{'='*60}")
    
    times = {
        "save_optimized": [],
        "save_fast": []
    }
    
    # Test optimized save (current method)
    for i in range(iterations):
        with GPUTimer("save_optimized", sync=False) as timer:
            image.save(output_path, "PNG", optimize=True)
        times["save_optimized"].append(timer.elapsed)
    
    # Test fast save (no optimization)
    for i in range(iterations):
        with GPUTimer("save_fast", sync=False) as timer:
            image.save(output_path, "PNG", optimize=False)
        times["save_fast"].append(timer.elapsed)
    
    # Calculate statistics
    results = {}
    for key, values in times.items():
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        results[key] = {
            "avg": avg,
            "std": std,
            "min": min_val,
            "max": max_val
        }
        logger.info(f"{key:15s}: {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms]")
    
    return results


def profile_full_pipeline(encoder: LatentEncoder, image_a: Image.Image, image_b: Image.Image, 
                          output_path: Path, iterations: int = 5) -> Dict[str, float]:
    """Profile complete interpolation pipeline: image A + B → interpolated frame"""
    logger.info(f"\n{'='*60}")
    logger.info("Profiling FULL INTERPOLATION PIPELINE")
    logger.info(f"{'='*60}")
    
    times = {
        "encode_a": [],
        "encode_b": [],
        "slerp": [],
        "decode": [],
        "save": [],
        "total": []
    }
    
    for i in range(iterations):
        with GPUTimer("total_pipeline") as timer_total:
            # Encode image A
            with GPUTimer("encode_a") as timer:
                latent_a = encoder.encode(image_a)
            times["encode_a"].append(timer.elapsed)
            
            # Encode image B
            with GPUTimer("encode_b") as timer:
                latent_b = encoder.encode(image_b)
            times["encode_b"].append(timer.elapsed)
            
            # Interpolate
            with GPUTimer("slerp") as timer:
                latent_interp = spherical_lerp(latent_a, latent_b, t=0.5)
            times["slerp"].append(timer.elapsed)
            
            # Decode
            with GPUTimer("decode") as timer:
                image_interp = encoder.decode(latent_interp)
            times["decode"].append(timer.elapsed)
            
            # Save
            with GPUTimer("save", sync=False) as timer:
                image_interp.save(output_path, "PNG", optimize=True)
            times["save"].append(timer.elapsed)
        
        times["total"].append(timer_total.elapsed)
    
    # Calculate statistics
    results = {}
    logger.info("\nBreakdown:")
    for key, values in times.items():
        avg = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        results[key] = {
            "avg": avg,
            "std": std,
            "min": min_val,
            "max": max_val
        }
        
        # Calculate percentage of total
        if key != "total":
            pct = (avg / results.get("total", {}).get("avg", avg)) * 100 if "total" in results else 0
            logger.info(f"{key:15s}: {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms] ({pct:5.1f}%)")
        else:
            logger.info(f"{key:15s}: {avg*1000:6.2f}ms (±{std*1000:4.2f}ms) [{min_val*1000:6.2f}-{max_val*1000:6.2f}ms]")
    
    # FPS calculation
    avg_total = results["total"]["avg"]
    fps = 1.0 / avg_total if avg_total > 0 else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"CURRENT PERFORMANCE: {avg_total*1000:.2f}ms per frame = {fps:.2f} FPS")
    logger.info(f"TARGET: 10-15 FPS (67-100ms per frame)")
    logger.info(f"SPEEDUP NEEDED: {avg_total / 0.1:.2f}x")
    logger.info(f"{'='*60}")
    
    return results


def check_gpu_utilization(encoder: LatentEncoder):
    """Verify VAE model is properly on GPU"""
    logger.info(f"\n{'='*60}")
    logger.info("GPU Utilization Check")
    logger.info(f"{'='*60}")
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
    
    # Check VAE model location
    if encoder.vae is not None:
        logger.info(f"\nVAE Model:")
        logger.info(f"  Device: {encoder.device}")
        
        # Check if model parameters are on GPU
        param_devices = set()
        for name, param in encoder.vae.named_parameters():
            param_devices.add(str(param.device))
        
        logger.info(f"  Parameter devices: {param_devices}")
        
        # Check model dtype
        first_param = next(encoder.vae.parameters())
        logger.info(f"  Model dtype: {first_param.dtype}")
        logger.info(f"  Model device: {first_param.device}")
        
        # Memory usage
        mem_info = get_gpu_memory_info()
        logger.info(f"\nGPU Memory:")
        logger.info(f"  Allocated: {mem_info['allocated_mb']:.1f} MB")
        logger.info(f"  Reserved: {mem_info['reserved_mb']:.1f} MB")
        logger.info(f"  Peak allocated: {mem_info['max_allocated_mb']:.1f} MB")
    else:
        logger.error("VAE not loaded!")


def main():
    parser = argparse.ArgumentParser(description="Profile VAE interpolation performance")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for profiling")
    parser.add_argument("--image-a", type=str, default=None, help="Path to first image")
    parser.add_argument("--image-b", type=str, default=None, help="Path to second image")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VAE INTERPOLATION PROFILER")
    logger.info("="*60)
    
    # Initialize encoder
    logger.info("\nInitializing LatentEncoder...")
    encoder = LatentEncoder(device="cuda", auto_load=True)
    logger.info("[OK] LatentEncoder initialized")
    
    # Check GPU utilization
    check_gpu_utilization(encoder)
    
    # Load test images
    if args.image_a and args.image_b:
        image_a = Image.open(args.image_a).convert('RGB')
        image_b = Image.open(args.image_b).convert('RGB')
    else:
        # Use seed images or create test images
        seed_dir = Path("seeds")
        if seed_dir.exists():
            seed_images = list(seed_dir.glob("*.png"))
            if len(seed_images) >= 2:
                image_a = Image.open(seed_images[0]).convert('RGB')
                image_b = Image.open(seed_images[1]).convert('RGB')
            else:
                logger.warning("Not enough seed images, creating test images")
                image_a = Image.new('RGB', (512, 256), color=(128, 128, 128))
                image_b = Image.new('RGB', (512, 256), color=(64, 64, 192))
        else:
            logger.warning("Seed directory not found, creating test images")
            image_a = Image.new('RGB', (512, 256), color=(128, 128, 128))
            image_b = Image.new('RGB', (512, 256), color=(64, 64, 192))
    
    logger.info(f"\nTest images: {image_a.size} {image_a.mode}")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "profile_test.png"
    
    # Profile individual components
    profile_encode(encoder, image_a, iterations=args.iterations)
    
    # Generate latents for decode test
    latent_a = encoder.encode(image_a)
    profile_decode(encoder, latent_a, iterations=args.iterations)
    
    # Profile slerp
    latent_b = encoder.encode(image_b)
    profile_slerp(latent_a, latent_b, iterations=args.iterations)
    
    # Profile image save
    test_image = encoder.decode(latent_a)
    profile_image_save(test_image, output_path, iterations=args.iterations)
    
    # Profile full pipeline
    profile_full_pipeline(encoder, image_a, image_b, output_path, iterations=args.iterations)
    
    logger.info("\n" + "="*60)
    logger.info("Profiling complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

