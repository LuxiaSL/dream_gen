"""
Quick test script to measure interpolation speed improvements

This script tests a single interpolation cycle to verify optimizations.
"""

import torch
import time
import sys
from pathlib import Path
from PIL import Image

# Add backend directory to path for imports (we're in backend/tools/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpolation.latent_encoder import LatentEncoder
from interpolation.spherical_lerp import spherical_lerp, precompute_slerp_params

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution-divisor", type=int, default=1, 
                       help="Interpolation resolution divisor (1=full, 2=half, 4=quarter)")
    parser.add_argument("--upscale-method", type=str, default="bilinear",
                       choices=["bilinear", "bicubic", "nearest"],
                       help="Upscaling method")
    args = parser.parse_args()
    
    print("="*70)
    print("INTERPOLATION SPEED TEST")
    if args.resolution_divisor > 1:
        print(f"Lower-res mode: {args.resolution_divisor}x downscale, {args.upscale_method} upscale")
    else:
        print("Full resolution mode")
    print("="*70)
    
    # Initialize encoder
    print("\nInitializing LatentEncoder...")
    encoder = LatentEncoder(
        device="cuda",
        auto_load=True,
        interpolation_resolution_divisor=args.resolution_divisor,
        upscale_method=args.upscale_method
    )
    print("[OK] VAE loaded")
    
    # Load test images
    seed_dir = Path("seeds")
    seed_images = list(seed_dir.glob("*.png"))
    if len(seed_images) < 2:
        print("[ERROR] Need at least 2 seed images")
        return
    
    # Load and resize to target resolution (512x256)
    image_a = Image.open(seed_images[0]).convert('RGB')
    image_b = Image.open(seed_images[1]).convert('RGB')
    
    # Resize to standard resolution
    target_size = (512, 256)
    if image_a.size != target_size:
        image_a = image_a.resize(target_size, Image.Resampling.LANCZOS)
    if image_b.size != target_size:
        image_b = image_b.resize(target_size, Image.Resampling.LANCZOS)
    
    print(f"Test images: {image_a.size}")
    
    # Warm-up GPU
    print("\nWarming up GPU...")
    for _ in range(3):
        latent = encoder.encode(image_a, for_interpolation=True)
        _ = encoder.decode(latent, upscale_to_target=True)
    
    # Test 1: Full pipeline timing
    print("\n" + "="*70)
    print("TEST 1: Full Interpolation Pipeline (7 frames)")
    print("="*70)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Encode both images (at interpolation resolution)
    latent_a = encoder.encode(image_a, for_interpolation=True)
    latent_b = encoder.encode(image_b, for_interpolation=True)
    
    # Pre-compute slerp parameters (OPTIMIZATION!)
    slerp_params = precompute_slerp_params(latent_a, latent_b)
    
    # Generate 7 interpolated frames
    frames = []
    for i in range(7):
        t = i / 7.0
        latent_interp = spherical_lerp(latent_a, latent_b, t, precomputed=slerp_params)
        image_interp = encoder.decode(latent_interp, upscale_to_target=True)
        frames.append(image_interp)
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"Time per frame: {total_time/7:.3f}s ({1/(total_time/7):.1f} FPS)")
    print(f"Encode time (amortized): {(total_time - (total_time/7)*7)/2:.3f}s per encode")
    
    # Test 2: Pure interpolation speed
    print("\n" + "="*70)
    print("TEST 2: Pure Interpolation Speed (decode only)")
    print("="*70)
    
    # Already have latents from previous test
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(7):
        t = i / 7.0
        latent_interp = spherical_lerp(latent_a, latent_b, t, precomputed=slerp_params)
        image_interp = encoder.decode(latent_interp, upscale_to_target=True)
    
    torch.cuda.synchronize()
    interp_time = time.perf_counter() - start
    
    print(f"\nTotal time: {interp_time:.3f}s")
    print(f"Time per frame: {interp_time/7:.3f}s ({1/(interp_time/7):.1f} FPS)")
    
    # Test 3: Component breakdown
    print("\n" + "="*70)
    print("TEST 3: Component Breakdown")
    print("="*70)
    
    # Encode
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = encoder.encode(image_a, for_interpolation=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    encode_time = sum(times) / len(times)
    print(f"Encode (avg): {encode_time*1000:.1f}ms")
    
    # Slerp
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = spherical_lerp(latent_a, latent_b, 0.5, precomputed=slerp_params)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    slerp_time = sum(times) / len(times)
    print(f"Slerp (avg):  {slerp_time*1000:.1f}ms")
    
    # Decode
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = encoder.decode(latent_a, upscale_to_target=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    decode_time = sum(times) / len(times)
    print(f"Decode (avg): {decode_time*1000:.1f}ms")
    
    # Save
    test_image = frames[0]
    times = []
    for _ in range(10):
        start = time.perf_counter()
        test_image.save("output/speed_test.png", "PNG", optimize=False, compress_level=1)
        times.append(time.perf_counter() - start)
    save_time = sum(times) / len(times)
    print(f"Save (avg):   {save_time*1000:.1f}ms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_per_frame = slerp_time + decode_time + save_time
    print(f"\nPer-frame breakdown:")
    print(f"  Slerp:  {slerp_time*1000:6.1f}ms ({slerp_time/total_per_frame*100:5.1f}%)")
    print(f"  Decode: {decode_time*1000:6.1f}ms ({decode_time/total_per_frame*100:5.1f}%)")
    print(f"  Save:   {save_time*1000:6.1f}ms ({save_time/total_per_frame*100:5.1f}%)")
    print(f"  TOTAL:  {total_per_frame*1000:6.1f}ms")
    print(f"\nPredicted FPS: {1/total_per_frame:.1f} FPS")
    print(f"Target FPS: 10-15 FPS")
    
    if 1/total_per_frame >= 10:
        print("\n✓ TARGET ACHIEVED!")
    elif 1/total_per_frame >= 7:
        print("\n⚠ Close to target, may need more optimization")
    else:
        print("\n✗ Below target, needs more work")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

