"""
Comprehensive Quality Comparison Test for Interpolation Optimization

Tests all combinations of:
- Resolution divisors (1, 2, 3, 4, 8)
- Upscale methods (bilinear, bicubic, nearest)
- Downsample methods (bilinear, bicubic, lanczos)

Generates visual outputs and performance metrics for comparison.
"""

import torch
import time
import sys
import json
import statistics
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add backend directory to path for imports (we're in backend/tools/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpolation.latent_encoder import LatentEncoder
from interpolation.spherical_lerp import spherical_lerp, precompute_slerp_params


def calculate_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate Structural Similarity Index (simplified version)
    Returns value between -1 and 1, where 1 is perfect similarity
    """
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Mean
    mu1 = arr1.mean()
    mu2 = arr2.mean()
    
    # Variance and covariance
    sigma1_sq = ((arr1 - mu1) ** 2).mean()
    sigma2_sq = ((arr2 - mu2) ** 2).mean()
    sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()
    
    # SSIM formula
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim)


def generate_test_configs(include_fractional: bool = False, include_multi_upscale: bool = False, 
                          exclude_div8: bool = False, focused_post_upscale: bool = False) -> List[Dict]:
    """
    Generate comprehensive test configuration matrix
    
    Args:
        include_fractional: Include fractional divisors (1.5, 2.5, etc.)
        include_multi_upscale: Include multi-pass upscaling experiments for ALL configs
        exclude_div8: Exclude divisor 8 (too degraded)
        focused_post_upscale: Add post-upscaling only for most promising configs
    
    Returns list of config dictionaries with all combinations
    """
    configs = []
    
    # Divisors to test
    divisors = [1, 2, 3, 4]
    if not exclude_div8:
        divisors.append(8)
    
    # Add fractional divisors if requested
    if include_fractional:
        divisors.extend([1.5, 2.5, 3.5])
    
    # Resampling methods
    upscale_methods = ["bilinear", "bicubic", "nearest"]
    downsample_methods = ["bilinear", "bicubic", "lanczos"]
    
    # Post-upscale multipliers for focused testing
    post_upscale_factors = [1.25, 1.5] if focused_post_upscale else []
    
    for divisor in divisors:
        if divisor == 1:
            # Baseline - no upscale/downsample needed
            configs.append({
                "name": "baseline_div1",
                "divisor": 1,
                "downsample": None,
                "upscale": None,
                "multi_upscale": None,
                "description": "Full resolution baseline (no scaling)"
            })
        else:
            # Test all combinations for scaled resolutions
            for down_method in downsample_methods:
                for up_method in upscale_methods:
                    divisor_str = str(divisor).replace('.', '_')
                    
                    # Standard config
                    configs.append({
                        "name": f"div{divisor_str}_{down_method}down_{up_method}up",
                        "divisor": divisor,
                        "downsample": down_method,
                        "upscale": up_method,
                        "multi_upscale": None,
                        "description": f"1/{divisor} res, {down_method} down, {up_method} up"
                    })
                    
                    # Focused post-upscaling: only for specific promising combinations
                    if focused_post_upscale:
                        # Only add post-upscale for most promising base methods
                        promising = (
                            (divisor in [2, 3] and down_method in ["bicubic", "lanczos"] and up_method in ["bilinear", "bicubic"]) or
                            (divisor == 4 and down_method == "bilinear" and up_method == "bilinear")
                        )
                        
                        if promising:
                            for factor in post_upscale_factors:
                                factor_str = str(factor).replace('.', '_')
                                configs.append({
                                    "name": f"div{divisor_str}_{down_method}down_{up_method}up_{factor_str}x",
                                    "divisor": divisor,
                                    "downsample": down_method,
                                    "upscale": up_method,
                                    "multi_upscale": factor,
                                    "description": f"1/{divisor} res, then post-upscale to {factor}x original"
                                })
                    
                    # Multi-pass upscaling for ALL configs if requested
                    elif include_multi_upscale and divisor >= 2:
                        # Upscale 1.5x higher than original
                        configs.append({
                            "name": f"div{divisor_str}_{down_method}down_{up_method}up_1_5x",
                            "divisor": divisor,
                            "downsample": down_method,
                            "upscale": up_method,
                            "multi_upscale": 1.5,
                            "description": f"1/{divisor} res, then upscale to 1.5x original"
                        })
    
    return configs


def run_test_config(
    config: Dict,
    image_a: Image.Image,
    image_b: Image.Image,
    output_dir: Path,
    num_frames: int = 20,
    baseline_frames: Optional[List[Image.Image]] = None
) -> Dict:
    """
    Run one test configuration and save results
    
    Args:
        config: Test configuration dictionary
        image_a: First image
        image_b: Second image
        output_dir: Output directory for this config
        num_frames: Number of frames to generate
        baseline_frames: Baseline frames for quality comparison (optional)
    
    Returns:
        Dictionary with performance metrics and quality scores
    """
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoder with config
    encoder = LatentEncoder(
        device="cuda",
        auto_load=True,
        interpolation_resolution_divisor=config['divisor'],
        upscale_method=config['upscale'] or "bilinear",
        downsample_method=config['downsample'] or "bilinear"
    )
    
    # Handle multi-upscale if specified
    if config.get('multi_upscale'):
        # Temporarily adjust target resolution for multi-pass upscaling
        original_target = encoder.target_resolution
    
    # Encode images
    print("Encoding images...")
    torch.cuda.synchronize()
    encode_start = time.perf_counter()
    
    latent_a = encoder.encode(image_a, for_interpolation=True)
    latent_b = encoder.encode(image_b, for_interpolation=True)
    
    torch.cuda.synchronize()
    encode_time = (time.perf_counter() - encode_start) / 2  # Average per encode
    
    # Pre-compute slerp parameters
    slerp_params = precompute_slerp_params(latent_a, latent_b)
    
    # Generate interpolated frames
    print(f"Generating {num_frames} frames...")
    frames = []
    times = {
        'slerp': [],
        'decode': [],
        'save': [],
        'total': []
    }
    quality_metrics = {
        'psnr': [],
        'ssim': []
    }
    
    for i in range(num_frames):
        torch.cuda.synchronize()
        frame_start = time.perf_counter()
        
        # Slerp
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        torch.cuda.synchronize()
        slerp_start = time.perf_counter()
        
        latent_interp = spherical_lerp(latent_a, latent_b, t, precomputed=slerp_params)
        
        torch.cuda.synchronize()
        slerp_time = time.perf_counter() - slerp_start
        
        # Decode
        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        
        image_interp = encoder.decode(latent_interp, upscale_to_target=True)
        
        # Multi-pass upscaling if requested
        if config.get('multi_upscale'):
            multi_factor = config['multi_upscale']
            original_size = (image_a.width, image_a.height)
            new_size = (int(original_size[0] * multi_factor), int(original_size[1] * multi_factor))
            
            # Use the specified upscale method
            resample = getattr(Image.Resampling, (config['upscale'] or 'bilinear').upper())
            image_interp = image_interp.resize(new_size, resample)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        
        # Save
        save_start = time.perf_counter()
        output_path = output_dir / f"frame_{i:03d}.png"
        image_interp.save(output_path, "PNG", optimize=False, compress_level=1)
        save_time = time.perf_counter() - save_start
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - frame_start
        
        # Record times
        times['slerp'].append(slerp_time)
        times['decode'].append(decode_time)
        times['save'].append(save_time)
        times['total'].append(total_time)
        
        frames.append(image_interp)
        
        # Calculate quality metrics vs baseline
        # Skip quality comparison for multi-upscale configs (different output resolution)
        if baseline_frames and i < len(baseline_frames) and not config.get('multi_upscale'):
            psnr = calculate_psnr(baseline_frames[i], image_interp)
            ssim = calculate_ssim(baseline_frames[i], image_interp)
            quality_metrics['psnr'].append(psnr)
            quality_metrics['ssim'].append(ssim)
        
        # Progress indicator
        if (i + 1) % 5 == 0 or i == num_frames - 1:
            print(f"  Progress: {i+1}/{num_frames} frames ({(i+1)/num_frames*100:.1f}%)")
    
    # Calculate average metrics
    avg_times = {k: sum(v) / len(v) for k, v in times.items()}
    fps = 1.0 / avg_times['total']
    
    results = {
        "config": config,
        "performance": {
            "encode_time_ms": encode_time * 1000,
            "avg_slerp_time_ms": avg_times['slerp'] * 1000,
            "avg_decode_time_ms": avg_times['decode'] * 1000,
            "avg_save_time_ms": avg_times['save'] * 1000,
            "avg_total_time_ms": avg_times['total'] * 1000,
            "fps": fps,
            "frames_generated": len(frames)
        },
        "quality": {
            "avg_psnr": sum(quality_metrics['psnr']) / len(quality_metrics['psnr']) if quality_metrics['psnr'] else None,
            "avg_ssim": sum(quality_metrics['ssim']) / len(quality_metrics['ssim']) if quality_metrics['ssim'] else None,
            "min_psnr": min(quality_metrics['psnr']) if quality_metrics['psnr'] else None,
            "min_ssim": min(quality_metrics['ssim']) if quality_metrics['ssim'] else None
        },
        "frames": [str(output_dir / f"frame_{i:03d}.png") for i in range(num_frames)]
    }
    
    # Print summary
    print(f"\nResults for {config['name']}:")
    print(f"  Average time per frame: {avg_times['total']*1000:.1f}ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Breakdown:")
    print(f"    Encode:  {encode_time*1000:.1f}ms")
    print(f"    Slerp:   {avg_times['slerp']*1000:.1f}ms")
    print(f"    Decode:  {avg_times['decode']*1000:.1f}ms")
    print(f"    Save:    {avg_times['save']*1000:.1f}ms")
    
    if quality_metrics['psnr']:
        print(f"  Quality vs baseline:")
        print(f"    PSNR: {results['quality']['avg_psnr']:.2f} dB (min: {results['quality']['min_psnr']:.2f})")
        print(f"    SSIM: {results['quality']['avg_ssim']:.4f} (min: {results['quality']['min_ssim']:.4f})")
    elif config.get('multi_upscale'):
        print(f"  Quality metrics: N/A (multi-upscale produces different resolution than baseline)")
    
    return results


def main():
    """Run comprehensive quality comparison tests"""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive interpolation quality testing')
    parser.add_argument('--frames', type=int, default=20, help='Number of frames to generate per config')
    parser.add_argument('--use-output-frames', action='store_true', help='Use real generated frames from output/ instead of seeds')
    parser.add_argument('--fractional', action='store_true', help='Include fractional divisors (1.5, 2.5, etc.)')
    parser.add_argument('--multi-upscale', action='store_true', help='Test multi-pass upscaling (1.5x) for ALL configs')
    parser.add_argument('--exclude-div8', action='store_true', help='Exclude divisor 8 (focus on div1-4)')
    parser.add_argument('--focused-post-upscale', action='store_true', help='Add post-upscaling for promising configs only')
    parser.add_argument('--configs-only', nargs='+', help='Test only specific configs (e.g., div2_bicubicdown_nearestup)')
    parser.add_argument('--image-a', type=str, help='Path to first image (default: first image in seeds/)')
    parser.add_argument('--image-b', type=str, help='Path to second image (default: second image in seeds/)')
    parser.add_argument('--sequences', type=int, default=1, help='Number of times to repeat the test (for long-run profiling)')
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE INTERPOLATION QUALITY COMPARISON TEST")
    print("="*70)
    print(f"\nFrames per config: {args.frames}")
    if args.use_output_frames:
        print("Using real generated frames from output/")
    if args.fractional:
        print("Including fractional divisors")
    if args.multi_upscale:
        print("Including multi-pass upscaling for ALL configs")
    if args.exclude_div8:
        print("Excluding divisor 8 (focusing on div1-4)")
    if args.focused_post_upscale:
        print("Adding post-upscaling for promising configs")
    
    # Generate test configurations
    configs = generate_test_configs(
        include_fractional=args.fractional,
        include_multi_upscale=args.multi_upscale,
        exclude_div8=args.exclude_div8,
        focused_post_upscale=args.focused_post_upscale
    )
    
    # Filter configs if specific ones requested
    if args.configs_only:
        configs = [c for c in configs if c['name'] in args.configs_only]
        print(f"\nTesting only: {', '.join(args.configs_only)}")
    
    print(f"\nThis will generate {len(configs)} test configurations")
    print("="*70)
    
    # Setup - output to root level
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(f"../output/quality_tests/run_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}")
    
    # Load test images
    seed_dir = Path("../seeds") if Path("../seeds").exists() else Path("seeds")
    
    # Check for explicit image paths
    if args.image_a and args.image_b:
        image_a_path = Path(args.image_a)
        image_b_path = Path(args.image_b)
        
        if not image_a_path.exists():
            print(f"[ERROR] Image A not found: {image_a_path}")
            return
        if not image_b_path.exists():
            print(f"[ERROR] Image B not found: {image_b_path}")
            return
        
        print(f"\nUsing specified images:")
        print(f"  Image A: {image_a_path}")
        print(f"  Image B: {image_b_path}")
        
        image_a = Image.open(image_a_path).convert('RGB')
        image_b = Image.open(image_b_path).convert('RGB')
        
        # Store paths for later reference
        test_image_paths = [str(image_a_path), str(image_b_path)]
        
    elif args.use_output_frames:
        # Use real generated frames from output folder
        output_dir = Path("../output") if Path("../output").exists() else Path("output")
        potential_frames = list(output_dir.glob("frame_*.png"))
        
        if len(potential_frames) < 2:
            print("[WARNING] Not enough frames in output/, falling back to seeds")
            seed_images = list(seed_dir.glob("*.png"))
        else:
            # Sort by frame number and pick diverse frames
            potential_frames.sort()
            # Pick frames from different parts of the sequence
            seed_images = [
                potential_frames[len(potential_frames) // 4],  # 25% through
                potential_frames[len(potential_frames) * 3 // 4]  # 75% through
            ]
            print(f"\nUsing real generated frames:")
            print(f"  Frame A: {seed_images[0].name}")
            print(f"  Frame B: {seed_images[1].name}")
        
        # Store paths for later reference
        test_image_paths = [str(seed_images[0]), str(seed_images[1])]
        
    else:
        seed_images = list(seed_dir.glob("*.png"))
        
        if len(seed_images) < 2:
            print("[ERROR] Need at least 2 images in seeds/")
            return
        
        print(f"\nLoading test images from seeds/...")
        image_a = Image.open(seed_images[0]).convert('RGB')
        image_b = Image.open(seed_images[1]).convert('RGB')
        
        # Store paths for later reference
        test_image_paths = [str(seed_images[0]), str(seed_images[1])]
    
    # Always resize to standard resolution for consistent testing
    target_size = (512, 256)
    if image_a.size != target_size:
        print(f"  Resizing Image A from {image_a.size} to {target_size}")
        image_a = image_a.resize(target_size, Image.Resampling.LANCZOS)
    if image_b.size != target_size:
        print(f"  Resizing Image B from {image_b.size} to {target_size}")
        image_b = image_b.resize(target_size, Image.Resampling.LANCZOS)
    
    print(f"  Final Image A: {image_a.size}")
    print(f"  Final Image B: {image_b.size}")
    
    # Configs were already generated above - use them!
    print(f"\nUsing {len(configs)} test configurations")
    
    # Check if we're doing multiple sequences
    if args.sequences > 1:
        print(f"\n{'='*70}")
        print(f"LONG-RUN MODE: {args.sequences} sequences")
        print(f"{'='*70}")
        print(f"Total frames: {args.sequences * args.frames * len(configs)}")
        print()
    
    # Run baseline first to get reference frames
    print("\n" + "="*70)
    print("RUNNING BASELINE TEST (for quality comparison)")
    print("="*70)
    
    baseline_config = configs[0]  # First config is always baseline
    
    # Run baseline once for reference (don't repeat in sequences)
    baseline_results = run_test_config(
        baseline_config,
        image_a,
        image_b,
        output_root / baseline_config['name'],
        num_frames=args.frames
    )
    
    # Load baseline frames for quality comparison
    baseline_frames = []
    for i in range(args.frames):
        frame_path = output_root / baseline_config['name'] / f"frame_{i:03d}.png"
        baseline_frames.append(Image.open(frame_path))
    
    # Store all results - baseline stored separately
    all_results = [baseline_results]
    
    # Run remaining tests (with sequences if specified)
    print("\n" + "="*70)
    print("RUNNING SCALED RESOLUTION TESTS")
    if args.sequences > 1:
        print(f"({args.sequences} sequences per config)")
    print("="*70)
    
    for i, config in enumerate(configs[1:], 1):
        print(f"\nTest {i}/{len(configs)-1}: {config['name']}")
        
        # Aggregate results across sequences
        sequence_fps_values = []
        
        for seq_num in range(args.sequences):
            if args.sequences > 1:
                print(f"  Sequence {seq_num + 1}/{args.sequences}...")
            
            # Determine output directory for this sequence
            if args.sequences > 1:
                output_dir = output_root / f"{config['name']}_seq{seq_num:03d}"
            else:
                output_dir = output_root / config['name']
            
            try:
                results = run_test_config(
                    config,
                    image_a,
                    image_b,
                    output_dir,
                    num_frames=args.frames,
                    baseline_frames=baseline_frames
                )
                
                # Track FPS for this sequence
                sequence_fps_values.append(results['performance']['fps'])
                
                # Only store first sequence results in all_results (for compatibility)
                if seq_num == 0:
                    all_results.append(results)
                
            except Exception as e:
                print(f"  [ERROR] Failed sequence {seq_num + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print sequence summary if multiple sequences
        try:
            if args.sequences > 1 and sequence_fps_values:
                avg_fps = statistics.mean(sequence_fps_values)
                min_fps = min(sequence_fps_values)
                max_fps = max(sequence_fps_values)
                stdev_fps = statistics.stdev(sequence_fps_values) if len(sequence_fps_values) > 1 else 0
                
                print(f"\n  Sequence Summary for {config['name']}:")
                print(f"    Average FPS: {avg_fps:.2f} (±{stdev_fps:.2f})")
                print(f"    Range:       {min_fps:.2f} - {max_fps:.2f} FPS")
                print(f"    Stability:   {(stdev_fps/avg_fps*100):.1f}% CV")
        except Exception as e:
            print(f"  [ERROR] Failed to print sequence summary: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    # Save comprehensive metrics
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    metrics_file = output_root / "performance_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "test_info": {
                "timestamp": timestamp,
                "total_configs": len(all_results),
                "test_images": test_image_paths,
                "target_fps": 15.0
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"Performance metrics saved: {metrics_file}")
    
    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY - TOP PERFORMERS")
    print("="*70)
    
    # Sort by FPS (descending)
    sorted_by_fps = sorted(all_results, key=lambda x: x['performance']['fps'], reverse=True)
    
    print("\nTop 10 by FPS:")
    print(f"{'Rank':<6} {'Config':<40} {'FPS':<8} {'PSNR':<8} {'SSIM':<8}")
    print("-" * 70)
    for i, result in enumerate(sorted_by_fps[:10], 1):
        config_name = result['config']['name']
        fps = result['performance']['fps']
        psnr = result['quality']['avg_psnr'] or 0
        ssim = result['quality']['avg_ssim'] or 0
        print(f"{i:<6} {config_name:<40} {fps:>6.2f}  {psnr:>6.1f}  {ssim:>6.4f}")
    
    # Find configs meeting 15 FPS target
    meeting_target = [r for r in all_results if r['performance']['fps'] >= 15.0]
    if meeting_target:
        print(f"\n{len(meeting_target)} configurations meet 15 FPS target:")
        # Sort by quality (SSIM)
        meeting_target.sort(key=lambda x: x['quality']['avg_ssim'] or 0, reverse=True)
        print(f"{'Config':<40} {'FPS':<8} {'PSNR':<8} {'SSIM':<8}")
        print("-" * 70)
        for result in meeting_target[:5]:  # Top 5
            config_name = result['config']['name']
            fps = result['performance']['fps']
            psnr = result['quality']['avg_psnr'] or 0
            ssim = result['quality']['avg_ssim'] or 0
            print(f"{config_name:<40} {fps:>6.2f}  {psnr:>6.1f}  {ssim:>6.4f}")
    else:
        print("\nNo configurations meet 15 FPS target")
        print("Closest:")
        closest = sorted_by_fps[1]  # Skip baseline
        print(f"  {closest['config']['name']}: {closest['performance']['fps']:.2f} FPS")
    
    print("\n" + "="*70)
    print(f"All results saved to: {output_root}")
    print("="*70)
    
    # Generate HTML comparison
    print("\nGenerating HTML comparison...")
    try:
        from tools.generate_comparison import generate_comparison_html
        html_path = generate_comparison_html(output_root)
        print(f"[OK] HTML comparison: {html_path}")
        print(f"     Open in browser: file:///{html_path.absolute()}")
    except Exception as e:
        print(f"[ERROR] Failed to generate HTML: {e}")
        print("You can manually generate it with:")
        print(f"  uv run backend/tools/generate_comparison.py {output_root}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    
    return output_root


if __name__ == "__main__":
    output_dir = main()

