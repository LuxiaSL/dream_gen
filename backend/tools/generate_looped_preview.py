#!/usr/bin/env python3
"""
Generate Looped Preview GIF for GitHub

Creates a compact, looping GIF animation between two selected keyframes
for use as a GitHub preview/thumbnail.

Pattern: keyframe_a -> interpolation_frames -> keyframe_b -> interpolation_frames -> keyframe_a

Features:
- Selectable keyframes (default: 024 and 010)
- Optimized for file size (target: < 1MB)
- Configurable frame count for interpolation
- Multiple compression strategies
- Real-time file size monitoring
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpolation.spherical_lerp import precompute_slerp_params, spherical_lerp
from interpolation.latent_encoder import LatentEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_keyframe(keyframes_dir: Path, keyframe_num: int) -> Optional[Path]:
    """
    Find keyframe by number.
    
    Args:
        keyframes_dir: Directory containing keyframes
        keyframe_num: Keyframe number (e.g., 24 for keyframe_024.png)
    
    Returns:
        Path to keyframe if found, None otherwise
    """
    formatted_num = str(keyframe_num).zfill(3)
    keyframe_path = keyframes_dir / f"keyframe_{formatted_num}.png"
    
    if keyframe_path.exists():
        return keyframe_path
    
    return None


def load_image_for_vae(image_path: Path, target_size: Tuple[int, int]) -> Image.Image:
    """
    Load and prepare image for VAE encoding.
    
    Args:
        image_path: Path to image
        target_size: Target resolution (width, height)
    
    Returns:
        PIL Image ready for VAE
    """
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image


def generate_looped_preview(
    keyframe_a_num: int,
    keyframe_b_num: int,
    output_dir: Path,
    input_dir: Path = None,
    interpolation_frames: int = 10,
    fps: int = 5,
    output_name: str = "preview.gif",
    quality: int = 85,
    target_size: Optional[Tuple[int, int]] = None,
    max_size_mb: float = 1.0,
) -> Path:
    """
    Generate a looped preview GIF between two keyframes.
    
    Args:
        keyframe_a_num: First keyframe number (e.g., 24)
        keyframe_b_num: Second keyframe number (e.g., 10)
        output_dir: Directory to save output GIF
        input_dir: Input directory containing keyframes (default: output)
        interpolation_frames: Number of frames between keyframes (default: 10)
        fps: Frames per second (default: 5)
        output_name: Name of output file (default: preview.gif)
        quality: Pillow GIF quality optimization level (default: 85)
        target_size: Override target resolution (width, height)
        max_size_mb: Target maximum file size in MB (default: 1.0)
    
    Returns:
        Path to generated GIF
    """
    
    if input_dir is None:
        input_dir = Path("output")
    
    keyframes_dir = input_dir / "keyframes"
    
    # Verify paths exist
    if not keyframes_dir.exists():
        raise FileNotFoundError(f"Keyframes directory not found: {keyframes_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find keyframes
    logger.info(f"🔍 Locating keyframes...")
    kf_a_path = find_keyframe(keyframes_dir, keyframe_a_num)
    kf_b_path = find_keyframe(keyframes_dir, keyframe_b_num)
    
    if kf_a_path is None:
        raise FileNotFoundError(f"Keyframe {keyframe_a_num:03d} not found in {keyframes_dir}")
    if kf_b_path is None:
        raise FileNotFoundError(f"Keyframe {keyframe_b_num:03d} not found in {keyframes_dir}")
    
    logger.info(f"  ✓ Found keyframe_{keyframe_a_num:03d}.png")
    logger.info(f"  ✓ Found keyframe_{keyframe_b_num:03d}.png")
    
    # Determine target size
    if target_size is None:
        img_a = Image.open(kf_a_path)
        target_size = img_a.size
        img_a.close()
    
    logger.info(f"  Target resolution: {target_size[0]}x{target_size[1]}")
    
    # Initialize VAE encoder
    logger.info(f"📦 Initializing VAE encoder...")
    try:
        encoder = LatentEncoder(
            vae_path=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            auto_load=True,
            target_resolution=target_size
        )
        logger.info(f"  ✓ VAE ready on {encoder.device}")
    except Exception as e:
        logger.error(f"Failed to initialize VAE: {e}")
        raise
    
    # Load and encode keyframes
    logger.info(f"🔐 Encoding keyframes...")
    try:
        img_a = load_image_for_vae(kf_a_path, target_size)
        latent_a = encoder.encode(img_a, for_interpolation=False)
        logger.info(f"  ✓ Encoded keyframe A ({keyframe_a_num:03d})")
        
        img_b = load_image_for_vae(kf_b_path, target_size)
        latent_b = encoder.encode(img_b, for_interpolation=False)
        logger.info(f"  ✓ Encoded keyframe B ({keyframe_b_num:03d})")
    except Exception as e:
        logger.error(f"Failed to encode keyframes: {e}")
        raise
    
    # Precompute slerp parameters
    logger.info(f"⚙️  Precomputing interpolation parameters...")
    try:
        slerp_params = precompute_slerp_params(latent_a, latent_b)
        logger.info(f"  ✓ Slerp parameters precomputed")
    except Exception as e:
        logger.error(f"Failed to precompute slerp: {e}")
        raise
    
    # Generate frames for the looped sequence
    # Pattern: A -> [interpolations] -> B -> [interpolations reversed] -> A
    logger.info(f"🎬 Generating interpolated frames...")
    frames = []
    frame_count = 0
    
    try:
        # Add keyframe A
        frames.append(img_a)
        frame_count += 1
        logger.info(f"  Frame {frame_count}: Keyframe A")
        
        # Generate interpolations A -> B
        logger.info(f"  Generating {interpolation_frames} frames (A → B)...")
        interp_frames_forward = []
        
        for i in range(1, interpolation_frames + 1):
            t = i / (interpolation_frames + 1)  # Don't include endpoints
            
            try:
                latent_interp = spherical_lerp(latent_a, latent_b, t, precomputed=slerp_params)
                img_interp = encoder.decode(latent_interp, upscale_to_target=False)
                interp_frames_forward.append(img_interp)
                frame_count += 1
                
                if i % 3 == 0 or i == interpolation_frames:
                    logger.info(f"    ✓ Frame {i}/{interpolation_frames}")
            except Exception as e:
                logger.error(f"Failed to generate interpolation frame {i}: {e}")
                raise
        
        frames.extend(interp_frames_forward)
        
        # Add keyframe B
        frames.append(img_b)
        frame_count += 1
        logger.info(f"  Frame {frame_count}: Keyframe B")
        
        # Generate interpolations B -> A (reversed)
        logger.info(f"  Generating {interpolation_frames} frames (B → A)...")
        
        for i in range(1, interpolation_frames + 1):
            t = 1.0 - (i / (interpolation_frames + 1))  # Reverse interpolation
            
            try:
                latent_interp = spherical_lerp(latent_a, latent_b, t, precomputed=slerp_params)
                img_interp = encoder.decode(latent_interp, upscale_to_target=False)
                frames.append(img_interp)
                frame_count += 1
                
                if i % 3 == 0 or i == interpolation_frames:
                    logger.info(f"    ✓ Frame {frame_count - interpolation_frames}/{interpolation_frames}")
            except Exception as e:
                logger.error(f"Failed to generate reverse interpolation frame {i}: {e}")
                raise
        
        total_frames = len(frames)
        logger.info(f"✓ Generated {total_frames} total frames")
        
    except Exception as e:
        logger.error(f"Frame generation failed: {e}")
        raise
    
    # Save as GIF with optimization
    logger.info(f"💾 Encoding GIF (target: < {max_size_mb:.1f}MB)...")
    output_path = output_dir / output_name
    
    duration_ms = int(1000 / fps)
    
    try:
        # First attempt: Standard quality
        logger.info(f"  Attempt 1: Standard encoding (quality={quality})")
        frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,  # Infinite loop
            optimize=True,
        )
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ File size: {file_size_mb:.2f}MB")
        
        # If file is too large, re-encode with higher compression
        if file_size_mb > max_size_mb:
            logger.info(f"  ⚠ File size exceeds limit ({file_size_mb:.2f}MB > {max_size_mb:.1f}MB)")
            logger.info(f"  Attempt 2: Aggressive downsampling...")
            
            # Downsample frames for smaller file size
            downsampled_frames = []
            downsample_factor = 2
            
            new_width = target_size[0] // downsample_factor
            new_height = target_size[1] // downsample_factor
            
            for i, frame in enumerate(frames):
                downsampled = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                downsampled_frames.append(downsampled)
                
                if (i + 1) % 5 == 0 or i == 0:
                    logger.info(f"    Downsampled {i + 1}/{len(frames)} frames")
            
            # Re-save with downsampled frames
            downsampled_frames[0].save(
                output_path,
                format="GIF",
                save_all=True,
                append_images=downsampled_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Downsampled file size: {file_size_mb:.2f}MB")
            
            # If still too large, reduce frame count
            if file_size_mb > max_size_mb:
                logger.info(f"  ⚠ Still exceeds limit. Reducing frame count...")
                logger.info(f"  Attempt 3: Frame decimation...")
                
                # Keep every Nth frame
                keep_every = 2
                decimated_frames = downsampled_frames[::keep_every]
                
                logger.info(f"    Keeping every {keep_every}th frame ({len(decimated_frames)} frames)")
                
                decimated_frames[0].save(
                    output_path,
                    format="GIF",
                    save_all=True,
                    append_images=decimated_frames[1:],
                    duration=duration_ms,
                    loop=0,
                    optimize=True,
                )
                
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ Decimated file size: {file_size_mb:.2f}MB")
        
        # Final statistics
        print(f"\n{'='*60}")
        print(f"✅ Preview GIF generated successfully!")
        print(f"{'='*60}")
        print(f"Output: {output_path.name}")
        print(f"Location: {output_path}")
        print(f"File size: {file_size_mb:.2f}MB")
        print(f"Target: < {max_size_mb:.1f}MB")
        
        if file_size_mb <= max_size_mb:
            print(f"✓ Within size limit!")
        else:
            print(f"⚠ Exceeds size limit (consider reducing frames or resolution)")
        
        print(f"\nAnimation details:")
        print(f"  Keyframe A: keyframe_{keyframe_a_num:03d}.png")
        print(f"  Keyframe B: keyframe_{keyframe_b_num:03d}.png")
        print(f"  Interpolation frames: {interpolation_frames}")
        print(f"  Total frames: {len(downsampled_frames) if file_size_mb > max_size_mb and 'downsampled_frames' in locals() else len(frames)}")
        print(f"  FPS: {fps}")
        print(f"  Duration: ~{(len(frames) / fps):.1f}s")
        print(f"  Resolution: {target_size[0]}x{target_size[1]}")
        print(f"{'='*60}\n")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save GIF: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate looped preview GIF between two keyframes for GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: keyframes 024 and 010, 10 interpolation frames
  uv run backend/tools/generate_looped_preview.py

  # Custom keyframes
  uv run backend/tools/generate_looped_preview.py --keyframe-a 30 --keyframe-b 15

  # More interpolation frames for smoother animation
  uv run backend/tools/generate_looped_preview.py --interpolation-frames 20

  # Faster animation
  uv run backend/tools/generate_looped_preview.py --fps 8

  # Custom output path
  uv run backend/tools/generate_looped_preview.py --output my_preview.gif

  # More aggressive size optimization
  uv run backend/tools/generate_looped_preview.py --max-size 0.8

  # Specific resolution
  uv run backend/tools/generate_looped_preview.py --resolution 256x128
        """
    )
    
    parser.add_argument(
        "--keyframe-a",
        type=int,
        default=24,
        help="First keyframe number (default: 24)"
    )
    
    parser.add_argument(
        "--keyframe-b",
        type=int,
        default=10,
        help="Second keyframe number (default: 10)"
    )
    
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        default=10,
        help="Number of interpolation frames between keyframes (default: 10)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        choices=range(1, 31),
        metavar="FPS",
        help="Frames per second (1-30, default: 5)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="preview.gif",
        help="Output filename (default: preview.gif)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output"),
        help="Input directory containing keyframes (default: output)"
    )
    
    parser.add_argument(
        "--max-size",
        type=float,
        default=1.0,
        help="Target maximum file size in MB (default: 1.0)"
    )
    
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Override target resolution (e.g., 512x256)"
    )
    
    args = parser.parse_args()
    
    # Parse resolution if provided
    target_size = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.lower().split('x'))
            target_size = (width, height)
            logger.info(f"Using custom resolution: {width}x{height}")
        except ValueError:
            logger.error(f"Invalid resolution format: {args.resolution}")
            return 1
    
    try:
        output_path = generate_looped_preview(
            keyframe_a_num=args.keyframe_a,
            keyframe_b_num=args.keyframe_b,
            output_dir=args.output_dir,
            input_dir=args.input_dir,
            interpolation_frames=args.interpolation_frames,
            fps=args.fps,
            output_name=args.output,
            target_size=target_size,
            max_size_mb=args.max_size,
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

