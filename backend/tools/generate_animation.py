#!/usr/bin/env python3
"""
Generate Animation from Keyframes and Interpolations

Creates a looping animation (WebP or GIF) from the keyframes and interpolation
frames in the output directory. Frames are sequenced as:
keyframe_001 -> interpolations_001-002 -> keyframe_002 -> interpolations_002-003 -> keyframe_003 -> ...

The animation loops back to the start, creating a continuous cycle.
"""

import argparse
import re
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from config.yaml to get target resolution.
    
    Args:
        config_path: Path to config.yaml (default: backend/config.yaml)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to backend/config.yaml relative to project root
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_target_resolution(config: dict) -> Tuple[int, int]:
    """
    Extract target resolution from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (width, height)
    """
    resolution = config.get("generation", {}).get("resolution", [512, 256])
    return tuple(resolution)


def parse_keyframe_name(filename: str) -> int:
    """
    Extract keyframe number from filename like 'keyframe_001.png'.
    
    Returns:
        Keyframe number (e.g., 1 for 'keyframe_001.png')
    """
    match = re.match(r'keyframe_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return -1


def parse_interpolation_name(filename: str) -> Tuple[int, int, int]:
    """
    Extract keyframe range and frame number from filename like '001-002_005.png'.
    
    Returns:
        Tuple of (start_keyframe, end_keyframe, frame_number)
        e.g., (1, 2, 5) for '001-002_005.png'
    """
    match = re.match(r'(\d+)-(\d+)_(\d+)\.png', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return (-1, -1, -1)


def collect_frames(output_dir: Path) -> List[Path]:
    """
    Collect and sort all keyframes and interpolations in proper sequence.
    
    The sequence follows: keyframe_N -> all interpolations N-(N+1) -> keyframe_(N+1)
    
    Returns:
        List of frame paths in proper sequential order
    """
    keyframes_dir = output_dir / "keyframes"
    interp_dir = output_dir / "interpolations"
    
    if not keyframes_dir.exists():
        raise FileNotFoundError(f"Keyframes directory not found: {keyframes_dir}")
    if not interp_dir.exists():
        raise FileNotFoundError(f"Interpolations directory not found: {interp_dir}")
    
    # Collect all keyframes
    keyframes = []
    for file in keyframes_dir.glob("keyframe_*.png"):
        num = parse_keyframe_name(file.name)
        if num > 0:
            keyframes.append((num, file))
    
    keyframes.sort(key=lambda x: x[0])
    
    if not keyframes:
        raise ValueError("No keyframes found in keyframes directory")
    
    # Collect all interpolations grouped by keyframe range
    interpolations = {}
    for file in interp_dir.glob("*-*_*.png"):
        start, end, frame_num = parse_interpolation_name(file.name)
        if start > 0 and end > 0 and frame_num > 0:
            key = (start, end)
            if key not in interpolations:
                interpolations[key] = []
            interpolations[key].append((frame_num, file))
    
    # Sort interpolations within each group
    for key in interpolations:
        interpolations[key].sort(key=lambda x: x[0])
    
    # Build the final sequence
    sequence = []
    
    for i, (keyframe_num, keyframe_path) in enumerate(keyframes):
        # Add the keyframe
        sequence.append(keyframe_path)
        
        # Add interpolations between this keyframe and the next
        if i < len(keyframes) - 1:
            next_keyframe_num = keyframes[i + 1][0]
            interp_key = (keyframe_num, next_keyframe_num)
            
            if interp_key in interpolations:
                for _, interp_path in interpolations[interp_key]:
                    sequence.append(interp_path)
    
    return sequence


def load_and_process_frame(
    frame_path: Path,
    target_resolution: Tuple[int, int],
    format: str,
    index: int,
) -> Tuple[int, Image.Image, Tuple[int, int]]:
    """
    Load and process a single frame (for parallel processing).
    
    Returns:
        Tuple of (index, processed_image, original_size)
    """
    img = Image.open(frame_path)
    original_size = img.size
    
    # Resize if needed
    if img.size != target_resolution:
        # Use Lanczos for significant downscaling, bilinear for minor adjustments
        size_ratio = (img.size[0] * img.size[1]) / (target_resolution[0] * target_resolution[1])
        if size_ratio > 2.0:
            # Significant downscaling - use Lanczos
            img = img.resize(target_resolution, Image.Resampling.LANCZOS)
        else:
            # Minor adjustment - use faster bilinear
            img = img.resize(target_resolution, Image.Resampling.BILINEAR)
    
    # Convert to RGB if needed
    if format.lower() == "gif" and img.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    
    return (index, img, original_size)


def generate_animation(
    frames: List[Path],
    output_path: Path,
    fps: int = 5,
    format: str = "webp",
    loop: bool = True,
    target_resolution: Optional[Tuple[int, int]] = None,
    fast_mode: bool = False,
) -> None:
    """
    Generate an animated image from a sequence of frames.
    
    Args:
        frames: List of frame paths in sequence order
        output_path: Output path for the animation
        fps: Frames per second (default: 5)
        format: Output format, 'webp' or 'gif' (default: 'webp')
        loop: Whether the animation should loop (default: True)
        target_resolution: Optional (width, height) to resize all frames to.
                          If None, uses the first frame's size.
        fast_mode: Use faster encoding at the cost of some quality/compression
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # First, quickly scan for target resolution if not provided
    if target_resolution is None:
        img = Image.open(frames[0])
        target_resolution = img.size
        img.close()
        print(f"Using first frame size as reference: {target_resolution[0]}x{target_resolution[1]}")
    else:
        print(f"Target resolution from config: {target_resolution[0]}x{target_resolution[1]}")
    
    # Load and process all frames in parallel
    print(f"Loading and processing {len(frames)} frames...")
    frame_sizes = {}
    processed_frames = [None] * len(frames)
    
    # Use ThreadPoolExecutor for I/O-bound operations (image loading)
    max_workers = min(8, len(frames))  # Don't spawn too many threads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                load_and_process_frame,
                frame_path,
                target_resolution,
                format,
                i
            ): i for i, frame_path in enumerate(frames)
        }
        
        # Collect results with progress indication
        completed = 0
        for future in as_completed(futures):
            index, img, original_size = future.result()
            processed_frames[index] = img
            
            # Track unique sizes for reporting
            if original_size not in frame_sizes:
                frame_sizes[original_size] = []
            frame_sizes[original_size].append(frames[index].name)
            
            completed += 1
            if completed % 20 == 0 or completed == len(frames):
                print(f"  Processed {completed}/{len(frames)} frames...")
    
    # Report any size mismatches
    if len(frame_sizes) > 1:
        print(f"⚠ Warning: Found {len(frame_sizes)} different frame sizes. Resized all to {target_resolution[0]}x{target_resolution[1]}:")
        for size, frame_list in frame_sizes.items():
            print(f"  - {size[0]}x{size[1]}: {len(frame_list)} frames (e.g., {frame_list[0]})")
    
    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)
    
    # Save animation
    print(f"Encoding {format.upper()} animation at {fps} FPS ({duration_ms}ms per frame)...")
    
    if format.lower() == "webp":
        # WebP supports better compression and quality
        # method: 0=fastest, 6=slowest/best (default: 4)
        webp_method = 3 if fast_mode else 4  # 4 is good balance, 3 for fast mode
        processed_frames[0].save(
            output_path,
            format="WebP",
            save_all=True,
            append_images=processed_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,  # 0 = infinite loop
            quality=85,
            method=webp_method,
        )
    elif format.lower() == "gif":
        # GIF fallback option
        processed_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=processed_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,  # 0 = infinite loop
            optimize=True,
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'webp' or 'gif'")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Animation saved to: {output_path}")
    print(f"  - Format: {format.upper()}")
    print(f"  - Resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {len(frames) / fps:.2f} seconds")
    print(f"  - File size: {file_size_mb:.2f} MB")


def generate_mp4(
    frames: List[Path],
    output_path: Path,
    fps: int = 15,
    target_resolution: Optional[Tuple[int, int]] = None,
    quality: int = 10,
) -> None:
    """
    Generate an MP4 video from a sequence of frames using imageio.
    
    Args:
        frames: List of frame paths in sequence order
        output_path: Output path for the video
        fps: Frames per second (default: 15)
        target_resolution: Optional (width, height) to resize all frames to.
                          If None, uses the first frame's size.
        quality: Video quality 1-10, where 10 is best (default: 10)
    """
    if not IMAGEIO_AVAILABLE:
        raise RuntimeError(
            "imageio is not installed. Install with: uv add imageio imageio-ffmpeg"
        )
    
    if not frames:
        raise ValueError("No frames provided")
    
    # First, quickly scan for target resolution if not provided
    if target_resolution is None:
        img = Image.open(frames[0])
        target_resolution = img.size
        img.close()
        print(f"Using first frame size as reference: {target_resolution[0]}x{target_resolution[1]}")
    else:
        print(f"Target resolution from config: {target_resolution[0]}x{target_resolution[1]}")
    
    # Load and process all frames in parallel
    print(f"Loading and processing {len(frames)} frames for MP4...")
    frame_sizes = {}
    processed_frames = [None] * len(frames)
    
    # Use ThreadPoolExecutor for I/O-bound operations (image loading)
    max_workers = min(8, len(frames))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (reuse existing load_and_process_frame)
        futures = {
            executor.submit(
                load_and_process_frame,
                frame_path,
                target_resolution,
                "mp4",  # format doesn't matter for RGB conversion
                i
            ): i for i, frame_path in enumerate(frames)
        }
        
        # Collect results with progress indication
        completed = 0
        for future in as_completed(futures):
            index, img, original_size = future.result()
            processed_frames[index] = img
            
            # Track unique sizes for reporting
            if original_size not in frame_sizes:
                frame_sizes[original_size] = []
            frame_sizes[original_size].append(frames[index].name)
            
            completed += 1
            if completed % 50 == 0 or completed == len(frames):
                print(f"  Processed {completed}/{len(frames)} frames...")
    
    # Report any size mismatches
    if len(frame_sizes) > 1:
        print(f"⚠ Warning: Found {len(frame_sizes)} different frame sizes. Resized all to {target_resolution[0]}x{target_resolution[1]}:")
        for size, frame_list in frame_sizes.items():
            print(f"  - {size[0]}x{size[1]}: {len(frame_list)} frames (e.g., {frame_list[0]})")
    
    # Convert PIL Images to numpy arrays for imageio
    print(f"Encoding MP4 video at {fps} FPS (quality: {quality}/10)...")
    import numpy as np
    frame_arrays = [np.array(img) for img in processed_frames]
    
    # Configure video writer with h264 codec
    # CRF (Constant Rate Factor): lower = better quality & larger files
    # CRF mapping for quality 1-10:
    #   quality 10 → CRF 18 (visually lossless, large files)
    #   quality 8  → CRF 22 (very high quality, default)
    #   quality 5  → CRF 28 (good quality, recommended for 512x256)
    #   quality 3  → CRF 32 (medium quality, smaller files)
    #   quality 1  → CRF 36 (acceptable quality, small files)
    crf_value = 38 - (quality * 2)  # Maps quality 1-10 to CRF 36-18
    
    writer_kwargs = {
        'codec': 'libx264',
        'quality': quality,
        'pixelformat': 'yuv420p',  # Compatible with most players
        'ffmpeg_params': [
            '-preset', 'medium',  # Balance between speed and compression
            '-crf', str(crf_value),
        ],
    }
    
    # Write video
    imageio.mimwrite(
        str(output_path),
        frame_arrays,
        fps=fps,
        **writer_kwargs
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    crf_used = 38 - (quality * 2)
    print(f"✓ MP4 video saved to: {output_path}")
    print(f"  - Format: MP4 (h264)")
    print(f"  - Resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {len(frames) / fps:.2f} seconds")
    print(f"  - Quality: {quality}/10 (CRF {crf_used})")
    print(f"  - File size: {file_size_mb:.2f} MB")


def generate_thumbnail_and_full(
    frames: List[Path],
    base_output_path: Path,
    fps: int = 15,
    thumbnail_duration: float = 10.0,
    target_resolution: Optional[Tuple[int, int]] = None,
    mp4_quality: int = 5,
) -> Tuple[Path, Path]:
    """
    Generate both a thumbnail WebP preview and full MP4 video.
    
    Args:
        frames: List of frame paths in sequence order
        base_output_path: Base output path (without extension)
        fps: Frames per second (default: 15)
        thumbnail_duration: Duration of thumbnail preview in seconds (default: 10.0)
        target_resolution: Optional (width, height) to resize all frames to
        mp4_quality: Video quality 1-10 for MP4 (default: 8)
    
    Returns:
        Tuple of (thumbnail_path, full_video_path)
    """
    # Calculate how many frames for thumbnail
    thumbnail_frame_count = int(fps * thumbnail_duration)
    thumbnail_frames = frames[:thumbnail_frame_count]
    
    # Generate paths
    thumbnail_path = base_output_path.parent / f"{base_output_path.stem}_preview.webp"
    full_path = base_output_path.parent / f"{base_output_path.stem}_full.mp4"
    
    print(f"\n{'='*60}")
    print(f"Generating dual output: thumbnail + full video")
    print(f"{'='*60}")
    
    # Generate thumbnail WebP (first N seconds)
    print(f"\n[1/2] Creating thumbnail preview ({thumbnail_duration}s, {len(thumbnail_frames)} frames)...")
    generate_animation(
        frames=thumbnail_frames,
        output_path=thumbnail_path,
        fps=fps,
        format="webp",
        loop=True,
        target_resolution=target_resolution,
        fast_mode=False,
    )
    
    # Generate full MP4
    print(f"\n[2/2] Creating full MP4 video ({len(frames) / fps:.1f}s, {len(frames)} frames)...")
    generate_mp4(
        frames=frames,
        output_path=full_path,
        fps=fps,
        target_resolution=target_resolution,
        quality=mp4_quality,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Dual output complete!")
    print(f"  - Preview (for README): {thumbnail_path.name}")
    print(f"  - Full video (for viewing): {full_path.name}")
    print(f"{'='*60}\n")
    
    return thumbnail_path, full_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate looping animation from keyframes and interpolations. "
                   "Automatically resizes frames to match config resolution to handle mixed frame sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate WebP at 5 FPS (default, uses config resolution)
  uv run backend/tools/generate_animation.py

  # Generate GIF at 4 FPS
  uv run backend/tools/generate_animation.py --format gif --fps 4

  # Generate MP4 video at 15 FPS (better for long animations!)
  uv run backend/tools/generate_animation.py --format mp4 --fps 15

  # Dual output: WebP preview (10s) + full MP4 (perfect for README + sharing)
  uv run backend/tools/generate_animation.py --dual-output --fps 15

  # Dual output with custom preview duration (15 seconds)
  uv run backend/tools/generate_animation.py --dual-output --thumbnail-duration 15

  # Custom output path
  uv run backend/tools/generate_animation.py --output my_animation.webp

  # Override resolution manually
  uv run backend/tools/generate_animation.py --resolution 512x256

  # Use custom input directory
  uv run backend/tools/generate_animation.py --input-dir path/to/output

  # High quality MP4 for archival (quality 10/10, large files!)
  uv run backend/tools/generate_animation.py --format mp4 --mp4-quality 10

  # Smaller file size while maintaining good quality
  uv run backend/tools/generate_animation.py --format mp4 --mp4-quality 3

Notes:
  - All frames are automatically resized to match the target resolution
  - Target resolution is loaded from backend/config.yaml by default
  - Use --resolution to override the config resolution
  - Mixed frame sizes are detected and reported with warnings
  - MP4 format is MUCH more efficient for long animations (5+ min)
  - Dual output mode creates both a WebP preview and full MP4 video
  - MP4 quality recommendations:
    * 10: Visually lossless (~100-300 MB for 5 min, overkill for 512x256)
    * 5-7: Good quality (~5-15 MB, recommended for 512x256)
    * 3-4: Medium quality (~3-8 MB, smaller files, still looks good)
    * 1-2: Lower quality (~1-3 MB, acceptable for previews)
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output"),
        help="Input directory containing keyframes/ and interpolations/ subdirs (default: output)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: output/finished_product.webp or .gif)",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        choices=range(1, 61),
        metavar="FPS",
        help="Frames per second (1-60, default: 5)",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="webp",
        choices=["webp", "gif", "mp4"],
        help="Output format: webp, gif, or mp4 (default: webp)",
    )
    
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable looping (animation plays once)",
    )
    
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Override target resolution (e.g., 512x256). If not specified, loads from config.yaml",
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: faster encoding with slightly larger file size",
    )
    
    parser.add_argument(
        "--dual-output",
        action="store_true",
        help="Generate both WebP preview and full MP4 video (ignores --format)",
    )
    
    parser.add_argument(
        "--thumbnail-duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Duration of WebP preview in dual-output mode (default: 10.0 seconds)",
    )
    
    parser.add_argument(
        "--mp4-quality",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="QUALITY",
        help="MP4 video quality 1-10, where 10 is best (default: 5, recommended for 512x256)",
    )
    
    args = parser.parse_args()
    
    # Determine target resolution
    target_resolution = None
    if args.resolution:
        # Parse manual resolution override
        try:
            width, height = map(int, args.resolution.lower().split('x'))
            target_resolution = (width, height)
            print(f"Using manual resolution override: {width}x{height}")
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'. Use format: WIDTHxHEIGHT (e.g., 512x256)")
            return 1
    else:
        # Load from config
        try:
            config = load_config()
            target_resolution = get_target_resolution(config)
            print(f"Loaded resolution from config: {target_resolution[0]}x{target_resolution[1]}")
        except Exception as e:
            print(f"Warning: Could not load config ({e}). Will use first frame size as reference.")
            target_resolution = None
    
    # Determine output path
    if args.output is None:
        if args.dual_output:
            # For dual output, use base name without extension
            args.output = args.input_dir / "finished_product"
        else:
            args.output = args.input_dir / f"finished_product.{args.format}"
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Collect frames
        print(f"Scanning {args.input_dir} for keyframes and interpolations...")
        frames = collect_frames(args.input_dir)
        
        if not frames:
            print("Error: No frames found!")
            return 1
        
        print(f"Found {len(frames)} frames in sequence")
        
        # Check if dual output mode
        if args.dual_output:
            # Generate both WebP preview and full MP4
            generate_thumbnail_and_full(
                frames=frames,
                base_output_path=args.output,
                fps=args.fps,
                thumbnail_duration=args.thumbnail_duration,
                target_resolution=target_resolution,
                mp4_quality=args.mp4_quality,
            )
        elif args.format == "mp4":
            # Generate MP4 only
            if not IMAGEIO_AVAILABLE:
                print("Error: imageio is required for MP4 generation.")
                print("Install with: uv add imageio imageio-ffmpeg")
                return 1
            
            generate_mp4(
                frames=frames,
                output_path=args.output,
                fps=args.fps,
                target_resolution=target_resolution,
                quality=args.mp4_quality,
            )
        else:
            # Generate WebP or GIF
            generate_animation(
                frames=frames,
                output_path=args.output,
                fps=args.fps,
                format=args.format,
                loop=not args.no_loop,
                target_resolution=target_resolution,
                fast_mode=args.fast,
            )
        
        print("\n✓ Animation generated successfully!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

