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

  # Custom output path
  uv run backend/tools/generate_animation.py --output my_animation.webp

  # Override resolution manually
  uv run backend/tools/generate_animation.py --resolution 512x256

  # Use custom input directory
  uv run backend/tools/generate_animation.py --input-dir path/to/output

Notes:
  - All frames are automatically resized to match the target resolution
  - Target resolution is loaded from backend/config.yaml by default
  - Use --resolution to override the config resolution
  - Mixed frame sizes are detected and reported with warnings
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
        choices=["webp", "gif"],
        help="Output format: webp or gif (default: webp)",
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
        
        # Generate animation
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

