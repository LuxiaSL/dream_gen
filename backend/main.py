"""
Dream Window CLI Entry Point

Minimal command-line interface for running Dream Window.
Delegates all logic to DreamController.

Run with:
    uv run backend/main.py
    uv run backend/main.py --test
    uv run backend/main.py --config path/to/config.yaml
"""

import argparse
from core.dream_controller import DreamController


def main():
    """Command-line entry point"""
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
