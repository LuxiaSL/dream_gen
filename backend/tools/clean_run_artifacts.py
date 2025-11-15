#!/usr/bin/env python3
"""
Clean Run Artifacts Script

This script clears all traces of previous runs by removing generated files
and emptying cache directories.

Usage:
    python backend/tools/clean_run_artifacts.py
"""

import os
import shutil
from pathlib import Path


def get_project_root():
    """Get the project root directory (2 levels up from this script)."""
    return Path(__file__).parent.parent.parent


def clear_directory(directory: Path, description: str):
    """Remove all files and subdirectories in the given directory."""
    if not directory.exists():
        print(f"[WARN] {description} does not exist: {directory}")
        return
    
    removed_count = 0
    for item in directory.iterdir():
        try:
            if item.is_file():
                item.unlink()
                removed_count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed_count += 1
        except Exception as e:
            print(f"[ERROR] Error removing {item}: {e}")
    
    print(f"[OK] Cleared {description}: removed {removed_count} item(s)")


def remove_file(file_path: Path, description: str):
    """Remove a specific file if it exists."""
    if not file_path.exists():
        print(f"[INFO] {description} does not exist: {file_path}")
        return
    
    try:
        file_path.unlink()
        print(f"[OK] Removed {description}")
    except Exception as e:
        print(f"[ERROR] Error removing {description}: {e}")


def truncate_file(file_path: Path, description: str):
    """Empty a file's contents while keeping the file."""
    if not file_path.exists():
        print(f"[INFO] {description} does not exist: {file_path}")
        return
    
    try:
        file_path.write_text("")
        print(f"[OK] Emptied {description}")
    except Exception as e:
        print(f"[ERROR] Error emptying {description}: {e}")


def main():
    """Main cleanup function."""
    print("Starting cleanup of previous run artifacts...\n")
    
    root = get_project_root()
    
    # Clear output directories
    clear_directory(root / "output" / "keyframes", "output/keyframes/*")
    clear_directory(root / "output" / "interpolations", "output/interpolations/*")
    
    # Remove specific output files
    remove_file(root / "output" / "current_frame.png", "output/current_frame.png")
    remove_file(root / "output" / "status.json", "output/status.json")
    
    # Clear or truncate log file
    truncate_file(root / "logs" / "dream_controller.log", "logs/dream_controller.log")
    
    # Clear cache directories
    clear_directory(root / "cache" / "images", "cache/images/*")
    clear_directory(root / "cache" / "metadata", "cache/metadata/*")
    
    print("\nCleanup complete!")


if __name__ == "__main__":
    main()

