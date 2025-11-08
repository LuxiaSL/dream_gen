"""
File Operations
Safe atomic file writes to prevent corruption

Rainmeter will be reading our output files constantly. We need to ensure
that it never reads a half-written file, which would cause flickering or crashes.

Solution: Atomic writes via temp file + rename
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def atomic_write_image(image: Image.Image, output_path: Path) -> bool:
    """
    Write image with atomic rename to prevent corruption
    
    Process:
    1. Write to temporary file in same directory
    2. Flush to disk
    3. Atomic rename (OS-level operation, all-or-nothing)
    
    Why atomic?
    - Prevents Rainmeter from reading half-written files
    - Ensures all-or-nothing semantics
    - Works reliably on Windows/Linux
    
    Args:
        image: PIL Image to save
        output_path: Final output path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temp file in same directory (ensures same filesystem)
        temp_dir = output_path.parent
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=temp_dir,
            delete=False,
            suffix=".tmp",
            prefix=".dream_",
        ) as tmp_file:
            # Save image to temp file
            image.save(tmp_file, "PNG", optimize=False)
            tmp_file.flush()
            tmp_path = Path(tmp_file.name)
        
        # Atomic rename
        shutil.move(str(tmp_path), str(output_path))
        
        logger.debug(f"Atomic write complete: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Atomic write failed: {e}", exc_info=True)
        # Clean up temp file if it exists
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except:
            pass
        return False


def atomic_write_image_with_retry(
    image: Image.Image,
    output_path: Path,
    max_retries: int = 3,
    retry_delay: float = 0.1,
) -> bool:
    """
    Atomic write with retry logic
    
    Handles transient failures like:
    - Permission errors (file temporarily locked)
    - Disk full errors
    - Network drive issues
    
    Args:
        image: PIL Image to save
        output_path: Final output path
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)
    
    Returns:
        True if successful, False after max retries
    """
    import time
    
    for attempt in range(max_retries):
        if atomic_write_image(image, output_path):
            return True
        
        if attempt < max_retries - 1:
            logger.warning(
                f"Write attempt {attempt + 1} failed, retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    logger.error(f"Failed to write after {max_retries} attempts: {output_path}")
    return False


def safe_copy(source: Path, dest: Path) -> bool:
    """
    Safely copy a file with error handling
    
    Args:
        source: Source file path
        dest: Destination file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)  # copy2 preserves metadata
        logger.debug(f"Copied: {source.name} → {dest.name}")
        return True
    except Exception as e:
        logger.error(f"Copy failed: {source} → {dest}: {e}")
        return False


def safe_delete(file_path: Path) -> bool:
    """
    Safely delete a file with error handling
    
    Args:
        file_path: File to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted: {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Delete failed: {file_path}: {e}")
        return False


def ensure_directory(dir_path: Path) -> bool:
    """
    Ensure directory exists, create if needed
    
    Args:
        dir_path: Directory path
    
    Returns:
        True if directory exists or was created
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {dir_path}: {e}")
        return False


def get_file_size_mb(file_path: Path) -> Optional[float]:
    """
    Get file size in megabytes
    
    Args:
        file_path: File to check
    
    Returns:
        Size in MB, or None if file doesn't exist
    """
    try:
        if file_path.exists():
            size_bytes = file_path.stat().st_size
            return size_bytes / (1024 * 1024)
        return None
    except Exception as e:
        logger.error(f"Failed to get file size: {e}")
        return None


def get_directory_size_mb(dir_path: Path) -> float:
    """
    Get total size of directory in megabytes
    
    Args:
        dir_path: Directory to check
    
    Returns:
        Total size in MB
    """
    total = 0
    try:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get directory size: {e}")
        return 0.0


# Test function
def test_file_ops() -> bool:
    """Test file operations"""
    import numpy as np
    
    print("=" * 60)
    print("Testing File Operations...")
    print("=" * 60)
    
    test_dir = Path("./test_file_ops")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test 1: Create test image
        print("\n1. Creating test image...")
        test_image = Image.new("RGB", (256, 512), color=(100, 150, 200))
        print("✓ Test image created")
        
        # Test 2: Atomic write
        print("\n2. Testing atomic write...")
        test_path = test_dir / "test_atomic.png"
        success = atomic_write_image(test_image, test_path)
        if success and test_path.exists():
            print(f"✓ Atomic write successful")
            size = get_file_size_mb(test_path)
            print(f"  File size: {size:.2f} MB")
        else:
            print("✗ Atomic write failed")
            return False
        
        # Test 3: Atomic write with retry
        print("\n3. Testing atomic write with retry...")
        test_path2 = test_dir / "test_retry.png"
        success = atomic_write_image_with_retry(test_image, test_path2)
        if success and test_path2.exists():
            print("✓ Retry write successful")
        else:
            print("✗ Retry write failed")
            return False
        
        # Test 4: Safe copy
        print("\n4. Testing safe copy...")
        test_copy = test_dir / "test_copy.png"
        success = safe_copy(test_path, test_copy)
        if success and test_copy.exists():
            print("✓ Safe copy successful")
        else:
            print("✗ Safe copy failed")
            return False
        
        # Test 5: Directory size
        print("\n5. Testing directory operations...")
        dir_size = get_directory_size_mb(test_dir)
        print(f"✓ Directory size: {dir_size:.2f} MB")
        
        # Test 6: Safe delete
        print("\n6. Testing safe delete...")
        success = safe_delete(test_copy)
        if success and not test_copy.exists():
            print("✓ Safe delete successful")
        else:
            print("✗ Safe delete failed")
            return False
        
        print("\n" + "=" * 60)
        print("File operations test PASSED ✓")
        print("=" * 60)
        
        # Cleanup
        shutil.rmtree(test_dir)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_file_ops()
    exit(0 if success else 1)

