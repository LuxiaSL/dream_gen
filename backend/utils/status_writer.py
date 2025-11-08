"""
Status Writer
Writes status.json for Rainmeter to read

This provides real-time information to the Rainmeter widget about
generation progress, performance, and system state.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .file_ops import atomic_write_image

logger = logging.getLogger(__name__)


class StatusWriter:
    """
    Write status information to JSON for Rainmeter
    
    Status JSON includes:
    - Frame count
    - Generation time
    - Current status (live/paused/error)
    - Current prompt
    - Cache information
    - Performance metrics
    - Uptime
    """

    def __init__(self, output_dir: Path):
        """
        Initialize status writer
        
        Args:
            output_dir: Directory to write status.json
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.output_dir / "status.json"
        self.start_time = datetime.now()
        
        # Initialize with default status
        self._write_initial_status()
        
        logger.info(f"StatusWriter initialized: {self.status_file}")

    def _write_initial_status(self) -> None:
        """Write initial status on startup"""
        initial_status = {
            "status": "initializing",
            "frame_number": 0,
            "generation_time": 0.0,
            "current_prompt": "",
            "current_mode": "idle",
            "cache_size": 0,
            "cache_hits": 0,
            "uptime_hours": 0.0,
            "last_update": datetime.now().isoformat(),
        }
        self._write_json(initial_status)

    def write_status(self, data: Dict[str, Any]) -> bool:
        """
        Write status update
        
        Args:
            data: Status data dictionary
        
        Returns:
            True if successful
        """
        # Add standard fields
        data.update({
            "last_update": datetime.now().isoformat(),
            "uptime_hours": self._get_uptime_hours(),
        })
        
        return self._write_json(data)

    def write_generation_status(
        self,
        frame_number: int,
        generation_time: float,
        prompt: str,
        mode: str = "img2img",
        cache_size: int = 0,
        cache_hits: int = 0,
    ) -> bool:
        """
        Write generation status (convenience method)
        
        Args:
            frame_number: Current frame number
            generation_time: Time taken to generate (seconds)
            prompt: Current prompt used
            mode: Generation mode (img2img/interpolate/hybrid)
            cache_size: Current cache size
            cache_hits: Total cache hits
        
        Returns:
            True if successful
        """
        status = {
            "status": "live",
            "frame_number": frame_number,
            "generation_time": round(generation_time, 2),
            "current_prompt": prompt[:100],  # Truncate long prompts
            "current_mode": mode,
            "cache_size": cache_size,
            "cache_hits": cache_hits,
        }
        return self.write_status(status)

    def write_paused_status(self, reason: str = "game_detected") -> bool:
        """
        Write paused status
        
        Args:
            reason: Reason for pause
        
        Returns:
            True if successful
        """
        status = {
            "status": "paused",
            "pause_reason": reason,
        }
        return self.write_status(status)

    def write_error_status(self, error_message: str) -> bool:
        """
        Write error status
        
        Args:
            error_message: Error description
        
        Returns:
            True if successful
        """
        status = {
            "status": "error",
            "error_message": error_message,
        }
        return self.write_status(status)

    def _write_json(self, data: Dict[str, Any]) -> bool:
        """
        Write JSON data atomically
        
        Uses temp file + rename for atomic writes
        
        Args:
            data: Data to write
        
        Returns:
            True if successful
        """
        import tempfile
        import shutil
        
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.output_dir,
                delete=False,
                suffix=".tmp",
                prefix=".status_",
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                tmp_path = tmp_file.name
            
            # Atomic rename
            shutil.move(tmp_path, str(self.status_file))
            
            logger.debug("Status updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write status: {e}")
            # Clean up temp file
            try:
                if "tmp_path" in locals():
                    Path(tmp_path).unlink()
            except:
                pass
            return False

    def _get_uptime_hours(self) -> float:
        """
        Get uptime in hours
        
        Returns:
            Uptime since initialization (hours)
        """
        elapsed = datetime.now() - self.start_time
        return round(elapsed.total_seconds() / 3600, 2)

    def read_status(self) -> Dict[str, Any]:
        """
        Read current status (for debugging)
        
        Returns:
            Current status dictionary
        """
        try:
            if self.status_file.exists():
                with open(self.status_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to read status: {e}")
            return {}


# Test function
def test_status_writer() -> bool:
    """Test status writer"""
    import time
    
    print("=" * 60)
    print("Testing StatusWriter...")
    print("=" * 60)
    
    test_dir = Path("./test_status")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create writer
        print("\n1. Creating status writer...")
        writer = StatusWriter(test_dir)
        print("✓ Writer created")
        
        # Check initial status
        print("\n2. Checking initial status...")
        status = writer.read_status()
        if status and status.get("status") == "initializing":
            print("✓ Initial status correct")
        else:
            print("✗ Initial status incorrect")
            return False
        
        # Write generation status
        print("\n3. Writing generation status...")
        success = writer.write_generation_status(
            frame_number=42,
            generation_time=1.85,
            prompt="ethereal digital angel",
            mode="img2img",
            cache_size=25,
            cache_hits=5,
        )
        if success:
            print("✓ Generation status written")
            status = writer.read_status()
            print(f"   Frame: {status.get('frame_number')}")
            print(f"   Gen time: {status.get('generation_time')}s")
            print(f"   Mode: {status.get('current_mode')}")
        else:
            print("✗ Generation status write failed")
            return False
        
        # Write paused status
        print("\n4. Writing paused status...")
        time.sleep(0.1)
        success = writer.write_paused_status("game_detected")
        if success:
            print("✓ Paused status written")
            status = writer.read_status()
            if status.get("status") == "paused":
                print(f"   Reason: {status.get('pause_reason')}")
            else:
                print("✗ Pause status not set correctly")
                return False
        else:
            print("✗ Paused status write failed")
            return False
        
        # Write error status
        print("\n5. Writing error status...")
        time.sleep(0.1)
        success = writer.write_error_status("Test error message")
        if success:
            print("✓ Error status written")
            status = writer.read_status()
            if status.get("status") == "error":
                print(f"   Error: {status.get('error_message')}")
            else:
                print("✗ Error status not set correctly")
                return False
        else:
            print("✗ Error status write failed")
            return False
        
        # Check uptime
        print("\n6. Checking uptime tracking...")
        time.sleep(0.1)
        writer.write_status({"test": "data"})
        status = writer.read_status()
        uptime = status.get("uptime_hours", 0)
        if uptime >= 0:
            print(f"✓ Uptime: {uptime} hours")
        else:
            print("✗ Uptime tracking failed")
            return False
        
        print("\n" + "=" * 60)
        print("StatusWriter test PASSED ✓")
        print("=" * 60)
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_status_writer()
    exit(0 if success else 1)

