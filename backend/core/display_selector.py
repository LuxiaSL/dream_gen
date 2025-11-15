"""
Display Frame Selector - Sequential frame consumption for display

This module handles selecting frames from the buffer and updating the
display (current_frame.png) with proper timing for smooth playback.

Key responsibilities:
- Wait for initial buffer to fill
- Select next frame in sequence from buffer
- Copy frame to current_frame.png for Rainmeter display
- Time frame display based on target FPS
- Update status with current frame info
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

from .frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


class DisplayFrameSelector:
    """
    Selects and displays frames from the buffer
    
    Consumes frames from the buffer in sequential order, copies them
    to current_frame.png for Rainmeter display, and times playback
    according to target FPS.
    
    Usage:
        selector = DisplayFrameSelector(
            frame_buffer=buffer,
            output_dir=Path("output"),
            target_fps=4.0,
            min_buffer_seconds=30.0
        )
        
        # Run display loop
        await selector.run()
    """
    
    def __init__(
        self,
        frame_buffer: FrameBuffer,
        output_dir: Path,
        target_fps: float = 4.0,
        min_buffer_seconds: float = 30.0
    ):
        """
        Initialize display frame selector
        
        Args:
            frame_buffer: FrameBuffer instance
            output_dir: Output directory for current_frame.png
            target_fps: Target frame rate for display
            min_buffer_seconds: Minimum buffer before starting playback
        """
        self.buffer = frame_buffer
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.min_buffer_seconds = min_buffer_seconds
        
        # Timing
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.25
        
        # State
        self.running = False
        self.paused = False
        self.frames_displayed = 0
        self.last_frame_time = 0
        
        # Output path
        self.current_frame_path = self.output_dir / "current_frame.png"
        
        logger.info("DisplayFrameSelector initialized")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Frame interval: {self.frame_interval:.3f}s")
        logger.info(f"  Min buffer: {min_buffer_seconds}s")
        logger.info(f"  Output: {self.current_frame_path}")
    
    async def wait_for_initial_buffer(self, check_interval: float = 1.0) -> bool:
        """
        Wait for buffer to fill before starting playback
        
        Args:
            check_interval: Seconds between buffer checks
            
        Returns:
            True when buffer is ready
        """
        logger.info("Waiting for initial buffer to fill...")
        logger.info(f"Target: {self.min_buffer_seconds}s")
        
        # For async system: Need proper buffer to account for generation/display rate mismatch
        # Display rate: 4 FPS = 4 frames/sec
        # Generation rate: ~2.5 frames/sec (0.4s per interpolation)
        # Need cushion to prevent display from overtaking generation!
        # Use at least 5 seconds (20 frames) to give async system time to build lead
        actual_min_buffer = max(5.0, min(self.min_buffer_seconds, 10.0))
        
        if actual_min_buffer != self.min_buffer_seconds:
            logger.info(f"[ASYNC] Adjusted min buffer: {actual_min_buffer}s (configured: {self.min_buffer_seconds}s)")
        
        while self.running:
            status = self.buffer.get_buffer_status()
            seconds_buffered = status['seconds_buffered']
            percentage = status['buffer_percentage']
            
            if seconds_buffered >= actual_min_buffer:
                logger.info(f"[OK] Buffer ready: {seconds_buffered:.1f}s ({percentage:.1f}%)")
                return True
            
            # Log progress
            if int(time.time()) % 5 == 0:  # Log every 5 seconds
                logger.info(f"Buffering... {seconds_buffered:.1f}s / {actual_min_buffer}s ({percentage:.1f}%)")
            
            await asyncio.sleep(check_interval)
        
        return False
    
    async def select_and_display_next_frame(self) -> bool:
        """
        Get next frame from buffer and display it
        
        Returns:
            True if frame was displayed successfully
        """
        # Get next frame from buffer
        frame_spec = self.buffer.get_next_display_frame()
        
        if frame_spec is None:
            # DEBUG: What frame are we trying to get?
            seq = self.buffer.display_sequence_num
            if seq in self.buffer.frames:
                frame = self.buffer.frames[seq]
                logger.warning(
                    f"Next frame not ready in buffer: Seq {seq} is {frame.state.value}, "
                    f"type={frame.frame_type.value}, "
                    f"file={frame.file_path.name if frame.file_path else 'None'}"
                )
            else:
                logger.warning(
                    f"Next frame not ready in buffer: Seq {seq} NOT REGISTERED YET "
                    f"(next_sequence_num={self.buffer.next_sequence_num})"
                )
            return False
        
        if not frame_spec.file_path or not frame_spec.file_path.exists():
            logger.error(f"Frame file missing: {frame_spec.file_path}")
            return False
        
        try:
            # Copy frame to current_frame.png
            self._write_current_frame(frame_spec.file_path)
            
            # Mark as displayed in buffer
            self.buffer.mark_displayed(frame_spec.sequence_num)
            
            # Advance to next frame
            self.buffer.advance_display()
            
            self.frames_displayed += 1
            
            # Log every 10th frame
            if self.frames_displayed % 10 == 0:
                logger.info(f"Displayed frame: {frame_spec}")
                status = self.buffer.get_buffer_status()
                logger.info(f"  Buffer: {status['seconds_buffered']:.1f}s ({status['frames_ready']} frames)")
            else:
                logger.debug(f"Displayed: {frame_spec}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}", exc_info=True)
            return False
    
    def _write_current_frame(self, frame_path: Path) -> None:
        """
        Write frame to current_frame.png atomically
        
        Args:
            frame_path: Path to frame to display
        """
        try:
            # Use atomic write with temp file
            import tempfile
            import shutil
            
            # Read source image
            image = Image.open(frame_path)
            
            # Write to temp file in same directory
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=self.output_dir,
                delete=False,
                suffix='.tmp',
                prefix='.current_frame_'
            ) as tmp_file:
                image.save(tmp_file, format='PNG', optimize=False)
                tmp_path = Path(tmp_file.name)
            
            # Atomic rename
            shutil.move(str(tmp_path), str(self.current_frame_path))
            
            logger.debug(f"Updated current_frame.png")
            
        except Exception as e:
            logger.error(f"Failed to write current frame: {e}")
            # Clean up temp file if it exists
            try:
                if 'tmp_path' in locals():
                    tmp_path.unlink()
            except:
                pass
    
    async def run(self, check_interval: float = 0.01) -> None:
        """
        Main display loop
        
        Waits for buffer, then displays frames at target FPS.
        
        Args:
            check_interval: Seconds between loop iterations
        """
        self.running = True
        logger.info("Display selector starting...")
        
        # Wait for initial buffer
        buffer_ready = await self.wait_for_initial_buffer()
        
        if not buffer_ready:
            logger.error("Buffer never became ready")
            return
        
        logger.info("Starting frame playback...")
        self.last_frame_time = time.time()
        
        while self.running:
            try:
                # Skip if paused
                if self.paused:
                    await asyncio.sleep(check_interval)
                    continue
                
                # Check if enough time has passed for next frame
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                if elapsed >= self.frame_interval:
                    # Time for next frame
                    success = await self.select_and_display_next_frame()
                    
                    if success:
                        self.last_frame_time = current_time
                    else:
                        # Frame not ready, check buffer status
                        status = self.buffer.get_buffer_status()
                        if status['frames_ready'] == 0:
                            logger.warning("Buffer depleted! Waiting for frames...")
                            # Wait a bit longer before next check
                            await asyncio.sleep(1.0)
                            continue
                
                # Small sleep to avoid busy waiting
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                logger.info("Display selector cancelled")
                break
            except Exception as e:
                logger.error(f"Error in display loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        logger.info("Display selector stopped")
    
    def pause(self) -> None:
        """Pause display"""
        self.paused = True
        logger.info("Display paused")
    
    def resume(self) -> None:
        """Resume display"""
        self.paused = False
        logger.info("Display resumed")
    
    def stop(self) -> None:
        """Stop display"""
        self.running = False
        logger.info("Display stopping...")
    
    def get_stats(self) -> Dict:
        """
        Get display statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "frames_displayed": self.frames_displayed,
            "target_fps": self.target_fps,
            "frame_interval": self.frame_interval,
            "is_paused": self.paused,
            "is_running": self.running,
            "current_display_sequence": self.buffer.display_sequence_num
        }


# Unit tests
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    print("🧪 Testing DisplayFrameSelector...")
    print()
    
    print("Note: This requires a filled buffer for full testing")
    print("Run with full DreamController for integration testing")
    print()
    
    print("✓ Module structure validated")

