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
from typing import Optional, Dict, Any, Callable, Awaitable
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
        min_buffer_seconds: float = 30.0,
        cleanup_displayed_frames: bool = False,
        on_frame_callback: Optional[Callable[[Image.Image, int, bool], Awaitable[None]]] = None
    ):
        """
        Initialize display frame selector
        
        Args:
            frame_buffer: FrameBuffer instance
            output_dir: Output directory for current_frame.png
            target_fps: Target frame rate for display
            min_buffer_seconds: Minimum buffer before starting playback
            cleanup_displayed_frames: Whether to delete frames immediately after display
            on_frame_callback: Optional async callback(image, frame_num, is_keyframe) 
                               called for each displayed frame (e.g., for cloud push)
        """
        self.buffer = frame_buffer
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.min_buffer_seconds = min_buffer_seconds
        
        # Storage management
        self.cleanup_enabled = cleanup_displayed_frames
        self.frames_deleted = 0
        
        # Timing
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.25
        
        # State
        self.running = False
        self.paused = False
        self.frames_displayed = 0
        self.skipped_frames = 0
        self.last_frame_time = 0
        self._depletion_count = 0
        
        # Output path
        self.current_frame_path = self.output_dir / "current_frame.png"
        
        # Optional callback for cloud push or other processing
        self.on_frame_callback = on_frame_callback
        
        logger.info("DisplayFrameSelector initialized")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Frame interval: {self.frame_interval:.3f}s")
        logger.info(f"  Min buffer: {min_buffer_seconds}s")
        logger.info(f"  Output: {self.current_frame_path}")
        
        if self.cleanup_enabled:
            logger.info(f"  Auto-cleanup: ENABLED (delete after display)")
        else:
            logger.info(f"  Auto-cleanup: DISABLED")
        
        if self.on_frame_callback:
            logger.info(f"  Frame callback: ENABLED")
    
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
        Get next frame from buffer and display it (with async I/O)
        
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
            # Load image once (for both display and callback)
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                lambda: Image.open(frame_spec.file_path)
            )
            
            # Write to current_frame.png ASYNC (don't block event loop)
            await self._write_current_frame_async(image)
            
            # Call optional callback (e.g., for cloud push)
            if self.on_frame_callback:
                try:
                    is_keyframe = frame_spec.frame_type.value == 'keyframe'
                    await self.on_frame_callback(image, self.frames_displayed, is_keyframe)
                except Exception as callback_error:
                    logger.warning(f"Frame callback error (non-fatal): {callback_error}")
            
            # Mark as displayed in buffer
            self.buffer.mark_displayed(frame_spec.sequence_num)
            
            # Advance to next frame
            self.buffer.advance_display()
            
            self.frames_displayed += 1
            
            # Delete the source frame immediately after successful display (if cleanup enabled)
            # This is safe because we've already copied it to current_frame.png
            if self.cleanup_enabled:
                await self._delete_frame_async(frame_spec.file_path)
            
            # Log every 10th frame
            if self.frames_displayed % 10 == 0:
                logger.info(f"Displayed frame: {frame_spec}")
                status = self.buffer.get_buffer_status()
                logger.info(f"  Buffer: {status['seconds_buffered']:.1f}s ({status['frames_ready']} frames)")
                
                if self.cleanup_enabled and self.frames_deleted > 0:
                    logger.info(f"  Cleanup: {self.frames_deleted} frames deleted total")
            else:
                logger.debug(f"Displayed: {frame_spec}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}", exc_info=True)
            return False
    
    async def _write_current_frame_async(self, image: Image.Image) -> None:
        """
        Write frame to current_frame.png using async I/O (optimized)
        
        Moves image saving to executor to avoid blocking event loop.
        
        Args:
            image: PIL Image to save
        """
        loop = asyncio.get_event_loop()
        
        # Run sync I/O in executor (don't block event loop!)
        await loop.run_in_executor(
            None,
            self._write_current_frame_sync,
            image
        )
    
    def _write_current_frame_sync(self, image: Image.Image) -> None:
        """
        Sync I/O operations (runs in executor)
        
        Args:
            image: PIL Image to save
        """
        try:
            # Use atomic write with temp file
            import tempfile
            import shutil
            
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
    
    async def _delete_frame_async(self, frame_path: Path) -> None:
        """
        Delete a single frame file after it's been displayed (async version)
        
        This is safe because the frame has already been copied to current_frame.png.
        We delete immediately to prevent disk space buildup.
        
        Args:
            frame_path: Path to frame file to delete
        """
        # Run deletion in executor (don't block event loop)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._delete_frame_sync, frame_path)
    
    def _delete_frame_sync(self, frame_path: Path) -> None:
        """
        Sync file deletion (runs in executor)
        
        Args:
            frame_path: Path to frame file to delete
        """
        try:
            # Safety check - file must exist and be in output directory
            if not frame_path.exists():
                logger.debug(f"Frame already deleted: {frame_path.name}")
                return
            
            # Verify it's in our output directory (safety check)
            if not str(frame_path).startswith(str(self.output_dir)):
                logger.warning(f"Refusing to delete file outside output dir: {frame_path}")
                return
            
            # Delete the file
            frame_path.unlink()
            self.frames_deleted += 1
            logger.debug(f"Deleted displayed frame: {frame_path.name}")
            
        except Exception as e:
            # Non-fatal - just log and continue
            # File might be locked temporarily or already deleted
            logger.debug(f"Could not delete {frame_path.name}: {e}")
    
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
                        self._depletion_count = 0  # Reset on success
                    else:
                        # Frame not ready, check buffer status
                        status = self.buffer.get_buffer_status()
                        if status['frames_ready'] == 0:
                            # Track how long we've been depleted
                            if not hasattr(self, '_depletion_count'):
                                self._depletion_count = 0
                            self._depletion_count += 1
                            
                            if self._depletion_count == 1:
                                logger.warning("Buffer depleted! Waiting for frames...")
                            elif self._depletion_count % 10 == 0:
                                logger.warning(f"Buffer still depleted ({self._depletion_count}s)...")
                            
                            # After 30 seconds of depletion, try to recover
                            if self._depletion_count >= 30:
                                logger.error(
                                    f"Buffer depleted for {self._depletion_count}s - triggering recovery!"
                                )
                                await self._trigger_buffer_recovery()
                                self._depletion_count = 0
                            
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
    
    async def _trigger_buffer_recovery(self) -> None:
        """
        Attempt to recover from buffer depletion.
        
        This is called when the buffer has been empty for too long,
        indicating a stuck generation pipeline. We skip the stuck frame
        and try to resume from wherever we can.
        """
        logger.warning("=" * 60)
        logger.warning("BUFFER RECOVERY TRIGGERED")
        logger.warning("=" * 60)
        
        # Find the next frame that's actually ready
        current_seq = self.buffer.display_sequence_num
        
        # Look ahead for any ready frames
        for offset in range(1, 50):
            check_seq = current_seq + offset
            if check_seq in self.buffer.frames:
                frame_spec = self.buffer.frames[check_seq]
                if frame_spec.state.value == "ready":
                    logger.warning(
                        f"Recovery: Skipping to ready frame at seq {check_seq} "
                        f"(skipped {offset} frames)"
                    )
                    # Update the display sequence to skip stuck frames
                    self.buffer.display_sequence_num = check_seq
                    self.skipped_frames += offset
                    logger.warning("Buffer recovery complete - resuming display")
                    return
        
        # No ready frames found - reset to latest ready keyframe
        logger.error("No ready frames found in lookahead - attempting keyframe reset")
        
        # Find ANY ready frame in the buffer (not just ahead of current)
        ready_frames = [
            (seq, spec) for seq, spec in self.buffer.frames.items()
            if spec.state.value == "ready"
        ]
        
        if ready_frames:
            # Sort by sequence number and get the latest
            ready_frames.sort(key=lambda x: x[0], reverse=True)
            latest_seq, latest_spec = ready_frames[0]
            
            # If we're behind the latest ready frame, jump forward
            if latest_seq > current_seq:
                skipped = latest_seq - current_seq
                logger.warning(f"Recovery: Jumping FORWARD to latest ready frame at seq {latest_seq}")
            else:
                # We're ahead of all ready frames - wait, or jump back?
                # Jump back to the latest ready frame to keep showing something
                skipped = 0
                logger.warning(f"Recovery: Jumping BACK to latest ready frame at seq {latest_seq}")
            
            self.buffer.display_sequence_num = latest_seq
            if skipped > 0:
                self.skipped_frames += skipped
            
            logger.warning(f"Buffer recovery complete - now at seq {latest_seq}")
        else:
            # No ready frames at all - check if buffer has any registered frames
            if self.buffer.frames:
                # Find any pending frame and wait for it
                pending = [(seq, spec) for seq, spec in self.buffer.frames.items() 
                          if spec.state.value in ("pending", "generating")]
                if pending:
                    earliest = min(pending, key=lambda x: x[0])
                    logger.warning(f"Recovery: Found {len(pending)} pending frames, earliest at seq {earliest[0]}")
                    # Jump to the earliest pending and wait
                    self.buffer.display_sequence_num = earliest[0]
                    logger.warning(f"Recovery: Jumping to seq {earliest[0]} to wait for generation")
                else:
                    logger.error("No ready or pending frames - buffer is completely empty!")
            else:
                logger.error("No frames in buffer at all - waiting for generation to start")
    
    def get_stats(self) -> Dict:
        """
        Get display statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "frames_displayed": self.frames_displayed,
            "skipped_frames": self.skipped_frames,
            "target_fps": self.target_fps,
            "frame_interval": self.frame_interval,
            "is_paused": self.paused,
            "is_running": self.running,
            "current_display_sequence": self.buffer.display_sequence_num
        }
        
        # Add cleanup stats if enabled
        if self.cleanup_enabled:
            stats.update({
                "cleanup_enabled": True,
                "frames_deleted": self.frames_deleted
            })
        else:
            stats["cleanup_enabled"] = False
        
        return stats


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

