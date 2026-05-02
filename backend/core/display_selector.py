"""
Display Frame Selector - Sequential frame consumption for streaming

Consumes frames from the FrameBuffer at target FPS and passes them
to a callback (frame pusher) for H.264 encoding and WebSocket push.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Awaitable
from PIL import Image

from .frame_buffer import FrameBuffer, FrameState
from utils.perf_stats import get_perf_stats

logger = logging.getLogger(__name__)


class DisplayFrameSelector:
    """
    Rate-governed frame consumer for the generation pipeline.

    Pulls frames from FrameBuffer in sequence order at target FPS,
    invokes the on_frame_callback (typically CloudFramePusher), and
    frees in-memory images after display.
    """

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        output_dir: Path,
        target_fps: float = 4.0,
        min_buffer_seconds: float = 30.0,
        cleanup_displayed_frames: bool = False,
        on_frame_callback: Optional[Callable[[Image.Image, int, bool, int, Optional[str]], Awaitable[None]]] = None,
        skip_disk_write: bool = False,
        **_kwargs,
    ):
        self.buffer = frame_buffer
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.min_buffer_seconds = min_buffer_seconds
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.25

        self.running = False
        self.paused = False
        self.frames_displayed = 0
        self.skipped_frames = 0
        self.last_frame_time = 0
        self._depletion_count = 0

        self._current_prompt: Optional[str] = None
        self._current_keyframe_num: int = 0

        self.on_frame_callback = on_frame_callback

        logger.info("DisplayFrameSelector initialized")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Frame interval: {self.frame_interval:.3f}s")
        logger.info(f"  Min buffer: {min_buffer_seconds}s")
    


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
        
        try:
            image = frame_spec.image
            if image is None and frame_spec.file_path and frame_spec.file_path.exists():
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(
                    None,
                    lambda: Image.open(frame_spec.file_path)
                )
            if image is None:
                logger.error(f"Frame {frame_spec.sequence_num}: no image in memory or on disk")
                return False

            # Update current prompt and keyframe number on keyframe display
            is_keyframe = frame_spec.frame_type.value == 'keyframe'
            if is_keyframe:
                if frame_spec.prompt:
                    self._current_prompt = frame_spec.prompt
                if frame_spec.keyframe_num is not None:
                    self._current_keyframe_num = frame_spec.keyframe_num
            
            # Call optional callback (e.g., for cloud push)
            if self.on_frame_callback:
                try:
                    await self.on_frame_callback(
                        image, 
                        self.frames_displayed, 
                        is_keyframe,
                        self._current_keyframe_num,
                        self._current_prompt
                    )
                except Exception as callback_error:
                    logger.warning(f"Frame callback error (non-fatal): {callback_error}")
            
            # Mark as displayed in buffer
            self.buffer.mark_displayed(frame_spec.sequence_num)
            
            # Advance to next frame
            self.buffer.advance_display()
            
            self.frames_displayed += 1
            
            # Record to perf stats (tracks actual display rate)
            get_perf_stats().record_display_frame()
            
            # Free in-memory image
            if frame_spec.image is not None:
                frame_spec.image = None

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
    
    async def run(self, check_interval: float = 0.001) -> None:
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
        
        # Find the last ready keyframe
        ready_keyframes = [
            (seq, spec) for seq, spec in self.buffer.frames.items()
            if spec.is_keyframe() and spec.state == FrameState.READY
        ]
        
        if ready_keyframes:
            # Sort by sequence number and get the latest
            ready_keyframes.sort(key=lambda x: x[0], reverse=True)
            latest_seq, latest_spec = ready_keyframes[0]
            
            skipped = latest_seq - current_seq if latest_seq > current_seq else 0
            logger.warning(f"Recovery: Jumping to latest keyframe at seq {latest_seq}")
            self.buffer.display_sequence_num = latest_seq
            if skipped > 0:
                self.skipped_frames += skipped
        else:
            logger.error("No ready keyframes found - waiting for generation")
    
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
        
        return stats

