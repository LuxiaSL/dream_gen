"""
Cloud Frame Pusher

Encodes frames to WebP format and pushes them to the VPS via WebSocket.
Handles both keyframes and interpolation frames based on configuration.

WebP encoding at 85% quality provides excellent compression (~40-70KB per
1024x512 frame) while maintaining visual quality for AI art.

Frame Message Format (v2):
  0x01 | metadata_len (4 bytes BE) | JSON metadata | WebP data

Metadata JSON:
  {
    "fn": frame_number,      // Sequential frame number
    "kf": keyframe_number,   // Current keyframe number
    "p": "prompt text"       // Prompt for this keyframe (optional)
  }

Self-healing features:
- Graceful handling of disconnections (continues encoding, queues frames)
- Connection status awareness for logging
- Keyframe priority for queue ordering during reconnection
"""

import asyncio
import json
import logging
import time
import io
from typing import Optional, Callable
from PIL import Image

from .websocket_client import VPSWebSocketClient, ConnectionState
from utils.perf_stats import get_perf_stats

logger = logging.getLogger(__name__)

# Type alias for push callbacks
PushCallback = Callable[[], None]

# Priority levels for frame queueing
PRIORITY_KEYFRAME = 10
PRIORITY_INTERPOLATION = 1


class CloudFramePusher:
    """
    Handles frame encoding and transmission to VPS
    
    Encodes PIL Images to WebP format and pushes via WebSocket.
    Tracks statistics for monitoring.
    
    Self-healing features:
    - Continues operating during disconnections (frames queued by WebSocket client)
    - Keyframes get higher queue priority than interpolations
    - Connection status logging for debugging
    """
    
    def __init__(self, websocket_client: VPSWebSocketClient, config: dict):
        """
        Initialize frame pusher
        
        Args:
            websocket_client: Connected VPS WebSocket client
            config: Cloud configuration dict containing:
                - frame_push.format: "webp" or "png"
                - frame_push.quality: WebP quality (1-100)
                - frame_push.include_interpolations: Push all frames or just keyframes
        """
        self.ws_client = websocket_client
        
        # Configuration
        frame_config = config.get('frame_push', {})
        self.format = frame_config.get('format', 'webp')
        self.quality = frame_config.get('quality', 85)
        self.include_interpolations = frame_config.get('include_interpolations', True)
        
        # Optional callback invoked after each successful push (for watchdog heartbeat, etc.)
        self._on_push_callback: Optional[PushCallback] = None
        
        # Statistics
        self.frames_pushed = 0
        self.keyframes_pushed = 0
        self.interpolations_pushed = 0
        self.frames_queued = 0  # Frames queued due to disconnect
        self.bytes_pushed = 0
        self.push_times = []  # Last N push times for avg calculation
        self.max_timing_samples = 100
        
        # Track disconnection state for logging
        self._last_connection_state = ConnectionState.DISCONNECTED
        self._disconnect_logged = False
        
        # Encoding buffer (reused to reduce allocations)
        self._buffer = io.BytesIO()
        
        logger.info(f"CloudFramePusher initialized: format={self.format}, quality={self.quality}")
    
    def set_push_callback(self, callback: Optional[PushCallback]) -> None:
        """
        Set a callback to be invoked after each successful frame push.
        
        Useful for watchdog heartbeat, metrics, etc. The callback is called
        synchronously after each successful push (not for skipped frames).
        
        Args:
            callback: Function to call after push, or None to clear
        """
        self._on_push_callback = callback
        if callback:
            logger.debug("Push callback registered")
        else:
            logger.debug("Push callback cleared")
    
    def _check_connection_state(self) -> None:
        """Log connection state changes for debugging"""
        current_state = self.ws_client.state
        
        if current_state != self._last_connection_state:
            if current_state == ConnectionState.CONNECTED:
                if self._disconnect_logged:
                    logger.info("VPS connection restored - resuming frame push")
                    self._disconnect_logged = False
            elif current_state == ConnectionState.RECONNECTING:
                if not self._disconnect_logged:
                    logger.warning("VPS connection lost - frames will be queued")
                    self._disconnect_logged = True
            elif current_state == ConnectionState.FAILED:
                logger.error("VPS connection failed (circuit breaker tripped)")
            
            self._last_connection_state = current_state
    
    async def push_frame(
        self,
        image: Image.Image,
        is_keyframe: bool = False,
        frame_number: int = 0,
        keyframe_number: int = 0,
        prompt: Optional[str] = None,
    ) -> bool:
        """
        Encode and push a frame to VPS with metadata
        
        Args:
            image: PIL Image to push
            is_keyframe: Whether this is a keyframe (vs interpolation)
            frame_number: Sequential frame number (server-authoritative)
            keyframe_number: Current keyframe number
            prompt: Prompt text for this frame's keyframe (optional)
        
        Returns:
            True if pushed successfully (or queued during disconnect)
        
        Message format (v2):
            0x01 | metadata_len (4 bytes BE) | JSON metadata | WebP data
        
        Note:
            During disconnection, frames are queued by the WebSocket client
            with keyframes getting higher priority. The function returns True
            if the frame was queued successfully.
        """
        # Check if we should push this frame
        if not is_keyframe and not self.include_interpolations:
            return True  # Skip interpolations, not an error
        
        # Log connection state changes
        self._check_connection_state()
        
        # Determine priority (keyframes are more important)
        priority = PRIORITY_KEYFRAME if is_keyframe else PRIORITY_INTERPOLATION
        
        start_time = time.time()
        
        try:
            # Encode frame (always encode, even if disconnected - will be queued)
            frame_bytes = self._encode_frame(image)
            encode_time = time.time() - start_time
            
            # Build metadata JSON
            metadata = {
                "fn": frame_number,
                "kf": keyframe_number,
            }
            if prompt:
                metadata["p"] = prompt
            
            metadata_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
            
            # Capture connection state BEFORE the send to determine if it will be queued
            # This avoids race conditions where connection drops after successful send
            was_connected_before = self.ws_client.connected
            
            # Push via WebSocket (with metadata and priority)
            # If disconnected, the WebSocket client will queue the frame
            push_start = time.time()
            success = await self.ws_client.send_frame_with_metadata(
                frame_bytes, metadata_bytes, priority=priority
            )
            push_time = time.time() - push_start
            
            total_time = time.time() - start_time
            
            if success:
                # Frame was queued if we weren't connected when we started the send
                was_queued = not was_connected_before
                
                # Update statistics
                self.frames_pushed += 1
                self.bytes_pushed += len(frame_bytes) + len(metadata_bytes) + 4
                
                if is_keyframe:
                    self.keyframes_pushed += 1
                else:
                    self.interpolations_pushed += 1
                
                if was_queued:
                    self.frames_queued += 1
                    # Don't log every queued frame, just periodically
                    if self.frames_queued % 10 == 1:
                        logger.info(
                            f"Frame {frame_number} queued (total queued: {self.frames_queued}, "
                            f"queue size: {self.ws_client.queue_size})"
                        )
                else:
                    # Track timing (only for actually sent frames)
                    self._record_timing(total_time)
                    
                    # Log frame push timing (always log for profiling)
                    logger.debug(
                        f"Pushed frame {frame_number}: {len(frame_bytes)/1024:.1f}KB "
                        f"(encode: {encode_time*1000:.1f}ms, push: {push_time*1000:.1f}ms)"
                    )
                    
                    # Log slow pushes for performance debugging
                    if total_time > 0.1:  # > 100ms is concerning
                        logger.info(
                            f"[PERF] Slow frame push {frame_number}: {total_time*1000:.1f}ms total "
                            f"(encode={encode_time*1000:.1f}ms, network={push_time*1000:.1f}ms, "
                            f"size={len(frame_bytes)/1024:.1f}KB)"
                        )
                    
                    # Invoke callback (for watchdog heartbeat, etc.)
                    # Only invoke for actually sent frames, not queued ones
                    if self._on_push_callback:
                        try:
                            self._on_push_callback()
                        except Exception as e:
                            logger.warning(f"Push callback failed: {e}")
                    
                    # Record to perf stats (tracks push throughput to VPS)
                    get_perf_stats().record_frame_push(total_time)
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to push frame: {e}")
            return False
    
    def _encode_frame(self, image: Image.Image) -> bytes:
        """
        Encode PIL Image to bytes
        
        Args:
            image: PIL Image to encode
        
        Returns:
            Encoded bytes
        """
        # Reset buffer
        self._buffer.seek(0)
        self._buffer.truncate()
        
        # Convert RGBA to RGB for formats that don't support alpha
        if image.mode == 'RGBA' and self.format in ('jpeg', 'jpg'):
            # Create white background and composite
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha as mask
            image = background
        elif image.mode == 'RGBA' and self.format == 'webp':
            # WebP supports RGBA, but RGB is smaller and we don't need alpha
            image = image.convert('RGB')
        elif image.mode not in ('RGB', 'L'):
            # Convert any other mode to RGB
            image = image.convert('RGB')
        
        if self.format == 'webp':
            # WebP encoding with quality setting
            image.save(
                self._buffer,
                format='WEBP',
                quality=self.quality,
                method=4,  # Compression method (0-6, 4 is good balance)
            )
        elif self.format == 'png':
            # PNG is lossless but larger
            image.save(self._buffer, format='PNG', optimize=True)
        else:
            # Default to JPEG as fallback
            image.save(self._buffer, format='JPEG', quality=self.quality)
        
        return self._buffer.getvalue()
    
    def _record_timing(self, time_seconds: float) -> None:
        """Record a timing sample"""
        self.push_times.append(time_seconds)
        if len(self.push_times) > self.max_timing_samples:
            self.push_times.pop(0)
    
    @property
    def average_push_time_ms(self) -> float:
        """Average push time in milliseconds"""
        if not self.push_times:
            return 0.0
        return sum(self.push_times) / len(self.push_times) * 1000
    
    @property
    def average_frame_size_kb(self) -> float:
        """Average frame size in KB"""
        if self.frames_pushed == 0:
            return 0.0
        return self.bytes_pushed / self.frames_pushed / 1024
    
    def get_stats(self) -> dict:
        """Get pusher statistics"""
        ws_stats = self.ws_client.get_stats()
        
        return {
            "frames_pushed": self.frames_pushed,
            "keyframes_pushed": self.keyframes_pushed,
            "interpolations_pushed": self.interpolations_pushed,
            "frames_queued": self.frames_queued,
            "bytes_pushed": self.bytes_pushed,
            "bytes_pushed_mb": round(self.bytes_pushed / 1024 / 1024, 2),
            "average_push_time_ms": round(self.average_push_time_ms, 2),
            "average_frame_size_kb": round(self.average_frame_size_kb, 2),
            "format": self.format,
            "quality": self.quality,
            # Include connection stats for visibility
            "connection": {
                "connected": ws_stats["connected"],
                "state": ws_stats["state"],
                "queue_size": ws_stats["queue_size"],
                "total_reconnects": ws_stats["total_reconnects"],
                "messages_dropped": ws_stats["messages_dropped"],
            }
        }


async def encode_frame_webp(image: Image.Image, quality: int = 85) -> bytes:
    """
    Utility function to encode a frame to WebP
    
    Runs encoding in thread pool to avoid blocking async loop.
    
    Args:
        image: PIL Image to encode
        quality: WebP quality (1-100)
    
    Returns:
        WebP-encoded bytes
    """
    loop = asyncio.get_event_loop()
    
    def _encode():
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP', quality=quality, method=4)
        return buffer.getvalue()
    
    return await loop.run_in_executor(None, _encode)

