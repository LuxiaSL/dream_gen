"""
Cloud Frame Pusher

Encodes frames to WebP format and pushes them to the VPS via WebSocket.
Handles both keyframes and interpolation frames based on configuration.

WebP encoding at 85% quality provides excellent compression (~40-70KB per
1024x512 frame) while maintaining visual quality for AI art.
"""

import asyncio
import logging
import time
import io
from typing import Optional, Callable
from PIL import Image

from .websocket_client import VPSWebSocketClient
from utils.perf_stats import get_perf_stats

logger = logging.getLogger(__name__)

# Type alias for push callbacks
PushCallback = Callable[[], None]


class CloudFramePusher:
    """
    Handles frame encoding and transmission to VPS
    
    Encodes PIL Images to WebP format and pushes via WebSocket.
    Tracks statistics for monitoring.
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
        self.bytes_pushed = 0
        self.push_times = []  # Last N push times for avg calculation
        self.max_timing_samples = 100
        
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
    
    async def push_frame(
        self,
        image: Image.Image,
        is_keyframe: bool = False,
        frame_number: int = 0,
        keyframe_number: int = 0,
    ) -> bool:
        """
        Encode and push a frame to VPS
        
        Args:
            image: PIL Image to push
            is_keyframe: Whether this is a keyframe (vs interpolation)
            frame_number: Sequential frame number
            keyframe_number: Current keyframe number
        
        Returns:
            True if pushed successfully
        """
        # Check if we should push this frame
        if not is_keyframe and not self.include_interpolations:
            return True  # Skip interpolations, not an error
        
        # Check connection
        if not self.ws_client.connected:
            logger.warning("Cannot push frame: not connected to VPS")
            return False
        
        start_time = time.time()
        
        try:
            # Encode frame
            frame_bytes = self._encode_frame(image)
            encode_time = time.time() - start_time
            
            # Push via WebSocket
            push_start = time.time()
            success = await self.ws_client.send_frame(frame_bytes)
            push_time = time.time() - push_start
            
            total_time = time.time() - start_time
            
            if success:
                # Update statistics
                self.frames_pushed += 1
                self.bytes_pushed += len(frame_bytes)
                
                if is_keyframe:
                    self.keyframes_pushed += 1
                else:
                    self.interpolations_pushed += 1
                
                # Track timing
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
        return {
            "frames_pushed": self.frames_pushed,
            "keyframes_pushed": self.keyframes_pushed,
            "interpolations_pushed": self.interpolations_pushed,
            "bytes_pushed": self.bytes_pushed,
            "bytes_pushed_mb": round(self.bytes_pushed / 1024 / 1024, 2),
            "average_push_time_ms": round(self.average_push_time_ms, 2),
            "average_frame_size_kb": round(self.average_frame_size_kb, 2),
            "format": self.format,
            "quality": self.quality,
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

