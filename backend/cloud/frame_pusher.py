"""
Cloud Frame Pusher

Encodes frames to H.264 and pushes them to the VPS via WebSocket.
Handles both keyframes and interpolation frames based on configuration.

H.264 encoding provides temporal compression — P-frames encode only
the delta between frames, which is tiny for dream-like content where
adjacent frames are nearly identical.

Frame Message Format (v2):
  0x01 | metadata_len (4 bytes BE) | JSON metadata | H.264 NAL data

Metadata JSON:
  {
    "fn": frame_number,      // Sequential frame number
    "kf": keyframe_number,   // Current keyframe number
    "vk": true/false,        // Video keyframe (H.264 I-frame)
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
from typing import Optional, Callable, Union

import numpy as np
from PIL import Image

from .websocket_client import VPSWebSocketClient, ConnectionState
from .video_encoder import VideoStreamEncoder
from utils.perf_stats import get_perf_stats

logger = logging.getLogger(__name__)

# Type alias for push callbacks
PushCallback = Callable[[], None]

# Priority levels for frame queueing
PRIORITY_KEYFRAME = 10
PRIORITY_INTERPOLATION = 1


class CloudFramePusher:
    """
    Encodes frames to H.264 and pushes to VPS via WebSocket.

    Uses VideoStreamEncoder for H.264 encoding with temporal compression.
    P-frames for dream-like content (slow drift) are very small (~5-15KB)
    compared to independent JPEG/WebP encoding (~50-75KB).

    Self-healing features:
    - Continues operating during disconnections (frames queued by WebSocket client)
    - Keyframes get higher queue priority than interpolations
    - Connection status logging for debugging
    """

    def __init__(self, websocket_client: VPSWebSocketClient, config: dict):
        """
        Initialize frame pusher with H.264 encoding.

        Args:
            websocket_client: Connected VPS WebSocket client
            config: Full configuration dict (needs cloud.frame_push and
                    generation.resolution / generation.hybrid for encoder setup)
        """
        self.ws_client = websocket_client

        # Extract cloud config for frame push settings
        cloud_config = config.get('cloud', {})
        frame_config = cloud_config.get('frame_push', {})
        self.include_interpolations = frame_config.get('include_interpolations', True)

        # Set up H.264 encoder
        resolution = config.get('generation', {}).get('resolution', [1024, 512])
        hybrid = config.get('generation', {}).get('hybrid', {})
        target_fps = hybrid.get('target_interpolation_fps', 17.0)

        h264_config = frame_config.get('h264', {})
        self._encoder = VideoStreamEncoder(
            width=resolution[0],
            height=resolution[1],
            fps=target_fps,
            keyframe_interval=h264_config.get('keyframe_interval', 34),
            crf=h264_config.get('crf', 23),
            max_bitrate=h264_config.get('max_bitrate', '2M'),
            preset=h264_config.get('preset', 'ultrafast'),
        )

        # Optional callback invoked after each successful push (for watchdog heartbeat, etc.)
        self._on_push_callback: Optional[PushCallback] = None

        # Statistics
        self.frames_pushed = 0
        self.keyframes_pushed = 0
        self.interpolations_pushed = 0
        self.frames_queued = 0  # Frames queued due to disconnect
        self.bytes_pushed = 0
        self.push_times: list[float] = []  # Last N push times for avg calculation
        self.max_timing_samples = 100
        self.video_keyframes_encoded = 0  # H.264 I-frames produced

        # Track disconnection state for logging
        self._last_connection_state = ConnectionState.DISCONNECTED
        self._disconnect_logged = False

        logger.info(
            f"CloudFramePusher initialized: H.264 {resolution[0]}x{resolution[1]} @ {target_fps}fps"
        )

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
        """Log connection state changes for debugging."""
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
        image: Union[Image.Image, np.ndarray],
        is_keyframe: bool = False,
        frame_number: int = 0,
        keyframe_number: int = 0,
        prompt: Optional[str] = None,
    ) -> bool:
        """
        Encode PIL image to H.264 and push to VPS.

        Args:
            image: PIL Image to push
            is_keyframe: Whether this is a generation keyframe (vs interpolation)
            frame_number: Sequential frame number (server-authoritative)
            keyframe_number: Current generation keyframe number
            prompt: Prompt text for this frame's keyframe (optional)

        Returns:
            True if pushed successfully (or queued during disconnect)

        Message format (v2):
            0x01 | metadata_len (4 bytes BE) | JSON metadata | H.264 NAL data

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
            # H.264 encode — run in executor to avoid blocking the display loop.
            # x264 occasionally spikes to 150-300ms on complex frames; without
            # the executor, this blocks the entire display pipeline.
            loop = asyncio.get_event_loop()
            nal_data, is_video_keyframe = await loop.run_in_executor(
                None, self._encoder.encode_frame, image
            )
            encode_time = time.time() - start_time

            if not nal_data:
                return True  # Encoder buffering (shouldn't happen with zerolatency)

            if is_video_keyframe:
                self.video_keyframes_encoded += 1

            # Build metadata JSON
            metadata: dict = {
                "fn": frame_number,
                "kf": keyframe_number,
                "vk": is_video_keyframe,
            }
            if prompt:
                metadata["p"] = prompt

            metadata_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')

            # Capture connection state BEFORE the send to determine if it will be queued
            was_connected_before = self.ws_client.connected

            # Push via WebSocket (with metadata and priority)
            push_start = time.time()
            success = await self.ws_client.send_frame_with_metadata(
                nal_data, metadata_bytes, priority=priority
            )
            push_time = time.time() - push_start

            total_time = time.time() - start_time

            if success:
                was_queued = not was_connected_before

                # Update statistics
                self.frames_pushed += 1
                self.bytes_pushed += len(nal_data) + len(metadata_bytes) + 4

                if is_keyframe:
                    self.keyframes_pushed += 1
                else:
                    self.interpolations_pushed += 1

                # Invoke callback on ANY successful frame processing (sent OR queued).
                # Critical for watchdog heartbeat — the system is "alive" if encoding,
                # even if VPS is disconnected and frames queue.
                if self._on_push_callback:
                    try:
                        self._on_push_callback()
                    except Exception as e:
                        logger.warning(f"Push callback failed: {e}")

                if was_queued:
                    self.frames_queued += 1
                    if self.frames_queued % 10 == 1:
                        logger.info(
                            f"Frame {frame_number} queued "
                            f"(total queued: {self.frames_queued}, "
                            f"queue size: {self.ws_client.queue_size})"
                        )
                else:
                    self._record_timing(total_time)

                    logger.debug(
                        f"Pushed frame {frame_number}: {len(nal_data)/1024:.1f}KB "
                        f"({'I' if is_video_keyframe else 'P'}-frame, "
                        f"encode: {encode_time*1000:.1f}ms, "
                        f"push: {push_time*1000:.1f}ms)"
                    )

                    if total_time > 0.1:  # > 100ms is concerning
                        logger.info(
                            f"[PERF] Slow frame push {frame_number}: {total_time*1000:.1f}ms total "
                            f"(encode={encode_time*1000:.1f}ms, network={push_time*1000:.1f}ms, "
                            f"size={len(nal_data)/1024:.1f}KB, "
                            f"type={'I' if is_video_keyframe else 'P'})"
                        )

                    get_perf_stats().record_frame_push(total_time)

            return success

        except Exception as e:
            logger.error(f"Failed to push frame: {e}")
            return False

    async def close(self) -> None:
        """Flush encoder and clean up resources."""
        try:
            final_data = self._encoder.flush()
            if final_data:
                # Best-effort send of final NAL units
                try:
                    from .websocket_client import MessageType
                    await self.ws_client.send_raw(
                        bytes([MessageType.FRAME]) + final_data
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error flushing encoder: {e}")
        finally:
            self._encoder.close()

    def _record_timing(self, time_seconds: float) -> None:
        """Record a timing sample."""
        self.push_times.append(time_seconds)
        if len(self.push_times) > self.max_timing_samples:
            self.push_times.pop(0)

    @property
    def average_push_time_ms(self) -> float:
        """Average push time in milliseconds."""
        if not self.push_times:
            return 0.0
        return sum(self.push_times) / len(self.push_times) * 1000

    @property
    def average_frame_size_kb(self) -> float:
        """Average frame size in KB."""
        if self.frames_pushed == 0:
            return 0.0
        return self.bytes_pushed / self.frames_pushed / 1024

    def get_stats(self) -> dict:
        """Get pusher statistics."""
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
            "format": "h264",
            "video_keyframes_encoded": self.video_keyframes_encoded,
            "encoder_frame_count": self._encoder.frame_count,
            "connection": {
                "connected": ws_stats["connected"],
                "state": ws_stats["state"],
                "queue_size": ws_stats["queue_size"],
                "total_reconnects": ws_stats["total_reconnects"],
                "messages_dropped": ws_stats["messages_dropped"],
            }
        }
