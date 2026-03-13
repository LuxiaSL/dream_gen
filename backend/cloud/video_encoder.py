"""
H.264 Video Stream Encoder

Encodes PIL images into H.264 Annex B NAL units for WebSocket streaming.
Uses PyAV (FFmpeg wrapper) for encoding. Outputs raw byte stream with
start codes (0x00000001) suitable for both WebCodecs VideoDecoder
and client-side fMP4 muxing.

Design decisions:
- Baseline profile: maximum browser compatibility (WebCodecs + MSE)
- zerolatency tune: no lookahead, no frame reordering, immediate output
- No B-frames: every frame is I or P, simplifies client decode
- repeat-headers: SPS/PPS included with every I-frame (self-contained keyframes)
- Annex B format: start codes between NAL units
"""

import logging
from fractions import Fraction
from typing import Optional

import av
from PIL import Image

logger = logging.getLogger(__name__)


class VideoStreamEncoder:
    """
    Encodes PIL images into H.264 NAL units for live streaming.

    Usage:
        encoder = VideoStreamEncoder(1024, 512, fps=17.0)

        for image in frame_generator():
            nal_data, is_keyframe = encoder.encode_frame(image)
            if nal_data:
                await pusher.push_video_frame(nal_data, is_keyframe)

        encoder.flush()
        encoder.close()

    Output: raw H.264 Annex B byte stream. Each encode_frame() returns
    one or more NAL units that can be fed directly to VideoDecoder or
    wrapped in fMP4 for MSE.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 17.0,
        keyframe_interval: int = 34,
        crf: int = 23,
        max_bitrate: str = "2M",
        preset: str = "ultrafast",
    ):
        """
        Args:
            width: Frame width (must be even for H.264)
            height: Frame height (must be even)
            fps: Target FPS (should match actual frame rate)
            keyframe_interval: Frames between I-frames (default 34 = ~2s at 17fps)
            crf: Constant Rate Factor (18=high quality, 23=medium, 28=low)
            max_bitrate: Bandwidth cap (e.g., "2M", "3M")
            preset: x264 preset — "ultrafast" for live streaming
        """
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(f"H.264 requires even dimensions, got {width}x{height}")

        self._closed: bool = True  # Set early in case __init__ fails partway
        self.width = width
        self.height = height
        self.fps = fps
        self.keyframe_interval = keyframe_interval
        self._frame_count: int = 0
        # Create codec context (no container — raw NAL output)
        codec = av.codec.Codec("libx264", "w")
        self._ctx = av.codec.CodecContext.create(codec)

        self._ctx.width = width
        self._ctx.height = height
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = Fraction(1, int(fps * 1000))
        self._ctx.framerate = Fraction(int(fps * 1000), 1000)
        self._ctx.gop_size = keyframe_interval
        self._ctx.max_b_frames = 0  # No B-frames for lower latency

        self._ctx.options = {
            "preset": preset,
            "tune": "zerolatency",      # No lookahead, immediate output
            "crf": str(crf),
            "maxrate": max_bitrate,
            "bufsize": max_bitrate,      # VBV buffer = 1 second
            "repeat-headers": "1",       # SPS/PPS with every I-frame
            "annexb": "1",               # Start codes, not length prefixes
            "profile": "baseline",       # Max WebCodecs/MSE compatibility
            "level": "3.1",              # Supports 1280x720@30fps
        }

        self._ctx.open()
        self._closed = False  # Successfully initialized
        logger.info(
            f"VideoStreamEncoder: {width}x{height} @ {fps}fps, "
            f"CRF={crf}, I-frame every {keyframe_interval} frames "
            f"({keyframe_interval / fps:.1f}s), preset={preset}"
        )

    def encode_frame(self, image: Image.Image) -> tuple[bytes, bool]:
        """
        Encode a PIL image to H.264 NAL units.

        Args:
            image: PIL Image (any mode — will be converted to RGB)

        Returns:
            (nal_bytes, is_keyframe) — nal_bytes may be empty if encoder
            is buffering (shouldn't happen with zerolatency, but handled).

        Raises:
            RuntimeError: If encoder has been closed.
        """
        if self._closed:
            raise RuntimeError("Encoder is closed")

        if image.mode != "RGB":
            image = image.convert("RGB")

        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.LANCZOS)

        frame = av.VideoFrame.from_image(image)
        frame = frame.reformat(format="yuv420p")
        frame.pts = int(self._frame_count * (1000 / self.fps))
        self._frame_count += 1

        packets = self._ctx.encode(frame)
        if not packets:
            return b"", False

        nal_data = b""
        is_keyframe = False
        for packet in packets:
            nal_data += bytes(packet)
            if packet.is_keyframe:
                is_keyframe = True

        return nal_data, is_keyframe

    def flush(self) -> bytes:
        """Flush encoder buffer on shutdown. Returns remaining NAL units."""
        if self._closed:
            return b""
        packets = self._ctx.encode(None)
        return b"".join(bytes(p) for p in packets)

    def close(self) -> None:
        """Close encoder and free resources."""
        if not self._closed:
            self._closed = True
            try:
                self._ctx.close()
            except Exception:
                pass
            logger.info(f"VideoStreamEncoder closed after {self._frame_count} frames")

    def __del__(self) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        """Number of frames encoded so far."""
        return self._frame_count
