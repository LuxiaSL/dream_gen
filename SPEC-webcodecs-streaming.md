# Spec: H.264 Video Streaming for Dream Window

## Status
- **Pacing Tuning**: DONE — applied in `config.b200.yaml`
- **This spec**: Ready for implementation

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Encoding format | H.264 only | Single code path, no JPEG/WebP |
| VPS pacing | Remove `FramePlaybackQueue` | Pass-through relay; decoder handles buffering |
| Primary browser decode | WebCodecs `VideoDecoder` | Chrome 94+, Edge 94+, hardware-accelerated |
| Safari fallback | MSE via client-side fMP4 muxer | Safari 17.1+ supports MSE; mux NAL→fMP4 in JS |
| Unsupported browsers | "Use a modern browser" message | No image-based fallback |
| Canvas blend | Remove entirely | H.264 temporal compression is inherently smooth |
| `/api/dreams/current` | Remove | No standalone image generation |
| OG meta image | Static placeholder | Not worth decoding I-frames just for previews |
| External API (programmatic) | Raw NAL over WebSocket | Document wire protocol for custom consumers |
| External API (casual) | HTTP MPEG-TS endpoint | `/api/dreams/stream` — playable in VLC/mpv/ffplay |

---

## Architecture

```
GPU (node1)                          VPS (aetherawi.red)                   Browser
─────────────────────────────────────────────────────────────────────────────────
PIL Image                            DreamWebSocketHub                     DreamViewer
  │                                    │                                     │
  ├─ VideoStreamEncoder.encode()       │                                     │
  │  PIL → YUV420 → H.264 NAL         │                                     │
  │  I-frame: ~30-50KB                 │                                     │
  │  P-frame: ~5-15KB                  │                                     │
  │                                    │                                     │
  ├─ WebSocket send                    │                                     │
  │  0x01 | meta_len | JSON | NAL ──► handle_gpu_message()                  │
  │                                    │                                     │
  │                                    ├─ Parse metadata + NAL data          │
  │                                    ├─ Cache I-frame for late joiners     │
  │                                    ├─ Update stats                       │
  │                                    │                                     │
  │                                    ├─ Broadcast to all viewers:          │
  │                                    │  JSON frame_meta ─────────────────► handleFrameMetaMessage()
  │                                    │  0x01 | NAL data ─────────────────► handleBinaryMessage()
  │                                    │                                     │
  │                                    │                              ┌──── WebCodecs path (Chrome/Edge):
  │                                    │                              │      EncodedVideoChunk → VideoDecoder
  │                                    │                              │      → VideoFrame → canvas.drawImage()
  │                                    │                              │
  │                                    │                              └──── MSE path (Safari):
  │                                    │                                     NAL → fMP4 muxer (JS)
  │                                    │                                     → MediaSource → <video> element
  │                                    │
  │                                    ├─ MPEG-TS endpoint (external):
  │                                    │  GET /api/dreams/stream ──────────► VLC / mpv / ffplay
  │                                    │  Wraps NAL units in MPEG-TS
  │                                    │  (chunked transfer encoding)
```

**Note on message type byte**: We keep `0x01` as the frame message type (not introducing `0x05`). The frame format changes from image bytes to H.264 NAL units, but the message type stays the same — this is a format change, not a new message category. The metadata `"vk"` flag distinguishes I-frames from P-frames.

---

## 1. GPU Side

### 1a. New file: `backend/cloud/video_encoder.py`

**Dependency**: `pip install av` (PyAV — thin FFmpeg wrapper, common in ML environments)

```python
"""
H.264 Video Stream Encoder

Encodes PIL images into H.264 Annex B NAL units for WebSocket streaming.
Uses PyAV (FFmpeg wrapper) for encoding. Outputs raw byte stream with
start codes (0x00000001) — suitable for both WebCodecs VideoDecoder
and client-side fMP4 muxing.

Design decisions:
- Baseline profile: maximum browser compatibility
- zerolatency tune: no lookahead, no frame reordering, immediate output
- No B-frames: every frame is I or P, simplifies client decode
- repeat-headers: SPS/PPS included with every I-frame (self-contained keyframes)
- Annex B format: start codes between NAL units
"""

import logging
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

        self.width = width
        self.height = height
        self.fps = fps
        self.keyframe_interval = keyframe_interval
        self._frame_count: int = 0
        self._closed: bool = False

        # Create codec context (no container — raw NAL output)
        codec = av.codec.Codec("libx264", "w")
        self._ctx = av.codec.CodecContext.create(codec)

        self._ctx.width = width
        self._ctx.height = height
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = av.Fraction(1, int(fps * 1000))
        self._ctx.framerate = av.Fraction(int(fps * 1000), 1000)
        self._ctx.gop_size = keyframe_interval
        self._ctx.max_b_frames = 0  # No B-frames — lower latency

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
            is buffering (shouldn't happen with zerolatency, but handled)

        Raises:
            RuntimeError: If encoder has been closed
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
            self._ctx.close()
            logger.info(f"VideoStreamEncoder closed after {self._frame_count} frames")

    def __del__(self) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count
```

### 1b. Changes to `backend/cloud/frame_pusher.py`

The entire class simplifies — no more image encoding, no format switching.

**Replace `_encode_frame()` and image-related logic entirely.** The pusher now only does: receive PIL image → encode H.264 → send NAL units.

Key changes:

```python
# Remove: io import, PIL Image.save logic, _buffer, format/quality config
# Add:
from .video_encoder import VideoStreamEncoder

class CloudFramePusher:
    """
    Encodes frames to H.264 and pushes to VPS via WebSocket.
    """

    def __init__(self, websocket_client: VPSWebSocketClient, config: dict):
        self.ws_client = websocket_client

        # H.264 encoder configuration
        frame_config = config.get('cloud', {}).get('frame_push', {})
        self.include_interpolations = frame_config.get('include_interpolations', True)

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

        # Push callback, stats, connection tracking (unchanged)
        self._on_push_callback: Optional[PushCallback] = None
        self.frames_pushed = 0
        self.keyframes_pushed = 0
        self.interpolations_pushed = 0
        self.frames_queued = 0
        self.bytes_pushed = 0
        self.push_times = []
        self.max_timing_samples = 100
        self._last_connection_state = ConnectionState.DISCONNECTED
        self._disconnect_logged = False

    async def push_frame(
        self,
        image: Image.Image,
        is_keyframe: bool = False,
        frame_number: int = 0,
        keyframe_number: int = 0,
        prompt: Optional[str] = None,
    ) -> bool:
        """Encode PIL image to H.264 and push to VPS."""
        if not is_keyframe and not self.include_interpolations:
            return True

        self._check_connection_state()
        priority = PRIORITY_KEYFRAME if is_keyframe else PRIORITY_INTERPOLATION
        start_time = time.time()

        try:
            # H.264 encode
            nal_data, is_video_keyframe = self._encoder.encode_frame(image)
            encode_time = time.time() - start_time

            if not nal_data:
                return True  # Encoder buffering

            # Build metadata
            metadata = {
                "fn": frame_number,
                "kf": keyframe_number,
                "vk": is_video_keyframe,
            }
            if prompt:
                metadata["p"] = prompt

            metadata_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')

            was_connected_before = self.ws_client.connected
            push_start = time.time()

            success = await self.ws_client.send_frame_with_metadata(
                nal_data, metadata_bytes, priority=priority
            )

            push_time = time.time() - push_start
            total_time = time.time() - start_time

            if success:
                was_queued = not was_connected_before
                self.frames_pushed += 1
                self.bytes_pushed += len(nal_data) + len(metadata_bytes) + 4

                if is_keyframe:
                    self.keyframes_pushed += 1
                else:
                    self.interpolations_pushed += 1

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
                            f"(total queued: {self.frames_queued})"
                        )
                else:
                    self._record_timing(total_time)
                    logger.debug(
                        f"Pushed frame {frame_number}: {len(nal_data)/1024:.1f}KB "
                        f"({'I' if is_video_keyframe else 'P'}-frame, "
                        f"encode: {encode_time*1000:.1f}ms, "
                        f"push: {push_time*1000:.1f}ms)"
                    )
                    get_perf_stats().record_frame_push(total_time)

            return success

        except Exception as e:
            logger.error(f"Failed to push frame: {e}")
            return False

    async def close(self) -> None:
        """Flush encoder and clean up."""
        final_data = self._encoder.flush()
        if final_data:
            try:
                # Best-effort send of final NAL units
                await self.ws_client.send_raw(
                    bytes([MessageType.FRAME])
                    + final_data
                )
            except Exception:
                pass
        self._encoder.close()

    # _check_connection_state, set_push_callback, _record_timing,
    # average_push_time_ms, average_frame_size_kb, get_stats
    # all remain unchanged
```

**Removed:**
- `_encode_frame()` method (image encoding)
- `_buffer` (BytesIO reuse)
- `format` / `quality` config
- `encode_frame_webp()` utility function

### 1c. Changes to `backend/cloud/websocket_client.py`

Minimal — remove the `VIDEO_FRAME` message type idea. We reuse `FRAME = 0x01`. The message format stays the same (type byte + metadata_len + JSON + payload), only the payload content changes from image bytes to NAL bytes.

**No changes needed to `websocket_client.py`** — `send_frame_with_metadata()` already sends:
```
0x01 | metadata_len (4B BE) | JSON metadata | payload bytes
```
It doesn't care whether `payload bytes` is JPEG or H.264. The metadata `"vk"` flag is the only new information, and that's in the JSON.

### 1d. Config changes: `backend/config.b200.yaml`

```yaml
cloud:
  frame_push:
    enabled: true
    include_interpolations: true

    # H.264 encoding
    h264:
      keyframe_interval: 34   # I-frame every ~2s (17fps * 2)
      crf: 23                 # Quality: 18=high, 23=medium, 28=low
      max_bitrate: "2M"       # Bandwidth cap for spikes during mutations
      preset: "ultrafast"     # CPU usage — ultrafast is fine for 1024x512
```

**Removed:** `format` and `quality` keys (no longer applicable).

---

## 2. VPS Side

### 2a. Remove `FramePlaybackQueue`

**Delete file**: `core/aethera/dreams/frame_playback.py`

**Remove from `__init__.py`** (if re-exported):
```python
# Remove: from .frame_playback import FramePlaybackQueue
```

### 2b. Simplify `FrameCache`

The `FrameCache` no longer stores frame image data (there are no standalone images). It becomes a **stats tracker** only — FPS calculation, frame counts, timestamps.

**File**: `core/aethera/dreams/frame_cache.py`

Rename to something clearer or simplify in place. The `CachedFrame` dataclass and deque of frames can be removed. Keep:
- `total_frames_received`
- `total_bytes_received`
- `_frame_timestamps` deque (for rolling FPS)
- `_session_start_time` / `_session_frames`
- `get_stats()` method
- `reset_session()` method

**But** — we still need to cache the **last I-frame** for late-joining viewers. This is a single `bytes` field, not a rolling buffer.

Simplified `FrameCache`:

```python
@dataclass
class StreamStats:
    """Lightweight stats tracker for H.264 stream (no frame storage)."""
    total_frames_received: int = 0
    total_bytes_received: int = 0

    # Rolling FPS
    _fps_window_seconds: float = 30.0
    _frame_timestamps: deque = field(default_factory=deque)
    _session_start_time: Optional[float] = None
    _session_frames: int = 0

    # Last I-frame for late joiners
    last_keyframe_nal: Optional[bytes] = None
    last_keyframe_number: int = 0
    last_keyframe_frame_number: int = 0
    last_keyframe_meta: Optional[dict] = None

    def record_frame(self, size_bytes: int, is_keyframe: bool,
                     nal_data: Optional[bytes] = None,
                     frame_number: int = 0,
                     keyframe_number: int = 0,
                     meta: Optional[dict] = None) -> None:
        self.total_frames_received += 1
        self.total_bytes_received += size_bytes
        self._session_frames += 1

        now = time.time()
        self._frame_timestamps.append(now)
        if self._session_start_time is None:
            self._session_start_time = now

        # Prune old timestamps
        cutoff = now - self._fps_window_seconds
        while self._frame_timestamps and self._frame_timestamps[0] < cutoff:
            self._frame_timestamps.popleft()

        # Cache I-frame
        if is_keyframe and nal_data is not None:
            self.last_keyframe_nal = nal_data
            self.last_keyframe_number = keyframe_number
            self.last_keyframe_frame_number = frame_number
            self.last_keyframe_meta = meta

    def reset_session(self) -> None:
        self._session_start_time = None
        self._session_frames = 0
        self._frame_timestamps.clear()
        self.last_keyframe_nal = None
        self.last_keyframe_meta = None

    def get_stats(self) -> dict:
        now = time.time()
        # Rolling FPS
        if len(self._frame_timestamps) >= 2:
            span = self._frame_timestamps[-1] - self._frame_timestamps[0]
            rolling_fps = (len(self._frame_timestamps) - 1) / span if span > 0 else 0.0
        else:
            rolling_fps = 0.0

        # Session FPS
        if self._session_start_time and self._session_frames > 0:
            elapsed = now - self._session_start_time
            session_fps = self._session_frames / elapsed if elapsed > 0 else 0.0
        else:
            session_fps = 0.0

        return {
            "total_frames_received": self.total_frames_received,
            "total_bytes_received": self.total_bytes_received,
            "average_fps": round(rolling_fps, 2),
            "session_fps": round(session_fps, 2),
            "has_keyframe": self.last_keyframe_nal is not None,
            "current_frame_number": self.last_keyframe_frame_number,
            "current_keyframe_number": self.last_keyframe_number,
        }
```

### 2c. Changes to `DreamWebSocketHub` (`core/aethera/dreams/websocket.py`)

**Major simplification** — remove playback queue, remove image broadcasting, add pass-through video relay.

```python
# Remove imports:
# from .frame_playback import FramePlaybackQueue

# Remove from __init__:
# self._playback_queue = ...
# self._playback_task = ...

# Remove methods:
# _on_frame_displayed
# _broadcast_frame (legacy)

# frame_cache becomes StreamStats (or renamed FrameCache with simplified internals)
```

**`__init__` changes:**

```python
def __init__(self, stream_stats, presence_tracker, gpu_manager=None):
    self.stats = stream_stats          # Was self.frame_cache
    self.presence = presence_tracker
    self.gpu_manager = gpu_manager

    self._viewers: Set[WebSocket] = set()
    self._gpu_websocket: Optional[WebSocket] = None
    self._lock = asyncio.Lock()

    self._status = "idle"
    self._status_message = "Waiting for connection..."
    self._last_frame_time: float = 0
    self._next_frame_number: int = 1
    self._current_prompt: Optional[str] = None
```

**`connect_gpu` simplifies** (no playback queue to start):

```python
async def connect_gpu(self, websocket: WebSocket) -> None:
    await websocket.accept()

    replacing = self._gpu_websocket is not None
    if replacing:
        logger.warning("Replacing stale GPU connection")
        old_ws = self._gpu_websocket
        self._gpu_websocket = None
        try:
            await old_ws.close(code=4001, reason="Replaced")
        except Exception:
            pass
        self.presence.set_gpu_running(False)

    self._gpu_websocket = websocket
    self.presence.set_gpu_running(True)

    if not replacing:
        self.stats.reset_session()
        self._next_frame_number = 1

    if self.gpu_manager:
        self.gpu_manager.on_gpu_connected()

    logger.info("GPU connected")
    await self._send_saved_state_to_gpu(websocket)
    await self.broadcast_status("ready", "Dreams flowing...")
```

**`disconnect_gpu` simplifies** (no playback queue to stop):

```python
async def disconnect_gpu(self) -> None:
    self._gpu_websocket = None
    self.presence.set_gpu_running(False)

    if self.gpu_manager:
        self.gpu_manager.on_gpu_disconnected()

    logger.info("GPU disconnected")
    await self.broadcast_status("idle", "Dream machine sleeping...")
```

**`_handle_gpu_frame` becomes the video pass-through:**

```python
async def _handle_gpu_frame(self, payload: bytes) -> None:
    """
    Handle H.264 frame from GPU — parse metadata, update stats,
    pass through to all viewers immediately.

    Frame format: metadata_len (4B BE) | JSON metadata | H.264 NAL data
    """
    self._last_frame_time = time.time()

    if self.gpu_manager:
        self.gpu_manager.on_frame_received()

    # Parse metadata
    frame_number = self._next_frame_number
    keyframe_number = 0
    prompt = None
    is_video_keyframe = False
    nal_data = payload

    if len(payload) > 4:
        try:
            metadata_len = int.from_bytes(payload[:4], 'big')
            if 0 < metadata_len < len(payload) - 4:
                metadata_bytes = payload[4:4 + metadata_len]
                nal_data = payload[4 + metadata_len:]

                metadata = json.loads(metadata_bytes.decode('utf-8'))
                frame_number = metadata.get('fn', frame_number)
                keyframe_number = metadata.get('kf', 0)
                is_video_keyframe = metadata.get('vk', False)

                if 'p' in metadata and isinstance(metadata['p'], str):
                    self._current_prompt = metadata['p']
                    prompt = metadata['p']
        except Exception as e:
            logger.debug(f"Metadata parse failed: {e}")

    # Update frame counter
    if frame_number == self._next_frame_number:
        self._next_frame_number += 1
    else:
        self._next_frame_number = frame_number + 1

    # Update stats + cache I-frame
    meta_for_cache = {"fn": frame_number, "kf": keyframe_number}
    if prompt:
        meta_for_cache["p"] = prompt

    self.stats.record_frame(
        size_bytes=len(nal_data),
        is_keyframe=is_video_keyframe,
        nal_data=nal_data if is_video_keyframe else None,
        frame_number=frame_number,
        keyframe_number=keyframe_number,
        meta=meta_for_cache if is_video_keyframe else None,
    )

    # Feed to MPEG-TS muxer (for /api/dreams/stream endpoint)
    if hasattr(self, '_mpegts_muxer') and self._mpegts_muxer:
        self._mpegts_muxer.feed_nal(nal_data, is_video_keyframe)

    # Pass through to all viewers
    await self._broadcast_video_frame(
        nal_data, frame_number, keyframe_number, prompt, is_video_keyframe
    )
```

**New `_broadcast_video_frame`:**

```python
async def _broadcast_video_frame(
    self,
    nal_data: bytes,
    frame_number: int,
    keyframe_number: int,
    prompt: Optional[str],
    is_video_keyframe: bool,
) -> None:
    """Broadcast H.264 frame to all connected viewers."""
    if not self._viewers:
        return

    meta_msg = {
        "type": "frame_meta",
        "fn": frame_number,
        "kf": keyframe_number,
        "vk": is_video_keyframe,
    }
    if prompt:
        meta_msg["p"] = prompt

    frame_message = bytes([MSG_FRAME]) + nal_data
    dead_viewers = set()

    async with self._lock:
        viewers = set(self._viewers)

    for viewer in viewers:
        try:
            await asyncio.wait_for(viewer.send_json(meta_msg), timeout=5.0)
            await asyncio.wait_for(viewer.send_bytes(frame_message), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            dead_viewers.add(viewer)

    if dead_viewers:
        async with self._lock:
            self._viewers -= dead_viewers
        for viewer in dead_viewers:
            await self.presence.on_viewer_disconnect(viewer)
```

**Late-joiner I-frame in `connect_viewer`:**

```python
async def connect_viewer(self, websocket: WebSocket) -> None:
    await websocket.accept()

    async with self._lock:
        self._viewers.add(websocket)

    await self.presence.on_viewer_connect(websocket)
    await self._send_status_to_viewer(websocket)

    # Send cached I-frame so viewer can start decoding immediately
    if self.stats.last_keyframe_nal:
        try:
            meta_msg = {
                "type": "frame_meta",
                **(self.stats.last_keyframe_meta or {}),
                "vk": True,
            }
            await asyncio.wait_for(websocket.send_json(meta_msg), timeout=5.0)
            await asyncio.wait_for(
                websocket.send_bytes(
                    bytes([MSG_FRAME]) + self.stats.last_keyframe_nal
                ),
                timeout=5.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Failed to send initial I-frame: {e}")
```

### 2d. MPEG-TS HTTP Endpoint

**New file**: `core/aethera/dreams/mpegts_muxer.py`

This wraps incoming H.264 NAL units into MPEG-TS packets for HTTP streaming. External players (VLC, mpv, ffplay) connect to `GET /api/dreams/stream` and receive a continuous MPEG-TS byte stream via chunked transfer encoding.

```python
"""
MPEG-TS Live Muxer

Wraps H.264 NAL units into MPEG-TS packets for HTTP streaming.
External players connect to GET /api/dreams/stream and receive
continuous MPEG-TS via chunked transfer encoding.

Uses PyAV to mux H.264 into MPEG-TS format. Each NAL unit from
the WebSocket is remuxed (not re-encoded) into TS packets and
pushed to all connected HTTP consumers.

Design:
- Single muxer instance fed by DreamWebSocketHub
- Multiple HTTP consumers read from a shared ring buffer
- Each consumer tracks its own read position
- I-frame: new consumers start from the latest I-frame
"""

import asyncio
import logging
import time
import io
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TSSegment:
    """A chunk of MPEG-TS data ready for HTTP consumers."""
    data: bytes
    frame_number: int
    is_keyframe: bool
    timestamp: float = field(default_factory=time.time)


class MpegTSMuxer:
    """
    Muxes H.264 NAL units into MPEG-TS format for HTTP streaming.

    Fed by DreamWebSocketHub.handle_gpu_message().
    Read by /api/dreams/stream HTTP endpoint.
    """

    def __init__(self, width: int = 1024, height: int = 512, fps: float = 17.0):
        self.width = width
        self.height = height
        self.fps = fps

        # Ring buffer of TS segments
        self._segments: deque[TSSegment] = deque(maxlen=300)  # ~17s at 17fps
        self._lock = asyncio.Lock()

        # Consumers waiting for new data
        self._waiters: list[asyncio.Event] = []

        # PyAV muxer writing to memory buffer
        self._output = io.BytesIO()
        self._container = av.open(self._output, mode='w', format='mpegts')
        self._stream = self._container.add_stream('h264', rate=int(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = 'yuv420p'
        self._stream.codec_context.time_base = av.Fraction(1, int(fps * 1000))

        self._frame_count = 0
        logger.info(f"MpegTSMuxer initialized: {width}x{height} @ {fps}fps")

    def feed_nal(self, nal_data: bytes, is_keyframe: bool) -> None:
        """
        Feed H.264 NAL units from GPU. Muxes into MPEG-TS and
        appends to ring buffer.

        Called synchronously from the WebSocket message handler.
        """
        # Create a packet from raw NAL data
        packet = av.Packet(nal_data)
        packet.stream = self._stream
        packet.pts = int(self._frame_count * (1000 / self.fps))
        packet.dts = packet.pts
        packet.is_keyframe = is_keyframe
        self._frame_count += 1

        # Reset output buffer
        self._output.seek(0)
        self._output.truncate()

        # Mux packet to MPEG-TS
        self._container.mux(packet)

        ts_data = self._output.getvalue()
        if ts_data:
            segment = TSSegment(
                data=ts_data,
                frame_number=self._frame_count,
                is_keyframe=is_keyframe,
            )
            self._segments.append(segment)

            # Wake up any waiting consumers
            for event in self._waiters:
                event.set()

    async def consume(self):
        """
        Async generator that yields MPEG-TS bytes for an HTTP consumer.

        Starts from the latest I-frame (if available) for fast playback start,
        then yields new segments as they arrive.
        """
        event = asyncio.Event()
        self._waiters.append(event)

        try:
            # Find latest I-frame to start from
            start_idx = 0
            segments = list(self._segments)
            for i in range(len(segments) - 1, -1, -1):
                if segments[i].is_keyframe:
                    start_idx = i
                    break

            # Yield backlog from I-frame
            for seg in segments[start_idx:]:
                yield seg.data

            # Track position
            last_frame = segments[-1].frame_number if segments else 0

            # Yield new segments as they arrive
            while True:
                event.clear()
                await event.wait()

                # Get new segments since last yield
                for seg in list(self._segments):
                    if seg.frame_number > last_frame:
                        yield seg.data
                        last_frame = seg.frame_number

        finally:
            self._waiters.remove(event)

    def close(self) -> None:
        """Close the muxer."""
        try:
            self._container.close()
        except Exception:
            pass
```

**New route in `core/aethera/api/dreams.py`:**

```python
from starlette.responses import StreamingResponse

@router.get("/api/dreams/stream")
async def dreams_video_stream(request: Request):
    """
    Live H.264 video stream in MPEG-TS container.

    Playable directly in:
    - VLC: vlc https://aetherawi.red/api/dreams/stream
    - mpv: mpv https://aetherawi.red/api/dreams/stream
    - ffplay: ffplay https://aetherawi.red/api/dreams/stream

    Uses chunked transfer encoding for continuous delivery.
    """
    hub = get_hub()
    hub.presence.on_api_access()

    if not hasattr(hub, '_mpegts_muxer') or hub._mpegts_muxer is None:
        return JSONResponse(
            {"error": "Stream not available — GPU not connected"},
            status_code=503
        )

    async def generate():
        async for chunk in hub._mpegts_muxer.consume():
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="video/mp2t",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff",
        }
    )
```

### 2e. Remove/Simplify API Endpoints

**Remove:**
- `GET /api/dreams/current` — no standalone images
- `GET /api/dreams/frames/recent` — no frame buffer
- `GET /api/dreams/frame/{frame_number}` — no frame buffer
- `GET /api/dreams/sse` — SSE was image-based; replaced by MPEG-TS

**Keep:**
- `GET /api/dreams/status` — update to reflect H.264 stats
- `GET /api/dreams/health` — unchanged
- `GET /api/dreams/embed` — update to reference WebSocket + stream URL
- `POST /api/dreams/stop` — unchanged
- `GET /api/dreams/stream` — new MPEG-TS endpoint
- All ComfyUI registry endpoints — unchanged
- All state management endpoints — unchanged

**Update `/api/dreams/status` response** to include video stream info:

```python
"stream": {
    "format": "h264",
    "profile": "baseline",
    "resolution": [1024, 512],
    "target_fps": 17.0,
    "endpoints": {
        "websocket": "wss://aetherawi.red/ws/dreams",
        "mpegts": "https://aetherawi.red/api/dreams/stream",
    }
}
```

---

## 3. Client Side

### 3a. Complete rewrite of `dreams.js`

The client detects WebCodecs support and falls back to MSE. No image-based playback path.

```javascript
/**
 * Dream Window — H.264 Video Stream Client
 *
 * Two decode paths:
 * 1. WebCodecs VideoDecoder (Chrome 94+, Edge 94+)
 *    - Receives raw H.264 NAL units via WebSocket
 *    - Feeds EncodedVideoChunk to VideoDecoder
 *    - Draws VideoFrame to canvas
 *
 * 2. MSE fallback (Safari 17.1+, Firefox)
 *    - Receives raw H.264 NAL units via WebSocket
 *    - Client-side JS muxes NAL → fMP4 segments
 *    - Feeds fMP4 to MediaSource → <video> element
 *
 * If neither is available, shows "unsupported browser" message.
 */

const MSG_FRAME = 0x01;

class DreamViewer {
    constructor(options = {}) {
        // DOM elements
        this.canvasId = options.canvasId || 'dream-canvas';
        this.loadingId = options.loadingId || 'dream-loading';
        this.errorId = options.errorId || 'dream-error';
        this.statusId = options.statusId || 'dream-status';

        this.canvas = document.getElementById(this.canvasId);
        this.ctx = this.canvas?.getContext('2d');
        this.loadingEl = document.getElementById(this.loadingId);
        this.errorEl = document.getElementById(this.errorId);
        this.statusEl = document.getElementById(this.statusId);

        // WebSocket
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.connected = false;

        // Frame tracking
        this.serverFrameNumber = 0;
        this.targetFps = 17.0;
        this.frameCount = 0;
        this.lastFrameTime = 0;

        // Decode path
        this.decodePath = null;  // 'webcodecs' | 'mse' | null
        this.videoDecoder = null;       // WebCodecs
        this.mediaSource = null;        // MSE
        this.sourceBuffer = null;       // MSE
        this.videoElement = null;       // MSE — hidden <video> for decode
        this.videoFrameCount = 0;
        this._lastFrameMetaIsVideoKeyframe = false;
        this._receivedFirstKeyframe = false;

        // MSE NAL buffer (accumulate until we can append)
        this._mseNalQueue = [];
        this._mseInitialized = false;

        // Stats elements
        this.frameCountEl = document.querySelector(
            '#dream-frame-count .dream-stat-value'
        );
        this.viewerCountEl = document.querySelector(
            '#dream-viewer-count .dream-stat-value'
        );
        this.connectionIndicator = document.querySelector(
            '#dream-connection-status .dream-stat-indicator'
        );
        this.connectionStatus = document.querySelector(
            '#dream-connection-status .dream-stat-value'
        );

        this.setupEventListeners();
    }

    // ==================== Initialization ====================

    setupEventListeners() {
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && !this.connected) {
                this.reconnectAttempts = 0;
                this.connect();
            }
        });

        const retryBtn = document.getElementById('dream-retry-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                this.hideError();
                this.reconnectAttempts = 0;
                this.connect();
            });
        }

        window.addEventListener('beforeunload', () => {
            if (this.ws) this.ws.close(1000, 'page_unload');
        });
    }

    _detectDecodePath() {
        // Prefer WebCodecs (lower latency, canvas output)
        try {
            if (typeof VideoDecoder === 'function'
                && typeof EncodedVideoChunk === 'function') {
                return 'webcodecs';
            }
        } catch {}

        // Fall back to MSE
        if (typeof MediaSource === 'function'
            && MediaSource.isTypeSupported('video/mp4; codecs="avc1.42001e"')) {
            return 'mse';
        }

        return null;
    }

    // ==================== WebCodecs Path ====================

    async _initWebCodecs() {
        const config = {
            codec: 'avc1.42001e',  // Baseline Level 3.0
            codedWidth: 1024,
            codedHeight: 512,
        };

        try {
            const support = await VideoDecoder.isConfigSupported(config);
            if (!support.supported) {
                console.warn('H.264 Baseline not supported');
                return false;
            }
        } catch (e) {
            console.warn('isConfigSupported failed:', e);
            return false;
        }

        this.videoDecoder = new VideoDecoder({
            output: (frame) => this._onVideoFrame(frame),
            error: (e) => {
                console.error('VideoDecoder error:', e);
                this._resetDecoder();
            },
        });

        this.videoDecoder.configure(config);
        console.log('WebCodecs VideoDecoder initialized');
        return true;
    }

    _onVideoFrame(frame) {
        if (!this.ctx) { frame.close(); return; }
        this.ctx.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height);
        frame.close();
        this.frameCount++;
        this.lastFrameTime = Date.now();
        this.hideLoading();
    }

    _feedWebCodecs(nalData) {
        if (!this.videoDecoder || this.videoDecoder.state === 'closed') return;

        const isKey = this._lastFrameMetaIsVideoKeyframe;

        // Must receive a keyframe first
        if (!isKey && !this._receivedFirstKeyframe) return;
        if (isKey) this._receivedFirstKeyframe = true;

        const timestamp = this.videoFrameCount * (1_000_000 / this.targetFps);
        this.videoFrameCount++;

        try {
            this.videoDecoder.decode(new EncodedVideoChunk({
                type: isKey ? 'key' : 'delta',
                timestamp: timestamp,
                data: nalData,
            }));
        } catch (e) {
            console.error('Decode failed:', e);
            if (!isKey) this._resetDecoder();
        }
    }

    // ==================== MSE Path ====================

    async _initMSE() {
        // Create hidden video element for decode
        this.videoElement = document.createElement('video');
        this.videoElement.muted = true;
        this.videoElement.autoplay = true;
        this.videoElement.playsInline = true;
        this.videoElement.style.display = 'none';
        document.body.appendChild(this.videoElement);

        this.mediaSource = new MediaSource();
        this.videoElement.src = URL.createObjectURL(this.mediaSource);

        return new Promise((resolve) => {
            this.mediaSource.addEventListener('sourceopen', () => {
                try {
                    this.sourceBuffer = this.mediaSource.addSourceBuffer(
                        'video/mp4; codecs="avc1.42001e"'
                    );
                    this.sourceBuffer.mode = 'sequence';
                    this.sourceBuffer.addEventListener('updateend', () => {
                        this._flushMSEQueue();
                    });
                    this._mseInitialized = true;
                    console.log('MSE initialized');

                    // Draw video frames to canvas
                    this._startMSECanvasLoop();
                    resolve(true);
                } catch (e) {
                    console.error('MSE addSourceBuffer failed:', e);
                    resolve(false);
                }
            });
        });
    }

    _startMSECanvasLoop() {
        // Continuously draw <video> to <canvas>
        const draw = () => {
            if (this.videoElement && this.ctx
                && this.videoElement.readyState >= 2) {
                this.ctx.drawImage(
                    this.videoElement, 0, 0,
                    this.canvas.width, this.canvas.height
                );
                this.hideLoading();
            }
            requestAnimationFrame(draw);
        };
        requestAnimationFrame(draw);
    }

    _feedMSE(nalData) {
        // For MSE we need to wrap NAL units in fMP4.
        // Queue the raw NAL data — the fMP4 muxer processes it.
        const isKey = this._lastFrameMetaIsVideoKeyframe;

        if (!isKey && !this._receivedFirstKeyframe) return;
        if (isKey) this._receivedFirstKeyframe = true;

        // Convert NAL to fMP4 segment and append
        const fmp4Segment = this._muxNALToFMP4(nalData, isKey);
        if (fmp4Segment) {
            this._mseNalQueue.push(fmp4Segment);
            this._flushMSEQueue();
        }
    }

    _flushMSEQueue() {
        if (!this.sourceBuffer || this.sourceBuffer.updating) return;
        if (this._mseNalQueue.length === 0) return;

        const segment = this._mseNalQueue.shift();
        try {
            this.sourceBuffer.appendBuffer(segment);
        } catch (e) {
            console.error('appendBuffer failed:', e);
            // Buffer may be full — remove old data
            if (this.sourceBuffer.buffered.length > 0) {
                try {
                    const start = this.sourceBuffer.buffered.start(0);
                    const end = this.sourceBuffer.buffered.end(0) - 5;
                    if (end > start) {
                        this.sourceBuffer.remove(start, end);
                    }
                } catch {}
            }
        }
    }

    _muxNALToFMP4(nalData, isKeyframe) {
        // Minimal fMP4 muxer for single H.264 stream
        // This is a simplified implementation — for production,
        // consider using jMuxer or a proven fMP4 library.
        //
        // The implementation needs to:
        // 1. Parse SPS/PPS from I-frame NAL units
        // 2. Generate init segment (ftyp + moov) on first keyframe
        // 3. Generate media segments (moof + mdat) for each frame
        //
        // TODO: Implement or integrate jMuxer
        // For now, this is a placeholder that will be filled
        // during implementation phase.
        return null;
    }

    // ==================== Shared Logic ====================

    _resetDecoder() {
        if (this.videoDecoder) {
            try { this.videoDecoder.close(); } catch {}
            this.videoDecoder = null;
        }
        this._receivedFirstKeyframe = false;
        this.videoFrameCount = 0;
        console.log('Decoder reset — waiting for I-frame');
    }

    // ==================== WebSocket ====================

    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) return;

        this.decodePath = this._detectDecodePath();

        if (!this.decodePath) {
            this.showError('Your browser does not support H.264 video decoding. Please use Chrome, Edge, or Safari 17.1+.');
            return;
        }

        this.setStatus('connecting', 'connecting...');
        this.setConnectionState('connecting');

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/dreams`;

        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';
            this.ws.onopen = () => this.handleOpen();
            this.ws.onmessage = (e) => this.handleMessage(e);
            this.ws.onclose = (e) => this.handleClose(e);
            this.ws.onerror = (e) => this.handleError(e);
        } catch (e) {
            console.error('WebSocket failed:', e);
            this.handleError(e);
        }
    }

    async handleOpen() {
        console.log(`Dream WebSocket connected (decode: ${this.decodePath})`);
        this.connected = true;
        this.reconnectAttempts = 0;
        this.setConnectionState('connected');

        // Reset state
        this._resetDecoder();
        this._receivedFirstKeyframe = false;
        this._lastFrameMetaIsVideoKeyframe = false;

        // Initialize decode path
        if (this.decodePath === 'webcodecs') {
            await this._initWebCodecs();
        } else if (this.decodePath === 'mse') {
            await this._initMSE();
        }

        this.startPingInterval();
    }

    handleMessage(event) {
        if (event.data instanceof ArrayBuffer) {
            this.handleBinaryMessage(event.data);
        } else {
            this.handleJsonMessage(event.data);
        }
    }

    handleBinaryMessage(data) {
        const view = new Uint8Array(data);
        if (view[0] !== MSG_FRAME) return;

        const nalData = data.slice(1);

        if (this.decodePath === 'webcodecs') {
            this._feedWebCodecs(nalData);
        } else if (this.decodePath === 'mse') {
            this._feedMSE(nalData);
        }
    }

    handleJsonMessage(data) {
        try {
            const msg = JSON.parse(data);
            switch (msg.type) {
                case 'status':
                    this.handleStatusMessage(msg);
                    break;
                case 'config':
                    if (msg.target_fps > 0) this.targetFps = msg.target_fps;
                    break;
                case 'frame_meta':
                    this.handleFrameMetaMessage(msg);
                    break;
                case 'pong':
                    break;
            }
        } catch (e) {
            console.error('JSON parse failed:', e);
        }
    }

    handleFrameMetaMessage(msg) {
        if (msg.fn !== undefined) {
            this.serverFrameNumber = msg.fn;
            if (this.frameCountEl) {
                this.frameCountEl.textContent =
                    this.serverFrameNumber.toLocaleString();
            }
        }
        this._lastFrameMetaIsVideoKeyframe = msg.vk === true;
    }

    handleStatusMessage(msg) {
        this.setStatus(msg.status, msg.message);
        if (msg.viewer_count !== undefined && this.viewerCountEl) {
            this.viewerCountEl.textContent = msg.viewer_count;
        }
        if (msg.target_fps > 0) this.targetFps = msg.target_fps;

        if (msg.status === 'starting' || msg.status === 'loading_models') {
            this.showLoading();
        } else if (msg.status === 'error') {
            this.showError(msg.message);
        }
    }

    handleClose(event) {
        this.connected = false;
        this.stopPingInterval();

        // Clean up decoder
        if (this.videoDecoder) {
            try { this.videoDecoder.close(); } catch {}
            this.videoDecoder = null;
        }
        if (this.videoElement) {
            this.videoElement.remove();
            this.videoElement = null;
        }
        this.mediaSource = null;
        this.sourceBuffer = null;

        if (event.code === 1000) {
            this.setConnectionState('offline');
            return;
        }
        this.scheduleReconnect();
    }

    handleError(error) {
        console.error('WebSocket error:', error);
        this.setConnectionState('error');
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.showError('Unable to connect after multiple attempts');
            return;
        }
        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1),
            30000
        );
        this.setStatus('reconnecting', `reconnecting in ${Math.round(delay / 1000)}s...`);
        this.setConnectionState('connecting');
        setTimeout(() => { if (!document.hidden) this.connect(); }, delay);
    }

    // ==================== UI Helpers ====================

    setStatus(status, message) {
        if (this.statusEl) {
            this.statusEl.textContent = message;
            this.statusEl.className = `dream-status dream-status-${status}`;
        }
    }

    setConnectionState(state) {
        if (this.connectionIndicator) {
            this.connectionIndicator.className = `dream-stat-indicator ${state}`;
        }
        if (this.connectionStatus) {
            const labels = {
                connected: 'live', connecting: 'connecting',
                offline: 'offline', error: 'error'
            };
            this.connectionStatus.textContent = labels[state] || state;
        }
    }

    showLoading() {
        this.loadingEl?.classList.remove('hidden');
        this.errorEl?.classList.add('hidden');
    }
    hideLoading() {
        this.loadingEl?.classList.add('hidden');
    }
    showError(message) {
        const msgEl = document.getElementById('dream-error-message');
        if (msgEl) msgEl.textContent = message;
        this.errorEl?.classList.remove('hidden');
        this.loadingEl?.classList.add('hidden');
        this.setConnectionState('error');
    }
    hideError() {
        this.errorEl?.classList.add('hidden');
    }

    startPingInterval() {
        this.pingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }
    stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
}

// ==================== Init ====================
document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(location.search);
    if (urlParams.get('embed') === '1') {
        document.body.classList.add('embed-mode');
    }

    window.dreamViewer = new DreamViewer();
    window.dreamViewer.connect();
});
```

### 3b. MSE fMP4 Muxer

The `_muxNALToFMP4` method above is a placeholder. For implementation, there are two options:

**Option A: Use jMuxer** (recommended for initial implementation)
- ~15KB minified, purpose-built for this exact use case
- Handles SPS/PPS parsing, init segment generation, fMP4 boxing
- Well-tested with Safari MSE
- Include via `<script>` tag or bundle

```javascript
// With jMuxer, the MSE path simplifies to:
_initMSE() {
    this.jmuxer = new JMuxer({
        node: 'dream-canvas',      // Can target a <video> element
        mode: 'video',
        fps: this.targetFps,
        flushingTime: 0,           // Immediate playback
        debug: false,
    });
}

_feedMSE(nalData) {
    if (!this._receivedFirstKeyframe) return;
    this.jmuxer.feed({
        video: new Uint8Array(nalData),
    });
}
```

**Option B: Hand-rolled minimal fMP4 muxer** (~200-300 lines)
- Zero dependencies
- Generates ftyp+moov init segment from SPS/PPS
- Generates moof+mdat segments per frame
- More control, but more code to maintain

**Decision for implementation**: Start with jMuxer for speed. Replace with hand-rolled if we need to eliminate the dependency later.

### 3c. HTML Template Changes

**File**: `core/aethera/templates/dreams/viewer.html`

Remove the OG image meta tag pointing to `/api/dreams/current`. Use a static placeholder instead.

```html
<!-- Replace dynamic OG image with static -->
<meta property="og:image" content="{{ request.base_url }}static/images/dreams-preview.png">
```

---

## 4. Wire Protocol Summary

### GPU → VPS
```
0x01 | metadata_len (4 bytes BE) | JSON metadata | H.264 NAL units
```

Metadata:
```json
{
    "fn": 12345,           // Frame number (sequential)
    "kf": 617,             // Keyframe number (generation keyframe, not video I-frame)
    "vk": true,            // Video keyframe (H.264 I-frame)
    "p": "ethereal..."     // Prompt text (optional, only on keyframe changes)
}
```

### VPS → Browser
```
JSON: {"type": "frame_meta", "fn": 12345, "kf": 617, "vk": true, "p": "..."}
Binary: 0x01 | H.264 NAL units
```

### VPS → External (MPEG-TS)
```
GET /api/dreams/stream
Content-Type: video/mp2t
Transfer-Encoding: chunked

[continuous MPEG-TS byte stream]
```

---

## 5. Performance Expectations

| Metric | Current (JPEG) | H.264 | Improvement |
|--------|----------------|-------|-------------|
| Encode time/frame | ~25ms | ~3-5ms | 5-8x faster |
| Avg frame size | 50-75 KB | 8-15 KB (P) / 30-50 KB (I) | 5-8x smaller |
| Bandwidth/viewer | 1.0-1.5 MB/s | 150-300 KB/s | 5x reduction |
| Browser decode | Software (Image API) | Hardware (VideoDecoder) | GPU-accelerated |
| E2E latency | ~200-500ms (3 buffers) | ~50-100ms (pass-through) | 3-5x lower |
| VPS CPU | Playback queue pacing | Pass-through only | Near zero |
| Client memory | frameQueue + Images | VideoDecoder internal | Lower |

---

## 6. Files Changed / Created

### New files
| File | Component | Description |
|------|-----------|-------------|
| `dream_gen/backend/cloud/video_encoder.py` | GPU | H.264 encoder (PyAV) |
| `core/aethera/dreams/mpegts_muxer.py` | VPS | MPEG-TS muxer for HTTP stream |

### Modified files
| File | Component | Changes |
|------|-----------|---------|
| `dream_gen/backend/cloud/frame_pusher.py` | GPU | Replace image encoding with VideoStreamEncoder |
| `dream_gen/backend/config.b200.yaml` | GPU | H.264 config, remove format/quality |
| `core/aethera/dreams/websocket.py` | VPS | Remove playback queue, add pass-through relay |
| `core/aethera/dreams/frame_cache.py` | VPS | Simplify to StreamStats (stats + I-frame cache) |
| `core/aethera/api/dreams.py` | VPS | Remove image endpoints, add MPEG-TS stream |
| `core/aethera/static/js/dreams.js` | Client | WebCodecs + MSE decode, remove image/blend |
| `core/aethera/templates/dreams/viewer.html` | Client | Static OG image, optional jMuxer script |

### Deleted files
| File | Component | Reason |
|------|-----------|--------|
| `core/aethera/dreams/frame_playback.py` | VPS | Replaced by pass-through relay |

---

## 7. Implementation Order

### Phase 1: GPU encoder (testable in isolation)
1. Create `video_encoder.py` with `VideoStreamEncoder`
2. Write a quick test: encode 100 PIL images, verify output is valid H.264
3. Modify `frame_pusher.py` — replace image encoding
4. Update `config.b200.yaml`

### Phase 2: VPS relay
5. Simplify `frame_cache.py` → `StreamStats`
6. Remove `frame_playback.py`
7. Update `websocket.py` — pass-through broadcast, I-frame cache
8. Update `dreams.py` — remove image endpoints
9. Add `mpegts_muxer.py` + `/api/dreams/stream` endpoint

### Phase 3: Browser client
10. Rewrite `dreams.js` — WebCodecs primary path
11. Add MSE fallback with jMuxer
12. Update `viewer.html` template
13. End-to-end test

### Phase 4: Polish
14. Update `/api/dreams/status` response format
15. Update `/api/dreams/embed` response
16. Add static OG preview image
17. Update `DREAMS_API.md` documentation
18. Verify `pyav` available on B200 node1
19. Deploy and test

---

## 8. Dependencies

| Dependency | Where | Version | Notes |
|------------|-------|---------|-------|
| `pyav` (`av`) | GPU + VPS | >=12.0 | FFmpeg wrapper for H.264 encode + MPEG-TS mux |
| `jMuxer` | Browser | 2.0.5+ | fMP4 muxer for MSE fallback (optional — can hand-roll later) |

Both are lightweight. `pyav` is commonly pre-installed in ML environments. `jMuxer` is ~15KB minified.
