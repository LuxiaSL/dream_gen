# Spec: B200 Tuning — Pacing & Video Streaming

## Context

Dream Window on B200 generates keyframes at ~6-7 FPS (was 2-3 on RunPod). All timing parameters were tuned for the old rate. The result: mutations, cache injections, and template swaps happen 3x too frequently. The visual evolution is too aggressive — it should feel slow, dreamy, and continuous.

Additionally, the frame delivery uses per-frame JPEG over WebSocket with a triple buffer (GPU-side, VPS playback queue, client-side). With higher FPS and a persistent connection, this should be replaced with proper video encoding.

## Part 1: Pacing Tuning

### Problem
At the old 2-3 FPS keyframe rate:
- `mutation base_probability: 0.12` = mutation every ~8 keyframes = every ~3 seconds
- `staleness_threshold: 25` = forced mutation every ~10 seconds
- `injection_cooldown: 5` = cache injection possible every ~2 seconds
- `warmup_keyframes: 10` = collapse detection starts after ~4 seconds

At the new 6-7 FPS keyframe rate, all of these are 3x faster:
- Mutation every ~1 second
- Forced mutation every ~3.5 seconds
- Cache injection every ~0.7 seconds
- Collapse detection after ~1.5 seconds

### Recommended config changes (`config.b200.yaml`)

```yaml
generation:
  sd:
    steps: 10
    cfg_scale: 7.0           # was 8.0 — slightly lower for dreamier output

  hybrid:
    interpolation_frames: 20
    target_interpolation_fps: 20.0
    keyframe_denoise: 0.15    # keep low for smooth drift

  img2img:
    denoise: 0.25             # was 0.3

fresh_generation:
  denoising:
    drift: 0.10               # was 0.15 — slower evolution per keyframe
    bend: 0.35                # was 0.45 — less dramatic mutations
    bend_frames: 8            # was 5 — longer transition period

  mutation:
    base_probability: 0.04    # was 0.12 — ~1 mutation per 25 keyframes (~4 seconds)
    staleness_threshold: 75   # was 25 — forced mutation every ~12 seconds
    category_weights:         # unchanged
      color_logic: 0.35
      atmosphere_field: 0.25
      light_behavior: 0.20
      temporal_state: 0.15
      texture_density: 0.05
    similarity_target: 0.55
    similarity_range: 0.25

cache:
  injection_cooldown: 15      # was 5 — cache injection at most every ~2.5 seconds
  seed_injection_cooldown: 15 # was 5
  warmup_keyframes: 30        # was 10 — let it establish baseline before detecting collapse
  collapse_detection_window: 50  # (if present) wider window for stability
  injection_probability: 0.08 # was 0.15 — less random injection

  seed_injection_floor: 0.01  # was 0.02
  seed_injection_max: 0.08    # was 0.15
  seed_injection_ramp: 100    # was 50

  diversity_check_interval: 25 # was 10
```

### Rationale
The goal: at 6-7 FPS keyframes with 20 interpolation frames each, viewer sees ~20 FPS. Visual evolution should be:
- **Drift**: Slow, barely perceptible per-second change. Like watching clouds.
- **Mutations**: Every 4-5 seconds, a subtle component shift (color palette, lighting).
- **Template swaps**: Every 30-60 seconds, a new visual theme emerges gradually.
- **Cache injections**: Rare, only when genuine collapse detected after extended drift.

### How to test
1. Update `config.b200.yaml` with the values above
2. Push to DreamGen repo, pull on node1
3. Cancel + resubmit Heimdall job
4. Watch `aetherawi.red/dreams` for 2-3 minutes — should feel meditative, not frenetic
5. Adjust `drift` denoise (0.08-0.15) and `base_probability` (0.03-0.06) by feel

---

## Part 2: Video Streaming (WebCodecs)

### Current architecture
```
DreamGen (node1)                    VPS (aetherawi.red)              Browser
  PIL image                           WebSocket hub                    JS client
  → JPEG encode (25ms)                → playback queue (pacing)        → receive JPEG
  → WS binary frame                   → broadcast to viewers           → decode
  → per frame: 75KB                   → per frame: 75KB                → draw to canvas
                                                                       → triple buffer
```

**Problems at 20 FPS:**
- 1.5 MB/s bandwidth per viewer (75KB × 20)
- No temporal compression (each frame independent)
- Triple buffer adds complexity + latency
- Client-side jitter from per-frame WebSocket delivery

### Proposed architecture
```
DreamGen (node1)                    VPS (aetherawi.red)              Browser
  PIL image                           WebSocket relay                  WebCodecs
  → H.264 encode (pyav/ffmpeg)        → pass through                   → VideoDecoder
  → I-frame every 2s, P-frames        → (no decode/re-encode)          → canvas render
  → ~200-500 KB/s total               → ~200-500 KB/s                  → native buffering
```

### Implementation plan

#### DreamGen side: `backend/cloud/video_encoder.py` (new file)
```python
class VideoStreamEncoder:
    """Encodes PIL images into H.264 NAL units for WebSocket streaming."""

    def __init__(self, width, height, fps, keyframe_interval=40, bitrate="2M"):
        # Use pyav (FFmpeg wrapper) for encoding
        # Output: raw H.264 Annex B NAL units (no container)
        # I-frame every keyframe_interval frames (~2s at 20fps)

    def encode_frame(self, pil_image: Image) -> bytes:
        # Returns encoded NAL unit(s) for this frame
        # I-frame: ~30-50KB, P-frame: ~5-15KB

    def flush(self) -> bytes:
        # Flush encoder buffer on shutdown
```

Replace the JPEG encode in `frame_pusher.py`:
```python
# Current:
jpeg_data = encode_jpeg(image, quality=80)
await ws.send_bytes(MSG_FRAME + jpeg_data)

# New:
nal_units = self.encoder.encode_frame(image)
await ws.send_bytes(MSG_VIDEO_FRAME + nal_units)
```

**Dependencies**: `pip install av` (PyAV — already a common ML dep, thin FFmpeg wrapper)

#### VPS side: Pass-through relay
The WebSocket hub already broadcasts binary frames to all viewers. Minimal changes:
- New message type `MSG_VIDEO_FRAME = 0x05` (alongside existing `MSG_FRAME = 0x01`)
- Hub passes through without decoding
- On new viewer connect: send the latest I-frame (keyframe) for fast join
- Store last I-frame in `frame_cache` for this purpose

#### Client side: WebCodecs VideoDecoder
```javascript
// Replace current image-per-frame approach:
const decoder = new VideoDecoder({
    output: (frame) => {
        ctx.drawImage(frame, 0, 0);
        frame.close();
    },
    error: (e) => console.error(e),
});

decoder.configure({
    codec: "avc1.42001e",  // H.264 Baseline
    codedWidth: 1024,
    codedHeight: 512,
});

// On WebSocket message:
ws.onmessage = (e) => {
    if (e.data[0] === MSG_VIDEO_FRAME) {
        const isKey = /* check NAL type */;
        decoder.decode(new EncodedVideoChunk({
            type: isKey ? "key" : "delta",
            timestamp: frameNum * (1000000 / fps),
            data: e.data.slice(1),
        }));
    }
};
```

**Browser support**: WebCodecs is in Chrome 94+, Edge 94+, Opera 80+. No Safari yet. Fallback to current JPEG-per-frame for unsupported browsers.

#### Migration path
1. Add `video_encoder.py` to DreamGen, gated by config `cloud.frame_push.format: "h264"` (alongside existing `"jpeg"`)
2. Add `MSG_VIDEO_FRAME` pass-through to VPS hub
3. Add WebCodecs client with JPEG fallback in the viewer template
4. Test with `format: "h264"` in config
5. Once stable, make it the default

### What this eliminates
- **Triple buffer** → VideoDecoder handles its own buffering natively
- **Playback queue on VPS** → pass-through relay, no pacing needed
- **JPEG encode/decode per frame** → H.264 temporal compression
- **1.5 MB/s bandwidth** → ~300 KB/s (5x reduction)
- **Client-side jitter** → VideoDecoder smooths playback automatically

### What stays
- WebSocket transport (low latency, bidirectional for control messages)
- Frame metadata (keyframe number, prompt) sent as JSON alongside video data
- State sync for resume functionality
- Auth token on WebSocket connect

### Estimated effort
- `video_encoder.py`: ~100 lines (pyav is very clean)
- VPS hub changes: ~30 lines (new message type + I-frame cache)
- Client JS: ~80 lines (WebCodecs decoder + fallback)
- Config: ~5 lines
- Testing: need to verify encoding doesn't add meaningful latency (pyav H.264 encode is typically <5ms for 1024x512)

---

## Priority order
1. **Pacing tuning** — config changes only, immediate visual improvement
2. **Video streaming** — architectural improvement, eliminates triple buffer
3. **Dead code cleanup** — separate effort, no urgency
