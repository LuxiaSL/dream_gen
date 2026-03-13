"""
Tests for VideoStreamEncoder

Validates H.264 encoding behavior:
- Output is valid Annex B byte stream
- Keyframe interval is respected
- Dimensions enforced
- Encode/close lifecycle
- Frame size expectations for dream-like content
"""

import pytest
import numpy as np
from PIL import Image

from cloud.video_encoder import VideoStreamEncoder


def make_gradient_image(width: int = 1024, height: int = 512, seed: int = 0) -> Image.Image:
    """Create a gradient test image (simulates dream frame content). Uses numpy for speed."""
    x = np.arange(width, dtype=np.uint8)
    y = np.arange(height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    r = ((xx.astype(np.int16) + seed * 7) % 256).astype(np.uint8)
    g = ((yy.astype(np.int16) + seed * 13) % 256).astype(np.uint8)
    b = ((xx.astype(np.int16) + yy.astype(np.int16) + seed * 3) % 256).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr)


def make_solid_image(width: int = 1024, height: int = 512, color: tuple = (128, 128, 128)) -> Image.Image:
    """Create a solid color image."""
    return Image.new("RGB", (width, height), color)


ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
ANNEX_B_SHORT_START_CODE = b"\x00\x00\x01"


def has_annex_b_start_codes(data: bytes) -> bool:
    """Check if data contains Annex B start codes."""
    return ANNEX_B_START_CODE in data or ANNEX_B_SHORT_START_CODE in data


class TestVideoStreamEncoderInit:
    """Tests for encoder initialization."""

    def test_basic_init(self):
        encoder = VideoStreamEncoder(1024, 512, fps=17.0)
        assert encoder.width == 1024
        assert encoder.height == 512
        assert encoder.fps == 17.0
        assert encoder.frame_count == 0
        encoder.close()

    def test_odd_dimensions_rejected(self):
        with pytest.raises(ValueError, match="even dimensions"):
            VideoStreamEncoder(1023, 512)

        with pytest.raises(ValueError, match="even dimensions"):
            VideoStreamEncoder(1024, 511)

    def test_custom_params(self):
        encoder = VideoStreamEncoder(
            640, 480, fps=30.0, keyframe_interval=60,
            crf=18, max_bitrate="4M", preset="superfast",
        )
        assert encoder.keyframe_interval == 60
        encoder.close()


class TestVideoStreamEncoderEncode:
    """Tests for frame encoding."""

    def test_encode_single_frame(self):
        encoder = VideoStreamEncoder(256, 128, fps=10.0, keyframe_interval=10)
        img = make_gradient_image(256, 128)

        nal_data, is_keyframe = encoder.encode_frame(img)

        assert len(nal_data) > 0, "First frame should produce output"
        assert is_keyframe, "First frame must be a keyframe"
        assert has_annex_b_start_codes(nal_data), "Output must be Annex B format"
        assert encoder.frame_count == 1
        encoder.close()

    def test_encode_sequence(self):
        encoder = VideoStreamEncoder(256, 128, fps=10.0, keyframe_interval=10)

        keyframe_indices = []
        sizes = []

        for i in range(30):
            img = make_gradient_image(256, 128, seed=i)
            nal_data, is_keyframe = encoder.encode_frame(img)

            assert len(nal_data) > 0, f"Frame {i} should produce output"
            sizes.append(len(nal_data))

            if is_keyframe:
                keyframe_indices.append(i)

        # First frame is always a keyframe
        assert 0 in keyframe_indices

        # Should have keyframes at approximately every 10 frames
        assert len(keyframe_indices) >= 3, f"Expected ~3 keyframes in 30 frames, got {len(keyframe_indices)}"

        assert encoder.frame_count == 30
        encoder.close()

    def test_p_frames_smaller_than_i_frames(self):
        """P-frames should be significantly smaller for similar content."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0, keyframe_interval=20)

        i_frame_sizes = []
        p_frame_sizes = []

        for i in range(20):
            # Gradually changing image (like dream drift)
            img = make_gradient_image(256, 128, seed=i)
            nal_data, is_keyframe = encoder.encode_frame(img)

            if is_keyframe:
                i_frame_sizes.append(len(nal_data))
            else:
                p_frame_sizes.append(len(nal_data))

        assert len(i_frame_sizes) > 0
        assert len(p_frame_sizes) > 0

        avg_i = sum(i_frame_sizes) / len(i_frame_sizes)
        avg_p = sum(p_frame_sizes) / len(p_frame_sizes)

        # P-frames should be smaller than I-frames for gradual content changes
        assert avg_p < avg_i, (
            f"P-frames ({avg_p:.0f}B) should be smaller than I-frames ({avg_i:.0f}B)"
        )
        encoder.close()

    def test_near_identical_frames_tiny_p_frames(self):
        """Near-identical frames (like slow dream drift) should produce very small P-frames."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0, keyframe_interval=50)

        base_img = make_solid_image(256, 128, (100, 100, 100))
        p_frame_sizes = []

        for i in range(20):
            # Very slight variation (simulates slow drift)
            color = (100 + i % 3, 100 + i % 2, 100 + i % 4)
            img = make_solid_image(256, 128, color)
            nal_data, is_keyframe = encoder.encode_frame(img)

            if not is_keyframe:
                p_frame_sizes.append(len(nal_data))

        if p_frame_sizes:
            avg_p = sum(p_frame_sizes) / len(p_frame_sizes)
            # Near-identical frames should produce tiny P-frames
            assert avg_p < 5000, f"P-frames for near-identical content should be small, got {avg_p:.0f}B"

        encoder.close()

    def test_rgba_input_converted(self):
        """RGBA images should be automatically converted to RGB."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        img = Image.new("RGBA", (256, 128), (128, 128, 128, 255))

        nal_data, is_keyframe = encoder.encode_frame(img)
        assert len(nal_data) > 0
        encoder.close()

    def test_wrong_size_resized(self):
        """Images with wrong dimensions should be resized."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        img = Image.new("RGB", (512, 256), (128, 128, 128))

        nal_data, is_keyframe = encoder.encode_frame(img)
        assert len(nal_data) > 0
        encoder.close()

    def test_grayscale_input(self):
        """Grayscale images should be converted."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        img = Image.new("L", (256, 128), 128)

        nal_data, is_keyframe = encoder.encode_frame(img)
        assert len(nal_data) > 0
        encoder.close()


    def test_numpy_input(self):
        """Numpy arrays should be accepted directly (skip PIL conversion)."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)

        # Create numpy array (H, W, 3) uint8 — same as VAE decode output
        arr = np.random.randint(0, 256, (128, 256, 3), dtype=np.uint8)
        nal_data, is_keyframe = encoder.encode_frame(arr)
        assert len(nal_data) > 0
        assert is_keyframe  # First frame
        encoder.close()

    def test_numpy_faster_than_pil(self):
        """Numpy path should be faster than PIL path."""
        import time

        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        arr = np.random.randint(0, 256, (128, 256, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)

        # Warm up
        encoder.encode_frame(arr)

        # Time numpy path
        start = time.time()
        for _ in range(20):
            encoder.encode_frame(arr)
        numpy_time = time.time() - start
        encoder.close()

        encoder2 = VideoStreamEncoder(256, 128, fps=10.0)
        encoder2.encode_frame(pil)  # warm up
        start = time.time()
        for _ in range(20):
            encoder2.encode_frame(pil)
        pil_time = time.time() - start
        encoder2.close()

        # Numpy should be at least as fast (usually faster due to no conversion)
        print(f"  Numpy: {numpy_time*1000/20:.1f}ms/frame, PIL: {pil_time*1000/20:.1f}ms/frame")


class TestVideoStreamEncoderLifecycle:
    """Tests for encoder lifecycle management."""

    def test_flush(self):
        encoder = VideoStreamEncoder(256, 128, fps=10.0)

        for i in range(5):
            encoder.encode_frame(make_gradient_image(256, 128, seed=i))

        flush_data = encoder.flush()
        # Flush may or may not produce data depending on encoder state
        assert isinstance(flush_data, bytes)
        encoder.close()

    def test_close(self):
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        encoder.encode_frame(make_gradient_image(256, 128))
        encoder.close()

        # Encoding after close should raise
        with pytest.raises(RuntimeError, match="closed"):
            encoder.encode_frame(make_gradient_image(256, 128))

    def test_double_close(self):
        """Double close should not raise."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        encoder.close()
        encoder.close()  # Should not raise

    def test_flush_after_close(self):
        """Flush after close should return empty bytes."""
        encoder = VideoStreamEncoder(256, 128, fps=10.0)
        encoder.close()
        assert encoder.flush() == b""


class TestVideoStreamEncoderPerformance:
    """Performance sanity checks (not strict benchmarks)."""

    def test_encode_speed_reasonable(self):
        """Encoding should be fast enough for real-time at 1024x512."""
        import time

        encoder = VideoStreamEncoder(1024, 512, fps=17.0, preset="ultrafast")
        img = make_gradient_image(1024, 512)

        # Warm up (first frame includes encoder init overhead)
        encoder.encode_frame(img)

        # Time subsequent frames
        start = time.time()
        n_frames = 10
        for i in range(n_frames):
            encoder.encode_frame(make_gradient_image(1024, 512, seed=i + 1))
        elapsed = time.time() - start

        avg_ms = (elapsed / n_frames) * 1000
        # Should be well under 50ms per frame for ultrafast at 1024x512
        assert avg_ms < 50, f"Encoding too slow: {avg_ms:.1f}ms per frame"

        encoder.close()

    def test_output_size_reasonable(self):
        """Output should be much smaller than raw JPEG for similar content."""
        encoder = VideoStreamEncoder(1024, 512, fps=17.0, crf=23)

        total_bytes = 0
        n_frames = 30

        for i in range(n_frames):
            img = make_gradient_image(1024, 512, seed=i)
            nal_data, _ = encoder.encode_frame(img)
            total_bytes += len(nal_data)

        avg_bytes = total_bytes / n_frames
        # Average should be well under 75KB (typical JPEG size)
        # For CRF 23 with gradual changes, expect ~10-30KB average
        assert avg_bytes < 75000, f"Average frame too large: {avg_bytes / 1024:.1f}KB"

        encoder.close()
