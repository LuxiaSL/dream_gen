"""
Tests for the keyframe display round-trip fix (interpolation_worker).

The displayed keyframe was the only frame skipping the VAE round-trip,
popping ~7.5x the baseline frame delta. The worker now assigns the
round-trip decode to the keyframe's in-memory display image.

Contracts:
- assignment sets spec.image for pending/ready keyframes
- displayed/failed keyframes are left alone (too late / pointless)
- unknown keyframe numbers are a no-op, never a raise
- the raw file path is never touched
"""

from pathlib import Path

import numpy as np
import pytest

from core.frame_buffer import FrameBuffer, FrameState
from core.workers.interpolation_worker import InterpolationWorker


@pytest.fixture
def buffer_and_worker(tmp_path: Path):
    buf = FrameBuffer(
        interpolation_frames=4, target_fps=4.0, output_dir=tmp_path
    )
    worker = InterpolationWorker(
        vae_access=None, frame_buffer=buf,
        config={"generation": {"hybrid": {"keyframe_display_roundtrip": True}}},
    )
    return buf, worker


def fake_image():
    return np.zeros((64, 128, 3), dtype=np.uint8)


def test_assigns_image_to_ready_keyframe(buffer_and_worker):
    buf, worker = buffer_and_worker
    seq = buf.register_keyframe(7)
    buf.mark_ready(seq, buf.frames[seq].file_path)

    img = fake_image()
    worker._assign_keyframe_display_image(7, img)

    spec = buf.frames[seq]
    assert spec.image is img
    # File path untouched — generation still reads the raw keyframe
    assert spec.file_path.name == "keyframe_007.png"


def test_skips_displayed_keyframe(buffer_and_worker):
    buf, worker = buffer_and_worker
    seq = buf.register_keyframe(3)
    buf.mark_ready(seq, buf.frames[seq].file_path)
    buf.frames[seq].state = FrameState.DISPLAYED

    worker._assign_keyframe_display_image(3, fake_image())
    assert buf.frames[seq].image is None


def test_skips_failed_keyframe(buffer_and_worker):
    buf, worker = buffer_and_worker
    seq = buf.register_keyframe(4)
    buf.frames[seq].state = FrameState.FAILED

    worker._assign_keyframe_display_image(4, fake_image())
    assert buf.frames[seq].image is None


def test_unknown_keyframe_is_noop(buffer_and_worker):
    _, worker = buffer_and_worker
    worker._assign_keyframe_display_image(999, fake_image())  # must not raise


def test_disabled_by_config(tmp_path):
    buf = FrameBuffer(interpolation_frames=4, target_fps=4.0, output_dir=tmp_path)
    worker = InterpolationWorker(
        vae_access=None, frame_buffer=buf,
        config={"generation": {"hybrid": {"keyframe_display_roundtrip": False}}},
    )
    assert worker.keyframe_display_roundtrip is False


def test_default_is_enabled(tmp_path):
    buf = FrameBuffer(interpolation_frames=4, target_fps=4.0, output_dir=tmp_path)
    worker = InterpolationWorker(vae_access=None, frame_buffer=buf, config={})
    assert worker.keyframe_display_roundtrip is True
