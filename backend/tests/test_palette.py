"""
Tests for palette grounding (utils/palette.py, 2026-09-25).

The img2img lineage drifts toward magenta; grounding recolors each re-anchor
init toward a prompt-faithful reference. These pin the color math and the
never-raise contract.
"""

import numpy as np
import pytest
from PIL import Image

from utils.palette import (
    lab_to_rgb, palette_summary, rgb_to_lab, transfer_palette, _stats,
)


def solid(rgb, size=(64, 32)):
    return Image.new("RGB", size, rgb)


def noisy(rgb, spread=30, seed=0, size=(128, 64)):
    rs = np.random.RandomState(seed)
    base = np.array(rgb, dtype=np.float64)
    arr = np.clip(base + rs.randn(size[1], size[0], 3) * spread, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def test_lab_reference_values():
    lab = rgb_to_lab(np.array([[[255, 255, 255], [255, 0, 0]]], dtype=np.uint8))
    white, red = lab[0, 0], lab[0, 1]
    assert white[0] == pytest.approx(100.0, abs=0.1)
    assert abs(white[1]) < 0.1 and abs(white[2]) < 0.1
    assert red == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_lab_roundtrip_is_near_lossless():
    rs = np.random.RandomState(1)
    rgb = rs.randint(0, 256, size=(40, 60, 3)).astype(np.uint8)
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_zero_strength_is_identity():
    src = noisy((200, 40, 160))
    out = transfer_palette(src, noisy((40, 80, 200), seed=2),
                           chroma_strength=0.0, luma_strength=0.0)
    assert out is src


def test_chroma_moves_toward_reference_by_strength():
    src = noisy((190, 60, 170), seed=3)     # magenta
    ref = noisy((60, 110, 190), seed=4)     # blue
    mu_s, _ = _stats(src)
    mu_r, _ = _stats(ref)
    out = transfer_palette(src, ref, chroma_strength=0.6, luma_strength=0.0)
    mu_o, _ = _stats(out)
    for ch in (1, 2):
        expected = mu_s[ch] + 0.6 * (mu_r[ch] - mu_s[ch])
        assert mu_o[ch] == pytest.approx(expected, abs=3.0)
    # lightness untouched when luma_strength is 0 (up to gamut clipping)
    assert mu_o[0] == pytest.approx(mu_s[0], abs=2.0)


def test_full_transfer_lands_on_reference_palette():
    src = noisy((180, 50, 150), seed=5)
    ref = noisy((70, 140, 90), seed=6)
    out = transfer_palette(src, ref, chroma_strength=1.0, luma_strength=1.0)
    mu_o, _ = _stats(out)
    mu_r, _ = _stats(ref)
    assert np.allclose(mu_o, mu_r, atol=4.0)


def test_flat_source_is_not_amplified_into_noise():
    src = solid((200, 30, 150))              # std 0 -> ratio must clamp
    ref = noisy((40, 90, 200), spread=60, seed=7)
    out = np.asarray(transfer_palette(src, ref, chroma_strength=1.0, luma_strength=1.0))
    assert np.isfinite(out).all()
    assert out.std(axis=(0, 1)).max() < 2.0  # still flat, just recolored


def test_never_raises_and_returns_source_on_bad_input():
    src = noisy((100, 100, 100))
    assert transfer_palette(src, None) is src
    assert transfer_palette(src, "not an image") is src  # type: ignore[arg-type]
    assert palette_summary("nope") is None  # type: ignore[arg-type]


def test_output_matches_source_size_and_mode():
    src = noisy((120, 60, 180), size=(96, 48))
    out = transfer_palette(src, noisy((30, 160, 120), seed=8))
    assert out.size == src.size and out.mode == "RGB"
