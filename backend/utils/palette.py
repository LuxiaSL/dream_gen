"""
Palette grounding — steer an image's colors toward a reference render.

The img2img lineage drifts in color: over repeated generations the chain
walks toward the model's own bias (measured 2026-09-25: magenta-crimson hue
share 0.87 across 27,505 chained keyframes vs median 0.02 for fresh txt2img
frames of the same templates). Color words in the prompt barely move it,
because img2img preserves low-frequency color above all else.

Grounding recolors an image toward a reference (a txt2img render of the
current prompt, or an era's fresh frame) by Reinhard mean/std transfer in
CIELAB, applied partially: chroma (a*, b*) carries the palette, lightness
(L*) carries structure and is touched only lightly. The img2img step that
follows harmonizes the recolored init, so no transfer seams survive.

Dependency-free (numpy + PIL). Public functions never raise.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# sRGB (D65) <-> XYZ
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_M_INV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0

STATS_SIZE = (256, 128)  # channel statistics are computed on a downscale


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 [...,3] sRGB -> float64 [...,3] CIELAB (L in 0..100)."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > _DELTA ** 3, np.cbrt(xyz),
                 xyz / (3 * _DELTA ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float [...,3] CIELAB -> uint8 [...,3] sRGB (out-of-gamut clipped)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f ** 3, 3 * _DELTA ** 2 * (f - 4.0 / 29.0)) * _WHITE
    lin = np.clip(xyz @ _M_INV.T, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


def _stats(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    lab = rgb_to_lab(np.asarray(img.convert("RGB").resize(STATS_SIZE)))
    flat = lab.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0)


def palette_summary(img: Image.Image) -> Optional[str]:
    """Short 'a*/b* mean, chroma' string for logs. None on failure."""
    try:
        mu, _ = _stats(img)
        chroma = float(np.hypot(mu[1], mu[2]))
        hue = float(np.degrees(np.arctan2(mu[2], mu[1])) % 360)
        return f"L{mu[0]:.0f} a{mu[1]:+.0f} b{mu[2]:+.0f} (hue {hue:.0f}, C {chroma:.0f})"
    except Exception:
        return None


def transfer_palette(
    src: Image.Image,
    ref: Optional[Image.Image],
    chroma_strength: float = 0.6,
    luma_strength: float = 0.2,
    std_clamp: Tuple[float, float] = (0.6, 1.6),
) -> Image.Image:
    """
    Move `src`'s color statistics toward `ref`'s.

    Each LAB channel is Reinhard-transferred (x - mu_s) * (sd_r/sd_s) + mu_r,
    then mixed back with the original by its strength (0 = untouched,
    1 = full transfer). The std ratio is clamped so a near-flat image isn't
    amplified into noise. Returns `src` unchanged on any failure.
    """
    try:
        if ref is None or (chroma_strength <= 0 and luma_strength <= 0):
            return src
        src_rgb = src.convert("RGB")
        lab = rgb_to_lab(np.asarray(src_rgb))
        mu_s, sd_s = _stats(src_rgb)
        mu_r, sd_r = _stats(ref)
        lo, hi = std_clamp
        out = lab.copy()
        for ch, strength in ((0, luma_strength), (1, chroma_strength), (2, chroma_strength)):
            strength = float(min(1.0, max(0.0, strength)))
            if strength == 0.0:
                continue
            ratio = float(np.clip(sd_r[ch] / max(sd_s[ch], 1e-3), lo, hi))
            target = (lab[..., ch] - mu_s[ch]) * ratio + mu_r[ch]
            out[..., ch] = lab[..., ch] + strength * (target - lab[..., ch])
        out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
        return Image.fromarray(lab_to_rgb(out), mode="RGB")
    except Exception:
        logger.warning("Palette transfer failed (using image unchanged)", exc_info=True)
        return src
