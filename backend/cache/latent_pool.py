"""
Pooled VAE-latent embeddings — a free perceptual distance metric.

Every keyframe is already VAE-encoded for slerp interpolation; average-pooling
that latent to a small fixed grid gives a compact embedding whose cosine
distance tracks joint color+structure difference. Unlike the chi-square color
histogram (bin-edge explosions on near-uniform images) and pHash-8 (64-bit
saturation), it is continuous and well-behaved within a motif — see the
2026-07-19 cache metric audit that motivated this module.

Used for cache admission gating and injection selection. The modulo schedule
still decides WHEN to intervene; this only decides WHAT gets admitted/injected.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pool target: (channels stay, spatial pooled to H=4, W=8) -> 4*4*8 = 128 dims
POOL_H = 4
POOL_W = 8


def pool_latent(latent: torch.Tensor) -> Optional[np.ndarray]:
    """
    Pool a VAE latent to a compact L2-normalized embedding vector.

    Accepts [C, H, W] or [B, C, H, W] (B=1). Returns float32 ndarray of
    C * POOL_H * POOL_W dims (128 for SD's 4-channel latents), unit norm.
    Never raises — returns None on any failure.
    """
    try:
        with torch.no_grad():
            t = latent
            if t.dim() == 4:
                t = t[0]
            if t.dim() != 3:
                return None
            pooled = F.adaptive_avg_pool2d(t.float(), (POOL_H, POOL_W))
            vec = pooled.flatten().detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8 or not np.isfinite(norm):
                return None
            return vec / norm
    except Exception:
        logger.debug("pool_latent failed", exc_info=True)
        return None


def cosine_dist(a: Union[np.ndarray, List[float]],
                b: Union[np.ndarray, List[float]]) -> float:
    """
    Cosine distance between two pooled embeddings (0 = identical, 2 = opposite).
    Inputs may be lists (deserialized cache entries) or ndarrays. Assumes
    unit-normalized inputs (as produced by pool_latent); re-normalizes
    defensively if not.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(1.0 - np.dot(va / na, vb / nb))
