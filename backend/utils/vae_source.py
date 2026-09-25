"""
Which VAE weights to load (config: generation.vae).

The stock SD 1.5 VAE shifts color toward magenta on every encode/decode round
trip (measured 2026-09-25, experiments/model_lab: a* +0.88 / b* -0.67 per
trip; a 40-step img2img loop goes 0.00 -> 0.79 magenta share). Stability's
decoder-only fine-tune `stabilityai/sd-vae-ft-ema` is color-neutral in the
same loop (0.00 -> 0.05) and keeps detail, while ft-mse blurs under repeated
round trips. The encoder is unchanged, so the UNet's latent space is too.

    generation:
      vae:
        model: "stabilityai/sd-vae-ft-ema"   # null = the pipeline's stock VAE
        cache_dir: "~/luxi-files/model-cache"
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def vae_source(config: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """(model_id, cache_dir) from config; (None, None) means stock. Never raises."""
    try:
        vae = ((config or {}).get("generation") or {}).get("vae") or {}
        model = vae.get("model") or None
        cache_dir = vae.get("cache_dir") or None
        if cache_dir:
            cache_dir = os.path.expanduser(str(cache_dir))
        return (str(model) if model else None), cache_dir
    except Exception:
        return None, None
