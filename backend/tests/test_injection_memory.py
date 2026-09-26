"""
Cache injection: which memory is recalled, and that it really is blended.

- Per-era cache: an admission still in flight when the era switches must not
  land in the new era's cache (seen live 2026-09-26: kf 21998 admitted one
  second after the fresh start at kf 22000, then injected twice into the new
  era — the elastic band returning through a race).
- The latent-first selection path must produce the configured blend, not a
  silent direct copy (its metadata once named the dual-metric locals).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cache.injection_strategy import CacheInjectionStrategy
from cache.manager import CacheManager


def unit(seed: int) -> np.ndarray:
    x = np.random.RandomState(seed).randn(128).astype(np.float32)
    return x / np.linalg.norm(x)


def emb(vec) -> dict:
    return {"color": [0.0] * 96, "struct": "ab" * 8, "latent": [float(x) for x in vec]}


def per_era_cfg(tmp: Path) -> dict:
    return {
        "system": {"cache_dir": str(tmp), "output_dir": str(tmp / "out")},
        "generation": {"cache": {
            "max_size": 10,
            "fresh_cache_per_era": True,
            "blend_weight": 0.35,
            "latent_admission": {"min_dist": 0.20},
            "self_regulation": {"eviction": "redundancy", "entry_ttl_minutes": 0},
        }},
    }


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "src.png"
    Image.new("RGB", (8, 8)).save(p)
    return p


def test_admission_from_the_previous_era_is_refused(tmp_path, src):
    mgr = CacheManager(per_era_cfg(tmp_path))
    mgr.switch_template("site_decay", era_start_kf=21000)
    mgr.add(src, "old era frame", {"keyframe_num": 21500}, emb(unit(1)))
    mgr.switch_template("material_collision", era_start_kf=22000)
    assert mgr.size() == 0

    late = mgr.add(src, "storm drain outfall pipe", {"keyframe_num": 21998}, emb(unit(2)))
    assert late is None, "a frame from before the switch must not enter the new era"
    assert mgr.size() == 0

    ok = mgr.add(src, "new era frame", {"keyframe_num": 22005}, emb(unit(3)))
    assert ok is not None and mgr.size() == 1


def test_admissions_without_a_keyframe_or_era_are_unaffected(tmp_path, src):
    mgr = CacheManager(per_era_cfg(tmp_path))
    assert mgr.add(src, "boot", {"keyframe_num": 5}, emb(unit(4))) is not None  # no era yet
    mgr.switch_template("liminal", era_start_kf=1000)
    assert mgr.add(src, "no kf", {}, emb(unit(5))) is not None


class FakeVAE:
    """Latents are 1x4x2x2 tensors filled with a per-image constant."""

    def __init__(self, values: dict):
        self.values = values
        self.decoded = []

    async def encode_async(self, path, for_interpolation=True):
        return torch.full((1, 4, 2, 2), float(self.values[Path(path).name]))

    async def decode_async(self, latent, upscale_to_target=True):
        self.decoded.append(float(latent.mean()))
        return Image.new("RGB", (8, 4))


def test_latent_selection_blends_and_names_the_memory(tmp_path, src):
    cfg = per_era_cfg(tmp_path)
    mgr = CacheManager(cfg)
    mgr.switch_template("liminal", era_start_kf=0)
    cur = unit(10)
    far = -cur  # cosine distance 2.0: certainly selected
    cid = mgr.add(src, "a door in the fog", {"keyframe_num": 7}, emb(far))
    cached_name = mgr.entries[cid].image_path.name

    current = tmp_path / "current.png"
    Image.new("RGB", (8, 4)).save(current)
    vae = FakeVAE({"current.png": 0.0, cached_name: 1.0})
    sim = SimpleNamespace(encode_image=lambda p: {"color": [0.0] * 96, "struct": "ab" * 8})
    out = tmp_path / "out"
    out.mkdir()
    strat = CacheInjectionStrategy(cfg, mgr, sim, vae_access=vae,
                                   buffer=SimpleNamespace(keyframe_dir=out))

    result = asyncio.run(strat.inject_dissimilar_keyframe(current, 42, current_latent_vec=cur))
    assert result is not None
    path, meta = result
    assert meta["type"] == "dissimilar_cache_injection", "must blend, not fall back to a copy"
    assert meta["selection"] == "latent" and meta["latent_dist"] > 1.9
    assert meta["memory_prompt"] == "a door in the fog" and meta["memory_keyframe"] == 7
    assert vae.decoded == [pytest.approx(0.35)], "35% memory, 65% present"
    assert path.exists()
