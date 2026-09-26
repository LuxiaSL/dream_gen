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


class VecVAE:
    """encode() returns a 1x4x4x8 latent whose pooled embedding is a given vector."""

    def __init__(self, vecs: dict):
        self.vecs = vecs
        self.encodes = []

    async def encode_async(self, path, for_interpolation=True):
        self.encodes.append(Path(path).name)
        return torch.tensor(self.vecs[Path(path).name]).reshape(1, 4, 4, 8)

    async def decode_async(self, latent, upscale_to_target=True):
        return Image.new("RGB", (8, 4))


def test_selection_uses_the_present_frame_itself_when_no_pooled_latent_is_ready(tmp_path, src):
    # The interpolation worker usually lags generation, so the orchestrator
    # often has no pooled latent to pass: the strategy must encode the
    # present frame itself (once, reused for the blend).
    cfg = per_era_cfg(tmp_path)
    mgr = CacheManager(cfg)
    mgr.switch_template("liminal", era_start_kf=0)
    cur = unit(30)
    near = cur + 0.01 * unit(31)
    near /= np.linalg.norm(near)
    id_far = mgr.add(src, "far memory", {"keyframe_num": 3}, emb(-cur))
    id_near = mgr.add(src, "near memory", {"keyframe_num": 4}, emb(near))

    current = tmp_path / "current.png"
    Image.new("RGB", (8, 4)).save(current)
    vae = VecVAE({"current.png": cur,
                  mgr.entries[id_far].image_path.name: -cur,
                  mgr.entries[id_near].image_path.name: near})
    sim = SimpleNamespace(encode_image=lambda p: {"color": [0.0] * 96, "struct": "ab" * 8})
    out = tmp_path / "out"
    out.mkdir()
    strat = CacheInjectionStrategy(cfg, mgr, sim, vae_access=vae,
                                   buffer=SimpleNamespace(keyframe_dir=out))

    _, meta = asyncio.run(strat.inject_dissimilar_keyframe(current, 9, current_latent_vec=None))
    assert meta["selection"] == "latent"
    assert meta["cache_id"] == id_far, "the near memory is under min_dist and never eligible"
    assert vae.encodes.count("current.png") == 1, "present encoded once, reused for the blend"


def _strategy_with(tmp_path, src, merger):
    cfg = per_era_cfg(tmp_path)
    mgr = CacheManager(cfg)
    mgr.switch_template("liminal", era_start_kf=0)
    cur = unit(40)
    mid = mgr.add(src, "a lantern", {"keyframe_num": 2}, emb([-x for x in cur]))
    current = tmp_path / "current.png"
    Image.new("RGB", (8, 4)).save(current)
    vae = VecVAE({"current.png": cur, mgr.entries[mid].image_path.name: [-x for x in cur]})
    sim = SimpleNamespace(encode_image=lambda p: {"color": [0.0] * 96, "struct": "ab" * 8})
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    strat = CacheInjectionStrategy(cfg, mgr, sim, vae_access=vae, buffer=SimpleNamespace(keyframe_dir=out))
    strat.merger = merger
    return strat, current, vae


def test_merger_replaces_the_blend_when_it_succeeds(tmp_path, src):
    calls = []

    async def merger(present, memory, target):
        calls.append((Path(present).name, Path(target).name))
        Image.new("RGB", (8, 4)).save(target)
        return True

    strat, current, vae = _strategy_with(tmp_path, src, merger)
    path, meta = asyncio.run(strat.inject_dissimilar_keyframe(current, 7, current_latent_vec=None))
    assert meta["type"] == "memory_merge" and meta["memory_prompt"] == "a lantern"
    assert calls == [("current.png", "keyframe_007.png")] and path.exists()


@pytest.mark.parametrize("behaviour", ["declines", "raises"])
def test_blend_is_the_fallback_when_the_merge_does_not_happen(tmp_path, src, behaviour):
    async def merger(present, memory, target):
        if behaviour == "raises":
            raise RuntimeError("no ip adapter")
        return False

    strat, current, _ = _strategy_with(tmp_path, src, merger)
    path, meta = asyncio.run(strat.inject_dissimilar_keyframe(current, 8, current_latent_vec=None))
    assert meta["type"] == "dissimilar_cache_injection" and path.exists()
