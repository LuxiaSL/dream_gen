"""
Tests for latent-gated cache admission + selection (cache/latent_pool.py).

Motivated by the 2026-07-19 metric audit: colorhist/phash cannot rank
within-motif difference, so the cache flooded with near-copies and injection
became self-reinforcing. Admission now gates on pooled-latent cosine distance
plus a metric-free temporal floor.

Requires torch (pool_latent) — runs on node1 / CI, not the minimal local venv.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cache.latent_pool import cosine_dist, pool_latent
from core.workers.cache_worker import CacheAnalysisWorker


class StubCache:
    def __init__(self):
        self.entries = []

    def add(self, path, prompt, params, embedding):
        e = type("E", (), {})()
        e.embedding = embedding
        e.cache_id = f"c{len(self.entries)}"
        self.entries.append(e)
        return e.cache_id

    def get_all(self):
        return self.entries

    def size(self):
        return len(self.entries)


class StubSim:
    def encode_image(self, path):
        return {"color": np.zeros(96), "struct": "ab" * 8}

    def to_serializable(self, e):
        out = {"color": list(e["color"]), "struct": e["struct"]}
        if e.get("latent") is not None:
            out["latent"] = [float(x) for x in e["latent"]]
        return out


def unit(seed: int) -> np.ndarray:
    x = np.random.RandomState(seed).randn(128).astype(np.float32)
    return x / np.linalg.norm(x)


CONFIG = {"generation": {"cache": {"latent_admission": {
    "enabled": True, "min_dist": 0.10, "min_interval_kf": 5,
    "min_interval_s": 0.0,  # wall floor off: tests admit in rapid succession
    "latent_wait_s": 0.5}}}}


@pytest.fixture
def frame(tmp_path: Path) -> Path:
    p = tmp_path / "kf.png"
    Image.new("RGB", (8, 8)).save(p)
    return p


def test_pool_latent_shape_and_norm():
    lat = torch.randn(4, 64, 128)
    v = pool_latent(lat)
    assert v is not None and v.shape == (128,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    assert np.allclose(v, pool_latent(lat.unsqueeze(0)))  # batched input


def test_pool_latent_degenerate_inputs():
    assert pool_latent(torch.zeros(4, 64, 128)) is None
    assert pool_latent(torch.randn(7)) is None


def test_cosine_dist_bounds_and_list_input():
    v = unit(1)
    assert cosine_dist(v, v) < 1e-6
    assert cosine_dist(v, list(-v)) > 1.99


async def _try_admit(worker, frame, kf, vec):
    ok, emb = await worker._analyze_frame_diversity(
        {"path": frame, "metadata": {"keyframe_num": kf}, "latent_vec": vec})
    if ok:
        await worker._cache_frame(
            {"path": frame, "prompt": "p", "metadata": {}}, emb)
        worker._last_admitted_kf = kf
    return ok


async def test_admission_floor_and_distance_gate(frame):
    w = CacheAnalysisWorker(StubCache(), StubSim(), config=CONFIG)
    a, b = unit(1), unit(2)
    near_a = a + 0.01 * unit(3)
    near_a /= np.linalg.norm(near_a)

    assert await _try_admit(w, frame, 10, a) is True     # first always in
    assert await _try_admit(w, frame, 11, b) is False    # temporal floor
    assert w.frames_rejected_floor == 1
    assert await _try_admit(w, frame, 20, near_a) is False  # too similar
    assert w.frames_rejected_similar == 1
    assert await _try_admit(w, frame, 30, b) is True     # distinct
    assert w.cache.size() == 2
    # latent survives serialization into stored entries
    assert all(e.embedding.get("latent") for e in w.cache.get_all())


async def test_no_latent_falls_back_to_floor_only(frame):
    w = CacheAnalysisWorker(StubCache(), StubSim(), config=CONFIG)
    ok, emb = await w._analyze_frame_diversity(
        {"path": frame, "metadata": {"keyframe_num": 1}, "latent_vec": None})
    assert ok is True
    assert emb.get("latent") is None


async def test_await_latent_provider_and_timeout():
    w = CacheAnalysisWorker(StubCache(), StubSim(), config=CONFIG)
    a = unit(1)
    w.latent_provider = {42: a}.get
    assert await w._await_latent(42) is a
    assert await w._await_latent(999) is None  # times out


async def test_disabled_admission_admits_everything(frame):
    cfg = {"generation": {"cache": {"latent_admission": {
        "enabled": False, "min_dist": 0.10, "min_interval_kf": 5}}}}
    w = CacheAnalysisWorker(StubCache(), StubSim(), config=cfg)
    a = unit(1)
    assert await _try_admit(w, frame, 1, a) is True
    assert await _try_admit(w, frame, 2, a) is True  # identical, still in


async def test_wall_clock_floor_covers_kfless_submissions(frame):
    """A submission without keyframe_num cannot bypass admission — the
    wall-clock floor rejects it (the injection double-cache bug)."""
    cfg = {"generation": {"cache": {"latent_admission": {
        "enabled": True, "min_dist": 0.10, "min_interval_kf": 5,
        "min_interval_s": 60.0, "latent_wait_s": 0.1}}}}
    w = CacheAnalysisWorker(StubCache(), StubSim(), config=cfg)

    assert await _try_admit(w, frame, 10, unit(1)) is True
    w._last_admission_ts = __import__("asyncio").get_event_loop().time()
    # kf-less, latent-less submission right after -> time floor rejects
    ok, _ = await w._analyze_frame_diversity(
        {"path": frame, "metadata": {}, "latent_vec": None})
    assert ok is False
    assert w.frames_rejected_floor >= 1
