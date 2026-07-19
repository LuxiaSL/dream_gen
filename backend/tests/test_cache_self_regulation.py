"""
Tests for the self-regulating cache (manager.py, 2026-07-19).

With a functional distance metric (cache/latent_pool.py) the cache maintains
its own diversity: redundancy eviction removes the older member of the
closest latent pair, TTL expiry keeps the cache tracking recent history, and
file handling is leak-free (evictions, expiry, fresh-start clears, and a
boot-time orphan sweep).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cache.manager import CacheManager


def unit(seed: int) -> np.ndarray:
    x = np.random.RandomState(seed).randn(128).astype(np.float32)
    return x / np.linalg.norm(x)


def emb(vec) -> dict:
    return {"color": [0.0] * 96, "struct": "ab" * 8,
            "latent": [float(x) for x in vec]}


def make_cfg(tmp: Path, max_size: int = 3, ttl: float = 0) -> dict:
    return {
        "system": {"cache_dir": str(tmp)},
        "generation": {"cache": {
            "max_size": max_size,
            "self_regulation": {"eviction": "redundancy",
                                "entry_ttl_minutes": ttl},
        }},
    }


@pytest.fixture
def src(tmp_path: Path) -> Path:
    p = tmp_path / "src.png"
    Image.new("RGB", (8, 8)).save(p)
    return p


def test_redundancy_eviction_removes_older_of_closest_pair(tmp_path, src):
    mgr = CacheManager(make_cfg(tmp_path, max_size=3))
    u1 = unit(1)
    id_a = mgr.add(src, "a", {}, emb(u1))
    id_b = mgr.add(src, "b", {}, emb(unit(2)))
    id_c = mgr.add(src, "c", {}, emb(unit(3)))
    a_file = mgr.entries[id_a].image_path

    near_u1 = u1 + 0.02 * unit(4)
    near_u1 /= np.linalg.norm(near_u1)
    id_d = mgr.add(src, "d", {}, emb(near_u1))  # closest pair (A, D); A older

    assert mgr.size() == 3
    assert id_a not in mgr.entries
    assert all(i in mgr.entries for i in (id_b, id_c, id_d))
    assert not a_file.exists()  # no file leak on eviction


def test_latent_less_entries_evict_first(tmp_path, src):
    mgr = CacheManager(make_cfg(tmp_path, max_size=2))
    id_nolat = mgr.add(src, "nolat", {}, {"color": [0.0] * 96, "struct": "ab" * 8})
    mgr.add(src, "x", {}, emb(unit(5)))
    mgr.add(src, "y", {}, emb(unit(6)))
    assert id_nolat not in mgr.entries


def test_ttl_sweep_expires_and_unlinks(tmp_path, src):
    mgr = CacheManager(make_cfg(tmp_path, max_size=10, ttl=30))
    id_old = mgr.add(src, "old", {}, emb(unit(7)))
    old_file = mgr.entries[id_old].image_path
    mgr.entries[id_old].timestamp = (
        datetime.now() - timedelta(minutes=60)
    ).isoformat()

    id_new = mgr.add(src, "new", {}, emb(unit(8)))  # add() triggers the sweep

    assert id_old not in mgr.entries
    assert not old_file.exists()
    assert id_new in mgr.entries


def test_orphan_sweep_on_load(tmp_path, src):
    cfg = make_cfg(tmp_path, max_size=10)
    mgr = CacheManager(cfg)
    id_k = mgr.add(src, "keep", {}, emb(unit(9)))
    kept_file = mgr.entries[id_k].image_path
    stray = mgr.image_dir / "stray_orphan.png"
    Image.new("RGB", (8, 8)).save(stray)

    mgr2 = CacheManager(cfg)  # load_cache runs the sweep

    assert not stray.exists()
    assert kept_file.exists()
    assert id_k in mgr2.entries


def test_fresh_start_switch_deletes_files(tmp_path, src):
    mgr = CacheManager(make_cfg(tmp_path, max_size=10))
    mgr.add(src, "p", {}, emb(unit(10)))
    mgr.add(src, "q", {}, emb(unit(11)))
    files = [e.image_path for e in mgr.entries.values()]

    # no current template id -> archive skipped -> fresh-start branch
    res = mgr.switch_template("brand_new_template")

    assert res["archived"] is False
    assert mgr.size() == 0
    assert not any(f.exists() for f in files)


def test_ttl_disabled_by_default(tmp_path, src):
    mgr = CacheManager(make_cfg(tmp_path, max_size=10, ttl=0))
    id_old = mgr.add(src, "old", {}, emb(unit(12)))
    mgr.entries[id_old].timestamp = (
        datetime.now() - timedelta(days=30)
    ).isoformat()
    mgr.add(src, "new", {}, emb(unit(13)))
    assert id_old in mgr.entries  # ttl 0 = never expire
