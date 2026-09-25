"""Tests for VAE weight selection (utils/vae_source.py, 2026-09-25)."""

import os
from pathlib import Path

import yaml

from utils.vae_source import vae_source

CONFIG = Path(__file__).resolve().parents[1] / "config.b200.yaml"


def test_stock_when_unset():
    assert vae_source({}) == (None, None)
    assert vae_source(None) == (None, None)
    assert vae_source({"generation": {"vae": {"model": None}}})[0] is None


def test_model_and_expanded_cache_dir():
    model, cache = vae_source({"generation": {"vae": {
        "model": "stabilityai/sd-vae-ft-ema", "cache_dir": "~/luxi-files/model-cache"}}})
    assert model == "stabilityai/sd-vae-ft-ema"
    assert cache == os.path.expanduser("~/luxi-files/model-cache")


def test_malformed_config_falls_back_to_stock():
    assert vae_source({"generation": "nonsense"}) == (None, None)
    assert vae_source({"generation": {"vae": ["x"]}}) == (None, None)


def test_shipping_config_parses():
    model, _ = vae_source(yaml.safe_load(open(CONFIG)))
    assert model in (None, "stabilityai/sd-vae-ft-ema")
