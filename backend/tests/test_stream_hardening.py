"""
Stream hardening (2026-09-25): console logging that stays small under Heimdall,
and full-VAE decodes that never hand the GPU a whole swap glide at once.
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fresh_root():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    for h in list(root.handlers):
        root.removeHandler(h)
    return root, saved


def _restore(root, saved):
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved[0]:
        root.addHandler(h)
    root.setLevel(saved[1])


def test_setup_logging_replaces_bootstrap_handler_and_honours_console_level(tmp_path, capsys):
    from core.dream_controller import setup_logging

    root, saved = _fresh_root()
    try:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # what heimdall_entry does first
        setup_logging(tmp_path, "INFO", "WARNING")
        assert len(root.handlers) == 2, "bootstrap handler must be removed"

        logging.getLogger("core.display_selector").debug("debug line")
        logging.getLogger("core.display_selector").info("chatty info line")
        logging.getLogger("core.async_orchestrator").warning("a real warning")
        logging.getLogger("heartbeat").info("alive: kf 1")
        for h in root.handlers:
            h.flush()
        out = capsys.readouterr()
        console = out.out + out.err
        assert "chatty info line" not in console
        assert "debug line" not in console
        assert console.count("a real warning") == 1, "warnings once, never duplicated"
        assert "alive: kf 1" in console, "heartbeat passes a WARNING console"
        text = (tmp_path / "dream_controller.log").read_text()
        assert "debug line" in text and "chatty info line" in text, "file keeps full detail"
    finally:
        _restore(root, saved)


def test_setup_logging_default_console_is_info(tmp_path, capsys):
    from core.dream_controller import setup_logging

    root, saved = _fresh_root()
    try:
        setup_logging(tmp_path, "INFO")
        logging.getLogger("x").info("info shows by default")
        for h in root.handlers:
            h.flush()
        assert "info shows by default" in capsys.readouterr().out
    finally:
        _restore(root, saved)


def test_full_vae_batch_decode_is_chunked():
    torch = pytest.importorskip("torch")
    from interpolation.latent_encoder import LatentEncoder

    class FakeVAE:
        def __init__(self):
            self.calls = []

        def decode(self, z):
            self.calls.append(z.shape[0])
            n, _, h, w = z.shape
            # stand-in pixels: each frame carries its latent's first value, so order is checkable
            return type("Out", (), {"sample": z[:, :1].repeat(1, 3, 8, 8).clamp(-1, 1)})()

    enc = LatentEncoder(auto_load=False, device="cpu")
    enc.vae = FakeVAE()
    enc.vae_scale_factor = 1.0
    enc.max_decode_batch = 24
    lat = torch.linspace(-0.9, 0.9, 50).view(50, 1, 1, 1).repeat(1, 4, 2, 2)
    imgs = enc._decode_batch_vae(lat, return_numpy=True)
    assert enc.vae.calls == [24, 24, 2]
    assert len(imgs) == 50
    firsts = [int(im[0, 0, 0]) for im in imgs]
    assert firsts == sorted(firsts), "frames come back in order"
