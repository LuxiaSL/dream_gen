#!/usr/bin/env python3
"""
Dreamgen model lab — measurement experiments run as Heimdall jobs on node1.

    python experiments/model_lab/lab.py dupe   # resolution -> duplicated-subject test
    python experiments/model_lab/lab.py vae    # decoder study: color bias, sharpness,
                                               # loopback drift, interpolation decodes

Run from the repo root. Outputs go to ~/luxi-files/dreamgen-lab/ (outside the
repo). SD 1.5 is read from the existing HF cache with local_files_only; the
only downloads (Stability's fine-tuned decoders) land under the lab dir.
Production settings are mirrored: euler + karras, 10 steps, cfg 7.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from prompts.combinatorial import CombinatorialPromptSystem  # noqa: E402
from utils.palette import rgb_to_lab  # noqa: E402

LAB = Path.home() / "luxi-files" / "dreamgen-lab"
HF_READ = Path.home() / ".cache" / "huggingface" / "hub"
HF_LAB = LAB / "hf-cache"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DECODERS = {"stock": None, "ft-ema": "stabilityai/sd-vae-ft-ema", "ft-mse": "stabilityai/sd-vae-ft-mse"}
STEPS, CFG = 10, 7.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pipes():
    from diffusers import (EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline,
                           StableDiffusionPipeline)
    t2i = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, cache_dir=HF_READ, local_files_only=True,
    ).to("cuda")
    t2i.scheduler = EulerDiscreteScheduler.from_config(t2i.scheduler.config, use_karras_sigmas=True)
    t2i.set_progress_bar_config(disable=True)
    i2i = StableDiffusionImg2ImgPipeline(**t2i.components)
    i2i.set_progress_bar_config(disable=True)
    return t2i, i2i


def prompt_system():
    cfg = yaml.safe_load(open(REPO / "backend" / "config.b200.yaml"))
    return CombinatorialPromptSystem(config=cfg)


def negative_for(ps, spec) -> str:
    try:
        ps.switch_template(spec["template_id"], spec["components"])
        return ps.get_negative_prompt()
    except Exception:
        return "blurry, low quality, text, watermark"


def magenta_share(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("RGB").convert("HSV"))
    h = np.histogram(hsv[:, :, 0], bins=32, range=(0, 256))[0] / hsv[:, :, 0].size
    return float(h[23:32].sum())


def lab_mean(img: Image.Image) -> np.ndarray:
    return rgb_to_lab(np.asarray(img.convert("RGB").resize((256, 128)))).reshape(-1, 3).mean(0)


def hf_energy(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


# --------------------------------------------------------------- dupe test
def run_dupe() -> None:
    out = LAB / "dupe"
    out.mkdir(parents=True, exist_ok=True)
    specs = json.load(open(REPO / "experiments" / "model_lab" / "prompts.json"))
    ps = prompt_system()
    t2i, _ = load_pipes()
    manifest = []
    for i, s in enumerate(specs):
        neg = negative_for(ps, s)
        for w, h in [(512, 512), (768, 384), (1024, 512)]:
            g = torch.Generator("cuda").manual_seed(s["seed"])
            img = t2i(s["prompt"], negative_prompt=neg, width=w, height=h,
                      num_inference_steps=STEPS, guidance_scale=CFG, generator=g).images[0]
            name = f"{i:02d}_{w}x{h}.jpg"
            img.save(out / name, quality=92)
            manifest.append({"i": i, "res": f"{w}x{h}", "file": name,
                             "template": s["template_id"], "prompt": s["prompt"]})
        log(f"dupe {i + 1}/{len(specs)}")
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    log(f"done: {len(manifest)} images in {out}")


# --------------------------------------------------------------- vae study
def load_decoders(stock_vae):
    from diffusers import AutoencoderKL
    decs = {"stock": stock_vae}
    for name, repo in DECODERS.items():
        if repo is None:
            continue
        try:
            decs[name] = AutoencoderKL.from_pretrained(
                repo, torch_dtype=torch.float16, cache_dir=HF_LAB, use_safetensors=True,
            ).to("cuda").eval()
            log(f"loaded decoder {name}")
        except Exception as e:
            log(f"decoder {name} unavailable: {e!r}")
    return decs


@torch.no_grad()
def encode(vae, img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1)[None].half().cuda()
    x = x / 127.5 - 1.0
    return vae.encode(x).latent_dist.mean * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, z: torch.Tensor) -> Image.Image:
    x = vae.decode(z / vae.config.scaling_factor).sample
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    af, bf = a.float().flatten(), b.float().flatten()
    omega = torch.acos(torch.clamp((af / af.norm()) @ (bf / bf.norm()), -1, 1))
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    so = torch.sin(omega)
    return ((torch.sin((1 - t) * omega) / so) * a.float() + (torch.sin(t * omega) / so) * b.float()).half()


def run_vae() -> None:
    out = LAB / "vae"
    out.mkdir(parents=True, exist_ok=True)
    dupe = LAB / "dupe"
    manifest = json.load(open(dupe / "manifest.json"))
    wide = [m for m in manifest if m["res"] == "1024x512"]
    imgs = [Image.open(dupe / m["file"]).convert("RGB") for m in wide]
    t2i, i2i = load_pipes()
    stock = t2i.vae
    decs = load_decoders(stock)
    results: dict = {"decoders": list(decs)}

    # A. single + repeated round trips (encoder is shared: Stability tuned decoders only)
    log("A: round-trip color bias / sharpness")
    rt = {}
    for name, dec in decs.items():
        dL, da, db, dmc, sharp, psnr = [], [], [], [], [], []
        for img in imgs:
            rec = decode(dec, encode(stock, img))
            d = lab_mean(rec) - lab_mean(img)
            dL.append(d[0]); da.append(d[1]); db.append(d[2])
            dmc.append(magenta_share(rec) - magenta_share(img))
            sharp.append(hf_energy(rec) / max(hf_energy(img), 1e-6))
            mse = np.mean((np.asarray(rec, np.float32) - np.asarray(img, np.float32)) ** 2)
            psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
        chain = []
        for img in imgs[:8]:
            x, traj = img, []
            for _ in range(10):
                x = decode(dec, encode(stock, x))
                traj.append([magenta_share(x), *lab_mean(x).tolist(), hf_energy(x)])
            chain.append(traj)
        c = np.array(chain)
        rt[name] = {
            "single": {"dL": float(np.mean(dL)), "da": float(np.mean(da)), "db": float(np.mean(db)),
                       "dmagenta": float(np.mean(dmc)), "sharp_ratio": float(np.mean(sharp)),
                       "psnr": float(np.mean(psnr))},
            "x10_mean_traj": c.mean(0).tolist(),  # [magenta, L, a, b, hf] per round trip
        }
        log(f"  {name}: {rt[name]['single']}")
    results["roundtrip"] = rt

    # B. loopback drift with the UNet: 40 re-anchor-strength img2img steps, no grounding
    log("B: loopback drift chains (denoise 0.70, 40 steps)")
    starts = sorted(range(len(wide)), key=lambda k: magenta_share(imgs[k]))[:6]
    ps = prompt_system()
    specs = json.load(open(REPO / "experiments" / "model_lab" / "prompts.json"))
    lb = {}
    for name, dec in decs.items():
        i2i.vae = dec
        trajs = []
        for k in starts:
            spec = wide[k]
            neg = negative_for(ps, specs[spec["i"]])
            x, traj = imgs[k], [[magenta_share(imgs[k]), *lab_mean(imgs[k]).tolist()]]
            for step in range(40):
                g = torch.Generator("cuda").manual_seed(7000 + 100 * k + step)
                x = i2i(spec["prompt"], image=x, strength=0.70, negative_prompt=neg,
                        num_inference_steps=STEPS, guidance_scale=CFG, generator=g).images[0]
                traj.append([magenta_share(x), *lab_mean(x).tolist()])
            trajs.append(traj)
            x.save(out / f"loopback_{name}_{k:02d}_step40.jpg", quality=90)
        t = np.array(trajs)
        lb[name] = {"mean_traj": t.mean(0).tolist(), "start_idx": starts}
        log(f"  {name}: magenta {t[:, 0, 0].mean():.3f} -> {t[:, -1, 0].mean():.3f}")
    i2i.vae = stock
    results["loopback"] = lb

    # C. interpolation decodes: slerp midpoints between keyframe-like pairs
    log("C: interpolation decode sheets")
    pairs = [(0, 1), (5, 17), (22, 30), (40, 45)]
    rows = []
    for a, b in pairs:
        za, zb = encode(stock, imgs[a]), encode(stock, imgs[b])
        row = []
        for name, dec in decs.items():
            for t in (0.25, 0.5, 0.75):
                row.append((name, t, decode(dec, slerp(za, zb, t)).resize((512, 256))))
        rows.append(row)
    W, H = 512, 256
    cols = len(rows[0])
    sheet = Image.new("RGB", (W * cols, H * len(rows)))
    for r, row in enumerate(rows):
        for c, (_, _, im) in enumerate(row):
            sheet.paste(im, (c * W, r * H))
    sheet.save(out / "interp_sheet.jpg", quality=90)
    results["interp_sheet_columns"] = [f"{n}@{t}" for n, t, _ in rows[0]]
    for name, dec in decs.items():  # full-res midpoint crops for sharpness review
        decode(dec, slerp(encode(stock, imgs[5]), encode(stock, imgs[17]), 0.5)).save(
            out / f"midpoint_full_{name}.png")

    json.dump(results, open(out / "results.json", "w"), indent=1)
    log(f"done -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"dupe": run_dupe, "vae": run_vae}.get(cmd, lambda: sys.exit(f"usage: lab.py dupe|vae (got {cmd!r})"))()
