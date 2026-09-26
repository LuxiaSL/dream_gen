#!/usr/bin/env python3
"""
Injection lab — how should a recalled memory enter the present?

    python experiments/model_lab/inject_lab.py <pairs.json> <out_dir>

Production today (cache/injection_strategy.py): the injected keyframe is a
LINEAR latent mix, 35% memory + 65% present, decoded as-is. A latent mix of
two different images decodes as a double exposure; it then becomes the
anchor, and the viewer watches a 150-frame slerp glide into that ghost.

This lab renders, for real (present, memory) pairs captured from the dream,
the candidates for a true merge ("(img+img)2img"):

  blend        production: decode(0.35 zM + 0.65 zP)
  redream      the blend re-dreamed: img2img at 0.55 with the present prompt
               (the UNet resolves the ghosting into one scene)
  redream+txt  as redream, with the prompt embedding also mixed 65/35 toward
               the memory's own prompt (both images AND both sets of words)
  ip           IP-Adapter: img2img from the present at 0.60, conditioned on
               the memory as an image prompt (present shape, memory meaning)
  ip+blend     IP-Adapter on the blend: img2img at 0.50 from the blend,
               conditioned on the memory
  ip+palette   as ip, but the present is first recolored toward the memory
               (utils/palette.py, chroma 0.6 / luma 0.2): IP carries substance,
               not color, so this brings the memory's palette too
  ip-plus      as ip, with the finer-grained IP-Adapter Plus
  ip-plus.8    ip-plus with the image prompt at 0.8 (memory dominant)

Pairs are chosen like production's latent selection: each present frame is
matched with the same-era memory farthest from it in pooled-latent space
(>= 0.20). Settings mirror production: euler karras, 10 steps, cfg 7,
1024x512. Outputs: one sheet per pair, an overview sheet, glide videos
(present -> injected keyframe, 150 frames at 30 fps, as the viewer sees it),
and metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import (CFG, STEPS, decode, encode, hf_energy, load_pipes, log,  # noqa: E402
                 magenta_share, negative_for, prompt_system, slerp)

from cache.latent_pool import cosine_dist, pool_latent  # noqa: E402  (backend/ is on sys.path via lab)
from utils.palette import transfer_palette  # noqa: E402

IP_REPO = "h94/IP-Adapter"
IP_CACHE = Path("/mnt/beegfs/luxia/hf-cache")
W, H = 1024, 512
BLEND = 0.35           # production generation.cache.blend_weight
MIN_DIST = 0.20        # production latent_admission.min_dist
GLIDE_FRAMES = 150     # production hybrid.injection_interpolation_frames
MAX_PAIRS = 18
GLIDE_PAIRS = 6

METHODS = ["blend", "redream", "redream+txt", "ip", "ip+blend", "ip+palette", "ip-plus", "ip-plus.8", "ip-plus.8+palette"]
GLIDE_METHODS = ["blend", "ip+palette", "ip-plus.8", "ip-plus.8+palette"]


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def squash(img: Image.Image) -> Image.Image:
    """Whole frame into a square: CLIP's processor would center-crop a 2:1
    frame and drop half the memory."""
    return img.resize((512, 512), Image.BICUBIC)


def build_pairs(spec: dict, vae) -> list[dict]:
    anchors, memories = spec["anchors"], spec["memories"]
    root = Path(spec["root"])
    ps = prompt_system()
    for a in anchors:  # production negatives come from the template + components
        if not a.get("negative"):
            a["negative"] = negative_for(ps, {"template_id": a["template"],
                                              "components": a.get("components") or {}})
    vec = {}
    for item in anchors + memories:
        img = Image.open(root / item["file"]).convert("RGB").resize((W, H))
        item["_img"] = img
        vec[item["file"]] = pool_latent(encode(vae, img))
    # Round-robin across eras (templates), anchors spread evenly through each
    # era, so every era is represented and consecutive pairs differ in era.
    by_era: dict[str, list[dict]] = {}
    for a in anchors:
        by_era.setdefault(a["template"], []).append(a)
    per_era = max(1, MAX_PAIRS // max(1, len(by_era)))
    queues = []
    for era_anchors in by_era.values():
        n = len(era_anchors)
        idx = sorted({round(i * (n - 1) / max(1, per_era - 1)) for i in range(min(per_era, n))})
        queues.append([era_anchors[i] for i in idx])
    order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]

    pairs, used = [], set()
    for a in order:
        cands = [(cosine_dist(vec[a["file"]], vec[m["file"]]), m) for m in memories
                 if m["template"] == a["template"]]
        cands = [(d, m) for d, m in cands if d >= MIN_DIST]
        if not cands:
            continue
        cands.sort(key=lambda x: (x[1]["file"] in used, -x[0]))  # prefer unused, then farthest
        d, m = cands[0]
        used.add(m["file"])
        pairs.append({"present": a, "memory": m, "dist": float(d)})
        if len(pairs) >= MAX_PAIRS:
            break
    from collections import Counter
    log(f"{len(pairs)} pairs from {len(anchors)} anchors x {len(memories)} memories: "
        f"{dict(Counter(p['present']['template'] for p in pairs))}")
    return pairs


@torch.no_grad()
def mixed_embeds(pipe, p_prompt: str, m_prompt: str, negative: str, w: float):
    pe, ne = pipe.encode_prompt(p_prompt, "cuda", 1, True, negative)
    me, _ = pipe.encode_prompt(m_prompt, "cuda", 1, True, negative)
    return (1 - w) * pe + w * me, ne


def i2i_kwargs(seed: int) -> dict:
    return dict(num_inference_steps=STEPS, guidance_scale=CFG,
                generator=torch.Generator("cuda").manual_seed(seed))


def run_plain(i2i, vae, pair: dict, seed: int) -> dict[str, Image.Image]:
    P, M = pair["present"], pair["memory"]
    zP, zM = encode(vae, P["_img"]), encode(vae, M["_img"])
    blend = decode(vae, BLEND * zM + (1 - BLEND) * zP)
    out = {"blend": blend}
    out["redream"] = i2i(P["prompt"], negative_prompt=P["negative"], image=blend,
                         strength=0.55, **i2i_kwargs(seed)).images[0]
    pe, ne = mixed_embeds(i2i, P["prompt"], M["prompt"], P["negative"], BLEND)
    out["redream+txt"] = i2i(prompt_embeds=pe, negative_prompt_embeds=ne, image=blend,
                             strength=0.55, **i2i_kwargs(seed)).images[0]
    return out


def run_ip(i2i, pair: dict, blend: Image.Image, seed: int, tag: str) -> dict[str, Image.Image]:
    P, M = pair["present"], pair["memory"]
    mem = squash(M["_img"])
    out = {}
    if tag == "ip":
        i2i.set_ip_adapter_scale(0.6)
        out["ip"] = i2i(P["prompt"], negative_prompt=P["negative"], image=P["_img"],
                        ip_adapter_image=mem, strength=0.60, **i2i_kwargs(seed)).images[0]
        i2i.set_ip_adapter_scale(0.5)
        out["ip+blend"] = i2i(P["prompt"], negative_prompt=P["negative"], image=blend,
                              ip_adapter_image=mem, strength=0.50, **i2i_kwargs(seed)).images[0]
        recolored = transfer_palette(P["_img"], M["_img"], 0.6, 0.2)
        i2i.set_ip_adapter_scale(0.6)
        out["ip+palette"] = i2i(P["prompt"], negative_prompt=P["negative"], image=recolored,
                                ip_adapter_image=mem, strength=0.60, **i2i_kwargs(seed)).images[0]
    else:
        for scale, name in [(0.6, "ip-plus"), (0.8, "ip-plus.8")]:
            i2i.set_ip_adapter_scale(scale)
            out[name] = i2i(P["prompt"], negative_prompt=P["negative"], image=P["_img"],
                            ip_adapter_image=mem, strength=0.60, **i2i_kwargs(seed)).images[0]
        # the two finalists combined: memory's palette on the init, IP-Plus at 0.8
        recolored = transfer_palette(P["_img"], M["_img"], 0.6, 0.2)
        i2i.set_ip_adapter_scale(0.8)
        out["ip-plus.8+palette"] = i2i(P["prompt"], negative_prompt=P["negative"], image=recolored,
                                       ip_adapter_image=mem, strength=0.60, **i2i_kwargs(seed)).images[0]
    return out


def metrics(vae, pair: dict, img: Image.Image) -> dict:
    v = pool_latent(encode(vae, img))
    return {
        "d_present": round(float(cosine_dist(v, pool_latent(encode(vae, pair["present"]["_img"])))), 3),
        "d_memory": round(float(cosine_dist(v, pool_latent(encode(vae, pair["memory"]["_img"])))), 3),
        "hf": round(hf_energy(img), 4),
        "magenta": round(magenta_share(img), 3),
    }


def pair_sheet(pair: dict, imgs: dict, mets: dict, path: Path) -> None:
    tw, th, lab_h = 400, 200, 34
    cells = [("present", pair["present"]["_img"], None), ("memory", pair["memory"]["_img"], None)]
    cells += [(k, imgs[k], mets[k]) for k in METHODS if k in imgs]
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    head = 52
    sheet = Image.new("RGB", (cols * tw, head + rows * (th + lab_h)), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    f, fs = font(13), font(11)
    d.text((6, 4), f"present: {pair['present']['prompt'][:170]}", fill=(210, 210, 210), font=fs)
    d.text((6, 20), f"memory:  {pair['memory']['prompt'][:170]}", fill=(210, 190, 150), font=fs)
    d.text((6, 36), f"{pair['present']['template']}  latent distance {pair['dist']:.2f}", fill=(150, 150, 150), font=fs)
    for i, (name, img, m) in enumerate(cells):
        x, y = (i % cols) * tw, head + (i // cols) * (th + lab_h)
        sheet.paste(img.resize((tw, th)), (x, y))
        d.text((x + 4, y + th + 2), name, fill=(240, 240, 240), font=f)
        if m:
            d.text((x + 4, y + th + 18), f"d(P) {m['d_present']:.2f}  d(M) {m['d_memory']:.2f}  hf {m['hf']:.3f}",
                   fill=(150, 170, 200), font=fs)
    sheet.save(path, quality=90)


@torch.no_grad()
def glide_video(vae, pair: dict, imgs: dict, path: Path) -> None:
    import av
    zP = encode(vae, pair["present"]["_img"])
    targets = {k: encode(vae, imgs[k]) for k in GLIDE_METHODS}
    cw, ch, lab_h = 512, 256, 22
    cols = 3
    names = GLIDE_METHODS
    rows = (len(names) + 1 + cols - 1) // cols
    fw, fh = cols * cw, rows * (ch + lab_h)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = fw, fh, "yuv420p"
    stream.options = {"crf": "18"}
    f = font(14)
    mem_small = pair["memory"]["_img"].resize((cw, ch))
    hold = 30  # 1 s on the landed keyframe
    for i in range(GLIDE_FRAMES + hold):
        t = min(1.0, i / (GLIDE_FRAMES - 1))
        frame = Image.new("RGB", (fw, fh), (10, 10, 12))
        d = ImageDraw.Draw(frame)
        for j, k in enumerate(names):
            img = decode(vae, slerp(zP, targets[k], t)).resize((cw, ch))
            x, y = (j % cols) * cw, (j // cols) * (ch + lab_h)
            frame.paste(img, (x, y))
            d.text((x + 6, y + ch + 3), k, fill=(235, 235, 235), font=f)
        j = len(names)
        x, y = (j % cols) * cw, (j // cols) * (ch + lab_h)
        frame.paste(mem_small, (x, y))
        d.text((x + 6, y + ch + 3), "the memory (reference)", fill=(210, 190, 150), font=f)
        for packet in stream.encode(av.VideoFrame.from_image(frame)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def overview(pairs: list[dict], all_imgs: list[dict], path: Path) -> None:
    tw, th = 256, 128
    cols = ["present", "memory"] + METHODS
    head = 24
    sheet = Image.new("RGB", (len(cols) * tw, head + len(pairs) * th), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    f = font(13)
    for c, name in enumerate(cols):
        d.text((c * tw + 4, 4), name, fill=(235, 235, 235), font=f)
    for r, (pair, imgs) in enumerate(zip(pairs, all_imgs)):
        row = {"present": pair["present"]["_img"], "memory": pair["memory"]["_img"], **imgs}
        for c, name in enumerate(cols):
            if name in row:
                sheet.paste(row[name].resize((tw, th)), (c * tw, head + r * th))
    sheet.save(path, quality=88)


def main() -> None:
    spec = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, i2i = load_pipes()
    vae = i2i.vae
    pairs = build_pairs(spec, vae)
    if not pairs:
        raise SystemExit("no pairs at >= min latent distance; nothing to compare")
    seeds = [1000 + i for i in range(len(pairs))]

    all_imgs = [run_plain(i2i, vae, p, s) for p, s in zip(pairs, seeds)]
    log(f"plain methods done ({time.time() - t0:.0f}s)")

    for weight, tag in [("ip-adapter_sd15.safetensors", "ip"), ("ip-adapter-plus_sd15.safetensors", "ip-plus")]:
        try:
            i2i.load_ip_adapter(IP_REPO, subfolder="models", weight_name=weight, cache_dir=IP_CACHE)
            for p, s, imgs in zip(pairs, seeds, all_imgs):
                imgs.update(run_ip(i2i, p, imgs["blend"], s, tag))
            i2i.unload_ip_adapter()
            log(f"{tag} done ({time.time() - t0:.0f}s)")
        except Exception as e:
            log(f"{tag} unavailable: {e!r}")

    report = []
    for i, (p, imgs) in enumerate(zip(pairs, all_imgs)):
        mets = {k: metrics(vae, p, img) for k, img in imgs.items()}
        pair_sheet(p, imgs, mets, out / f"pair_{i:02d}.jpg")
        for k, img in imgs.items():
            img.save(out / f"pair_{i:02d}_{k.replace('+', '_')}.jpg", quality=92)
        report.append({"pair": i, "template": p["present"]["template"], "dist": p["dist"],
                       "present": p["present"]["file"], "memory": p["memory"]["file"],
                       "present_prompt": p["present"]["prompt"], "memory_prompt": p["memory"]["prompt"],
                       "metrics": mets})
    json.dump(report, open(out / "metrics.json", "w"), indent=1)
    overview(pairs, all_imgs, out / "overview.jpg")
    log(f"sheets done ({time.time() - t0:.0f}s)")

    for i in range(min(GLIDE_PAIRS, len(pairs))):
        glide_video(vae, pairs[i], all_imgs[i], out / f"glide_{i:02d}.mp4")
        log(f"glide {i} done ({time.time() - t0:.0f}s)")
    log(f"inject_lab: finished OK, {len(pairs)} pairs in {out}")


if __name__ == "__main__":
    main()
