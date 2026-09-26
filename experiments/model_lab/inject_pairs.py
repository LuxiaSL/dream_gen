#!/usr/bin/env python3
"""
Label captured anchors and cache memories with their prompt and template
(from the chronicle export), producing the pairs.json inject_lab.py reads.

    python3 -I inject_pairs.py <raw_dir> <since_id> <out pairs.json> [remote_root]

<raw_dir> is a local copy of pairs_raw/ (anchors/, cache/, index/ — images
may be absent: only file names and the cache index JSONs are read).
Anchors are named anchor_<session kf>_<unix ts>.png; memories carry their
keyframe_num and prompt in the cache index snapshots. Each file is matched
to the chronicle record with the same session keyframe whose timestamp is
closest to the file's (sessions restart keyframe numbering, the clock
tells them apart).
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

EXPORT = "https://aetherawi.red/api/dreams/chronicle/export"
MAX_SKEW_S = 180


def fetch_records(since_id: int) -> list[dict]:
    out = []
    while True:
        url = f"{EXPORT}?since_id={since_id}&limit=2000&include_embeddings=false"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        if not page.get("count"):
            return out
        out.extend(page["records"])
        since_id = page["next_since_id"]


def unix(ts: str) -> float:
    from datetime import datetime, timezone
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp()


def main():
    raw, since_id, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    remote_root = sys.argv[4] if len(sys.argv) > 4 else str(raw)
    recs = fetch_records(since_id)
    by_kf: dict[int, list[dict]] = {}
    for r in recs:
        r["_t"] = unix(r["ts"])
        by_kf.setdefault(r["keyframe"], []).append(r)
    print(f"{len(recs)} chronicle records")

    def match(kf: int, t: float):
        cands = [r for r in by_kf.get(kf, []) if abs(r["_t"] - t) < MAX_SKEW_S]
        return min(cands, key=lambda r: abs(r["_t"] - t)) if cands else None

    anchors = []
    for f in sorted((raw / "anchors").iterdir()):
        m = re.match(r"anchor_(\d+)_(\d+)\.png", f.name)
        if not m:
            continue
        r = match(int(m.group(1)), float(m.group(2)))
        if r is None or r["prompt"] in ("", "injected"):
            continue
        anchors.append({"file": f"anchors/{f.name}", "kf": r["keyframe"], "session": r["session_id"][:8],
                        "prompt": r["prompt"], "template": r["template_id"], "components": r["components"]})

    entries = {}
    for idx in sorted((raw / "index").glob("*.json")):
        for e in json.load(open(idx)).get("entries", []):
            entries[Path(e["image_path"]).name] = e
    memories = []
    for f in sorted((raw / "cache").iterdir()):
        e = entries.get(f.name)
        m = re.match(r"cache_\d+_(\d+)\.png", f.name)
        if e is None or not m:
            continue
        kf = e.get("generation_params", {}).get("keyframe_num")
        r = match(kf, float(m.group(1))) if isinstance(kf, int) else None
        if r is None:
            continue
        memories.append({"file": f"cache/{f.name}", "kf": kf, "session": r["session_id"][:8],
                         "prompt": e.get("prompt") or r["prompt"], "template": r["template_id"]})

    json.dump({"root": remote_root, "anchors": anchors, "memories": memories}, open(out, "w"), indent=1)
    from collections import Counter
    print(f"{len(anchors)} anchors {dict(Counter(a['template'] for a in anchors))}")
    print(f"{len(memories)} memories {dict(Counter(m['template'] for m in memories))}")


if __name__ == "__main__":
    main()
