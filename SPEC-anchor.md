# Spec: Anchor Walking — wandering without dissolving

> **Status:** Implemented 2026-07-19 (same-day build with Luxia)
> **Revert point:** git tag `pre-anchor-walking`
> **Depends on:** latent-gated cache (e27d651), self-regulating cache (88188b0)

## Problem

The keyframe chain was a Markov random walk: every keyframe derived from its
predecessor at low denoise. One knob served two masters — low denoise
preserves the previous frame but cannot express the prompt (12–18% of the
schedule only adjusts texture); high denoise expresses the prompt but breaks
continuity. No value gives both, so the chain lost BOTH anchors: structure
eroded generation-by-generation into the model's own attractor (stripes,
mountains, magenta voids), while mutated components never got enough denoise
to matter. Interventions (injections/swaps) were transient nudges the chain
forgot within ~30 frames.

## Design

The lineage advances **anchor → anchor**, not frame → frame:

- **Anchor**: the era's reference image, kept as a stable file
  (`output/anchor/`, rotating pair since queued jobs may reference the
  previous one).
- **Drift keyframes** (most frames): chain locally from the previous frame at
  `drift` denoise, exactly as before — this is the visible motion. But the
  walk is *discarded from the lineage* at the next mutation: at most
  `mutation_interval` chained steps ever compound.
- **Mutation → re-anchor**: the post-mutation keyframe is generated **from
  the anchor** at `bend` denoise (0.60 — enough for the swapped components to
  genuinely restructure), and its result is **promoted to be the new
  anchor**. The lineage is a sequence of deliberate, detail-preserving,
  high-denoise steps between clean images.
- **Interventions set anchors** (durable, not nudges): template swaps set a
  fresh-frame anchor and resample the era noise seed (hard era boundary);
  cache injections promote the blended recall to anchor (memory becomes part
  of the lineage); resume/bootstrap images anchor their sessions.
- **Noise pinning**: one RNG seed per era passed to every generation;
  resampled only at swaps. Removes per-frame noise jitter from the walk.
- **Re-anchor glide**: mutation keyframe pairs get
  `mutation_interpolation_frames` (60, ~2s) instead of 20, smoothing the
  larger jump.
- `bend_frames` lingering is legacy-chained-mode only; under anchoring, bend
  applies to exactly the re-anchor keyframe.

Mush cannot compound: every era's quality is bounded by its anchor, and every
anchor is either a fresh txt2img frame, a curated cache recall, or a strong
regeneration of its predecessor-anchor.

## Companion cache tuning (same change)

- `blend_weight: 0.35` — recall suggests (~1/3); anchor promotion is what
  makes it durable now, the blend no longer needs to shout.
- `cache_injection_lockout_after_swap: 96` — two injection intervals of
  silence after each template swap, so a fresh era accumulates its own cache
  before recall can pull it back toward the previous era's archive.
- `min_dist` stays 0.20: ~3 admissions per 2000 analyzed is intended —
  restrictive over long horizons is the goal.

## Config (fresh_generation.anchor_walking)

```yaml
anchor_walking:
  enabled: true
  mutation_interpolation_frames: 60
  noise_pinning: true
```

`enabled: false` restores the pure chained behavior (tag
`pre-anchor-walking` restores the pre-anchor code entirely).

## Failure philosophy

Anchor bookkeeping never raises: a failed promotion keeps the old anchor; a
missing anchor file falls back to chained generation for that keyframe. The
worst failure mode is the previous status quo.

## What to watch

- Settle frames orbit the anchor instead of walking away from it; mutations
  visibly re-compose rather than smear.
- `[ANCHOR]` log lines at mutations/swaps/injections.
- Cache admissions should rise naturally (re-anchor jumps clear `min_dist`
  more often than chained drift did).
- Era-scale look coherence in the chronicle (the segmentation question
  changes shape under anchoring — eras get real boundaries for free).
