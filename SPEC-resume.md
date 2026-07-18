# Spec: Save/Resume — the dream picks up where it left off

> **Status:** Implemented with Phase 1.5 of the chronicle
> **Date:** 2026-07-17
> **Companion:** SPEC-chronicle.md

## Problem

Every restart reboots the dream from a fresh txt2img seed: keyframe counters
reset to 1, the aesthetic thread is cut, and the chronicle sees an artificial
hard boundary at every reboot. Generation is intermittent (no dedicated
machine), so over the lifelong chronicle these cuts dominate: eras that
should span a shutdown get split, and "keyframe 525" says nothing about how
long the dream has actually lived.

## Design: local-first checkpoint + lifetime epoch

The RunPod-era state round-trip (GPU→VPS→GPU via SCP/CTRL_LOAD_STATE) is
obsolete — node1's disk persists between runs. Resume is therefore
**GPU-local**: a small checkpoint under `output/resume/`, written
periodically and at shutdown, loaded at boot if compatible.

### What persists (per checkpoint, ~1.5MB)

| Item | Purpose |
|---|---|
| `last_keyframe.png` (full res) | Visual continuity: next session's first keyframe is img2img'd from it at drift denoise — the dream continues mid-thought |
| Prompt state: template_id + components + mutation counters | The combinatorial system resumes the same aesthetic sentence it was speaking |
| `lifetime_keyframes` epoch | Continuous numbering across sessions |
| Previous chronicle `session_id` | Session stitching in the chronicle |
| Config fingerprint (resolution, model) | Compatibility gate — mismatch → clean fresh start |

Deliberately **not** persisted: frame buffer, interpolation state, H.264
encoder state (ephemeral; refill in seconds), BEND mode (resume in DRIFT),
collapse-detector histories (re-baseline is safer than stale history).
The injection cache already persists on node1 disk (startup sweep leaves
`cache/` untouched).

### Boot behavior

```
load_resume_state(resume_dir, config)
  ├─ no checkpoint / corrupt / fingerprint mismatch / image missing
  │    → None → fresh start (fresh-buffer bootstrap, epoch 0,
  │      chronicle emits session_start)          [never crashes the boot]
  └─ valid
       → orchestrator bootstraps from last_keyframe.png,
         prompt_manager.switch_template(saved template, components)
         + counters restored, epoch = saved lifetime_keyframes,
         chronicle emits session_resume(detail=prev session_id)
```

Checkpoint cadence: every `resume.checkpoint_interval_keyframes` (default
50, ~40s) via atomic tmp+rename in the executor, plus a final save in
`orchestrator.stop()`. A crash loses at most one interval — acceptable.

### Chronicle integration

- `KeyframeRecord.lifetime_keyframe = epoch + local keyframe` on every
  record; the VPS stores it (`chronicle_keyframe.lifetime_keyframe`,
  nullable, guarded ALTER migration) and surfaces it in
  `/api/dreams/chronicle/current`.
- New event kind `session_resume` (fresh boots keep `session_start`).
  The Phase 2 segmenter treats `session_start` as a hard era boundary but
  `session_resume` as *continuity* — an era can span a shutdown, and the
  biographer can truthfully write "the dream slept mid-thought."
- The VPS chronicle DB remains the independent ledger: lifetime numbering
  is derivable from it (sum over sessions) as a cross-check on the local
  epoch; local epoch wins for display, ledger wins for audits.

### Known limitations (accepted)

- Multi-component template slots (`{category:N}`) restore only the primary
  component per category — same limitation as the existing fresh-buffer
  bootstrap path, drift re-diversifies within seconds.
- Local GPU keyframe numbering still starts at 1 per session (buffer/worker
  internals untouched); *lifetime* numbering lives in the chronicle records
  and the resume epoch. Display surfaces adopt lifetime numbers as they get
  touched (chronicle first).
- If the checkpoint is older than the cache retention horizon the resume
  image may look "ancient" relative to the last streamed frame (state saved
  every ~40s, so in practice: no).

## Config

```yaml
resume:
  enabled: true
  checkpoint_interval_keyframes: 50
```
