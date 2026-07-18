# Spec: The Chronicle — the dream's own memory

> **Status:** Draft for review
> **Author:** Claude (Fable), with Luxia
> **Date:** 2026-07-17
> **Depends on:** current `main` (b951d70) + uncommitted dual-embedding prompt system

## Problem

The dream generates ~112,000 keyframes a day and remembers none of them. Whole
aesthetic eras rise, evolve, and dissolve unobserved — the only records of what
the system has ever dreamed are two hand-written journal entries covering a
combined ~0.5% of its output. Yet the pipeline already computes everything a
memoir needs at the moment each keyframe is born: the prompt, the template, the
component words, the dual-metric embeddings, and the dramatic events (mutation,
cache injection, template switch). It throws all of it away.

The chronicle persists a thin record of that stream, segments it into named
eras, consolidates old raw data into durable narrative memory (with LLM help),
and renders it at `aetherawi.red/dreams/chronicle` — so the dream keeps its own
history and a visitor can ask "what did it do last night?"

## Goals

1. **Record**: persist a lightweight per-keyframe record + tiered thumbnails,
   with zero risk to the generation loop (fire-and-forget everywhere).
2. **Segment**: split history into eras using events + embedding trajectory.
3. **Consolidate**: when raw records age out, compress each era into a durable
   summary — mechanical stats + an LLM-written narrative memory.
4. **Narrate**: a nightly "biographer" writes a short diary entry for the day,
   in the tradition of `dream_watch_journal.md`.
5. **Serve**: `/dreams/chronicle` timeline page + JSON API.

## Non-goals

- Not a frame archive. We keep sparse thumbnails, not video. The H.264 stream
  remains ephemeral.
- Not real-time. Seconds-to-minutes of lag is fine everywhere.
- No new infrastructure. SQLite + files on the VPS, one new websocket message
  type, Claude API calls from the VPS only.

## Architecture

```
GPU instance (this repo)                VPS (aethera core)
┌─────────────────────────┐             ┌──────────────────────────────────┐
│ async_orchestrator      │             │ aethera/dreams/chronicle/        │
│   │ (after keyframe +   │             │   store.py     (sqlmodel, CRUD)  │
│   │  event bookkeeping) │             │   segmenter.py (era detection)   │
│   ▼                     │  ws 0x05    │   consolidate.py (aging + LLM)   │
│ backend/chronicle/      │ ──────────► │   biographer.py  (nightly LLM)   │
│   recorder.py           │  batched    │   routes: /dreams/chronicle      │
│   (record + thumbnail)  │  JSON+webp  │           /api/dreams/chronicle/*│
└─────────────────────────┘             │ data/chronicle.sqlite            │
                                        │ data/chronicle/thumbs/YYYYMMDD/  │
                                        └──────────────────────────────────┘
```

Division of labor: the GPU side only **observes and emits** (cheap, dumb,
crash-proof). All storage, segmentation, and LLM work happens on the VPS,
which is always on and already runs the FastAPI app.

---

## 1. Data model

Pydantic throughout (VPS side uses sqlmodel, which is pydantic + SQLAlchemy —
same stack as the blog's `blog.sqlite` / `irc.sqlite`; the chronicle gets its
own `chronicle.sqlite`, matching the one-DB-per-concern convention).

### 1.1 KeyframeRecord (wire format, GPU → VPS)

```python
# backend/chronicle/models.py  (GPU side, plain pydantic)
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChronicleEvent(BaseModel):
    """What happened at this keyframe, if anything."""
    kind: Literal["mutation", "forced_mutation", "cache_injection",
                  "seed_injection", "template_switch", "session_start",
                  "session_resume"]
    detail: str = ""          # e.g. "color_logic: 'verdigris' -> 'oxblood'"

class KeyframeRecord(BaseModel):
    """One keyframe's memoir entry. ~500 bytes without thumbnail."""
    session_id: str            # uuid per GPU boot, for resume stitching
    keyframe: int              # monotonic within session
    frame: int
    ts: float                  # unix epoch, GPU clock
    prompt: str
    negative: str = ""
    template_id: str
    components: dict[str, str] # {category: word}, primary component per cat
    events: list[ChronicleEvent] = Field(default_factory=list)
    # embeddings for segmentation (VPS never recomputes)
    color_hist: Optional[list[float]] = None   # 96-dim, rounded 4dp
    phash: Optional[str] = None                # hex string
    # thumbnail present only on sampled/event keyframes
    thumb_webp_b64: Optional[str] = None       # 256x128 webp, ~8-15KB
```

### 1.2 VPS tables (sqlmodel)

```python
# aethera/dreams/chronicle/models.py
class Keyframe(SQLModel, table=True):        # raw window, TTL ~14 days
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    keyframe: int
    frame: int
    ts: datetime = Field(index=True)
    prompt: str
    template_id: str = Field(index=True)
    components: str                          # JSON
    events: str                              # JSON list, "" if none
    color_hist: bytes | None                 # np.float32.tobytes(), 384B
    phash: str | None
    thumb_path: str | None                   # relative path if thumbnail kept
    era_id: int | None = Field(default=None, index=True)

class Era(SQLModel, table=True):             # kept forever
    id: int | None = Field(default=None, primary_key=True)
    started_at: datetime
    ended_at: datetime | None                # None = current era
    title: str                               # mechanical: "cupola · oxidized copper · charcoal"
    template_id: str
    dominant_components: str                 # JSON {category: word}
    keyframe_count: int = 0
    mutation_count: int = 0
    injection_count: int = 0
    boundary_kind: str                       # what opened it: template_switch | drift | injection | session_start
    exemplar_prompts: str                    # JSON, up to 5
    representative_thumbs: str               # JSON list of thumb paths, ≤10
    # filled at consolidation time:
    consolidated: bool = False
    narrative: str | None = None             # LLM-written memory (see §5)

class DiaryEntry(SQLModel, table=True):      # kept forever
    id: int | None = Field(default=None, primary_key=True)
    kind: str = Field(index=True)            # "session" | "daily"
    date: date = Field(index=True)           # unique per date for kind="daily"
    session_id: str | None = Field(default=None, index=True)  # set for kind="session"
    headline: str
    body_md: str                             # biographer's entry, markdown
    era_ids: str                             # JSON list of eras covered
    model: str                               # which Claude wrote it
    usage_json: str                          # token usage for cost tracking
```

Thumbnails live on disk (`data/chronicle/thumbs/YYYYMMDD/<session>_<kf>.webp`),
the DB stores paths. SQLite stays small; disk usage is bounded by retention.

---

## 2. GPU side: ChronicleRecorder

**New:** `backend/chronicle/__init__.py`, `models.py`, `recorder.py`.

### 2.1 Hook point

`AsyncOrchestrator` already owns the moment where everything is known: the
keyframe is generated, the cache worker has computed embeddings, injection
decisions were made inline, and `state_sync.on_keyframe_complete()` is called.
The recorder hooks in the same place:

```python
# async_orchestrator, after keyframe bookkeeping (same site as state_sync call)
if self.chronicle is not None:
    try:
        await self.chronicle.on_keyframe(
            keyframe_index=kf_index,
            frame_index=frame_index,
            prompt=prompt, negative=negative,
            template_id=self.prompt_manager.get_current_template_id(),
            components=self.prompt_manager.get_current_components(),
            events=events_this_keyframe,     # collected during the kf, see below
            embedding=embedding,             # {'color': ndarray, 'struct': hex}
            image=keyframe_image,            # PIL, already in memory
        )
    except Exception:
        logger.debug("chronicle hook failed", exc_info=True)  # NEVER raises upward
```

`events_this_keyframe` is a small list the orchestrator appends to at the four
places it already logs `[MUTATE]`, `[FORCE_MUTATE]`, injection, and template
switch — one line each, no new state machinery.

### 2.2 Thumbnail policy

- **Metadata**: every keyframe (~500 B each; ~1.3/s → trivial).
- **Thumbnail**: every `thumbnail_interval_s` seconds (default 30) **or**
  whenever `events` is non-empty. Mutations and injections are exactly the
  moments worth illustrating; drift between them is visually redundant.
- Encode: PIL resize to 256×128, `save(webp, quality=70)` — ~8–15 KB, and
  runs in the existing thread-pool executor so the loop never blocks.

At 1.3 kf/s this yields ~2,900 scheduled thumbs/day + event thumbs (~4–6k/day
total, ~60 MB/day pre-retention).

### 2.3 Batching + transport

Records accumulate in the recorder and flush every `flush_interval_s` (default
5 s) or 50 records, whichever first, as one message:

```
MessageType.CHRONICLE = 0x05   # GPU → VPS, JSON payload:
{ "type": "chronicle_batch", "records": [KeyframeRecord, ...] }
```

- Reuses `VPSWebSocketClient._send_binary` with **priority 0 and no
  queue-on-disconnect** (unlike frames): the chronicle is impressionist —
  a lost batch during a reconnect is a shrug. Exception: batches containing
  events are queued (they're the structural beats).
- `websocket.py` on the VPS adds one `elif msg_type == MSG_CHRONICLE` branch →
  `chronicle_store.ingest(batch)`.

### 2.4 Config (GPU side, `backend/config.*.yaml`)

```yaml
chronicle:
  enabled: true
  thumbnail_interval_s: 30
  flush_interval_s: 5
  flush_max_records: 50
  thumbnail_size: [256, 128]
  webp_quality: 70
```

---

## 3. VPS side: store + ingest

**New module:** `aethera/dreams/chronicle/` with its own `chronicle.sqlite`
(init in `main.py` lifespan next to `init_db()` / `init_irc_db()`).

`store.ingest(batch)`:
1. Decode thumbnails to disk, insert `Keyframe` rows (single transaction).
2. Feed each record to the **segmenter** (in-process, incremental).
3. Update the live era's counters.

All ingest work is wrapped so a malformed batch logs and drops — the chronicle
must never take down the stream hub it shares a process with.

---

## 4. Era segmentation

An era is "a stretch of the dream with a coherent look." Boundaries come from
two signals, cheapest first:

**Hard boundaries (event-driven, immediate):**
- `template_switch` → always closes the era.
- `session_start` (fresh boot, not resume) → always closes the era.

**Soft boundaries (drift detection, confirmed):**
- Maintain an EMA of the color-hist centroid and phash consensus over the last
  N=60 keyframes (~45 s). When the rolling window's mean similarity to the
  era's running centroid drops below a threshold **and stays below it for
  `confirm_keyframes` (default 120)**, close the era at the point the drop
  began. This is the collapse detector's math pointed backwards — same
  encoders, same similarity functions, no new dependencies (`color_encoder`
  and `phash_encoder` are already importable on the VPS via the shared repo,
  or the 3 similarity functions get vendored into `segmenter.py` — decide at
  implementation; vendoring ~50 lines is probably cleaner than importing
  torch-adjacent modules into the blog process).
- `cache_injection` events don't force a boundary but reset the confirmation
  counter (injections often *cause* legitimate drift).

**Era titling (mechanical, instant):** on close, title =
`{template_id} · {most persistent subject/material component} · {most persistent medium_render}`,
e.g. `material_study · bismuth crystal · anaglyph red cyan`. Computed by
counting component persistence across the era's keyframes. The LLM may later
improve it (§5), but the mechanical title is always present — the chronicle
never blocks on an API.

**Representatives:** on close, pick ≤10 thumbnails spread across the era:
always the first and last, then greedy farthest-point sampling in color-hist
space among the era's thumbnailed keyframes (maximizes visual coverage; ~20
lines of numpy).

Thresholds are declared **provisional** until Phase 1 data exists — we tune
against a real week of history, not a priori. `min_era_keyframes` (default
200, ~2.5 min) merges blips into their neighbor.

---

## 5. Consolidation — aging raw memory into narrative (LLM)

Nightly job (asyncio task in the FastAPI lifespan, or a cron hitting an
internal endpoint — implementation's choice; lifespan task is simpler):

1. Find eras fully outside the raw window (`ended_at < now - 14d`) with
   `consolidated == False`.
2. For each, build a **consolidation packet**: mechanical stats, exemplar
   prompts, event timeline, and the representative thumbnails (as images).
3. One Claude call per era (batched — see §7) with structured output:

```python
class EraMemory(BaseModel):
    """What the dream keeps when the raw footage fades."""
    improved_title: str          # ≤6 words, evocative but honest
    narrative: str               # 100-200 words: what this era looked like,
                                 # how it moved, how it ended. Written from
                                 # the packet ONLY — no invention of visuals
                                 # not evidenced by thumbnails/prompts.
    motifs: list[str]            # recurring visual motifs worth indexing
    continuity_note: str         # relationship to the previous era, if visible

resp = client.messages.parse(
    model=..., max_tokens=2048,
    system=CONSOLIDATOR_SYSTEM,          # frozen text, cache_control ephemeral
    messages=[{"role": "user", "content": [*thumb_image_blocks, packet_text]}],
    output_format=EraMemory,
)
```

4. Write `narrative`/`improved_title`/motifs onto the `Era` row, set
   `consolidated=True`, then **delete the era's raw `Keyframe` rows and
   non-representative thumbnails**. Deletion happens only after a successful
   DB write of the consolidation — if the API is down, raw data simply lives
   longer (grace window; hard-delete fallback at 30 d with mechanical-only
   summary so disk is always bounded).

This is the memory-consolidation shape you asked for: recent past is vivid
(every keyframe queryable), old past is narrative (a paragraph, a title, ten
images) — and the LLM does the lossy compression the way hippocampus-to-cortex
consolidation does, from evidence, at night.

## 6. The Biographer — session notes + nightly diary (LLM)

**Decision (2026-07-17): both cadences.** Two complementary writings:

**6a. Session note — "an entry per dream."** When a GPU session ends
(detected on the VPS by websocket disconnect + no reconnect within the
existing grace window), a short note is written for that session:

```python
class SessionNote(BaseModel):
    body_md: str        # 60-150 words: the arc of this one dream —
                        # where it started, where it went, how it was cut off
    headline: str       # one line, e.g. "the lotus became an eye became a door"
```

Input: that session's eras, exemplar prompts, first/last thumbnails. Sessions
shorter than `min_session_minutes` (default 10) or with <2 eras get a
mechanical stub instead of an API call.

**6b. Nightly synthesis.** The existing nightly job, now reading the day's
session notes *plus* era data, writing the day's entry:

```python
class Diary(BaseModel):
    body_md: str        # 150-400 words, first person plural is forbidden;
                        # written as an observer's field notes on the day
    headline: str       # one line for the chronicle page card
```

`DiaryEntry` gains a `kind` column (`"session" | "daily"`), with
`session_id` set on session notes and `date` unique only for `kind="daily"`.
The page shows the daily entry at the top of each day, session notes inline
where their sessions sit in the timeline.

Voice (both): the system prompt seeds 2–3 short excerpts from
`dream_watch_journal.md` as style reference — the biographer inherits the
tradition, it doesn't invent one. Guardrails: describe only what the evidence
shows; name eras by their titles; never claim to have watched continuously;
note gaps honestly ("the stream was down 0300–0700").

If the day had no generation, skip — no entries, no calls.

**Publication (decision): direct.** Notes and diary entries go live as
written; the admin panel gets edit/delete after the fact, no review queue.
The morning-surprise of an unreviewed entry is part of the charm.

## 7. Model + cost

Per the current API reference:

- **Model:** `claude-opus-4-8` for both consolidator and biographer
  ($5/$25 per MTok). These are exactly the judgment-heavy, low-volume calls
  where model quality is the product — the narratives are kept *forever*.
- **Batch API** (50% discount) for consolidation: it's a pile of independent
  era packets with a 24 h deadline — the textbook batch case. The biographer
  calls (session notes + nightly synthesis) run as normal requests — they're
  small, and session notes want to appear promptly after a dream ends.
- **Structured outputs** via `client.messages.parse(...)` + the pydantic
  models above — no hand-parsing, validation for free.
- **Prompt caching:** frozen system prompts with `cache_control: ephemeral`.

Cost envelope (generous): consolidation ~5–15 era-calls/day × (~3k input incl.
images + 500 out) ≈ $0.15/day batched; session notes ~1–4/day × (~2k in +
300 out) ≈ $0.05/day; nightly synthesis 1 call × (~8k in + 800 out)
≈ $0.06/day. **≈ $6–9/month** — rounding error next to the GPU. A
`chronicle.llm.daily_budget_calls` config cap (default 40) guards runaways;
on budget exhaustion, mechanical-only fallback.

API key: `ANTHROPIC_API_KEY` env var on the VPS only. The GPU side never
holds it and never calls the API.

## 8. HTTP surface

```
GET /dreams/chronicle                     # HTML page (Jinja, HTMX expansion)
GET /dreams/chronicle?day=2026-07-17      # a specific day
GET /api/dreams/chronicle/eras?limit=20&before=<id>     # JSON, paginated
GET /api/dreams/chronicle/eras/{id}       # full era incl. narrative + thumbs
GET /api/dreams/chronicle/diary/{date}    # diary entry
GET /api/dreams/chronicle/current         # live era snapshot (title, age, counters)
GET /chronicle/thumbs/{path}              # static thumbnails (nginx-cacheable)
```

Page layout (matches blog styling, semantic HTML, JSON-LD `CreativeWorkSeries`
for the SEO focus):

```
┌───────────────────────────────────────────────────────────┐
│ THE CHRONICLE          what the dream remembers           │
│                                                           │
│ ── Jul 17 ──────────────────────  ▁▂▅▃▂▆▂▁ 41 eras today  │
│ ❝ headline from the biographer… ❞            [read entry] │
│                                                           │
│ ● 22:52–23:08  specimen · oxblood lace · botanical        │
│   [▓][▓][▓][▓]   412 kf · 3 mutations · 1 injection       │
│   "pressed specimen of oxblood lace, botanical plate…"    │
│   ▸ expand: full thumb run + prompts (HTMX)               │
│ ● 22:39–22:52  environmental · sediment delta · …         │
│ …                                                         │
│ ── ARCHIVE ──  Jul 16 ▸   Jul 15 ▸   (consolidated eras   │
│                show narrative + 10 thumbs, no raw)        │
│ footer: link → the hand-written watch journals            │
└───────────────────────────────────────────────────────────┘
```

Session notes render inline in the timeline at their session's position;
the daily diary entry heads each day. The hand-written watch journals are
**linked** from the page footer (decision: link, don't embed — the chronicle
stands on machine records; the journals remain their own artifact).

## 9. Retention summary

| Data | Where | Kept |
|---|---|---|
| Keyframe rows + embeddings | chronicle.sqlite | 14 d (30 d hard cap) |
| All thumbnails | disk | 14 d |
| Representative thumbnails (≤10/era) | disk | forever |
| Era rows + narratives + motifs | chronicle.sqlite | forever |
| Diary entries | chronicle.sqlite | forever |

Steady-state disk: ~1 GB rolling window + ~2 MB/day permanent. Years of
permanent chronicle fit in single-digit GB.

## 10. Failure philosophy

The chronicle is a **remora, not an organ the host depends on**:

- Every GPU-side hook is wrapped; a chronicle bug can cost records, never frames.
- Transport is lossy by design (except event batches).
- Segmentation errors self-heal: a missed boundary makes one long era, a
  spurious one makes two short eras — both are cosmetic.
- LLM outages degrade to mechanical titles/summaries; nothing blocks.
- VPS ingest failures log and drop; the stream hub is untouched.

## 11. Testing

- `backend/tests/test_chronicle_recorder.py`: thumbnail policy, batching,
  never-raises contract (inject a poisoned image, assert loop continues).
- VPS: ingest round-trip (wire → DB → thumbs), segmenter unit tests with
  synthetic embedding streams (stable / drifting / switching), retention job
  idempotency, consolidation with a mocked `messages.parse`.
- One live calibration run: Phase 1 deployed alone for ≥3 days before any
  segmentation thresholds are frozen.

## 12. Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| **1. Record** | GPU recorder + 0x05 transport + VPS ingest + retention job + `/api/.../current` | One full GPU session recorded end-to-end; disk growth as predicted. (Generation is intermittent — no dedicated machine — so calibration is per-session, accumulating across however many runs happen.) |
| **2. Segment** | segmenter + Era table + mechanical titles + representatives; tune thresholds on accumulated Phase 1 sessions | Eras on real data match eyeball judgment on the thumb runs |
| **3. Serve** | chronicle page + API + SEO + links to the watch journals | Page live at /dreams/chronicle |
| **4. Consolidate** | aging job + EraMemory LLM call + deletion discipline | An era consolidated end-to-end; raw rows gone, narrative kept |
| **5. Biographer** | nightly diary + page integration | A morning where you read what the dream did overnight |

Each phase ships and runs alone before the next starts — matching "carefully,
over time."

## Resolved decisions (2026-07-17, with Luxia)

1. **Segmenter code sharing**: **vendor** the ~50 lines of similarity math
   into `aethera/dreams/chronicle/segmenter.py`, with a comment pointing at
   the source functions in this repo. The blog process stays free of
   dream_gen's import chain.
2. **Biographer cadence**: **both** — short per-session notes ("an entry per
   dream") plus a nightly synthesis. See §6.
3. **Publication**: **direct** — entries publish as written; admin panel gets
   edit/delete, no review queue.
4. **Journal linkage**: **link, don't embed** — the chronicle page footer
   links to the hand-written watch journals; they remain their own artifact.
