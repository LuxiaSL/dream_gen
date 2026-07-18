"""
ChronicleRecorder - observes keyframes, ships memoir records to the VPS

Design contract (SPEC-chronicle.md §2, §10):
- The chronicle is a remora, not an organ the host depends on. Every public
  method is wrapped so a chronicle failure can cost records, never frames.
- on_keyframe() is O(microseconds): it enqueues a job and returns. All image
  work (load, thumbnail, embeddings) happens on a background task via the
  default thread-pool executor.
- Transport is lossy by design: the work queue drops oldest on overflow and
  batches are not queued across disconnects. A lost batch is a shrug.

Thumbnail policy:
- metadata for every keyframe (~500B)
- thumbnail every `thumbnail_interval_s` seconds OR whenever the keyframe
  carries events (mutation/injection/template switch) - the dramatic beats
  are exactly the moments worth illustrating.
"""

import asyncio
import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from backend.utils.color_encoder import ColorHistogramEncoder
    from backend.utils.phash_encoder import PHashEncoder
except ImportError:  # entry points that put backend/ itself on sys.path
    from utils.color_encoder import ColorHistogramEncoder
    from utils.phash_encoder import PHashEncoder

from .models import ChronicleBatch, ChronicleEvent, KeyframeRecord

logger = logging.getLogger(__name__)


class ChronicleRecorder:
    """
    Collects per-keyframe records and flushes them to the VPS in batches.

    Usage (mirrors CloudFramePusher / CloudStateSync):
        recorder = ChronicleRecorder(vps_client, config)
        orchestrator.chronicle = recorder
        ...
        await recorder.close()   # on shutdown: final flush
    """

    def __init__(self, websocket_client: Any, config: dict):
        self.ws_client = websocket_client

        ch = config.get("chronicle", {})
        self.enabled: bool = ch.get("enabled", True)
        self.thumbnail_interval_s: float = float(ch.get("thumbnail_interval_s", 30))
        self.flush_interval_s: float = float(ch.get("flush_interval_s", 5))
        self.flush_max_records: int = int(ch.get("flush_max_records", 50))
        thumb_size = ch.get("thumbnail_size", [256, 128])
        self.thumb_size: tuple[int, int] = (int(thumb_size[0]), int(thumb_size[1]))
        self.webp_quality: int = int(ch.get("webp_quality", 70))
        self.max_queue: int = int(ch.get("max_queue", 200))

        # One session per process boot; the VPS stitches history by session_id
        self.session_id: str = uuid.uuid4().hex

        # Resume continuity (SPEC-resume.md), set by DreamController when a
        # checkpoint was loaded: lifetime numbering offset + prior session id
        self.epoch_offset: int = 0
        self.resumed_from: Optional[str] = None

        # Embedding encoders (numpy/PIL only - cheap, no GPU)
        self._color = ColorHistogramEncoder()
        self._phash = PHashEncoder()

        # Work queue: on_keyframe() enqueues, _worker() consumes
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        self._pending: list[KeyframeRecord] = []
        self._last_flush: float = time.time()
        self._last_thumb_ts: float = 0.0
        self._session_started = False

        self._worker_task: Optional[asyncio.Task] = None
        self._closed = False

        # Statistics
        self.records_made = 0
        self.records_dropped = 0
        self.thumbs_made = 0
        self.batches_sent = 0
        self.batches_failed = 0

        logger.info(
            f"ChronicleRecorder initialized: session={self.session_id[:8]}, "
            f"thumb every {self.thumbnail_interval_s:.0f}s + events, "
            f"flush {self.flush_interval_s:.0f}s/{self.flush_max_records} records"
        )

    # ------------------------------------------------------------------ #
    # Public API (never raises)                                          #
    # ------------------------------------------------------------------ #

    async def on_keyframe(
        self,
        *,
        keyframe_num: int,
        sequence_num: int,
        prompt: str,
        negative: str = "",
        template_id: str = "",
        components: Optional[dict] = None,
        events: Optional[list[dict]] = None,
        image_path: Optional[Path] = None,
    ) -> None:
        """
        Record a completed keyframe. Cheap: enqueues and returns.

        `events` is a list of {"kind": ..., "detail": ...} dicts collected by
        the orchestrator for this keyframe number.
        """
        if not self.enabled or self._closed:
            return
        try:
            self._ensure_worker()

            event_models = []
            if not self._session_started:
                if self.resumed_from:
                    event_models.append(
                        ChronicleEvent(kind="session_resume", detail=self.resumed_from)
                    )
                else:
                    event_models.append(ChronicleEvent(kind="session_start"))
                self._session_started = True
            for ev in events or []:
                try:
                    event_models.append(ChronicleEvent(**ev))
                except Exception:
                    logger.debug(f"Chronicle: bad event skipped: {ev!r}")

            now = time.time()
            want_thumb = bool(event_models) or (
                now - self._last_thumb_ts >= self.thumbnail_interval_s
            )
            if want_thumb:
                self._last_thumb_ts = now

            job = {
                "keyframe": keyframe_num,
                "lifetime_keyframe": self.epoch_offset + keyframe_num,
                "sequence": sequence_num,
                "ts": now,
                "prompt": prompt or "",
                "negative": negative or "",
                "template_id": template_id or "",
                "components": dict(components or {}),
                "events": event_models,
                "image_path": Path(image_path) if image_path else None,
                "want_thumb": want_thumb,
            }

            if self._queue.full():
                # Lossy by design: drop the oldest job, keep the newest
                try:
                    self._queue.get_nowait()
                    self.records_dropped += 1
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(job)
        except Exception:
            logger.debug("Chronicle on_keyframe failed", exc_info=True)

    async def close(self) -> None:
        """Stop the worker, process the remaining queue, flush what's left."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._worker_task:
                # Let the worker drain briefly, then cancel
                try:
                    await asyncio.wait_for(self._queue.join(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._flush()
            logger.info(
                f"Chronicle closed: {self.records_made} records, "
                f"{self.thumbs_made} thumbs, {self.batches_sent} batches sent, "
                f"{self.records_dropped} dropped"
            )
        except Exception:
            logger.debug("Chronicle close failed", exc_info=True)

    def get_stats(self) -> dict:
        return {
            "session_id": self.session_id,
            "records_made": self.records_made,
            "records_dropped": self.records_dropped,
            "thumbs_made": self.thumbs_made,
            "batches_sent": self.batches_sent,
            "batches_failed": self.batches_failed,
            "pending": len(self._pending),
            "queued": self._queue.qsize(),
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.get_running_loop().create_task(self._worker())

    async def _worker(self) -> None:
        """Consume jobs, build records (image work in executor), flush on schedule."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                try:
                    job = await asyncio.wait_for(
                        self._queue.get(), timeout=self.flush_interval_s
                    )
                except asyncio.TimeoutError:
                    # Idle: time-based flush still fires
                    await self._maybe_flush()
                    continue

                try:
                    color_hist, phash_hex, thumb_b64 = await loop.run_in_executor(
                        None,
                        self._process_image,
                        job["image_path"],
                        job["want_thumb"],
                    )
                    record = KeyframeRecord(
                        session_id=self.session_id,
                        keyframe=job["keyframe"],
                        lifetime_keyframe=job["lifetime_keyframe"],
                        sequence=job["sequence"],
                        ts=job["ts"],
                        prompt=job["prompt"],
                        negative=job["negative"],
                        template_id=job["template_id"],
                        components=job["components"],
                        events=job["events"],
                        color_hist=color_hist,
                        phash=phash_hex,
                        thumb_webp_b64=thumb_b64,
                    )
                    self._pending.append(record)
                    self.records_made += 1
                    if thumb_b64:
                        self.thumbs_made += 1
                except Exception:
                    logger.debug("Chronicle record build failed", exc_info=True)
                finally:
                    self._queue.task_done()

                await self._maybe_flush()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Chronicle worker error", exc_info=True)
                await asyncio.sleep(1.0)

    def _process_image(
        self, image_path: Optional[Path], want_thumb: bool
    ) -> tuple[Optional[list[float]], Optional[str], Optional[str]]:
        """
        Runs in executor. Load once; derive thumbnail + both embeddings.

        The keyframe file may already be cleaned up by the display selector -
        in that case the record ships metadata-only, which is fine.
        """
        if image_path is None:
            return None, None, None
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                hist = self._color.encode_image(img)
                color_list = (
                    [round(float(x), 4) for x in hist] if hist is not None else None
                )

                phash_obj = self._phash.encode_image(img)
                phash_hex = (
                    self._phash.to_serializable(phash_obj)
                    if phash_obj is not None
                    else None
                )

                thumb_b64 = None
                if want_thumb:
                    thumb = img.resize(self.thumb_size, Image.LANCZOS)
                    buf = io.BytesIO()
                    thumb.save(buf, format="WEBP", quality=self.webp_quality)
                    thumb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                return color_list, phash_hex, thumb_b64
        except Exception:
            logger.debug(f"Chronicle image processing failed: {image_path}")
            return None, None, None

    async def _maybe_flush(self) -> None:
        if not self._pending:
            return
        if (
            len(self._pending) >= self.flush_max_records
            or time.time() - self._last_flush >= self.flush_interval_s
        ):
            await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        batch = ChronicleBatch(records=self._pending)
        self._pending = []
        self._last_flush = time.time()
        try:
            payload = batch.model_dump_json(exclude_none=True).encode("utf-8")
            sent = await self.ws_client.send_chronicle(payload)
            if sent:
                self.batches_sent += 1
                logger.debug(
                    f"Chronicle batch sent: {len(batch.records)} records, "
                    f"{len(payload) / 1024:.1f}KB"
                )
            else:
                self.batches_failed += 1
        except Exception:
            self.batches_failed += 1
            logger.debug("Chronicle flush failed", exc_info=True)
