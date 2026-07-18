"""
Tests for ChronicleRecorder

Validates the recorder's contracts (SPEC-chronicle.md §2, §10):
- Records are built with embeddings and shipped in batches
- Thumbnail policy: interval-based + always on events
- Never-raises contract: poisoned/missing images still produce records
- Queue overflow drops oldest instead of blocking
- Wire format round-trips through ChronicleBatch
"""

import asyncio
import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from chronicle import ChronicleRecorder
from chronicle.models import ChronicleBatch


class FakeWSClient:
    """Captures chronicle payloads; can be told to fail."""

    def __init__(self, fail: bool = False):
        self.payloads: list[bytes] = []
        self.fail = fail

    async def send_chronicle(self, payload: bytes) -> bool:
        if self.fail:
            return False
        self.payloads.append(payload)
        return True

    def batches(self) -> list[ChronicleBatch]:
        return [ChronicleBatch(**json.loads(p)) for p in self.payloads]


def make_config(**chronicle_overrides) -> dict:
    chronicle = {
        "enabled": True,
        "thumbnail_interval_s": 30,
        "flush_interval_s": 0.2,
        "flush_max_records": 50,
        "thumbnail_size": [64, 32],
        "webp_quality": 70,
        "max_queue": 200,
    }
    chronicle.update(chronicle_overrides)
    return {"chronicle": chronicle}


@pytest.fixture
def keyframe_png(tmp_path: Path) -> Path:
    path = tmp_path / "kf_0001.png"
    img = Image.new("RGB", (128, 64))
    for x in range(128):
        for y in range(64):
            img.putpixel((x, y), (x * 2 % 256, y * 4 % 256, (x + y) % 256))
    img.save(path)
    return path


async def drain(recorder: ChronicleRecorder) -> None:
    """Wait until the recorder's work queue is fully processed."""
    await asyncio.wait_for(recorder._queue.join(), timeout=5.0)


async def test_record_flush_and_wire_roundtrip(keyframe_png):
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config())

    await rec.on_keyframe(
        keyframe_num=1, sequence_num=10, prompt="a dream of verdigris",
        negative="low quality", template_id="material_study",
        components={"color_logic": "verdigris"},
        events=[], image_path=keyframe_png,
    )
    await drain(rec)
    await rec._flush()

    batches = ws.batches()
    assert len(batches) == 1
    records = batches[0].records
    assert len(records) == 1
    r = records[0]

    assert r.keyframe == 1 and r.sequence == 10
    assert r.prompt == "a dream of verdigris"
    assert r.template_id == "material_study"
    assert r.components == {"color_logic": "verdigris"}
    # First record of a session always carries session_start
    assert [e.kind for e in r.events] == ["session_start"]
    # Embeddings computed from the image
    assert r.color_hist is not None and len(r.color_hist) == 96
    assert r.phash is not None and len(r.phash) > 0
    # session_start is an event -> thumbnail must be present and decodable
    assert r.thumb_webp_b64 is not None
    thumb = Image.open(
        __import__("io").BytesIO(base64.b64decode(r.thumb_webp_b64))
    )
    assert thumb.size == (64, 32)

    await rec.close()


async def test_thumbnail_policy(keyframe_png):
    """Within the interval: no thumb. Events: always a thumb."""
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config(thumbnail_interval_s=9999))

    # kf 1: session_start event -> thumb
    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=keyframe_png)
    # kf 2: no events, interval far away -> no thumb
    await rec.on_keyframe(keyframe_num=2, sequence_num=2, prompt="p",
                          events=[], image_path=keyframe_png)
    # kf 3: mutation event -> thumb despite interval
    await rec.on_keyframe(
        keyframe_num=3, sequence_num=3, prompt="p",
        events=[{"kind": "mutation", "detail": "color_logic: 'a' -> 'b'"}],
        image_path=keyframe_png,
    )
    await drain(rec)
    await rec._flush()

    records = {r.keyframe: r for r in ws.batches()[0].records}
    assert records[1].thumb_webp_b64 is not None
    assert records[2].thumb_webp_b64 is None
    assert records[3].thumb_webp_b64 is not None
    assert records[3].events[0].kind == "mutation"
    assert "color_logic" in records[3].events[0].detail

    await rec.close()


async def test_never_raises_on_poisoned_images(tmp_path):
    """Missing and corrupt images still produce metadata-only records."""
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config())

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a png at all")

    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=tmp_path / "does_not_exist.png")
    await rec.on_keyframe(keyframe_num=2, sequence_num=2, prompt="p",
                          events=[], image_path=corrupt)
    await rec.on_keyframe(keyframe_num=3, sequence_num=3, prompt="p",
                          events=[], image_path=None)
    await drain(rec)
    await rec._flush()

    records = ws.batches()[0].records
    assert len(records) == 3
    for r in records:
        assert r.color_hist is None
        assert r.phash is None
        assert r.thumb_webp_b64 is None
        assert r.prompt == "p"

    await rec.close()


async def test_bad_events_are_skipped_not_fatal(keyframe_png):
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config())

    await rec.on_keyframe(
        keyframe_num=1, sequence_num=1, prompt="p",
        events=[{"kind": "not_a_real_kind"}, {"kind": "mutation", "detail": "ok"}],
        image_path=keyframe_png,
    )
    await drain(rec)
    await rec._flush()

    r = ws.batches()[0].records[0]
    kinds = [e.kind for e in r.events]
    assert "mutation" in kinds and "session_start" in kinds
    assert "not_a_real_kind" not in kinds

    await rec.close()


async def test_size_based_flush(keyframe_png):
    """flush_max_records triggers a flush without waiting for the interval."""
    ws = FakeWSClient()
    rec = ChronicleRecorder(
        ws, make_config(flush_max_records=3, flush_interval_s=9999)
    )

    for i in range(1, 5):
        await rec.on_keyframe(keyframe_num=i, sequence_num=i, prompt="p",
                              events=[], image_path=keyframe_png)
    await drain(rec)

    assert len(ws.payloads) >= 1
    assert len(ws.batches()[0].records) == 3

    await rec.close()


async def test_queue_overflow_drops_oldest():
    """With a tiny queue and no worker drain, oldest jobs drop, newest survive."""
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config(max_queue=3))
    # Prevent the worker from draining so the queue actually fills
    rec._worker_task = asyncio.get_running_loop().create_task(asyncio.sleep(3600))

    for i in range(1, 8):
        await rec.on_keyframe(keyframe_num=i, sequence_num=i, prompt="p",
                              events=[], image_path=None)

    assert rec._queue.qsize() == 3
    assert rec.records_dropped == 4
    kept = [rec._queue.get_nowait()["keyframe"] for _ in range(3)]
    assert kept == [5, 6, 7]

    rec._worker_task.cancel()
    rec._closed = True


async def test_send_failure_is_counted_not_raised(keyframe_png):
    ws = FakeWSClient(fail=True)
    rec = ChronicleRecorder(ws, make_config())

    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=keyframe_png)
    await drain(rec)
    await rec._flush()

    assert rec.batches_failed >= 1
    assert ws.payloads == []

    await rec.close()


async def test_disabled_recorder_is_inert(keyframe_png):
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config(enabled=False))

    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=keyframe_png)
    await asyncio.sleep(0.05)

    assert rec._worker_task is None
    assert ws.payloads == []
    await rec.close()


async def test_resume_continuity(keyframe_png):
    """Resumed sessions emit session_resume (not session_start) and carry
    lifetime keyframe numbering offset by the epoch."""
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config())
    rec.epoch_offset = 15000
    rec.resumed_from = "prev_session_abc"

    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=keyframe_png)
    await drain(rec)
    await rec._flush()

    r = ws.batches()[0].records[0]
    kinds = [e.kind for e in r.events]
    assert kinds == ["session_resume"]
    assert r.events[0].detail == "prev_session_abc"
    assert r.lifetime_keyframe == 15001
    assert r.keyframe == 1

    await rec.close()


async def test_fresh_session_lifetime_equals_local(keyframe_png):
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config())

    await rec.on_keyframe(keyframe_num=7, sequence_num=7, prompt="p",
                          events=[], image_path=keyframe_png)
    await drain(rec)
    await rec._flush()

    r = ws.batches()[0].records[0]
    assert [e.kind for e in r.events] == ["session_start"]
    assert r.lifetime_keyframe == 7

    await rec.close()


async def test_close_flushes_remaining(keyframe_png):
    ws = FakeWSClient()
    rec = ChronicleRecorder(ws, make_config(flush_interval_s=9999,
                                            flush_max_records=9999))

    await rec.on_keyframe(keyframe_num=1, sequence_num=1, prompt="p",
                          events=[], image_path=keyframe_png)
    await drain(rec)
    assert ws.payloads == []  # nothing flushed yet

    await rec.close()
    assert len(ws.payloads) == 1
    assert ws.batches()[0].records[0].keyframe == 1
