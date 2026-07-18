"""
Chronicle wire models (GPU -> VPS)

One KeyframeRecord per generated keyframe, batched into a single
MessageType.CHRONICLE websocket message:

    0x05 | JSON payload: {"type": "chronicle_batch", "records": [...]}

Records are ~500 bytes without a thumbnail; thumbnails ride along as
base64 webp only on sampled/event keyframes (see recorder thumbnail policy).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

EventKind = Literal[
    "mutation",
    "forced_mutation",
    "cache_injection",
    "seed_injection",
    "template_switch",
    "session_start",  # fresh boot - hard era boundary for the segmenter
    "session_resume",  # continued from a checkpoint - continuity, NOT a boundary
]


class ChronicleEvent(BaseModel):
    """What happened at this keyframe, if anything."""

    kind: EventKind
    detail: str = ""  # e.g. "color_logic: 'verdigris' -> 'oxblood'"


class KeyframeRecord(BaseModel):
    """One keyframe's memoir entry."""

    session_id: str  # unique per GPU process boot
    keyframe: int  # keyframe number (monotonic within session)
    lifetime_keyframe: Optional[int] = None  # epoch + keyframe (SPEC-resume.md)
    sequence: int  # buffer sequence number of the keyframe
    ts: float  # unix epoch, GPU clock
    prompt: str
    negative: str = ""
    template_id: str = ""
    components: dict[str, str] = Field(default_factory=dict)
    events: list[ChronicleEvent] = Field(default_factory=list)

    # Embeddings for VPS-side era segmentation (computed here, never on VPS)
    color_hist: Optional[list[float]] = None  # 96-dim ColorHist, rounded 4dp
    phash: Optional[str] = None  # pHash-8 hex string

    # Present only on sampled/event keyframes
    thumb_webp_b64: Optional[str] = None  # 256x128 webp, ~8-15KB


class ChronicleBatch(BaseModel):
    """Wire envelope for a flush."""

    type: Literal["chronicle_batch"] = "chronicle_batch"
    records: list[KeyframeRecord] = Field(default_factory=list)
