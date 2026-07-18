"""
Chronicle - the dream's own memory (GPU side)

Observes the generation pipeline and ships lightweight per-keyframe records
(plus tiered thumbnails) to the VPS over the existing websocket, where they
become the /dreams/chronicle timeline. See SPEC-chronicle.md.
"""

from .models import ChronicleEvent, KeyframeRecord
from .recorder import ChronicleRecorder

__all__ = ["ChronicleEvent", "KeyframeRecord", "ChronicleRecorder"]
