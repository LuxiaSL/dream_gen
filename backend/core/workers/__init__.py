"""
Async Worker Classes for Parallelized Generation

Provides three independent workers that run concurrently to eliminate
blocking operations and achieve 2x+ FPS improvement:

- KeyframeWorker: HTTP I/O bound (ComfyUI generation)
- InterpolationWorker: GPU bound (VAE encode/decode/slerp)
- CacheAnalysisWorker: CPU bound (similarity analysis)

Each worker operates on a queue-based architecture with asyncio for
clean coordination and graceful shutdown.
"""

from .keyframe_worker import KeyframeWorker
from .interpolation_worker import InterpolationWorker
from .cache_worker import CacheAnalysisWorker

__all__ = [
    "KeyframeWorker",
    "InterpolationWorker",
    "CacheAnalysisWorker",
]

