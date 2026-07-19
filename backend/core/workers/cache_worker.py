"""
Cache Analysis Worker - Async Frame Diversity Analysis

Handles cache population decisions via async queue, using current diversity
logic with designed hooks for future advanced monitoring.

This worker is CPU bound (similarity calculations), so operations run
concurrently without blocking GPU-bound workers.

PHASE 1 (Current Implementation):
- Basic async diversity checking
- Selective cache population
- Current average-based similarity logic

PHASE 2 (Future Enhancement - Hooks Present):
- Continuous diversity matrix (O(N²) background)
- Smart redundancy-based eviction
- Max-similarity acceptance logic
- Cluster detection
- Adaptive threshold tuning

See: docs/ASYNC_CACHE_MONITORING_DESIGN.md for Phase 2 details
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CacheAnalysisWorker:
    """
    Async worker for cache frame analysis and population
    
    Responsibilities (Phase 1):
    - Maintain queue of frames for analysis
    - Encode images with dual-metric similarity system
    - Check diversity (current average-based logic)
    - Add diverse frames to cache
    - NO injection decisions (orchestrator handles that inline)
    
    Future Capabilities (Phase 2 - Hooks Present):
    - Background diversity matrix updates
    - Smart eviction candidate selection
    - Cluster detection
    - Adaptive threshold recommendations
    
    Queue Flow:
        Coordinator → submit_frame() → analysis_queue
        ↓
        run() loop processes frames
        ↓
        Adds to cache if diverse (no output queue needed)
    
    Usage:
        worker = CacheAnalysisWorker(
            cache=cache_manager,
            similarity_manager=similarity_mgr,
            config=config
        )
        
        # Start worker loop
        asyncio.create_task(worker.run())
        
        # Submit frame for analysis
        await worker.submit_frame(
            frame_path=Path("keyframe_005.png"),
            prompt="ethereal dreamscape...",
            metadata={'denoise': 0.6, 'type': 'keyframe'}
        )
    """
    
    def __init__(
        self,
        cache,  # CacheManager instance
        similarity_manager,  # DualMetricSimilarityManager instance
        config: Dict[str, Any],
        max_queue_size: int = 20
    ):
        """
        Initialize cache analysis worker
        
        Args:
            cache: CacheManager instance
            similarity_manager: DualMetricSimilarityManager instance
            config: Configuration dictionary
            max_queue_size: Maximum pending analyses (backpressure control)
        """
        self.cache = cache
        self.similarity_manager = similarity_manager
        self.config = config

        # Queue
        self.analysis_queue = asyncio.Queue(maxsize=max_queue_size)

        # State
        self.running = False
        self.processing = False

        # Latent-gated admission (cache/latent_pool.py). The metric audit
        # showed colorhist/phash can't rank within-motif difference, so the
        # cache flooded with near-copies and injection became self-reinforcing.
        # Admission now requires a minimum pooled-latent cosine distance to
        # the nearest existing entry, plus a metric-free temporal floor.
        la = config.get('generation', {}).get('cache', {}).get('latent_admission', {})
        self.admission_enabled = bool(la.get('enabled', True))
        self.admission_min_dist = float(la.get('min_dist', 0.10))
        self.admission_min_interval_kf = int(la.get('min_interval_kf', 12))
        self.admission_latent_wait_s = float(la.get('latent_wait_s', 4.0))
        # kf_num -> Optional[np.ndarray]; wired by the orchestrator to the
        # interpolation worker's pooled-latent stash
        self.latent_provider = None
        self._last_admitted_kf: 'int | None' = None

        # Statistics
        self.frames_analyzed = 0
        self.frames_cached = 0
        self.frames_skipped = 0
        self.frames_rejected_floor = 0
        self.frames_rejected_similar = 0

        logger.info(
            f"CacheAnalysisWorker initialized (max queue: {max_queue_size}, "
            f"latent admission: {'on' if self.admission_enabled else 'off'}, "
            f"min_dist={self.admission_min_dist}, "
            f"min_interval={self.admission_min_interval_kf}kf)"
        )
    
    async def submit_frame(
        self,
        frame_path: Path,
        prompt: str,
        metadata: Dict[str, Any],
        latent_vec=None,
    ) -> None:
        """
        Submit a frame for cache analysis

        Args:
            frame_path: Path to frame image
            prompt: Generation prompt used
            metadata: Additional metadata (denoise, type, keyframe_num, etc.)
            latent_vec: Optional pre-pooled latent embedding (np.ndarray).
                        If absent, the worker asks latent_provider by
                        metadata['keyframe_num'].
        """
        frame = {
            'path': frame_path,
            'prompt': prompt,
            'metadata': metadata,
            'latent_vec': latent_vec,
        }
        
        # Check for queue backlog (skip if falling behind)
        if self.analysis_queue.qsize() >= self.analysis_queue.maxsize * 0.8:
            logger.warning(
                f"Cache analysis queue near capacity "
                f"({self.analysis_queue.qsize()}/{self.analysis_queue.maxsize}) - "
                f"skipping frame to prevent backlog"
            )
            self.frames_skipped += 1
            return
        
        await self.analysis_queue.put(frame)
        
        logger.debug(
            f"Submitted frame for analysis: {frame_path.name} "
            f"(queue depth: {self.analysis_queue.qsize()})"
        )
    
    async def _analyze_frame_diversity(
        self,
        frame: Dict[str, Any]
    ) -> tuple[bool, Optional[Any]]:
        """
        Analyze frame diversity and decide if should cache
        
        Phase 1: Current average-based diversity check
        Phase 2: Will support max-similarity logic
        
        Abstraction allows swapping strategies without changing interface.
        
        Args:
            frame: Frame dictionary with path, prompt, metadata
            
        Returns:
            Tuple of (should_cache, embedding)
        """
        frame_path = frame['path']
        kf_num = frame.get('metadata', {}).get('keyframe_num')

        try:
            # === 1. Temporal floor (metric-free flood-stop) ===
            # Applies before any encoding: at most one admission per
            # min_interval_kf keyframes, no matter what the metrics say.
            if (
                self.admission_enabled
                and kf_num is not None
                and self._last_admitted_kf is not None
                and 0 <= kf_num - self._last_admitted_kf < self.admission_min_interval_kf
            ):
                self.frames_rejected_floor += 1
                logger.debug(
                    f"Admission floor: {frame_path.name} "
                    f"(kf {kf_num}, last admitted {self._last_admitted_kf})"
                )
                return False, None

            # Encode image with dual-metric similarity system
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.similarity_manager.encode_image,
                frame_path
            )

            if embedding is None:
                logger.warning(f"Failed to encode frame: {frame_path.name}")
                return False, None

            # === 2. Latent distance gate ===
            latent_vec = frame.get('latent_vec')
            if latent_vec is None:
                latent_vec = await self._await_latent(kf_num)

            if latent_vec is not None:
                embedding['latent'] = latent_vec
                if self.admission_enabled:
                    min_dist = self._min_dist_to_cache(latent_vec)
                    if min_dist is not None and min_dist < self.admission_min_dist:
                        self.frames_rejected_similar += 1
                        logger.debug(
                            f"Admission rejected {frame_path.name}: "
                            f"min latent dist {min_dist:.4f} < {self.admission_min_dist}"
                        )
                        return False, None
            # No latent available -> the temporal floor above is the only
            # gate (remora philosophy: a broken latent path must not kill
            # the cache entirely)

            return True, embedding

        except Exception as e:
            logger.error(f"Error analyzing frame diversity: {e}", exc_info=True)
            return False, None

    async def _await_latent(self, kf_num):
        """
        Fetch the pooled latent for a keyframe, waiting briefly if the
        interpolation worker hasn't encoded it yet. Returns None on
        timeout/failure — never raises.
        """
        if self.latent_provider is None or kf_num is None:
            return None
        deadline = asyncio.get_event_loop().time() + self.admission_latent_wait_s
        while True:
            try:
                vec = self.latent_provider(kf_num)
            except Exception:
                return None
            if vec is not None:
                return vec
            if asyncio.get_event_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.5)

    def _min_dist_to_cache(self, vec):
        """Min cosine distance from vec to existing cache entries (None if no
        entry has a latent embedding yet)."""
        try:
            from cache.latent_pool import cosine_dist
        except ImportError:
            return None
        dists = []
        for entry in self.cache.get_all():
            emb = getattr(entry, 'embedding', None)
            if isinstance(emb, dict) and emb.get('latent') is not None:
                try:
                    dists.append(cosine_dist(vec, emb['latent']))
                except Exception:
                    pass
        return min(dists) if dists else None
    
    async def _cache_frame(
        self,
        frame: Dict[str, Any],
        embedding: Any
    ) -> bool:
        """
        Add frame to cache
        
        Args:
            frame: Frame dictionary
            embedding: Dual-metric embedding
            
        Returns:
            True if successfully added
        """
        try:
            # Convert to serializable format if needed
            if isinstance(embedding, dict) and 'color' in embedding:
                # Dual-metric: convert to serializable
                embedding = self.similarity_manager.to_serializable(embedding)
            
            # Prepare generation params
            generation_params = {
                "model": self.config.get("generation", {}).get("model", "sd15"),
                "resolution": self.config.get("generation", {}).get("resolution", [512, 256])
            }
            generation_params.update(frame['metadata'])
            
            # Add to cache (run in executor)
            loop = asyncio.get_event_loop()
            cache_id = await loop.run_in_executor(
                None,
                self.cache.add,
                frame['path'],
                frame['prompt'],
                generation_params,
                embedding
            )
            
            logger.debug(
                f"Added frame to cache: {cache_id} "
                f"(total: {self.cache.size()})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add frame to cache: {e}", exc_info=True)
            return False
    
    async def run(self) -> None:
        """
        Main worker loop
        
        Processes frame analysis requests from queue.
        Runs until stop() is called.
        """
        self.running = True
        logger.info("CacheAnalysisWorker started")

        while self.running:
            try:
                # Get next frame (with timeout to allow checking running flag)
                try:
                    frame = await asyncio.wait_for(
                        self.analysis_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No frame available, continue loop
                    continue
                
                self.processing = True
                
                logger.debug(f"Analyzing frame: {frame['path'].name}")
                
                try:
                    # Analyze diversity
                    should_cache, embedding = await self._analyze_frame_diversity(frame)
                    
                    if should_cache and embedding is not None:
                        # Add to cache
                        success = await self._cache_frame(frame, embedding)

                        if success:
                            self.frames_cached += 1
                            kf = frame.get('metadata', {}).get('keyframe_num')
                            if kf is not None:
                                self._last_admitted_kf = kf
                            logger.debug(f"Frame cached: {frame['path'].name}")
                        else:
                            self.frames_skipped += 1
                    else:
                        self.frames_skipped += 1

                    self.frames_analyzed += 1

                    # Heartbeat so cache health is visible in logs (the cache
                    # was silently dead for two months once; never again)
                    if self.frames_analyzed % 100 == 0:
                        logger.info(
                            f"[CACHE_HEALTH] analyzed={self.frames_analyzed} "
                            f"cached={self.frames_cached} size={self.cache.size()} "
                            f"floor_rej={self.frames_rejected_floor} "
                            f"sim_rej={self.frames_rejected_similar} "
                            f"queue={self.analysis_queue.qsize()}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error analyzing frame {frame['path'].name}: {e}",
                        exc_info=True
                    )
                
                finally:
                    # Mark task as done
                    self.analysis_queue.task_done()
                    self.processing = False
            
            except asyncio.CancelledError:
                logger.info("CacheAnalysisWorker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache analysis worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error

        logger.info("CacheAnalysisWorker stopped")
    
    def stop(self) -> None:
        """
        Stop the worker gracefully
        
        The worker will finish processing the current frame and then exit.
        """
        logger.info("Stopping CacheAnalysisWorker...")
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics
        
        Returns:
            Dictionary with worker stats
        """
        cache_rate = 0.0
        if self.frames_analyzed > 0:
            cache_rate = self.frames_cached / self.frames_analyzed
        
        return {
            'frames_analyzed': self.frames_analyzed,
            'frames_cached': self.frames_cached,
            'frames_skipped': self.frames_skipped,
            'frames_rejected_floor': self.frames_rejected_floor,
            'frames_rejected_similar': self.frames_rejected_similar,
            'cache_rate': cache_rate,
            'queue_depth': self.analysis_queue.qsize(),
            'is_processing': self.processing,
            'is_running': self.running,
        }

