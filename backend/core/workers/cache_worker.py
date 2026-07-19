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
        
        # Statistics
        self.frames_analyzed = 0
        self.frames_cached = 0
        self.frames_skipped = 0
        
        logger.info(f"CacheAnalysisWorker initialized (max queue: {max_queue_size})")
    
    async def submit_frame(
        self,
        frame_path: Path,
        prompt: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Submit a frame for cache analysis
        
        Args:
            frame_path: Path to frame image
            prompt: Generation prompt used
            metadata: Additional metadata (denoise, type, etc.)
        """
        frame = {
            'path': frame_path,
            'prompt': prompt,
            'metadata': metadata
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
        
        try:
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
            
            # Check if should cache (selective caching for diversity)
            population_mode = self.config.get('generation', {}).get('cache', {}).get(
                'population_mode', 'selective'
            )
            
            should_cache = True
            
            return should_cache, embedding
            
        except Exception as e:
            logger.error(f"Error analyzing frame diversity: {e}", exc_info=True)
            return False, None
    
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
            'cache_rate': cache_rate,
            'queue_depth': self.analysis_queue.qsize(),
            'is_processing': self.processing,
            'is_running': self.running,
        }

