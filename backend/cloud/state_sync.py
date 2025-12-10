"""
Cloud State Synchronization

Handles periodic state snapshots for VPS persistence.
State is pushed every N keyframes and on graceful shutdown.

State includes:
- Last keyframe latent (for interpolation continuity)
- Generation counters (frame, keyframe, theme)
- Cache metadata (LRU state)
- Similarity embeddings (on shutdown only)

Uses msgpack for efficient binary serialization.
"""

import asyncio
import logging
import time
import json
from typing import Optional, Any
from pathlib import Path
import numpy as np

from .websocket_client import VPSWebSocketClient

logger = logging.getLogger(__name__)

# Try to import msgpack, fall back to json if not available
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    logger.warning("msgpack not available, using JSON for state serialization (less efficient)")


class CloudStateSync:
    """
    Handles state synchronization between GPU and VPS
    
    Periodically pushes state snapshots to VPS for:
    - Resume capability after GPU restart
    - Progress tracking
    - Cache state preservation
    """
    
    def __init__(self, websocket_client: VPSWebSocketClient, config: dict):
        """
        Initialize state sync
        
        Args:
            websocket_client: Connected VPS WebSocket client
            config: Cloud configuration dict containing:
                - state_sync.interval_keyframes: Push state every N keyframes
                - state_sync.push_on_shutdown: Always push on graceful shutdown
        """
        self.ws_client = websocket_client
        
        # Configuration
        sync_config = config.get('state_sync', {})
        self.save_interval = sync_config.get('interval_keyframes', 10)
        self.push_on_shutdown = sync_config.get('push_on_shutdown', True)
        
        # State tracking
        self.keyframe_count = 0
        self.last_push_keyframe = 0
        self.last_push_time = 0
        
        # Cached state components
        self._last_latent: Optional[np.ndarray] = None
        self._last_state: Optional[dict] = None
        
        # Statistics
        self.pushes_completed = 0
        self.bytes_pushed = 0
        self.push_times = []
        self.max_timing_samples = 20
        
        logger.info(f"CloudStateSync initialized: interval={self.save_interval} keyframes")
    
    async def on_keyframe_complete(
        self,
        keyframe_latent: Any,  # torch.Tensor
        generation_state: dict,
        cache_manager: Optional[Any] = None,
        similarity_manager: Optional[Any] = None,
    ) -> bool:
        """
        Called after each keyframe generation
        
        Saves state locally and pushes to VPS every N keyframes.
        
        Args:
            keyframe_latent: Latent tensor from VAE encoding
            generation_state: Dict with frame_count, keyframe_count, theme_index, etc.
            cache_manager: Optional cache manager for metadata
            similarity_manager: Optional similarity manager (for full snapshots)
        
        Returns:
            True if state was pushed to VPS
        """
        self.keyframe_count += 1
        
        # Convert latent to numpy for storage
        if hasattr(keyframe_latent, 'cpu'):
            self._last_latent = keyframe_latent.cpu().numpy()
        else:
            self._last_latent = keyframe_latent
        
        self._last_state = generation_state.copy()
        
        # Check if we should push
        if self.keyframe_count - self.last_push_keyframe >= self.save_interval:
            return await self._push_state(
                include_cache=False,  # Light push for periodic saves
                cache_manager=cache_manager,
            )
        
        return False
    
    async def on_shutdown(
        self,
        cache_manager: Optional[Any] = None,
        similarity_manager: Optional[Any] = None,
    ) -> bool:
        """
        Called on graceful shutdown
        
        Pushes full state including cache data and embeddings.
        
        Args:
            cache_manager: Cache manager for metadata
            similarity_manager: Similarity manager for embeddings
        
        Returns:
            True if state was pushed successfully
        """
        if not self.push_on_shutdown:
            return True
        
        logger.info("Pushing final state on shutdown...")
        
        return await self._push_state(
            include_cache=True,
            cache_manager=cache_manager,
            similarity_manager=similarity_manager,
        )
    
    async def _push_state(
        self,
        include_cache: bool = False,
        cache_manager: Optional[Any] = None,
        similarity_manager: Optional[Any] = None,
    ) -> bool:
        """
        Push state snapshot to VPS
        
        Args:
            include_cache: Include cache metadata and embeddings
            cache_manager: Cache manager instance
            similarity_manager: Similarity manager instance
        
        Returns:
            True if pushed successfully
        """
        if not self.ws_client.connected:
            logger.warning("Cannot push state: not connected to VPS")
            return False
        
        if self._last_latent is None or self._last_state is None:
            logger.warning("No state to push")
            return False
        
        start_time = time.time()
        
        try:
            # Build state bundle
            bundle = {
                "latent": self._last_latent.tobytes(),
                "latent_shape": list(self._last_latent.shape),
                "latent_dtype": str(self._last_latent.dtype),
                "state": self._last_state,
                "timestamp": time.time(),
                "keyframe_count": self.keyframe_count,
            }
            
            # Add cache metadata if requested
            if include_cache and cache_manager is not None:
                try:
                    bundle["cache_meta"] = cache_manager.get_metadata()
                except Exception as e:
                    logger.warning(f"Could not get cache metadata: {e}")
            
            # Add similarity embeddings if requested (shutdown only)
            if include_cache and similarity_manager is not None:
                try:
                    embeddings = similarity_manager.serialize()
                    if embeddings:
                        bundle["embeddings"] = embeddings
                except Exception as e:
                    logger.warning(f"Could not serialize embeddings: {e}")
            
            # Serialize
            if HAS_MSGPACK:
                state_bytes = msgpack.packb(bundle, use_bin_type=True)
            else:
                # Fallback: JSON with base64 for binary data
                import base64
                json_bundle = bundle.copy()
                json_bundle["latent"] = base64.b64encode(bundle["latent"]).decode('ascii')
                if "embeddings" in json_bundle:
                    json_bundle["embeddings"] = base64.b64encode(bundle["embeddings"]).decode('ascii')
                state_bytes = json.dumps(json_bundle).encode('utf-8')
            
            # Push to VPS
            success = await self.ws_client.send_state(state_bytes)
            
            push_time = time.time() - start_time
            
            if success:
                self.last_push_keyframe = self.keyframe_count
                self.last_push_time = time.time()
                self.pushes_completed += 1
                self.bytes_pushed += len(state_bytes)
                self._record_timing(push_time)
                
                logger.info(
                    f"Pushed state snapshot: {len(state_bytes)/1024:.1f}KB "
                    f"(keyframe {self.keyframe_count}, {push_time*1000:.1f}ms)"
                )
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to push state: {e}")
            return False
    
    def _record_timing(self, time_seconds: float) -> None:
        """Record a timing sample"""
        self.push_times.append(time_seconds)
        if len(self.push_times) > self.max_timing_samples:
            self.push_times.pop(0)
    
    @property
    def average_push_time_ms(self) -> float:
        """Average push time in milliseconds"""
        if not self.push_times:
            return 0.0
        return sum(self.push_times) / len(self.push_times) * 1000
    
    def get_stats(self) -> dict:
        """Get sync statistics"""
        return {
            "keyframe_count": self.keyframe_count,
            "pushes_completed": self.pushes_completed,
            "bytes_pushed": self.bytes_pushed,
            "bytes_pushed_mb": round(self.bytes_pushed / 1024 / 1024, 2),
            "average_push_time_ms": round(self.average_push_time_ms, 2),
            "last_push_keyframe": self.last_push_keyframe,
            "save_interval": self.save_interval,
        }


def deserialize_state(state_bytes: bytes) -> dict:
    """
    Deserialize a state bundle received from VPS
    
    Args:
        state_bytes: Serialized state bytes
    
    Returns:
        State bundle dict with restored numpy arrays
    """
    if HAS_MSGPACK:
        bundle = msgpack.unpackb(state_bytes, raw=False)
    else:
        import base64
        bundle = json.loads(state_bytes.decode('utf-8'))
        # Decode base64 binary data
        bundle["latent"] = base64.b64decode(bundle["latent"])
        if "embeddings" in bundle:
            bundle["embeddings"] = base64.b64decode(bundle["embeddings"])
    
    # Restore latent as numpy array
    latent_bytes = bundle["latent"]
    latent_shape = tuple(bundle["latent_shape"])
    latent_dtype = np.dtype(bundle["latent_dtype"])
    bundle["latent"] = np.frombuffer(latent_bytes, dtype=latent_dtype).reshape(latent_shape)
    
    return bundle

