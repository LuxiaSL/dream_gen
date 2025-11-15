"""
Shared Resources for Async Workers

Provides thread-safe access to shared resources (VAE, etc.) across
concurrent async workers to prevent CUDA conflicts and race conditions.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Any
import torch

logger = logging.getLogger(__name__)


class SharedVAEAccess:
    """
    Thread-safe async wrapper for VAE encoder/decoder
    
    Provides exclusive access to the VAE (LatentEncoder) via asyncio lock
    to prevent CUDA context conflicts when multiple workers need VAE access.
    
    Use Cases:
    - InterpolationWorker: Frequent encode/decode for frame generation
    - Injection logic: Occasional VAE blending (inline in orchestrator)
    
    Lock Contention:
    - InterpolationWorker: ~40 ops × 50ms = 2s total
    - Injection: ~15% probability × 150ms = rare, short
    - Expected contention: minimal (<50ms waits)
    
    Usage:
        vae_access = SharedVAEAccess(latent_encoder)
        
        # Async encode
        latent = await vae_access.encode_async(image_path, for_interpolation=True)
        
        # Async decode
        image = await vae_access.decode_async(latent, upscale_to_target=True)
    """
    
    def __init__(self, latent_encoder):
        """
        Initialize shared VAE access
        
        Args:
            latent_encoder: LatentEncoder instance (from interpolation.latent_encoder)
        """
        self.encoder = latent_encoder
        self.lock = asyncio.Lock()
        
        # Statistics for monitoring lock contention
        self.lock_wait_times = []
        self.lock_acquisitions = 0
        self.max_wait_time = 0.0
        
        logger.info("SharedVAEAccess initialized with asyncio lock")
    
    async def encode_async(
        self, 
        image: Path, 
        for_interpolation: bool = False
    ) -> torch.Tensor:
        """
        Encode image to latent space (async, thread-safe)
        
        Acquires exclusive lock before calling VAE encoder to prevent
        CUDA context conflicts from concurrent access.
        
        Args:
            image: Path to image file
            for_interpolation: If True, may use lower resolution for performance
            
        Returns:
            Latent tensor (on GPU)
        """
        import time
        wait_start = time.time()
        
        async with self.lock:
            # Track lock wait time
            wait_time = time.time() - wait_start
            self.lock_wait_times.append(wait_time)
            self.lock_acquisitions += 1
            self.max_wait_time = max(self.max_wait_time, wait_time)
            
            # Log if significant contention detected
            if wait_time > 0.1:  # 100ms threshold
                logger.warning(
                    f"VAE lock contention: waited {wait_time*1000:.1f}ms "
                    f"(acquisitions: {self.lock_acquisitions})"
                )
            
            # Run encode in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            latent = await loop.run_in_executor(
                None,
                self.encoder.encode,
                image,
                for_interpolation
            )
            
            return latent
    
    async def decode_async(
        self,
        latent: torch.Tensor,
        upscale_to_target: bool = False
    ) -> Any:
        """
        Decode latent to image (async, thread-safe)
        
        Acquires exclusive lock before calling VAE decoder to prevent
        CUDA context conflicts from concurrent access.
        
        Args:
            latent: Latent tensor (on GPU)
            upscale_to_target: If True, upscale to target resolution
            
        Returns:
            PIL Image
        """
        import time
        wait_start = time.time()
        
        async with self.lock:
            # Track lock wait time
            wait_time = time.time() - wait_start
            self.lock_wait_times.append(wait_time)
            self.lock_acquisitions += 1
            self.max_wait_time = max(self.max_wait_time, wait_time)
            
            # Log if significant contention detected
            if wait_time > 0.1:  # 100ms threshold
                logger.warning(
                    f"VAE lock contention: waited {wait_time*1000:.1f}ms "
                    f"(acquisitions: {self.lock_acquisitions})"
                )
            
            # Run decode in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                self.encoder.decode,
                latent,
                upscale_to_target
            )
            
            return image
    
    def get_lock_stats(self) -> dict:
        """
        Get lock contention statistics
        
        Returns:
            Dictionary with lock statistics:
            - acquisitions: Total number of lock acquisitions
            - avg_wait_time_ms: Average wait time in milliseconds
            - max_wait_time_ms: Maximum wait time in milliseconds
            - recent_wait_times_ms: Last 10 wait times
        """
        if not self.lock_wait_times:
            return {
                "acquisitions": 0,
                "avg_wait_time_ms": 0.0,
                "max_wait_time_ms": 0.0,
                "recent_wait_times_ms": []
            }
        
        avg_wait = sum(self.lock_wait_times) / len(self.lock_wait_times)
        recent_waits = self.lock_wait_times[-10:]
        
        return {
            "acquisitions": self.lock_acquisitions,
            "avg_wait_time_ms": avg_wait * 1000,
            "max_wait_time_ms": self.max_wait_time * 1000,
            "recent_wait_times_ms": [w * 1000 for w in recent_waits]
        }
    
    def reset_stats(self):
        """Reset lock statistics"""
        self.lock_wait_times = []
        self.lock_acquisitions = 0
        self.max_wait_time = 0.0
        logger.debug("Lock statistics reset")

