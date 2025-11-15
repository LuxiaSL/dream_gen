"""
Keyframe Worker - Async HTTP I/O for ComfyUI Generation

Handles keyframe generation requests via async queue, eliminating the
blocking HTTP polling that previously serialized the pipeline.

This worker is I/O bound (waiting for ComfyUI HTTP/WebSocket), so it
yields to the event loop during the ~2s generation time, allowing
other workers to run concurrently.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class KeyframeWorker:
    """
    Async worker for keyframe generation via ComfyUI
    
    Responsibilities:
    - Maintain queue of generation requests
    - Execute generations via async generator (non-blocking HTTP)
    - Report completions to coordinator
    - Handle errors and retries
    
    Queue Flow:
        Coordinator → submit_request() → request_queue
        ↓
        run() loop processes requests
        ↓
        result_queue → Coordinator reads completions
    
    Usage:
        worker = KeyframeWorker(generator, config)
        
        # Start worker loop
        asyncio.create_task(worker.run())
        
        # Submit generation request
        await worker.submit_request(
            current_image=path,
            keyframe_num=5,
            prompt="ethereal dreamscape..."
        )
        
        # Wait for result
        result = await worker.result_queue.get()
        # {'keyframe_num': 5, 'path': Path(...), 'prompt': '...'}
    """
    
    def __init__(
        self,
        generator,
        frame_buffer,
        config: Dict[str, Any],
        max_queue_size: int = 5
    ):
        """
        Initialize keyframe worker
        
        Args:
            generator: DreamGenerator instance (with async methods)
            frame_buffer: FrameBuffer instance for path management
            config: Configuration dictionary
            max_queue_size: Maximum pending requests (backpressure control)
        """
        self.generator = generator
        self.frame_buffer = frame_buffer
        self.config = config
        
        # Queues
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_queue = asyncio.Queue()
        
        # State
        self.running = False
        self.processing = False
        
        # Statistics
        self.requests_processed = 0
        self.requests_failed = 0
        self.total_generation_time = 0.0
        
        logger.info(f"KeyframeWorker initialized (max queue: {max_queue_size})")
    
    async def submit_request(
        self,
        current_image: Path,
        keyframe_num: int,
        sequence_num: int,
        prompt: str
    ) -> None:
        """
        Submit a keyframe generation request
        
        This will block if the queue is full (backpressure), preventing
        memory exhaustion from too many pending requests.
        
        Args:
            current_image: Path to current image (for img2img)
            keyframe_num: Keyframe number to generate
            sequence_num: Sequence number in buffer (pre-registered by orchestrator)
            prompt: Generation prompt
        """
        request = {
            'keyframe_num': keyframe_num,
            'sequence_num': sequence_num,
            'current_image': current_image,
            'prompt': prompt
        }
        
        # This will block if queue is full (backpressure)
        await self.request_queue.put(request)
        
        logger.debug(
            f"Submitted keyframe request: KF{keyframe_num} "
            f"(queue depth: {self.request_queue.qsize()})"
        )
    
    async def run(self) -> None:
        """
        Main worker loop
        
        Processes generation requests from queue and reports completions.
        Runs until stop() is called.
        """
        self.running = True
        logger.info("KeyframeWorker started")
        
        while self.running:
            try:
                # Get next request (with timeout to allow checking running flag)
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No request available, continue loop
                    continue
                
                self.processing = True
                
                # Extract request data
                keyframe_num = request['keyframe_num']
                sequence_num = request['sequence_num']
                current_image = request['current_image']
                prompt = request['prompt']
                
                logger.info(f"Processing keyframe request: KF{keyframe_num} (seq {sequence_num})")
                
                # Mark as generating in buffer
                self.frame_buffer.mark_generating(sequence_num)
                
                # Get denoise from config
                denoise = self.config['generation']['hybrid']['keyframe_denoise']
                
                # Generate keyframe (ASYNC - doesn't block event loop!)
                import time
                start_time = time.time()
                
                try:
                    keyframe_path = await self.generator.generate_from_image_async(
                        image_path=current_image,
                        prompt=prompt,
                        denoise=denoise
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if keyframe_path and keyframe_path.exists():
                        # Move to keyframe directory with proper naming
                        # Generator outputs to: output/frame_XXXXX.png
                        # We need: output/keyframes/keyframe_XXX.png
                        target_path = self.frame_buffer.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
                        
                        # Move to target location (removes duplicate from output root)
                        import shutil
                        shutil.move(str(keyframe_path), str(target_path))
                        
                        logger.debug(f"Moved keyframe: {keyframe_path.name} -> {target_path}")
                        
                        # Success! Report to coordinator
                        result = {
                            'keyframe_num': keyframe_num,
                            'sequence_num': sequence_num,
                            'path': target_path,  # Use the final path
                            'prompt': prompt,
                            'generation_time': elapsed
                        }
                        
                        await self.result_queue.put(result)
                        
                        self.requests_processed += 1
                        self.total_generation_time += elapsed
                        
                        logger.info(
                            f"[OK] Keyframe {keyframe_num} generated in {elapsed:.2f}s "
                            f"(total: {self.requests_processed})"
                        )
                    else:
                        # Generation failed
                        logger.error(f"Keyframe {keyframe_num} generation failed")
                        self.requests_failed += 1
                        
                        # Could implement retry logic here
                        # For now, just report failure and continue
                
                except Exception as e:
                    logger.error(f"Error generating keyframe {keyframe_num}: {e}", exc_info=True)
                    self.requests_failed += 1
                
                finally:
                    # Mark task as done
                    self.request_queue.task_done()
                    self.processing = False
            
            except asyncio.CancelledError:
                logger.info("KeyframeWorker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in keyframe worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error
        
        logger.info("KeyframeWorker stopped")
    
    def stop(self) -> None:
        """
        Stop the worker gracefully
        
        The worker will finish processing the current request and then exit.
        """
        logger.info("Stopping KeyframeWorker...")
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics
        
        Returns:
            Dictionary with worker stats
        """
        avg_time = 0.0
        if self.requests_processed > 0:
            avg_time = self.total_generation_time / self.requests_processed
        
        return {
            'requests_processed': self.requests_processed,
            'requests_failed': self.requests_failed,
            'avg_generation_time': avg_time,
            'queue_depth': self.request_queue.qsize(),
            'result_queue_depth': self.result_queue.qsize(),
            'is_processing': self.processing,
            'is_running': self.running
        }

