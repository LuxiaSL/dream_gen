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

from utils.vram_profiler import dump_vram_on_oom

logger = logging.getLogger(__name__)


class KeyframeWorker:
    """
    Async worker for keyframe generation via ComfyUI
    
    Responsibilities:
    - Maintain queue of generation requests
    - Execute generations via async generator (non-blocking HTTP)
    - Report completions to coordinator
    - Handle errors and retries with exponential backoff
    
    Queue Flow:
        Coordinator → submit_request() → request_queue
        ↓
        run() loop processes requests (with retries)
        ↓
        result_queue → Coordinator reads completions (success or failure)
    
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
        # Success: {'keyframe_num': 5, 'path': Path(...), 'prompt': '...', 'success': True}
        # Failure: {'keyframe_num': 5, 'success': False, 'error': '...'}
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 10.0  # seconds
    
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
        
        # === Retry state tracking ===
        # When in retry mode, we must protect the source image from cleanup
        self.in_retry_mode = False
        self.retry_source_image: Optional[Path] = None  # Protected during retries
        self.current_request: Optional[Dict[str, Any]] = None  # Cached for resubmit
        
        # Statistics
        self.requests_processed = 0
        self.requests_failed = 0
        self.requests_retried = 0
        self.total_generation_time = 0.0
        
        logger.info(f"KeyframeWorker initialized (max queue: {max_queue_size}, max retries: {self.MAX_RETRIES})")
    
    def is_source_image_protected(self, image_path: Path) -> bool:
        """
        Check if an image is protected from cleanup (needed for retry).
        
        Call this before deleting any keyframe image to ensure we don't
        delete the source image while retrying generation.
        
        Args:
            image_path: Path to check
            
        Returns:
            True if this image should NOT be deleted (protected for retry)
        """
        if not self.in_retry_mode or self.retry_source_image is None:
            return False
        
        # Check if paths match (normalize for comparison)
        try:
            return image_path.resolve() == self.retry_source_image.resolve()
        except Exception:
            return str(image_path) == str(self.retry_source_image)
    
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
        Implements retry logic with exponential backoff for transient errors.
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
                
                # === Cache request for potential resubmit ===
                self.current_request = request.copy()
                
                logger.info(f"Processing keyframe request: KF{keyframe_num} (seq {sequence_num})")
                
                # Mark as generating in buffer
                self.frame_buffer.mark_generating(sequence_num)
                
                # Get denoise from config
                denoise = self.config['generation']['hybrid']['keyframe_denoise']
                
                # Generate keyframe with retry logic
                import time
                import shutil
                
                success = False
                last_error = None
                total_elapsed = 0.0
                
                for attempt in range(self.MAX_RETRIES):
                    # === Enter retry mode to protect source image ===
                    if attempt > 0:
                        self.in_retry_mode = True
                        self.retry_source_image = current_image
                        logger.info(f"  Retry mode: protecting source image {current_image.name}")
                    
                    start_time = time.time()
                    keyframe_path = None
                    
                    try:
                        # Wrap generation in timeout to prevent infinite hangs
                        # ComfyUI sometimes hangs on WebSocket wait
                        generation_timeout = self.config.get('performance', {}).get('generation_timeout', 180)
                        
                        try:
                            keyframe_path = await asyncio.wait_for(
                                self.generator.generate_from_image_async(
                                    image_path=current_image,
                                    prompt=prompt,
                                    denoise=denoise
                                ),
                                timeout=generation_timeout
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                f"Keyframe {keyframe_num} generation timed out after {generation_timeout}s "
                                f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                            )
                            last_error = f"Generation timeout after {generation_timeout}s"
                            
                            # Dump VRAM diagnostics on timeout (likely GPU contention)
                            dump_vram_on_oom(
                                context=f"Keyframe {keyframe_num} generation timeout",
                                exception=None
                            )
                            
                            # Recovery handled below in common path
                            keyframe_path = None
                        
                        elapsed = time.time() - start_time
                        total_elapsed += elapsed
                        
                        if keyframe_path and keyframe_path.exists():
                            # Move to keyframe directory with proper naming
                            target_path = self.frame_buffer.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
                            shutil.move(str(keyframe_path), str(target_path))
                            
                            logger.debug(f"Moved keyframe: {keyframe_path.name} -> {target_path}")
                            
                            # === SUCCESS - Clear retry state ===
                            self.in_retry_mode = False
                            self.retry_source_image = None
                            self.current_request = None
                            
                            # Report to coordinator
                            result = {
                                'keyframe_num': keyframe_num,
                                'sequence_num': sequence_num,
                                'path': target_path,
                                'prompt': prompt,
                                'generation_time': total_elapsed,
                                'success': True,
                                'retries': attempt
                            }
                            
                            await self.result_queue.put(result)
                            
                            self.requests_processed += 1
                            self.total_generation_time += total_elapsed
                            
                            if attempt > 0:
                                logger.info(
                                    f"[OK] Keyframe {keyframe_num} generated in {total_elapsed:.2f}s "
                                    f"(after {attempt} retries, total: {self.requests_processed})"
                                )
                            else:
                                logger.info(
                                    f"[OK] Keyframe {keyframe_num} generated in {elapsed:.2f}s "
                                    f"(total: {self.requests_processed})"
                                )
                            
                            success = True
                            break
                        else:
                            # Generation returned None or file doesn't exist
                            last_error = last_error or "Generation returned no output"
                            logger.warning(f"Keyframe {keyframe_num} attempt {attempt + 1}/{self.MAX_RETRIES} failed: {last_error}")
                            # Fall through to common recovery path
                    
                    except Exception as e:
                        elapsed = time.time() - start_time
                        total_elapsed += elapsed
                        last_error = str(e)
                        logger.error(f"Keyframe {keyframe_num} attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                    
                    # === COMMON RECOVERY PATH FOR ALL FAILURES ===
                    # This handles: timeout, no output, exceptions - ALL failure types
                    if attempt < self.MAX_RETRIES - 1:
                        logger.warning(f"Attempting ComfyUI recovery before retry {attempt + 2}...")
                        
                        # Interrupt and clear ComfyUI queue
                        try:
                            self.generator.interrupt_and_clear_queue()
                        except Exception as recover_err:
                            logger.error(f"ComfyUI recovery failed: {recover_err}")
                        
                        # Verify ComfyUI is still responsive
                        if not self.generator.is_comfyui_responsive():
                            logger.error("ComfyUI not responding! Cannot retry.")
                            last_error = "ComfyUI unresponsive"
                            break  # Don't retry if ComfyUI is down
                        
                        # Exponential backoff before retry
                        retry_delay = min(5.0 * (2 ** attempt), 30.0)
                        logger.info(f"Waiting {retry_delay:.0f}s before retry...")
                        self.requests_retried += 1
                        await asyncio.sleep(retry_delay)
                
                # === FINAL CLEANUP ===
                # Clear retry state regardless of outcome
                self.in_retry_mode = False
                self.retry_source_image = None
                
                # If all retries failed, report failure to orchestrator
                if not success:
                    logger.error(f"[FAIL] Keyframe {keyframe_num} failed after {self.MAX_RETRIES} attempts: {last_error}")
                    self.requests_failed += 1
                    
                    # Final ComfyUI cleanup before reporting failure
                    try:
                        self.generator.interrupt_and_clear_queue()
                    except Exception:
                        pass  # Best effort
                    
                    # CRITICAL: Report failure so orchestrator can recover
                    failure_result = {
                        'keyframe_num': keyframe_num,
                        'sequence_num': sequence_num,
                        'success': False,
                        'error': last_error,
                        'retries': self.MAX_RETRIES,
                        'source_image': str(current_image)  # Include for recovery info
                    }
                    await self.result_queue.put(failure_result)
                
                # Clear cached request
                self.current_request = None
                
                # Mark task as done
                self.request_queue.task_done()
                self.processing = False
            
            except asyncio.CancelledError:
                logger.info("KeyframeWorker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in keyframe worker loop: {e}", exc_info=True)
                self.processing = False
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
            'requests_retried': self.requests_retried,
            'avg_generation_time': avg_time,
            'queue_depth': self.request_queue.qsize(),
            'result_queue_depth': self.result_queue.qsize(),
            'is_processing': self.processing,
            'is_running': self.running
        }

