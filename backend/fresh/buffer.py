"""
Fresh Frame Buffer - Per-Template Pre-Generation System

Maintains one pre-generated fresh frame per template. At startup, generates
txt2img frames for ALL templates. When a frame is consumed, regenerates that
template's frame with fresh components in the background.

Key Design:
- Buffer = Dict[template_id, BufferedFrame] - one frame per template
- Startup: Populate entire buffer (one frame per template)
- Selection: Random template excluding recently used (pool resets when all used)
- After consumption: Regenerate that template with new components
- No seed fallback - fresh frames only

Integration:
- Orchestrator calls populate_all() at startup before generation loop
- At seed injection: call select_and_consume() to get random template's frame
- Background task regenerates consumed template's frame
- Template + components travel with frame for mutation starting point
"""

import asyncio
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    """
    A pre-generated fresh frame with its context
    
    Attributes:
        path: Path to the generated image file
        template_id: Template ID used for generation
        components: Component values used (e.g. {'color': 'cyan', 'mood': 'ethereal'})
        prompt: Full positive prompt string
        negative_prompt: Full negative prompt string
        generated_at: Timestamp when frame was generated
    """
    path: Path
    template_id: str
    components: Dict[str, str]
    prompt: str
    negative_prompt: str
    generated_at: float


class FreshFrameBuffer:
    """
    Per-template fresh frame buffer for infinite generation
    
    Maintains one pre-generated frame for each template. This ensures:
    - Instant template switches (frame already generated)
    - Full template coverage (all templates available)
    - Fresh aesthetics (regenerate with new components after use)
    
    Usage:
        buffer = FreshFrameBuffer(generator, prompt_system, config)
        
        # At startup (REQUIRED before generation loop):
        await buffer.populate_all()
        
        # At seed injection:
        frame = await buffer.select_and_consume()
        prompt_system.switch_template(frame.template_id, frame.components)
        
        # Background regeneration happens automatically
    """
    
    def __init__(
        self,
        generator,  # DreamGenerator instance
        prompt_system,  # CombinatorialPromptSystem instance
        config: Dict[str, Any],
        output_dir: Optional[Path] = None
    ):
        """
        Initialize fresh frame buffer
        
        Args:
            generator: DreamGenerator instance with generate_from_prompt() method
            prompt_system: CombinatorialPromptSystem instance
            config: Configuration dictionary
            output_dir: Directory for fresh frame output (defaults to frames/fresh/)
        """
        self.generator = generator
        self.prompt_system = prompt_system
        self.config = config
        
        # Output directory for fresh frames
        if output_dir:
            self.output_dir = output_dir
        else:
            base_output = Path(config['system']['output_dir'])
            self.output_dir = base_output / 'frames' / 'fresh'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === Per-Template Buffer ===
        # {template_id: BufferedFrame}
        self._buffer: Dict[str, BufferedFrame] = {}
        
        # Track which templates are currently being regenerated
        self._regenerating: Set[str] = set()
        
        # === Selection Pool (for random-excluding-recent) ===
        # Templates that haven't been used in current cycle
        self._available_pool: Set[str] = set()
        
        # Templates used in current cycle (clears when all used)
        self._used_this_cycle: Set[str] = set()
        
        # === Locks ===
        self._buffer_lock = asyncio.Lock()
        self._regen_lock = asyncio.Lock()
        
        # === Statistics ===
        self._frame_counter: int = 0
        self._total_generated: int = 0
        self._total_consumed: int = 0
        self._generation_times: List[float] = []
        self._startup_complete: bool = False
        
        # === Config ===
        self._max_retries = 3  # Crash after this many consecutive failures
        
        logger.info("FreshFrameBuffer initialized (per-template mode)")
        logger.info(f"  Output directory: {self.output_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STARTUP - Populate entire buffer
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def populate_all(self) -> None:
        """
        Populate buffer with one fresh frame per template
        
        MUST be called before starting generation loop.
        Generates txt2img frame for each available template.
        
        Raises:
            RuntimeError: If generation fails repeatedly
        """
        templates = self.prompt_system.get_available_templates()
        
        if not templates:
            raise RuntimeError("No templates available in prompt system")
        
        logger.info("=" * 60)
        logger.info(f"[FRESH] Populating buffer for {len(templates)} templates...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for i, template_id in enumerate(templates):
            logger.info(f"[FRESH] Generating {i+1}/{len(templates)}: '{template_id}'")
            
            success = await self._generate_for_template(template_id)
            
            if not success:
                raise RuntimeError(
                    f"Failed to generate fresh frame for template '{template_id}'. "
                    "Check ComfyUI connection and model availability."
                )
        
        elapsed = time.time() - start_time
        
        # Initialize selection pool with all templates
        self._available_pool = set(templates)
        self._used_this_cycle = set()
        self._startup_complete = True
        
        logger.info("=" * 60)
        logger.info(f"[FRESH] Buffer populated: {len(self._buffer)} frames in {elapsed:.1f}s")
        logger.info(f"[FRESH] Avg generation time: {elapsed/len(templates):.2f}s per frame")
        logger.info("=" * 60)
    
    async def _generate_for_template(self, template_id: str) -> bool:
        """
        Generate fresh frame for a specific template
        
        Args:
            template_id: Template to generate for
            
        Returns:
            True if successful, False on failure
        """
        template = self.prompt_system.templates.get(template_id)
        if not template:
            logger.error(f"[FRESH] Template '{template_id}' not found")
            return False
        
        retries = 0
        while retries < self._max_retries:
            try:
                start_time = time.time()
                
                # Build prompt for this template
                prompt, negative, components = self._build_prompt_for_template(template)
                
                logger.debug(f"[FRESH] Template '{template_id}' prompt: {prompt[:80]}...")
                
                # Generate txt2img
                result_path = await self._generate_txt2img(prompt, negative)
                
                if result_path and result_path.exists():
                    elapsed = time.time() - start_time
                    self._generation_times.append(elapsed)
                    if len(self._generation_times) > 50:
                        self._generation_times.pop(0)
                    
                    # Copy to our output directory with template-specific naming
                    self._frame_counter += 1
                    final_path = self.output_dir / f"fresh_{template_id}_{self._frame_counter:05d}.png"
                    shutil.copy2(result_path, final_path)
                    
                    # Store in buffer
                    async with self._buffer_lock:
                        self._buffer[template_id] = BufferedFrame(
                            path=final_path,
                            template_id=template_id,
                            components=components,
                            prompt=prompt,
                            negative_prompt=negative,
                            generated_at=time.time()
                        )
                    
                    self._total_generated += 1
                    
                    logger.info(
                        f"[FRESH] Generated '{template_id}' in {elapsed:.2f}s"
                    )
                    return True
                else:
                    logger.warning(f"[FRESH] Generation returned no path for '{template_id}', retry {retries+1}/{self._max_retries}")
                    retries += 1
                    
            except Exception as e:
                logger.error(f"[FRESH] Generation error for '{template_id}': {e}", exc_info=True)
                retries += 1
        
        logger.error(f"[FRESH] Failed to generate '{template_id}' after {self._max_retries} retries")
        return False
    
    def _build_prompt_for_template(self, template) -> tuple:
        """
        Build prompt string for a template without affecting current state
        
        Creates a one-off prompt for pre-generation without switching
        the prompt system's active template.
        
        Args:
            template: Template object to build prompt for
            
        Returns:
            Tuple of (prompt, negative_prompt, components_dict)
        """
        # Track what we select for each category
        selected_components: Dict[str, str] = {}
        category_pools: Dict[str, list] = {}
        category_index: Dict[str, int] = {}
        
        # Pre-select components for each category needed by template
        for category, count, _ in template.slots:
            if category not in self.prompt_system.components:
                continue
            
            if category not in category_pools:
                pool = self.prompt_system.components[category]
                total_needed = sum(c for cat, c, _ in template.slots if cat == category)
                category_pools[category] = random.sample(pool, min(total_needed, len(pool)))
        
        # Build the prompt by filling slots
        prompt = template.structure
        
        for category, count, separator in template.slots:
            if category not in category_pools:
                continue
            
            start_idx = category_index.get(category, 0)
            available = category_pools[category]
            end_idx = min(start_idx + count, len(available))
            slot_components = available[start_idx:end_idx]
            category_index[category] = end_idx
            
            if slot_components:
                replacement = separator.join(c.word for c in slot_components)
                # Track first component of each category for components dict
                if category not in selected_components and slot_components:
                    selected_components[category] = slot_components[0].word
            else:
                replacement = f"[missing {category}]"
            
            # Build the pattern to replace
            if count > 1:
                if separator != " ":
                    pattern = f"{{{category}:{count}:{separator}}}"
                else:
                    pattern = f"{{{category}:{count}}}"
            else:
                pattern = f"{{{category}}}"
            
            prompt = prompt.replace(pattern, replacement, 1)
        
        # Build negative prompt from semantic opposites
        negatable_categories = {'adjective', 'color', 'style', 'mood'}
        opposites = []
        
        for category, components in category_pools.items():
            if category in negatable_categories:
                for comp in components:
                    if comp.opposite:
                        opposites.append(comp.opposite)
        
        base_negatives = ["low quality", "blurry", "text", "watermark"]
        negative_prompt = ", ".join(opposites + base_negatives)
        
        return prompt, negative_prompt, selected_components
    
    async def _generate_txt2img(self, prompt: str, negative_prompt: str) -> Optional[Path]:
        """
        Generate a fresh frame using txt2img
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            
        Returns:
            Path to generated image, or None on failure
        """
        # Check if generator has async txt2img method
        if hasattr(self.generator, 'generate_from_prompt_async'):
            return await self.generator.generate_from_prompt_async(
                prompt=prompt,
                negative_prompt=negative_prompt
            )
        else:
            # Fall back to sync method in executor to not block event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generator.generate_from_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt
                )
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SELECTION & CONSUMPTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def select_and_consume(self) -> BufferedFrame:
        """
        Select random template and consume its buffered frame
        
        Selection uses random-excluding-recently-used strategy:
        - Pick from templates not used in current cycle
        - When all templates used once, reset pool
        
        After consumption, triggers background regeneration for that template.
        
        Returns:
            BufferedFrame with path, template_id, components, prompts
            
        Raises:
            RuntimeError: If buffer not populated or generation fails
        """
        if not self._startup_complete:
            raise RuntimeError(
                "Fresh frame buffer not populated. "
                "Call populate_all() before starting generation."
            )
        
        # Check if we need to wait (outside lock to avoid deadlock)
        needs_wait = False
        async with self._buffer_lock:
            # Reset pool if all templates used
            if not self._available_pool:
                logger.info("[FRESH] All templates used once, resetting selection pool")
                self._available_pool = set(self._buffer.keys())
                self._used_this_cycle = set()
            
            # Filter to templates that have ready frames
            ready_templates = [
                t for t in self._available_pool 
                if t in self._buffer and self._buffer[t] is not None
            ]
            
            if not ready_templates:
                # All available templates are being regenerated - need to wait
                # Set flag to wait OUTSIDE lock to avoid deadlock with regeneration tasks
                logger.warning("[FRESH] No ready frames, will wait for regeneration...")
                needs_wait = True
        
        # Wait outside the lock if needed
        if needs_wait:
            frame = await self._wait_for_any_frame()
            if frame is None:
                raise RuntimeError("No fresh frames available and regeneration failed")
            return frame
        
        # Normal path: select and consume with lock
        async with self._buffer_lock:
            # Re-check ready templates (state may have changed)
            ready_templates = [
                t for t in self._available_pool 
                if t in self._buffer and self._buffer[t] is not None
            ]
            
            if not ready_templates:
                # Race condition: became empty between checks, recurse to wait path
                logger.warning("[FRESH] Ready templates emptied during selection, retrying...")
        
        # Handle race condition by recursing (will hit wait path)
        if not ready_templates:
            return await self.select_and_consume()
        
        # Perform selection inside lock
        async with self._buffer_lock:
            # Random selection from available pool
            selected_template = random.choice(ready_templates)
            
            # Consume the frame
            frame = self._buffer[selected_template]
            self._buffer[selected_template] = None  # Mark as consumed
            
            # Update selection tracking
            self._available_pool.discard(selected_template)
            self._used_this_cycle.add(selected_template)
            
            self._total_consumed += 1
            age = time.time() - frame.generated_at
            
            logger.info(
                f"[FRESH] Consumed '{selected_template}' (age: {age:.1f}s, "
                f"pool: {len(self._available_pool)}/{len(self._buffer)} remaining)"
            )
        
        # Trigger background regeneration for consumed template
        asyncio.create_task(self._regenerate_template(selected_template))
        
        return frame
    
    async def _wait_for_any_frame(self) -> Optional[BufferedFrame]:
        """
        Wait for any frame to become available
        
        Called when all ready frames are consumed and some are regenerating.
        Polls buffer until a frame is ready.
        
        Returns:
            BufferedFrame if one becomes available, None on timeout
        """
        timeout = 120  # 2 minute max wait
        start_time = time.time()
        poll_interval = 0.5
        
        logger.info("[FRESH] Waiting for frame regeneration...")
        
        while time.time() - start_time < timeout:
            # Check if any frame is ready (with lock for thread safety)
            async with self._buffer_lock:
                for template_id, frame in self._buffer.items():
                    if frame is not None and template_id in self._available_pool:
                        # Found a ready frame - consume it
                        self._buffer[template_id] = None
                        self._available_pool.discard(template_id)
                        self._used_this_cycle.add(template_id)
                        self._total_consumed += 1
                        
                        logger.info(f"[FRESH] Frame ready after wait: '{template_id}'")
                        
                        # Trigger regeneration (outside lock via task)
                        asyncio.create_task(self._regenerate_template(template_id))
                        
                        return frame
            
            # Sleep outside lock to allow regeneration tasks to complete
            await asyncio.sleep(poll_interval)
        
        logger.error(f"[FRESH] Timeout waiting for frame ({timeout}s)")
        return None
    
    async def _regenerate_template(self, template_id: str) -> None:
        """
        Regenerate frame for a consumed template (background task)
        
        Called automatically after a frame is consumed.
        Uses fresh components for variety.
        
        Args:
            template_id: Template to regenerate
        """
        # Prevent concurrent regeneration of same template
        async with self._regen_lock:
            if template_id in self._regenerating:
                logger.debug(f"[FRESH] '{template_id}' already regenerating, skipping")
                return
            self._regenerating.add(template_id)
        
        try:
            logger.info(f"[FRESH] Regenerating '{template_id}' with fresh components...")
            
            success = await self._generate_for_template(template_id)
            
            if success:
                logger.info(f"[FRESH] Regenerated '{template_id}'")
            else:
                logger.error(f"[FRESH] Failed to regenerate '{template_id}'")
                
        finally:
            async with self._regen_lock:
                self._regenerating.discard(template_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & STATS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def is_ready(self) -> bool:
        """
        Check if buffer has at least one ready frame
        
        Returns:
            True if at least one frame is available
        """
        return any(f is not None for f in self._buffer.values())
    
    def is_fully_populated(self) -> bool:
        """
        Check if all templates have ready frames
        
        Returns:
            True if all templates have frames
        """
        return all(f is not None for f in self._buffer.values())
    
    def get_ready_count(self) -> int:
        """Get count of ready frames"""
        return sum(1 for f in self._buffer.values() if f is not None)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer stats
        """
        avg_gen_time = (
            sum(self._generation_times) / len(self._generation_times)
            if self._generation_times else 0.0
        )
        
        ready_frames = [t for t, f in self._buffer.items() if f is not None]
        regenerating = list(self._regenerating)
        
        return {
            "startup_complete": self._startup_complete,
            "total_templates": len(self._buffer),
            "ready_count": len(ready_frames),
            "ready_templates": ready_frames,
            "regenerating": regenerating,
            "available_pool_size": len(self._available_pool),
            "used_this_cycle": len(self._used_this_cycle),
            "total_generated": self._total_generated,
            "total_consumed": self._total_consumed,
            "avg_generation_time": avg_gen_time,
        }
    
    def clear(self) -> None:
        """
        Clear the buffer
        
        Useful for shutdown or when changing generation parameters.
        """
        logger.info("[FRESH] Clearing buffer")
        self._buffer.clear()
        self._available_pool.clear()
        self._used_this_cycle.clear()
        self._startup_complete = False
    
    def peek(self) -> Optional[BufferedFrame]:
        """
        Peek at any available frame without consuming
        
        Returns:
            BufferedFrame if any available, None otherwise
        """
        for frame in self._buffer.values():
            if frame is not None:
                return frame
        return None
