"""
Fresh Frame Buffer

Pre-generates the next fresh frame during idle periods so template switches 
are instantaneous. When the orchestrator has buffer capacity and is waiting,
this buffer generates a txt2img frame with a NEW template ready for the next
seed injection trigger.

Key Design:
- Buffer holds exactly ONE pre-generated frame (simplicity > complexity)
- Generation happens during orchestrator idle/throttle periods
- Frame includes template context for proper cache/detector management
- Fallback to synchronous generation if buffer not ready (shouldn't happen normally)

Integration:
- Orchestrator calls ensure_ready() during throttle periods
- inject_seed_frame() calls consume() to get the buffered frame
- Template switch coordination happens atomically at consume time

Performance:
- No visible delay on template switch (frame already generated)
- Background generation utilizes otherwise-idle GPU time
- Single-frame buffer prevents resource waste
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    Pre-generates fresh frames for seamless template switching
    
    The buffer maintains at most ONE pre-generated frame. This simplicity
    is intentional: we only need one frame ready for the next template switch.
    
    Responsibilities:
    - Pre-generate next fresh frame during idle time (txt2img)
    - Select a NEW template different from current
    - Track template context (template_id, components, prompt)
    - Provide instant frame consumption for seed injection
    - Coordinate with prompt system for template selection
    
    Usage:
        buffer = FreshFrameBuffer(generator=gen, prompt_system=prompts, config=config)
        
        # During idle periods:
        await buffer.ensure_ready()
        
        # At seed injection:
        frame = buffer.consume()
        if frame:
            # Use frame.path, switch to frame.template_id
            prompt_system.switch_template(frame.template_id, frame.components)
            cache_manager.switch_template(frame.template_id)
            collapse_detector.reset_for_template(frame.template_id)
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
        
        # Buffer state
        self._buffered_frame: Optional[BufferedFrame] = None
        self._is_generating: bool = False
        self._generation_lock = asyncio.Lock()
        self._frame_counter: int = 0
        
        # Statistics
        self._total_generated: int = 0
        self._total_consumed: int = 0
        self._generation_times: list = []
        
        logger.info("FreshFrameBuffer initialized")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def is_ready(self) -> bool:
        """
        Check if buffer has a ready frame
        
        Returns:
            True if a frame is buffered and ready for consumption
        """
        return self._buffered_frame is not None
    
    def is_generating(self) -> bool:
        """
        Check if currently generating a frame
        
        Returns:
            True if a generation is in progress
        """
        return self._is_generating
    
    async def ensure_ready(self) -> bool:
        """
        Ensure buffer has a ready frame, generating if needed
        
        Called during orchestrator idle/throttle periods. If buffer is empty
        and not currently generating, starts background generation.
        
        This is idempotent: calling multiple times is safe.
        
        Returns:
            True if buffer is now ready (or will be soon)
        """
        # Already have a frame
        if self._buffered_frame is not None:
            return True
        
        # Already generating
        if self._is_generating:
            logger.debug("Fresh frame generation already in progress")
            return True
        
        # Start generation
        async with self._generation_lock:
            # Double-check after acquiring lock
            if self._buffered_frame is not None or self._is_generating:
                return True
            
            self._is_generating = True
        
        try:
            logger.info("[FRESH] Starting pre-generation of next fresh frame...")
            start_time = time.time()
            
            # Select next template (different from current)
            current_template_id = self.prompt_system.get_current_template_id()
            available_templates = self.prompt_system.get_available_templates()
            
            # Pick a different template
            candidates = [t for t in available_templates if t != current_template_id]
            if not candidates:
                candidates = available_templates
            
            import random
            next_template_id = random.choice(candidates) if candidates else current_template_id
            
            # Generate prompt for the new template WITHOUT switching current state
            # We need to temporarily build a prompt for the new template
            template = self.prompt_system.templates.get(next_template_id)
            if not template:
                logger.error(f"Template '{next_template_id}' not found")
                return False
            
            # Build prompt and components for the new template
            prompt, negative, components = self._build_prompt_for_template(template)
            
            logger.info(f"[FRESH] Pre-generating with template '{next_template_id}'")
            logger.debug(f"  Prompt: {prompt[:80]}...")
            
            # Generate the frame using txt2img
            result_path = await self._generate_txt2img(prompt, negative)
            
            if result_path:
                elapsed = time.time() - start_time
                self._generation_times.append(elapsed)
                if len(self._generation_times) > 20:
                    self._generation_times.pop(0)
                
                # Move/rename to our output directory with consistent naming
                self._frame_counter += 1
                final_path = self.output_dir / f"fresh_{self._frame_counter:05d}.png"
                
                # Copy result to our directory (generator may have its own location)
                import shutil
                shutil.copy2(result_path, final_path)
                
                # Create buffered frame
                self._buffered_frame = BufferedFrame(
                    path=final_path,
                    template_id=next_template_id,
                    components=components,
                    prompt=prompt,
                    negative_prompt=negative,
                    generated_at=time.time()
                )
                
                self._total_generated += 1
                
                logger.info(
                    f"[FRESH] Pre-generated frame ready in {elapsed:.2f}s "
                    f"(template: '{next_template_id}')"
                )
                return True
            else:
                logger.error("[FRESH] Pre-generation failed")
                return False
                
        except Exception as e:
            logger.error(f"[FRESH] Pre-generation error: {e}", exc_info=True)
            return False
        finally:
            self._is_generating = False
    
    def _build_prompt_for_template(self, template) -> tuple:
        """
        Build prompt string for a template without affecting current state
        
        This creates a one-off prompt for pre-generation without switching
        the prompt system's active template.
        
        Args:
            template: Template object to build prompt for
            
        Returns:
            Tuple of (prompt, negative_prompt, components_dict)
        """
        import random
        
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
        
        Uses the generator's async API if available, otherwise falls back to sync.
        
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
    
    def consume(self) -> Optional[BufferedFrame]:
        """
        Consume the buffered frame
        
        Called at seed injection time. Returns the buffered frame and clears
        the buffer, triggering the next pre-generation cycle.
        
        Returns:
            BufferedFrame if available, None if buffer is empty
        """
        frame = self._buffered_frame
        self._buffered_frame = None
        
        if frame:
            self._total_consumed += 1
            age = time.time() - frame.generated_at
            logger.info(
                f"[FRESH] Consumed buffered frame (age: {age:.1f}s, "
                f"template: '{frame.template_id}')"
            )
        else:
            logger.warning("[FRESH] No buffered frame available to consume")
        
        return frame
    
    def peek(self) -> Optional[BufferedFrame]:
        """
        Peek at the buffered frame without consuming it
        
        Returns:
            BufferedFrame if available, None otherwise
        """
        return self._buffered_frame
    
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
        
        return {
            "is_ready": self.is_ready(),
            "is_generating": self.is_generating(),
            "buffered_template": (
                self._buffered_frame.template_id 
                if self._buffered_frame else None
            ),
            "total_generated": self._total_generated,
            "total_consumed": self._total_consumed,
            "avg_generation_time": avg_gen_time,
            "buffer_age": (
                time.time() - self._buffered_frame.generated_at
                if self._buffered_frame else 0.0
            )
        }
    
    def clear(self) -> None:
        """
        Clear the buffer
        
        Useful for shutdown or when changing generation parameters.
        """
        if self._buffered_frame:
            logger.info(f"[FRESH] Clearing buffered frame (template: '{self._buffered_frame.template_id}')")
        self._buffered_frame = None


# Test function
async def test_fresh_frame_buffer() -> bool:
    """
    Test FreshFrameBuffer functionality
    
    Note: This test uses mocks since it doesn't require ComfyUI.
    """
    print("=" * 60)
    print("Testing FreshFrameBuffer...")
    print("=" * 60)
    
    import tempfile
    from unittest.mock import MagicMock, AsyncMock
    from PIL import Image
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image for mock generation
            test_img_path = Path(tmpdir) / "test_output.png"
            Image.new("RGB", (512, 512), color=(100, 150, 200)).save(test_img_path)
            
            # Create mock generator (without async method to force sync fallback)
            mock_generator = MagicMock()
            mock_generator.generate_from_prompt = MagicMock(return_value=test_img_path)
            # Remove async method so it falls back to sync in executor
            del mock_generator.generate_from_prompt_async
            
            # Create mock prompt system
            mock_prompt = MagicMock()
            mock_prompt.get_current_template_id.return_value = "template_a"
            mock_prompt.get_available_templates.return_value = ["template_a", "template_b"]
            
            # Create mock template for template_b (the one that will be selected)
            mock_template = MagicMock()
            mock_template.id = "template_b"
            mock_template.structure = "{adjective} {noun} in {setting}"
            mock_template.slots = [("adjective", 1, " "), ("noun", 1, " "), ("setting", 1, " ")]
            mock_prompt.templates = {
                "template_a": mock_template,  # Needed as fallback
                "template_b": mock_template
            }
            
            # Create mock components
            mock_comp_adj = MagicMock()
            mock_comp_adj.word = "ethereal"
            mock_comp_adj.opposite = "harsh"
            
            mock_comp_noun = MagicMock()
            mock_comp_noun.word = "angel"
            mock_comp_noun.opposite = None
            
            mock_comp_setting = MagicMock()
            mock_comp_setting.word = "void"
            mock_comp_setting.opposite = None
            
            mock_prompt.components = {
                "adjective": [mock_comp_adj],
                "noun": [mock_comp_noun],
                "setting": [mock_comp_setting]
            }
            
            # Create config
            config = {
                "system": {
                    "output_dir": tmpdir
                }
            }
            
            print("\n1. Creating FreshFrameBuffer...")
            buffer = FreshFrameBuffer(
                generator=mock_generator,
                prompt_system=mock_prompt,
                config=config
            )
            print("✓ Buffer created")
            
            print("\n2. Checking initial state...")
            assert not buffer.is_ready(), "Buffer should start empty"
            assert not buffer.is_generating(), "Should not be generating initially"
            print("✓ Initial state correct")
            
            print("\n3. Testing ensure_ready()...")
            success = await buffer.ensure_ready()
            assert success, "ensure_ready should succeed"
            assert buffer.is_ready(), "Buffer should be ready after ensure_ready"
            print("✓ ensure_ready() works")
            
            print("\n4. Testing peek()...")
            frame = buffer.peek()
            assert frame is not None, "peek should return frame"
            assert frame.template_id == "template_b", f"Expected template_b, got {frame.template_id}"
            assert buffer.is_ready(), "Buffer should still be ready after peek"
            print(f"✓ peek() works (template: {frame.template_id})")
            
            print("\n5. Testing consume()...")
            frame = buffer.consume()
            assert frame is not None, "consume should return frame"
            assert not buffer.is_ready(), "Buffer should be empty after consume"
            print(f"✓ consume() works (got template: {frame.template_id})")
            
            print("\n6. Testing consume on empty buffer...")
            frame = buffer.consume()
            assert frame is None, "consume on empty should return None"
            print("✓ Empty consume returns None")
            
            print("\n7. Testing get_stats()...")
            stats = buffer.get_stats()
            assert "total_generated" in stats
            assert "total_consumed" in stats
            assert stats["total_generated"] == 1
            assert stats["total_consumed"] == 1
            print(f"✓ Stats: generated={stats['total_generated']}, consumed={stats['total_consumed']}")
            
            print("\n8. Testing clear()...")
            await buffer.ensure_ready()
            buffer.clear()
            assert not buffer.is_ready(), "Buffer should be empty after clear"
            print("✓ clear() works")
            
            print("\n" + "=" * 60)
            print("FreshFrameBuffer test PASSED ✓")
            print("=" * 60)
            return True
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    success = asyncio.run(test_fresh_frame_buffer())
    exit(0 if success else 1)

