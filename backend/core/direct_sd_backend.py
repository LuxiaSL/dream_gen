"""
Direct Stable Diffusion Backend

Drop-in replacement for DreamGenerator that uses diffusers directly
instead of going through ComfyUI's HTTP API. Eliminates all serialization,
disk I/O, and network overhead for significantly higher throughput.

Implements the same interface as DreamGenerator so the orchestrator,
keyframe worker, and all other components work unchanged.

Production config: SD 1.5, euler sampler, karras scheduler, CFG 8.0, 15 steps.
"""

import asyncio
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DirectSDBackend:
    """
    Direct diffusers-based image generation backend.

    Replaces ComfyUI HTTP round-trips with in-process pipeline calls.
    Shares the same public interface as DreamGenerator for drop-in use.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Paths
        self.output_dir = Path(config["system"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / ".temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Resolution
        self.target_width, self.target_height = config["generation"]["resolution"]

        # Generation parameters from config
        model_type = config.get("generation", {}).get("model", "sd15")
        if model_type in ("sd15", "sd21-unclip"):
            sd_config = config["generation"].get("sd", {})
            self.default_steps = sd_config.get("steps", 15)
            self.default_cfg = sd_config.get("cfg_scale", 8.0)
            self.sampler_name = sd_config.get("sampler", "euler")
            self.scheduler_name = sd_config.get("scheduler", "karras")
        else:
            raise ValueError(f"Direct backend only supports SD 1.5 currently, got: {model_type}")

        # FPS throttling
        perf_config = config.get("performance", {})
        self.max_generation_fps: Optional[float] = perf_config.get("max_generation_fps")
        self.enable_compile = perf_config.get("enable_torch_compile", False)

        # State
        self.frame_count = 0
        self.generation_times: list[float] = []
        self._shutdown_requested = False
        self._pipeline_lock = threading.Lock()

        # Device
        gpu_id = config.get("system", {}).get("gpu_id", 0)
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Pipelines (lazy-loaded)
        self._txt2img_pipe = None
        self._img2img_pipe = None
        self._loaded = False

        logger.info(
            f"DirectSDBackend initialized "
            f"(device={self.device}, steps={self.default_steps}, "
            f"cfg={self.default_cfg}, scheduler={self.scheduler_name}, "
            f"compile={self.enable_compile}, throttle={self.max_generation_fps})"
        )

    # ================================================================
    # Pipeline Management
    # ================================================================

    def _ensure_loaded(self) -> None:
        """Load pipelines on first use (lazy initialization)."""
        if self._loaded:
            return

        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            EulerDiscreteScheduler,
        )

        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        logger.info(f"Loading SD 1.5 pipeline from {model_id} (fp16)...")
        load_start = time.time()

        # Load txt2img pipeline (shares all components)
        self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Configure scheduler: euler + karras
        scheduler_config = self._txt2img_pipe.scheduler.config
        use_karras = self.scheduler_name == "karras"
        self._txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
            scheduler_config,
            use_karras_sigmas=use_karras,
        )

        # Disable progress bars
        self._txt2img_pipe.set_progress_bar_config(disable=True)

        # Build img2img pipeline sharing all components (zero extra VRAM)
        self._img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self._txt2img_pipe.vae,
            text_encoder=self._txt2img_pipe.text_encoder,
            tokenizer=self._txt2img_pipe.tokenizer,
            unet=self._txt2img_pipe.unet,
            scheduler=self._txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        self._img2img_pipe.set_progress_bar_config(disable=True)

        load_time = time.time() - load_start
        logger.info(f"Pipeline loaded in {load_time:.1f}s")

        # Optional: torch.compile UNet for ~30% speedup
        if self.enable_compile:
            try:
                logger.info("Compiling UNet with torch.compile (first inference will be slow)...")
                self._txt2img_pipe.unet = torch.compile(
                    self._txt2img_pipe.unet,
                    mode="reduce-overhead",
                )
                # img2img shares the same unet object, so it's already compiled
                logger.info("UNet compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed (will use eager mode): {e}")

        self._loaded = True

    def share_vae(self, external_vae) -> None:
        """
        Replace pipeline VAE with a shared instance from LatentEncoder.

        This avoids loading two copies of the VAE (~160MB fp16 each).
        Called by DreamController after LatentEncoder initialization.

        Args:
            external_vae: AutoencoderKL instance from LatentEncoder
        """
        self._ensure_loaded()

        old_vae = self._txt2img_pipe.vae
        self._txt2img_pipe.vae = external_vae
        self._img2img_pipe.vae = external_vae

        # Help GC free the old copy
        del old_vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("VAE shared with LatentEncoder (duplicate freed)")

    # ================================================================
    # Image Resize (copied from DreamGenerator — pure PIL)
    # ================================================================

    def _resize_image_for_generation(self, image_path: Path) -> Path:
        """Resize image to target resolution if needed."""
        img = Image.open(image_path)
        current_w, current_h = img.size

        if current_w == self.target_width and current_h == self.target_height:
            return image_path

        logger.info(f"Resizing {current_w}x{current_h} → {self.target_width}x{self.target_height}")
        img_resized = img.resize(
            (self.target_width, self.target_height),
            Image.Resampling.LANCZOS,
        )

        temp_path = self.temp_dir / f"resized_{image_path.name}"
        img_resized.save(temp_path, format="PNG", optimize=True)
        return temp_path

    # ================================================================
    # Generation Methods
    # ================================================================

    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """Generate image from text prompt (txt2img)."""
        start_time = time.time()

        if negative_prompt is None:
            negative_prompt = self._default_negative_prompt()

        logger.info(f"[txt2img] prompt: {prompt[:60]}...")

        self._ensure_loaded()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            with self._pipeline_lock:
                result = self._txt2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.target_width,
                    height=self.target_height,
                    num_inference_steps=self.default_steps,
                    guidance_scale=self.default_cfg,
                    generator=generator,
                )
        except Exception as e:
            logger.error(f"txt2img generation failed: {e}", exc_info=True)
            return None

        return self._save_result(result.images[0], start_time)

    async def generate_from_prompt_async(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """Async wrapper for txt2img (runs in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_from_prompt, prompt, negative_prompt, seed,
        )

    def generate_from_image(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        denoise: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """Generate image from existing image (img2img morphing)."""
        start_time = time.time()

        if not image_path.exists():
            logger.error(f"Source image not found: {image_path}")
            return None

        if negative_prompt is None:
            negative_prompt = self._default_negative_prompt()

        if denoise is None:
            denoise = self.config["generation"]["img2img"]["denoise"]

        logger.info(f"[img2img] {image_path.name} denoise={denoise:.2f}")
        logger.debug(f"Prompt: {prompt[:60]}...")

        self._ensure_loaded()

        # Load and resize input image
        resized_path = self._resize_image_for_generation(image_path)
        input_image = Image.open(resized_path).convert("RGB")

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            with self._pipeline_lock:
                result = self._img2img_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=denoise,
                    num_inference_steps=self.default_steps,
                    guidance_scale=self.default_cfg,
                    generator=generator,
                )
        except Exception as e:
            logger.error(f"img2img generation failed: {e}", exc_info=True)
            return None

        output_path = self._save_result(result.images[0], start_time)

        # FPS throttling: sleep if generation was too fast
        if output_path and self.max_generation_fps:
            elapsed = time.time() - start_time
            min_interval = 1.0 / self.max_generation_fps
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        return output_path

    async def generate_from_image_async(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        denoise: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """Async wrapper for img2img (runs in executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate_from_image,
            image_path, prompt, negative_prompt, denoise, seed,
        )

    # ================================================================
    # Compatibility Stubs (match DreamGenerator interface)
    # ================================================================

    def interrupt_and_clear_queue(self) -> bool:
        """No external queue to manage — no-op."""
        logger.debug("interrupt_and_clear_queue called (no-op in direct mode)")
        return True

    def is_comfyui_responsive(self) -> bool:
        """Health check: returns True if pipeline is loaded."""
        return self._loaded

    def get_queue_status(self) -> dict:
        """No queue — always empty."""
        return {"running": 0, "pending": 0}

    def get_performance_stats(self) -> Dict[str, float]:
        """Get generation timing statistics."""
        if not self.generation_times:
            return {}
        return {
            "avg_time": sum(self.generation_times) / len(self.generation_times),
            "min_time": min(self.generation_times),
            "max_time": max(self.generation_times),
            "total_frames": self.frame_count,
        }

    def close(self) -> None:
        """Release GPU resources."""
        self._txt2img_pipe = None
        self._img2img_pipe = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DirectSDBackend closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _default_negative_prompt(self) -> str:
        """Get default negative prompt from config."""
        prompts_config = self.config.get("prompts", {})
        theme_pairs = prompts_config.get("theme_pairs", [])
        if theme_pairs:
            return theme_pairs[0].get("negative", "")
        return prompts_config.get("negative", "")

    def _save_result(self, image: Image.Image, start_time: float) -> Optional[Path]:
        """Save generated PIL image to output directory."""
        self.frame_count += 1
        dest_filename = f"frame_{self.frame_count:05d}.png"
        dest_path = self.output_dir / dest_filename

        try:
            image.save(dest_path, format="PNG")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return None

        elapsed = time.time() - start_time
        self.generation_times.append(elapsed)
        if len(self.generation_times) > 100:
            self.generation_times.pop(0)

        logger.info(f"Generation time: {elapsed:.2f}s ({1/elapsed:.1f} FPS)")
        return dest_path

    @property
    def last_seed(self) -> Optional[int]:
        """Compatibility property for state restore."""
        return getattr(self, "_last_seed", None)

    @last_seed.setter
    def last_seed(self, value: int) -> None:
        self._last_seed = value
