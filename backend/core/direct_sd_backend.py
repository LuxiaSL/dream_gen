"""
Direct Stable Diffusion Backend

Drop-in replacement for DreamGenerator that uses diffusers directly
instead of going through ComfyUI's HTTP API. Eliminates all serialization,
disk I/O, and network overhead for significantly higher throughput.

Implements the same interface as DreamGenerator so the orchestrator,
keyframe worker, and all other components work unchanged.

Optimizations over naive pipeline calls:
  - CLIP embedding cache: skip re-encoding when prompt unchanged (~50ms saved)
  - Latent cache: skip VAE encode when input is our own last output (~60ms saved)
  - Manual UNet loop: bypass pipeline wrapper overhead (~50ms saved)
  - JPEG save instead of PNG (~80ms saved)
  - output_type="latent" to get cacheable latent + single VAE decode

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

        self.do_cfg = self.default_cfg > 1.0

        # FPS throttling
        perf_config = config.get("performance", {})
        self.max_generation_fps: Optional[float] = perf_config.get("max_generation_fps")
        self.enable_compile = perf_config.get("enable_torch_compile", False)

        # State
        self.frame_count = 0
        self.generation_times: list[float] = []
        self._shutdown_requested = False
        self._pipeline_lock = threading.Lock()

        # Device — supports multi-GPU via gpu_devices config
        default_gpu = config.get("system", {}).get("gpu_id", 0)
        gpu_devices = config.get("system", {}).get("gpu_devices", {})
        unet_gpu = gpu_devices.get("unet", default_gpu)
        vae_gpu = gpu_devices.get("vae", default_gpu)

        self.unet_device = f"cuda:{unet_gpu}" if torch.cuda.is_available() else "cpu"
        self.vae_device = f"cuda:{vae_gpu}" if torch.cuda.is_available() else "cpu"
        self.device = self.unet_device  # Default device (backward compat)
        self.multi_gpu = self.unet_device != self.vae_device

        if self.multi_gpu:
            logger.info(f"Multi-GPU: UNet on {self.unet_device}, VAE on {self.vae_device}")

        # Pipelines (lazy-loaded)
        self._txt2img_pipe = None
        self._img2img_pipe = None
        self._unet = None
        self._vae = None
        self._scheduler = None
        self._image_processor = None
        self._loaded = False

        # CLIP embedding cache — skip re-encoding when prompt unchanged
        self._cached_prompt: Optional[str] = None
        self._cached_negative: Optional[str] = None
        self._cached_prompt_embeds = None
        self._cached_negative_embeds = None

        # Latent cache — skip VAE encode when input is our own last output
        self._last_output_path: Optional[Path] = None
        self._last_output_latent = None  # torch.Tensor on GPU

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

        # Store direct references for the manual UNet loop
        self._unet = self._txt2img_pipe.unet
        self._vae = self._txt2img_pipe.vae
        self._scheduler = self._txt2img_pipe.scheduler
        self._image_processor = self._img2img_pipe.image_processor

        load_time = time.time() - load_start
        logger.info(f"Pipeline loaded in {load_time:.1f}s")

        # Optional: torch.compile UNet for ~30% speedup
        if self.enable_compile:
            try:
                logger.info("Compiling UNet with torch.compile (first inference will be slow)...")
                self._unet = torch.compile(
                    self._unet,
                    mode="default",  # "reduce-overhead" breaks with variable step counts
                )
                # Update references
                self._txt2img_pipe.unet = self._unet
                logger.info("UNet compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed (will use eager mode): {e}")

        # Multi-GPU: move VAE to dedicated device after pipeline load
        if self.multi_gpu:
            logger.info(f"Moving pipeline VAE to {self.vae_device}...")
            self._vae = self._vae.to(self.vae_device)
            self._txt2img_pipe.vae = self._vae
            self._img2img_pipe.vae = self._vae
            logger.info(f"[OK] VAE on {self.vae_device}, UNet on {self.unet_device}")

        self._loaded = True

    def share_vae(self, external_vae) -> None:
        """
        Replace pipeline VAE with a shared instance from LatentEncoder.

        This avoids loading two copies of the VAE (~160MB fp16 each).
        Called by DreamController after LatentEncoder initialization.
        """
        self._ensure_loaded()

        old_vae = self._vae
        self._vae = external_vae
        self._txt2img_pipe.vae = external_vae
        self._img2img_pipe.vae = external_vae

        del old_vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("VAE shared with LatentEncoder (duplicate freed)")

    # ================================================================
    # Image Resize (from DreamGenerator — pure PIL)
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
    # Caching
    # ================================================================

    def _get_prompt_embeds(self, prompt: str, negative_prompt: str):
        """Get CLIP embeddings, using cache if prompt unchanged."""
        if prompt == self._cached_prompt and negative_prompt == self._cached_negative:
            return self._cached_prompt_embeds, self._cached_negative_embeds

        pipe = self._txt2img_pipe
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_cfg,
            negative_prompt=negative_prompt,
        )

        self._cached_prompt = prompt
        self._cached_negative = negative_prompt
        self._cached_prompt_embeds = prompt_embeds
        self._cached_negative_embeds = negative_embeds

        logger.debug("CLIP embeddings encoded (cache miss)")
        return prompt_embeds, negative_embeds

    def _has_cached_latent(self, image_path: Path) -> bool:
        """Check if we have a cached latent for this image path."""
        return (
            self._last_output_latent is not None
            and self._last_output_path is not None
            and image_path.resolve() == self._last_output_path.resolve()
        )

    # ================================================================
    # Manual UNet Loop — bypasses pipeline wrapper overhead
    # ================================================================

    def _run_unet_loop(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        denoise: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Run the denoising loop directly on a latent tensor.

        This replaces the Img2Img pipeline's __call__ method for the hot path,
        eliminating per-call overhead: timestep setup validation, image preprocessing,
        safety checks, callback management, etc.

        Args:
            latent: Clean latent to denoise from (scaled, on device)
            prompt_embeds: Pre-computed CLIP embeddings for prompt
            negative_embeds: Pre-computed CLIP embeddings for negative prompt
            denoise: Strength parameter (0.0-1.0)
            generator: Torch random generator for noise

        Returns:
            Denoised latent tensor (still scaled, needs VAE decode)
        """
        scheduler = self._scheduler
        unet = self._unet

        # Set up timesteps for img2img (only run the last N steps based on denoise)
        scheduler.set_timesteps(self.default_steps, device=self.device)
        # Compute how many steps to actually run
        init_timestep = min(int(self.default_steps * denoise), self.default_steps)
        t_start = max(self.default_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        num_steps = len(timesteps)

        if num_steps == 0:
            return latent

        # Add noise at the appropriate level for img2img
        noise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=latent.dtype)
        latent = scheduler.add_noise(latent, noise, timesteps[:1])

        # Build combined embeddings for classifier-free guidance
        if self.do_cfg:
            prompt_combined = torch.cat([negative_embeds, prompt_embeds])
        else:
            prompt_combined = prompt_embeds

        # Denoising loop
        with torch.no_grad():
            for t in timesteps:
                # Expand latent for classifier-free guidance (unconditional + conditional)
                latent_input = torch.cat([latent] * 2) if self.do_cfg else latent
                latent_input = scheduler.scale_model_input(latent_input, t)

                # UNet prediction
                noise_pred = unet(latent_input, t, encoder_hidden_states=prompt_combined).sample

                # Classifier-free guidance
                if self.do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.default_cfg * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latent = scheduler.step(noise_pred, t, latent, generator=generator).prev_sample

        return latent

    def _vae_encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to latent space."""
        # Preprocess: PIL → tensor → normalize to [-1, 1]
        processed = self._image_processor.preprocess(image).to(
            device=self.device, dtype=torch.float16,
        )
        with torch.no_grad():
            latent = self._vae.encode(processed).latent_dist.sample()
        return latent * self._vae.config.scaling_factor

    def _vae_decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode a latent tensor to PIL image."""
        with torch.no_grad():
            decoded = self._vae.decode(
                latent / self._vae.config.scaling_factor,
                return_dict=False,
            )[0]
        return self._image_processor.postprocess(decoded, output_type="pil")[0]

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

        prompt_embeds, negative_embeds = self._get_prompt_embeds(prompt, negative_prompt)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            with self._pipeline_lock:
                # txt2img still uses the pipeline (not the hot path)
                result = self._txt2img_pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    width=self.target_width,
                    height=self.target_height,
                    num_inference_steps=self.default_steps,
                    guidance_scale=self.default_cfg,
                    generator=generator,
                    output_type="latent",
                )

                output_latent = result.images
                image = self._vae_decode_latent(output_latent)

        except Exception as e:
            logger.error(f"txt2img generation failed: {e}", exc_info=True)
            return None

        output_path = self._save_result(image, start_time)

        if output_path is not None:
            self._last_output_latent = output_latent.detach()
            self._last_output_path = output_path

        return output_path

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

        # --- Cached CLIP embeddings ---
        prompt_embeds, negative_embeds = self._get_prompt_embeds(prompt, negative_prompt)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            with self._pipeline_lock:
                # Get input latent — cached or encode from disk
                if self._has_cached_latent(image_path):
                    input_latent = self._last_output_latent
                    logger.debug("Using cached latent (skipped VAE encode + disk read)")
                else:
                    resized_path = self._resize_image_for_generation(image_path)
                    input_image = Image.open(resized_path).convert("RGB")
                    input_latent = self._vae_encode_image(input_image)
                    logger.debug("Encoded input from disk (cache miss)")

                # Manual UNet loop — bypasses pipeline wrapper
                output_latent = self._run_unet_loop(
                    latent=input_latent,
                    prompt_embeds=prompt_embeds,
                    negative_embeds=negative_embeds,
                    denoise=denoise,
                    generator=generator,
                )

                # VAE decode to PIL
                image = self._vae_decode_latent(output_latent)

        except Exception as e:
            logger.error(f"img2img generation failed: {e}", exc_info=True)
            self._last_output_latent = None
            self._last_output_path = None
            return None

        output_path = self._save_result(image, start_time)

        # Cache output latent for next frame
        if output_path is not None:
            self._last_output_latent = output_latent.detach()
            self._last_output_path = output_path

        # FPS throttling
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
        self._unet = None
        self._vae = None
        self._scheduler = None
        self._image_processor = None
        self._loaded = False
        self._last_output_latent = None
        self._last_output_path = None
        self._cached_prompt_embeds = None
        self._cached_negative_embeds = None
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
        """Save generated PIL image to output directory as JPEG (fast encode)."""
        self.frame_count += 1
        dest_filename = f"frame_{self.frame_count:05d}.jpg"
        dest_path = self.output_dir / dest_filename

        try:
            image.save(dest_path, format="JPEG", quality=95)
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
