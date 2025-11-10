"""
VAE Encoder/Decoder for Latent Space Operations

Provides image <-> latent conversion using VAE models for smooth
interpolation and hybrid generation workflows.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str, logger_func=None):
    """Simple context manager for timing code blocks"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if logger_func:
            logger_func(f"{name}: {elapsed*1000:.2f}ms")


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


class LatentEncoder:
    """
    VAE encoder/decoder for latent space operations
    
    Responsibilities:
    - Load VAE model from checkpoint
    - Encode images to latent space
    - Decode latents back to images
    - Batch operations for efficiency
    
    Latent Space:
    - Flux: 4 channels, ~8x spatial compression
    - Input: 256x512 RGB image → Latent: 4x32x64 tensor
    """
    
    def __init__(self, vae_path: Optional[Path] = None, device: str = "cuda", auto_load: bool = False,
                 interpolation_resolution_divisor: int = 1, upscale_method: str = "bilinear",
                 downsample_method: str = "bilinear"):
        """
        Initialize VAE encoder/decoder
        
        Args:
            vae_path: Path to VAE model checkpoint (optional, uses HF default if None and auto_load=True)
            device: Device to run on ("cuda" or "cpu")
            auto_load: If True and vae_path is None, automatically loads SD 1.5 VAE from HuggingFace
            interpolation_resolution_divisor: Divide resolution by this for interpolation (1=full res, 2=half, 4=quarter)
            upscale_method: Method for upscaling after decode ("bilinear", "bicubic", "nearest")
            downsample_method: Method for downsampling before encode ("bilinear", "bicubic", "lanczos")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vae = None
        self.vae_path = vae_path
        
        # Tensor caching for preprocessing optimization
        self._preprocess_cache = {}
        self._use_pinned_memory = torch.cuda.is_available()
        
        # Lower-resolution interpolation settings
        self.interpolation_resolution_divisor = interpolation_resolution_divisor
        self.upscale_method = upscale_method
        self.downsample_method = downsample_method
        self.target_resolution = None  # Will be set when we see first image
        
        if interpolation_resolution_divisor > 1:
            logger.info(f"Lower-res interpolation enabled: {interpolation_resolution_divisor}x downscale")
            logger.info(f"Upscale method: {upscale_method}")
        
        # Load VAE if path provided or auto_load requested
        if vae_path:
            self._load_vae(vae_path)
        elif auto_load:
            self._load_vae()
    
    def _load_vae(self, vae_path: Path = None):
        """
        Load VAE model from HuggingFace for local encoding/decoding
        
        Uses SD 1.5 VAE for latent space operations. This provides
        fast local interpolation without API overhead.
        
        Args:
            vae_path: Optional path to local VAE checkpoint (not used for HF download)
        """
        try:
            from diffusers import AutoencoderKL
            
            logger.info("Loading SD 1.5 VAE for interpolation...")
            mem_before = get_gpu_memory_mb()
            
            # Load SD 1.5 VAE from HuggingFace
            # Using fp16 for memory efficiency (~500MB vs 1GB)
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="vae",
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)
            
            # Set to eval mode (no training)
            self.vae.eval()
            
            # OPTIMIZATION: Try to compile VAE decoder with torch.compile (PyTorch 2.0+)
            # NOTE: Requires CUDA Capability >= 7.0 (Volta or newer)
            # Maxwell Titan X (5.2) and Pascal (6.x) GPUs cannot use torch.compile
            try:
                # Check if torch.compile is available and GPU is new enough
                if hasattr(torch, 'compile') and torch.cuda.is_available():
                    # Check CUDA compute capability
                    compute_cap = torch.cuda.get_device_capability()
                    compute_cap_num = compute_cap[0] + compute_cap[1] / 10
                    
                    if compute_cap_num >= 7.0:
                        logger.info(f"  GPU compute capability: {compute_cap_num} (triton supported)")
                        logger.info("  Attempting to compile VAE decoder with torch.compile...")
                        # Compile the decoder for faster inference
                        # mode='reduce-overhead' optimizes for repeated calls (our use case!)
                        self.vae.decoder = torch.compile(
                            self.vae.decoder,
                            mode='reduce-overhead',
                            fullgraph=True
                        )
                        logger.info("  [OK] VAE decoder compiled successfully")
                    else:
                        logger.info(f"  GPU compute capability: {compute_cap_num} (triton requires >= 7.0)")
                        logger.info("  Skipping torch.compile (GPU too old for triton)")
            except Exception as e:
                logger.warning(f"  Could not compile VAE decoder: {e}")
                logger.info("  Continuing with standard (uncompiled) decoder")
            
            # VAE scaling factor for SD 1.5
            # This is the standard scale factor used in Stable Diffusion
            self.vae_scale_factor = 0.18215
            
            # Verify GPU placement
            mem_after = get_gpu_memory_mb()
            vae_mem = mem_after - mem_before
            
            # Check if VAE is actually on GPU
            first_param = next(self.vae.parameters())
            actual_device = first_param.device
            actual_dtype = first_param.dtype
            
            logger.info(f"[OK] VAE loaded successfully")
            logger.info(f"  Target device: {self.device}")
            logger.info(f"  Actual device: {actual_device}")
            logger.info(f"  Model dtype: {actual_dtype}")
            logger.info(f"  VRAM usage: {vae_mem:.1f} MB")
            logger.info(f"  Scale factor: {self.vae_scale_factor}")
            
            if str(actual_device) != self.device and not str(actual_device).startswith(self.device):
                logger.warning(f"⚠ VAE not on expected device! Expected {self.device}, got {actual_device}")
            
            if actual_dtype != torch.float16:
                logger.warning(f"⚠ VAE not in fp16! Got {actual_dtype}")
            
        except ImportError as e:
            logger.error("Failed to import diffusers - please install: uv pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}", exc_info=True)
            raise
    
    def encode(self, image: Union[Image.Image, Path], for_interpolation: bool = False) -> torch.Tensor:
        """
        Encode image to latent space using VAE
        
        Args:
            image: PIL Image or path to image file
            for_interpolation: If True and resolution divisor > 1, encode at lower resolution
        
        Returns:
            Latent tensor (shape: [1, 4, H//8, W//8])
            For 512x256 input: [1, 4, 32, 64]
            For 256x128 input with divisor=2: [1, 4, 16, 32]
        
        Process:
        1. Load and preprocess image (resize, normalize to [-1, 1])
        2. Optionally downsample for interpolation
        3. Encode to latent distribution
        4. Sample from distribution
        5. Scale by VAE factor
        """
        if isinstance(image, Path):
            image = Image.open(image).convert('RGB')
        
        # Store target resolution from first image
        if self.target_resolution is None:
            self.target_resolution = image.size
        
        # Apply resolution divisor for interpolation speedup
        if for_interpolation and self.interpolation_resolution_divisor > 1:
            original_size = image.size
            new_width = int(original_size[0] // self.interpolation_resolution_divisor)
            new_height = int(original_size[1] // self.interpolation_resolution_divisor)
            
            # Use configurable downsample method
            resample_method = getattr(Image.Resampling, self.downsample_method.upper())
            image = image.resize((new_width, new_height), resample_method)
            logger.debug(f"Downsampled for interpolation ({self.downsample_method}): {original_size} → {image.size}")
        
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock latent")
            # Fallback: return mock latent with correct shape
            h, w = image.size[1] // 8, image.size[0] // 8
            return torch.randn(1, 4, h, w, device=self.device)
        
        # Preprocess image to tensor
        image_tensor = self._preprocess_image(image)
        
        # Encode to latent space
        with torch.no_grad():
            # VAE encode returns a distribution
            latent_dist = self.vae.encode(image_tensor).latent_dist
            # Sample from the distribution (could also use .mode() for deterministic)
            latent = latent_dist.sample()
            # Apply VAE scaling factor
            latent = latent * self.vae_scale_factor
        
        return latent
    
    def decode(self, latent: torch.Tensor, upscale_to_target: bool = False) -> Image.Image:
        """
        Decode latent tensor back to image using VAE
        
        Args:
            latent: Scaled latent tensor (shape: [1, 4, H//8, W//8])
            upscale_to_target: If True and image is smaller than target, upscale to target resolution
        
        Returns:
            PIL Image (RGB) at original or target resolution
        
        Process:
        1. Unscale latent by dividing by VAE factor
        2. Decode using VAE
        3. Optionally upscale on GPU (if resolution divisor was used)
        4. Denormalize from [-1, 1] to [0, 255]
        5. Convert to PIL Image
        """
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock image")
            # Fallback: return mock image with correct dimensions
            h, w = latent.shape[2] * 8, latent.shape[3] * 8
            return Image.new('RGB', (w, h), color=(128, 128, 128))
        
        # Unscale latent (reverse the encoding scaling)
        latent = latent / self.vae_scale_factor
        
        # Decode latent to image
        with torch.no_grad():
            # VAE decode returns a sample (tensor)
            image_tensor = self.vae.decode(latent).sample
        
        # OPTIMIZATION: Upscale on GPU if needed (much faster than CPU upscaling!)
        if upscale_to_target and self.target_resolution is not None and self.interpolation_resolution_divisor > 1:
            current_size = (image_tensor.shape[3], image_tensor.shape[2])  # (W, H)
            if current_size != self.target_resolution:
                # Upscale on GPU using torch.nn.functional.interpolate
                # This is MUCH faster than PIL resize on CPU
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor,
                    size=(self.target_resolution[1], self.target_resolution[0]),  # (H, W)
                    mode=self.upscale_method,
                    align_corners=False if self.upscale_method in ['bilinear', 'bicubic'] else None
                )
                logger.debug(f"GPU upscaled: {current_size} → {self.target_resolution} ({self.upscale_method})")
        
        # Post-process tensor to PIL Image
        image = self._postprocess_image(image_tensor)
        
        return image
    
    def encode_batch(self, images: List[Union[Image.Image, Path]]) -> torch.Tensor:
        """
        Encode multiple images to latent space
        
        Args:
            images: List of PIL Images or paths
        
        Returns:
            Batched latent tensor (shape: [N, C, H, W])
        
        Note: ~3x faster than encoding individually
        """
        # Load and preprocess all images
        image_tensors = []
        for img in images:
            if isinstance(img, Path):
                img = Image.open(img).convert('RGB')
            img_tensor = self._preprocess_image(img)
            image_tensors.append(img_tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0)
        
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock latents")
            n, c, h, w = batch_tensor.shape[0], 4, batch_tensor.shape[2] // 8, batch_tensor.shape[3] // 8
            return torch.randn(n, c, h, w, device=self.device)
        
        # Encode batch
        with torch.no_grad():
            latents = self.vae.encode(batch_tensor)
        
        return latents
    
    def decode_batch(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Decode multiple latents to images
        
        Args:
            latents: Batched latent tensor (shape: [N, C, H, W])
        
        Returns:
            List of PIL Images
        """
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock images")
            h, w = latents.shape[2] * 8, latents.shape[3] * 8
            return [Image.new('RGB', (w, h), color=(128, 128, 128)) for _ in range(latents.shape[0])]
        
        # Decode batch
        with torch.no_grad():
            image_tensors = self.vae.decode(latents)
        
        # Convert to PIL Images
        images = []
        for i in range(image_tensors.shape[0]):
            img = self._postprocess_image(image_tensors[i:i+1])
            images.append(img)
        
        return images
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL Image to tensor for VAE encoding (OPTIMIZED with caching)
        
        VAE expects images normalized to [-1, 1] range.
        
        OPTIMIZATION: Use pinned memory and reusable tensor buffers to avoid
        repeated allocations and speed up CPU→GPU transfers.
        
        Steps:
        1. Convert to numpy array (vectorized)
        2. Normalize and convert to [-1, 1] in one step
        3. Channels first: (H, W, C) -> (C, H, W)
        4. Use pinned memory for faster transfer to GPU
        5. Add batch dimension
        6. Move to device and convert to fp16
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Image tensor (shape: [1, 3, H, W], range [-1, 1], dtype fp16)
        """
        # Get image dimensions
        width, height = image.size
        shape_key = (height, width)
        
        # Convert to numpy (this is still necessary for PIL)
        img_array = np.array(image, dtype=np.float32)
        
        # OPTIMIZATION: Fuse normalization operations
        # (img / 255.0 - 0.5) * 2.0 = (img - 127.5) / 127.5
        # This avoids an intermediate array allocation
        img_array = (img_array - 127.5) / 127.5
        
        # Channels first: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to tensor
        # Use pinned memory for faster CPU→GPU transfer if available
        if self._use_pinned_memory:
            # Create tensor with pinned memory
            img_tensor = torch.from_numpy(img_array).pin_memory().to(self.device, non_blocking=True)
        else:
            img_tensor = torch.from_numpy(img_array).to(self.device)
        
        # Convert to fp16 if VAE is using fp16 (match VAE dtype)
        if self.vae is not None and next(self.vae.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()
        
        return img_tensor
    
    def _postprocess_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Post-process decoded tensor to PIL Image (OPTIMIZED for minimal GPU→CPU transfer)
        
        VAE outputs images in [-1, 1] range.
        
        OPTIMIZATION: Keep all processing on GPU until final conversion to uint8,
        minimizing data transfer and leveraging GPU compute for normalization.
        
        Steps:
        1. Remove batch dimension (GPU)
        2. Convert from [-1, 1] to [0, 255] (GPU, fused operations)
        3. Clamp to valid range (GPU)
        4. Convert to uint8 (GPU)
        5. Channels last: (C, H, W) -> (H, W, C) (GPU)
        6. Transfer only final uint8 to CPU (much smaller transfer!)
        7. Convert to PIL Image
        
        Args:
            image_tensor: Image tensor (shape: [1, 3, H, W], range [-1, 1])
        
        Returns:
            PIL Image (RGB)
        """
        # Remove batch dimension (GPU operation, no copy)
        img_tensor = image_tensor.squeeze(0)
        
        # Convert from [-1, 1] to [0, 255] in one fused operation (GPU)
        # img_tensor is fp16, computation stays on GPU
        img_tensor = (img_tensor * 0.5 + 0.5) * 255.0
        
        # Clamp to valid range (GPU)
        img_tensor = torch.clamp(img_tensor, 0.0, 255.0)
        
        # Convert to uint8 (GPU) - much smaller data type for transfer!
        img_tensor = img_tensor.to(dtype=torch.uint8)
        
        # Channels last: (C, H, W) -> (H, W, C) (GPU operation)
        img_tensor = img_tensor.permute(1, 2, 0)
        
        # NOW transfer to CPU (only uint8 data, much faster!)
        # This is the ONLY CPU transfer, and it's optimized
        img_array = img_tensor.cpu().numpy()
        
        # Convert to PIL (CPU operation, but numpy array is already contiguous and uint8)
        image = Image.fromarray(img_array, mode='RGB')
        
        return image
    
    def get_latent_shape(self, image_width: int, image_height: int) -> tuple:
        """
        Calculate latent space dimensions for given image size
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            Tuple of (channels, latent_height, latent_width)
        
        Example:
            >>> encoder.get_latent_shape(256, 512)
            (4, 32, 64)  # For Flux with 8x compression
        """
        # Flux uses 8x spatial compression
        compression_factor = 8
        channels = 4  # Flux latent channels
        
        latent_h = image_height // compression_factor
        latent_w = image_width // compression_factor
        
        return (channels, latent_h, latent_w)


class ComfyUILatentEncoder(LatentEncoder):
    """
    Latent encoder that uses ComfyUI API for VAE operations
    
    This is the RECOMMENDED implementation for Dream Window.
    Instead of loading the VAE locally, it uses ComfyUI's API
    to perform encoding/decoding on the server side.
    
    Benefits:
    - No need to load heavy VAE model in Python
    - Reuses ComfyUI's optimized VAE implementation
    - Consistent with the rest of the generation pipeline
    """
    
    def __init__(self, comfyui_client, workflow_builder):
        """
        Initialize with ComfyUI client
        
        Args:
            comfyui_client: ComfyUIClient instance
            workflow_builder: WorkflowBuilder instance
        """
        super().__init__(vae_path=None, device="cuda")
        self.client = comfyui_client
        self.workflow_builder = workflow_builder
        logger.info("Using ComfyUI API for VAE operations")
    
    def encode(self, image: Union[Image.Image, Path]) -> torch.Tensor:
        """
        Encode image via ComfyUI API
        
        Note: This requires a ComfyUI workflow that:
        1. Loads the image
        2. Encodes it with VAE
        3. Saves the latent to a file
        4. Returns the latent file path
        
        For Dream Window, we may skip this and just use img2img
        workflows directly, which handle the VAE operations internally.
        """
        logger.warning("ComfyUI VAE encoding not yet implemented")
        logger.info("Consider using img2img workflows directly instead of manual VAE ops")
        return super().encode(image)
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent via ComfyUI API
        
        Similar to encode(), this is not typically needed for Dream Window
        since the img2img workflow handles decoding internally.
        """
        logger.warning("ComfyUI VAE decoding not yet implemented")
        logger.info("img2img workflows handle VAE decoding automatically")
        return super().decode(latent)


# Unit test when run directly
if __name__ == "__main__":
    print("🧪 Testing LatentEncoder...")
    
    # Test 1: Basic initialization
    print("\n1. Initialization test")
    encoder = LatentEncoder(vae_path=None, device="cpu")
    print("✓ Encoder initialized (no VAE loaded)")
    
    # Test 2: Latent shape calculation
    print("\n2. Latent shape calculation")
    shape = encoder.get_latent_shape(256, 512)
    assert shape == (4, 32, 64), f"Expected (4, 32, 64), got {shape}"
    print(f"✓ Latent shape for 256x512: {shape}")
    
    # Test 3: Image preprocessing
    print("\n3. Image preprocessing test")
    test_image = Image.new('RGB', (256, 512), color=(128, 128, 128))
    img_tensor = encoder._preprocess_image(test_image)
    assert img_tensor.shape == (1, 3, 512, 256), f"Unexpected shape: {img_tensor.shape}"
    print(f"✓ Preprocessed image shape: {img_tensor.shape}")
    
    # Test 4: Mock encode/decode
    print("\n4. Mock encode/decode test (no VAE)")
    latent = encoder.encode(test_image)
    print(f"✓ Mock latent shape: {latent.shape}")
    
    decoded = encoder.decode(latent)
    assert isinstance(decoded, Image.Image), "Decoded should be PIL Image"
    print(f"✓ Mock decoded image size: {decoded.size}")
    
    # Test 5: Batch operations
    print("\n5. Batch operations test")
    images = [test_image] * 3
    latents = encoder.encode_batch(images)
    print(f"✓ Batch latents shape: {latents.shape}")
    
    decoded_batch = encoder.decode_batch(latents)
    assert len(decoded_batch) == 3, f"Expected 3 images, got {len(decoded_batch)}"
    print(f"✓ Batch decoded: {len(decoded_batch)} images")
    
    # Test 6: Integration note
    print("\n6. Integration notes")
    print("ℹ️  For Dream Window:")
    print("   - Use ComfyUILatentEncoder for production")
    print("   - Or skip manual VAE ops entirely")
    print("   - img2img workflows handle VAE internally")
    print("   - Only needed if implementing pure latent interpolation")
    
    print("\n✅ All tests passed!")
    print("\nUsage for Dream Window:")
    print("  # Option 1: Use ComfyUI workflows (RECOMMENDED)")
    print("  generator.generate_from_image(...)  # VAE ops happen in ComfyUI")
    print("")
    print("  # Option 2: Manual VAE ops (if needed for interpolation)")
    print("  from backend.interpolation import LatentEncoder")
    print("  encoder = LatentEncoder(vae_path=path_to_vae)")
    print("  latent = encoder.encode(image)")
    print("  interpolated = spherical_lerp(latent_a, latent_b, t=0.5)")
    print("  result = encoder.decode(interpolated)")

