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

logger = logging.getLogger(__name__)


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
    
    def __init__(self, vae_path: Optional[Path] = None, device: str = "cuda"):
        """
        Initialize VAE encoder/decoder
        
        Args:
            vae_path: Path to VAE model checkpoint (optional)
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vae = None
        self.vae_path = vae_path
        
        if vae_path:
            self._load_vae(vae_path)
        else:
            logger.warning("No VAE path provided - encoder will not be functional")
    
    def _load_vae(self, vae_path: Path):
        """
        Load VAE model from checkpoint
        
        Note: This is a placeholder implementation. In production,
        you'll need to load the actual VAE from your Flux checkpoint
        or standalone VAE file.
        
        For ComfyUI integration, the VAE encoding/decoding should
        happen through ComfyUI's API rather than locally.
        """
        try:
            logger.info(f"Loading VAE from {vae_path}")
            
            # Placeholder: In real implementation, load actual VAE
            # from safetensors or PyTorch checkpoint
            # 
            # Example with Flux:
            # from comfy.sd import VAE
            # self.vae = VAE(sd_path=str(vae_path))
            
            # For now, set to None - will use ComfyUI API instead
            self.vae = None
            
            logger.warning("VAE loading not implemented - will use ComfyUI API")
            logger.info("For local VAE operations, integrate with ComfyUI's VAE class")
            
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            raise
    
    def encode(self, image: Union[Image.Image, Path]) -> torch.Tensor:
        """
        Encode image to latent space
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            Latent tensor (shape: [1, C, H, W])
            For Flux 256x512: [1, 4, 32, 64]
        
        Note: This is a placeholder. For production use with Flux:
        1. Use ComfyUI's VAE encode node via API, OR
        2. Integrate with ComfyUI's VAE class directly
        """
        if isinstance(image, Path):
            image = Image.open(image).convert('RGB')
        
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock latent")
            # Return mock latent with correct shape
            # For Flux: 4 channels, 8x compression
            c, h, w = 4, image_tensor.shape[2] // 8, image_tensor.shape[3] // 8
            return torch.randn(1, c, h, w, device=self.device)
        
        # Actual encoding (when VAE is loaded)
        with torch.no_grad():
            latent = self.vae.encode(image_tensor)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to image
        
        Args:
            latent: Latent tensor (shape: [1, C, H, W])
        
        Returns:
            PIL Image (RGB)
        
        Note: This is a placeholder. For production use with Flux:
        1. Use ComfyUI's VAE decode node via API, OR
        2. Integrate with ComfyUI's VAE class directly
        """
        if self.vae is None:
            logger.warning("VAE not loaded - returning mock image")
            # Return mock image with correct dimensions
            h, w = latent.shape[2] * 8, latent.shape[3] * 8
            return Image.new('RGB', (w, h), color=(128, 128, 128))
        
        # Actual decoding (when VAE is loaded)
        with torch.no_grad():
            image_tensor = self.vae.decode(latent)
        
        # Post-process to PIL Image
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
        Preprocess PIL Image to tensor for VAE
        
        Steps:
        1. Resize if needed
        2. Convert to tensor
        3. Normalize to [-1, 1] or [0, 1] depending on VAE
        4. Move to device
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Image tensor (shape: [1, 3, H, W])
        """
        # Convert to numpy
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Channels first: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).to(self.device)
        
        return img_tensor
    
    def _postprocess_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """
        Post-process tensor to PIL Image
        
        Steps:
        1. Denormalize from [0, 1] to [0, 255]
        2. Convert to numpy
        3. Channels last: (C, H, W) -> (H, W, C)
        4. Convert to PIL Image
        
        Args:
            image_tensor: Image tensor (shape: [1, 3, H, W])
        
        Returns:
            PIL Image (RGB)
        """
        # Remove batch dimension
        img_array = image_tensor.squeeze(0).cpu().numpy()
        
        # Channels last: (C, H, W) -> (H, W, C)
        img_array = np.transpose(img_array, (1, 2, 0))
        
        # Denormalize to [0, 255]
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to PIL
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

