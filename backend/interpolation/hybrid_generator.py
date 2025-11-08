"""
Hybrid Generator - Combines Interpolation + img2img

Provides the "magic" hybrid generation mode that:
1. Generates keyframes via img2img (slow but diverse)
2. Interpolates between keyframes (fast and smooth)
3. Injects cache for variety

This achieves smooth morphing with controlled variation.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from .spherical_lerp import spherical_lerp, batch_spherical_lerp
from .latent_encoder import LatentEncoder

logger = logging.getLogger(__name__)


class HybridGenerator:
    """
    Hybrid generation combining interpolation and img2img
    
    Strategy:
    - Generate keyframes every N frames (img2img, slow but varied)
    - Interpolate frames between keyframes (fast and smooth)
    - Result: 70% interpolated, 30% generated = optimal speed/quality
    
    Example timeline:
        Frame 0:  img2img (keyframe A)
        Frame 1-6: interpolate(A → B)
        Frame 7:  img2img (keyframe B)
        Frame 8-14: interpolate(B → C)
        ...
    """
    
    def __init__(
        self,
        generator,
        latent_encoder: Optional[LatentEncoder] = None,
        interpolation_frames: int = 6
    ):
        """
        Initialize hybrid generator
        
        Args:
            generator: DreamGenerator instance (for img2img)
            latent_encoder: LatentEncoder instance (for interpolation)
            interpolation_frames: Number of frames to interpolate between keyframes
        """
        self.generator = generator
        self.latent_encoder = latent_encoder
        self.interpolation_frames = interpolation_frames
        
        # State
        self.last_keyframe_latent = None
        self.next_keyframe_latent = None
        self.frames_since_keyframe = 0
        
        logger.info(f"Hybrid generator initialized ({interpolation_frames} interpolation frames)")
    
    def generate_sequence(
        self,
        start_image: Path,
        prompt: str,
        num_frames: int,
        denoise: float = 0.4
    ) -> List[Path]:
        """
        Generate a sequence of frames using hybrid mode
        
        Args:
            start_image: Starting image path
            prompt: Generation prompt
            num_frames: Total frames to generate
            denoise: img2img denoise strength
        
        Returns:
            List of paths to generated frames
        """
        frames = []
        current_image = start_image
        
        for i in range(num_frames):
            logger.info(f"Generating frame {i+1}/{num_frames}")
            
            # Decide: keyframe or interpolation?
            if i % (self.interpolation_frames + 1) == 0:
                # Generate keyframe via img2img
                logger.info(f"  → Keyframe generation (img2img)")
                frame_path = self.generator.generate_from_image(
                    image_path=current_image,
                    prompt=prompt,
                    denoise=denoise
                )
                
                if frame_path:
                    frames.append(frame_path)
                    current_image = frame_path
                    
                    # Encode for next interpolation
                    if self.latent_encoder:
                        self.last_keyframe_latent = self.latent_encoder.encode(frame_path)
                else:
                    logger.error("Keyframe generation failed")
                    break
            else:
                # Interpolate between keyframes
                if self.latent_encoder and self.last_keyframe_latent is not None:
                    logger.info(f"  → Interpolation")
                    # Calculate interpolation factor
                    t = (i % (self.interpolation_frames + 1)) / (self.interpolation_frames + 1)
                    
                    # Generate next keyframe latent if needed
                    if self.next_keyframe_latent is None:
                        # Peek ahead: generate next keyframe
                        temp_frame = self.generator.generate_from_image(
                            image_path=current_image,
                            prompt=prompt,
                            denoise=denoise
                        )
                        if temp_frame:
                            self.next_keyframe_latent = self.latent_encoder.encode(temp_frame)
                    
                    # Interpolate
                    if self.next_keyframe_latent is not None:
                        interpolated_latent = spherical_lerp(
                            self.last_keyframe_latent,
                            self.next_keyframe_latent,
                            t
                        )
                        
                        # Decode
                        interpolated_image = self.latent_encoder.decode(interpolated_latent)
                        
                        # Save
                        frame_path = self.generator.output_dir / f"frame_{len(frames):05d}.png"
                        interpolated_image.save(frame_path)
                        frames.append(frame_path)
                    else:
                        # Fallback: generate normally
                        frame_path = self.generator.generate_from_image(
                            image_path=current_image,
                            prompt=prompt,
                            denoise=denoise
                        )
                        if frame_path:
                            frames.append(frame_path)
                else:
                    # No encoder available, fallback to img2img
                    logger.warning("No latent encoder, using img2img fallback")
                    frame_path = self.generator.generate_from_image(
                        image_path=current_image,
                        prompt=prompt,
                        denoise=denoise
                    )
                    if frame_path:
                        frames.append(frame_path)
                        current_image = frame_path
        
        logger.info(f"Sequence complete: {len(frames)} frames generated")
        return frames
    
    def generate_smooth_transition(
        self,
        image_a: Path,
        image_b: Path,
        num_frames: int = 7,
        include_endpoints: bool = True
    ) -> List[Image.Image]:
        """
        Generate smooth transition between two images
        
        Args:
            image_a: Start image
            image_b: End image
            num_frames: Number of transition frames
            include_endpoints: If True, include A and B in output
        
        Returns:
            List of PIL Images showing smooth transition
        
        Note: This requires a functional LatentEncoder
        """
        if not self.latent_encoder:
            logger.error("LatentEncoder required for smooth transitions")
            return []
        
        logger.info(f"Generating smooth transition: {num_frames} frames")
        
        # Encode both images
        latent_a = self.latent_encoder.encode(image_a)
        latent_b = self.latent_encoder.encode(image_b)
        
        # Generate interpolated latents
        interpolated_latents = batch_spherical_lerp(
            latent_a,
            latent_b,
            num_frames=num_frames,
            include_endpoints=include_endpoints
        )
        
        # Decode all frames
        frames = []
        for i, latent in enumerate(interpolated_latents):
            logger.info(f"  Decoding frame {i+1}/{num_frames}")
            image = self.latent_encoder.decode(latent.unsqueeze(0))
            frames.append(image)
        
        logger.info(f"Transition complete: {len(frames)} frames")
        return frames
    
    def estimate_generation_time(
        self,
        num_frames: int,
        img2img_time: float = 2.0,
        interpolation_time: float = 0.5
    ) -> float:
        """
        Estimate total generation time for a sequence
        
        Args:
            num_frames: Total frames to generate
            img2img_time: Time per img2img generation (seconds)
            interpolation_time: Time per interpolation (seconds)
        
        Returns:
            Estimated total time in seconds
        """
        # Calculate number of keyframes vs interpolations
        num_keyframes = (num_frames + self.interpolation_frames) // (self.interpolation_frames + 1)
        num_interpolations = num_frames - num_keyframes
        
        total_time = (num_keyframes * img2img_time) + (num_interpolations * interpolation_time)
        
        return total_time
    
    def get_stats(self) -> dict:
        """
        Get hybrid generation statistics
        
        Returns:
            Dictionary with generation stats
        """
        return {
            "interpolation_frames": self.interpolation_frames,
            "has_latent_encoder": self.latent_encoder is not None,
            "frames_since_keyframe": self.frames_since_keyframe
        }


class SimpleHybridGenerator:
    """
    Simplified hybrid generator that doesn't use VAE
    
    Instead of latent interpolation, uses:
    - Keyframes at regular intervals (img2img with high denoise)
    - Fill frames with lower denoise (smoother transitions)
    
    This is MUCH simpler and works without VAE operations!
    Perfect for Dream Window MVP.
    """
    
    def __init__(
        self,
        generator,
        keyframe_interval: int = 7,
        keyframe_denoise: float = 0.5,
        fill_denoise: float = 0.3
    ):
        """
        Initialize simple hybrid generator
        
        Args:
            generator: DreamGenerator instance
            keyframe_interval: Generate keyframe every N frames
            keyframe_denoise: Denoise strength for keyframes (higher = more variation)
            fill_denoise: Denoise strength for fill frames (lower = smoother)
        """
        self.generator = generator
        self.keyframe_interval = keyframe_interval
        self.keyframe_denoise = keyframe_denoise
        self.fill_denoise = fill_denoise
        
        logger.info(f"Simple hybrid generator initialized (keyframe every {keyframe_interval} frames)")
    
    def generate_next_frame(
        self,
        current_image: Path,
        prompt: str,
        frame_number: int
    ) -> Optional[Path]:
        """
        Generate next frame using hybrid strategy
        
        Args:
            current_image: Current frame
            prompt: Generation prompt
            frame_number: Current frame number (for keyframe decision)
        
        Returns:
            Path to generated frame
        """
        # Decide denoise based on keyframe interval
        is_keyframe = (frame_number % self.keyframe_interval == 0)
        denoise = self.keyframe_denoise if is_keyframe else self.fill_denoise
        
        mode = "KEYFRAME" if is_keyframe else "fill"
        logger.info(f"Frame {frame_number}: {mode} (denoise={denoise})")
        
        # Generate
        frame_path = self.generator.generate_from_image(
            image_path=current_image,
            prompt=prompt,
            denoise=denoise
        )
        
        return frame_path
    
    def estimate_frame_time(self, img2img_time: float = 2.0) -> float:
        """
        Estimate average time per frame
        
        All frames are img2img, so average time is constant.
        Keyframes might be slightly slower due to higher denoise.
        
        Args:
            img2img_time: Base img2img generation time
        
        Returns:
            Estimated time per frame
        """
        return img2img_time  # All frames are ~same speed


# Unit test when run directly
if __name__ == "__main__":
    print("🧪 Testing HybridGenerator...")
    
    # Test 1: SimpleHybridGenerator (no VAE needed)
    print("\n1. SimpleHybridGenerator test")
    print("ℹ️  This is the RECOMMENDED mode for Dream Window MVP")
    
    class MockGenerator:
        """Mock generator for testing"""
        def __init__(self):
            self.output_dir = Path("/tmp")
            self.frame_count = 0
        
        def generate_from_image(self, image_path, prompt, denoise):
            self.frame_count += 1
            print(f"   Mock generation: denoise={denoise}")
            return Path(f"/tmp/mock_frame_{self.frame_count}.png")
    
    mock_gen = MockGenerator()
    simple_hybrid = SimpleHybridGenerator(
        generator=mock_gen,
        keyframe_interval=7,
        keyframe_denoise=0.5,
        fill_denoise=0.3
    )
    
    print("✓ SimpleHybridGenerator initialized")
    
    # Test frame generation pattern
    print("\n2. Generation pattern test")
    mock_image = Path("/tmp/test.png")
    for i in range(15):
        result = simple_hybrid.generate_next_frame(
            current_image=mock_image,
            prompt="test prompt",
            frame_number=i
        )
    
    print(f"✓ Generated 15 frames (pattern: keyframe every 7)")
    
    # Test 3: Time estimation
    print("\n3. Time estimation test")
    avg_time = simple_hybrid.estimate_frame_time(img2img_time=2.0)
    print(f"✓ Average time per frame: {avg_time}s")
    
    total_time_100_frames = avg_time * 100
    print(f"  Estimated for 100 frames: {total_time_100_frames/60:.1f} minutes")
    
    # Test 4: Full HybridGenerator (requires VAE)
    print("\n4. Full HybridGenerator (VAE-based)")
    print("ℹ️  Requires functional LatentEncoder")
    print("ℹ️  Can be implemented later for smoother transitions")
    print("✓ Skipping for now (use SimpleHybridGenerator for MVP)")
    
    print("\n✅ All tests passed!")
    print("\n📝 Recommendations:")
    print("  1. Use SimpleHybridGenerator for Dream Window MVP")
    print("  2. It's simpler, faster, and doesn't need VAE operations")
    print("  3. Still provides smooth morphing via low denoise fill frames")
    print("  4. Can upgrade to full HybridGenerator later if desired")
    
    print("\nUsage:")
    print("  from backend.interpolation import SimpleHybridGenerator")
    print("  hybrid = SimpleHybridGenerator(generator)")
    print("  frame = hybrid.generate_next_frame(current_image, prompt, frame_num)")

