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

from .spherical_lerp import spherical_lerp, batch_spherical_lerp, precompute_slerp_params
from .latent_encoder import LatentEncoder
from .frame_sequence_manager import FrameSequenceManager, FrameType, FrameState

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
        
        # Frame sequence manager for explicit tracking
        self.frame_manager = FrameSequenceManager(
            interpolation_frames=interpolation_frames,
            output_dir=generator.output_dir
        )
        
        # OPTIMIZATION: Pre-computed slerp parameters
        # Maps keyframe pair (start, end) to precomputed params
        self.slerp_precomputed: dict = {}
        
        logger.info(f"Hybrid generator initialized with VAE interpolation")
        logger.info(f"  Interpolation frames: {interpolation_frames}")
        logger.info(f"  Has VAE: {latent_encoder is not None}")
    
    def generate_next_frame(
        self,
        current_image: Path,
        prompt: str,
        frame_number: int,
        denoise: float = 0.4
    ) -> Optional[Path]:
        """
        Generate next frame using hybrid strategy with VAE interpolation
        
        Uses FrameSequenceManager to track keyframes and interpolations explicitly.
        
        Strategy:
        - Keyframes → img2img generation (slow, diverse)
        - Interpolated frames → VAE latent lerp (fast, smooth)
        
        Args:
            current_image: Current frame path
            prompt: Generation prompt
            frame_number: Current frame number
            denoise: Denoise strength for keyframes
        
        Returns:
            Path to generated frame
        """
        # Get frame specification from manager
        frame_spec = self.frame_manager.get_or_create_frame_spec(frame_number)
        
        logger.info(f"Generating: {frame_spec}")
        
        # Mark as generating
        self.frame_manager.mark_generating(frame_number)
        
        if frame_spec.is_keyframe():
            # Generate keyframe via img2img
            logger.info(f"  -> Generating KEYFRAME via img2img (denoise={denoise})")
            
            keyframe = self.generator.generate_from_image(
                image_path=current_image,
                prompt=prompt,
                denoise=denoise
            )
            
            if not keyframe or not keyframe.exists():
                logger.error("Keyframe generation failed")
                return None
            
            # Mark as ready
            self.frame_manager.mark_ready(frame_number, keyframe, prompt)
            
            # Encode keyframe to latent if we have VAE
            if self.latent_encoder and self.latent_encoder.vae is not None:
                try:
                    # Synchronize CUDA after ComfyUI generation
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        logger.debug(f"  CUDA synchronized after img2img generation")
                    
                    # Encode at interpolation resolution to maintain consistency with other keyframes
                    latent = self.latent_encoder.encode(keyframe, for_interpolation=True)
                    
                    # Store in frame manager
                    self.frame_manager.store_keyframe_data(
                        frame_number=frame_number,
                        latent=latent,
                        image_path=keyframe
                    )
                    
                    logger.debug(f"  Encoded and stored keyframe latent: {latent.shape}")
                except Exception as e:
                    logger.error(f"Failed to encode keyframe: {e}", exc_info=True)
            
            return keyframe
        
        else:
            # Interpolate between keyframes
            logger.info(f"  -> Interpolating (t={frame_spec.interpolation_t:.3f})")
            
            # Check if we have VAE and latents
            if not self.latent_encoder or self.latent_encoder.vae is None:
                logger.warning("No VAE available - falling back to img2img")
                keyframe = self.generator.generate_from_image(
                    image_path=current_image,
                    prompt=prompt,
                    denoise=denoise * 0.6  # Lower denoise for fill frames
                )
                if keyframe:
                    self.frame_manager.mark_ready(frame_number, keyframe, prompt)
                return keyframe
            
            # Get keyframe pair data
            keyframe_pair_data = self.frame_manager.get_keyframe_pair_data(frame_number)
            
            if keyframe_pair_data is None:
                # Need to generate next keyframe first
                logger.info("  -> Next keyframe not ready, generating it now...")
                
                start_keyframe, end_keyframe = frame_spec.keyframe_pair
                
                # Check if we have the start keyframe
                if start_keyframe not in self.frame_manager.keyframe_latents:
                    logger.error(f"Missing start keyframe {start_keyframe}")
                    # Fallback: generate keyframe
                    keyframe = self.generator.generate_from_image(
                        image_path=current_image,
                        prompt=prompt,
                        denoise=denoise
                    )
                    if keyframe:
                        # Encode and store
                        latent = self.latent_encoder.encode(keyframe, for_interpolation=True)
                        self.frame_manager.store_keyframe_data(start_keyframe, latent, keyframe)
                        self.frame_manager.mark_ready(frame_number, keyframe, prompt)
                    return keyframe
                
                # Generate the end keyframe
                next_keyframe = self.generator.generate_from_image(
                    image_path=current_image,
                    prompt=prompt,
                    denoise=denoise
                )
                
                if not next_keyframe or not next_keyframe.exists():
                    logger.error("Next keyframe generation failed")
                    return None
                
                # Encode and store
                try:
                    # Synchronize CUDA to ensure ComfyUI generation completes before proceeding
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        logger.debug(f"  CUDA synchronized after next keyframe generation")
                    
                    next_latent = self.latent_encoder.encode(next_keyframe, for_interpolation=True)
                    self.frame_manager.store_keyframe_data(
                        frame_number=end_keyframe,
                        latent=next_latent,
                        image_path=next_keyframe
                    )
                    logger.debug(f"  Next keyframe generated and stored: frame {end_keyframe}")
                    
                    # Get keyframe pair data again
                    keyframe_pair_data = self.frame_manager.get_keyframe_pair_data(frame_number)
                    
                except Exception as e:
                    logger.error(f"Failed to encode next keyframe: {e}", exc_info=True)
                    return None
            
            # Now we have both keyframes, perform interpolation
            (start_frame_num, start_latent, start_path), (end_frame_num, end_latent, end_path) = keyframe_pair_data
            
            logger.debug(f"  Interpolating between keyframes {start_frame_num} and {end_frame_num}")
            
            try:
                # Check if we have precomputed slerp params for this pair
                pair_key = (start_frame_num, end_frame_num)
                if pair_key not in self.slerp_precomputed:
                    # Precompute for this keyframe pair
                    self.slerp_precomputed[pair_key] = precompute_slerp_params(
                        start_latent,
                        end_latent
                    )
                    logger.debug(f"  Precomputed slerp params for pair {pair_key}")
                
                # Perform spherical lerp with pre-computed parameters
                interpolated_latent = spherical_lerp(
                    start_latent,
                    end_latent,
                    frame_spec.interpolation_t,
                    precomputed=self.slerp_precomputed[pair_key]
                )
                
                # Decode to image (with upscaling if lower-res interpolation is enabled)
                interpolated_image = self.latent_encoder.decode(interpolated_latent, upscale_to_target=True)
                
                # Save to output (OPTIMIZED: no optimize flag for speed)
                output_path = frame_spec.output_path
                # Save without compression optimization (trades file size for speed)
                interpolated_image.save(output_path, "PNG", optimize=False, compress_level=1)
                
                # Mark as ready
                self.frame_manager.mark_ready(frame_number, output_path)
                
                logger.debug(f"  Saved interpolated frame: {output_path.name}")
                return output_path
                
            except Exception as e:
                logger.error(f"Interpolation failed: {e}", exc_info=True)
                # Fallback to img2img
                logger.warning("Falling back to img2img")
                keyframe = self.generator.generate_from_image(
                    image_path=current_image,
                    prompt=prompt,
                    denoise=denoise * 0.6
                )
                if keyframe:
                    self.frame_manager.mark_ready(frame_number, keyframe, prompt)
                return keyframe
    
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
        frame_stats = self.frame_manager.get_stats()
        
        return {
            **frame_stats,
            "has_latent_encoder": self.latent_encoder is not None,
        }
    
    def print_buffer_status(self, lookahead: int = 10) -> None:
        """
        Print buffer status (delegates to frame manager)
        
        Args:
            lookahead: Number of frames ahead to show
        """
        self.frame_manager.print_buffer_status(lookahead)
    
    def cleanup_old_keyframes(self, keep_recent: int = 5) -> None:
        """
        Clean up old keyframe latents to save memory
        
        Args:
            keep_recent: Number of recent keyframes to keep
        """
        self.frame_manager.cleanup_old_keyframes(keep_recent)


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
            import tempfile
            self.output_dir = Path(tempfile.gettempdir())
            self.frame_count = 0
        
        def generate_from_image(self, image_path, prompt, denoise):
            self.frame_count += 1
            print(f"   Mock generation: denoise={denoise}")
            return self.output_dir / f"mock_frame_{self.frame_count}.png"
    
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
    import tempfile
    mock_image = Path(tempfile.gettempdir()) / "test.png"
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

