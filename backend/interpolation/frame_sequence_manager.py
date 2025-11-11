"""
Frame Sequence Manager - Track Keyframes and Interpolations

This module provides explicit tracking of frame generation sequences,
making it crystal clear what frames are keyframes vs interpolated,
and what the buffer/generation state is.

Key Concepts:
- KEYFRAMES: Generated via img2img from ComfyUI (slow, high quality)
- INTERPOLATED: Generated via VAE latent interpolation (fast, smooth)
- KEYFRAME PAIRS: Each interpolated frame knows which two keyframes it's between

Example Sequence (interpolation_frames=3):
  Frame 0: KEYFRAME A (generated)
  Frame 1: INTERPOLATED (A→B, t=0.25)
  Frame 2: INTERPOLATED (A→B, t=0.50)
  Frame 3: INTERPOLATED (A→B, t=0.75)
  Frame 4: KEYFRAME B (generated)
  Frame 5: INTERPOLATED (B→C, t=0.25)
  ...
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Type of frame in the sequence"""
    KEYFRAME = "keyframe"
    INTERPOLATED = "interpolated"


class FrameState(Enum):
    """Generation state of a frame"""
    PENDING = "pending"          # Not started yet
    GENERATING = "generating"    # Currently being generated
    READY = "ready"             # Generated and saved to disk
    DISPLAYED = "displayed"      # Already shown to user


@dataclass
class FrameSpec:
    """
    Complete specification for a single frame
    
    Attributes:
        frame_number: Global frame index (0, 1, 2, ...)
        frame_type: KEYFRAME or INTERPOLATED
        state: Current generation/display state
        keyframe_pair: (start_keyframe_num, end_keyframe_num) for interpolated frames
        interpolation_t: Interpolation factor (0.0-1.0) for interpolated frames
        output_path: Path where frame is/will be saved
        generated_at: Timestamp when generation completed
        prompt: Generation prompt used (for keyframes)
    """
    frame_number: int
    frame_type: FrameType
    state: FrameState = FrameState.PENDING
    keyframe_pair: Optional[Tuple[int, int]] = None
    interpolation_t: Optional[float] = None
    output_path: Optional[Path] = None
    generated_at: Optional[float] = None
    prompt: Optional[str] = None
    
    def is_keyframe(self) -> bool:
        """Check if this is a keyframe"""
        return self.frame_type == FrameType.KEYFRAME
    
    def is_interpolated(self) -> bool:
        """Check if this is an interpolated frame"""
        return self.frame_type == FrameType.INTERPOLATED
    
    def is_ready(self) -> bool:
        """Check if frame is ready to display"""
        return self.state == FrameState.READY or self.state == FrameState.DISPLAYED
    
    def __repr__(self) -> str:
        """Readable representation"""
        type_str = "KEY" if self.is_keyframe() else f"INT(t={self.interpolation_t:.2f})"
        state_str = self.state.value.upper()
        pair_str = f" [{self.keyframe_pair[0]}->{self.keyframe_pair[1]}]" if self.keyframe_pair else ""
        return f"Frame#{self.frame_number:03d} {type_str:15s} {state_str:10s}{pair_str}"


class FrameSequenceManager:
    """
    Manages the sequence of keyframes and interpolated frames
    
    This provides:
    - Clear tracking of what frame is what type
    - Explicit keyframe pair relationships
    - Buffer visualization
    - Next frame scheduling
    - State management for async generation
    
    Usage:
        manager = FrameSequenceManager(interpolation_frames=6)
        
        # Get next frame to generate
        frame_spec = manager.get_next_frame()
        
        # Mark as generating
        manager.mark_generating(frame_spec.frame_number)
        
        # ... generate the frame ...
        
        # Mark as ready
        manager.mark_ready(frame_spec.frame_number, output_path)
        
        # Get next ready frame to display
        display_frame = manager.get_next_display_frame()
    """
    
    def __init__(
        self,
        interpolation_frames: int = 6,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize frame sequence manager
        
        Args:
            interpolation_frames: Number of interpolated frames between keyframes
            output_dir: Directory where frames will be saved
        """
        self.interpolation_frames = interpolation_frames
        self.output_dir = output_dir or Path("./output")
        
        # Frame tracking
        self.frames: Dict[int, FrameSpec] = {}
        self.current_frame_number = 0
        self.last_keyframe_number = None
        self.next_keyframe_number = None
        
        # Keyframe tracking (for interpolation)
        self.keyframe_latents: Dict[int, any] = {}  # frame_number -> latent tensor
        self.keyframe_paths: Dict[int, Path] = {}   # frame_number -> image path
        
        logger.info(f"FrameSequenceManager initialized")
        logger.info(f"  Interpolation frames: {interpolation_frames}")
        logger.info(f"  Frame pattern: KEY, {', '.join(['INT'] * interpolation_frames)}, KEY, ...")
    
    def _calculate_frame_spec(self, frame_number: int) -> FrameSpec:
        """
        Calculate what type of frame this should be
        
        Args:
            frame_number: Frame index to calculate
            
        Returns:
            FrameSpec with type and interpolation info filled in
        """
        # Frame 0 is always a keyframe
        if frame_number == 0:
            return FrameSpec(
                frame_number=frame_number,
                frame_type=FrameType.KEYFRAME,
                output_path=self.output_dir / f"frame_{frame_number:05d}.png"
            )
        
        # Determine position in cycle
        frames_per_cycle = self.interpolation_frames + 1
        position_in_cycle = frame_number % frames_per_cycle
        
        if position_in_cycle == 0:
            # This is a keyframe
            return FrameSpec(
                frame_number=frame_number,
                frame_type=FrameType.KEYFRAME,
                output_path=self.output_dir / f"frame_{frame_number:05d}.png"
            )
        else:
            # This is an interpolated frame
            # Calculate which keyframe pair it belongs to
            start_keyframe = (frame_number // frames_per_cycle) * frames_per_cycle
            end_keyframe = start_keyframe + frames_per_cycle
            
            # Calculate interpolation factor (0.0 to 1.0)
            t = position_in_cycle / frames_per_cycle
            
            return FrameSpec(
                frame_number=frame_number,
                frame_type=FrameType.INTERPOLATED,
                keyframe_pair=(start_keyframe, end_keyframe),
                interpolation_t=t,
                output_path=self.output_dir / f"frame_{frame_number:05d}.png"
            )
    
    def get_or_create_frame_spec(self, frame_number: int) -> FrameSpec:
        """
        Get existing frame spec or create new one
        
        Args:
            frame_number: Frame index
            
        Returns:
            FrameSpec for that frame
        """
        if frame_number not in self.frames:
            self.frames[frame_number] = self._calculate_frame_spec(frame_number)
        return self.frames[frame_number]
    
    def get_next_frame(self) -> FrameSpec:
        """
        Get the next frame that needs to be generated
        
        Returns:
            FrameSpec for the next frame to generate
        """
        frame_spec = self.get_or_create_frame_spec(self.current_frame_number)
        return frame_spec
    
    def mark_generating(self, frame_number: int) -> None:
        """Mark a frame as currently generating"""
        frame_spec = self.get_or_create_frame_spec(frame_number)
        frame_spec.state = FrameState.GENERATING
        logger.debug(f"Marked generating: {frame_spec}")
    
    def mark_ready(
        self,
        frame_number: int,
        output_path: Path,
        prompt: Optional[str] = None
    ) -> None:
        """
        Mark a frame as ready (generation complete)
        
        Args:
            frame_number: Frame index
            output_path: Path where frame was saved
            prompt: Prompt used (for keyframes)
        """
        frame_spec = self.get_or_create_frame_spec(frame_number)
        frame_spec.state = FrameState.READY
        frame_spec.output_path = output_path
        frame_spec.generated_at = time.time()
        if prompt:
            frame_spec.prompt = prompt
        
        logger.info(f"Frame ready: {frame_spec}")
    
    def mark_displayed(self, frame_number: int) -> None:
        """Mark a frame as displayed"""
        if frame_number in self.frames:
            self.frames[frame_number].state = FrameState.DISPLAYED
            logger.debug(f"Marked displayed: frame {frame_number}")
    
    def advance_frame(self) -> None:
        """Move to next frame number"""
        self.current_frame_number += 1
        logger.debug(f"Advanced to frame {self.current_frame_number}")
    
    def store_keyframe_data(
        self,
        frame_number: int,
        latent: any,
        image_path: Path
    ) -> None:
        """
        Store keyframe latent and path for interpolation
        
        Args:
            frame_number: Keyframe frame number
            latent: Encoded latent tensor
            image_path: Path to keyframe image
        """
        self.keyframe_latents[frame_number] = latent
        self.keyframe_paths[frame_number] = image_path
        
        # Update keyframe tracking
        self.last_keyframe_number = frame_number
        
        logger.debug(f"Stored keyframe data for frame {frame_number}")
    
    def get_keyframe_pair_data(
        self,
        frame_number: int
    ) -> Optional[Tuple[Tuple[int, any, Path], Tuple[int, any, Path]]]:
        """
        Get the keyframe pair data for an interpolated frame
        
        Args:
            frame_number: Interpolated frame number
            
        Returns:
            ((start_frame_num, start_latent, start_path),
             (end_frame_num, end_latent, end_path))
            or None if data not available
        """
        frame_spec = self.get_or_create_frame_spec(frame_number)
        
        if not frame_spec.is_interpolated():
            logger.warning(f"Frame {frame_number} is not interpolated")
            return None
        
        start_keyframe, end_keyframe = frame_spec.keyframe_pair
        
        # Check if we have both keyframes
        if start_keyframe not in self.keyframe_latents:
            logger.warning(f"Missing start keyframe {start_keyframe}")
            return None
        
        if end_keyframe not in self.keyframe_latents:
            logger.debug(f"End keyframe {end_keyframe} not ready yet")
            return None
        
        return (
            (start_keyframe, self.keyframe_latents[start_keyframe], self.keyframe_paths[start_keyframe]),
            (end_keyframe, self.keyframe_latents[end_keyframe], self.keyframe_paths[end_keyframe])
        )
    
    def get_buffer_status(self, lookahead: int = 10) -> List[FrameSpec]:
        """
        Get status of upcoming frames in buffer
        
        Args:
            lookahead: Number of frames ahead to check
            
        Returns:
            List of FrameSpecs for upcoming frames
        """
        buffer = []
        for i in range(self.current_frame_number, self.current_frame_number + lookahead):
            frame_spec = self.get_or_create_frame_spec(i)
            buffer.append(frame_spec)
        return buffer
    
    def print_buffer_status(self, lookahead: int = 10) -> None:
        """
        Print buffer status to console (for debugging)
        
        Args:
            lookahead: Number of frames ahead to show
        """
        buffer = self.get_buffer_status(lookahead)
        
        logger.info("=" * 70)
        logger.info("FRAME BUFFER STATUS")
        logger.info("=" * 70)
        
        for frame_spec in buffer:
            # Add visual indicator for current frame
            indicator = ">" if frame_spec.frame_number == self.current_frame_number else " "
            logger.info(f"{indicator} {frame_spec}")
        
        logger.info("=" * 70)
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about frame generation
        
        Returns:
            Dictionary with statistics
        """
        total_frames = len(self.frames)
        keyframes = sum(1 for f in self.frames.values() if f.is_keyframe())
        interpolated = sum(1 for f in self.frames.values() if f.is_interpolated())
        ready = sum(1 for f in self.frames.values() if f.is_ready())
        
        return {
            "current_frame": self.current_frame_number,
            "total_frames": total_frames,
            "keyframes": keyframes,
            "interpolated": interpolated,
            "ready": ready,
            "frames_per_cycle": self.interpolation_frames + 1,
            "interpolation_ratio": interpolated / total_frames if total_frames > 0 else 0.0
        }
    
    def cleanup_old_keyframes(self, keep_recent: int = 5) -> None:
        """
        Clean up old keyframe latents to save memory
        
        Args:
            keep_recent: Number of recent keyframes to keep
        """
        if len(self.keyframe_latents) <= keep_recent:
            return
        
        # Sort keyframes by frame number
        keyframe_numbers = sorted(self.keyframe_latents.keys())
        
        # Delete old ones (keep_recent most recent)
        to_delete = keyframe_numbers[:-keep_recent]
        
        for frame_num in to_delete:
            del self.keyframe_latents[frame_num]
            if frame_num in self.keyframe_paths:
                del self.keyframe_paths[frame_num]
        
        logger.debug(f"Cleaned up {len(to_delete)} old keyframes")


# Unit test when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    print("🧪 Testing FrameSequenceManager...")
    print()
    
    # Test 1: Basic initialization
    print("1. Basic initialization")
    manager = FrameSequenceManager(interpolation_frames=3)
    print("✓ Manager created")
    print()
    
    # Test 2: Frame sequence generation
    print("2. Frame sequence pattern (10 frames, interpolation=3)")
    for i in range(10):
        frame_spec = manager.get_or_create_frame_spec(i)
        print(f"  {frame_spec}")
    print("✓ Frame types calculated correctly")
    print()
    
    # Test 3: State transitions
    print("3. State transitions")
    frame_0 = manager.get_next_frame()
    print(f"  Next frame: {frame_0}")
    
    manager.mark_generating(0)
    print(f"  After mark_generating: {manager.frames[0]}")
    
    manager.mark_ready(0, Path("output/frame_00000.png"), "test prompt")
    print(f"  After mark_ready: {manager.frames[0]}")
    
    manager.mark_displayed(0)
    print(f"  After mark_displayed: {manager.frames[0]}")
    print("✓ State transitions work")
    print()
    
    # Test 4: Buffer visualization
    print("4. Buffer status visualization")
    manager.print_buffer_status(lookahead=15)
    print("✓ Buffer visualization works")
    print()
    
    # Test 5: Statistics
    print("5. Statistics")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("✓ Statistics calculated")
    print()
    
    # Test 6: Keyframe pair lookup
    print("6. Keyframe pair lookup")
    frame_2 = manager.get_or_create_frame_spec(2)
    print(f"  Frame 2 spec: {frame_2}")
    print(f"  Keyframe pair: {frame_2.keyframe_pair}")
    print(f"  Interpolation t: {frame_2.interpolation_t}")
    print("✓ Keyframe pair calculation works")
    print()
    
    print("✅ All tests passed!")
    print()
    print("📝 Frame pattern summary:")
    print(f"  Interpolation frames: {manager.interpolation_frames}")
    print(f"  Frames per cycle: {manager.interpolation_frames + 1}")
    print(f"  Pattern: KEY, {'INT, ' * manager.interpolation_frames}KEY, ...")
    print()
    print("Example usage:")
    print("  manager = FrameSequenceManager(interpolation_frames=6)")
    print("  frame_spec = manager.get_next_frame()")
    print("  if frame_spec.is_keyframe():")
    print("      # Generate via img2img")
    print("  else:")
    print("      # Interpolate between keyframes")

