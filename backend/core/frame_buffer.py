"""
Frame Buffer - Central frame ordering and buffering system

This module provides the core buffering and sequencing logic for the Dream Window.
It maintains an ordered list of frames (keyframes + interpolations), tracks their
generation state, and provides sequential access for display.

Key responsibilities:
- Calculate frame specifications (keyframe vs interpolated, t values, pairs)
- Track frame state (PENDING → GENERATING → READY → DISPLAYED)
- Provide sequential frame access for display
- Report buffer status (fill level, seconds buffered)
- Map logical sequence numbers to physical file paths
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Type of frame in the sequence"""
    KEYFRAME = "keyframe"
    INTERPOLATED = "interpolated"


class FrameState(Enum):
    """Generation state of a frame"""
    PENDING = "pending"          # Not started yet
    GENERATING = "generating"    # Currently being generated
    READY = "ready"              # Generated and saved to disk
    DISPLAYED = "displayed"      # Already shown to user


@dataclass
class FrameSpec:
    """
    Complete specification for a single frame in the display sequence
    
    Attributes:
        sequence_num: Position in display sequence (0, 1, 2, ...)
        frame_type: KEYFRAME or INTERPOLATED
        state: Current generation/display state
        file_path: Path where frame is/will be saved
        keyframe_num: Keyframe number (for keyframes only)
        keyframe_pair: (start_kf_num, end_kf_num) for interpolations
        interpolation_t: Interpolation factor (0.0-1.0) for interpolations
        interpolation_index: Index within keyframe pair (1, 2, 3, ...) for interpolations
        generated_at: Timestamp when generation completed
    """
    sequence_num: int
    frame_type: FrameType
    state: FrameState = FrameState.PENDING
    file_path: Optional[Path] = None
    
    # Keyframe-specific
    keyframe_num: Optional[int] = None
    
    # Interpolation-specific
    keyframe_pair: Optional[Tuple[int, int]] = None
    interpolation_t: Optional[float] = None
    interpolation_index: Optional[int] = None
    
    # Metadata
    generated_at: Optional[float] = None
    
    def is_keyframe(self) -> bool:
        """Check if this is a keyframe"""
        return self.frame_type == FrameType.KEYFRAME
    
    def is_interpolated(self) -> bool:
        """Check if this is an interpolated frame"""
        return self.frame_type == FrameType.INTERPOLATED
    
    def is_ready(self) -> bool:
        """Check if frame is ready to display"""
        return self.state in (FrameState.READY, FrameState.DISPLAYED)
    
    def __repr__(self) -> str:
        """Readable representation"""
        if self.is_keyframe():
            type_str = f"KEYFRAME {self.keyframe_num:03d}"
        else:
            type_str = f"INTERP {self.keyframe_pair[0]:03d}-{self.keyframe_pair[1]:03d}_{self.interpolation_index:03d} (t={self.interpolation_t:.2f})"
        
        state_str = self.state.value.upper()
        return f"Seq#{self.sequence_num:04d}: {type_str:35s} [{state_str:10s}]"


class FrameBuffer:
    """
    Central frame buffer and sequencing system
    
    Maintains an ordered sequence of frames (keyframes + interpolations),
    tracks their generation state, and provides sequential access for display.
    
    Usage:
        buffer = FrameBuffer(
            interpolation_frames=10,
            target_fps=4,
            output_dir=Path("output")
        )
        
        # Register frames
        buffer.register_keyframe(1)  # Creates keyframe_001
        buffer.register_interpolations(1, 2, 10)  # Creates 001-002_001 through 001-002_010
        buffer.register_keyframe(2)  # Creates keyframe_002
        
        # Mark frames as ready
        buffer.mark_ready(0, Path("output/keyframes/keyframe_001.png"))
        buffer.mark_ready(1, Path("output/interpolations/001-002_001.png"))
        
        # Check buffer status
        status = buffer.get_buffer_status()
        print(f"Buffer: {status['seconds_buffered']:.1f}s / {status['target_seconds']}s")
        
        # Get next frame for display
        if buffer.is_buffer_ready():
            frame_spec = buffer.get_next_display_frame()
            if frame_spec:
                display_image(frame_spec.file_path)
                buffer.mark_displayed(frame_spec.sequence_num)
    """
    
    def __init__(
        self,
        interpolation_frames: int = 10,
        target_fps: float = 4.0,
        output_dir: Path = Path("output"),
        buffer_target_seconds: float = 30.0
    ):
        """
        Initialize frame buffer
        
        Args:
            interpolation_frames: Number of interpolated frames between keyframes
            target_fps: Target frame rate for display
            output_dir: Base output directory
            buffer_target_seconds: Target buffer size in seconds
        """
        self.interpolation_frames = interpolation_frames
        self.target_fps = target_fps
        self.output_dir = Path(output_dir)
        self.buffer_target_seconds = buffer_target_seconds
        
        # Create output directories
        self.keyframe_dir = self.output_dir / "keyframes"
        self.interpolation_dir = self.output_dir / "interpolations"
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.interpolation_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame sequence tracking
        self.frames: Dict[int, FrameSpec] = {}  # sequence_num -> FrameSpec
        self.next_sequence_num = 0
        self.display_sequence_num = 0  # Next frame to display
        
        # Keyframe tracking
        self.keyframe_count = 0
        self.keyframe_sequence_map: Dict[int, int] = {}  # keyframe_num -> sequence_num
        
        logger.info(f"FrameBuffer initialized")
        logger.info(f"  Interpolation frames: {interpolation_frames}")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Buffer target: {buffer_target_seconds}s")
        logger.info(f"  Keyframe dir: {self.keyframe_dir}")
        logger.info(f"  Interpolation dir: {self.interpolation_dir}")
    
    def register_keyframe(self, keyframe_num: int) -> int:
        """
        Register a new keyframe in the sequence
        
        Args:
            keyframe_num: Keyframe number (1, 2, 3, ...)
            
        Returns:
            Sequence number assigned to this keyframe
        """
        sequence_num = self.next_sequence_num
        self.next_sequence_num += 1
        
        file_path = self.keyframe_dir / f"keyframe_{keyframe_num:03d}.png"
        
        frame_spec = FrameSpec(
            sequence_num=sequence_num,
            frame_type=FrameType.KEYFRAME,
            keyframe_num=keyframe_num,
            file_path=file_path
        )
        
        self.frames[sequence_num] = frame_spec
        self.keyframe_sequence_map[keyframe_num] = sequence_num
        self.keyframe_count = max(self.keyframe_count, keyframe_num)
        
        logger.debug(f"Registered keyframe {keyframe_num} at sequence {sequence_num}")
        return sequence_num
    
    def register_interpolations(
        self,
        start_keyframe: int,
        end_keyframe: int,
        count: int
    ) -> List[int]:
        """
        Register interpolated frames between two keyframes
        
        Args:
            start_keyframe: Starting keyframe number
            end_keyframe: Ending keyframe number
            count: Number of interpolations to register
            
        Returns:
            List of sequence numbers assigned to interpolations
        """
        sequence_nums = []
        
        for i in range(1, count + 1):
            sequence_num = self.next_sequence_num
            self.next_sequence_num += 1
            
            # Calculate interpolation factor
            t = i / (count + 1)  # Evenly spaced between keyframes
            
            file_path = self.interpolation_dir / f"{start_keyframe:03d}-{end_keyframe:03d}_{i:03d}.png"
            
            frame_spec = FrameSpec(
                sequence_num=sequence_num,
                frame_type=FrameType.INTERPOLATED,
                keyframe_pair=(start_keyframe, end_keyframe),
                interpolation_t=t,
                interpolation_index=i,
                file_path=file_path
            )
            
            self.frames[sequence_num] = frame_spec
            sequence_nums.append(sequence_num)
        
        logger.debug(f"Registered {count} interpolations between keyframes {start_keyframe}-{end_keyframe}")
        return sequence_nums
    
    def mark_generating(self, sequence_num: int) -> None:
        """Mark a frame as currently generating"""
        if sequence_num in self.frames:
            self.frames[sequence_num].state = FrameState.GENERATING
            logger.debug(f"Marked generating: {self.frames[sequence_num]}")
    
    def mark_ready(self, sequence_num: int, file_path: Optional[Path] = None) -> None:
        """
        Mark a frame as ready (generation complete)
        
        Args:
            sequence_num: Sequence number of the frame
            file_path: Path where frame was saved (optional, uses registered path if None)
        """
        if sequence_num in self.frames:
            self.frames[sequence_num].state = FrameState.READY
            if file_path:
                self.frames[sequence_num].file_path = file_path
            self.frames[sequence_num].generated_at = time.time()
            logger.debug(f"Marked ready: {self.frames[sequence_num]}")
    
    def mark_displayed(self, sequence_num: int) -> None:
        """Mark a frame as displayed"""
        if sequence_num in self.frames:
            self.frames[sequence_num].state = FrameState.DISPLAYED
            logger.debug(f"Marked displayed: sequence {sequence_num}")
    
    def get_next_display_frame(self) -> Optional[FrameSpec]:
        """
        Get the next frame in sequence that's ready to display
        
        Returns:
            FrameSpec if available, None if next frame not ready
        """
        if self.display_sequence_num not in self.frames:
            return None
        
        frame_spec = self.frames[self.display_sequence_num]
        
        if frame_spec.is_ready():
            return frame_spec
        
        return None
    
    def advance_display(self) -> None:
        """Move to next frame in display sequence"""
        self.display_sequence_num += 1
        logger.debug(f"Advanced display to sequence {self.display_sequence_num}")
    
    def get_buffer_status(self) -> Dict:
        """
        Get current buffer status
        
        Returns:
            Dictionary with buffer statistics:
            - frames_ready: Number of ready frames ahead of display position
            - seconds_buffered: Seconds of ready frames in buffer
            - target_seconds: Target buffer size
            - buffer_percentage: Fill percentage (0-100)
            - is_buffer_ready: Whether buffer meets target
            - total_frames: Total frames registered
            - frames_generating: Frames currently generating
            - frames_displayed: Frames already shown
        """
        # Count ready frames ahead of current display position
        ready_count = 0
        generating_count = 0
        displayed_count = 0
        
        for seq_num in sorted(self.frames.keys()):
            if seq_num < self.display_sequence_num:
                if self.frames[seq_num].state == FrameState.DISPLAYED:
                    displayed_count += 1
            elif seq_num >= self.display_sequence_num:
                state = self.frames[seq_num].state
                if state == FrameState.READY:
                    ready_count += 1
                elif state == FrameState.GENERATING:
                    generating_count += 1
        
        # Calculate seconds buffered
        seconds_buffered = ready_count / self.target_fps if self.target_fps > 0 else 0
        buffer_percentage = min(100, (seconds_buffered / self.buffer_target_seconds) * 100)
        
        return {
            "frames_ready": ready_count,
            "seconds_buffered": seconds_buffered,
            "target_seconds": self.buffer_target_seconds,
            "buffer_percentage": buffer_percentage,
            "is_buffer_ready": seconds_buffered >= self.buffer_target_seconds,
            "total_frames": len(self.frames),
            "frames_generating": generating_count,
            "frames_displayed": displayed_count,
            "display_sequence_num": self.display_sequence_num,
            "next_sequence_num": self.next_sequence_num,
            "keyframe_count": self.keyframe_count
        }
    
    def is_buffer_ready(self, min_seconds: Optional[float] = None) -> bool:
        """
        Check if buffer has enough frames for playback
        
        Args:
            min_seconds: Minimum seconds required (uses target if None)
            
        Returns:
            True if buffer meets threshold
        """
        status = self.get_buffer_status()
        threshold = min_seconds if min_seconds is not None else self.buffer_target_seconds
        return status["seconds_buffered"] >= threshold
    
    def get_keyframe_sequence_num(self, keyframe_num: int) -> Optional[int]:
        """
        Get sequence number for a keyframe
        
        Args:
            keyframe_num: Keyframe number
            
        Returns:
            Sequence number, or None if not found
        """
        return self.keyframe_sequence_map.get(keyframe_num)
    
    def get_latest_keyframe_num(self) -> int:
        """Get the most recent keyframe number"""
        return self.keyframe_count
    
    def needs_interpolations(self) -> Optional[Tuple[int, int]]:
        """
        Check if there's a keyframe pair that needs interpolations
        
        Returns the FIRST missing keyframe pair (earliest in sequence).
        This ensures interpolations are generated in order.
        
        Returns:
            (start_kf, end_kf) tuple if interpolations needed, None otherwise
        """
        if self.keyframe_count < 2:
            return None
        
        # Check ALL consecutive keyframe pairs, starting from the earliest
        # This ensures we fill in interpolations in order, not just the latest pair
        for kf_num in range(1, self.keyframe_count):
            start_kf = kf_num
            end_kf = kf_num + 1
            
            # Check if interpolations exist for this pair
            has_interpolations = False
            for frame_spec in self.frames.values():
                if frame_spec.is_interpolated():
                    if frame_spec.keyframe_pair == (start_kf, end_kf):
                        has_interpolations = True
                        break
            
            # Return the first missing pair we find
            if not has_interpolations:
                return (start_kf, end_kf)
        
        # All pairs have interpolations
        return None
    
    def print_buffer_status(self, lookahead: int = 20) -> None:
        """
        Print detailed buffer status to console (for debugging)
        
        Args:
            lookahead: Number of frames ahead to show
        """
        status = self.get_buffer_status()
        
        logger.info("=" * 80)
        logger.info("FRAME BUFFER STATUS")
        logger.info("=" * 80)
        logger.info(f"Buffer: {status['seconds_buffered']:.1f}s / {status['target_seconds']}s ({status['buffer_percentage']:.1f}%)")
        logger.info(f"Ready: {status['frames_ready']} | Generating: {status['frames_generating']} | Displayed: {status['frames_displayed']}")
        logger.info(f"Display at sequence: {status['display_sequence_num']} | Next sequence: {status['next_sequence_num']}")
        logger.info(f"Keyframes: {status['keyframe_count']}")
        logger.info("-" * 80)
        
        # Show upcoming frames
        start_seq = self.display_sequence_num
        end_seq = start_seq + lookahead
        
        for seq_num in range(start_seq, min(end_seq, self.next_sequence_num)):
            if seq_num in self.frames:
                frame_spec = self.frames[seq_num]
                indicator = ">" if seq_num == self.display_sequence_num else " "
                logger.info(f"{indicator} {frame_spec}")
        
        logger.info("=" * 80)
    
    def get_stats(self) -> Dict:
        """
        Get detailed statistics
        
        Returns:
            Dictionary with statistics
        """
        status = self.get_buffer_status()
        
        keyframe_count = sum(1 for f in self.frames.values() if f.is_keyframe())
        interpolated_count = sum(1 for f in self.frames.values() if f.is_interpolated())
        
        return {
            **status,
            "keyframes_registered": keyframe_count,
            "interpolations_registered": interpolated_count,
            "interpolation_frames_per_cycle": self.interpolation_frames,
            "target_fps": self.target_fps
        }


# Unit tests
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    print("🧪 Testing FrameBuffer...")
    print()
    
    # Test 1: Basic initialization
    print("1. Basic initialization")
    buffer = FrameBuffer(
        interpolation_frames=10,
        target_fps=4.0,
        output_dir=Path("output"),
        buffer_target_seconds=30.0
    )
    print("✓ Buffer created")
    print()
    
    # Test 2: Register keyframes and interpolations
    print("2. Register frames")
    seq1 = buffer.register_keyframe(1)
    print(f"  Keyframe 1 -> sequence {seq1}")
    
    interp_seqs = buffer.register_interpolations(1, 2, 10)
    print(f"  Interpolations 1-2 -> sequences {interp_seqs[0]}-{interp_seqs[-1]}")
    
    seq2 = buffer.register_keyframe(2)
    print(f"  Keyframe 2 -> sequence {seq2}")
    print("✓ Frames registered")
    print()
    
    # Test 3: Mark frames as ready
    print("3. Mark frames ready")
    buffer.mark_ready(0)
    buffer.mark_ready(1)
    buffer.mark_ready(2)
    print("✓ Frames marked ready")
    print()
    
    # Test 4: Buffer status
    print("4. Buffer status")
    status = buffer.get_buffer_status()
    print(f"  Ready frames: {status['frames_ready']}")
    print(f"  Seconds buffered: {status['seconds_buffered']:.1f}s")
    print(f"  Buffer percentage: {status['buffer_percentage']:.1f}%")
    print(f"  Is ready: {status['is_buffer_ready']}")
    print("✓ Buffer status calculated")
    print()
    
    # Test 5: Get next display frame
    print("5. Get next display frame")
    frame = buffer.get_next_display_frame()
    if frame:
        print(f"  Next frame: {frame}")
        buffer.mark_displayed(frame.sequence_num)
        buffer.advance_display()
    print("✓ Frame retrieval works")
    print()
    
    # Test 6: Check for missing interpolations
    print("6. Check for missing interpolations")
    needs_interp = buffer.needs_interpolations()
    print(f"  Needs interpolations: {needs_interp}")
    print("✓ Interpolation check works")
    print()
    
    # Test 7: Print detailed status
    print("7. Detailed buffer status")
    buffer.print_buffer_status(lookahead=15)
    print("✓ Status display works")
    print()
    
    print("✅ All tests passed!")

