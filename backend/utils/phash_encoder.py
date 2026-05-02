"""
Perceptual Hash Encoder
Encodes images to 8×8 perceptual hashes for structural similarity comparison

This encoder specializes in detecting compositional and structural changes.
It's blind to color, making it complementary to color-based encoders.

Key Features:
- 8×8 DCT-based perceptual hash (64 bits)
- Grayscale conversion (color-blind)
- Hamming distance → similarity metric
- Extremely fast: ~2.7ms per image
- No GPU/model required

Use Cases:
- Detect structural drift (wireframe patterns, composition)
- Detect pose/layout changes
- Cache diversity evaluation (structural component)
- Mode collapse detection (structural component)
"""

import logging
from pathlib import Path
from typing import Optional, Union

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

logger = logging.getLogger(__name__)


class PHashEncoder:
    """
    Perceptual hash encoder for detecting structural changes
    
    Uses DCT-based perceptual hashing to create 8×8 = 64-bit fingerprints
    of image structure. Focuses on spatial layout and brightness patterns,
    completely ignoring color information.
    
    Usage:
        encoder = PHashEncoder(hash_size=8)
        hash_obj = encoder.encode_image("image.png")
        similarity = PHashEncoder.similarity(hash1, hash2)
    """
    
    def __init__(self, hash_size: int = 8):
        """
        Initialize perceptual hash encoder
        
        Args:
            hash_size: DCT hash size (8×8 = 64 bits)
                      8 is optimal for speed/quality tradeoff
                      Tested: 8 (fast, good), 16 (slower, overkill)
        """
        if not HAS_IMAGEHASH:
            raise ImportError(
                "imagehash library required for PHashEncoder. "
                "Install with: pip install imagehash"
            )
        
        self.hash_size = hash_size
        logger.debug(f"PHashEncoder initialized (hash_size={hash_size})")
    
    def encode_image(self, image_input: Union[Path, str, "Image.Image"]) -> Optional[imagehash.ImageHash]:
        """
        Encode image to perceptual hash
        
        Process (handled by imagehash library):
        1. Resize to 32×32 (or 8×hash_size)
        2. Convert to grayscale
        3. Compute DCT (Discrete Cosine Transform)
        4. Extract low-frequency 8×8 coefficients
        5. Threshold to binary hash
        
        Args:
            image_input: Path to image file OR PIL Image object (for performance)
        
        Returns:
            ImageHash object (8×8 = 64 bits)
            None if encoding fails
        """
        try:
            # Handle input: Path or PIL Image
            if isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                # Load from path
                if isinstance(image_input, str):
                    image_input = Path(image_input)
                img = Image.open(image_input).convert('RGB')
            
            # Compute perceptual hash
            # This automatically converts to grayscale and applies DCT
            hash_obj = imagehash.phash(img, hash_size=self.hash_size)
            
            return hash_obj
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_input}: {e}")
            return None
    
    @staticmethod
    def similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """
        Compute similarity from Hamming distance
        
        Hamming distance counts the number of differing bits between two hashes.
        We convert this to a similarity score where 1.0 = identical.
        
        Args:
            hash1: First perceptual hash
            hash2: Second perceptual hash
        
        Returns:
            Similarity score (higher = more similar)
            Range: 0.0-1.0
            
        Interpretation (for 8×8 = 64 bits):
            0.90-1.00: Nearly identical structures
            0.75-0.90: Very similar (same composition)
            0.60-0.75: Moderately similar (related layouts)
            0.40-0.60: Dissimilar (different compositions)
            <0.40:     Very different structures
        """
        # Compute Hamming distance (number of differing bits)
        distance = hash1 - hash2  # imagehash overloads - operator
        
        # Convert to similarity (0 distance = 1.0 similarity)
        # Max distance = hash_size^2 (64 for 8×8)
        max_distance = hash1.hash.size  # Total number of bits
        similarity = 1.0 - (distance / max_distance)
        
        return float(similarity)
    
    @staticmethod
    def to_serializable(hash_obj: imagehash.ImageHash) -> str:
        """
        Convert hash to JSON-serializable format
        
        Args:
            hash_obj: ImageHash object
        
        Returns:
            Hexadecimal string representation
        """
        return str(hash_obj)
    
    @staticmethod
    def from_serializable(hash_str: str) -> imagehash.ImageHash:
        """
        Convert serialized hash back to ImageHash object
        
        Args:
            hash_str: Hexadecimal string from JSON
        
        Returns:
            ImageHash object
        """
        return imagehash.hex_to_hash(hash_str)

