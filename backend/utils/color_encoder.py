"""
Color Histogram Encoder
Encodes images to HSV color histograms for color-based similarity comparison

This encoder specializes in detecting color palette drift and brightness changes.
It's blind to spatial structure, making it complementary to structural encoders.

Key Features:
- HSV color space (better than RGB for perceptual similarity)
- 32 bins per channel (H, S, V) = 96-dim feature vector
- Histogram intersection similarity metric
- Fast: ~39ms per image
- No GPU/model required

Use Cases:
- Detect color drift (mono → magenta → cyan)
- Detect brightness distribution changes
- Cache diversity evaluation (color component)
- Mode collapse detection (color component)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ColorHistogramEncoder:
    """
    HSV color histogram encoder for detecting color palette changes
    
    Converts images to 96-dimensional color histograms (32 bins × 3 channels)
    for fast color similarity comparison. Complements structural encoders
    by focusing purely on color distribution.
    
    Usage:
        encoder = ColorHistogramEncoder(bins_per_channel=32)
        hist = encoder.encode_image("image.png")
        similarity = ColorHistogramEncoder.similarity(hist1, hist2)
    """
    
    def __init__(self, bins_per_channel: int = 32):
        """
        Initialize color histogram encoder
        
        Args:
            bins_per_channel: Number of histogram bins per channel (H, S, V)
                             32 bins = 96-dim vector, good balance of detail vs speed
                             Tested: 16 (too coarse), 32 (optimal), 64 (overkill)
        """
        self.bins = bins_per_channel
        logger.debug(f"ColorHistogramEncoder initialized (bins={bins_per_channel})")
    
    def encode_image(self, image_input: Union[Path, str, Image.Image]) -> Optional[np.ndarray]:
        """
        Encode image to HSV color histogram
        
        Process:
        1. Load image and convert RGB → HSV
        2. Compute histogram for each channel (H, S, V)
        3. Normalize each histogram (sum to 1.0)
        4. Concatenate [h_hist, s_hist, v_hist] → 96-dim vector
        
        Args:
            image_input: Path to image file OR PIL Image object (for performance)
        
        Returns:
            96-dim normalized histogram (32 bins × 3 channels)
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
            
            # Convert to HSV for better color representation
            # HSV separates color (H) from intensity (V), making it more perceptual
            hsv = img.convert('HSV')
            h, s, v = hsv.split()
            
            # Compute histograms (32 bins each channel)
            # Range is 0-256 for PIL's HSV mode
            h_hist = np.histogram(np.array(h), bins=self.bins, range=(0, 256))[0]
            s_hist = np.histogram(np.array(s), bins=self.bins, range=(0, 256))[0]
            v_hist = np.histogram(np.array(v), bins=self.bins, range=(0, 256))[0]
            
            # Normalize histograms (each sums to 1.0)
            # This makes them invariant to image size
            h_hist = h_hist.astype(np.float32) / h_hist.sum()
            s_hist = s_hist.astype(np.float32) / s_hist.sum()
            v_hist = v_hist.astype(np.float32) / v_hist.sum()
            
            # Concatenate into single 96-dim feature vector
            histogram = np.concatenate([h_hist, s_hist, v_hist])
            
            return histogram
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_input}: {e}")
            return None
    
    @staticmethod
    def similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute histogram intersection similarity
        
        Histogram intersection is a standard similarity metric for histograms.
        It computes the sum of minimum values across all bins, representing
        the "overlap" between two distributions.
        
        Args:
            hist1: First histogram (96-dim)
            hist2: Second histogram (96-dim)
        
        Returns:
            Similarity score (higher = more similar)
            Range: ~0.8-2.3 for typical aesthetic variations
            
        Interpretation:
            2.0-2.3: Nearly identical color palettes
            1.5-2.0: Very similar (same palette, slight shifts)
            1.0-1.5: Moderately similar (related palettes)
            0.8-1.0: Dissimilar (different color schemes)
            <0.8:    Very different (e.g., mono → magenta)
        """
        # Histogram intersection: sum of minimum values across all bins
        intersection = np.minimum(hist1, hist2).sum()
        return float(intersection)
    
    @staticmethod
    def to_serializable(histogram: np.ndarray) -> list:
        """
        Convert histogram to JSON-serializable format
        
        Args:
            histogram: NumPy histogram array
        
        Returns:
            List of floats (can be serialized to JSON)
        """
        return histogram.tolist()
    
    @staticmethod
    def from_serializable(histogram_list: list) -> np.ndarray:
        """
        Convert serialized histogram back to NumPy array
        
        Args:
            histogram_list: List of floats from JSON
        
        Returns:
            NumPy histogram array
        """
        return np.array(histogram_list, dtype=np.float32)

