"""
Dual-Metric Similarity Manager
Coordinates ColorHist + pHash-8 with OR logic for mode collapse detection

This instantiates a dual-watchdog system:
- ColorHist: Detects color palette drift (mono → magenta → cyan)
- pHash-8: Detects structural drift (wireframe patterns, composition)
- OR Logic: Either metric triggers = injection/caching

Architecture:
    ColorHist + pHash-8 → OR Logic → Collapse Detection
         ↓           ↓
    Color drift   Structural drift
         ↓           ↓
         └─────OR────┘
               ↓
         Should inject?
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.color_encoder import ColorHistogramEncoder
from utils.phash_encoder import PHashEncoder

logger = logging.getLogger(__name__)


class DualMetricSimilarityManager:
    """
    Dual-metric similarity manager with OR logic
    
    Coordinates ColorHistogramEncoder and PHashEncoder to provide
    comprehensive collapse detection that catches both color AND
    structural drift.
    
    Usage:
        manager = DualMetricSimilarityManager(config)

        # Encode image with both metrics
        embedding = manager.encode_image(image_path)
        # Returns: {'color': hist[96], 'struct': hash_hex}
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dual-metric manager
        
        Args:
            config: Configuration dictionary with cache settings
        """
        self.config = config
        cache_config = config['generation']['cache']
        
        # Initialize encoders
        color_config = cache_config.get('color_histogram', {})
        phash_config = cache_config.get('phash', {})
        
        self.color_encoder = ColorHistogramEncoder(
            bins_per_channel=color_config.get('bins_per_channel', 32)
        )
        
        self.phash_encoder = PHashEncoder(
            hash_size=phash_config.get('hash_size', 8)
        )
        
        # Thresholds for collapse detection
        self.color_threshold = color_config.get('diversity_threshold', 1.80)
        self.struct_threshold = phash_config.get('diversity_threshold', 0.65)
        
        # OR vs AND logic
        self.injection_logic = cache_config.get('injection_logic', 'any')  # 'any' = OR, 'all' = AND
        
        logger.info("DualMetricSimilarityManager initialized")
        logger.info(f"  Color threshold: {self.color_threshold:.2f}")
        logger.info(f"  Struct threshold: {self.struct_threshold:.2f}")
        logger.info(f"  Injection logic: {self.injection_logic} (OR logic)" if self.injection_logic == 'any' else f"  Injection logic: {self.injection_logic} (AND logic)")
    
    def encode_image(self, image_input: Union[Path, str, 'Image.Image']) -> Optional[Dict[str, Any]]:
        """
        Encode image with BOTH metrics
        
        Args:
            image_input: Path to image file OR PIL Image object (for performance)
        
        Returns:
            Dictionary with dual embeddings:
            {
                'color': np.ndarray[96],  # ColorHist embedding
                'struct': str             # pHash hex string
            }
            None if encoding fails
        """
        try:
            # Encode with color histogram
            color_hist = self.color_encoder.encode_image(image_input)
            if color_hist is None:
                logger.warning(f"Color encoding failed for {image_input}")
                return None
            
            # Encode with perceptual hash
            phash_obj = self.phash_encoder.encode_image(image_input)
            if phash_obj is None:
                logger.warning(f"Structural encoding failed for {image_input}")
                return None
            
            # Convert to serializable format
            embedding = {
                'color': color_hist,  # Keep as numpy array for internal use
                'struct': str(phash_obj)  # Convert to hex string
            }
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_input}: {e}", exc_info=True)
            return None
    
    def get_color_similarity(
        self,
        embedding1: Dict[str, Any],
        embedding2: Dict[str, Any]
    ) -> float:
        """
        Get color similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Color similarity score
        """
        return self.color_encoder.similarity(
            embedding1['color'],
            embedding2['color']
        )
    
    def get_struct_similarity(
        self,
        embedding1: Dict[str, Any],
        embedding2: Dict[str, Any]
    ) -> float:
        """
        Get structural similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Structural similarity score
        """
        hash1 = self.phash_encoder.from_serializable(embedding1['struct'])
        hash2 = self.phash_encoder.from_serializable(embedding2['struct'])
        
        return self.phash_encoder.similarity(hash1, hash2)
    
    def to_serializable(self, embedding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert embedding to JSON-serializable format
        
        Args:
            embedding: Dual embedding
        
        Returns:
            JSON-serializable dictionary
        """
        serialized = {
            'color': self.color_encoder.to_serializable(embedding['color']),
            'struct': embedding['struct']  # Already a string
        }
        # Pooled latent (cache/latent_pool.py) rides along when present
        if embedding.get('latent') is not None:
            lat = embedding['latent']
            serialized['latent'] = [float(x) for x in lat]
        return serialized
    
    def from_serializable(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert serialized embedding back to working format
        
        Args:
            serialized: JSON-loaded dictionary
        
        Returns:
            Dual embedding
        """
        embedding = {
            'color': self.color_encoder.from_serializable(serialized['color']),
            'struct': serialized['struct']  # Already a string
        }
        if serialized.get('latent') is not None:
            embedding['latent'] = serialized['latent']  # list of floats
        return embedding

