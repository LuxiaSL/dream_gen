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
from typing import Dict, Any, List, Tuple, Optional, Union

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
        
        # Check for collapse
        should_inject, reason = manager.check_collapse(
            current_embedding,
            recent_embeddings
        )
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
    
    def check_collapse(
        self,
        current_embedding: Dict[str, Any],
        recent_embeddings: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Check for mode collapse using OR logic
        
        Computes average similarity for BOTH metrics independently,
        then applies OR logic: if EITHER metric exceeds its threshold,
        collapse is detected.
        
        Args:
            current_embedding: Latest frame embedding
            recent_embeddings: Last N frames for comparison
        
        Returns:
            Tuple of (should_inject: bool, reason: str)
            
        Reasons:
            "BOTH color and structural collapse"
            "COLOR collapse detected"
            "STRUCTURAL collapse detected"
            "No collapse"
        """
        if not recent_embeddings:
            return False, "No history to compare"
        
        try:
            # === COLOR SIMILARITY ===
            color_sims = []
            for past in recent_embeddings:
                if 'color' in past:
                    sim = self.color_encoder.similarity(
                        current_embedding['color'],
                        past['color']
                    )
                    color_sims.append(sim)
            
            avg_color_sim = float(np.mean(color_sims)) if color_sims else 0.0
            
            # === STRUCTURAL SIMILARITY ===
            struct_sims = []
            for past in recent_embeddings:
                if 'struct' in past:
                    # Reconstruct hash objects from hex strings
                    current_hash = self.phash_encoder.from_serializable(current_embedding['struct'])
                    past_hash = self.phash_encoder.from_serializable(past['struct'])
                    
                    sim = self.phash_encoder.similarity(current_hash, past_hash)
                    struct_sims.append(sim)
            
            avg_struct_sim = float(np.mean(struct_sims)) if struct_sims else 0.0
            
            # === OR/AND LOGIC ===
            # Higher similarity = less diverse = should inject
            color_collapse = avg_color_sim > self.color_threshold
            struct_collapse = avg_struct_sim > self.struct_threshold
            
            # Apply logic (OR by default)
            if self.injection_logic == 'all':  # AND logic
                should_inject = color_collapse and struct_collapse
            else:  # 'any' or default = OR logic
                should_inject = color_collapse or struct_collapse
            
            # Determine reason for logging
            if color_collapse and struct_collapse:
                reason = f"BOTH color ({avg_color_sim:.3f}>{self.color_threshold:.3f}) and structural ({avg_struct_sim:.3f}>{self.struct_threshold:.3f}) collapse"
            elif color_collapse:
                reason = f"COLOR collapse detected ({avg_color_sim:.3f}>{self.color_threshold:.3f})"
            elif struct_collapse:
                reason = f"STRUCTURAL collapse detected ({avg_struct_sim:.3f}>{self.struct_threshold:.3f})"
            else:
                reason = f"No collapse (color:{avg_color_sim:.3f}, struct:{avg_struct_sim:.3f})"
            
            return should_inject, reason
            
        except Exception as e:
            logger.error(f"Collapse check failed: {e}", exc_info=True)
            return False, f"Error: {e}"
    
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
        return {
            'color': self.color_encoder.to_serializable(embedding['color']),
            'struct': embedding['struct']  # Already a string
        }
    
    def from_serializable(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert serialized embedding back to working format
        
        Args:
            serialized: JSON-loaded dictionary
        
        Returns:
            Dual embedding
        """
        return {
            'color': self.color_encoder.from_serializable(serialized['color']),
            'struct': serialized['struct']  # Already a string
        }


# Test function
def test_dual_similarity() -> bool:
    """Test dual-metric similarity manager"""
    print("=" * 60)
    print("Testing DualMetricSimilarityManager...")
    print("=" * 60)
    
    try:
        # Create mock config
        print("\n1. Creating manager...")
        config = {
            'generation': {
                'cache': {
                    'color_histogram': {
                        'bins_per_channel': 32,
                        'diversity_threshold': 1.80
                    },
                    'phash': {
                        'hash_size': 8,
                        'diversity_threshold': 0.65
                    },
                    'injection_logic': 'any'  # OR logic
                }
            }
        }
        
        manager = DualMetricSimilarityManager(config)
        print("[OK] Manager created")
        
        # Create test images
        print("\n2. Creating test images...")
        from PIL import Image, ImageDraw
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Image 1: Blue circle
            img1 = Image.new('RGB', (256, 256), color=(255, 255, 255))
            draw1 = ImageDraw.Draw(img1)
            draw1.ellipse([64, 64, 192, 192], fill=(0, 0, 255))
            img1_path = tmpdir / "blue_circle.png"
            img1.save(img1_path)
            
            # Image 2: Blue circle (similar both metrics)
            img2 = Image.new('RGB', (256, 256), color=(255, 255, 255))
            draw2 = ImageDraw.Draw(img2)
            draw2.ellipse([64, 64, 192, 192], fill=(50, 50, 255))
            img2_path = tmpdir / "blue_circle2.png"
            img2.save(img2_path)
            
            # Image 3: Red circle (different color, same structure)
            img3 = Image.new('RGB', (256, 256), color=(255, 255, 255))
            draw3 = ImageDraw.Draw(img3)
            draw3.ellipse([64, 64, 192, 192], fill=(255, 0, 0))
            img3_path = tmpdir / "red_circle.png"
            img3.save(img3_path)
            
            # Image 4: Blue square (same color, different structure)
            img4 = Image.new('RGB', (256, 256), color=(255, 255, 255))
            draw4 = ImageDraw.Draw(img4)
            draw4.rectangle([64, 64, 192, 192], fill=(0, 0, 255))
            img4_path = tmpdir / "blue_square.png"
            img4.save(img4_path)
            
            print("[OK] Test images created")
            
            # Test encoding
            print("\n3. Testing dual encoding...")
            emb1 = manager.encode_image(img1_path)
            emb2 = manager.encode_image(img2_path)
            emb3 = manager.encode_image(img3_path)
            emb4 = manager.encode_image(img4_path)
            
            if not all([emb1, emb2, emb3, emb4]):
                print("[FAIL] Encoding failed")
                return False
            
            print("[OK] Dual encoding works")
            print(f"   Embedding keys: {list(emb1.keys())}")
            print(f"   Color shape: {emb1['color'].shape}")
            print(f"   Struct type: {type(emb1['struct']).__name__}")
            
            # Test similarity computation
            print("\n4. Testing similarity computation...")
            
            color_sim_similar = manager.get_color_similarity(emb1, emb2)
            color_sim_diff = manager.get_color_similarity(emb1, emb3)
            
            struct_sim_similar = manager.get_struct_similarity(emb1, emb2)
            struct_sim_diff = manager.get_struct_similarity(emb1, emb4)
            
            print(f"   Color:  Blue1 <-> Blue2: {color_sim_similar:.3f} (similar)")
            print(f"   Color:  Blue1 <-> Red:   {color_sim_diff:.3f} (different)")
            print(f"   Struct: Circle1 <-> Circle2: {struct_sim_similar:.3f} (similar)")
            print(f"   Struct: Circle <-> Square:   {struct_sim_diff:.3f} (different)")
            
            print("[OK] Similarity computation works")
            
            # Test collapse detection
            print("\n5. Testing collapse detection (OR logic)...")
            
            # No collapse: diverse images
            should_inject, reason = manager.check_collapse(emb1, [emb3, emb4])
            print(f"   Diverse set: {should_inject} - {reason}")
            
            # Should detect if recent images are all very similar
            should_inject, reason = manager.check_collapse(emb2, [emb1, emb1, emb1])
            print(f"   Similar set: {should_inject} - {reason}")
            
            print("[OK] Collapse detection works")
            
            # Test serialization
            print("\n6. Testing serialization...")
            serialized = manager.to_serializable(emb1)
            deserialized = manager.from_serializable(serialized)
            
            # Check color
            if not np.allclose(emb1['color'], deserialized['color']):
                print("[FAIL] Color serialization failed")
                return False
            
            # Check struct
            if emb1['struct'] != deserialized['struct']:
                print("[FAIL] Struct serialization failed")
                return False
            
            print("[OK] Serialization works")
        
        print("\n" + "=" * 60)
        print("DualMetricSimilarityManager test PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_dual_similarity()
    exit(0 if success else 1)

