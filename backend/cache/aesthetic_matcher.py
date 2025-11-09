"""
Aesthetic Matcher
Uses CLIP embeddings to find visually similar cached images

CLIP (Contrastive Language-Image Pre-Training) encodes images into a
512-dimensional semantic space where similar images are close together.

This allows us to:
1. Find cached images similar to the current frame
2. Inject variety while maintaining aesthetic coherence
3. Prevent mode collapse by intelligently mixing in diverse frames
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class AestheticMatcher:
    """
    Match images based on CLIP embeddings
    
    This is the KEY to intelligent cache injection!
    
    How it works:
    1. Encode images to 512-dim CLIP embeddings
    2. Compute cosine similarity between embeddings
    3. Find cached images with similar aesthetic
    4. Use weighted random selection (higher similarity = higher probability)
    
    Result: Variety without losing coherence!
    """

    def __init__(self):
        """Initialize aesthetic matcher with CLIP model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load CLIP model
        
        Model: openai/clip-vit-base-patch32
        - Size: ~600MB download (one-time)
        - Embedding dimension: 512
        - Fast inference: ~50-100ms per image
        """
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("Loading CLIP model...")
            model_name = "openai/clip-vit-base-patch32"

            self.processor = CLIPProcessor.from_pretrained(model_name)
            # Use safetensors format (modern, safer, and doesn't require torch 2.6+)
            self.model = CLIPModel.from_pretrained(
                model_name,
                use_safetensors=True  # Force safetensors format
            ).to(self.device)
            self.model.eval()  # Inference mode

            logger.info(f"CLIP model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}", exc_info=True)
            raise RuntimeError(
                "CLIP model required for aesthetic matching. "
                "Install with: pip install transformers"
            ) from e

    def encode_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Encode image to CLIP embedding
        
        Process:
        1. Load and preprocess image
        2. Pass through CLIP vision encoder
        3. Normalize to unit vector (for cosine similarity)
        
        Args:
            image_path: Path to image file
        
        Returns:
            512-dimensional normalized embedding, or None on failure
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize to unit vector
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)

            logger.debug(f"Encoded image: {image_path.name}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    def encode_images_batch(
        self, image_paths: List[Path], batch_size: int = 8
    ) -> List[Optional[np.ndarray]]:
        """
        Encode multiple images in batches (more efficient)
        
        Batching is ~3x faster than encoding individually
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images per batch
        
        Returns:
            List of embeddings (None for failed images)
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            try:
                # Load images
                images = []
                valid_indices = []
                for j, path in enumerate(batch_paths):
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        valid_indices.append(i + j)
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")
                        embeddings.append(None)

                if not images:
                    continue

                # Preprocess batch
                inputs = self.processor(
                    images=images, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                # Normalize
                batch_embeddings = image_features.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )

                # Add to results
                for emb in batch_embeddings:
                    embeddings.append(emb)

                logger.debug(f"Encoded batch: {len(images)} images")

            except Exception as e:
                logger.error(f"Failed to encode batch: {e}")
                # Add Nones for failed batch
                for _ in batch_paths:
                    embeddings.append(None)

        return embeddings

    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between embeddings
        
        For normalized vectors: similarity = dot product
        
        Args:
            embedding1: First embedding (normalized)
            embedding2: Second embedding (normalized)
        
        Returns:
            Similarity from -1 (opposite) to 1 (identical)
            
        Interpretation:
            0.9-1.0: Nearly identical
            0.8-0.9: Very similar (same subject, pose)
            0.7-0.8: Similar aesthetic (GOOD for cache injection)
            0.6-0.7: Related style
            < 0.6:   Different aesthetic
        """
        return float(np.dot(embedding1, embedding2))

    def find_similar(
        self,
        target_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        threshold: float = 0.7,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find similar images from candidates
        
        This is used for cache injection: find cached images that
        are similar enough to maintain aesthetic but different enough
        to add variety.
        
        Args:
            target_embedding: Target image embedding
            candidate_embeddings: List of (cache_id, embedding) tuples
            threshold: Minimum similarity threshold (0.7 recommended)
            top_k: Return top K matches
        
        Returns:
            List of (cache_id, similarity) tuples, sorted by similarity
        """
        similarities = []

        for cache_id, embedding in candidate_embeddings:
            if embedding is None:
                continue

            similarity = self.cosine_similarity(target_embedding, embedding)

            if similarity >= threshold:
                similarities.append((cache_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        result = similarities[:top_k]

        if result:
            logger.debug(
                f"Found {len(result)} similar images "
                f"(best: {result[0][1]:.3f})"
            )

        return result

    def weighted_random_selection(
        self, candidates: List[Tuple[str, float]]
    ) -> Optional[str]:
        """
        Select candidate with weighted random (higher similarity = higher probability)
        
        This adds variety while still favoring more similar images.
        
        Uses softmax-like weighting:
        - Amplify differences with exponential
        - Normalize to probabilities
        - Random selection
        
        Args:
            candidates: List of (cache_id, similarity) tuples
        
        Returns:
            Selected cache_id, or None if no candidates
        """
        if not candidates:
            return None

        # Extract cache IDs and similarities
        cache_ids = [c[0] for c in candidates]
        similarities = np.array([c[1] for c in candidates])

        # Amplify differences (softmax-like)
        weights = np.exp(similarities * 2)  # Amplify by 2x
        weights = weights / weights.sum()  # Normalize

        # Random selection with weights
        selected = np.random.choice(cache_ids, p=weights)

        logger.debug(f"Selected {selected} (weighted random)")
        return str(selected)


# Test function
def test_aesthetic_matcher() -> bool:
    """Test CLIP encoding and similarity"""
    from pathlib import Path

    print("=" * 60)
    print("Testing AestheticMatcher...")
    print("=" * 60)

    # Check for seed images
    seed_dir = Path("seeds")
    if not seed_dir.exists():
        print("✗ seeds/ directory not found")
        return False

    seed_images = list(seed_dir.glob("*.png"))
    if len(seed_images) < 2:
        print("✗ Need at least 2 seed images")
        return False

    print(f"Found {len(seed_images)} seed images")

    try:
        # Create matcher
        print("\n1. Loading CLIP model...")
        matcher = AestheticMatcher()
        print(f"✓ CLIP model loaded on {matcher.device}")

        # Test single encoding
        print("\n2. Testing single image encoding...")
        embedding1 = matcher.encode_image(seed_images[0])
        if embedding1 is not None:
            print(f"✓ Encoded: shape={embedding1.shape}")
            print(f"  Norm: {np.linalg.norm(embedding1):.3f} (should be ~1.0)")
        else:
            print("✗ Encoding failed")
            return False

        # Test self-similarity
        print("\n3. Testing self-similarity...")
        similarity = matcher.cosine_similarity(embedding1, embedding1)
        print(f"✓ Self-similarity: {similarity:.3f} (should be 1.0)")
        if not (0.99 < similarity < 1.01):
            print("✗ Self-similarity not 1.0!")
            return False

        # Test cross-similarity
        print("\n4. Testing cross-similarity...")
        embedding2 = matcher.encode_image(seed_images[1])
        if embedding2 is not None:
            similarity = matcher.cosine_similarity(embedding1, embedding2)
            print(f"✓ Cross-similarity: {similarity:.3f}")
            print("  Interpretation:")
            if similarity > 0.9:
                print("    → Nearly identical images")
            elif similarity > 0.8:
                print("    → Very similar aesthetic")
            elif similarity > 0.7:
                print("    → Similar aesthetic (good for cache injection)")
            elif similarity > 0.6:
                print("    → Related style")
            else:
                print("    → Different aesthetic")
        else:
            print("✗ Encoding failed")
            return False

        # Test batch encoding
        print("\n5. Testing batch encoding...")
        embeddings = matcher.encode_images_batch(seed_images[:4], batch_size=2)
        valid_embeddings = [e for e in embeddings if e is not None]
        print(f"✓ Batch encoded: {len(valid_embeddings)}/{len(seed_images[:4])}")

        # Test find_similar
        print("\n6. Testing similarity search...")
        candidates = [
            (f"cache_{i}", emb)
            for i, emb in enumerate(embeddings)
            if emb is not None
        ]
        similar = matcher.find_similar(
            embedding1, candidates, threshold=0.0, top_k=3
        )
        print(f"✓ Found {len(similar)} matches:")
        for cache_id, sim in similar[:3]:
            print(f"   {cache_id}: {sim:.3f}")

        # Test weighted random selection
        print("\n7. Testing weighted random selection...")
        if similar:
            selected = matcher.weighted_random_selection(similar)
            print(f"✓ Selected: {selected}")
        else:
            print("  (No candidates for selection)")

        print("\n" + "=" * 60)
        print("AestheticMatcher test PASSED ✓")
        print("=" * 60)
        print("\nNote: CLIP model downloaded (~600MB) and cached for future use")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    success = test_aesthetic_matcher()
    exit(0 if success else 1)

