"""
Compute CLIP Text Embeddings for Prompt Components

Generates semantic embeddings for all prompt components using CLIP's text encoder.
CLIP embeddings understand visual concepts - they capture how words relate in
image-generation space, not just linguistic similarity.

This enables similarity-guided component mutation: when switching from "ethereal"
to a new adjective, prefer components that are visually related but distinct.

Architecture:
    - components.yaml: Human-editable file with word + opposite (no embeddings)
    - components_embeddings.npz: Auto-generated binary file with embeddings
    
    The combinatorial prompt system loads both and merges them at runtime.

Usage:
    # Compute embeddings (writes to components_embeddings.npz)
    uv run python backend/tools/compute_embeddings.py
    
    # Dry run (preview without writing)
    uv run python backend/tools/compute_embeddings.py --dry-run
    
    # Use a different model
    uv run python backend/tools/compute_embeddings.py --model openai/clip-vit-large-patch14
    
    # Also update opposites in components.yaml
    uv run python backend/tools/compute_embeddings.py --update-opposites

Output:
    - prompts/components_embeddings.npz: Binary file with embeddings
      Structure: {
          "adjective/ethereal": array([...]),
          "adjective/digital": array([...]),
          ...
          "_metadata": {"model": "...", "dim": 512, "timestamp": "..."}
      }
    
    - Optionally updates prompts/components.yaml with computed opposites

Dependencies:
    - transformers (for CLIP)
    - torch
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class CLIPTextEmbedder:
    """
    CLIP text encoder for computing visual-semantic embeddings
    
    Uses CLIP's text encoder to embed words/phrases into a space where
    visually similar concepts are close together.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP text encoder
        
        Args:
            model_name: HuggingFace model identifier for CLIP
        """
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info(f"Loading CLIP model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.projection_dim
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding
        
        Args:
            text: Text string to encode
            
        Returns:
            Normalized embedding vector (512-dim for base model)
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text features from CLIP
        text_features = self.model.get_text_features(**inputs)
        
        # Normalize to unit vector
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts in a single batch
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (n_texts, embedding_dim), normalized
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (assumes normalized)"""
    return float(np.dot(a, b))


def find_semantic_opposite(
    word: str,
    embedding: np.ndarray,
    pool: list[dict[str, Any]],
    pool_embeddings: np.ndarray
) -> str | None:
    """
    Find the most semantically distant component in the pool
    
    For negative prompt generation, we want the opposite of a component.
    This finds the component with lowest cosine similarity.
    
    Args:
        word: Current component word
        embedding: Current component embedding
        pool: List of component dicts in the category
        pool_embeddings: Array of embeddings for all components in pool
        
    Returns:
        Word of most distant component, or None if pool too small
    """
    if len(pool) < 2:
        return None
    
    # Compute similarities to all others
    similarities = pool_embeddings @ embedding
    
    # Find minimum similarity (most distant), excluding self
    min_sim = float('inf')
    opposite_word = None
    
    for i, item in enumerate(pool):
        if item['word'] == word:
            continue
        if similarities[i] < min_sim:
            min_sim = similarities[i]
            opposite_word = item['word']
    
    return opposite_word


def compute_all_embeddings(
    components_path: Path,
    embedder: CLIPTextEmbedder,
    compute_opposites: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    """
    Compute embeddings for all components
    
    Args:
        components_path: Path to components.yaml
        embedder: CLIPTextEmbedder instance
        compute_opposites: Whether to compute semantic opposites
        
    Returns:
        Tuple of:
        - embeddings_dict: {category/word: embedding_array}
        - components_data: Original YAML data (for potential opposite updates)
        - opposites_dict: {category/word: opposite_word} for negatable categories
    """
    # Load existing components
    with open(components_path) as f:
        data = yaml.safe_load(f)
    
    components = data.get('components', {})
    
    # Categories that get semantic opposites for negative prompts
    # Updated for v2 category names
    negatable_categories = {
        'color_logic', 'light_behavior', 'atmosphere_field', 
        'temporal_state', 'texture_density', 'medium_render'
    }
    
    embeddings_dict: dict[str, np.ndarray] = {}
    opposites_dict: dict[str, str] = {}
    
    # Process each category
    for category, items in components.items():
        logger.info(f"Processing category: {category} ({len(items)} items)")
        
        # Handle both formats:
        # - Old: list of dicts with 'word' key
        # - New: simple list of strings
        if items and isinstance(items[0], dict):
            words = [item['word'] for item in items]
        else:
            words = items if items else []
        
        if not words:
            logger.warning(f"  Skipping empty category: {category}")
            continue
        
        embeddings = embedder.encode_batch(words)
        
        # Store embeddings with category/word keys
        for i, word in enumerate(words):
            key = f"{category}/{word}"
            embeddings_dict[key] = embeddings[i]
        
        # Compute opposites for negatable categories using greedy pairing
        if compute_opposites and category in negatable_categories:
            logger.info(f"  Computing semantic opposite pairs for {category}")
            
            category_opposites = compute_greedy_opposite_pairs(words, embeddings)
            
            for word, opposite in category_opposites.items():
                key = f"{category}/{word}"
                opposites_dict[key] = opposite
            
            # Log pairing stats
            unique_opposites = len(set(category_opposites.values()))
            logger.info(f"    Created {len(category_opposites)} pairs with {unique_opposites} unique opposites")
    
    return embeddings_dict, data, opposites_dict


def compute_greedy_opposite_pairs(
    words: list[str],
    embeddings: np.ndarray
) -> dict[str, str]:
    """
    Compute semantic opposites using greedy maximum-distance pairing.
    
    Algorithm:
    1. Compute full pairwise similarity matrix
    2. Find the most distant pair (lowest similarity)
    3. Assign them as mutual opposites
    4. Remove both from the pool
    5. Repeat until pool is exhausted
    6. If odd count, last word gets paired with most distant remaining (already paired) word
    
    This ensures each word has a unique opposite (no duplicate assignments).
    
    Args:
        words: List of component words
        embeddings: Corresponding embeddings array (normalized)
        
    Returns:
        Dict mapping each word to its opposite
    """
    n = len(words)
    if n < 2:
        return {}
    
    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.T
    
    # Track which indices are still available for pairing
    available = set(range(n))
    opposites: dict[str, str] = {}
    
    while len(available) >= 2:
        # Find most distant pair among available indices
        min_sim = float('inf')
        best_pair = None
        
        available_list = list(available)
        for idx_a, i in enumerate(available_list):
            for j in available_list[idx_a + 1:]:
                if sim_matrix[i, j] < min_sim:
                    min_sim = sim_matrix[i, j]
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        # Assign mutual opposites
        opposites[words[i]] = words[j]
        opposites[words[j]] = words[i]
        
        # Remove both from pool
        available.remove(i)
        available.remove(j)
    
    # Handle odd count: last word pairs with most distant already-paired word
    if len(available) == 1:
        last_idx = available.pop()
        last_word = words[last_idx]
        
        # Find most distant among all OTHER words
        min_sim = float('inf')
        best_opposite = None
        
        for i in range(n):
            if i == last_idx:
                continue
            if sim_matrix[last_idx, i] < min_sim:
                min_sim = sim_matrix[last_idx, i]
                best_opposite = words[i]
        
        if best_opposite:
            opposites[last_word] = best_opposite
    
    return opposites


def find_semantic_opposite_simple(
    word: str,
    embedding: np.ndarray,
    words: list[str],
    pool_embeddings: np.ndarray
) -> str | None:
    """
    DEPRECATED: Use compute_greedy_opposite_pairs instead.
    
    Find the most semantically distant component in the pool (simple list version)
    """
    if len(words) < 2:
        return None
    
    # Compute similarities to all others
    similarities = pool_embeddings @ embedding
    
    # Find minimum similarity (most distant), excluding self
    min_sim = float('inf')
    opposite_word = None
    
    for i, w in enumerate(words):
        if w == word:
            continue
        if similarities[i] < min_sim:
            min_sim = similarities[i]
            opposite_word = w
    
    return opposite_word


def save_embeddings_npz(
    embeddings_dict: dict[str, np.ndarray],
    output_path: Path,
    model_name: str
) -> None:
    """
    Save embeddings to compressed numpy archive
    
    Format:
        - Keys are "category/word" (e.g., "adjective/ethereal")
        - Special key "_metadata_json" contains model info
    """
    # Prepare metadata
    metadata = {
        "model": model_name,
        "dim": len(next(iter(embeddings_dict.values()))),
        "count": len(embeddings_dict),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Convert to saveable format
    save_dict = {k: v for k, v in embeddings_dict.items()}
    save_dict["_metadata_json"] = np.array(json.dumps(metadata))
    
    np.savez_compressed(output_path, **save_dict)
    
    # Report file size
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved {len(embeddings_dict)} embeddings to {output_path} ({size_kb:.1f} KB)")


def update_yaml_opposites(
    components_path: Path,
    opposites_dict: dict[str, str]
) -> None:
    """
    Update components.yaml with computed opposites
    
    Converts simple list format to dict format with word + opposite
    """
    with open(components_path) as f:
        data = yaml.safe_load(f)
    
    components = data.get('components', {})
    
    # Convert to new format with opposites
    new_components = {}
    for category, items in components.items():
        new_items = []
        
        # Handle both formats
        if items and isinstance(items[0], dict):
            words = [item['word'] for item in items]
        else:
            words = items if items else []
        
        for word in words:
            key = f"{category}/{word}"
            item_dict = {'word': word}
            if key in opposites_dict:
                item_dict['opposite'] = opposites_dict[key]
            new_items.append(item_dict)
        
        new_components[category] = new_items
    
    data['components'] = new_components
    
    # Write back with clean formatting
    output = generate_clean_yaml(data)
    with open(components_path, 'w') as f:
        f.write(output)
    
    logger.info(f"Updated opposites in {components_path}")


def generate_clean_yaml(data: dict[str, Any]) -> str:
    """
    Generate clean YAML without inline embeddings
    """
    header = '''# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT POOLS v2 - Building blocks for combinatorial prompts
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each component has:
#   - word: The text inserted into templates
#   - opposite: Semantic opposite for negative prompt generation (nullable)
#
# Embeddings are stored separately in components_embeddings.npz
# Run `uv run python backend/tools/compute_embeddings.py` to regenerate
#
# 12 Visual Axis Categories:
#   subject_form       - Primary visual entity (archetypal forms)
#   material_substance - What it's made of (concrete materials)
#   texture_density    - Surface quality (tactile descriptors)
#   light_behavior     - How light interacts (specific phenomena)
#   color_logic        - Palette relationships (not single colors)
#   atmosphere_field   - Environmental media (fog, particles, fields)
#   phenomenon_pattern - Visual processes (organic, digital, mathematical)
#   spatial_logic      - Compositional arrangement (geometry, flow)
#   scale_perspective  - Viewing position and magnification
#   temporal_state     - Moment in process (decay, growth, transformation)
#   setting_location   - Environmental context (specific places)
#   medium_render      - Artistic technique / imaging method
#
# Categories with semantic opposites for negative prompts:
#   color_logic, light_behavior, atmosphere_field, temporal_state, 
#   texture_density, medium_render
# ═══════════════════════════════════════════════════════════════════════════════

'''
    
    content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return header + content


def print_similarity_analysis(
    embeddings_dict: dict[str, np.ndarray],
    opposites_dict: dict[str, str]
) -> None:
    """Print analysis of computed embeddings"""
    
    print("\n" + "=" * 70)
    print("EMBEDDING ANALYSIS")
    print("=" * 70)
    
    # Group by category
    categories: dict[str, list[tuple[str, np.ndarray]]] = {}
    for key, emb in embeddings_dict.items():
        if key.startswith("_"):
            continue
        category, word = key.split("/", 1)
        if category not in categories:
            categories[category] = []
        categories[category].append((word, emb))
    
    for category, items in sorted(categories.items()):
        words = [w for w, _ in items]
        embeddings = np.array([e for _, e in items])
        
        # Compute pairwise similarities
        similarity_matrix = embeddings @ embeddings.T
        
        # Get statistics (excluding diagonal)
        n = len(items)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])
        
        similarities = np.array(similarities)
        
        print(f"\n{category.upper()} ({n} items)")
        print(f"  Similarity: min={similarities.min():.3f}, "
              f"mean={similarities.mean():.3f}, max={similarities.max():.3f}")
        
        # Find most similar pair
        max_idx = np.argmax(similarity_matrix - np.eye(n) * 2)
        i, j = max_idx // n, max_idx % n
        print(f"  Most similar: '{words[i]}' <-> '{words[j]}' ({similarity_matrix[i, j]:.3f})")
        
        # Find most distant pair
        min_idx = np.argmin(similarity_matrix + np.eye(n) * 2)
        i, j = min_idx // n, min_idx % n
        print(f"  Most distant: '{words[i]}' <-> '{words[j]}' ({similarity_matrix[i, j]:.3f})")
        
        # Show opposites for this category
        category_opposites = [
            (w, opposites_dict.get(f"{category}/{w}"))
            for w in words
            if f"{category}/{w}" in opposites_dict
        ]
        if category_opposites:
            print(f"  Opposites:")
            for word, opp in category_opposites[:3]:
                print(f"    '{word}' -> '{opp}'")
            if len(category_opposites) > 3:
                print(f"    ... and {len(category_opposites) - 3} more")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP embeddings for prompt components"
    )
    parser.add_argument(
        '--model',
        default='openai/clip-vit-base-patch32',
        help='CLIP model to use (default: openai/clip-vit-base-patch32)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing to file'
    )
    parser.add_argument(
        '--no-opposites',
        action='store_true',
        help='Skip computing semantic opposites'
    )
    parser.add_argument(
        '--update-opposites',
        action='store_true',
        help='Also update opposites in components.yaml'
    )
    parser.add_argument(
        '--components',
        type=Path,
        help='Path to components.yaml (default: prompts/components.yaml)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Path for embeddings output (default: prompts/components_embeddings.npz)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    components_path = args.components or project_root / "prompts" / "components.yaml"
    embeddings_path = args.output or project_root / "prompts" / "components_embeddings.npz"
    
    if not components_path.exists():
        logger.error(f"Components file not found: {components_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("CLIP Text Embedding Computation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Components: {components_path}")
    print(f"Embeddings output: {embeddings_path}")
    print(f"Compute opposites: {not args.no_opposites}")
    print(f"Update YAML opposites: {args.update_opposites}")
    print(f"Dry run: {args.dry_run}")
    
    # Initialize embedder
    try:
        embedder = CLIPTextEmbedder(model_name=args.model)
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        logger.error("Install transformers: pip install transformers")
        sys.exit(1)
    
    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings_dict, components_data, opposites_dict = compute_all_embeddings(
        components_path=components_path,
        embedder=embedder,
        compute_opposites=not args.no_opposites
    )
    
    # Print analysis
    print_similarity_analysis(embeddings_dict, opposites_dict)
    
    # Write output
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Not writing files")
        print("=" * 70)
        print(f"\nWould write {len(embeddings_dict)} embeddings to: {embeddings_path}")
        print(f"Embedding dimension: {len(next(iter(embeddings_dict.values())))}")
        
        # Preview a few
        print("\nPreview (first 3 keys):")
        for i, (key, emb) in enumerate(embeddings_dict.items()):
            if i >= 3:
                break
            print(f"  {key}: [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")
        
        if opposites_dict:
            print(f"\nComputed {len(opposites_dict)} opposites")
    else:
        # Save embeddings to npz
        save_embeddings_npz(embeddings_dict, embeddings_path, args.model)
        
        # Optionally update YAML with opposites (and remove inline embeddings)
        if args.update_opposites:
            update_yaml_opposites(components_path, opposites_dict)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

