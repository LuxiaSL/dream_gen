#!/usr/bin/env python3
"""
Diversity Selection Tool
═══════════════════════════════════════════════════════════════════════════════

Takes generated component candidates and selects the most diverse subset using
farthest-point sampling in CLIP embedding space.

This is the second stage of the pipeline:
  1. generate_components.py → ~200 candidates per category
  2. select_diverse_components.py → 40 most orthogonal per category (this tool)
  3. Manual review pass → final components.yaml

Algorithm: Greedy farthest-point sampling
- Start with seed components (from original YAML)
- Iteratively add the candidate that is MOST DISTANT from all selected
- This maximizes coverage of the visual-semantic space

Usage:
    # Select 40 most diverse from generated candidates
    uv run python backend/tools/select_diverse_components.py \
        --input prompts/generated_components_20260123.yaml \
        --output prompts/selected_components.yaml
    
    # Select different count
    uv run python backend/tools/select_diverse_components.py \
        --input prompts/generated_components.yaml \
        --k 50
    
    # Preview without writing
    uv run python backend/tools/select_diverse_components.py \
        --input prompts/generated_components.yaml \
        --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════


class CLIPTextEmbedder:
    """CLIP text encoder for computing visual-semantic embeddings"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info(f"Loading CLIP model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim
        self.model.eval()
    
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts, returns normalized embeddings"""
        import torch
        
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERSITY SELECTION ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════


def select_diverse_subset(
    embeddings: np.ndarray,
    words: list[str],
    k: int = 40,
    seed_indices: list[int] | None = None,
) -> list[int]:
    """
    Greedy farthest-point sampling.
    
    Returns indices of k most mutually distant embeddings.
    
    Algorithm:
    1. Start with seed_indices (or first item if none)
    2. Precompute full similarity matrix
    3. Iteratively select the point with maximum minimum distance to selected set
    
    This is O(k*n) after the O(n²) similarity precompute, but n is only ~200.
    
    Args:
        embeddings: (n, dim) array of normalized embeddings
        words: List of words corresponding to embeddings
        k: Number of items to select
        seed_indices: Optional starting indices (seeds from original YAML)
    
    Returns:
        List of selected indices
    """
    n = len(embeddings)
    
    if k >= n:
        logger.warning(f"Requested {k} items but only {n} available, returning all")
        return list(range(n))
    
    # Start with provided seeds or first item
    selected = list(seed_indices) if seed_indices else [0]
    selected_set = set(selected)
    
    # Precompute similarity matrix (embeddings assumed normalized, so dot = cosine sim)
    sim_matrix = embeddings @ embeddings.T
    
    # Track minimum similarity to selected set for each candidate
    # Initialize with similarities to seed(s)
    min_sim_to_selected = np.ones(n) * np.inf
    for seed_idx in selected:
        min_sim_to_selected = np.minimum(min_sim_to_selected, sim_matrix[:, seed_idx])
    
    while len(selected) < k:
        # Mask already-selected
        min_sim_to_selected_masked = min_sim_to_selected.copy()
        for idx in selected_set:
            min_sim_to_selected_masked[idx] = np.inf  # Already selected
        
        # Find candidate with MINIMUM similarity to selected set (= maximum distance)
        best_idx = int(np.argmin(min_sim_to_selected_masked))
        
        if best_idx in selected_set:
            logger.warning("No more candidates available")
            break
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update min similarities with new selection
        min_sim_to_selected = np.minimum(min_sim_to_selected, sim_matrix[:, best_idx])
    
    return selected


def analyze_selection(
    embeddings: np.ndarray,
    words: list[str],
    selected_indices: list[int],
    full_embeddings: np.ndarray | None = None,
    redundancy_threshold: float = 0.85,
) -> dict[str, Any]:
    """
    Analyze the diversity of selected components.
    
    Args:
        embeddings: Full embedding matrix
        words: All words
        selected_indices: Indices of selected words
        full_embeddings: Original full embeddings for before/after comparison
        redundancy_threshold: Threshold for redundant pair warning
    """
    selected_emb = embeddings[selected_indices]
    selected_words = [words[i] for i in selected_indices]
    
    # Pairwise similarities within selection
    sim_matrix = selected_emb @ selected_emb.T
    
    # Get off-diagonal similarities
    n = len(selected_indices)
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = sim_matrix[mask]
    
    # Check for redundant pairs above threshold
    redundant_count = int(np.sum(sim_matrix[mask] > redundancy_threshold) // 2)  # Divide by 2 for symmetric matrix
    if redundant_count > 0:
        logger.warning(f"  ⚠️  {redundant_count} pairs still above {redundancy_threshold} threshold")
    
    # Find most similar pair
    sim_matrix_copy = sim_matrix.copy()
    np.fill_diagonal(sim_matrix_copy, -np.inf)
    max_idx = np.unravel_index(np.argmax(sim_matrix_copy), sim_matrix_copy.shape)
    most_similar = (
        selected_words[max_idx[0]],
        selected_words[max_idx[1]],
        float(sim_matrix_copy[max_idx]),
    )
    
    # Find most distant pair
    np.fill_diagonal(sim_matrix_copy, np.inf)
    min_idx = np.unravel_index(np.argmin(sim_matrix_copy), sim_matrix_copy.shape)
    most_distant = (
        selected_words[min_idx[0]],
        selected_words[min_idx[1]],
        float(sim_matrix_copy[min_idx]),
    )
    
    analysis = {
        "count": len(selected_indices),
        "mean_similarity": float(pairwise_sims.mean()),
        "min_similarity": float(pairwise_sims.min()),
        "max_similarity": float(pairwise_sims.max()),
        "most_similar_pair": most_similar,
        "most_distant_pair": most_distant,
        "redundant_pairs_above_threshold": redundant_count,
    }
    
    # Before/after comparison if full embeddings provided
    if full_embeddings is not None:
        full_sim = full_embeddings @ full_embeddings.T
        full_n = len(full_embeddings)
        full_mask = ~np.eye(full_n, dtype=bool)
        full_pairwise = full_sim[full_mask]
        
        original_mean = float(full_pairwise.mean())
        improvement = original_mean - analysis["mean_similarity"]
        
        analysis["original_count"] = full_n
        analysis["original_mean_similarity"] = original_mean
        analysis["improvement"] = improvement
        analysis["improvement_pct"] = (improvement / original_mean) * 100 if original_mean > 0 else 0
    
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def process_category(
    category: str,
    candidates: list[str],
    seeds: list[str],
    embedder: CLIPTextEmbedder,
    k: int,
) -> tuple[list[str], dict[str, Any]]:
    """
    Process a single category: embed all candidates, select diverse subset.
    
    Args:
        category: Category name
        candidates: All candidate words (including seeds)
        seeds: Original seed words to prioritize
        embedder: CLIP embedder
        k: Number to select
    
    Returns:
        Tuple of (selected_words, analysis_dict)
    """
    logger.info(f"Processing {category}: {len(candidates)} candidates → {k} selected")
    
    # Embed all candidates
    embeddings = embedder.encode_batch(candidates)
    
    # Find seed indices
    seed_indices = []
    for seed in seeds:
        if seed in candidates:
            seed_indices.append(candidates.index(seed))
    
    if seed_indices:
        logger.info(f"  Starting with {len(seed_indices)} seeds")
    
    # Select diverse subset
    selected_indices = select_diverse_subset(
        embeddings=embeddings,
        words=candidates,
        k=k,
        seed_indices=seed_indices if seed_indices else None,
    )
    
    # Analyze selection with before/after comparison
    analysis = analyze_selection(
        embeddings=embeddings,
        words=candidates,
        selected_indices=selected_indices,
        full_embeddings=embeddings,  # Pass full for before/after comparison
    )
    
    selected_words = [candidates[i] for i in selected_indices]
    
    # Log results with improvement
    if "improvement" in analysis:
        logger.info(
            f"  Mean similarity: {analysis['original_mean_similarity']:.3f} → {analysis['mean_similarity']:.3f} "
            f"(↓{analysis['improvement']:.3f}, {analysis['improvement_pct']:.1f}% improvement)"
        )
    else:
        logger.info(f"  Mean similarity: {analysis['mean_similarity']:.3f}")
    
    logger.info(f"  Most similar: '{analysis['most_similar_pair'][0]}' <-> '{analysis['most_similar_pair'][1]}' ({analysis['most_similar_pair'][2]:.3f})")
    
    return selected_words, analysis


def main():
    parser = argparse.ArgumentParser(
        description="Select most diverse components using farthest-point sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input YAML with generated candidates",
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        help="YAML with seed components (default: prompts/components_v2.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output YAML for selected components",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        help="Number of components to select per category (default: 40)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to process (default: all)",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model for embeddings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    seeds_path = args.seeds or project_root / "prompts" / "components_v2.yaml"
    
    # Load input candidates
    with open(args.input) as f:
        input_data = yaml.safe_load(f)
    
    candidates_by_category = input_data.get("components", {})
    
    # Load seeds
    seeds_by_category: dict[str, list[str]] = {}
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds_data = yaml.safe_load(f)
        for cat, items in seeds_data.get("components", {}).items():
            seeds_by_category[cat] = [item["word"] for item in items]
    
    # Filter categories if specified
    if args.categories:
        candidates_by_category = {
            k: v for k, v in candidates_by_category.items()
            if k in args.categories
        }
    
    if not candidates_by_category:
        logger.error("No categories found in input")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("DIVERSITY SELECTION")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Categories: {list(candidates_by_category.keys())}")
    logger.info(f"Selecting: {args.k} per category")
    logger.info(f"CLIP model: {args.model}")
    
    # Initialize embedder
    try:
        embedder = CLIPTextEmbedder(model_name=args.model)
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        logger.error("Install: pip install transformers torch")
        sys.exit(1)
    
    # Process each category
    results: dict[str, list[str]] = {}
    all_analysis: dict[str, dict[str, Any]] = {}
    
    for category, candidates in candidates_by_category.items():
        seeds = seeds_by_category.get(category, [])
        
        selected, analysis = process_category(
            category=category,
            candidates=candidates,
            seeds=seeds,
            embedder=embedder,
            k=args.k,
        )
        
        results[category] = selected
        all_analysis[category] = analysis
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Category':<25} {'From':>6} {'To':>4} {'Before':>8} {'After':>8} {'Improve':>8} {'Redund':>7}")
    logger.info("-" * 75)
    
    total_redundant = 0
    for category, analysis in all_analysis.items():
        orig_count = analysis.get('original_count', '?')
        orig_sim = analysis.get('original_mean_similarity', 0)
        improvement = analysis.get('improvement', 0)
        redundant = analysis.get('redundant_pairs_above_threshold', 0)
        total_redundant += redundant
        
        logger.info(
            f"{category:<25} {orig_count:>6} {analysis['count']:>4} "
            f"{orig_sim:>8.3f} {analysis['mean_similarity']:>8.3f} "
            f"{improvement:>+8.3f} {redundant:>7}"
        )
    
    if total_redundant > 0:
        logger.warning(f"\n⚠️  Total redundant pairs (>0.85): {total_redundant}")
    
    # Output
    if args.dry_run:
        logger.info("\n[DRY RUN] Would write to: {args.output or 'auto-generated path'}")
        logger.info("\nPreview of selected components:")
        for category, words in results.items():
            logger.info(f"\n{category}:")
            for w in words[:10]:
                logger.info(f"  - {w}")
            if len(words) > 10:
                logger.info(f"  ... and {len(words) - 10} more")
    else:
        output_path = args.output or project_root / "prompts" / f"selected_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "selection_params": {
                "k": args.k,
                "model": args.model,
                "input": str(args.input),
            },
            "analysis": all_analysis,
            "components": results,
        }
        
        with open(output_path, "w") as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, width=120)
        
        logger.info(f"\nSaved to: {output_path}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()

