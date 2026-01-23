#!/usr/bin/env python3
"""
Component Quality Analysis
═══════════════════════════════════════════════════════════════════════════════

Analyzes generated components before diversity selection to identify:
1. Intra-category similarity distributions (mean, min, max)
2. Cross-category contamination (terms closer to wrong category)
3. Redundancy pairs (similarity > threshold)

Usage:
    uv run python backend/tools/analyze_component_quality.py \
        --input prompts/generation_checkpoint.yaml
    
    # Custom redundancy threshold
    uv run python backend/tools/analyze_component_quality.py \
        --input prompts/generation_checkpoint.yaml \
        --redundancy-threshold 0.90
"""

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
    
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple texts in batches, returns normalized embeddings"""
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(all_embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryStats:
    """Statistics for a single category"""
    name: str
    count: int
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    std_similarity: float


@dataclass 
class RedundantPair:
    """A pair of terms with high similarity"""
    word1: str
    word2: str
    similarity: float


@dataclass
class ContaminatedTerm:
    """A term that embeds closer to a different category"""
    word: str
    assigned_category: str
    closest_category: str
    similarity_to_assigned: float
    similarity_to_closest: float


def compute_intra_category_stats(
    words: list[str],
    embeddings: np.ndarray,
) -> CategoryStats:
    """Compute similarity statistics within a category"""
    n = len(words)
    
    if n < 2:
        return CategoryStats(
            name="",
            count=n,
            mean_similarity=0.0,
            min_similarity=0.0,
            max_similarity=0.0,
            std_similarity=0.0,
        )
    
    # Compute pairwise similarities
    sim_matrix = embeddings @ embeddings.T
    
    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    
    return CategoryStats(
        name="",
        count=n,
        mean_similarity=float(pairwise_sims.mean()),
        min_similarity=float(pairwise_sims.min()),
        max_similarity=float(pairwise_sims.max()),
        std_similarity=float(pairwise_sims.std()),
    )


def find_redundant_pairs(
    words: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> list[RedundantPair]:
    """Find pairs of terms with similarity above threshold"""
    n = len(words)
    redundant = []
    
    # Compute pairwise similarities
    sim_matrix = embeddings @ embeddings.T
    
    # Find pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                redundant.append(RedundantPair(
                    word1=words[i],
                    word2=words[j],
                    similarity=float(sim_matrix[i, j]),
                ))
    
    # Sort by similarity descending
    redundant.sort(key=lambda x: x.similarity, reverse=True)
    
    return redundant


def check_cross_category_contamination(
    categories: dict[str, tuple[list[str], np.ndarray]],
) -> list[ContaminatedTerm]:
    """
    Find terms that embed closer to a different category's centroid
    than their assigned category.
    """
    # Compute category centroids
    centroids = {}
    for cat_name, (words, embeddings) in categories.items():
        centroids[cat_name] = embeddings.mean(axis=0)
        # Normalize centroid
        centroids[cat_name] = centroids[cat_name] / np.linalg.norm(centroids[cat_name])
    
    cat_names = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[c] for c in cat_names])
    
    contaminated = []
    
    for assigned_cat, (words, embeddings) in categories.items():
        assigned_idx = cat_names.index(assigned_cat)
        
        # Compute similarities to all category centroids
        sims_to_centroids = embeddings @ centroid_matrix.T  # (n_words, n_categories)
        
        for i, word in enumerate(words):
            sim_to_assigned = sims_to_centroids[i, assigned_idx]
            
            # Find closest category
            closest_idx = np.argmax(sims_to_centroids[i])
            closest_cat = cat_names[closest_idx]
            sim_to_closest = sims_to_centroids[i, closest_idx]
            
            # Check if closer to different category
            if closest_cat != assigned_cat:
                contaminated.append(ContaminatedTerm(
                    word=word,
                    assigned_category=assigned_cat,
                    closest_category=closest_cat,
                    similarity_to_assigned=float(sim_to_assigned),
                    similarity_to_closest=float(sim_to_closest),
                ))
    
    # Sort by contamination severity (difference in similarities)
    contaminated.sort(
        key=lambda x: x.similarity_to_closest - x.similarity_to_assigned,
        reverse=True,
    )
    
    return contaminated


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Analyze component quality before diversity selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input YAML with generated components",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for redundancy detection (default: 0.85)",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model for embeddings",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Optional: save detailed report to YAML",
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
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load components
    with open(args.input) as f:
        data = yaml.safe_load(f)
    
    components_by_category = data.get("components", {})
    
    print("=" * 70)
    print("COMPONENT QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Categories: {len(components_by_category)}")
    print(f"Redundancy threshold: {args.redundancy_threshold}")
    print()
    
    # Initialize embedder
    try:
        embedder = CLIPTextEmbedder(model_name=args.model)
    except Exception as e:
        logger.error(f"Failed to load CLIP: {e}")
        sys.exit(1)
    
    # Embed all categories
    print("\nEmbedding all components...")
    categories_with_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    
    for cat_name, words in components_by_category.items():
        print(f"  {cat_name}: {len(words)} words")
        embeddings = embedder.encode_batch(words)
        categories_with_embeddings[cat_name] = (words, embeddings)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. INTRA-CATEGORY SIMILARITY DISTRIBUTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("1. INTRA-CATEGORY SIMILARITY DISTRIBUTIONS")
    print("=" * 70)
    print(f"{'Category':<25} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    
    all_stats = {}
    for cat_name, (words, embeddings) in categories_with_embeddings.items():
        stats = compute_intra_category_stats(words, embeddings)
        stats.name = cat_name
        all_stats[cat_name] = stats
        
        print(f"{cat_name:<25} {stats.count:>6} {stats.mean_similarity:>8.3f} "
              f"{stats.std_similarity:>8.3f} {stats.min_similarity:>8.3f} {stats.max_similarity:>8.3f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. REDUNDANT PAIRS (similarity > threshold)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print(f"2. REDUNDANT PAIRS (similarity > {args.redundancy_threshold})")
    print("=" * 70)
    
    all_redundant: dict[str, list[RedundantPair]] = {}
    total_redundant = 0
    
    for cat_name, (words, embeddings) in categories_with_embeddings.items():
        redundant = find_redundant_pairs(words, embeddings, args.redundancy_threshold)
        all_redundant[cat_name] = redundant
        total_redundant += len(redundant)
        
        if redundant:
            print(f"\n{cat_name} ({len(redundant)} pairs):")
            for pair in redundant[:10]:  # Show top 10
                print(f"  {pair.similarity:.3f}: '{pair.word1}' ↔ '{pair.word2}'")
            if len(redundant) > 10:
                print(f"  ... and {len(redundant) - 10} more pairs")
    
    if total_redundant == 0:
        print("\nNo redundant pairs found above threshold!")
    else:
        print(f"\nTotal redundant pairs: {total_redundant}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. CROSS-CATEGORY CONTAMINATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("3. CROSS-CATEGORY CONTAMINATION")
    print("=" * 70)
    print("Terms that embed closer to a different category's centroid:\n")
    
    contaminated = check_cross_category_contamination(categories_with_embeddings)
    
    if not contaminated:
        print("No contaminated terms found!")
    else:
        # Group by assigned category
        by_category: dict[str, list[ContaminatedTerm]] = defaultdict(list)
        for term in contaminated:
            by_category[term.assigned_category].append(term)
        
        print(f"{'Category':<25} {'Contaminated':>12} {'% of Total':>12}")
        print("-" * 50)
        
        for cat_name in sorted(by_category.keys()):
            terms = by_category[cat_name]
            total = len(categories_with_embeddings[cat_name][0])
            pct = len(terms) / total * 100
            print(f"{cat_name:<25} {len(terms):>12} {pct:>11.1f}%")
        
        print(f"\nTotal contaminated: {len(contaminated)}")
        
        # Show worst offenders
        print(f"\nTop 20 most contaminated terms:")
        print(f"{'Word':<30} {'Assigned':<20} {'→ Closest':<20} {'Δ Sim':>8}")
        print("-" * 80)
        
        for term in contaminated[:20]:
            delta = term.similarity_to_closest - term.similarity_to_assigned
            print(f"{term.word:<30} {term.assigned_category:<20} "
                  f"→ {term.closest_category:<20} {delta:>+.3f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find categories with issues
    high_similarity_cats = [
        (name, stats) for name, stats in all_stats.items()
        if stats.mean_similarity > 0.50
    ]
    
    if high_similarity_cats:
        print("\n⚠️  Categories with high mean similarity (>0.50):")
        for name, stats in sorted(high_similarity_cats, key=lambda x: -x[1].mean_similarity):
            print(f"  - {name}: {stats.mean_similarity:.3f}")
    
    cats_with_redundancy = [
        (name, pairs) for name, pairs in all_redundant.items()
        if pairs
    ]
    
    if cats_with_redundancy:
        print(f"\n⚠️  Categories with redundant pairs (>{args.redundancy_threshold}):")
        for name, pairs in sorted(cats_with_redundancy, key=lambda x: -len(x[1])):
            print(f"  - {name}: {len(pairs)} pairs")
    
    if contaminated:
        worst_cats = sorted(
            by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:5]
        print(f"\n⚠️  Categories with most contamination:")
        for name, terms in worst_cats:
            print(f"  - {name}: {len(terms)} terms")
    
    print("\n✅ Analysis complete!")
    
    # Optional: save detailed report
    if args.output:
        report = {
            "input": str(args.input),
            "redundancy_threshold": args.redundancy_threshold,
            "intra_category_stats": {
                name: {
                    "count": s.count,
                    "mean": s.mean_similarity,
                    "std": s.std_similarity,
                    "min": s.min_similarity,
                    "max": s.max_similarity,
                }
                for name, s in all_stats.items()
            },
            "redundant_pairs": {
                name: [
                    {"word1": p.word1, "word2": p.word2, "similarity": p.similarity}
                    for p in pairs
                ]
                for name, pairs in all_redundant.items()
                if pairs
            },
            "contaminated_terms": [
                {
                    "word": t.word,
                    "assigned": t.assigned_category,
                    "closest": t.closest_category,
                    "sim_assigned": t.similarity_to_assigned,
                    "sim_closest": t.similarity_to_closest,
                }
                for t in contaminated
            ],
        }
        
        with open(args.output, "w") as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()

