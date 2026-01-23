"""
Find Maximally Diverse Components for Prompt Categories

Given a large pool of candidate words, selects the N most visually-diverse
components using CLIP embeddings and max-min diversity selection.

Algorithm:
    1. Embed all candidates with CLIP
    2. Start with the two most distant candidates
    3. Iteratively add the candidate that maximizes minimum distance
       to all already-selected items
    4. This produces a set with maximum spread in embedding space

Usage:
    # Find diverse adjectives from the extended word pool
    uv run python backend/tools/find_diverse_components.py adjective --count 12
    
    # Optimize all categories and write to components.yaml
    uv run python backend/tools/find_diverse_components.py --all --write
    
    # Use custom candidates file
    uv run python backend/tools/find_diverse_components.py adjective --candidates my_words.txt
    
    # Analyze existing components for diversity
    uv run python backend/tools/find_diverse_components.py --analyze

Word pools are defined in word_sources.py - edit that file to add more candidates.
The more diverse the selected components, the more visual variety in generated images.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Import extended word pools from companion module
try:
    from word_sources import WORD_POOLS, get_word_pool, get_pool_stats
except ImportError:
    # Fallback if run from different directory
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from word_sources import WORD_POOLS, get_word_pool, get_pool_stats


def greedy_max_min_selection(
    embeddings: np.ndarray,
    words: list[str],
    n_select: int,
    seed_indices: Optional[tuple[int, int]] = None
) -> list[int]:
    """
    Greedy max-min diversity selection
    
    Selects items that maximize the minimum pairwise distance.
    This spreads selected items as far apart as possible in embedding space.
    
    Args:
        embeddings: (n_candidates, embed_dim) array
        words: List of candidate words
        n_select: Number of items to select
        seed_indices: Optional starting pair indices
        
    Returns:
        List of selected indices
    """
    n_candidates = len(embeddings)
    n_select = min(n_select, n_candidates)
    
    # Compute full similarity matrix
    sim_matrix = embeddings @ embeddings.T
    
    # Initialize with most distant pair
    if seed_indices is None:
        # Find minimum similarity (most distant pair)
        # Mask diagonal
        masked = sim_matrix + np.eye(n_candidates) * 2
        min_idx = np.argmin(masked)
        i, j = min_idx // n_candidates, min_idx % n_candidates
        selected = [i, j]
        logger.info(f"Starting with most distant pair: '{words[i]}' <-> '{words[j]}' (sim={sim_matrix[i,j]:.3f})")
    else:
        selected = list(seed_indices)
    
    # Greedily add items that maximize min distance to selected set
    while len(selected) < n_select:
        best_idx = -1
        best_min_dist = -1
        
        for idx in range(n_candidates):
            if idx in selected:
                continue
            
            # Min similarity to any selected item
            # (lower similarity = more distant = better)
            min_sim = min(sim_matrix[idx, s] for s in selected)
            
            # We want to minimize similarity (maximize distance)
            # So we want the candidate with the LOWEST max similarity
            if best_idx == -1 or min_sim < best_min_dist:
                best_idx = idx
                best_min_dist = min_sim
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        logger.debug(f"Added '{words[best_idx]}' (min_sim={best_min_dist:.3f})")
    
    return selected


def analyze_diversity(embeddings: np.ndarray, words: list[str]) -> dict:
    """Compute diversity statistics for a set of embeddings"""
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    
    # Get upper triangle (excluding diagonal)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(sim_matrix[i, j])
    
    similarities = np.array(similarities)
    
    # Find most/least similar pairs
    masked = sim_matrix + np.eye(n) * 2
    min_idx = np.argmin(masked)
    i_min, j_min = min_idx // n, min_idx % n
    
    max_idx = np.argmax(sim_matrix - np.eye(n) * 2)
    i_max, j_max = max_idx // n, max_idx % n
    
    return {
        "min": float(similarities.min()),
        "max": float(similarities.max()),
        "mean": float(similarities.mean()),
        "std": float(similarities.std()),
        "range": float(similarities.max() - similarities.min()),
        "most_similar": (words[i_max], words[j_max], float(sim_matrix[i_max, j_max])),
        "most_distant": (words[i_min], words[j_min], float(sim_matrix[i_min, j_min])),
    }


def load_clip_embedder():
    """Load CLIP embedder (reuse from compute_embeddings)"""
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
        
        model_name = "openai/clip-vit-base-patch32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading CLIP model: {model_name} on {device}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        
        def encode_batch(texts: list[str]) -> np.ndarray:
            with torch.no_grad():
                inputs = processor(text=texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                features = model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()
        
        return encode_batch
    
    except ImportError as e:
        logger.error(f"Failed to load CLIP: {e}")
        logger.error("Install: uv sync --all-extras")
        sys.exit(1)


def write_optimized_components(
    components_path: Path,
    results: dict[str, list[str]]
) -> None:
    """
    Write optimized components to components.yaml
    
    Preserves structure but replaces component words for specified categories.
    Opposites are set to None (re-run compute_embeddings.py to compute them).
    """
    # Load existing file to preserve any other data
    if components_path.exists():
        with open(components_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    
    if "components" not in data:
        data["components"] = {}
    
    # Update specified categories
    for category, words in results.items():
        data["components"][category] = [
            {"word": word, "opposite": None}
            for word in words
        ]
    
    # Write with header
    header = '''# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT POOLS - Building blocks for combinatorial prompts
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each component has:
#   - word: The text inserted into templates
#   - opposite: Semantic opposite for negative prompt generation (nullable)
#
# Embeddings are stored separately in components_embeddings.npz
# Run `uv run python backend/tools/compute_embeddings.py` to regenerate
#
# Components optimized for maximum visual diversity using CLIP embeddings.
# Edit word_sources.py to expand candidate pools for re-optimization.
# ═══════════════════════════════════════════════════════════════════════════════

'''
    
    content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    with open(components_path, 'w') as f:
        f.write(header + content)


def export_for_review(results: dict[str, list[str]], export_path: Path) -> None:
    """Export selections to a YAML file for manual review.
    
    The file can be edited to prune unwanted words, then imported with --import-file.
    """
    header = """# Selected Components for Review
# ================================
# Edit this file to prune unwanted components.
# Delete any words you don't want, keeping at least a few per category.
# Then import with:
#   uv run python backend/tools/find_diverse_components.py --import-file {path} --write
#
# Categories with online sources: adjective, mood, noun
# Categories with curated pools: color, style, effect, setting

""".format(path=export_path.name)
    
    content = yaml.dump(results, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w') as f:
        f.write(header + content)


def import_reviewed_file(import_path: Path) -> dict[str, list[str]]:
    """Import a reviewed selection file."""
    with open(import_path) as f:
        data = yaml.safe_load(f)
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"Invalid format: expected dict, got {type(data)}")
    
    results = {}
    for category, words in data.items():
        if category not in WORD_POOLS:
            logger.warning(f"Unknown category '{category}', skipping")
            continue
        if not isinstance(words, list):
            logger.warning(f"Invalid words for '{category}', skipping")
            continue
        results[category] = [str(w).strip() for w in words if w]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Find diverse components for prompt categories")
    parser.add_argument(
        "category",
        nargs="?",
        choices=list(WORD_POOLS.keys()),
        help="Category to optimize"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=12,
        help="Number of components to select (default: 12)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze diversity of current components.yaml"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="File with candidate words (one per line)"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Expand pool with online sources (adjective/mood/noun only)"
    )
    parser.add_argument(
        "--no-pos-filter",
        action="store_true",
        help="Skip part-of-speech filtering (faster but noisier results)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize all categories"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write optimized components to components.yaml"
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="Export selections to YAML file for manual review before committing"
    )
    parser.add_argument(
        "--import-file",
        type=Path,
        metavar="FILE",
        dest="import_file",
        help="Import reviewed selections from YAML and write to components.yaml"
    )
    parser.add_argument(
        "--pool-stats",
        action="store_true",
        help="Show word pool statistics"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    components_path = project_root / "prompts" / "components.yaml"
    
    # Handle import of reviewed file
    if args.import_file:
        if not args.import_file.exists():
            logger.error(f"Import file not found: {args.import_file}")
            sys.exit(1)
        
        print("=" * 70)
        print(f"IMPORTING: {args.import_file}")
        print("=" * 70)
        
        reviewed = import_reviewed_file(args.import_file)
        
        for category, words in reviewed.items():
            print(f"\n{category}: {len(words)} components")
            for word in words:
                print(f"  - {word}")
        
        if args.write:
            print("\n" + "-" * 70)
            print("Writing to components.yaml...")
            write_optimized_components(components_path, reviewed)
            print(f"\nUpdated: {components_path}")
            print("\nNext step:")
            print("  uv run python backend/tools/compute_embeddings.py --update-opposites")
        else:
            print("\n" + "-" * 70)
            print("Add --write to apply these changes to components.yaml")
        
        return
    
    if args.pool_stats:
        print("=" * 50)
        print("WORD POOL STATISTICS")
        print("=" * 50)
        stats = get_pool_stats()
        total = 0
        for category, count in sorted(stats.items()):
            print(f"  {category:<12}: {count:>4} candidates")
            total += count
        print("-" * 30)
        print(f"  {'TOTAL':<12}: {total:>4} candidates")
        print("\nEdit word_sources.py to add more candidates")
        return
    
    if args.analyze:
        # Analyze current components
        print("=" * 70)
        print("ANALYZING CURRENT COMPONENT DIVERSITY")
        print("=" * 70)
        
        embeddings_path = project_root / "prompts" / "components_embeddings.npz"
        if not embeddings_path.exists():
            logger.error(f"Embeddings not found: {embeddings_path}")
            logger.error("Run: uv run python backend/tools/compute_embeddings.py")
            sys.exit(1)
        
        data = np.load(embeddings_path)
        
        # Group by category
        categories: dict[str, tuple[list[str], list[np.ndarray]]] = {}
        for key in data.files:
            if key.startswith("_"):
                continue
            category, word = key.split("/", 1)
            if category not in categories:
                categories[category] = ([], [])
            categories[category][0].append(word)
            categories[category][1].append(data[key])
        
        # Analyze each
        results = []
        for category in sorted(categories.keys()):
            words, embs = categories[category]
            embeddings = np.array(embs)
            stats = analyze_diversity(embeddings, words)
            stats["category"] = category
            stats["count"] = len(words)
            results.append(stats)
        
        # Print sorted by range (best diversity first)
        print(f"\n{'Category':<12} {'Count':>5} {'Range':>7} {'Min':>7} {'Mean':>7} {'Max':>7}")
        print("-" * 55)
        for r in sorted(results, key=lambda x: -x["range"]):
            print(f"{r['category']:<12} {r['count']:>5} {r['range']:>7.3f} "
                  f"{r['min']:>7.3f} {r['mean']:>7.3f} {r['max']:>7.3f}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        for r in sorted(results, key=lambda x: x["range"]):
            if r["range"] < 0.15:
                print(f"\n{r['category'].upper()}: Low diversity (range={r['range']:.3f})")
                print(f"  Most similar: '{r['most_similar'][0]}' <-> '{r['most_similar'][1]}' ({r['most_similar'][2]:.3f})")
                print(f"  Consider replacing one of these with something more distinct")
        
        return
    
    # Need category for optimization
    if not args.category and not args.all:
        parser.print_help()
        print("\nSpecify a category to optimize, or use --analyze or --pool-stats")
        sys.exit(1)
    
    # Load embedder
    encode = load_clip_embedder()
    
    # Categories to process
    categories = list(WORD_POOLS.keys()) if args.all else [args.category]
    
    # Store results for potential writing
    all_results: dict[str, list[str]] = {}
    
    for category in categories:
        print("\n" + "=" * 70)
        print(f"OPTIMIZING: {category.upper()}")
        print("=" * 70)
        
        # Get candidates
        online_categories = ["adjective", "mood", "noun"]
        use_online = args.online and category in online_categories
        
        if args.candidates and args.candidates.exists():
            candidates = [line.strip() for line in args.candidates.read_text().splitlines() if line.strip()]
        elif use_online:
            from word_sources import expand_pool_online
            print(f"Fetching expanded pool from online sources...")
            candidates = expand_pool_online(
                category, 
                max_words=300,
                filter_pos=not args.no_pos_filter
            )
        else:
            candidates = get_word_pool(category)
            if args.online and category not in online_categories:
                print(f"  (online not available for {category}, using curated pool)")
        
        if len(candidates) < args.count:
            logger.warning(f"Only {len(candidates)} candidates, need {args.count}")
        
        print(f"Candidates: {len(candidates)}")
        print(f"Selecting: {args.count}")
        
        # Embed all candidates
        print("\nComputing embeddings...")
        embeddings = encode(candidates)
        
        # Analyze full pool
        full_stats = analyze_diversity(embeddings, candidates)
        print(f"\nFull pool diversity: range={full_stats['range']:.3f}, mean={full_stats['mean']:.3f}")
        
        # Select diverse subset
        print("\nSelecting diverse components...")
        selected_indices = greedy_max_min_selection(embeddings, candidates, args.count)
        selected_words = [candidates[i] for i in selected_indices]
        selected_embeddings = embeddings[selected_indices]
        
        # Store for writing
        all_results[category] = selected_words
        
        # Analyze selected
        selected_stats = analyze_diversity(selected_embeddings, selected_words)
        
        print(f"\nSelected ({len(selected_words)} items):")
        for word in selected_words:
            print(f"  - {word}")
        
        print(f"\nDiversity improvement:")
        print(f"  Range: {full_stats['range']:.3f} -> {selected_stats['range']:.3f} "
              f"({'+' if selected_stats['range'] > full_stats['range'] else ''}"
              f"{(selected_stats['range'] - full_stats['range']):.3f})")
        print(f"  Mean:  {full_stats['mean']:.3f} -> {selected_stats['mean']:.3f}")
        print(f"  Min:   {full_stats['min']:.3f} -> {selected_stats['min']:.3f}")
        
        print(f"\n  Most similar: '{selected_stats['most_similar'][0]}' <-> "
              f"'{selected_stats['most_similar'][1]}' ({selected_stats['most_similar'][2]:.3f})")
        print(f"  Most distant: '{selected_stats['most_distant'][0]}' <-> "
              f"'{selected_stats['most_distant'][1]}' ({selected_stats['most_distant'][2]:.3f})")
    
    # Export for review if requested
    if args.export and all_results:
        print("\n" + "=" * 70)
        print(f"EXPORTING TO: {args.export}")
        print("=" * 70)
        
        export_for_review(all_results, args.export)
        
        print(f"\nExported {sum(len(v) for v in all_results.values())} components across {len(all_results)} categories")
        print("\nNext steps:")
        print(f"  1. Review and prune: {args.export}")
        print(f"  2. Apply with: uv run python backend/tools/find_diverse_components.py --import {args.export} --write")
        print("  3. Generate embeddings: uv run python backend/tools/compute_embeddings.py --update-opposites")
    
    # Write results if requested
    elif args.write and all_results:
        print("\n" + "=" * 70)
        print("WRITING TO components.yaml")
        print("=" * 70)
        
        write_optimized_components(components_path, all_results)
        
        print(f"\nUpdated: {components_path}")
        print("\nNext steps:")
        print("  1. Review the changes in components.yaml")
        print("  2. Run: uv run python backend/tools/compute_embeddings.py --update-opposites")
    elif all_results:
        print("\n" + "=" * 70)
        print("To apply these selections:")
        print("  --export FILE  : Export for manual review first")
        print("  --write        : Write directly to components.yaml")
        print("  Then run: uv run python backend/tools/compute_embeddings.py --update-opposites")
        print("=" * 70)


if __name__ == "__main__":
    main()

