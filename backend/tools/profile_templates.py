#!/usr/bin/env python3
"""Profile templates for diversity potential.

Analyzes each template's diversity by:
1. Internal diversity — variety within a single template
2. Cross-template uniqueness — how distinct from other templates
3. Uses stable component selection for fair comparison
4. Computes embeddings once for efficiency
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import numpy as np
from typing import Callable
import re
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from find_diverse_components import load_clip_embedder


def encode_batched(
    encode_fn: Callable[[list[str]], np.ndarray],
    prompts: list[str],
    batch_size: int = 1000,
    desc: str = "Encoding"
) -> np.ndarray:
    """Encode prompts in batches to avoid OOM errors.
    
    Args:
        encode_fn: The base encoding function
        prompts: List of prompts to encode
        batch_size: Number of prompts per batch
        desc: Description for progress display
    
    Returns:
        Stacked numpy array of all embeddings
    """
    if len(prompts) <= batch_size:
        return encode_fn(prompts)
    
    all_embeddings = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  {desc}: batch {batch_num}/{n_batches} ({len(batch)} prompts)...", end=" ", flush=True)
        embeddings = encode_fn(batch)
        all_embeddings.append(embeddings)
        print("done")
    
    return np.vstack(all_embeddings)


@dataclass
class TemplateProfile:
    """Statistical profile of a template's diversity potential."""
    name: str
    structure: str
    n_samples: int
    
    # Internal diversity (within template)
    internal_mean: float = 0.0
    internal_min: float = 0.0
    internal_max: float = 0.0
    internal_range: float = 0.0
    
    # Cross-template uniqueness (vs other templates)
    cross_mean: float = 0.0  # mean similarity to other template prompts
    cross_min: float = 0.0
    cross_max: float = 0.0
    
    # Structural info
    n_slots: int = 0
    slot_categories: list[str] = field(default_factory=list)
    multi_component_slots: int = 0
    theoretical_combinations: int = 0
    
    def internal_score(self) -> float:
        """Internal diversity: lower mean + higher range = better."""
        return (1 - self.internal_mean) * 0.6 + self.internal_range * 0.4
    
    def uniqueness_score(self) -> float:
        """Cross-template uniqueness: lower mean = more distinct."""
        return 1 - self.cross_mean
    
    def combined_score(self) -> float:
        """Combined: internal diversity + uniqueness."""
        return self.internal_score() * 0.5 + self.uniqueness_score() * 0.5


def load_templates(path: Path) -> dict[str, dict]:
    """Load templates from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    templates_list = data.get("templates", [])
    return {t["id"]: t for t in templates_list if "id" in t}


def load_components(path: Path) -> dict[str, list[str]]:
    """Load components from YAML.
    
    Handles both formats:
    - Old: components.yaml with list of {word: "...", ...} dicts
    - New: selected_components.yaml with simple list of strings
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    
    components = {}
    for category, items in data.get("components", {}).items():
        if isinstance(items, list):
            if items and isinstance(items[0], dict):
                # Old format: list of dicts with "word" key
                components[category] = [item["word"] for item in items]
            else:
                # New format: simple list of strings
                components[category] = items
        else:
            components[category] = []
    return components


def parse_template_slots(structure: str) -> list[tuple[str, int, str]]:
    """Parse template slots: returns [(category, count, separator), ...]"""
    slots = []
    for match in re.finditer(r'\{(\w+)(?::(\d+))?(?::([^}]*))?\}', structure):
        category = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        sep = match.group(3) if match.group(3) else " "
        slots.append((category, count, sep))
    return slots


def compute_theoretical_combinations(
    slots: list[tuple[str, int, str]], 
    components: dict[str, list[str]]
) -> int:
    """Compute total possible combinations for a template."""
    total = 1
    for category, count, _ in slots:
        if category not in components:
            continue
        n = len(components[category])
        if count == 1:
            total *= n
        else:
            from math import comb
            total *= comb(n, min(count, n))
    return total


class StableComponentSelector:
    """Selects components in a deterministic, stable way across templates.
    
    Uses round-robin through sorted component lists, ensuring the same
    component selections are used for fair cross-template comparison.
    """
    
    def __init__(self, components: dict[str, list[str]], seed: int = 42):
        self.components = {k: sorted(v) for k, v in components.items()}
        self.positions = defaultdict(int)
        self.seed = seed
        self._rng = None
    
    def reset(self):
        """Reset positions for a new run."""
        self.positions = defaultdict(int)
        import random
        self._rng = random.Random(self.seed)
    
    def select(self, category: str, count: int = 1) -> list[str]:
        """Select components using stable round-robin."""
        if category not in self.components:
            return []
        
        words = self.components[category]
        if not words:
            return []
        
        selected = []
        for _ in range(count):
            pos = self.positions[category] % len(words)
            selected.append(words[pos])
            self.positions[category] += 1
        
        return selected


def generate_prompts_stable(
    template: dict,
    selector: StableComponentSelector,
    n_prompts: int
) -> list[str]:
    """Generate prompts with stable component selection.
    
    Tracks used words per-category within each prompt to avoid
    duplicates when the same category appears in multiple slots.
    """
    structure = template.get("structure", "")
    slots = parse_template_slots(structure)
    
    prompts = []
    for _ in range(n_prompts):
        prompt = structure
        # Track used words per category within this prompt
        used_in_prompt: dict[str, set[str]] = defaultdict(set)
        
        for category, count, sep in slots:
            # Get candidates, excluding already-used words in this prompt
            selected = []
            attempts = 0
            max_attempts = count * 3  # Prevent infinite loop
            
            while len(selected) < count and attempts < max_attempts:
                candidates = selector.select(category, 1)
                if not candidates:
                    break
                word = candidates[0]
                if word not in used_in_prompt[category]:
                    selected.append(word)
                    used_in_prompt[category].add(word)
                attempts += 1
            
            if not selected:
                continue
            
            replacement = sep.join(selected)
            
            if count > 1:
                if sep != " ":
                    pattern = f"{{{category}:{count}:{sep}}}"
                else:
                    pattern = f"{{{category}:{count}}}"
            else:
                pattern = f"{{{category}}}"
            
            prompt = prompt.replace(pattern, replacement, 1)
        
        prompts.append(prompt)
    
    return prompts


def compute_similarity_stats(embeddings: np.ndarray) -> dict:
    """Compute similarity statistics for a set of embeddings."""
    if len(embeddings) < 2:
        return {"mean": 0, "min": 0, "max": 0, "range": 0}
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    # Compute pairwise similarities
    sim_matrix = embeddings_norm @ embeddings_norm.T
    
    # Get upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(len(embeddings), k=1)
    similarities = sim_matrix[triu_indices]
    
    return {
        "mean": float(np.mean(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "range": float(np.max(similarities) - np.min(similarities))
    }


def compute_cross_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> dict:
    """Compute similarity between two sets of embeddings."""
    # Normalize both
    norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
    
    # Cross similarity matrix
    sim_matrix = norm_a @ norm_b.T
    
    return {
        "mean": float(np.mean(sim_matrix)),
        "min": float(np.min(sim_matrix)),
        "max": float(np.max(sim_matrix))
    }


def profile_all_templates(
    templates: dict[str, dict],
    components: dict[str, list[str]],
    encode: Callable[[list[str]], np.ndarray],
    n_samples: int = 500,
    seed: int = 42,
    use_random: bool = False,
    batch_size: int = 1000
) -> tuple[list[TemplateProfile], dict[str, list[str]], dict[str, np.ndarray]]:
    """Profile all templates with single embedding computation.
    
    Returns:
        - profiles: List of TemplateProfile objects
        - all_prompts: Dict of template_name -> list of prompts
        - template_embeddings: Dict of template_name -> embeddings array
    """
    selection_mode = "random" if use_random else "stable round-robin"
    
    # Phase 1: Generate all prompts
    print(f"\n[1/3] Generating prompts ({selection_mode} selection)...")
    all_prompts: dict[str, list[str]] = {}
    
    if use_random:
        for name, template in templates.items():
            prompts = generate_prompts_random(template, components, n_samples, seed=seed)
            all_prompts[name] = prompts
            print(f"  {name}: {len(prompts)} prompts")
    else:
        selector = StableComponentSelector(components, seed=seed)
        for name, template in templates.items():
            selector.reset()  # Same starting point for each template
            prompts = generate_prompts_stable(template, selector, n_samples)
            all_prompts[name] = prompts
            print(f"  {name}: {len(prompts)} prompts")
    
    # Phase 2: Compute all embeddings in batches
    print("\n[2/3] Computing embeddings (batched)...")
    flat_prompts = []
    prompt_indices = {}  # template_name -> (start, end) indices
    
    idx = 0
    for name, prompts in all_prompts.items():
        prompt_indices[name] = (idx, idx + len(prompts))
        flat_prompts.extend(prompts)
        idx += len(prompts)
    
    print(f"  Total prompts: {len(flat_prompts)}")
    all_embeddings = encode_batched(encode, flat_prompts, batch_size=batch_size)
    print(f"  Embeddings shape: {all_embeddings.shape}")
    
    # Extract per-template embeddings
    template_embeddings: dict[str, np.ndarray] = {}
    for name, (start, end) in prompt_indices.items():
        template_embeddings[name] = all_embeddings[start:end]
    
    # Phase 3: Compute statistics
    print("\n[3/3] Computing diversity statistics...")
    profiles = []
    
    for name, template in templates.items():
        print(f"  {name}...", end=" ", flush=True)
        
        structure = template.get("structure", "")
        slots = parse_template_slots(structure)
        emb = template_embeddings[name]
        
        # Internal diversity
        internal = compute_similarity_stats(emb)
        
        # Cross-template similarity (vs all other templates combined)
        other_embeddings = []
        for other_name, other_emb in template_embeddings.items():
            if other_name != name:
                other_embeddings.append(other_emb)
        
        if other_embeddings:
            combined_others = np.vstack(other_embeddings)
            cross = compute_cross_similarity(emb, combined_others)
        else:
            cross = {"mean": 0, "min": 0, "max": 0}
        
        profile = TemplateProfile(
            name=name,
            structure=structure,
            n_samples=n_samples,
            internal_mean=internal["mean"],
            internal_min=internal["min"],
            internal_max=internal["max"],
            internal_range=internal["range"],
            cross_mean=cross["mean"],
            cross_min=cross["min"],
            cross_max=cross["max"],
            n_slots=len(slots),
            slot_categories=[s[0] for s in slots],
            multi_component_slots=sum(1 for s in slots if s[1] > 1),
            theoretical_combinations=compute_theoretical_combinations(slots, components)
        )
        profiles.append(profile)
        print(f"internal={profile.internal_score():.3f}, unique={profile.uniqueness_score():.3f}")
    
    return profiles, all_prompts, template_embeddings


def compute_template_pairwise_matrix(
    template_embeddings: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise similarity matrix between templates.
    
    Returns template names and NxN matrix of mean cross-similarities.
    """
    names = list(template_embeddings.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                # Diagonal: internal mean similarity
                stats = compute_similarity_stats(template_embeddings[name_a])
                matrix[i, j] = stats["mean"]
            else:
                # Off-diagonal: cross similarity
                cross = compute_cross_similarity(
                    template_embeddings[name_a],
                    template_embeddings[name_b]
                )
                matrix[i, j] = cross["mean"]
    
    return names, matrix


def print_category_usage_matrix(templates: dict[str, dict], components: dict[str, list[str]]):
    """Show which categories each template uses."""
    categories = sorted(components.keys())
    
    print("\n" + "=" * 110)
    print("CATEGORY USAGE BY TEMPLATE")
    print("=" * 110)
    
    # Header - abbreviated category names
    cat_abbrev = [c[:6] for c in categories]
    print(f"{'Template':<20}", end="")
    for ca in cat_abbrev:
        print(f"{ca:>8}", end="")
    print(f"{'Total':>8}")
    print("-" * (20 + 8 * (len(categories) + 1)))
    
    for name, template in templates.items():
        slots = parse_template_slots(template.get("structure", ""))
        slot_cats = [s[0] for s in slots]
        
        print(f"{name:<20}", end="")
        count = 0
        for cat in categories:
            n = slot_cats.count(cat)
            if n > 0:
                print(f"{n:>8}", end="")
                count += n
            else:
                print(f"{'·':>8}", end="")
        print(f"{count:>8}")


def extract_fixed_phrases(templates: dict[str, dict]):
    """Extract non-slot text from each template."""
    print("\n" + "=" * 110)
    print("FIXED PHRASES PER TEMPLATE")
    print("=" * 110)
    
    for name, template in templates.items():
        structure = template.get("structure", "")
        # Remove all {slot} patterns
        fixed = re.sub(r'\{[^}]+\}', '___', structure)
        # Clean up multiple underscores and normalize spacing
        fixed = re.sub(r'_+', ' ___ ', fixed).strip()
        fixed = re.sub(r'\s+', ' ', fixed)
        print(f"\n{name}:")
        print(f"  {fixed}")


def print_profiles(profiles: list[TemplateProfile], baseline_profiles: list[TemplateProfile] | None = None):
    """Print template profiles with optional baseline comparison."""
    print("\n" + "=" * 110)
    print("TEMPLATE DIVERSITY PROFILES")
    print("=" * 110)
    
    # Build baseline lookup if provided
    baseline_lookup = {}
    if baseline_profiles:
        baseline_lookup = {p.name: p for p in baseline_profiles}
    
    # Sort by combined score
    profiles_sorted = sorted(profiles, key=lambda p: p.combined_score(), reverse=True)
    
    # Header with optional delta columns
    if baseline_lookup:
        print(f"\n{'Rank':<5} {'Template':<18} {'Combined':>9} {'Δ':>6} {'Internal':>9} {'Δ':>6} "
              f"{'Unique':>9} {'Δ':>6} {'Int.Mean':>9} {'Cross.Mean':>11}")
        print("-" * 120)
    else:
        print(f"\n{'Rank':<5} {'Template':<18} {'Combined':>9} {'Internal':>9} {'Unique':>9} "
              f"{'Int.Mean':>9} {'Int.Range':>10} {'Cross.Mean':>11}")
        print("-" * 110)
    
    for rank, p in enumerate(profiles_sorted, 1):
        if baseline_lookup and p.name in baseline_lookup:
            base = baseline_lookup[p.name]
            d_comb = p.combined_score() - base.combined_score()
            d_int = p.internal_score() - base.internal_score()
            d_uniq = p.uniqueness_score() - base.uniqueness_score()
            d_comb_str = f"{d_comb:+.3f}" if d_comb != 0 else "   ·  "
            d_int_str = f"{d_int:+.3f}" if d_int != 0 else "   ·  "
            d_uniq_str = f"{d_uniq:+.3f}" if d_uniq != 0 else "   ·  "
            print(f"{rank:<5} {p.name:<18} {p.combined_score():>9.3f} {d_comb_str:>6} "
                  f"{p.internal_score():>9.3f} {d_int_str:>6} {p.uniqueness_score():>9.3f} {d_uniq_str:>6} "
                  f"{p.internal_mean:>9.3f} {p.cross_mean:>11.3f}")
        else:
            print(f"{rank:<5} {p.name:<18} {p.combined_score():>9.3f} {p.internal_score():>9.3f} "
                  f"{p.uniqueness_score():>9.3f} {p.internal_mean:>9.3f} {p.internal_range:>10.3f} "
                  f"{p.cross_mean:>11.3f}")
    
    # Best/worst analysis
    print("\n" + "-" * 110)
    best = profiles_sorted[0]
    print(f"🏆 Best overall: {best.name} (combined={best.combined_score():.3f})")
    
    best_internal = max(profiles, key=lambda p: p.internal_score())
    print(f"📊 Best internal diversity: {best_internal.name} (score={best_internal.internal_score():.3f})")
    
    best_unique = max(profiles, key=lambda p: p.uniqueness_score())
    print(f"🎯 Most unique: {best_unique.name} (score={best_unique.uniqueness_score():.3f})")


def print_pairwise_matrix(names: list[str], matrix: np.ndarray):
    """Print cross-template similarity matrix."""
    print("\n" + "=" * 110)
    print("CROSS-TEMPLATE SIMILARITY MATRIX")
    print("=" * 110)
    print("(diagonal = internal similarity, off-diagonal = cross-template similarity)")
    
    # Truncate names for display
    short_names = [n[:10] for n in names]
    
    # Header
    print(f"\n{'':>12}", end="")
    for sn in short_names:
        print(f"{sn:>11}", end="")
    print()
    print("-" * (12 + 11 * len(names)))
    
    # Rows
    for i, name in enumerate(names):
        print(f"{short_names[i]:>12}", end="")
        for j in range(len(names)):
            val = matrix[i, j]
            # Highlight diagonal differently
            if i == j:
                print(f"  [{val:.3f}]", end="")
            else:
                print(f"   {val:.3f} ", end="")
        print()
    
    # Find most/least similar pairs
    print("\n" + "-" * 110)
    
    # Off-diagonal only
    min_sim = float('inf')
    max_sim = float('-inf')
    min_pair = None
    max_pair = None
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if matrix[i, j] < min_sim:
                min_sim = matrix[i, j]
                min_pair = (names[i], names[j])
            if matrix[i, j] > max_sim:
                max_sim = matrix[i, j]
                max_pair = (names[i], names[j])
    
    if min_pair:
        print(f"Most distinct pair: {min_pair[0]} ↔ {min_pair[1]} (sim={min_sim:.3f})")
    if max_pair:
        print(f"Most similar pair: {max_pair[0]} ↔ {max_pair[1]} (sim={max_sim:.3f})")


def print_template_clusters(names: list[str], matrix: np.ndarray, threshold: float = 0.6):
    """Identify template clusters based on similarity."""
    print("\n" + "=" * 110)
    print(f"TEMPLATE CLUSTERS (threshold={threshold})")
    print("=" * 110)
    
    # Simple clustering: templates with cross-sim > threshold are grouped
    clusters = []
    assigned = set()
    
    for i, name in enumerate(names):
        if name in assigned:
            continue
        
        cluster = [name]
        assigned.add(name)
        
        for j, other_name in enumerate(names):
            if other_name in assigned or i == j:
                continue
            if matrix[i, j] > threshold:
                cluster.append(other_name)
                assigned.add(other_name)
        
        clusters.append(cluster)
    
    for i, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:
            print(f"\nCluster {i} (similar templates):")
            for name in cluster:
                print(f"  - {name}")
        else:
            print(f"\n🎯 Unique: {cluster[0]}")


def generate_prompts_random(
    template: dict,
    components: dict[str, list[str]],
    n_prompts: int,
    seed: int = 42
) -> list[str]:
    """Generate prompts with pure random component selection.
    
    Tracks used words per-category within each prompt to avoid
    duplicates when the same category appears in multiple slots
    (e.g., "{temporal_state} becoming {temporal_state}").
    """
    import random
    rng = random.Random(seed)
    
    structure = template.get("structure", "")
    slots = parse_template_slots(structure)
    
    prompts = []
    for _ in range(n_prompts):
        prompt = structure
        # Track used words per category within this prompt
        used_in_prompt: dict[str, set[str]] = defaultdict(set)
        
        for category, count, sep in slots:
            if category not in components:
                continue
            
            words = components[category]
            # Exclude words already used in this prompt for this category
            available = [w for w in words if w not in used_in_prompt[category]]
            if not available:
                # Fallback: if all used, allow repeats (shouldn't happen with ~30 words)
                available = words
            
            sample_size = min(count, len(available))
            selected = rng.sample(available, sample_size)
            
            # Mark these as used
            used_in_prompt[category].update(selected)
            
            replacement = sep.join(selected)
            
            if count > 1:
                if sep != " ":
                    pattern = f"{{{category}:{count}:{sep}}}"
                else:
                    pattern = f"{{{category}:{count}}}"
            else:
                pattern = f"{{{category}}}"
            
            prompt = prompt.replace(pattern, replacement, 1)
        
        prompts.append(prompt)
    
    return prompts


def save_samples_to_log(
    all_prompts: dict[str, list[str]], 
    output_path: Path, 
    n_samples: int
):
    """Save sample prompts to a log file for review."""
    with open(output_path, 'w') as f:
        f.write("# Template Sample Log\n")
        f.write(f"# Generated: {n_samples} samples per template\n")
        f.write("# For quality review and prompt tuning\n")
        f.write("=" * 80 + "\n\n")
        
        for template_name, prompts in all_prompts.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"TEMPLATE: {template_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            for i, prompt in enumerate(prompts[:n_samples], 1):
                f.write(f"{i:3}. {prompt}\n")
            
            f.write("\n")
    
    print(f"\n📝 Saved {n_samples} samples per template to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile template diversity potential")
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=500,
        help="Number of prompts per template (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use pure random selection (default: stable round-robin)"
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Show cross-template similarity matrix"
    )
    parser.add_argument(
        "--show-clusters",
        action="store_true",
        help="Show template clusters"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        metavar="N",
        help="Show N sample prompts from each template"
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        metavar="N",
        help="Save N sample prompts per template to a log file"
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=Path("template_samples.log"),
        help="Output path for sample log (default: template_samples.log)"
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.55,
        help="Similarity threshold for clustering (default: 0.55)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for embedding computation (default: 1000)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against baseline template stats (YAML with previous run)"
    )
    parser.add_argument(
        "--save-stats",
        type=Path,
        help="Save profile stats to YAML for future baseline comparison"
    )
    parser.add_argument(
        "--show-categories",
        action="store_true",
        help="Show category usage matrix by template"
    )
    parser.add_argument(
        "--show-phrases",
        action="store_true",
        help="Show fixed phrases per template"
    )
    parser.add_argument(
        "--templates",
        type=Path,
        help="Path to templates YAML (default: prompts/templates.yaml)"
    )
    parser.add_argument(
        "--components",
        type=Path,
        help="Path to components YAML (default: prompts/components.yaml)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    templates_path = args.templates or (project_root / "prompts" / "templates.yaml")
    components_path = args.components or (project_root / "prompts" / "components.yaml")
    
    print("Loading resources...")
    templates = load_templates(templates_path)
    components = load_components(components_path)
    encode = load_clip_embedder()
    
    selection_mode = "random" if args.random else "stable round-robin"
    print(f"Templates: {len(templates)}")
    print(f"Components: {sum(len(v) for v in components.values())} across {len(components)} categories")
    print(f"Samples per template: {args.samples}")
    print(f"Selection mode: {selection_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    
    # Show structural analysis before profiling
    if args.show_categories:
        print_category_usage_matrix(templates, components)
    
    if args.show_phrases:
        extract_fixed_phrases(templates)
    
    # Profile all templates
    profiles, all_prompts, template_embeddings = profile_all_templates(
        templates, components, encode, 
        n_samples=args.samples, seed=args.seed,
        use_random=args.random, batch_size=args.batch_size
    )
    
    # Load baseline if provided
    baseline_profiles = None
    if args.baseline and args.baseline.exists():
        print(f"\nLoading baseline from {args.baseline}...")
        with open(args.baseline) as f:
            baseline_data = yaml.safe_load(f)
        baseline_profiles = []
        for name, stats in baseline_data.get("profiles", {}).items():
            bp = TemplateProfile(
                name=name,
                structure=stats.get("structure", ""),
                n_samples=stats.get("n_samples", 0),
                internal_mean=stats.get("internal_mean", 0),
                internal_min=stats.get("internal_min", 0),
                internal_max=stats.get("internal_max", 0),
                internal_range=stats.get("internal_range", 0),
                cross_mean=stats.get("cross_mean", 0),
                cross_min=stats.get("cross_min", 0),
                cross_max=stats.get("cross_max", 0),
            )
            baseline_profiles.append(bp)
        print(f"Loaded {len(baseline_profiles)} baseline profiles")
    
    # Print main results
    print_profiles(profiles, baseline_profiles)
    
    # Save stats for future baseline comparison
    if args.save_stats:
        stats_data = {"profiles": {}}
        for p in profiles:
            stats_data["profiles"][p.name] = {
                "structure": p.structure,
                "n_samples": p.n_samples,
                "internal_mean": p.internal_mean,
                "internal_min": p.internal_min,
                "internal_max": p.internal_max,
                "internal_range": p.internal_range,
                "cross_mean": p.cross_mean,
                "cross_min": p.cross_min,
                "cross_max": p.cross_max,
                "combined_score": p.combined_score(),
                "internal_score": p.internal_score(),
                "uniqueness_score": p.uniqueness_score(),
            }
        with open(args.save_stats, 'w') as f:
            yaml.dump(stats_data, f, default_flow_style=False)
        print(f"\n📊 Saved stats to {args.save_stats}")
    
    # Cross-template matrix - reuse embeddings from profiling
    if args.show_matrix or args.show_clusters:
        print("\nComputing cross-template matrix (reusing embeddings)...")
        names, matrix = compute_template_pairwise_matrix(template_embeddings)
        
        if args.show_matrix:
            print_pairwise_matrix(names, matrix)
        
        if args.show_clusters:
            print_template_clusters(names, matrix, args.cluster_threshold)
    
    # Save samples to log file if requested
    if args.save_samples:
        save_samples_to_log(all_prompts, args.output_log, args.save_samples)
    
    # Show samples if requested
    if args.show_samples:
        print("\n" + "=" * 110)
        print(f"SAMPLE PROMPTS ({args.show_samples} per template)")
        print("=" * 110)
        
        for name, prompts in all_prompts.items():
            print(f"\n{name}:")
            for p in prompts[:args.show_samples]:
                truncated = p[:95] + "..." if len(p) > 95 else p
                print(f"  • {truncated}")


if __name__ == "__main__":
    main()
