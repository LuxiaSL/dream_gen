#!/usr/bin/env python3
"""Analyze diversity of a component selection YAML without modifying components.yaml.

Also supports prompt simulation to measure end-to-end diversity.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
from typing import Callable

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from find_diverse_components import analyze_diversity, load_clip_embedder


def load_selection(path: Path) -> dict[str, list[str]]:
    """Load a selection YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if isinstance(v, list)}


def analyze_categories(
    selection: dict[str, list[str]], 
    encode: Callable[[list[str]], np.ndarray]
) -> dict[str, dict]:
    """Analyze diversity for each category."""
    results = {}
    
    for category, words in selection.items():
        if len(words) < 2:
            continue
            
        embeddings = encode(words)
        stats = analyze_diversity(embeddings, words)
        stats["count"] = len(words)
        stats["category"] = category
        results[category] = stats
    
    return results


def print_comparison(baseline: dict, proposed: dict) -> None:
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("DIVERSITY COMPARISON: Baseline vs Proposed")
    print("=" * 80)
    
    # Header
    print(f"\n{'Category':<12} {'Count':>6} {'Count':>6}  {'Range':>7} {'Range':>7}  {'Δ Range':>8}")
    print(f"{'':12} {'(old)':>6} {'(new)':>6}  {'(old)':>7} {'(new)':>7}  {'':>8}")
    print("-" * 80)
    
    all_categories = set(baseline.keys()) | set(proposed.keys())
    
    improvements = []
    for cat in sorted(all_categories):
        b = baseline.get(cat, {})
        p = proposed.get(cat, {})
        
        b_count = b.get("count", 0)
        p_count = p.get("count", 0)
        b_range = b.get("range", 0)
        p_range = p.get("range", 0)
        
        delta = p_range - b_range
        delta_str = f"{'+' if delta > 0 else ''}{delta:.3f}"
        indicator = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        
        print(f"{cat:<12} {b_count:>6} {p_count:>6}  {b_range:>7.3f} {p_range:>7.3f}  {delta_str:>7} {indicator}")
        improvements.append((cat, delta))
    
    print("-" * 80)
    avg_improvement = np.mean([d for _, d in improvements])
    print(f"{'Average':12} {'':>6} {'':>6}  {'':>7} {'':>7}  {'+' if avg_improvement > 0 else ''}{avg_improvement:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    improved = sum(1 for _, d in improvements if d > 0.01)
    degraded = sum(1 for _, d in improvements if d < -0.01)
    
    print(f"  Categories improved: {improved}/{len(improvements)}")
    print(f"  Categories degraded: {degraded}/{len(improvements)}")
    print(f"  Average range change: {'+' if avg_improvement > 0 else ''}{avg_improvement:.3f}")


def load_templates(path: Path) -> dict[str, dict]:
    """Load templates from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    
    templates_list = data.get("templates", [])
    # Convert list to dict keyed by id
    return {t["id"]: t for t in templates_list if "id" in t}


def generate_prompts(
    templates: dict[str, dict],
    components: dict[str, list[str]],
    n_per_template: int = 50,
    seed: int = 42
) -> dict[str, list[str]]:
    """Generate sample prompts for each template."""
    import random
    import re
    rng = random.Random(seed)
    
    prompts_by_template = {}
    
    for template_name, template_data in templates.items():
        structure = template_data.get("structure", "")
        if not structure:
            continue
        
        prompts = []
        for _ in range(n_per_template):
            prompt = structure
            
            # Find all placeholders like {category}, {category:N}, {category:N:sep}
            placeholders = re.findall(r'\{(\w+)(?::(\d+))?(?::([^}]*))?\}', prompt)
            
            for category, count_str, separator in placeholders:
                if category not in components or not components[category]:
                    continue
                
                words = components[category]
                count = int(count_str) if count_str else 1
                sep = separator if separator else " "
                
                # Sample without replacement if possible
                sample_size = min(count, len(words))
                selected = rng.sample(words, sample_size)
                replacement = sep.join(selected)
                
                # Replace the placeholder
                if count_str:
                    if separator:
                        pattern = f"{{{category}:{count_str}:{separator}}}"
                    else:
                        pattern = f"{{{category}:{count_str}}}"
                else:
                    pattern = f"{{{category}}}"
                
                prompt = prompt.replace(pattern, replacement, 1)
            
            prompts.append(prompt)
        
        prompts_by_template[template_name] = prompts
    
    return prompts_by_template


def analyze_prompt_diversity(
    prompts_by_template: dict[str, list[str]],
    encode: Callable[[list[str]], np.ndarray]
) -> dict:
    """Analyze diversity of generated prompts."""
    results = {
        "per_template": {},
        "global": {}
    }
    
    all_prompts = []
    all_embeddings = []
    
    for template_name, prompts in prompts_by_template.items():
        if not prompts:
            continue
            
        embeddings = encode(prompts)
        stats = analyze_diversity(embeddings, prompts)
        stats["count"] = len(prompts)
        results["per_template"][template_name] = stats
        
        all_prompts.extend(prompts)
        all_embeddings.append(embeddings)
    
    # Global analysis
    if all_embeddings:
        combined = np.vstack(all_embeddings)
        global_stats = analyze_diversity(combined, all_prompts)
        global_stats["count"] = len(all_prompts)
        results["global"] = global_stats
    
    return results


def print_prompt_analysis(results: dict) -> None:
    """Print prompt diversity analysis."""
    print("\n" + "=" * 80)
    print("PROMPT DIVERSITY ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'Template':<20} {'Count':>6} {'Range':>8} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print("-" * 80)
    
    for name, stats in sorted(results["per_template"].items()):
        print(f"{name:<20} {stats['count']:>6} {stats['range']:>8.3f} "
              f"{stats['min']:>8.3f} {stats['mean']:>8.3f} {stats['max']:>8.3f}")
    
    print("-" * 80)
    g = results["global"]
    print(f"{'GLOBAL':<20} {g['count']:>6} {g['range']:>8.3f} "
          f"{g['min']:>8.3f} {g['mean']:>8.3f} {g['max']:>8.3f}")
    
    print("\n" + "-" * 80)
    print("Interpretation:")
    print("  Range = max_sim - min_sim (higher = more diverse)")
    print("  Mean = average pairwise similarity (lower = more diverse)")
    print("  Min similarity = most different pair (lower = better coverage)")


def main():
    parser = argparse.ArgumentParser(description="Analyze component selection diversity")
    parser.add_argument(
        "selection",
        type=Path,
        help="Path to selection YAML file"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against baseline selection YAML"
    )
    parser.add_argument(
        "--simulate-prompts",
        type=int,
        metavar="N",
        help="Generate N prompts per template and analyze diversity"
    )
    parser.add_argument(
        "--templates",
        type=Path,
        help="Path to templates.yaml (for prompt simulation)"
    )
    
    args = parser.parse_args()
    
    if not args.selection.exists():
        print(f"Error: {args.selection} not found")
        sys.exit(1)
    
    # Load embedder
    print("Loading CLIP model...")
    encode = load_clip_embedder()
    
    # Load and analyze proposed selection
    print(f"\nAnalyzing: {args.selection}")
    proposed = load_selection(args.selection)
    proposed_stats = analyze_categories(proposed, encode)
    
    # Compare with baseline if provided
    if args.baseline:
        if not args.baseline.exists():
            print(f"Warning: baseline {args.baseline} not found, skipping comparison")
        else:
            print(f"Comparing with baseline: {args.baseline}")
            baseline = load_selection(args.baseline)
            baseline_stats = analyze_categories(baseline, encode)
            print_comparison(baseline_stats, proposed_stats)
    else:
        # Just print proposed stats
        print("\n" + "=" * 80)
        print("DIVERSITY ANALYSIS")
        print("=" * 80)
        print(f"\n{'Category':<12} {'Count':>6} {'Range':>8} {'Min':>8} {'Mean':>8} {'Max':>8}")
        print("-" * 80)
        
        for cat in sorted(proposed_stats.keys()):
            s = proposed_stats[cat]
            print(f"{cat:<12} {s['count']:>6} {s['range']:>8.3f} "
                  f"{s['min']:>8.3f} {s['mean']:>8.3f} {s['max']:>8.3f}")
    
    # Prompt simulation
    if args.simulate_prompts:
        project_root = Path(__file__).parent.parent.parent
        templates_path = args.templates or project_root / "prompts" / "templates.yaml"
        
        if not templates_path.exists():
            print(f"\nError: templates not found at {templates_path}")
            sys.exit(1)
        
        print(f"\nSimulating {args.simulate_prompts} prompts per template...")
        templates = load_templates(templates_path)
        
        prompts = generate_prompts(templates, proposed, n_per_template=args.simulate_prompts)
        prompt_results = analyze_prompt_diversity(prompts, encode)
        print_prompt_analysis(prompt_results)
        
        # Show sample prompts
        print("\n" + "=" * 80)
        print("SAMPLE PROMPTS (3 per template)")
        print("=" * 80)
        for name, template_prompts in prompts.items():
            print(f"\n{name}:")
            for p in template_prompts[:3]:
                print(f"  • {p[:100]}{'...' if len(p) > 100 else ''}")


if __name__ == "__main__":
    main()

