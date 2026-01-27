#!/usr/bin/env python3
"""
Visualize Intrinsic Metrics Over Time
=====================================

Plots temporal trends in image health metrics to identify collapse patterns.

Usage:
    python plot_metric.py intrinsic_metrics.json --output plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Use a nicer style
plt.style.use('seaborn-v0_8-whitegrid')


def load_results(json_path: Path) -> Dict:
    with open(json_path) as f:
        return json.load(f)


def get_dataset_name(json_path: Path) -> str:
    """Extract a readable name from the json file path"""
    stem = json_path.stem
    if 'calibration' in stem.lower() or 'intrinsic' in stem.lower():
        return 'Calibration'
    elif 'keyframe' in stem.lower():
        return 'Keyframes'
    return stem


def plot_temporal_trends(data: Dict, output_dir: Path, dataset_name: str = ''):
    """Plot key metrics over frame index"""
    per_frame = data['per_frame']
    n_frames = len(per_frame)
    
    # Key metrics to track
    key_metrics = [
        ('hue_entropy', 'Hue Entropy (color diversity)'),
        ('value_std', 'Value StdDev (dynamic range)'),
        ('laplacian_variance', 'Laplacian Var (sharpness)'),
        ('high_freq_ratio', 'High Freq Ratio (detail)'),
        ('color_correlation_mean', 'Color Correlation (desaturation)'),
        ('edge_density', 'Edge Density'),
    ]
    
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(14, 3 * len(key_metrics)))
    title = f'Intrinsic Image Metrics Over Time - {dataset_name}' if dataset_name else 'Intrinsic Image Metrics Over Time'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    x = np.arange(n_frames)
    
    # Color by category if available
    categories = [f.get('category', 'unknown') for f in per_frame]
    has_categories = len(set(categories)) > 1 and 'unknown' not in categories
    
    for ax, (metric, title) in zip(axes, key_metrics):
        values = np.array([f[metric] for f in per_frame])
        
        # Raw values with category coloring if available
        if has_categories:
            cat_colors = {'broad': '#55A868', 'deep': '#C44E52', 'intervention': '#4C72B0'}
            for cat, color in cat_colors.items():
                mask = np.array([c == cat for c in categories])
                if mask.any():
                    ax.scatter(x[mask], values[mask], alpha=0.4, s=5, color=color, label=cat)
        else:
            ax.plot(x, values, alpha=0.3, linewidth=0.5, color='#4C72B0', label='Raw')
        
        # Rolling average (window=20)
        window = min(20, n_frames // 5) if n_frames > 10 else n_frames
        if n_frames > window and window > 1:
            rolling = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, n_frames), rolling, linewidth=2, color='#333333', label=f'{window}-frame avg')
        
        # Mark P10/P90 thresholds
        p10 = np.percentile(values, 10)
        p90 = np.percentile(values, 90)
        ax.axhline(p10, color='#C44E52', linestyle='--', alpha=0.7, label=f'P10={p10:.3f}')
        ax.axhline(p90, color='#55A868', linestyle='--', alpha=0.7, label=f'P90={p90:.3f}')
        
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Frame Index')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: temporal_trends.png")


def plot_metric_distributions(data: Dict, output_dir: Path, dataset_name: str = ''):
    """Plot distributions of each metric"""
    per_frame = data['per_frame']
    
    # Filter out non-numeric fields
    exclude_fields = ('filename', 'index', 'template', 'category')
    metrics = [k for k in per_frame[0].keys() if k not in exclude_fields]
    
    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes = np.array(axes).flatten()
    
    for i, metric in enumerate(metrics):
        values = [f[metric] for f in per_frame]
        ax = axes[i]
        
        ax.hist(values, bins=50, edgecolor='white', alpha=0.8, color='#4C72B0')
        ax.axvline(np.mean(values), color='#C44E52', linestyle='--', linewidth=2, label=f'Mean={np.mean(values):.3f}')
        ax.axvline(np.percentile(values, 10), color='#DD8452', linestyle=':', linewidth=2, label='P10')
        ax.axvline(np.percentile(values, 90), color='#55A868', linestyle=':', linewidth=2, label='P90')
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    title = f'Metric Distributions - {dataset_name}' if dataset_name else 'Metric Distributions'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: metric_distributions.png")


def plot_template_comparison(data: Dict, output_dir: Path, dataset_name: str = ''):
    """Compare metrics across templates"""
    template_stats = data.get('template_stats', {})
    if not template_stats:
        print("No template data available - skipping template_comparison.png")
        return
    
    key_metrics = ['hue_entropy', 'value_std', 'laplacian_variance', 'high_freq_ratio']
    templates = sorted(template_stats.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(templates)))
    
    for ax, metric in zip(axes, key_metrics):
        means = [template_stats[t][metric]['mean'] for t in templates]
        stds = [template_stats[t][metric]['stdev'] for t in templates]
        
        x = np.arange(len(templates))
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color=colors, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(templates, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    title = f'Per-Template Metric Comparison - {dataset_name}' if dataset_name else 'Per-Template Metric Comparison'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'template_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: template_comparison.png")


def plot_category_comparison(data: Dict, output_dir: Path, dataset_name: str = ''):
    """Compare metrics across categories (broad/deep/intervention)"""
    category_stats = data.get('category_stats', {})
    if not category_stats or len(category_stats) < 2:
        print("No category data available - skipping category_comparison.png")
        return
    
    # Filter out 'unknown' category
    categories = [c for c in ['broad', 'deep', 'intervention'] if c in category_stats]
    if len(categories) < 2:
        print("Need at least 2 categories - skipping category_comparison.png")
        return
    
    key_metrics = ['hue_entropy', 'value_std', 'laplacian_variance', 'high_freq_ratio', 
                   'unique_colors_q16', 'color_correlation_mean']
    
    # Category colors - green for healthy (broad), red for collapse (deep), blue for intervention
    cat_colors = {'broad': '#55A868', 'deep': '#C44E52', 'intervention': '#4C72B0'}
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    x = np.arange(len(categories))
    width = 0.6
    
    for ax, metric in zip(axes, key_metrics):
        means = []
        stds = []
        colors = []
        
        for cat in categories:
            stats = category_stats[cat].get(metric, {})
            means.append(stats.get('mean', 0))
            stds.append(stats.get('stdev', 0))
            colors.append(cat_colors.get(cat, '#888888'))
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, 
                      edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], label=f'{c} (n={category_stats[c].get("count", "?")})') 
                      for c in categories]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(categories), 
               bbox_to_anchor=(0.5, 0.02), fontsize=11)
    
    title = f'Category Comparison (Collapse Detection) - {dataset_name}' if dataset_name else 'Category Comparison (Collapse Detection)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'category_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: category_comparison.png")


def plot_collapse_detection_demo(data: Dict, output_dir: Path, dataset_name: str = ''):
    """
    Demo: What collapse detection would look like using these metrics
    """
    per_frame = data['per_frame']
    n_frames = len(per_frame)
    
    if n_frames < 10:
        print("Not enough frames for collapse detection demo - skipping")
        return
    
    # Create a composite "health score"
    # Normalize each metric to 0-1 range, then combine
    metrics_for_health = ['hue_entropy', 'value_std', 'laplacian_variance', 'high_freq_ratio']
    
    normalized = {}
    for metric in metrics_for_health:
        values = np.array([f[metric] for f in per_frame])
        # Normalize to 0-1 using percentiles (robust to outliers)
        p5, p95 = np.percentile(values, 5), np.percentile(values, 95)
        normalized[metric] = np.clip((values - p5) / (p95 - p5 + 1e-10), 0, 1)
    
    # Health score = average of normalized metrics
    health_scores = np.mean([normalized[m] for m in metrics_for_health], axis=0)
    
    # Rolling average (adaptive window)
    window = min(30, n_frames // 5) if n_frames > 30 else max(3, n_frames // 3)
    health_rolling = np.convolve(health_scores, np.ones(window)/window, mode='valid')
    
    # Detect "collapse" as sustained decline
    # Simple: where rolling health drops below threshold
    threshold = np.percentile(health_rolling, 20)  # Bottom 20% = potential collapse
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Check for category data to color points
    categories = [f.get('category', 'unknown') for f in per_frame]
    has_categories = len(set(categories)) > 1 and 'unknown' not in categories
    
    # Top: Health score
    if has_categories:
        cat_colors = {'broad': '#55A868', 'deep': '#C44E52', 'intervention': '#4C72B0'}
        for cat, color in cat_colors.items():
            mask = np.array([c == cat for c in categories])
            if mask.any():
                ax1.scatter(np.arange(n_frames)[mask], health_scores[mask], alpha=0.5, s=10, color=color, label=cat)
    else:
        ax1.plot(health_scores, alpha=0.3, linewidth=0.5, color='#4C72B0', label='Raw')
    
    ax1.plot(np.arange(window-1, n_frames), health_rolling, linewidth=2.5, color='#333333', label=f'{window}-frame avg')
    ax1.axhline(threshold, color='#C44E52', linestyle='--', linewidth=2, label=f'Alert threshold (P20={threshold:.3f})')
    ax1.fill_between(np.arange(window-1, n_frames), 0, 1, 
                     where=health_rolling < threshold, alpha=0.2, color='#C44E52', label='Potential collapse')
    ax1.set_ylabel('Health Score', fontsize=11)
    ax1.set_title('Composite Image Health Score (higher = healthier)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Bottom: Individual metrics (normalized)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for metric, color in zip(metrics_for_health, colors):
        ax2.plot(normalized[metric], alpha=0.6, linewidth=1.5, color=color, label=metric)
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Normalized Value', fontsize=11)
    ax2.set_title('Individual Metrics (normalized)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    title = f'Collapse Detection Demo - {dataset_name}' if dataset_name else 'Collapse Detection Demo'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'collapse_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: collapse_detection_demo.png")


def main():
    parser = argparse.ArgumentParser(description="Plot intrinsic metrics results")
    parser.add_argument('json_file', type=str, help='JSON file from metric_v2.py')
    parser.add_argument('--output', '-o', type=str, default='plots', help='Output directory')
    parser.add_argument('--name', '-n', type=str, default=None, help='Dataset name for titles')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    data = load_results(json_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset name
    dataset_name = args.name if args.name else get_dataset_name(json_path)
    
    print(f"Loaded {data['meta']['num_images']} frames from {dataset_name}")
    
    # Show category breakdown if available
    if 'categories' in data['meta']:
        cats = data['meta']['categories']
        print(f"Categories: {', '.join(f'{k}={v}' for k, v in cats.items())}")
    
    plot_temporal_trends(data, output_dir, dataset_name)
    plot_metric_distributions(data, output_dir, dataset_name)
    plot_category_comparison(data, output_dir, dataset_name)
    plot_template_comparison(data, output_dir, dataset_name)
    plot_collapse_detection_demo(data, output_dir, dataset_name)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()