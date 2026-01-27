#!/usr/bin/env python3
"""
Plot Metrics v2
===============

Visualize intrinsic metrics from metric_v2.py output.

Generates:
  - Time series plots for each metric
  - Distribution histograms
  - Correlation heatmap
  - Category comparison (if applicable)

Usage:
    python plot_metrics_v2.py metrics.json --output-dir plots/
    python plot_metrics_v2.py metrics.json --output-dir plots/ --highlight 60,70  # highlight frame range
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_time_series(data: dict, output_dir: Path, highlight_range: Optional[Tuple[int, int]] = None):
    """Plot each metric over time (frame index)."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    # Get frame numbers or indices
    if 'frame_number' in per_frame[0]:
        x = [f['frame_number'] for f in per_frame]
        xlabel = 'Frame Number'
    else:
        x = [f['index'] for f in per_frame]
        xlabel = 'Frame Index'
    
    # Create subplots - 4 columns
    n_metrics = len(metric_names)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        y = [f[metric] for f in per_frame]
        
        ax.plot(x, y, linewidth=0.8, alpha=0.8)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        
        # Add highlight region if specified
        if highlight_range:
            ax.axvspan(highlight_range[0], highlight_range[1], 
                       alpha=0.2, color='red', label='Highlight')
        
        # Add global mean line
        mean = data['global_stats'][metric]['mean']
        ax.axhline(mean, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=150)
    plt.close()
    print(f"Saved: temporal_trends.png")


def plot_distributions(data: dict, output_dir: Path):
    """Plot distribution histograms for each metric."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    n_metrics = len(metric_names)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        values = [f[metric] for f in per_frame]
        
        ax.hist(values, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        
        # Add percentile markers
        stats = data['global_stats'][metric]
        ax.axvline(stats['p10'], color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='P10')
        ax.axvline(stats['p50'], color='green', linestyle='-', linewidth=0.8, alpha=0.7, label='P50')
        ax.axvline(stats['p90'], color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='P90')
    
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150)
    plt.close()
    print(f"Saved: metric_distributions.png")


def plot_correlation_matrix(data: dict, output_dir: Path):
    """Plot correlation heatmap between all metrics."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    # Build matrix
    n = len(metric_names)
    values = np.zeros((len(per_frame), n))
    for i, frame in enumerate(per_frame):
        for j, metric in enumerate(metric_names):
            values[i, j] = frame[metric]
    
    corr = np.corrcoef(values.T)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(metric_names, fontsize=8)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{corr[i, j]:.2f}',
                          ha='center', va='center', fontsize=6,
                          color='white' if abs(corr[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Metric Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150)
    plt.close()
    print(f"Saved: correlation_matrix.png")


def plot_category_comparison(data: dict, output_dir: Path):
    """Plot category comparison if multiple categories exist."""
    category_stats = data.get('category_stats', {})
    
    # Filter out unknown and single-frame categories
    categories = [c for c in category_stats if c != 'unknown' and category_stats[c]['count'] > 1]
    
    if len(categories) < 2:
        print("Skipping category comparison (need at least 2 categories)")
        return
    
    metric_names = data['meta']['metric_names']
    
    # Select key metrics for comparison
    key_metrics = [m for m in metric_names if m in category_stats[categories[0]]][:8]
    
    n_metrics = len(key_metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for i, metric in enumerate(key_metrics):
        ax = axes[i]
        
        x = range(len(categories))
        means = []
        stds = []
        
        for cat in categories:
            stats = category_stats[cat].get(metric, {})
            means.append(stats.get('mean', 0))
            stds.append(stats.get('stdev', 0))
        
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors[:len(categories)], alpha=0.7)
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
    
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: category_comparison.png")


def plot_collapse_detection(data: dict, output_dir: Path, highlight_range: Optional[Tuple[int, int]] = None):
    """
    Plot metrics that are useful for collapse detection together.
    Shows both raw values and rolling averages.
    """
    per_frame = data['per_frame']
    
    # Metrics to plot together
    collapse_metrics = [
        ('local_autocorrelation', 'lower = more noise'),
        ('laplacian_variance', 'high = sharp edges OR noise'),
        ('block_coherence_variance', 'high = varied OR chaotic'),
        ('gradient_direction_entropy', 'high = edges point randomly'),
    ]
    
    # Filter to metrics that exist
    collapse_metrics = [(m, d) for m, d in collapse_metrics if m in per_frame[0]]
    
    if not collapse_metrics:
        print("No collapse detection metrics found")
        return
    
    if 'frame_number' in per_frame[0]:
        x = [f['frame_number'] for f in per_frame]
        xlabel = 'Frame Number'
    else:
        x = [f['index'] for f in per_frame]
        xlabel = 'Frame Index'
    
    fig, axes = plt.subplots(len(collapse_metrics), 1, figsize=(14, 3 * len(collapse_metrics)), 
                             sharex=True)
    if len(collapse_metrics) == 1:
        axes = [axes]
    
    for ax, (metric, desc) in zip(axes, collapse_metrics):
        y = np.array([f[metric] for f in per_frame])
        
        # Raw values
        ax.plot(x, y, linewidth=0.5, alpha=0.4, color='blue', label='raw')
        
        # Rolling average (window=10)
        window = 10
        if len(y) > window:
            rolling = np.convolve(y, np.ones(window)/window, mode='valid')
            x_rolling = x[window-1:]
            ax.plot(x_rolling, rolling, linewidth=1.5, color='blue', label=f'rolling avg (w={window})')
        
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f'{metric} - {desc}', fontsize=10)
        ax.legend(loc='upper right', fontsize=7)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add global stats
        stats = data['global_stats'][metric]
        ax.axhline(stats['p10'], color='green', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.axhline(stats['p90'], color='red', linestyle=':', linewidth=0.8, alpha=0.6)
        
        # Highlight region
        if highlight_range:
            ax.axvspan(highlight_range[0], highlight_range[1], 
                       alpha=0.2, color='red')
    
    axes[-1].set_xlabel(xlabel, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'collapse_detection_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: collapse_detection_analysis.png")


def plot_delta_analysis(data: dict, output_dir: Path, highlight_range: Optional[Tuple[int, int]] = None):
    """Plot rate of change (delta) analysis."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    # Key metrics for delta analysis
    key_metrics = ['local_autocorrelation', 'laplacian_variance', 'edge_density', 
                   'local_contrast_mean', 'block_coherence_variance']
    key_metrics = [m for m in key_metrics if m in metric_names]
    
    fig, axes = plt.subplots(len(key_metrics), 2, figsize=(16, 3 * len(key_metrics)))
    
    for i, metric in enumerate(key_metrics):
        values = np.array([f[metric] for f in per_frame])
        
        # Compute deltas
        deltas = np.diff(values)
        delta_frames = frames[1:]
        
        # Rolling delta std (volatility indicator)
        window = 10
        rolling_std = []
        for j in range(len(deltas) - window + 1):
            rolling_std.append(np.std(deltas[j:j+window]))
        rolling_frames = delta_frames[window-1:]
        
        # Left plot: raw deltas
        ax1 = axes[i, 0]
        ax1.plot(delta_frames, deltas, linewidth=0.5, alpha=0.6, color='blue')
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax1.fill_between(delta_frames, 0, deltas, alpha=0.3, 
                        where=[d > 0 for d in deltas], color='green')
        ax1.fill_between(delta_frames, 0, deltas, alpha=0.3, 
                        where=[d < 0 for d in deltas], color='red')
        ax1.set_ylabel(f'{metric}\nΔ per frame', fontsize=8)
        ax1.set_title(f'{metric} - Frame-to-Frame Change', fontsize=9)
        ax1.tick_params(axis='both', labelsize=7)
        
        if highlight_range:
            ax1.axvspan(highlight_range[0], highlight_range[1], alpha=0.2, color='orange')
        
        # Right plot: rolling volatility
        ax2 = axes[i, 1]
        ax2.plot(rolling_frames, rolling_std, linewidth=1, color='purple')
        ax2.set_ylabel('Rolling Δ Std\n(volatility)', fontsize=8)
        ax2.set_title(f'{metric} - Volatility (window={window})', fontsize=9)
        ax2.tick_params(axis='both', labelsize=7)
        
        # Add threshold line (mean + 2*std)
        vol_mean = np.mean(rolling_std)
        vol_std = np.std(rolling_std)
        ax2.axhline(vol_mean + 2*vol_std, color='red', linestyle='--', linewidth=0.8, 
                   alpha=0.7, label='Anomaly threshold')
        
        if highlight_range:
            ax2.axvspan(highlight_range[0], highlight_range[1], alpha=0.2, color='orange')
    
    axes[-1, 0].set_xlabel('Frame Number', fontsize=9)
    axes[-1, 1].set_xlabel('Frame Number', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'delta_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: delta_analysis.png")


def plot_cumulative_change(data: dict, output_dir: Path, highlight_range: Optional[Tuple[int, int]] = None):
    """Plot cumulative change from start - shows drift over time."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    key_metrics = ['local_autocorrelation', 'laplacian_variance', 'edge_density', 
                   'local_contrast_mean', 'block_coherence_variance', 'saturation_mean']
    key_metrics = [m for m in key_metrics if m in metric_names]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_metrics)))
    
    for metric, color in zip(key_metrics, colors):
        values = np.array([f[metric] for f in per_frame])
        
        # Normalize to [0, 1] range for comparison
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 0:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            normalized = values * 0
        
        # Compute cumulative deviation from initial value
        cumulative = np.cumsum(np.diff(normalized, prepend=normalized[0]))
        
        ax.plot(frames, cumulative, linewidth=1.5, label=metric, color=color, alpha=0.8)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Frame Number', fontsize=10)
    ax.set_ylabel('Cumulative Change (normalized)', fontsize=10)
    ax.set_title('Cumulative Metric Drift Over Time', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    
    if highlight_range:
        ax.axvspan(highlight_range[0], highlight_range[1], alpha=0.2, color='red', label='Collapse')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_change.png', dpi=150)
    plt.close()
    print(f"Saved: cumulative_change.png")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from metric_v2.py output")
    parser.add_argument('metrics_json', type=str, help='Path to metrics JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='plots', 
                        help='Output directory for plots')
    parser.add_argument('--highlight', type=str, default=None,
                        help='Frame range to highlight (e.g., "60,70")')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not found. Install with: pip install matplotlib")
        return 1
    
    data = load_metrics(args.metrics_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    highlight_range = None
    if args.highlight:
        parts = args.highlight.split(',')
        highlight_range = (int(parts[0]), int(parts[1]))
    
    print(f"Generating plots for {data['meta']['num_images']} frames...")
    print(f"Metrics: {', '.join(data['meta']['metric_names'])}")
    print()
    
    plot_time_series(data, output_dir, highlight_range)
    plot_distributions(data, output_dir)
    plot_correlation_matrix(data, output_dir)
    plot_category_comparison(data, output_dir)
    plot_collapse_detection(data, output_dir, highlight_range)
    plot_delta_analysis(data, output_dir, highlight_range)
    plot_cumulative_change(data, output_dir, highlight_range)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    exit(main() or 0)

