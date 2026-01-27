#!/usr/bin/env python3
"""
Analyze Metrics v2 - Rate of Change & Pattern Detection
========================================================

Analyzes temporal patterns in metrics, focusing on:
- Rate of change (deltas) over time
- Windowed statistics
- Pattern matching to known collapse regions
- Anomaly detection based on delta behavior

Usage:
    python analyze_metrics_v2.py metrics.json --collapse-range 60,70
    python analyze_metrics_v2.py metrics.json --collapse-range 60,70 --window 5
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowStats:
    """Statistics for a window of frames."""
    start_frame: int
    end_frame: int
    mean: float
    std: float
    min_val: float
    max_val: float
    delta_mean: float  # Mean of frame-to-frame changes
    delta_std: float   # Std of frame-to-frame changes
    trend: float       # Linear trend (slope)
    total_change: float  # End - Start value


def compute_window_stats(values: List[float], frames: List[int], start_idx: int, end_idx: int) -> WindowStats:
    """Compute statistics for a window of values."""
    window_vals = values[start_idx:end_idx+1]
    window_frames = frames[start_idx:end_idx+1]
    
    # Frame-to-frame deltas
    deltas = [window_vals[i+1] - window_vals[i] for i in range(len(window_vals)-1)]
    
    # Linear trend (simple linear regression slope)
    if len(window_vals) > 1:
        x = np.arange(len(window_vals))
        coeffs = np.polyfit(x, window_vals, 1)
        trend = coeffs[0]
    else:
        trend = 0.0
    
    return WindowStats(
        start_frame=window_frames[0],
        end_frame=window_frames[-1],
        mean=float(np.mean(window_vals)),
        std=float(np.std(window_vals)),
        min_val=float(np.min(window_vals)),
        max_val=float(np.max(window_vals)),
        delta_mean=float(np.mean(deltas)) if deltas else 0.0,
        delta_std=float(np.std(deltas)) if deltas else 0.0,
        trend=float(trend),
        total_change=float(window_vals[-1] - window_vals[0])
    )


def compute_rolling_deltas(values: List[float], window: int = 1) -> List[float]:
    """Compute rolling deltas (rate of change) with optional smoothing window."""
    if window == 1:
        return [values[i+1] - values[i] for i in range(len(values)-1)]
    else:
        # Smoothed delta: average change over window
        deltas = []
        for i in range(len(values) - window):
            delta = (values[i + window] - values[i]) / window
            deltas.append(delta)
        return deltas


def find_similar_regions(
    values: List[float],
    frames: List[int],
    reference_start: int,
    reference_end: int,
    window_size: int,
    threshold_factor: float = 0.8
) -> List[Tuple[int, int, float]]:
    """
    Find regions with similar statistical behavior to a reference region.
    
    Returns list of (start_frame, end_frame, similarity_score) tuples.
    """
    # Get reference window indices
    ref_start_idx = next((i for i, f in enumerate(frames) if f >= reference_start), 0)
    ref_end_idx = next((i for i, f in enumerate(frames) if f >= reference_end), len(frames)-1)
    
    ref_stats = compute_window_stats(values, frames, ref_start_idx, ref_end_idx)
    
    # Slide window across all data
    similar_regions = []
    step = max(1, window_size // 2)
    
    for start_idx in range(0, len(values) - window_size, step):
        end_idx = start_idx + window_size - 1
        
        # Skip if overlaps with reference region
        if not (frames[end_idx] < reference_start or frames[start_idx] > reference_end):
            continue
        
        window_stats = compute_window_stats(values, frames, start_idx, end_idx)
        
        # Compute similarity based on multiple features
        # Normalize differences by reference values
        features = []
        
        # Delta behavior similarity (most important for collapse detection)
        if ref_stats.delta_std > 0:
            delta_std_sim = 1.0 - min(1.0, abs(window_stats.delta_std - ref_stats.delta_std) / ref_stats.delta_std)
            features.append(delta_std_sim * 2)  # Weight higher
        
        if abs(ref_stats.delta_mean) > 0.001:
            delta_mean_sim = 1.0 - min(1.0, abs(window_stats.delta_mean - ref_stats.delta_mean) / abs(ref_stats.delta_mean))
            features.append(delta_mean_sim * 2)
        
        # Trend similarity
        if abs(ref_stats.trend) > 0.001:
            trend_sim = 1.0 - min(1.0, abs(window_stats.trend - ref_stats.trend) / abs(ref_stats.trend))
            features.append(trend_sim)
        
        # Value range similarity
        if ref_stats.std > 0:
            std_sim = 1.0 - min(1.0, abs(window_stats.std - ref_stats.std) / ref_stats.std)
            features.append(std_sim)
        
        if features:
            similarity = np.mean(features)
            if similarity >= threshold_factor:
                similar_regions.append((frames[start_idx], frames[end_idx], similarity))
    
    # Sort by similarity score
    similar_regions.sort(key=lambda x: x[2], reverse=True)
    
    return similar_regions


def analyze_collapse_region(data: dict, collapse_start: int, collapse_end: int, window: int = 5):
    """Detailed analysis of collapse region behavior."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    # Get frame numbers
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    # Find collapse region indices
    collapse_start_idx = next((i for i, f in enumerate(frames) if f >= collapse_start), 0)
    collapse_end_idx = next((i for i, f in enumerate(frames) if f >= collapse_end), len(frames)-1)
    
    print("=" * 100)
    print(f"COLLAPSE REGION ANALYSIS: Frames {collapse_start} - {collapse_end}")
    print(f"Window size for rolling stats: {window} frames")
    print("=" * 100)
    
    results = {}
    
    for metric in metric_names:
        values = [f[metric] for f in per_frame]
        
        # Global stats for reference
        global_mean = np.mean(values)
        global_std = np.std(values)
        
        # Collapse region stats
        collapse_stats = compute_window_stats(values, frames, collapse_start_idx, collapse_end_idx)
        
        # Pre-collapse stats (same length window before)
        pre_length = collapse_end_idx - collapse_start_idx
        pre_start_idx = max(0, collapse_start_idx - pre_length)
        pre_stats = compute_window_stats(values, frames, pre_start_idx, collapse_start_idx - 1) if collapse_start_idx > 0 else None
        
        # Post-collapse stats
        post_end_idx = min(len(values) - 1, collapse_end_idx + pre_length)
        post_stats = compute_window_stats(values, frames, collapse_end_idx + 1, post_end_idx) if collapse_end_idx < len(values) - 1 else None
        
        # Compute deltas
        deltas = compute_rolling_deltas(values, window=1)
        collapse_deltas = deltas[collapse_start_idx:collapse_end_idx]
        
        results[metric] = {
            'global_mean': global_mean,
            'global_std': global_std,
            'collapse_stats': collapse_stats,
            'pre_stats': pre_stats,
            'post_stats': post_stats,
            'collapse_deltas': collapse_deltas,
        }
    
    return results


def print_detailed_analysis(results: dict, collapse_start: int, collapse_end: int):
    """Print detailed analysis results."""
    
    print("\n" + "=" * 100)
    print("RATE OF CHANGE ANALYSIS (Frame-to-Frame Deltas)")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} | {'Collapse Δ Mean':>15} | {'Collapse Δ Std':>15} | {'Global Δ Std':>15} | {'Ratio':>10}")
    print("-" * 100)
    
    # Sort metrics by how unusual their collapse behavior is
    metric_scores = []
    for metric, data in results.items():
        collapse_stats = data['collapse_stats']
        
        # Compute global delta std for comparison
        global_vals = [f for f in data.get('all_deltas', [])] if 'all_deltas' in data else []
        
        # Use collapse delta std vs expected
        score = abs(collapse_stats.delta_std / (data['global_std'] + 1e-10))
        metric_scores.append((metric, score, data))
    
    metric_scores.sort(key=lambda x: x[1], reverse=True)
    
    for metric, score, data in metric_scores:
        cs = data['collapse_stats']
        print(f"{metric:<30} | {cs.delta_mean:>+15.4f} | {cs.delta_std:>15.4f} | {data['global_std']:>15.4f} | {score:>10.2f}x")
    
    print("\n" + "=" * 100)
    print("TREND ANALYSIS (Linear Slope During Windows)")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} | {'Pre-Collapse':>15} | {'Collapse':>15} | {'Post-Collapse':>15} | {'Direction':<20}")
    print("-" * 120)
    
    for metric, data in results.items():
        cs = data['collapse_stats']
        pre = data['pre_stats']
        post = data['post_stats']
        
        pre_trend = f"{pre.trend:>+15.4f}" if pre else f"{'N/A':>15}"
        post_trend = f"{post.trend:>+15.4f}" if post else f"{'N/A':>15}"
        
        # Determine direction pattern
        if cs.trend > 0.1:
            direction = "↑ RISING"
        elif cs.trend < -0.1:
            direction = "↓ FALLING"
        else:
            direction = "→ STABLE"
        
        # Add volatility indicator
        if cs.delta_std > data['global_std'] * 1.5:
            direction += " (volatile)"
        
        print(f"{metric:<30} | {pre_trend} | {cs.trend:>+15.4f} | {post_trend} | {direction:<20}")
    
    print("\n" + "=" * 100)
    print("TOTAL CHANGE DURING COLLAPSE")
    print("=" * 100)
    
    print(f"\n{'Metric':<30} | {'Start Value':>15} | {'End Value':>15} | {'Total Change':>15} | {'% Change':>12}")
    print("-" * 100)
    
    for metric, data in results.items():
        cs = data['collapse_stats']
        start_val = cs.mean - cs.total_change / 2  # Approximate
        end_val = cs.mean + cs.total_change / 2
        pct_change = (cs.total_change / (abs(cs.mean) + 1e-10)) * 100
        
        print(f"{metric:<30} | {start_val:>15.4f} | {end_val:>15.4f} | {cs.total_change:>+15.4f} | {pct_change:>+11.1f}%")


def find_and_print_similar_regions(data: dict, results: dict, collapse_start: int, collapse_end: int, top_n: int = 5):
    """Find regions with similar behavior to collapse zone."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    window_size = collapse_end - collapse_start
    
    print("\n" + "=" * 100)
    print(f"SIMILAR REGIONS (matching collapse pattern, window={window_size} frames)")
    print("=" * 100)
    
    # Key metrics for collapse detection
    key_metrics = ['local_autocorrelation', 'laplacian_variance', 'block_coherence_variance', 
                   'edge_density', 'local_contrast_mean']
    key_metrics = [m for m in key_metrics if m in metric_names]
    
    all_similar = {}
    
    for metric in key_metrics:
        values = [f[metric] for f in per_frame]
        similar = find_similar_regions(values, frames, collapse_start, collapse_end, window_size, threshold_factor=0.6)
        all_similar[metric] = similar[:top_n]
        
        print(f"\n{metric}:")
        if similar:
            for start, end, score in similar[:top_n]:
                print(f"  Frames {start:>4} - {end:>4}  (similarity: {score:.2%})")
        else:
            print("  No similar regions found")
    
    # Find consensus regions (appear in multiple metrics)
    print("\n" + "-" * 100)
    print("CONSENSUS REGIONS (similar behavior across multiple metrics):")
    
    region_counts = {}
    for metric, regions in all_similar.items():
        for start, end, score in regions:
            key = (start, end)
            if key not in region_counts:
                region_counts[key] = {'metrics': [], 'scores': []}
            region_counts[key]['metrics'].append(metric)
            region_counts[key]['scores'].append(score)
    
    # Sort by number of metrics that flagged this region
    consensus = [(k, v) for k, v in region_counts.items() if len(v['metrics']) >= 2]
    consensus.sort(key=lambda x: (len(x[1]['metrics']), np.mean(x[1]['scores'])), reverse=True)
    
    if consensus:
        for (start, end), info in consensus[:10]:
            avg_score = np.mean(info['scores'])
            metrics_str = ", ".join(info['metrics'][:3])
            if len(info['metrics']) > 3:
                metrics_str += f" +{len(info['metrics'])-3} more"
            print(f"  Frames {start:>4} - {end:>4}  ({len(info['metrics'])} metrics, avg score: {avg_score:.2%})")
            print(f"    Metrics: {metrics_str}")
    else:
        print("  No consensus regions found")


def compute_windowed_stats_timeline(data: dict, window_size: int = 10):
    """Compute windowed statistics across the entire timeline."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    print("\n" + "=" * 100)
    print(f"WINDOWED STATISTICS (window={window_size} frames)")
    print("=" * 100)
    
    # Key metrics for detailed timeline
    key_metrics = ['local_autocorrelation', 'laplacian_variance', 'block_coherence_variance', 'edge_density']
    key_metrics = [m for m in key_metrics if m in metric_names]
    
    for metric in key_metrics:
        values = [f[metric] for f in per_frame]
        
        print(f"\n{metric}:")
        print(f"  {'Window':<15} | {'Mean':>12} | {'Std':>12} | {'Δ Mean':>12} | {'Δ Std':>12} | {'Trend':>12}")
        print("  " + "-" * 90)
        
        # Compute stats for each window
        windows = []
        for start_idx in range(0, len(values) - window_size + 1, window_size // 2):
            end_idx = min(start_idx + window_size - 1, len(values) - 1)
            stats = compute_window_stats(values, frames, start_idx, end_idx)
            windows.append(stats)
            
            # Flag unusual windows
            flag = ""
            if abs(stats.delta_std) > np.std([w.delta_std for w in windows]) * 2 if len(windows) > 3 else False:
                flag = " ⚠️"
            
            print(f"  {stats.start_frame:>4} - {stats.end_frame:<4}    | {stats.mean:>12.4f} | {stats.std:>12.4f} | "
                  f"{stats.delta_mean:>+12.4f} | {stats.delta_std:>12.4f} | {stats.trend:>+12.4f}{flag}")


def export_delta_data(data: dict, output_path: str, window: int = 1):
    """Export delta data for external analysis/plotting."""
    per_frame = data['per_frame']
    metric_names = data['meta']['metric_names']
    
    if 'frame_number' in per_frame[0]:
        frames = [f['frame_number'] for f in per_frame]
    else:
        frames = [f['index'] for f in per_frame]
    
    # Compute deltas for all metrics
    delta_data = {
        'frames': frames[window:],  # Frames after first delta
        'window': window,
        'deltas': {},
        'rolling_stats': {},
    }
    
    for metric in metric_names:
        values = [f[metric] for f in per_frame]
        deltas = compute_rolling_deltas(values, window=window)
        delta_data['deltas'][metric] = deltas
        
        # Rolling stats (mean, std over 10-frame windows)
        rolling_window = 10
        if len(deltas) >= rolling_window:
            rolling_mean = []
            rolling_std = []
            for i in range(len(deltas) - rolling_window + 1):
                window_deltas = deltas[i:i+rolling_window]
                rolling_mean.append(float(np.mean(window_deltas)))
                rolling_std.append(float(np.std(window_deltas)))
            delta_data['rolling_stats'][metric] = {
                'frames': frames[window + rolling_window - 1:window + rolling_window - 1 + len(rolling_mean)],
                'rolling_delta_mean': rolling_mean,
                'rolling_delta_std': rolling_std,
            }
    
    with open(output_path, 'w') as f:
        json.dump(delta_data, f, indent=2)
    
    print(f"\nDelta data exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze rate of change in metrics")
    parser.add_argument('metrics_json', type=str, help='Path to metrics JSON file')
    parser.add_argument('--collapse-range', '-c', type=str, required=True,
                        help='Known collapse range (e.g., "60,70")')
    parser.add_argument('--window', '-w', type=int, default=5,
                        help='Window size for rolling statistics')
    parser.add_argument('--export-deltas', '-e', type=str, default=None,
                        help='Export delta data to JSON file')
    parser.add_argument('--windowed-timeline', action='store_true',
                        help='Show windowed statistics timeline')
    
    args = parser.parse_args()
    
    with open(args.metrics_json) as f:
        data = json.load(f)
    
    # Parse collapse range
    parts = args.collapse_range.split(',')
    collapse_start, collapse_end = int(parts[0]), int(parts[1])
    
    print(f"Loaded {data['meta']['num_images']} frames")
    print(f"Metrics: {len(data['meta']['metric_names'])}")
    
    # Analyze collapse region
    results = analyze_collapse_region(data, collapse_start, collapse_end, args.window)
    
    # Print detailed analysis
    print_detailed_analysis(results, collapse_start, collapse_end)
    
    # Find similar regions
    find_and_print_similar_regions(data, results, collapse_start, collapse_end)
    
    # Windowed timeline if requested
    if args.windowed_timeline:
        compute_windowed_stats_timeline(data, window_size=args.window * 2)
    
    # Export deltas if requested
    if args.export_deltas:
        export_delta_data(data, args.export_deltas, window=args.window)
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"""
Key observations from collapse region {collapse_start}-{collapse_end}:

The delta (rate of change) metrics reveal:
- How rapidly each metric is changing during collapse
- The volatility (delta std) indicates instability
- Trend shows directional movement

Look for regions where multiple metrics show similar delta patterns to the collapse zone.
These are potential undetected collapse events or collapse precursors.
""")


if __name__ == "__main__":
    main()

