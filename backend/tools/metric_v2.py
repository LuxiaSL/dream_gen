#!/usr/bin/env python3
"""
Intrinsic Image Metrics v2
==========================

Computes per-frame intrinsic metrics for image analysis. Outputs detailed
statistics suitable for plotting and external analysis.

Metrics computed:
  - Color: hue_entropy, saturation_mean/std, value_mean/std, unique_colors_q16, color_correlation_mean
  - Structure: edge_density, laplacian_variance, high_freq_ratio, local_contrast_mean, gradient_entropy
  - Chaos detection: gradient_direction_entropy, local_autocorrelation, block_coherence_variance, laplacian_edge_ratio

Usage:
    python metric_v2.py /path/to/images --output results.json
    python metric_v2.py /path/to/images --output results.json --profile  # with timing data
"""

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import time

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_intrinsic_metrics(img: Image.Image, profile: bool = False, 
                               fft_downsample: int = 1) -> Dict[str, float]:
    """
    Compute intrinsic metrics for a single image.
    
    Args:
        img: PIL Image
        profile: If True, return timing data in metrics['_timings']
        fft_downsample: Factor to downsample image for FFT (1=full, 2=half, etc.)
    
    Returns:
        Dictionary of metric name -> float value
    """
    timings = {} if profile else None
    
    def timed(name):
        class Timer:
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            def __exit__(self, *args):
                if timings is not None:
                    timings[name] = (time.perf_counter() - self.start) * 1000
        return Timer()
    
    with timed('image_conversion'):
        rgb = np.array(img.convert('RGB'))
        hsv = np.array(img.convert('HSV'))
        gray = np.array(img.convert('L')).astype(np.float32)
    
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    metrics = {}
    
    # === COLOR METRICS ===
    
    with timed('hue_entropy'):
        h_hist, _ = np.histogram(h.flatten(), bins=32, range=(0, 256))
        h_hist = h_hist / (h_hist.sum() + 1e-10)
        h_hist = h_hist[h_hist > 0]
        metrics['hue_entropy'] = float(-np.sum(h_hist * np.log2(h_hist + 1e-10)))
    
    with timed('saturation_stats'):
        metrics['saturation_mean'] = float(s.mean())
        metrics['saturation_std'] = float(s.std())
    
    with timed('value_stats'):
        metrics['value_mean'] = float(v.mean())
        metrics['value_std'] = float(v.std())
    
    with timed('unique_colors'):
        quantized = rgb // 16
        packed = (quantized[:,:,0].astype(np.int32) << 8) | \
                 (quantized[:,:,1].astype(np.int32) << 4) | \
                  quantized[:,:,2].astype(np.int32)
        metrics['unique_colors_q16'] = len(np.unique(packed))
    
    with timed('color_correlation'):
        r = rgb[:,:,0].flatten().astype(np.float64)
        g = rgb[:,:,1].flatten().astype(np.float64)
        b = rgb[:,:,2].flatten().astype(np.float64)
        corr_matrix = np.corrcoef([r, g, b])
        metrics['color_correlation_mean'] = float(
            (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
        )
    
    # === STRUCTURE METRICS ===
    
    with timed('edge_density'):
        gx = np.abs(gray[:, 1:] - gray[:, :-1])
        gy = np.abs(gray[1:, :] - gray[:-1, :])
        metrics['edge_density'] = float((gx.mean() + gy.mean()) / 2)
    
    with timed('laplacian_variance'):
        lap = (
            4 * gray[1:-1, 1:-1] -
            gray[:-2, 1:-1] -
            gray[2:, 1:-1] -
            gray[1:-1, :-2] -
            gray[1:-1, 2:]
        )
        metrics['laplacian_variance'] = float(lap.var())
    
    with timed('high_freq_ratio'):
        if fft_downsample > 1:
            gray_fft = gray[::fft_downsample, ::fft_downsample]
        else:
            gray_fft = gray
            
        f = np.fft.fft2(gray_fft)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        height, width = magnitude.shape
        cy, cx = height // 2, width // 2
        r = min(height, width) // 8
        y, x = np.ogrid[:height, :width]
        high_freq_mask = ((x - cx) ** 2 + (y - cy) ** 2) > r ** 2
        
        total_energy = magnitude.sum() + 1e-10
        metrics['high_freq_ratio'] = float(magnitude[high_freq_mask].sum() / total_energy)
    
    with timed('local_contrast'):
        if HAS_SCIPY:
            local_mean = uniform_filter(gray, size=16)
            local_sqr_mean = uniform_filter(gray ** 2, size=16)
            local_var = local_sqr_mean - local_mean ** 2
            metrics['local_contrast_mean'] = float(np.sqrt(np.maximum(local_var, 0)).mean())
        else:
            metrics['local_contrast_mean'] = float(gray.std())
    
    with timed('gradient_entropy'):
        grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
        grad_hist, _ = np.histogram(grad_mag.flatten(), bins=32, range=(0, grad_mag.max() + 1))
        grad_hist = grad_hist / (grad_hist.sum() + 1e-10)
        grad_hist = grad_hist[grad_hist > 0]
        metrics['gradient_entropy'] = float(-np.sum(grad_hist * np.log2(grad_hist + 1e-10)))
    
    # === CHAOS DETECTION METRICS ===
    
    with timed('gradient_direction_entropy'):
        gx_dir = (gray[:, 1:] - gray[:, :-1])[:-1, :]
        gy_dir = (gray[1:, :] - gray[:-1, :])[:, :-1]
        
        theta = np.arctan2(gy_dir, gx_dir)
        mag = np.sqrt(gx_dir ** 2 + gy_dir ** 2)
        
        num_bins = 16
        bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        dir_hist = np.zeros(num_bins, dtype=np.float64)
        
        for i in range(num_bins):
            mask = (theta >= bin_edges[i]) & (theta < bin_edges[i + 1])
            dir_hist[i] = mag[mask].sum()
        
        dir_hist = dir_hist / (dir_hist.sum() + 1e-10)
        dir_hist = dir_hist[dir_hist > 0]
        metrics['gradient_direction_entropy'] = float(-np.sum(dir_hist * np.log2(dir_hist + 1e-10)))
    
    with timed('local_autocorrelation'):
        correlations = []
        
        # (1,0), (0,1), (1,1), (2,0), (0,2) offsets
        for shift_fn in [
            lambda g: (g[:, :-1], g[:, 1:]),
            lambda g: (g[:-1, :], g[1:, :]),
            lambda g: (g[:-1, :-1], g[1:, 1:]),
            lambda g: (g[:, :-2], g[:, 2:]),
            lambda g: (g[:-2, :], g[2:, :]),
        ]:
            img1, img2 = shift_fn(gray)
            corr = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
            correlations.append(corr)
        
        metrics['local_autocorrelation'] = float(np.mean(correlations))
    
    with timed('block_coherence_variance'):
        block_size = 32
        gx_blk = np.abs(gray[:, 1:] - gray[:, :-1])
        gy_blk = np.abs(gray[1:, :] - gray[:-1, :])
        grad_magnitude = np.sqrt(gx_blk[:-1, :] ** 2 + gy_blk[:, :-1] ** 2)
        
        gh, gw = grad_magnitude.shape
        n_blocks_y = gh // block_size
        n_blocks_x = gw // block_size
        
        if n_blocks_y > 0 and n_blocks_x > 0:
            block_means = []
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    block = grad_magnitude[
                        by * block_size : (by + 1) * block_size,
                        bx * block_size : (bx + 1) * block_size
                    ]
                    block_means.append(block.mean())
            metrics['block_coherence_variance'] = float(np.var(block_means))
        else:
            metrics['block_coherence_variance'] = 0.0
    
    with timed('laplacian_edge_ratio'):
        lap = (
            4 * gray[1:-1, 1:-1] -
            gray[:-2, 1:-1] -
            gray[2:, 1:-1] -
            gray[1:-1, :-2] -
            gray[1:-1, 2:]
        )
        lap_abs = np.abs(lap)
        
        gx_lap = np.abs(gray[1:-1, 2:] - gray[1:-1, :-2]) / 2
        gy_lap = np.abs(gray[2:, 1:-1] - gray[:-2, 1:-1]) / 2
        grad_mag_lap = np.sqrt(gx_lap ** 2 + gy_lap ** 2)
        
        threshold = np.median(grad_mag_lap) * 1.5
        edge_mask = grad_mag_lap > threshold
        
        lap_at_edges = lap_abs[edge_mask].sum()
        lap_total = lap_abs.sum() + 1e-10
        
        metrics['laplacian_edge_ratio'] = float(lap_at_edges / lap_total)
    
    if profile and timings:
        metrics['_timings'] = timings
    
    return metrics


def extract_template(filename: str) -> Optional[str]:
    """Extract template name from filename if present."""
    templates = [
        'abstract_field', 'atmospheric_depth', 'environmental', 'liminal',
        'material_collision', 'material_study', 'minimal_object', 'process_state',
        'ruin_state', 'specimen', 'temporal_diptych', 'textural_macro'
    ]
    stem = Path(filename).stem.lower()
    for t in templates:
        if t in stem:
            return t
    return None


def extract_category(filename: str) -> str:
    """Extract category from filename prefix."""
    stem = Path(filename).stem.lower()
    
    if stem.startswith('broad_'):
        return 'broad'
    elif stem.startswith('deep_'):
        return 'deep'
    elif stem.startswith('intervention_'):
        return 'intervention'
    else:
        return 'unknown'


def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename like keyframe_064.png or frame_00001.png."""
    import re
    stem = Path(filename).stem
    match = re.search(r'(\d+)$', stem)
    if match:
        return int(match.group(1))
    return None


@dataclass
class MetricStats:
    count: int
    mean: float
    stdev: float
    min_val: float
    max_val: float
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float


def compute_stats(values: List[float]) -> MetricStats:
    arr = np.array(values)
    return MetricStats(
        count=len(values),
        mean=float(arr.mean()),
        stdev=float(arr.std()),
        min_val=float(arr.min()),
        max_val=float(arr.max()),
        p5=float(np.percentile(arr, 5)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
    )


def main():
    parser = argparse.ArgumentParser(description="Compute intrinsic image metrics")
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', '-o', type=str, default='intrinsic_metrics.json')
    parser.add_argument('--limit', '-n', type=int, default=None, help='Limit number of images')
    parser.add_argument('--profile', '-p', action='store_true', help='Enable per-metric timing')
    parser.add_argument('--fft-downsample', type=int, default=1, 
                        help='Downsample factor for FFT (1=full, 2=half)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        list(image_dir.glob('*.png')) + 
        list(image_dir.glob('*.jpg'))
    )
    
    if args.limit:
        image_paths = image_paths[:args.limit]
    
    if not args.quiet:
        print(f"Processing {len(image_paths)} images...")
    
    all_metrics: List[Dict] = []
    template_metrics: Dict[str, List[Dict]] = defaultdict(list)
    category_metrics: Dict[str, List[Dict]] = defaultdict(list)
    all_timings: List[Dict] = []
    
    start_time = time.time()
    
    for i, path in enumerate(image_paths):
        if not args.quiet and i % 100 == 0 and i > 0:
            print(f"  {i}/{len(image_paths)}...")
        
        try:
            img = Image.open(path).convert('RGB')
            metrics = compute_intrinsic_metrics(img, profile=args.profile, 
                                                fft_downsample=args.fft_downsample)
            
            if '_timings' in metrics:
                all_timings.append(metrics.pop('_timings'))
            
            metrics['filename'] = path.name
            metrics['index'] = i
            
            frame_num = extract_frame_number(path.name)
            if frame_num is not None:
                metrics['frame_number'] = frame_num
            
            template = extract_template(path.name)
            if template:
                metrics['template'] = template
                template_metrics[template].append(metrics)
            
            category = extract_category(path.name)
            metrics['category'] = category
            category_metrics[category].append(metrics)
            
            all_metrics.append(metrics)
            
        except Exception as e:
            if not args.quiet:
                print(f"Error on {path.name}: {e}")
    
    elapsed = time.time() - start_time
    
    if not args.quiet:
        print(f"Processed {len(all_metrics)} images in {elapsed:.1f}s ({elapsed/len(all_metrics)*1000:.1f}ms/image)")
    
    # Compute statistics
    metric_names = [k for k in all_metrics[0].keys() 
                    if k not in ('filename', 'index', 'template', 'category', 'frame_number')]
    
    # Global stats
    global_stats = {}
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        global_stats[name] = asdict(compute_stats(values))
    
    # Category stats
    category_stats = {}
    for category, cat_metrics in category_metrics.items():
        category_stats[category] = {'count': len(cat_metrics)}
        for name in metric_names:
            values = [m[name] for m in cat_metrics]
            if len(values) >= 2:
                category_stats[category][name] = asdict(compute_stats(values))
            else:
                category_stats[category][name] = {'count': len(values), 'mean': values[0] if values else 0}
    
    # Template stats
    template_stats = {}
    for template, tmpl_metrics in template_metrics.items():
        template_stats[template] = {'count': len(tmpl_metrics)}
        for name in metric_names:
            values = [m[name] for m in tmpl_metrics]
            if len(values) >= 2:
                template_stats[template][name] = asdict(compute_stats(values))
            else:
                template_stats[template][name] = {'count': len(values), 'mean': values[0] if values else 0}
    
    # Temporal stats (frame-to-frame deltas)
    temporal_stats = {}
    if len(all_metrics) > 1:
        consecutive_deltas = defaultdict(list)
        for i in range(1, len(all_metrics)):
            for name in metric_names:
                delta = all_metrics[i][name] - all_metrics[i-1][name]
                consecutive_deltas[name].append(delta)
        
        for name in metric_names:
            deltas = consecutive_deltas[name]
            abs_deltas = [abs(d) for d in deltas]
            temporal_stats[name] = {
                'mean_delta': float(np.mean(deltas)),
                'stdev_delta': float(np.std(deltas)),
                'abs_mean_delta': float(np.mean(abs_deltas)),
                'abs_max_delta': float(np.max(abs_deltas)),
            }
    
    # Performance stats
    performance_stats = {}
    if all_timings:
        for key in all_timings[0].keys():
            values = [t[key] for t in all_timings]
            performance_stats[key] = {
                'mean_ms': float(np.mean(values)),
                'stdev_ms': float(np.std(values)),
                'min_ms': float(np.min(values)),
                'max_ms': float(np.max(values)),
            }
    
    # Output
    output = {
        'meta': {
            'num_images': len(all_metrics),
            'num_templates': len(template_metrics),
            'num_categories': len([c for c in category_metrics if c != 'unknown']),
            'categories': {cat: len(imgs) for cat, imgs in category_metrics.items()},
            'templates': list(template_metrics.keys()),
            'metric_names': metric_names,
            'elapsed_seconds': elapsed,
            'ms_per_image': elapsed / len(all_metrics) * 1000 if all_metrics else 0,
        },
        'global_stats': global_stats,
        'category_stats': category_stats,
        'template_stats': template_stats,
        'temporal_stats': temporal_stats,
        'performance': performance_stats if performance_stats else None,
        'per_frame': all_metrics,
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    if not args.quiet:
        print(f"Results saved to {output_path}")
        print(f"\nMetrics computed: {', '.join(metric_names)}")


if __name__ == "__main__":
    if not HAS_SCIPY:
        print("Warning: scipy not found, local_contrast will use fallback.")
    main()
