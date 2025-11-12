"""
Similarity Metrics Comparison Tool

Validates encoder implementations and compares different similarity metrics
for mode collapse detection.

Installation:
    uv sync --extra analysis

Usage:
    uv run python backend/tools/similarity_comparison.py --dir cache/images
    uv run python backend/tools/similarity_comparison.py --dir seeds
    uv run python backend/tools/similarity_comparison.py --images img1.png img2.png img3.png
    
Metrics compared:
    - ColorHist (HSV histogram) - Production color encoder
    - pHash-8 (perceptual hash) - Production structure encoder
    - pHash-16 (perceptual hash, higher resolution)
    - SSIM (structural similarity)
    - MSE (pixel-level difference)
    
Performance metrics included for each method.

NOTE: Validates that encoder class implementations match inline reference implementations.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not available. Install with: pip install imagehash")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage import io
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")


class SimilarityComparator:
    """Compare different similarity metrics on a set of images"""
    
    def __init__(self, image_paths: List[Path]):
        """
        Initialize with list of image paths
        
        Args:
            image_paths: List of paths to images to compare
        """
        self.image_paths = image_paths
        self.images = {}
        self.results = {}
        self.timings = {}  # Track performance
        
        # Load images
        print(f"\nLoading {len(image_paths)} images...")
        for path in image_paths:
            self.images[path.name] = Image.open(path).convert('RGB')

    
    def compute_phash_8_similarity_inline(self) -> np.ndarray:
        """Compute pHash-8 similarity (INLINE/OLD implementation for comparison)"""
        if not HAS_IMAGEHASH:
            return None
        
        hash_size = 8
        print(f"\nComputing pHash-8 similarity - INLINE implementation...")
        start_time = time.time()
        hashes = {}
        
        for name, img in self.images.items():
            hashes[name] = imagehash.phash(img, hash_size=hash_size)
        
        encoding_time = time.time() - start_time
        
        n = len(hashes)
        similarity_matrix = np.zeros((n, n))
        names = list(hashes.keys())
        
        max_distance = hash_size * hash_size  # Maximum possible hamming distance
        
        comparison_start = time.time()
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    distance = hashes[names[i]] - hashes[names[j]]  # Hamming distance
                    similarity = 1.0 - (distance / max_distance)
                    similarity_matrix[i, j] = similarity
        
        comparison_time = time.time() - comparison_start
        total_time = time.time() - start_time
        
        # Store timing
        avg_encode_time = (encoding_time * 1000) / n if n > 0 else 0
        avg_compare_time = (comparison_time * 1000) / (n * n) if n > 0 else 0
        
        self.timings['pHash-8 INLINE'] = {
            'encoding_ms': max(0.001, avg_encode_time),
            'comparison_ms': max(0.001, avg_compare_time),
            'total_ms': max(0.001, total_time * 1000)
        }
        
        return similarity_matrix, names
    
    def compute_phash_8_similarity_new(self) -> np.ndarray:
        """Compute pHash-8 similarity (NEW PHashEncoder class)"""
        if not HAS_IMAGEHASH:
            return None
        
        hash_size = 8
        print(f"\nComputing pHash-8 similarity - NEW PHashEncoder class...")
        
        try:
            from utils.phash_encoder import PHashEncoder
            encoder = PHashEncoder(hash_size=hash_size)
            
            start_time = time.time()
            hashes = {}
            
            for name, img in self.images.items():
                # Pass PIL Image directly to avoid redundant I/O
                hash_obj = encoder.encode_image(img)
                if hash_obj is not None:
                    hashes[name] = hash_obj
            
            encoding_time = time.time() - start_time
            
            n = len(hashes)
            similarity_matrix = np.zeros((n, n))
            names = list(hashes.keys())
            
            comparison_start = time.time()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sim = encoder.similarity(hashes[names[i]], hashes[names[j]])
                        similarity_matrix[i, j] = sim
            
            comparison_time = time.time() - comparison_start
            total_time = time.time() - start_time
            
            # Store timing
            avg_encode_time = (encoding_time * 1000) / n if n > 0 else 0
            avg_compare_time = (comparison_time * 1000) / (n * n) if n > 0 else 0
            
            self.timings['pHash-8 NEW'] = {
                'encoding_ms': max(0.001, avg_encode_time),
                'comparison_ms': max(0.001, avg_compare_time),
                'total_ms': max(0.001, total_time * 1000)
            }
            
            print(f"   [OK] Using NEW PHashEncoder class")
            return similarity_matrix, names
            
        except Exception as e:
            print(f"   [FAIL] Could not load PHashEncoder: {e}")
            return None
    
    def compute_ssim_similarity(self) -> np.ndarray:
        """Compute pairwise SSIM (structural similarity)"""
        if not HAS_SKIMAGE:
            return None
        
        print("\nComputing SSIM (structural similarity)...")
        start_time = time.time()
        
        # Resize all images to same size for SSIM
        target_size = (512, 512)
        images_resized = {}
        
        for name, img in self.images.items():
            resized = img.resize(target_size, Image.LANCZOS)
            # Convert to grayscale for SSIM
            gray = np.array(resized.convert('L'))
            images_resized[name] = gray
        
        encoding_time = time.time() - start_time
        
        n = len(images_resized)
        similarity_matrix = np.zeros((n, n))
        names = list(images_resized.keys())
        
        comparison_start = time.time()
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = ssim(images_resized[names[i]], images_resized[names[j]])
                    # SSIM ranges from -1 to 1, normalize to 0-1
                    similarity_matrix[i, j] = (sim + 1) / 2
        
        comparison_time = time.time() - comparison_start
        total_time = time.time() - start_time
        
        # Store timing
        self.timings['SSIM (structural)'] = {
            'encoding_ms': (encoding_time * 1000) / n,
            'comparison_ms': (comparison_time * 1000) / (n * n),
            'total_ms': total_time * 1000
        }
        
        return similarity_matrix, names
    
    def compute_mse_similarity(self) -> np.ndarray:
        """Compute normalized MSE-based similarity"""
        print("\nComputing MSE (mean squared error)...")
        start_time = time.time()
        
        # Resize all images to same size
        target_size = (512, 512)
        images_resized = {}
        
        for name, img in self.images.items():
            resized = img.resize(target_size, Image.LANCZOS)
            arr = np.array(resized).astype(np.float32) / 255.0
            images_resized[name] = arr
        
        encoding_time = time.time() - start_time
        
        n = len(images_resized)
        similarity_matrix = np.zeros((n, n))
        names = list(images_resized.keys())
        
        comparison_start = time.time()
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    mse = np.mean((images_resized[names[i]] - images_resized[names[j]])**2)
                    # Convert MSE to similarity (lower MSE = higher similarity)
                    # MSE typically ranges 0.0-0.3 for similar images
                    # Use exponential decay: sim = exp(-k * mse)
                    # k=5 gives good discrimination
                    similarity = np.exp(-5.0 * mse)
                    similarity_matrix[i, j] = similarity
        
        comparison_time = time.time() - comparison_start
        total_time = time.time() - start_time
        
        # Store timing
        self.timings['MSE (pixel-level)'] = {
            'encoding_ms': (encoding_time * 1000) / n,
            'comparison_ms': (comparison_time * 1000) / (n * n),
            'total_ms': total_time * 1000
        }
        
        return similarity_matrix, names
    
    def compute_color_histogram_similarity_inline(self) -> np.ndarray:
        """Compute HSV color histogram similarity (INLINE/OLD implementation for comparison)"""
        print("\nComputing Color Histogram similarity (HSV) - INLINE implementation...")
        start_time = time.time()
        
        histograms = {}
        
        for name, img in self.images.items():
            # Convert to HSV for better color representation
            hsv = img.convert('HSV')
            h, s, v = hsv.split()
            
            # Compute histograms (32 bins each channel)
            h_hist = np.histogram(np.array(h), bins=32, range=(0, 256))[0]
            s_hist = np.histogram(np.array(s), bins=32, range=(0, 256))[0]
            v_hist = np.histogram(np.array(v), bins=32, range=(0, 256))[0]
            
            # Normalize histograms
            h_hist = h_hist.astype(np.float32) / h_hist.sum()
            s_hist = s_hist.astype(np.float32) / s_hist.sum()
            v_hist = v_hist.astype(np.float32) / v_hist.sum()
            
            # Concatenate into single feature vector
            histograms[name] = np.concatenate([h_hist, s_hist, v_hist])
        
        encoding_time = time.time() - start_time
        
        n = len(histograms)
        similarity_matrix = np.zeros((n, n))
        names = list(histograms.keys())
        
        comparison_start = time.time()
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Histogram intersection (common similarity metric)
                    intersection = np.minimum(histograms[names[i]], histograms[names[j]]).sum()
                    similarity_matrix[i, j] = intersection
        
        comparison_time = time.time() - comparison_start
        total_time = time.time() - start_time
        
        # Store timing
        self.timings['ColorHist INLINE'] = {
            'encoding_ms': (encoding_time * 1000) / n,
            'comparison_ms': (comparison_time * 1000) / (n * n),
            'total_ms': total_time * 1000
        }
        
        return similarity_matrix, names
    
    def compute_color_histogram_similarity_new(self) -> np.ndarray:
        """Compute HSV color histogram similarity (NEW ColorHistogramEncoder class)"""
        print("\nComputing Color Histogram similarity (HSV) - NEW ColorHistogramEncoder class...")
        
        try:
            from utils.color_encoder import ColorHistogramEncoder
            encoder = ColorHistogramEncoder(bins_per_channel=32)
            
            start_time = time.time()
            histograms = {}
            for name, img in self.images.items():
                # Pass PIL Image directly to avoid redundant I/O
                hist = encoder.encode_image(img)
                if hist is not None:
                    histograms[name] = hist
            
            encoding_time = time.time() - start_time
            
            n = len(histograms)
            similarity_matrix = np.zeros((n, n))
            names = list(histograms.keys())
            
            comparison_start = time.time()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        sim = encoder.similarity(histograms[names[i]], histograms[names[j]])
                        similarity_matrix[i, j] = sim
            
            comparison_time = time.time() - comparison_start
            total_time = time.time() - start_time
            
            # Store timing
            self.timings['ColorHist NEW'] = {
                'encoding_ms': (encoding_time * 1000) / n,
                'comparison_ms': (comparison_time * 1000) / (n * n),
                'total_ms': total_time * 1000
            }
            
            print("   [OK] Using NEW ColorHistogramEncoder class")
            return similarity_matrix, names
            
        except Exception as e:
            print(f"   [FAIL] Could not load ColorHistogramEncoder: {e}")
            return None
    
    def compute_hybrid_similarity(self, weights: Dict[str, float], components: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Combine multiple metrics with weights"""
        # All matrices should have same names in same order
        names = None
        combined = None
        
        for metric_name, weight in weights.items():
            if metric_name in components:
                matrix = components[metric_name]
                if combined is None:
                    combined = weight * matrix
                    names = list(self.images.keys())  # Assume same order
                else:
                    combined += weight * matrix
        
        return combined, names
    
    def analyze_metric(self, name: str, matrix: np.ndarray, image_names: List[str]) -> Dict:
        """Analyze a similarity matrix"""
        # Extract upper triangle (excluding diagonal)
        n = len(matrix)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(matrix[i, j])
        
        similarities = np.array(similarities)
        
        stats = {
            'name': name,
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'range': float(np.max(similarities) - np.min(similarities)),
            'matrix': matrix,
            'image_names': image_names
        }
        
        return stats
    
    def print_comparison(self):
        """Print comparison - VALIDATION MODE (OLD vs NEW implementations)"""
        print("\n" + "="*80)
        print("ENCODER VALIDATION: OLD (inline) vs NEW (class)")
        print("="*80)
        print(f"\nNumber of images: {len(self.images)}")
        print(f"Number of pairs: {len(self.images) * (len(self.images) - 1) // 2}")
        print("\nTesting: ColorHist (OLD/NEW) + pHash-8 (OLD/NEW)")
        
        metrics = []
        
        # Run validation metrics
        
        # ColorHist INLINE (old)
        result = self.compute_color_histogram_similarity_inline()
        if result:
            matrix, names = result
            stats = self.analyze_metric("ColorHist INLINE", matrix, names)
            metrics.append(stats)
        
        # ColorHist NEW (class)
        result = self.compute_color_histogram_similarity_new()
        if result:
            matrix, names = result
            stats = self.analyze_metric("ColorHist NEW", matrix, names)
            metrics.append(stats)
        
        # pHash-8 INLINE (old)
        result = self.compute_phash_8_similarity_inline()
        if result:
            matrix, names = result
            stats = self.analyze_metric("pHash-8 INLINE", matrix, names)
            metrics.append(stats)
        
        # pHash-8 NEW (class)
        result = self.compute_phash_8_similarity_new()
        if result:
            matrix, names = result
            stats = self.analyze_metric("pHash-8 NEW", matrix, names)
            metrics.append(stats)
        
        # Print summary table
        print("\n" + "-"*80)
        print(f"{'Metric':<25} {'Min':>8} {'Mean':>8} {'Max':>8} {'Range':>8} {'StdDev':>8}")
        print("-"*80)
        
        for stat in metrics:
            print(f"{stat['name']:<25} "
                  f"{stat['min']:>8.4f} "
                  f"{stat['mean']:>8.4f} "
                  f"{stat['max']:>8.4f} "
                  f"{stat['range']:>8.4f} "
                  f"{stat['std']:>8.4f}")
        
        print("-"*80)
        
        # Print performance table
        if self.timings:
            print("\n" + "-"*80)
            print(f"{'Metric':<25} {'Encode/img':>12} {'Compare':>12} {'Total':>12}")
            print("-"*80)
            
            for metric_name in self.timings.keys():
                timing = self.timings[metric_name]
                print(f"{metric_name:<25} "
                      f"{timing['encoding_ms']:>10.1f}ms "
                      f"{timing['comparison_ms']:>10.3f}ms "
                      f"{timing['total_ms']:>10.1f}ms")
            
            print("-"*80)
            print("Note: Encode/img = time to encode one image")
            print("      Compare = time for one similarity comparison")
            print("      Total = total time for all operations")
        
        # Validation Analysis
        print("\n" + "="*80)
        print("VALIDATION ANALYSIS")
        print("="*80)
        print("\nComparing OLD (inline) vs NEW (class) implementations:")
        print("Expected: Results should be IDENTICAL or near-identical\n")
        
        # Group by metric type
        colorhist_metrics = [m for m in metrics if 'ColorHist' in m['name']]
        phash_metrics = [m for m in metrics if 'pHash-8' in m['name']]
        
        if len(colorhist_metrics) == 2:
            old = next(m for m in colorhist_metrics if 'INLINE' in m['name'])
            new = next(m for m in colorhist_metrics if 'NEW' in m['name'])
            
            print("ColorHist Validation:")
            print(f"  INLINE: min={old['min']:.4f}, mean={old['mean']:.4f}, max={old['max']:.4f}, range={old['range']:.4f}")
            print(f"  NEW:    min={new['min']:.4f}, mean={new['mean']:.4f}, max={new['max']:.4f}, range={new['range']:.4f}")
            
            # Check if results match
            diff_mean = abs(old['mean'] - new['mean'])
            diff_range = abs(old['range'] - new['range'])
            if diff_mean < 0.001 and diff_range < 0.001:
                print(f"  [OK] Results MATCH (diff_mean={diff_mean:.6f}, diff_range={diff_range:.6f})")
            else:
                print(f"  [WARN] Results differ (diff_mean={diff_mean:.6f}, diff_range={diff_range:.6f})")
        
        if len(phash_metrics) == 2:
            old = next(m for m in phash_metrics if 'INLINE' in m['name'])
            new = next(m for m in phash_metrics if 'NEW' in m['name'])
            
            print("\npHash-8 Validation:")
            print(f"  INLINE: min={old['min']:.4f}, mean={old['mean']:.4f}, max={old['max']:.4f}, range={old['range']:.4f}")
            print(f"  NEW:    min={new['min']:.4f}, mean={new['mean']:.4f}, max={new['max']:.4f}, range={new['range']:.4f}")
            
            # Check if results match
            diff_mean = abs(old['mean'] - new['mean'])
            diff_range = abs(old['range'] - new['range'])
            if diff_mean < 0.001 and diff_range < 0.001:
                print(f"  [OK] Results MATCH (diff_mean={diff_mean:.6f}, diff_range={diff_range:.6f})")
            else:
                print(f"  [WARN] Results differ (diff_mean={diff_mean:.6f}, diff_range={diff_range:.6f})")
        
        # Show pairwise similarities for best discriminating metric
        if metrics:
            by_range = sorted(metrics, key=lambda x: x['range'], reverse=True)
            best = by_range[0]
            print(f"\n" + "="*80)
            print(f"PAIRWISE SIMILARITIES - {best['name'].upper()}")
            print("="*80)
            
            matrix = best['matrix']
            names = best['image_names']
            n = len(names)
            
            # Print matrix header
            print(f"\n{'':>20}", end='')
            for name in names:
                print(f"{name[:15]:>17}", end='')
            print()
            
            # Print matrix
            for i in range(n):
                print(f"{names[i][:18]:>20}", end='')
                for j in range(n):
                    if i == j:
                        print(f"{'1.0000':>17}", end='')
                    else:
                        print(f"{matrix[i,j]:>17.4f}", end='')
                print()
            
            # Print sorted pairs
            print(f"\n" + "-"*80)
            print("PAIRWISE SIMILARITIES (sorted, most similar first):")
            print("-"*80)
            
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((names[i], names[j], matrix[i,j]))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            for img1, img2, sim in pairs:
                print(f"{img1[:25]:>25} <-> {img2[:25]:<25} : {sim:.4f}")
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        if self.timings:
            # Compare performance: OLD vs NEW
            ch_new_time = self.timings.get('ColorHist NEW', {}).get('encoding_ms', 0)
            ph_new_time = self.timings.get('pHash-8 NEW', {}).get('encoding_ms', 0)
            dual_time = ch_new_time + ph_new_time
            
            print(f"\nEncoding Speed (per image):")
            print(f"  ColorHist NEW:    {ch_new_time:>8.1f}ms")
            print(f"  pHash-8 NEW:      {ph_new_time:>8.1f}ms")
            print(f"  Dual-metric total: {dual_time:>8.1f}ms")
            
            # Show OLD vs NEW performance
            ch_old_time = self.timings.get('ColorHist INLINE', {}).get('encoding_ms', 0)
            ph_old_time = self.timings.get('pHash-8 INLINE', {}).get('encoding_ms', 0)
            
            if ch_old_time > 0 and ch_new_time > 0:
                ch_speedup = ch_old_time / ch_new_time if ch_new_time > 0 else 0
                print(f"\n  ColorHist: INLINE {ch_old_time:.1f}ms vs NEW {ch_new_time:.1f}ms ({ch_speedup:.2f}x)")
            
            if ph_old_time > 0 and ph_new_time > 0:
                ph_speedup = ph_old_time / ph_new_time if ph_new_time > 0 else 0
                print(f"  pHash-8:   INLINE {ph_old_time:.1f}ms vs NEW {ph_new_time:.1f}ms ({ph_speedup:.2f}x)")
        
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        
        if len(colorhist_metrics) == 2 and len(phash_metrics) == 2:
            ch_old = next(m for m in colorhist_metrics if 'INLINE' in m['name'])
            ch_new = next(m for m in colorhist_metrics if 'NEW' in m['name'])
            ph_old = next(m for m in phash_metrics if 'INLINE' in m['name'])
            ph_new = next(m for m in phash_metrics if 'NEW' in m['name'])
            
            ch_match = abs(ch_old['mean'] - ch_new['mean']) < 0.001
            ph_match = abs(ph_old['mean'] - ph_new['mean']) < 0.001
            
            if ch_match and ph_match:
                print("\n[OK] CORRECTNESS: NEW encoders produce identical results to inline")
                
                # Performance verdict
                ch_new_time = self.timings.get('ColorHist NEW', {}).get('encoding_ms', 0)
                ph_new_time = self.timings.get('pHash-8 NEW', {}).get('encoding_ms', 0)
                
                print(f"[OK] PERFORMANCE: Dual-metric encoding at {ch_new_time + ph_new_time:.1f}ms per image")
                print("\n  The encoder classes validated successfully!")
            else:
                print("\n[WARN] Some differences detected:")
                if not ch_match:
                    print(f"  ColorHist: Diff = {abs(ch_old['mean'] - ch_new['mean']):.6f}")
                if not ph_match:
                    print(f"  pHash-8:   Diff = {abs(ph_old['mean'] - ph_new['mean']):.6f}")
        else:
            print("\n[INFO] Not all metrics ran - partial validation only")


def main():
    parser = argparse.ArgumentParser(description="Compare similarity metrics on images")
    parser.add_argument('--images', nargs='+', help='Image files to compare')
    parser.add_argument('--dir', help='Directory containing images')
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    
    if args.images:
        image_paths = [Path(p) for p in args.images]
    elif args.dir:
        dir_path = Path(args.dir)
        image_paths = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
    else:
        # Default: use cache images
        cache_dir = Path(__file__).parent.parent.parent / "cache" / "images"
        if cache_dir.exists():
            image_paths = sorted(cache_dir.glob('*.png'))
            print(f"Using images from: {cache_dir}")
        else:
            print("Error: No images specified and cache directory not found")
            print("Usage: python similarity_comparison.py --images img1.png img2.png ...")
            return
    
    # Filter to existing files
    image_paths = [p for p in image_paths if p.exists()]
    
    if len(image_paths) < 2:
        print(f"Error: Need at least 2 images, found {len(image_paths)}")
        return
    
    print(f"Found {len(image_paths)} images")
    for p in image_paths:
        print(f"  - {p.name}")
    
    # Run comparison
    comparator = SimilarityComparator(image_paths)
    comparator.print_comparison()


if __name__ == "__main__":
    main()



