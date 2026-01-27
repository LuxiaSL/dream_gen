#!/usr/bin/env python3
"""
Similarity Metrics Benchmark Suite
==================================

Comprehensive benchmark comparing different perceptual hash and similarity
algorithms for mode collapse detection in dream_gen.

PROBLEM STATEMENT
-----------------
pHash-8 is showing noise-like behavior where consecutive frames have 
similarity ~0.836 with high variance (0.44-1.0 range), making it hard
to discriminate between actually-similar and different frames.

METRICS TESTED
--------------
Hash-based (fast, ~2-5ms/image):
- pHash-8:  DCT-based, 64-bit (current production)
- pHash-16: DCT-based, 256-bit (higher resolution)
- dHash-8:  Difference hash, 64-bit (gradient-based)
- dHash-16: Difference hash, 256-bit
- wHash-8:  Wavelet hash, 64-bit (DWT-based)
- wHash-16: Wavelet hash, 256-bit

Color-based (fast, ~5-40ms/image):
- ColorHist: HSV histogram intersection (current production)

Structural (slower, ~50-200ms/image):
- SSIM: Structural Similarity Index (perceptual quality metric)

ANALYSIS MODES
--------------
1. PAIRWISE: All-pairs similarity matrix analysis
   - Distribution stats (mean, std, percentiles)
   - Discrimination range (min-max spread)

2. TEMPORAL: Consecutive frame similarity
   - Measures "noise floor" - how similar adjacent frames are
   - Key for detecting actual vs spurious changes

3. CATEGORICAL: Within vs across template groups
   - Tests if metrics can distinguish different aesthetic styles
   - Computes discrimination ratio (across/within similarity)

4. PERFORMANCE: Encoding and comparison speed
   - ms/image for encoding
   - µs/comparison for similarity

Usage:
------
    # Run on keyframe directory with template grouping
    uv run python backend/tools/metric_benchmark.py /path/to/keyframes --output results.json
    
    # Run on calibration data
    uv run python backend/tools/metric_benchmark.py /workspace/calibration --output calibration_metrics.json
    
    # Quick test (first 50 images only)
    uv run python backend/tools/metric_benchmark.py /path/to/keyframes --limit 50
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not available. Install with: pip install imagehash")

try:
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. SSIM will be unavailable.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENCODER IMPLEMENTATIONS
# =============================================================================

class BaseEncoder:
    """Base class for similarity encoders"""
    name: str = "base"
    
    def encode(self, image: Image.Image) -> Any:
        """Encode image to embedding"""
        raise NotImplementedError
    
    def similarity(self, emb1: Any, emb2: Any) -> float:
        """Compute similarity between two embeddings (0-1, higher = more similar)"""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class PHashEncoder(BaseEncoder):
    """Perceptual hash using DCT"""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.name = f"phash-{hash_size}"
    
    def encode(self, image: Image.Image) -> 'imagehash.ImageHash':
        return imagehash.phash(image.convert('RGB'), hash_size=self.hash_size)
    
    def similarity(self, emb1, emb2) -> float:
        distance = emb1 - emb2  # Hamming distance
        max_distance = emb1.hash.size
        return 1.0 - (distance / max_distance)


class DHashEncoder(BaseEncoder):
    """Difference hash based on gradient"""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.name = f"dhash-{hash_size}"
    
    def encode(self, image: Image.Image) -> 'imagehash.ImageHash':
        return imagehash.dhash(image.convert('RGB'), hash_size=self.hash_size)
    
    def similarity(self, emb1, emb2) -> float:
        distance = emb1 - emb2
        max_distance = emb1.hash.size
        return 1.0 - (distance / max_distance)


class WHashEncoder(BaseEncoder):
    """Wavelet hash using DWT"""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.name = f"whash-{hash_size}"
    
    def encode(self, image: Image.Image) -> 'imagehash.ImageHash':
        return imagehash.whash(image.convert('RGB'), hash_size=self.hash_size)
    
    def similarity(self, emb1, emb2) -> float:
        distance = emb1 - emb2
        max_distance = emb1.hash.size
        return 1.0 - (distance / max_distance)


class AHashEncoder(BaseEncoder):
    """Average hash (simple but fast baseline)"""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.name = f"ahash-{hash_size}"
    
    def encode(self, image: Image.Image) -> 'imagehash.ImageHash':
        return imagehash.average_hash(image.convert('RGB'), hash_size=self.hash_size)
    
    def similarity(self, emb1, emb2) -> float:
        distance = emb1 - emb2
        max_distance = emb1.hash.size
        return 1.0 - (distance / max_distance)


class ColorHistEncoder(BaseEncoder):
    """HSV color histogram (production encoder)"""
    
    def __init__(self, bins: int = 32):
        self.bins = bins
        self.name = f"colorhist-{bins}"
    
    def encode(self, image: Image.Image) -> np.ndarray:
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        
        h_hist = np.histogram(np.array(h), bins=self.bins, range=(0, 256))[0]
        s_hist = np.histogram(np.array(s), bins=self.bins, range=(0, 256))[0]
        v_hist = np.histogram(np.array(v), bins=self.bins, range=(0, 256))[0]
        
        h_hist = h_hist.astype(np.float32) / h_hist.sum()
        s_hist = s_hist.astype(np.float32) / s_hist.sum()
        v_hist = v_hist.astype(np.float32) / v_hist.sum()
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # Histogram intersection - returns ~0-3 range (one per channel)
        # Normalize to 0-1 by dividing by 3
        intersection = np.minimum(emb1, emb2).sum()
        return float(intersection / 3.0)


class SSIMEncoder(BaseEncoder):
    """Structural Similarity Index"""
    
    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self.name = f"ssim-{target_size}"
    
    def encode(self, image: Image.Image) -> np.ndarray:
        # Resize and convert to grayscale
        resized = image.resize((self.target_size, self.target_size), Image.LANCZOS)
        gray = np.array(resized.convert('L'))
        return gray
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        score = ssim_func(emb1, emb2)
        # SSIM ranges from -1 to 1, normalize to 0-1
        return float((score + 1) / 2)


class CropResistantHashEncoder(BaseEncoder):
    """Crop-resistant hash - handles shifted/cropped images better"""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self.name = f"crhash-{hash_size}"
    
    def encode(self, image: Image.Image) -> 'imagehash.ImageHash':
        return imagehash.crop_resistant_hash(image.convert('RGB'), hash_size=self.hash_size)
    
    def similarity(self, emb1, emb2) -> float:
        # crop_resistant_hash returns ImageMultiHash
        # Use its built-in matching
        distance = emb1 - emb2  # Uses best_match internally
        # Normalize - crop resistant hash can have larger distances
        max_distance = self.hash_size * self.hash_size
        return max(0.0, 1.0 - (distance / max_distance))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DistributionStats:
    """Statistical distribution of similarity values"""
    values: List[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def mean(self) -> float:
        return round(statistics.mean(self.values), 4) if self.values else 0.0
    
    @property
    def stdev(self) -> float:
        return round(statistics.stdev(self.values), 4) if len(self.values) > 1 else 0.0
    
    @property
    def median(self) -> float:
        return round(statistics.median(self.values), 4) if self.values else 0.0
    
    def percentile(self, p: int) -> float:
        return round(float(np.percentile(self.values, p)), 4) if self.values else 0.0
    
    @property
    def range(self) -> float:
        return round(max(self.values) - min(self.values), 4) if self.values else 0.0
    
    def to_dict(self) -> dict:
        if not self.values:
            return {'count': 0}
        return {
            'count': self.count,
            'min': round(min(self.values), 4),
            'max': round(max(self.values), 4),
            'mean': self.mean,
            'median': self.median,
            'stdev': self.stdev,
            'range': self.range,
            'p5': self.percentile(5),
            'p10': self.percentile(10),
            'p25': self.percentile(25),
            'p75': self.percentile(75),
            'p90': self.percentile(90),
            'p95': self.percentile(95),
        }


@dataclass  
class MetricResults:
    """Results for a single metric"""
    name: str
    
    # Performance
    encode_time_ms: float = 0.0
    compare_time_us: float = 0.0  # microseconds
    
    # All-pairs distribution
    all_pairs: DistributionStats = field(default_factory=DistributionStats)
    
    # Consecutive frames (temporal)
    consecutive: DistributionStats = field(default_factory=DistributionStats)
    
    # Within-template vs across-template
    within_template: DistributionStats = field(default_factory=DistributionStats)
    across_template: DistributionStats = field(default_factory=DistributionStats)
    discrimination_ratio: float = 0.0  # across_mean / within_mean (higher = better discrimination)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'performance': {
                'encode_ms': round(self.encode_time_ms, 3),
                'compare_us': round(self.compare_time_us, 3),
            },
            'all_pairs': self.all_pairs.to_dict(),
            'consecutive': self.consecutive.to_dict(),
            'within_template': self.within_template.to_dict(),
            'across_template': self.across_template.to_dict(),
            'discrimination_ratio': round(self.discrimination_ratio, 4),
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class MetricBenchmark:
    """
    Comprehensive metric benchmark suite
    """
    
    def __init__(self, image_paths: List[Path], template_groups: Optional[Dict[str, List[int]]] = None):
        """
        Initialize benchmark
        
        Args:
            image_paths: Sorted list of image paths
            template_groups: Optional mapping of template_id -> list of image indices
        """
        self.image_paths = image_paths
        self.template_groups = template_groups or {}
        
        # Load images
        logger.info(f"Loading {len(image_paths)} images...")
        self.images: List[Image.Image] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                self.images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                self.images.append(None)
        
        valid_count = sum(1 for img in self.images if img is not None)
        logger.info(f"Loaded {valid_count}/{len(image_paths)} images")
        
        # Initialize encoders
        self.encoders: List[BaseEncoder] = []
        self._init_encoders()
        
        # Results
        self.results: Dict[str, MetricResults] = {}
    
    def _init_encoders(self):
        """Initialize all encoder variants"""
        if HAS_IMAGEHASH:
            # Hash-based encoders
            self.encoders.extend([
                PHashEncoder(hash_size=8),
                PHashEncoder(hash_size=16),
                DHashEncoder(hash_size=8),
                DHashEncoder(hash_size=16),
                WHashEncoder(hash_size=8),
                WHashEncoder(hash_size=16),
                AHashEncoder(hash_size=8),
                AHashEncoder(hash_size=16),
            ])
            
            # Crop-resistant hash (slower, may not work well for our use case)
            # self.encoders.append(CropResistantHashEncoder(hash_size=8))
        
        # Color histogram
        self.encoders.append(ColorHistEncoder(bins=32))
        
        # SSIM (slower but high quality)
        if HAS_SKIMAGE:
            self.encoders.extend([
                SSIMEncoder(target_size=128),  # Faster
                SSIMEncoder(target_size=256),  # Higher quality
            ])
        
        logger.info(f"Initialized {len(self.encoders)} encoders: {[e.name for e in self.encoders]}")
    
    def run_benchmark(self) -> Dict[str, MetricResults]:
        """Run full benchmark suite on all encoders"""
        logger.info("=" * 70)
        logger.info("RUNNING METRIC BENCHMARK")
        logger.info("=" * 70)
        
        for encoder in self.encoders:
            logger.info(f"\nBenchmarking: {encoder.name}")
            self.results[encoder.name] = self._benchmark_encoder(encoder)
        
        return self.results
    
    def _benchmark_encoder(self, encoder: BaseEncoder) -> MetricResults:
        """Benchmark a single encoder"""
        result = MetricResults(name=encoder.name)
        
        # Step 1: Encode all images and measure time
        logger.info(f"  Encoding {len(self.images)} images...")
        embeddings = []
        encode_times = []
        
        for i, img in enumerate(self.images):
            if img is None:
                embeddings.append(None)
                continue
            
            try:
                start = time.perf_counter()
                emb = encoder.encode(img)
                encode_time = (time.perf_counter() - start) * 1000
                encode_times.append(encode_time)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"  Failed to encode image {i}: {e}")
                embeddings.append(None)
        
        result.encode_time_ms = statistics.mean(encode_times) if encode_times else 0.0
        logger.info(f"  Encode time: {result.encode_time_ms:.2f}ms/image")
        
        # Step 2: Compute all-pairs similarity and measure comparison time
        logger.info(f"  Computing pairwise similarities...")
        all_sims = []
        consecutive_sims = []
        compare_times = []
        
        n = len(embeddings)
        valid_pairs = 0
        
        for i in range(n):
            if embeddings[i] is None:
                continue
            
            for j in range(i + 1, n):
                if embeddings[j] is None:
                    continue
                
                try:
                    start = time.perf_counter()
                    sim = encoder.similarity(embeddings[i], embeddings[j])
                    compare_time = (time.perf_counter() - start) * 1_000_000  # microseconds
                    
                    all_sims.append(sim)
                    compare_times.append(compare_time)
                    valid_pairs += 1
                    
                    # Track consecutive frames
                    if j == i + 1:
                        consecutive_sims.append(sim)
                        
                except Exception as e:
                    logger.warning(f"  Failed to compare {i} vs {j}: {e}")
        
        result.all_pairs = DistributionStats(all_sims)
        result.consecutive = DistributionStats(consecutive_sims)
        result.compare_time_us = statistics.mean(compare_times) if compare_times else 0.0
        
        logger.info(f"  Compare time: {result.compare_time_us:.2f}µs/pair")
        logger.info(f"  All-pairs: mean={result.all_pairs.mean:.4f}, std={result.all_pairs.stdev:.4f}, range={result.all_pairs.range:.4f}")
        logger.info(f"  Consecutive: mean={result.consecutive.mean:.4f}, std={result.consecutive.stdev:.4f}")
        
        # Step 3: Within vs across template analysis
        if self.template_groups:
            within_sims = []
            across_sims = []
            
            # Build index -> template mapping
            idx_to_template = {}
            for template_id, indices in self.template_groups.items():
                for idx in indices:
                    idx_to_template[idx] = template_id
            
            for i in range(n):
                if embeddings[i] is None or i not in idx_to_template:
                    continue
                
                for j in range(i + 1, n):
                    if embeddings[j] is None or j not in idx_to_template:
                        continue
                    
                    try:
                        sim = encoder.similarity(embeddings[i], embeddings[j])
                        
                        if idx_to_template[i] == idx_to_template[j]:
                            within_sims.append(sim)
                        else:
                            across_sims.append(sim)
                    except:
                        pass
            
            result.within_template = DistributionStats(within_sims)
            result.across_template = DistributionStats(across_sims)
            
            # Discrimination ratio: lower across-template similarity = better discrimination
            # We want within > across, so ratio = within / across (higher = better)
            if result.across_template.mean > 0:
                result.discrimination_ratio = result.within_template.mean / result.across_template.mean
            
            logger.info(f"  Within-template: mean={result.within_template.mean:.4f}")
            logger.info(f"  Across-template: mean={result.across_template.mean:.4f}")
            logger.info(f"  Discrimination ratio: {result.discrimination_ratio:.4f}")
        
        return result
    
    def print_summary(self):
        """Print summary comparison table"""
        print("\n" + "=" * 100)
        print("METRIC BENCHMARK SUMMARY")
        print("=" * 100)
        
        # Performance table
        print("\n--- PERFORMANCE ---")
        print(f"{'Metric':<20} {'Encode (ms)':<15} {'Compare (µs)':<15}")
        print("-" * 50)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1].encode_time_ms):
            print(f"{name:<20} {result.encode_time_ms:>12.2f}   {result.compare_time_us:>12.2f}")
        
        # Discrimination table
        print("\n--- ALL-PAIRS DISTRIBUTION ---")
        print(f"{'Metric':<20} {'Mean':<10} {'StdDev':<10} {'Range':<10} {'P10':<10} {'P90':<10}")
        print("-" * 70)
        
        for name, result in sorted(self.results.items(), key=lambda x: -x[1].all_pairs.range):
            ap = result.all_pairs
            print(f"{name:<20} {ap.mean:>8.4f}   {ap.stdev:>8.4f}   {ap.range:>8.4f}   {ap.percentile(10):>8.4f}   {ap.percentile(90):>8.4f}")
        
        # Consecutive (temporal noise)
        print("\n--- CONSECUTIVE FRAME SIMILARITY (temporal noise floor) ---")
        print(f"{'Metric':<20} {'Mean':<10} {'StdDev':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)
        
        for name, result in sorted(self.results.items(), key=lambda x: x[1].consecutive.stdev):
            c = result.consecutive
            if c.count > 0:
                print(f"{name:<20} {c.mean:>8.4f}   {c.stdev:>8.4f}   {c.percentile(0) if c.count else 0:>8.4f}   {c.percentile(100) if c.count else 0:>8.4f}")
        
        # Template discrimination (if available)
        if any(r.within_template.count > 0 for r in self.results.values()):
            print("\n--- TEMPLATE DISCRIMINATION (within/across) ---")
            print(f"{'Metric':<20} {'Within':<10} {'Across':<10} {'Ratio':<10} {'Assessment'}")
            print("-" * 70)
            
            for name, result in sorted(self.results.items(), key=lambda x: -x[1].discrimination_ratio):
                if result.within_template.count > 0:
                    ratio = result.discrimination_ratio
                    assessment = "🟢 Excellent" if ratio > 1.3 else "🟡 Good" if ratio > 1.1 else "🔴 Poor"
                    print(f"{name:<20} {result.within_template.mean:>8.4f}   {result.across_template.mean:>8.4f}   {ratio:>8.4f}   {assessment}")
        
        # Recommendations
        print("\n--- RECOMMENDATIONS ---")
        
        # Best overall discriminator
        best_range = max(self.results.items(), key=lambda x: x[1].all_pairs.range)
        print(f"Best discrimination (range): {best_range[0]} (range={best_range[1].all_pairs.range:.4f})")
        
        # Most stable consecutive
        best_stable = min(self.results.items(), key=lambda x: x[1].consecutive.stdev if x[1].consecutive.count > 0 else float('inf'))
        print(f"Most stable temporal: {best_stable[0]} (consecutive stdev={best_stable[1].consecutive.stdev:.4f})")
        
        # Fastest encoder
        fastest = min(self.results.items(), key=lambda x: x[1].encode_time_ms)
        print(f"Fastest encoder: {fastest[0]} ({fastest[1].encode_time_ms:.2f}ms)")
        
        # Best template discrimination
        if any(r.discrimination_ratio > 0 for r in self.results.values()):
            best_disc = max(self.results.items(), key=lambda x: x[1].discrimination_ratio)
            print(f"Best template discrimination: {best_disc[0]} (ratio={best_disc[1].discrimination_ratio:.4f})")
        
        print("\n" + "=" * 100)
    
    def save_results(self, output_path: Path):
        """Save results to JSON"""
        output = {
            'meta': {
                'num_images': len(self.images),
                'valid_images': sum(1 for img in self.images if img is not None),
                'num_templates': len(self.template_groups),
                'encoders': [e.name for e in self.encoders],
            },
            'results': {name: result.to_dict() for name, result in self.results.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def extract_template_from_filename(filename: str) -> Optional[str]:
    """
    Extract template ID from filename patterns
    
    Patterns:
    - broad_0001_material_study.png -> material_study
    - deep_0001_atmospheric_depth.png -> atmospheric_depth
    - keyframe_0001.png -> None (unknown)
    - calibration_broad_0001_abstract_field.png -> abstract_field
    """
    # Known templates
    templates = [
        'abstract_field', 'atmospheric_depth', 'environmental', 'liminal',
        'material_collision', 'material_study', 'minimal_object', 'process_state',
        'ruin_state', 'specimen', 'temporal_diptych', 'textural_macro'
    ]
    
    # Check for template in filename
    stem = Path(filename).stem.lower()
    
    for template in templates:
        if template in stem:
            return template
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark similarity metrics for mode collapse detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on calibration keyframes
    %(prog)s /workspace/calibration
    
    # Run on captured keyframes with limit
    %(prog)s ~/runpod-capture/output/keyframes --limit 100
    
    # Save results
    %(prog)s /path/to/images --output benchmark_results.json
"""
    )
    
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file')
    parser.add_argument('--limit', '-n', type=int, default=None, help='Limit number of images')
    parser.add_argument('--no-ssim', action='store_true', help='Skip SSIM (slower)')
    
    args = parser.parse_args()
    
    # Find images
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        return 1
    
    # Collect image paths
    image_paths = sorted(
        list(image_dir.glob('*.png')) + 
        list(image_dir.glob('*.jpg')) +
        list(image_dir.glob('*.jpeg'))
    )
    
    if not image_paths:
        print(f"Error: No images found in {image_dir}")
        return 1
    
    if args.limit:
        image_paths = image_paths[:args.limit]
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Extract template groups from filenames
    template_groups = defaultdict(list)
    for i, path in enumerate(image_paths):
        template = extract_template_from_filename(path.name)
        if template:
            template_groups[template].append(i)
    
    if template_groups:
        print(f"Detected {len(template_groups)} template groups:")
        for template, indices in sorted(template_groups.items()):
            print(f"  {template}: {len(indices)} images")
    
    # Remove SSIM if requested
    if args.no_ssim:
        global HAS_SKIMAGE
        HAS_SKIMAGE = False
    
    # Run benchmark
    benchmark = MetricBenchmark(image_paths, dict(template_groups) if template_groups else None)
    benchmark.run_benchmark()
    benchmark.print_summary()
    
    # Save results
    if args.output:
        benchmark.save_results(Path(args.output))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

