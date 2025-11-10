"""
Performance Correlation Analysis for Interpolation Methods

Analyzes performance metrics JSON to find correlations between:
- Upscale methods (bilinear, bicubic, nearest)
- Downsample methods (bilinear, bicubic, lanczos)
- Divisor values (1, 2, 3, 4, 8)
- Performance metrics (FPS, quality, etc.)

Goal: Determine if certain method combinations consistently produce better/worse performance
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


@dataclass
class MethodStats:
    """Statistics for a particular method combination"""
    method: str
    values: List[float]
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0
    
    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0
    
    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0
    
    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0
    
    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0
    
    @property
    def cv(self) -> float:
        """Coefficient of variation - relative stability (lower = more stable)"""
        if self.mean == 0:
            return 0.0
        return (self.stdev / self.mean) * 100 if self.values else 0.0


def load_metrics(json_path: Path) -> Dict:
    """Load performance metrics from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_by_category(results: List[Dict], category: str) -> Dict[str, List[Dict]]:
    """
    Extract results grouped by a specific category
    
    Args:
        results: List of test results
        category: 'upscale', 'downsample', or 'divisor'
    
    Returns:
        Dictionary mapping category values to list of results
    """
    grouped = defaultdict(list)
    
    for result in results:
        config = result['config']
        
        # Skip baseline and multi-upscale configs for method analysis
        if config['divisor'] == 1 or config.get('multi_upscale'):
            continue
        
        if category == 'upscale':
            key = config.get('upscale')
        elif category == 'downsample':
            key = config.get('downsample')
        elif category == 'divisor':
            key = str(config.get('divisor'))
        elif category == 'combination':
            # Upscale + Downsample combination
            key = f"{config.get('downsample')}+{config.get('upscale')}"
        else:
            raise ValueError(f"Unknown category: {category}")
        
        if key:
            grouped[key].append(result)
    
    return grouped


def calculate_method_stats(grouped_results: Dict[str, List[Dict]], 
                          metric: str = 'fps') -> Dict[str, MethodStats]:
    """
    Calculate statistics for each method
    
    Args:
        grouped_results: Results grouped by method
        metric: Which metric to analyze ('fps', 'avg_ssim', 'avg_psnr', etc.)
    
    Returns:
        Dictionary mapping method names to their statistics
    """
    stats = {}
    
    for method, results in grouped_results.items():
        values = []
        
        for result in results:
            if metric == 'fps':
                val = result['performance']['fps']
            elif metric in ['avg_ssim', 'avg_psnr', 'min_ssim', 'min_psnr']:
                val = result['quality'].get(metric)
                if val is None:
                    continue
            else:
                # Try performance metrics
                val = result['performance'].get(metric)
                if val is None:
                    continue
            
            values.append(val)
        
        if values:
            stats[method] = MethodStats(method=method, values=values)
    
    return stats


def analyze_correlation_by_divisor(results: List[Dict], method_category: str) -> Dict:
    """
    Analyze if a method (upscale/downsample) correlates with performance across divisors
    
    Returns analysis showing whether method choice matters or is random
    """
    analysis = {
        'category': method_category,
        'by_divisor': {},
        'overall_ranking': {},
        'consistency_score': 0.0
    }
    
    divisors = ['2', '3', '4', '8']
    
    # Get FPS for each method at each divisor
    method_rankings = defaultdict(list)  # method -> [rank at div2, rank at div3, ...]
    
    for divisor in divisors:
        # Filter to this divisor
        divisor_results = [r for r in results 
                          if r['config']['divisor'] == int(divisor) 
                          and not r['config'].get('multi_upscale')]
        
        if not divisor_results:
            continue
        
        # Group by method and calculate average FPS
        method_fps = defaultdict(list)
        for result in divisor_results:
            method = result['config'].get(method_category)
            if method:
                method_fps[method].append(result['performance']['fps'])
        
        # Average FPS per method
        method_avg_fps = {m: statistics.mean(fps) for m, fps in method_fps.items()}
        
        # Rank methods (1 = fastest)
        ranked = sorted(method_avg_fps.items(), key=lambda x: x[1], reverse=True)
        
        analysis['by_divisor'][divisor] = {
            'ranked': [(m, f) for m, f in ranked],
            'spread': max(method_avg_fps.values()) - min(method_avg_fps.values()) if method_avg_fps else 0
        }
        
        # Track rankings
        for rank, (method, fps) in enumerate(ranked, 1):
            method_rankings[method].append(rank)
    
    # Calculate overall ranking consistency
    # If a method is always fastest/slowest, rankings will have low variance
    # If random, rankings will vary wildly
    consistency_scores = {}
    for method, ranks in method_rankings.items():
        if len(ranks) > 1:
            # Lower variance = more consistent
            variance = statistics.variance(ranks)
            avg_rank = statistics.mean(ranks)
            consistency_scores[method] = {
                'avg_rank': avg_rank,
                'rank_variance': variance,
                'ranks': ranks
            }
    
    analysis['overall_ranking'] = consistency_scores
    
    # Overall consistency: how much do rankings vary?
    if consistency_scores:
        avg_variance = statistics.mean([s['rank_variance'] for s in consistency_scores.values()])
        # Convert to 0-1 score (0 = completely random, 1 = perfectly consistent)
        # Max variance for 3 methods across 4 divisors ≈ 0.67
        analysis['consistency_score'] = max(0, 1 - (avg_variance / 0.67))
    
    return analysis


def analyze_combination_patterns(results: List[Dict]) -> Dict:
    """
    Analyze if specific up/down combinations consistently perform better/worse
    """
    # Group by combination
    combo_results = extract_by_category(results, 'combination')
    
    # Calculate stats for each combo
    combo_stats = calculate_method_stats(combo_results, 'fps')
    
    # Also get quality
    combo_quality = calculate_method_stats(combo_results, 'avg_ssim')
    
    # Rank combinations
    ranked_by_fps = sorted(combo_stats.items(), 
                          key=lambda x: x[1].mean, 
                          reverse=True)
    
    ranked_by_quality = sorted(combo_quality.items(),
                              key=lambda x: x[1].mean,
                              reverse=True)
    
    analysis = {
        'total_combinations': len(combo_stats),
        'ranked_by_fps': [(name, stats.mean, stats.stdev) for name, stats in ranked_by_fps],
        'ranked_by_quality': [(name, stats.mean, stats.stdev) for name, stats in ranked_by_quality],
        'best_balanced': []
    }
    
    # Find best balanced (high FPS, high quality)
    balanced = []
    for combo in combo_stats.keys():
        if combo in combo_quality:
            fps_mean = combo_stats[combo].mean
            quality_mean = combo_quality[combo].mean
            
            # Normalize to 0-1 scale (rough approximation)
            fps_norm = fps_mean / 35.0  # Assuming max ~35 FPS
            quality_norm = quality_mean  # SSIM already 0-1
            
            # Combined score
            balance_score = (fps_norm + quality_norm) / 2
            balanced.append((combo, balance_score, fps_mean, quality_mean))
    
    balanced.sort(key=lambda x: x[1], reverse=True)
    analysis['best_balanced'] = balanced[:10]
    
    return analysis


def analyze_divisor_scaling(results: List[Dict]) -> Dict:
    """
    Analyze how FPS scales with divisor size (is it predictable?)
    """
    divisor_groups = extract_by_category(results, 'divisor')
    
    analysis = {
        'by_divisor': {},
        'scaling_pattern': 'unknown'
    }
    
    divisors_sorted = sorted([float(d) for d in divisor_groups.keys()])
    
    avg_fps_by_divisor = []
    for divisor in divisors_sorted:
        div_str = str(divisor)
        fps_values = [r['performance']['fps'] for r in divisor_groups[div_str]]
        
        if not fps_values:
            continue
        
        avg_fps = statistics.mean(fps_values)
        variance = statistics.variance(fps_values) if len(fps_values) > 1 else 0
        
        analysis['by_divisor'][divisor] = {
            'avg_fps': avg_fps,
            'variance': variance,
            'min': min(fps_values),
            'max': max(fps_values),
            'spread': max(fps_values) - min(fps_values),
            'configs_tested': len(fps_values)
        }
        
        avg_fps_by_divisor.append((divisor, avg_fps))
    
    # Check if FPS increases consistently with divisor
    if len(avg_fps_by_divisor) > 2:
        fps_increases = [avg_fps_by_divisor[i+1][1] > avg_fps_by_divisor[i][1] 
                        for i in range(len(avg_fps_by_divisor)-1)]
        
        if all(fps_increases):
            analysis['scaling_pattern'] = 'monotonic_increase'
        elif sum(fps_increases) / len(fps_increases) > 0.66:
            analysis['scaling_pattern'] = 'mostly_increasing'
        else:
            analysis['scaling_pattern'] = 'irregular'
    
    return analysis


def calculate_variance_ratio(results: List[Dict], category: str) -> float:
    """
    Calculate how much variance is explained by the category
    
    High ratio = category matters a lot
    Low ratio = category doesn't matter much (random)
    """
    grouped = extract_by_category(results, category)
    stats = calculate_method_stats(grouped, 'fps')
    
    # Within-group variance (how much variation within each method)
    within_variance = []
    for method_stats in stats.values():
        if len(method_stats.values) > 1:
            within_variance.extend(method_stats.values)
    
    # Between-group variance (how different are the methods' means)
    method_means = [s.mean for s in stats.values()]
    
    if len(within_variance) <= 1 or len(method_means) <= 1:
        return 0.0
    
    total_variance = statistics.variance(within_variance)
    between_variance = statistics.variance(method_means)
    
    # Ratio: higher = category explains more variance
    return between_variance / total_variance if total_variance > 0 else 0.0


def analyze_stability_within_groups(results: List[Dict]) -> Dict:
    """
    Analyze FPS stability within each method combination across divisors
    
    Lower variance/CV = more predictable performance
    """
    combo_results = extract_by_category(results, 'combination')
    
    stability_analysis = {}
    
    for combo, combo_results_list in combo_results.items():
        fps_values = [r['performance']['fps'] for r in combo_results_list]
        
        if len(fps_values) < 2:
            continue
        
        mean_fps = statistics.mean(fps_values)
        stdev_fps = statistics.stdev(fps_values)
        cv = (stdev_fps / mean_fps * 100) if mean_fps > 0 else 0
        
        # Group by divisor to see if certain divisors are outliers
        by_divisor = defaultdict(list)
        for r in combo_results_list:
            by_divisor[r['config']['divisor']].append(r['performance']['fps'])
        
        divisor_means = {d: statistics.mean(fps) for d, fps in by_divisor.items()}
        
        stability_analysis[combo] = {
            'mean': mean_fps,
            'stdev': stdev_fps,
            'cv': cv,  # Coefficient of variation (%)
            'min': min(fps_values),
            'max': max(fps_values),
            'range': max(fps_values) - min(fps_values),
            'n_samples': len(fps_values),
            'by_divisor': divisor_means
        }
    
    return stability_analysis


def perform_linear_regression(results: List[Dict]) -> Dict:
    """
    Linear regression to quantify FPS contribution of each method
    
    Model: FPS = β0 + β1*divisor + β2*upscale + β3*downsample + ε
    """
    # Prepare data
    data = []
    for r in results:
        data.append({
            'fps': r['performance']['fps'],
            'divisor': r['config']['divisor'],
            'upscale': r['config'].get('upscale', 'none'),
            'downsample': r['config'].get('downsample', 'none')
        })
    
    # Convert to numpy arrays
    fps = np.array([d['fps'] for d in data])
    divisor = np.array([d['divisor'] for d in data]).reshape(-1, 1)
    
    # Encode categorical variables
    upscale_encoder = LabelEncoder()
    downsample_encoder = LabelEncoder()
    
    upscale_encoded = upscale_encoder.fit_transform([d['upscale'] for d in data]).reshape(-1, 1)
    downsample_encoded = downsample_encoder.fit_transform([d['downsample'] for d in data]).reshape(-1, 1)
    
    # Combine features
    X = np.hstack([divisor, upscale_encoded, downsample_encoded])
    y = fps
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    r_squared = model.score(X, y)
    
    # Get coefficients
    coef_divisor = model.coef_[0]
    coef_upscale = model.coef_[1]
    coef_downsample = model.coef_[2]
    intercept = model.intercept_
    
    # Calculate standardized coefficients (to compare importance)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean()) / y.std()
    model_std = LinearRegression()
    model_std.fit(X_std, y_std)
    
    analysis = {
        'r_squared': r_squared,
        'intercept': intercept,
        'coefficients': {
            'divisor': coef_divisor,
            'upscale': coef_upscale,
            'downsample': coef_downsample
        },
        'standardized_coefficients': {
            'divisor': model_std.coef_[0],
            'upscale': model_std.coef_[1],
            'downsample': model_std.coef_[2]
        },
        'encoders': {
            'upscale': dict(zip(upscale_encoder.classes_, range(len(upscale_encoder.classes_)))),
            'downsample': dict(zip(downsample_encoder.classes_, range(len(downsample_encoder.classes_))))
        },
        'interpretation': {}
    }
    
    # Interpret coefficients
    # Positive coefficient = increases FPS
    # Negative coefficient = decreases FPS
    
    analysis['interpretation']['divisor'] = f"Each +1 divisor → +{coef_divisor:.2f} FPS"
    
    # For categorical, need to map back
    upscale_effects = {}
    for method, code in analysis['encoders']['upscale'].items():
        if method != 'none':
            effect = coef_upscale * code
            upscale_effects[method] = effect
    
    downsample_effects = {}
    for method, code in analysis['encoders']['downsample'].items():
        if method != 'none':
            effect = coef_downsample * code
            downsample_effects[method] = effect
    
    analysis['interpretation']['upscale_effects'] = upscale_effects
    analysis['interpretation']['downsample_effects'] = downsample_effects
    
    return analysis


def perform_anova(results: List[Dict]) -> Dict:
    """
    ANOVA to test if method choice significantly affects FPS
    
    H0: All methods have the same mean FPS
    H1: At least one method differs
    """
    # ANOVA for upscale methods
    upscale_groups = extract_by_category(results, 'upscale')
    upscale_fps_groups = [[r['performance']['fps'] for r in group] 
                         for group in upscale_groups.values()]
    
    if len(upscale_fps_groups) >= 2:
        f_stat_up, p_value_up = stats.f_oneway(*upscale_fps_groups)
    else:
        f_stat_up, p_value_up = 0, 1.0
    
    # ANOVA for downsample methods
    downsample_groups = extract_by_category(results, 'downsample')
    downsample_fps_groups = [[r['performance']['fps'] for r in group]
                            for group in downsample_groups.values()]
    
    if len(downsample_fps_groups) >= 2:
        f_stat_down, p_value_down = stats.f_oneway(*downsample_fps_groups)
    else:
        f_stat_down, p_value_down = 0, 1.0
    
    return {
        'upscale': {
            'f_statistic': f_stat_up,
            'p_value': p_value_up,
            'significant': p_value_up < 0.05,
            'interpretation': 'Upscale method SIGNIFICANTLY affects FPS' if p_value_up < 0.05 
                            else 'Upscale method does NOT significantly affect FPS'
        },
        'downsample': {
            'f_statistic': f_stat_down,
            'p_value': p_value_down,
            'significant': p_value_down < 0.05,
            'interpretation': 'Downsample method SIGNIFICANTLY affects FPS' if p_value_down < 0.05
                            else 'Downsample method does NOT significantly affect FPS'
        }
    }


def analyze_variance_decomposition(results: List[Dict]) -> Dict:
    """
    Decompose total variance into components:
    - Between divisors
    - Between upscale methods (within divisor)
    - Between downsample methods (within divisor)
    - Residual (measurement noise)
    """
    all_fps = [r['performance']['fps'] for r in results]
    total_variance = statistics.variance(all_fps) if len(all_fps) > 1 else 0
    
    # Variance between divisors
    divisor_groups = extract_by_category(results, 'divisor')
    divisor_means = [statistics.mean([r['performance']['fps'] for r in group])
                    for group in divisor_groups.values()]
    variance_between_divisors = statistics.variance(divisor_means) if len(divisor_means) > 1 else 0
    
    # For each divisor, calculate variance due to methods
    method_variances = []
    for divisor, divisor_results in divisor_groups.items():
        # Group by combination within this divisor
        combo_results = extract_by_category(divisor_results, 'combination')
        if len(combo_results) > 1:
            combo_means = [statistics.mean([r['performance']['fps'] for r in group])
                          for group in combo_results.values()]
            if len(combo_means) > 1:
                method_variances.append(statistics.variance(combo_means))
    
    variance_due_to_methods = statistics.mean(method_variances) if method_variances else 0
    
    # Residual variance (within each combo)
    residual_variances = []
    combo_all = extract_by_category(results, 'combination')
    for combo_results in combo_all.values():
        fps_values = [r['performance']['fps'] for r in combo_results]
        if len(fps_values) > 1:
            residual_variances.append(statistics.variance(fps_values))
    
    variance_residual = statistics.mean(residual_variances) if residual_variances else 0
    
    # Calculate percentages
    total_explained = variance_between_divisors + variance_due_to_methods + variance_residual
    
    return {
        'total_variance': total_variance,
        'components': {
            'divisor': {
                'variance': variance_between_divisors,
                'percent': (variance_between_divisors / total_variance * 100) if total_variance > 0 else 0
            },
            'method_choice': {
                'variance': variance_due_to_methods,
                'percent': (variance_due_to_methods / total_variance * 100) if total_variance > 0 else 0
            },
            'residual': {
                'variance': variance_residual,
                'percent': (variance_residual / total_variance * 100) if total_variance > 0 else 0
            }
        },
        'total_explained': total_explained
    }


def print_analysis_report(data: Dict, output_file: Optional[Path] = None):
    """Generate and print comprehensive analysis report"""
    
    results = data['results']
    
    # Filter out baseline and multi-upscale
    filtered_results = [r for r in results 
                       if r['config']['divisor'] != 1 
                       and not r['config'].get('multi_upscale')]
    
    report_lines = []
    
    def add_line(line: str = ""):
        report_lines.append(line)
        print(line)
    
    add_line("=" * 80)
    add_line("INTERPOLATION PERFORMANCE CORRELATION ANALYSIS")
    add_line("=" * 80)
    add_line()
    add_line(f"Dataset: {data['test_info']['total_configs']} configurations tested")
    add_line(f"Timestamp: {data['test_info']['timestamp']}")
    add_line(f"Analyzing: {len(filtered_results)} non-baseline configurations")
    add_line()
    
    # 1. Overall variance analysis
    add_line("=" * 80)
    add_line("1. VARIANCE ANALYSIS - What factors matter?")
    add_line("=" * 80)
    add_line()
    
    upscale_variance_ratio = calculate_variance_ratio(filtered_results, 'upscale')
    downsample_variance_ratio = calculate_variance_ratio(filtered_results, 'downsample')
    divisor_variance_ratio = calculate_variance_ratio(filtered_results, 'divisor')
    
    add_line(f"Variance explained by each factor (higher = more important):")
    add_line(f"  • Divisor:      {divisor_variance_ratio:.4f}")
    add_line(f"  • Downsample:   {downsample_variance_ratio:.4f}")
    add_line(f"  • Upscale:      {upscale_variance_ratio:.4f}")
    add_line()
    
    # Interpret results
    if divisor_variance_ratio > 0.5:
        add_line("→ DIVISOR is the dominant factor in performance")
    if downsample_variance_ratio < 0.1 and upscale_variance_ratio < 0.1:
        add_line("→ SCALING METHODS have minimal impact (performance is RANDOM across methods)")
    elif downsample_variance_ratio > 0.2 or upscale_variance_ratio > 0.2:
        add_line("→ SCALING METHODS do affect performance (non-random)")
    
    add_line()
    
    # 2. Stability analysis
    add_line("=" * 80)
    add_line("2. STABILITY ANALYSIS - Which combinations are most consistent?")
    add_line("=" * 80)
    add_line()
    
    stability_analysis = analyze_stability_within_groups(filtered_results)
    
    # Sort by coefficient of variation (lower = more stable)
    sorted_by_stability = sorted(stability_analysis.items(), key=lambda x: x[1]['cv'])
    
    add_line("Method Combinations Ranked by Stability (lower CV% = more consistent):")
    add_line()
    add_line(f"{'Rank':<6} {'Combination':<25} {'Mean FPS':<12} {'StdDev':<10} {'CV%':<10} {'Range':<12}")
    add_line("-" * 85)
    
    for i, (combo, stats) in enumerate(sorted_by_stability, 1):
        down, up = combo.split('+')
        combo_str = f"{down[:3]}→{up[:3]}"
        add_line(f"{i:<6} {combo:<25} {stats['mean']:>10.2f}  {stats['stdev']:>8.2f}  "
                f"{stats['cv']:>8.1f}  {stats['range']:>10.2f}")
    
    add_line()
    add_line("Interpretation:")
    most_stable = sorted_by_stability[0]
    least_stable = sorted_by_stability[-1]
    add_line(f"  Most stable:  {most_stable[0]} (CV = {most_stable[1]['cv']:.1f}%)")
    add_line(f"  Least stable: {least_stable[0]} (CV = {least_stable[1]['cv']:.1f}%)")
    add_line()
    
    if most_stable[1]['cv'] < 5:
        add_line("  → Most stable combo has VERY low variance - highly predictable!")
    elif most_stable[1]['cv'] < 15:
        add_line("  → Most stable combo has reasonable variance - moderately predictable")
    else:
        add_line("  → Even 'most stable' combo is quite variable - performance is noisy")
    
    add_line()
    
    # 3. Linear Regression Analysis
    add_line("=" * 80)
    add_line("3. LINEAR REGRESSION - Quantifying method contributions")
    add_line("=" * 80)
    add_line()
    
    regression = perform_linear_regression(filtered_results)
    
    add_line(f"Model: FPS = β0 + β1·divisor + β2·upscale + β3·downsample + ε")
    add_line()
    add_line(f"R² Score: {regression['r_squared']:.4f}")
    add_line(f"  ({regression['r_squared']*100:.1f}% of FPS variance explained by the model)")
    add_line()
    
    add_line("Coefficients (raw):")
    add_line(f"  Intercept:   {regression['intercept']:>8.2f}")
    add_line(f"  Divisor:     {regression['coefficients']['divisor']:>8.2f}  "
            f"({regression['interpretation']['divisor']})")
    add_line(f"  Upscale:     {regression['coefficients']['upscale']:>8.2f}")
    add_line(f"  Downsample:  {regression['coefficients']['downsample']:>8.2f}")
    add_line()
    
    add_line("Standardized Coefficients (comparable importance):")
    std_coefs = regression['standardized_coefficients']
    add_line(f"  Divisor:     {std_coefs['divisor']:>8.2f}  (importance rank: 1)")
    add_line(f"  Upscale:     {std_coefs['upscale']:>8.2f}  "
            f"(importance rank: {2 if abs(std_coefs['upscale']) > abs(std_coefs['downsample']) else 3})")
    add_line(f"  Downsample:  {std_coefs['downsample']:>8.2f}  "
            f"(importance rank: {2 if abs(std_coefs['downsample']) > abs(std_coefs['upscale']) else 3})")
    add_line()
    
    add_line("Interpretation:")
    if regression['r_squared'] > 0.9:
        add_line("  ✓ Excellent model fit - FPS is highly predictable from these factors")
    elif regression['r_squared'] > 0.7:
        add_line("  ✓ Good model fit - most variance is explained")
    elif regression['r_squared'] > 0.5:
        add_line("  ⚠ Moderate fit - some unexplained variance remains")
    else:
        add_line("  ✗ Poor fit - other factors or noise dominate")
    
    add_line()
    
    # 4. ANOVA - Statistical Significance
    add_line("=" * 80)
    add_line("4. ANOVA - Are method differences statistically significant?")
    add_line("=" * 80)
    add_line()
    
    anova = perform_anova(filtered_results)
    
    add_line("Upscale Method:")
    add_line(f"  F-statistic: {anova['upscale']['f_statistic']:.4f}")
    add_line(f"  p-value:     {anova['upscale']['p_value']:.6f}")
    add_line(f"  Result:      {anova['upscale']['interpretation']}")
    add_line()
    
    add_line("Downsample Method:")
    add_line(f"  F-statistic: {anova['downsample']['f_statistic']:.4f}")
    add_line(f"  p-value:     {anova['downsample']['p_value']:.6f}")
    add_line(f"  Result:      {anova['downsample']['interpretation']}")
    add_line()
    
    add_line("Interpretation (α = 0.05):")
    if anova['upscale']['significant'] and anova['downsample']['significant']:
        add_line("  ✓ BOTH upscale and downsample methods significantly affect FPS")
        add_line("    → Method choice MATTERS for performance")
    elif anova['upscale']['significant']:
        add_line("  ✓ Upscale method affects FPS, but downsample does not")
        add_line("    → Focus on choosing the right upscale method")
    elif anova['downsample']['significant']:
        add_line("  ✓ Downsample method affects FPS, but upscale does not")
        add_line("    → Focus on choosing the right downsample method")
    else:
        add_line("  ✗ Neither method significantly affects FPS")
        add_line("    → Differences may be due to random variation/noise")
    
    add_line()
    
    # 5. Variance Decomposition
    add_line("=" * 80)
    add_line("5. VARIANCE DECOMPOSITION - Where does variability come from?")
    add_line("=" * 80)
    add_line()
    
    var_decomp = analyze_variance_decomposition(filtered_results)
    
    add_line(f"Total FPS Variance: {var_decomp['total_variance']:.2f}")
    add_line()
    add_line("Breakdown:")
    add_line(f"  Divisor:       {var_decomp['components']['divisor']['variance']:>8.2f}  "
            f"({var_decomp['components']['divisor']['percent']:>5.1f}%)")
    add_line(f"  Method Choice: {var_decomp['components']['method_choice']['variance']:>8.2f}  "
            f"({var_decomp['components']['method_choice']['percent']:>5.1f}%)")
    add_line(f"  Residual:      {var_decomp['components']['residual']['variance']:>8.2f}  "
            f"({var_decomp['components']['residual']['percent']:>5.1f}%)")
    add_line()
    
    add_line("Interpretation:")
    divisor_pct = var_decomp['components']['divisor']['percent']
    method_pct = var_decomp['components']['method_choice']['percent']
    residual_pct = var_decomp['components']['residual']['percent']
    
    if divisor_pct > 70:
        add_line("  → Divisor dominates (~{:.0f}% of variance)".format(divisor_pct))
    if method_pct > 10:
        add_line("  → Method choice matters (~{:.0f}% of variance)".format(method_pct))
    if residual_pct > 20:
        add_line("  → High residual variance (~{:.0f}%) - measurement noise or other factors".format(residual_pct))
    
    add_line()
    
    # 6. Upscale method analysis
    add_line("=" * 80)
    add_line("6. UPSCALE METHOD ANALYSIS")
    add_line("=" * 80)
    add_line()
    
    upscale_analysis = analyze_correlation_by_divisor(filtered_results, 'upscale')
    
    add_line(f"Consistency Score: {upscale_analysis['consistency_score']:.2f} / 1.00")
    add_line(f"  (0.00 = completely random, 1.00 = perfectly consistent)")
    add_line()
    
    add_line("Average Rankings by Divisor:")
    for divisor, data in upscale_analysis['by_divisor'].items():
        add_line(f"\n  Divisor {divisor}:")
        for i, (method, fps) in enumerate(data['ranked'], 1):
            add_line(f"    {i}. {method:10} - {fps:.2f} FPS")
        add_line(f"    Spread: {data['spread']:.2f} FPS")
    
    add_line("\n  Overall Ranking Consistency:")
    for method, stats in sorted(upscale_analysis['overall_ranking'].items(), 
                                key=lambda x: x[1]['avg_rank']):
        add_line(f"    {method:10} - Avg Rank: {stats['avg_rank']:.2f}, "
                f"Variance: {stats['rank_variance']:.2f}, Ranks: {stats['ranks']}")
    
    add_line()
    
    # 7. Downsample method analysis
    add_line("=" * 80)
    add_line("7. DOWNSAMPLE METHOD ANALYSIS")
    add_line("=" * 80)
    add_line()
    
    downsample_analysis = analyze_correlation_by_divisor(filtered_results, 'downsample')
    
    add_line(f"Consistency Score: {downsample_analysis['consistency_score']:.2f} / 1.00")
    add_line()
    
    add_line("Average Rankings by Divisor:")
    for divisor, data in downsample_analysis['by_divisor'].items():
        add_line(f"\n  Divisor {divisor}:")
        for i, (method, fps) in enumerate(data['ranked'], 1):
            add_line(f"    {i}. {method:10} - {fps:.2f} FPS")
        add_line(f"    Spread: {data['spread']:.2f} FPS")
    
    add_line("\n  Overall Ranking Consistency:")
    for method, stats in sorted(downsample_analysis['overall_ranking'].items(),
                                key=lambda x: x[1]['avg_rank']):
        add_line(f"    {method:10} - Avg Rank: {stats['avg_rank']:.2f}, "
                f"Variance: {stats['rank_variance']:.2f}, Ranks: {stats['ranks']}")
    
    add_line()
    
    # 8. Combination analysis
    add_line("=" * 80)
    add_line("8. METHOD COMBINATION ANALYSIS")
    add_line("=" * 80)
    add_line()
    
    combo_analysis = analyze_combination_patterns(filtered_results)
    
    add_line(f"Total combinations tested: {combo_analysis['total_combinations']}")
    add_line()
    
    add_line("Top 10 by FPS (average across all divisors):")
    for i, (combo, fps, stdev) in enumerate(combo_analysis['ranked_by_fps'][:10], 1):
        down, up = combo.split('+')
        add_line(f"  {i:2}. {down:10} → {up:10}  {fps:6.2f} FPS (±{stdev:.2f})")
    
    add_line()
    add_line("Top 10 by Quality (SSIM):")
    for i, (combo, ssim, stdev) in enumerate(combo_analysis['ranked_by_quality'][:10], 1):
        down, up = combo.split('+')
        add_line(f"  {i:2}. {down:10} → {up:10}  {ssim:.4f} SSIM (±{stdev:.4f})")
    
    add_line()
    add_line("Best Balanced (FPS + Quality):")
    for i, (combo, score, fps, quality) in enumerate(combo_analysis['best_balanced'][:10], 1):
        down, up = combo.split('+')
        add_line(f"  {i:2}. {down:10} → {up:10}  "
                f"Score: {score:.3f} (FPS: {fps:.2f}, SSIM: {quality:.4f})")
    
    add_line()
    
    # 9. Divisor scaling analysis
    add_line("=" * 80)
    add_line("9. DIVISOR SCALING ANALYSIS")
    add_line("=" * 80)
    add_line()
    
    divisor_analysis = analyze_divisor_scaling(filtered_results)
    
    add_line(f"Scaling Pattern: {divisor_analysis['scaling_pattern'].upper()}")
    add_line()
    
    add_line("Performance by Divisor:")
    add_line(f"{'Divisor':<10} {'Avg FPS':<12} {'Min-Max':<15} {'Spread':<10} {'Variance':<10}")
    add_line("-" * 65)
    
    for divisor in sorted(divisor_analysis['by_divisor'].keys()):
        data = divisor_analysis['by_divisor'][divisor]
        add_line(f"{divisor:<10} {data['avg_fps']:>10.2f}  "
                f"{data['min']:>6.2f}-{data['max']:<6.2f}  "
                f"{data['spread']:>8.2f}  {data['variance']:>10.2f}")
    
    add_line()
    
    # 10. Key insights
    add_line("=" * 80)
    add_line("10. KEY INSIGHTS")
    add_line("=" * 80)
    add_line()
    
    # Determine if methods matter
    if upscale_analysis['consistency_score'] < 0.3 and downsample_analysis['consistency_score'] < 0.3:
        add_line("✗ Scaling method choice appears RANDOM / INCONSISTENT")
        add_line("  → Performance varies chaotically across divisors")
        add_line("  → No clear winner for upscale or downsample methods")
        add_line("  → Recommendation: Choose based on AESTHETICS, not performance")
    else:
        add_line("✓ Scaling method choice shows CONSISTENT patterns")
        add_line("  → Some methods consistently outperform others")
        if upscale_analysis['consistency_score'] > downsample_analysis['consistency_score']:
            add_line("  → UPSCALE method matters more than downsample")
        else:
            add_line("  → DOWNSAMPLE method matters more than upscale")
    
    add_line()
    
    # Divisor insights
    if divisor_analysis['scaling_pattern'] == 'monotonic_increase':
        add_line("✓ FPS scales predictably with divisor")
        add_line("  → Higher divisor = consistently higher FPS")
    else:
        add_line("✗ FPS scaling is irregular across divisors")
        add_line("  → Performance doesn't always improve with higher divisor")
    
    add_line()
    
    # Spread analysis
    max_spread = max([d['spread'] for d in divisor_analysis['by_divisor'].values()])
    if max_spread > 5:
        add_line(f"⚠ High spread detected ({max_spread:.2f} FPS difference within same divisor)")
        add_line("  → Method choice CAN make a difference")
        add_line("  → Review top combinations for optimal config")
    else:
        add_line(f"✓ Low spread ({max_spread:.2f} FPS) - methods perform similarly")
        add_line("  → Any method choice is reasonable")
    
    add_line()
    
    # Final recommendation
    add_line("=" * 80)
    add_line("RECOMMENDATION")
    add_line("=" * 80)
    add_line()
    
    best_combo = combo_analysis['best_balanced'][0]
    down, up = best_combo[0].split('+')
    
    add_line(f"For optimal balance (performance + quality):")
    add_line(f"  → Downsample: {down}")
    add_line(f"  → Upscale:    {up}")
    add_line(f"  → Expected:   {best_combo[2]:.2f} FPS, {best_combo[3]:.4f} SSIM")
    add_line()
    
    if upscale_analysis['consistency_score'] < 0.3 and downsample_analysis['consistency_score'] < 0.3:
        add_line("However, since methods show random behavior:")
        add_line("  → Feel free to experiment based on visual aesthetics")
        add_line("  → Performance differences are likely noise / measurement variance")
    
    add_line()
    add_line("=" * 80)
    
    # Save report
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        add_line()
        add_line(f"Report saved to: {output_file}")


def main():
    """Run correlation analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze performance correlations')
    parser.add_argument('metrics_json', type=str, nargs='?',
                       help='Path to performance_metrics.json file')
    parser.add_argument('--output', type=str,
                       help='Output file for report (optional)')
    args = parser.parse_args()
    
    # Find metrics file
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
    else:
        # Find most recent test
        quality_tests_dir = Path("../output/quality_tests")
        if not quality_tests_dir.exists():
            quality_tests_dir = Path("output/quality_tests")
        
        if not quality_tests_dir.exists():
            print("[ERROR] No quality tests found")
            print("Usage: python analyze_performance_correlations.py [path/to/performance_metrics.json]")
            return
        
        # Find most recent run
        runs = sorted(quality_tests_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
        if not runs:
            print("[ERROR] No test runs found")
            return
        
        metrics_path = runs[0] / "performance_metrics.json"
        print(f"Using most recent test: {runs[0].name}")
        print()
    
    if not metrics_path.exists():
        print(f"[ERROR] File not found: {metrics_path}")
        return
    
    # Load data
    print(f"Loading metrics from: {metrics_path}")
    data = load_metrics(metrics_path)
    print()
    
    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = metrics_path.parent / "correlation_analysis.txt"
    
    # Run analysis
    print_analysis_report(data, output_path)


if __name__ == "__main__":
    main()

