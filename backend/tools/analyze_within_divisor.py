"""
Within-Divisor Performance Analysis

Controls for divisor variance by analyzing each divisor independently.
Determines what factors actually matter when divisor is held constant.

Also generates visualizations for better data interpretation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def load_metrics(json_path: Path) -> Dict:
    """Load performance metrics from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_all_runs(quality_tests_dir: Path) -> Dict:
    """
    Load and combine metrics from all test runs
    
    Returns a combined dataset with all results from all runs
    """
    runs = sorted(quality_tests_dir.glob("run_*"), key=lambda p: p.name)
    
    if not runs:
        print("[ERROR] No test runs found")
        return None
    
    print(f"Found {len(runs)} test runs:")
    for run in runs:
        print(f"  - {run.name}")
    print()
    
    # Load all metrics
    all_results = []
    test_info_list = []
    
    for run in runs:
        metrics_file = run / "performance_metrics.json"
        if not metrics_file.exists():
            print(f"[WARNING] Skipping {run.name} - no metrics file")
            continue
        
        try:
            data = load_metrics(metrics_file)
            test_info_list.append({
                'run': run.name,
                'timestamp': data['test_info']['timestamp'],
                'configs': data['test_info']['total_configs'],
                'images': data['test_info'].get('test_images', ['unknown', 'unknown'])
            })
            
            # Add run identifier to each result
            for result in data['results']:
                result['_run_source'] = run.name
            
            all_results.extend(data['results'])
            
        except Exception as e:
            print(f"[ERROR] Failed to load {run.name}: {e}")
            continue
    
    if not all_results:
        print("[ERROR] No valid results found")
        return None
    
    print(f"Combined {len(all_results)} total test results from {len(test_info_list)} runs")
    print()
    
    # Create combined dataset
    combined = {
        'test_info': {
            'combined_from': len(test_info_list),
            'total_configs': len(all_results),
            'runs': test_info_list,
            'target_fps': 15.0
        },
        'results': all_results
    }
    
    return combined


def analyze_single_divisor(results: List[Dict], divisor: float) -> Dict:
    """
    Analyze performance for a single divisor value
    
    Returns detailed breakdown of method effects when divisor is controlled
    """
    # Filter to this divisor only
    divisor_results = [r for r in results if r['config']['divisor'] == divisor]
    
    if not divisor_results:
        return None
    
    analysis = {
        'divisor': divisor,
        'n_configs': len(divisor_results),
        'overall_stats': {},
        'by_upscale': {},
        'by_downsample': {},
        'by_combination': {},
        'anova': {},
        'best_combo': None
    }
    
    # Overall stats for this divisor
    all_fps = [r['performance']['fps'] for r in divisor_results]
    analysis['overall_stats'] = {
        'mean_fps': statistics.mean(all_fps),
        'stdev_fps': statistics.stdev(all_fps) if len(all_fps) > 1 else 0,
        'min_fps': min(all_fps),
        'max_fps': max(all_fps),
        'range': max(all_fps) - min(all_fps),
        'cv': (statistics.stdev(all_fps) / statistics.mean(all_fps) * 100) if len(all_fps) > 1 and statistics.mean(all_fps) > 0 else 0
    }
    
    # Group by upscale method
    by_upscale = defaultdict(list)
    for r in divisor_results:
        method = r['config'].get('upscale')
        if method:
            by_upscale[method].append(r['performance']['fps'])
    
    for method, fps_values in by_upscale.items():
        analysis['by_upscale'][method] = {
            'mean': statistics.mean(fps_values),
            'stdev': statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
            'n': len(fps_values)
        }
    
    # Group by downsample method
    by_downsample = defaultdict(list)
    for r in divisor_results:
        method = r['config'].get('downsample')
        if method:
            by_downsample[method].append(r['performance']['fps'])
    
    for method, fps_values in by_downsample.items():
        analysis['by_downsample'][method] = {
            'mean': statistics.mean(fps_values),
            'stdev': statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
            'n': len(fps_values)
        }
    
    # Group by combination
    by_combo = defaultdict(lambda: {'fps': [], 'ssim': []})
    for r in divisor_results:
        up = r['config'].get('upscale')
        down = r['config'].get('downsample')
        if up and down:
            combo = f"{down}+{up}"
            by_combo[combo]['fps'].append(r['performance']['fps'])
            if r['quality'].get('avg_ssim'):
                by_combo[combo]['ssim'].append(r['quality']['avg_ssim'])
    
    for combo, data in by_combo.items():
        analysis['by_combination'][combo] = {
            'mean_fps': statistics.mean(data['fps']),
            'mean_ssim': statistics.mean(data['ssim']) if data['ssim'] else None,
            'stdev_fps': statistics.stdev(data['fps']) if len(data['fps']) > 1 else 0,
            'n': len(data['fps'])
        }
    
    # ANOVA for upscale methods
    upscale_groups = [fps for fps in by_upscale.values()]
    if len(upscale_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*upscale_groups)
        analysis['anova']['upscale'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # ANOVA for downsample methods
    downsample_groups = [fps for fps in by_downsample.values()]
    if len(downsample_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*downsample_groups)
        analysis['anova']['downsample'] = {
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Find best combination
    if analysis['by_combination']:
        best = max(analysis['by_combination'].items(), 
                  key=lambda x: x[1]['mean_fps'])
        analysis['best_combo'] = {
            'name': best[0],
            'fps': best[1]['mean_fps'],
            'ssim': best[1]['mean_ssim']
        }
    
    return analysis


def create_visualizations(data: Dict, output_dir: Path):
    """
    Create comprehensive visualizations for the analysis
    """
    results = [r for r in data['results'] 
              if r['config']['divisor'] != 1 
              and not r['config'].get('multi_upscale')]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: FPS by Method across Divisors
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1a. Upscale methods
    ax1 = fig.add_subplot(gs[0, 0])
    upscale_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        divisor = r['config']['divisor']
        up = r['config'].get('upscale')
        if up:
            upscale_data[divisor][up].append(r['performance']['fps'])
    
    divisors = sorted(upscale_data.keys())
    methods = ['nearest', 'bilinear', 'bicubic']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    x = np.arange(len(divisors))
    width = 0.25
    
    for i, method in enumerate(methods):
        means = [statistics.mean(upscale_data[d][method]) if upscale_data[d][method] else 0 
                for d in divisors]
        stdevs = [statistics.stdev(upscale_data[d][method]) if len(upscale_data[d][method]) > 1 else 0
                 for d in divisors]
        ax1.bar(x + i * width, means, width, label=method.capitalize(), 
               yerr=stdevs, capsize=5, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Divisor', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax1.set_title('Upscale Method Performance by Divisor', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([str(d) for d in divisors])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1b. Downsample methods
    ax2 = fig.add_subplot(gs[0, 1])
    downsample_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        divisor = r['config']['divisor']
        down = r['config'].get('downsample')
        if down:
            downsample_data[divisor][down].append(r['performance']['fps'])
    
    methods_down = ['bilinear', 'bicubic', 'lanczos']
    colors_down = ['#3498db', '#2ecc71', '#9b59b6']
    
    for i, method in enumerate(methods_down):
        means = [statistics.mean(downsample_data[d][method]) if downsample_data[d][method] else 0 
                for d in divisors]
        stdevs = [statistics.stdev(downsample_data[d][method]) if len(downsample_data[d][method]) > 1 else 0
                 for d in divisors]
        ax2.bar(x + i * width, means, width, label=method.capitalize(),
               yerr=stdevs, capsize=5, color=colors_down[i], alpha=0.8)
    
    ax2.set_xlabel('Divisor', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax2.set_title('Downsample Method Performance by Divisor', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(d) for d in divisors])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1c. Scatter: FPS vs SSIM colored by divisor
    ax3 = fig.add_subplot(gs[1, 0])
    
    divisor_colors = {1.5: '#e74c3c', 2: '#3498db', 2.5: '#2ecc71', 
                     3: '#f39c12', 3.5: '#9b59b6', 4: '#1abc9c', 8: '#34495e'}
    
    for r in results:
        if r['quality'].get('avg_ssim'):
            divisor = r['config']['divisor']
            fps = r['performance']['fps']
            ssim = r['quality']['avg_ssim']
            color = divisor_colors.get(divisor, '#95a5a6')
            ax3.scatter(ssim, fps, c=color, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax3.set_xlabel('SSIM (Similarity to Baseline)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax3.set_title('FPS vs Quality Trade-off', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=divisor_colors[d], 
                                     label=f'Div {d}', alpha=0.6)
                      for d in sorted(divisor_colors.keys()) if d in [r['config']['divisor'] for r in results]]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # 1d. Stability heatmap (CV%)
    ax4 = fig.add_subplot(gs[1, 1])
    
    combo_stability = defaultdict(lambda: defaultdict(list))
    for r in results:
        divisor = r['config']['divisor']
        up = r['config'].get('upscale')
        down = r['config'].get('downsample')
        if up and down:
            combo = f"{down[:3]}→{up[:3]}"
            combo_stability[combo][divisor].append(r['performance']['fps'])
    
    # Calculate CV for each combo at each divisor
    combos = sorted(combo_stability.keys())
    divisors_for_heatmap = sorted(set(r['config']['divisor'] for r in results))
    
    heatmap_data = []
    for combo in combos:
        row = []
        for div in divisors_for_heatmap:
            fps_values = combo_stability[combo][div]
            if len(fps_values) > 1:
                cv = statistics.stdev(fps_values) / statistics.mean(fps_values) * 100
            else:
                cv = 0
            row.append(cv)
        heatmap_data.append(row)
    
    im = ax4.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    ax4.set_xticks(range(len(divisors_for_heatmap)))
    ax4.set_xticklabels([str(d) for d in divisors_for_heatmap])
    ax4.set_yticks(range(len(combos)))
    ax4.set_yticklabels(combos)
    ax4.set_xlabel('Divisor', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Method Combination', fontsize=12, fontweight='bold')
    ax4.set_title('Stability (CV%) - Lower is Better', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('CV%', rotation=270, labelpad=20)
    
    plt.suptitle('Interpolation Method Performance Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"[✓] Saved: {output_dir / 'performance_analysis.png'}")
    plt.close()
    
    # Figure 2: Per-Divisor Detailed Analysis
    key_divisors = [2, 2.5, 3, 4]
    available_divisors = [d for d in key_divisors if d in [r['config']['divisor'] for r in results]]
    
    if available_divisors:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, divisor in enumerate(available_divisors[:4]):
            ax = axes[idx]
            
            # Get data for this divisor
            div_results = [r for r in results if r['config']['divisor'] == divisor]
            
            combo_data = defaultdict(lambda: {'fps': [], 'ssim': []})
            for r in div_results:
                up = r['config'].get('upscale')
                down = r['config'].get('downsample')
                if up and down:
                    combo = f"{down[:3]}→{up[:3]}"
                    combo_data[combo]['fps'].append(r['performance']['fps'])
                    if r['quality'].get('avg_ssim'):
                        combo_data[combo]['ssim'].append(r['quality']['avg_ssim'])
            
            # Create bar chart
            combos = sorted(combo_data.keys())
            fps_means = [statistics.mean(combo_data[c]['fps']) for c in combos]
            fps_stdevs = [statistics.stdev(combo_data[c]['fps']) if len(combo_data[c]['fps']) > 1 else 0 
                         for c in combos]
            
            x_pos = np.arange(len(combos))
            bars = ax.bar(x_pos, fps_means, yerr=fps_stdevs, capsize=5, 
                         color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Highlight best combo
            best_idx = fps_means.index(max(fps_means))
            bars[best_idx].set_color('#2ecc71')
            bars[best_idx].set_alpha(0.9)
            
            ax.set_xlabel('Method Combination', fontsize=11, fontweight='bold')
            ax.set_ylabel('FPS', fontsize=11, fontweight='bold')
            ax.set_title(f'Divisor {divisor} - Method Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(combos, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean line
            mean_fps = statistics.mean(fps_means)
            ax.axhline(y=mean_fps, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.text(0.02, mean_fps, f'Mean: {mean_fps:.1f}', 
                   transform=ax.get_yaxis_transform(), fontsize=9,
                   verticalalignment='bottom', color='red')
        
        # Hide unused subplots
        for idx in range(len(available_divisors), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_divisor_analysis.png', dpi=300, bbox_inches='tight')
        print(f"[✓] Saved: {output_dir / 'per_divisor_analysis.png'}")
        plt.close()


def print_within_divisor_report(data: Dict, output_file: Optional[Path] = None):
    """Generate report analyzing each divisor independently"""
    
    results = [r for r in data['results'] 
              if r['config']['divisor'] != 1 
              and not r['config'].get('multi_upscale')]
    
    report_lines = []
    
    def add_line(line: str = ""):
        report_lines.append(line)
        print(line)
    
    add_line("=" * 80)
    add_line("WITHIN-DIVISOR ANALYSIS - Controlling for Divisor Variance")
    add_line("=" * 80)
    add_line()
    
    # Check if this is a combined analysis
    if 'combined_from' in data['test_info']:
        add_line(f"COMBINED ANALYSIS from {data['test_info']['combined_from']} test runs:")
        add_line()
        for run_info in data['test_info']['runs']:
            imgs = run_info['images']
            img_a = Path(imgs[0]).name if imgs and len(imgs) > 0 else 'unknown'
            img_b = Path(imgs[1]).name if imgs and len(imgs) > 1 else 'unknown'
            add_line(f"  • {run_info['run']}: {img_a} → {img_b} ({run_info['configs']} configs)")
        add_line()
        add_line(f"Total configurations analyzed: {data['test_info']['total_configs']}")
        add_line()
    
    add_line("This analysis examines what factors affect performance when divisor is")
    add_line("held constant, removing the dominant divisor effect from consideration.")
    add_line()
    
    # Note about available methods
    add_line("=" * 80)
    add_line("AVAILABLE RESAMPLING METHODS IN PIL/PILLOW")
    add_line("=" * 80)
    add_line()
    add_line("Currently tested:")
    add_line("  • Upscale:   NEAREST, BILINEAR, BICUBIC")
    add_line("  • Downsample: BILINEAR, BICUBIC, LANCZOS")
    add_line()
    add_line("Also available but not yet tested:")
    add_line("  • BOX - Simple averaging, fast for downsampling")
    add_line("  • HAMMING - Windowed sinc filter, between bilinear and lanczos")
    add_line()
    add_line("Note: Advanced methods like DCCI, EWMI, or deep learning approaches")
    add_line("would require custom implementation beyond PIL.")
    add_line()
    
    # Get unique divisors
    divisors = sorted(set(r['config']['divisor'] for r in results))
    
    add_line("=" * 80)
    add_line(f"ANALYZING {len(divisors)} DIVISOR VALUES INDEPENDENTLY")
    add_line("=" * 80)
    add_line()
    
    # Analyze each divisor
    all_divisor_analyses = {}
    
    for divisor in divisors:
        analysis = analyze_single_divisor(results, divisor)
        if not analysis:
            continue
        
        all_divisor_analyses[divisor] = analysis
        
        add_line("-" * 80)
        add_line(f"DIVISOR: {divisor}")
        add_line("-" * 80)
        add_line()
        
        # Overall stats
        stats = analysis['overall_stats']
        add_line(f"Overall Performance at 1/{divisor} resolution:")
        add_line(f"  Mean FPS:  {stats['mean_fps']:.2f} ± {stats['stdev_fps']:.2f}")
        add_line(f"  Range:     {stats['min_fps']:.2f} - {stats['max_fps']:.2f} FPS")
        add_line(f"  Spread:    {stats['range']:.2f} FPS")
        add_line(f"  CV:        {stats['cv']:.1f}%")
        add_line()
        
        # Upscale method ranking
        add_line("Upscale Method Rankings:")
        upscale_sorted = sorted(analysis['by_upscale'].items(), 
                               key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, data) in enumerate(upscale_sorted, 1):
            add_line(f"  {i}. {method:10} {data['mean']:>6.2f} FPS (±{data['stdev']:.2f})")
        
        # ANOVA for upscale
        if 'upscale' in analysis['anova']:
            anova_up = analysis['anova']['upscale']
            sig_str = "[SIGNIFICANT - method choice matters]" if anova_up['significant'] else "[not significant - differences may be noise]"
            add_line(f"  → p-value: {anova_up['p_value']:.4f} {sig_str}")
        add_line()
        
        # Downsample method ranking
        add_line("Downsample Method Rankings:")
        downsample_sorted = sorted(analysis['by_downsample'].items(),
                                   key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, data) in enumerate(downsample_sorted, 1):
            add_line(f"  {i}. {method:10} {data['mean']:>6.2f} FPS (±{data['stdev']:.2f})")
        
        # ANOVA for downsample
        if 'downsample' in analysis['anova']:
            anova_down = analysis['anova']['downsample']
            sig_str = "[SIGNIFICANT - method choice matters]" if anova_down['significant'] else "[not significant - differences may be noise]"
            add_line(f"  → p-value: {anova_down['p_value']:.4f} {sig_str}")
        add_line()
        
        # Top 3 combinations
        add_line("Top 3 Combinations:")
        combo_sorted = sorted(analysis['by_combination'].items(),
                            key=lambda x: x[1]['mean_fps'], reverse=True)
        for i, (combo, data) in enumerate(combo_sorted[:3], 1):
            ssim_str = f", SSIM {data['mean_ssim']:.4f}" if data['mean_ssim'] else ""
            add_line(f"  {i}. {combo:25} {data['mean_fps']:>6.2f} FPS{ssim_str}")
        
        add_line()
    
    # Summary comparison across divisors
    add_line("=" * 80)
    add_line("CROSS-DIVISOR SUMMARY")
    add_line("=" * 80)
    add_line()
    
    add_line("Best Combination at Each Divisor:")
    add_line(f"{'Divisor':<10} {'Best Combo':<25} {'FPS':<10} {'SSIM':<10}")
    add_line("-" * 60)
    
    for divisor in sorted(all_divisor_analyses.keys()):
        analysis = all_divisor_analyses[divisor]
        if analysis['best_combo']:
            best = analysis['best_combo']
            ssim_str = f"{best['ssim']:.4f}" if best['ssim'] else "N/A"
            add_line(f"{divisor:<10} {best['name']:<25} {best['fps']:<10.2f} {ssim_str:<10}")
    
    add_line()
    
    # Statistical significance summary
    add_line("=" * 80)
    add_line("STATISTICAL SIGNIFICANCE SUMMARY")
    add_line("=" * 80)
    add_line()
    
    add_line("Does method choice significantly affect FPS at each divisor?")
    add_line()
    add_line(f"{'Divisor':<10} {'Upscale Sig?':<15} {'Downsample Sig?':<20} {'Interpretation':<30}")
    add_line("-" * 80)
    
    for divisor in sorted(all_divisor_analyses.keys()):
        analysis = all_divisor_analyses[divisor]
        
        up_sig = "Yes (p<0.05)" if analysis['anova'].get('upscale', {}).get('significant') else "No"
        down_sig = "Yes (p<0.05)" if analysis['anova'].get('downsample', {}).get('significant') else "No"
        
        if up_sig.startswith("Yes") and down_sig.startswith("Yes"):
            interp = "Both matter"
        elif up_sig.startswith("Yes"):
            interp = "Upscale matters most"
        elif down_sig.startswith("Yes"):
            interp = "Downsample matters most"
        else:
            interp = "Differences are noise"
        
        add_line(f"{divisor:<10} {up_sig:<15} {down_sig:<20} {interp:<30}")
    
    add_line()
    
    # Key insights
    add_line("=" * 80)
    add_line("KEY INSIGHTS - Within-Divisor Analysis")
    add_line("=" * 80)
    add_line()
    
    # Check consistency of winners
    upscale_winners = []
    downsample_winners = []
    
    for analysis in all_divisor_analyses.values():
        up_sorted = sorted(analysis['by_upscale'].items(), 
                          key=lambda x: x[1]['mean'], reverse=True)
        down_sorted = sorted(analysis['by_downsample'].items(),
                           key=lambda x: x[1]['mean'], reverse=True)
        if up_sorted:
            upscale_winners.append(up_sorted[0][0])
        if down_sorted:
            downsample_winners.append(down_sorted[0][0])
    
    # Count most common winner
    from collections import Counter
    up_counter = Counter(upscale_winners)
    down_counter = Counter(downsample_winners)
    
    if up_counter:
        most_common_up = up_counter.most_common(1)[0]
        if most_common_up[1] == len(upscale_winners):
            add_line(f"✓ Upscale: '{most_common_up[0]}' wins at EVERY divisor (perfectly consistent)")
        else:
            add_line(f"⚠ Upscale: '{most_common_up[0]}' wins most often ({most_common_up[1]}/{len(upscale_winners)} times)")
    
    if down_counter:
        most_common_down = down_counter.most_common(1)[0]
        if most_common_down[1] == len(downsample_winners):
            add_line(f"✓ Downsample: '{most_common_down[0]}' wins at EVERY divisor (perfectly consistent)")
        else:
            add_line(f"✗ Downsample: No consistent winner - '{most_common_down[0]}' only wins {most_common_down[1]}/{len(downsample_winners)} times")
    
    add_line()
    
    # Check if any divisor shows significant effects
    any_significant = False
    for analysis in all_divisor_analyses.values():
        if analysis['anova'].get('upscale', {}).get('significant') or \
           analysis['anova'].get('downsample', {}).get('significant'):
            any_significant = True
            break
    
    if not any_significant:
        add_line("✗ WARNING: No divisor shows statistically significant method effects")
        add_line("  → FPS differences between methods are within measurement noise")
        add_line("  → Choose based on VISUAL QUALITY, not performance metrics")
    else:
        add_line("✓ Some divisors show significant method effects")
        add_line("  → Method choice CAN affect performance at certain resolutions")
    
    add_line()
    add_line("=" * 80)
    
    # Save report
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        add_line()
        add_line(f"Report saved to: {output_file}")


def main():
    """Run within-divisor analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze performance within each divisor')
    parser.add_argument('metrics_json', type=str, nargs='?',
                       help='Path to performance_metrics.json file')
    parser.add_argument('--output', type=str,
                       help='Output file for report (optional)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--all-runs', action='store_true',
                       help='Combine and analyze ALL test runs in quality_tests/')
    args = parser.parse_args()
    
    # Determine if we're analyzing all runs or a single run
    if args.all_runs:
        # Find quality tests directory
        quality_tests_dir = Path("../output/quality_tests")
        if not quality_tests_dir.exists():
            quality_tests_dir = Path("output/quality_tests")
        
        if not quality_tests_dir.exists():
            print("[ERROR] quality_tests directory not found")
            return
        
        print("=" * 80)
        print("COMBINED ANALYSIS - ALL TEST RUNS")
        print("=" * 80)
        print()
        
        # Load all runs
        data = load_all_runs(quality_tests_dir)
        if not data:
            return
        
        # Determine output path
        output_path = quality_tests_dir / "combined_analysis.txt"
        if args.output:
            output_path = Path(args.output)
        
        viz_output_dir = quality_tests_dir
        
    else:
        # Single run analysis
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
                print("Usage: uv run backend/tools/analyze_within_divisor.py [path/to/performance_metrics.json]")
                print("       uv run backend/tools/analyze_within_divisor.py --all-runs")
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
            output_path = metrics_path.parent / "within_divisor_analysis.txt"
        
        viz_output_dir = metrics_path.parent
    
    # Run analysis
    print_within_divisor_report(data, output_path)
    
    # Generate visualizations
    if not args.no_viz:
        print()
        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        print()
        try:
            create_visualizations(data, viz_output_dir)
            print()
            print("[✓] Visualizations complete!")
        except Exception as e:
            print(f"[ERROR] Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

