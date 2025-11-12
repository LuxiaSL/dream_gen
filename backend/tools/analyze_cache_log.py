"""
Cache Log Analyzer
Analyzes dream_controller.log to understand cache behavior and dual-metric performance

Usage:
    python backend/tools/analyze_cache_log.py [log_file]
    
    # Or from backend directory:
    python tools/analyze_cache_log.py ../logs/dream_controller.log

Output:
    - Cache population statistics
    - Injection patterns and frequencies
    - Similarity distributions (ColorHist and pHash)
    - Convergence detection analysis
    - Timeline of events
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import statistics


class CacheLogAnalyzer:
    """Analyzes dream_controller.log for cache and injection patterns"""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lines = []
        self.keyframes_generated = []
        self.cache_injections = []
        self.seed_injections = []
        self.cache_additions = []
        self.diversity_checks = []
        self.dissimilar_selections = []
        self.convergence_events = []
        
    def load_log(self):
        """Load log file"""
        if not self.log_path.exists():
            print(f"❌ Log file not found: {self.log_path}")
            sys.exit(1)
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        
        print(f"✓ Loaded {len(self.lines)} log lines from {self.log_path.name}")
    
    def parse_log(self):
        """Parse log file for relevant events"""
        
        for i, line in enumerate(self.lines):
            # Keyframe generation
            if "[OK] Keyframe" in line:
                match = re.search(r'\[OK\] Keyframe (\d+)', line)
                if match:
                    kf_num = int(match.group(1))
                    
                    if "from DISSIMILAR CACHE" in line:
                        self.keyframes_generated.append((kf_num, "cache"))
                        self.cache_injections.append(kf_num)
                    elif "from SEED" in line:
                        self.keyframes_generated.append((kf_num, "seed"))
                        self.seed_injections.append(kf_num)
                    elif "generated in" in line:
                        self.keyframes_generated.append((kf_num, "generated"))
            
            # Cache additions
            if "Added keyframe to cache:" in line:
                match = re.search(r'cache_(\d+)_\d+ \(total: (\d+)\)', line)
                if match:
                    cache_id = f"cache_{match.group(1)}"
                    total = int(match.group(2))
                    self.cache_additions.append((cache_id, total))
            
            # Diversity checks (dual-metric)
            if "Frame is diverse" in line or "Frame is redundant" in line:
                # Extract color and struct similarities
                match = re.search(r'color:([\d.]+).*struct:([\d.]+)', line)
                if match:
                    color_sim = float(match.group(1))
                    struct_sim = float(match.group(2))
                    is_diverse = "is diverse" in line
                    self.diversity_checks.append((color_sim, struct_sim, is_diverse))
            
            # Dissimilar selections
            if "[DISSIMILAR] Selected" in line:
                # Extract cache ID and similarities
                match_dual = re.search(r'cache_(\d+).*color:([\d.]+).*struct:([\d.]+).*dissim:([\d.]+)', line)
                
                if match_dual:
                    cache_id = f"cache_{match_dual.group(1)}"
                    color_sim = float(match_dual.group(2))
                    struct_sim = float(match_dual.group(3))
                    dissim = float(match_dual.group(4))
                    self.dissimilar_selections.append((cache_id, color_sim, struct_sim, dissim))
            
            # Convergence detection
            if "[CONVERGING]" in line or "[COLLAPSE]" in line:
                self.convergence_events.append(line.strip())
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        
        print("\n" + "="*70)
        print("🎨 DUAL-METRIC CACHE ANALYSIS")
        print("="*70)
        
        # Keyframe generation summary
        total_kf = len(self.keyframes_generated)
        generated_kf = sum(1 for _, t in self.keyframes_generated if t == "generated")
        cache_kf = sum(1 for _, t in self.keyframes_generated if t == "cache")
        seed_kf = sum(1 for _, t in self.keyframes_generated if t == "seed")
        
        print(f"\n📊 KEYFRAME GENERATION SUMMARY (Total: {total_kf})")
        print(f"  {'Generated:':<20} {generated_kf:>3} ({self._pct(generated_kf, total_kf)})")
        print(f"  {'Cache Injections:':<20} {cache_kf:>3} ({self._pct(cache_kf, total_kf)})")
        print(f"  {'Seed Injections:':<20} {seed_kf:>3} ({self._pct(seed_kf, total_kf)})")
        
        # Cache injection timeline
        if self.cache_injections:
            print(f"\n  Cache injection timeline: KF {self.cache_injections}")
        if self.seed_injections:
            print(f"  Seed injection timeline:  KF {self.seed_injections}")
        
        # Cache population
        print(f"\n💾 CACHE POPULATION")
        print(f"  {'Frames cached:':<20} {len(self.cache_additions)}")
        if self.cache_additions:
            final_size = self.cache_additions[-1][1]
            print(f"  {'Final cache size:':<20} {final_size}")
            print(f"\n  Cache growth timeline:")
            for cache_id, total in self.cache_additions:
                print(f"    {cache_id} -> total: {total}")
        
        # Diversity check analysis
        if self.diversity_checks:
            print(f"\n🎯 DIVERSITY CHECK ANALYSIS ({len(self.diversity_checks)} checks)")
            
            diverse = [c for c in self.diversity_checks if c[2]]
            redundant = [c for c in self.diversity_checks if not c[2]]
            
            print(f"  {'Diverse (cached):':<20} {len(diverse):>3} ({self._pct(len(diverse), len(self.diversity_checks))})")
            print(f"  {'Redundant (skipped):':<20} {len(redundant):>3} ({self._pct(len(redundant), len(self.diversity_checks))})")
            
            # Color similarity stats
            all_color_sims = [c[0] for c in self.diversity_checks]
            diverse_color_sims = [c[0] for c in diverse]
            redundant_color_sims = [c[0] for c in redundant]
            
            print(f"\n  📊 ColorHist Similarity Distribution:")
            print(f"    {'All frames:':<20} min={min(all_color_sims):.3f}, max={max(all_color_sims):.3f}, mean={statistics.mean(all_color_sims):.3f}")
            if diverse_color_sims:
                print(f"    {'Diverse frames:':<20} min={min(diverse_color_sims):.3f}, max={max(diverse_color_sims):.3f}, mean={statistics.mean(diverse_color_sims):.3f}")
            if redundant_color_sims:
                print(f"    {'Redundant frames:':<20} min={min(redundant_color_sims):.3f}, max={max(redundant_color_sims):.3f}, mean={statistics.mean(redundant_color_sims):.3f}")
            
            # Struct similarity stats
            all_struct_sims = [c[1] for c in self.diversity_checks]
            diverse_struct_sims = [c[1] for c in diverse]
            redundant_struct_sims = [c[1] for c in redundant]
            
            print(f"\n  🏗️  pHash-8 Similarity Distribution:")
            print(f"    {'All frames:':<20} min={min(all_struct_sims):.3f}, max={max(all_struct_sims):.3f}, mean={statistics.mean(all_struct_sims):.3f}")
            if diverse_struct_sims:
                print(f"    {'Diverse frames:':<20} min={min(diverse_struct_sims):.3f}, max={max(diverse_struct_sims):.3f}, mean={statistics.mean(diverse_struct_sims):.3f}")
            if redundant_struct_sims:
                print(f"    {'Redundant frames:':<20} min={min(redundant_struct_sims):.3f}, max={max(redundant_struct_sims):.3f}, mean={statistics.mean(redundant_struct_sims):.3f}")
        
        # Dissimilar selection analysis
        if self.dissimilar_selections:
            print(f"\n🔄 INJECTION SELECTION ANALYSIS ({len(self.dissimilar_selections)} injections)")
            
            # Count which cache frames were selected
            cache_ids = [s[0] for s in self.dissimilar_selections]
            cache_counter = Counter(cache_ids)
            
            print(f"  Cache frame usage:")
            for cache_id, count in cache_counter.most_common():
                pct = self._pct(count, len(self.dissimilar_selections))
                print(f"    {cache_id}: {count} times ({pct})")
            
            # Check for looping (>50% on single frame)
            if cache_counter.most_common(1)[0][1] / len(self.dissimilar_selections) > 0.5:
                print(f"\n  ⚠️  LOOPING DETECTED: {cache_counter.most_common(1)[0][0]} used >50% of time")
            
            # Similarity analysis
            dual_metric_selections = [s for s in self.dissimilar_selections if s[2] is not None]
            
            if dual_metric_selections:
                print(f"\n  📊 Dual-Metric Injection Similarities:")
                color_sims = [s[1] for s in dual_metric_selections]
                struct_sims = [s[2] for s in dual_metric_selections]
                
                print(f"    {'ColorHist:':<15} min={min(color_sims):.3f}, max={max(color_sims):.3f}, mean={statistics.mean(color_sims):.3f}")
                print(f"    {'pHash-8:':<15} min={min(struct_sims):.3f}, max={max(struct_sims):.3f}, mean={statistics.mean(struct_sims):.3f}")
        
        # Convergence analysis
        if self.convergence_events:
            print(f"\n⚠️  CONVERGENCE EVENTS ({len(self.convergence_events)})")
            for event in self.convergence_events[:10]:  # Show first 10
                # Extract just the important part
                if "[CONVERGING]" in event:
                    print(f"  {event.split('[CONVERGING]')[1].strip()}")
                elif "[COLLAPSE]" in event:
                    print(f"  ❌ {event.split('[COLLAPSE]')[1].strip()}")
            
            if len(self.convergence_events) > 10:
                print(f"  ... and {len(self.convergence_events) - 10} more")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        
        if self.diversity_checks:
            diverse_rate = len([c for c in self.diversity_checks if c[2]]) / len(self.diversity_checks)
            
            # Calculate suggested thresholds based on observed data
            all_color = [c[0] for c in self.diversity_checks]
            all_struct = [c[1] for c in self.diversity_checks]
            
            color_mean = statistics.mean(all_color)
            struct_mean = statistics.mean(all_struct)
            
            # Target: allow ~30-40% of frames to be cached
            # Strategy: set threshold at ~60th percentile (allows bottom 40%)
            sorted_color = sorted(all_color)
            sorted_struct = sorted(all_struct)
            
            color_40th_pct = sorted_color[int(len(sorted_color) * 0.4)] if sorted_color else color_mean
            struct_40th_pct = sorted_struct[int(len(sorted_struct) * 0.4)] if sorted_struct else struct_mean
            
            if diverse_rate < 0.20:
                print(f"  ⚠️  Cache acceptance rate too low ({diverse_rate:.1%})")
                print(f"      Current ColorHist mean: {color_mean:.3f}")
                print(f"      → Suggested diversity_threshold: {color_40th_pct:.2f} (40th percentile)")
                print(f"      Current pHash mean: {struct_mean:.3f}")
                print(f"      → Suggested diversity_threshold: {struct_40th_pct:.2f} (40th percentile)")
            elif diverse_rate > 0.50:
                print(f"  ⚠️  Cache acceptance rate too high ({diverse_rate:.1%})")
                print(f"      → Decrease diversity_threshold (ColorHist: -0.1, pHash: -0.05)")
            else:
                print(f"  ✓ Cache acceptance rate healthy ({diverse_rate:.1%})")
        
        if self.dissimilar_selections:
            cache_counter = Counter([s[0] for s in self.dissimilar_selections])
            max_usage = cache_counter.most_common(1)[0][1] / len(self.dissimilar_selections)
            
            if max_usage > 0.5:
                print(f"  ⚠️  Injection looping detected ({max_usage:.1%} on single frame)")
                print(f"      → Widen dissimilarity_range or increase anti-loop penalty")
                
                # Suggest ranges based on observed injection similarities
                dual_metric_inj = [s for s in self.dissimilar_selections if s[2] is not None]
                if dual_metric_inj:
                    color_inj_sims = [s[1] for s in dual_metric_inj]
                    struct_inj_sims = [s[2] for s in dual_metric_inj]
                    
                    print(f"      Observed injection ColorHist: {min(color_inj_sims):.2f}-{max(color_inj_sims):.2f}")
                    print(f"      → Suggested range: [{min(color_inj_sims)*0.9:.2f}, {max(color_inj_sims)*1.1:.2f}]")
                    print(f"      Observed injection pHash: {min(struct_inj_sims):.2f}-{max(struct_inj_sims):.2f}")
                    print(f"      → Suggested range: [{min(struct_inj_sims)*0.9:.2f}, {max(struct_inj_sims)*1.1:.2f}]")
            elif max_usage < 0.3:
                print(f"  ✓ Good injection rotation (max {max_usage:.1%} on any frame)")
            
            # Check if all selections are same frame
            if len(cache_counter) == 1:
                print(f"  ⚠️  ALL injections from same cache frame!")
                print(f"      → Dissimilarity ranges likely wrong for your aesthetic")
        
        if len(self.cache_additions) < total_kf * 0.2:
            print(f"  ⚠️  Cache under-populated ({len(self.cache_additions)} frames)")
            print(f"      → Thresholds too strict for this aesthetic space")
        
        print("\n" + "="*70)
    
    def _pct(self, value: int, total: int) -> str:
        """Format percentage"""
        if total == 0:
            return "0.0%"
        return f"{value/total*100:.1f}%"
    
    def print_timeline(self, max_events: int = 20):
        """Print timeline of key events"""
        print(f"\n📅 EVENT TIMELINE (first {max_events} events)")
        print("-"*70)
        
        # Create timeline from keyframes
        events = []
        for kf_num, kf_type in self.keyframes_generated[:max_events]:
            if kf_type == "generated":
                events.append(f"KF{kf_num:03d}: Generated")
            elif kf_type == "cache":
                # Find which cache was used
                for selection in self.dissimilar_selections:
                    # Match by proximity in log
                    events.append(f"KF{kf_num:03d}: CACHE INJECTION ({selection[0]})")
                    break
                else:
                    events.append(f"KF{kf_num:03d}: CACHE INJECTION")
            elif kf_type == "seed":
                events.append(f"KF{kf_num:03d}: SEED INJECTION")
        
        for event in events:
            print(f"  {event}")
        
        if len(self.keyframes_generated) > max_events:
            print(f"  ... and {len(self.keyframes_generated) - max_events} more keyframes")
    
    def analyze(self):
        """Run full analysis"""
        self.load_log()
        self.parse_log()
        self.print_summary()
        self.print_timeline()
    
    def export_stats(self, output_path: Optional[Path] = None):
        """Export statistics to JSON file"""
        import json
        
        stats = {
            "total_keyframes": len(self.keyframes_generated),
            "generated": sum(1 for _, t in self.keyframes_generated if t == "generated"),
            "cache_injections": len(self.cache_injections),
            "seed_injections": len(self.seed_injections),
            "cache_size": len(self.cache_additions),
            "diversity_checks": {
                "total": len(self.diversity_checks),
                "diverse": sum(1 for _, _, d in self.diversity_checks if d),
                "redundant": sum(1 for _, _, d in self.diversity_checks if not d),
            }
        }
        
        if self.diversity_checks:
            all_color = [c[0] for c in self.diversity_checks]
            all_struct = [c[1] for c in self.diversity_checks]
            
            stats["color_similarity"] = {
                "min": min(all_color),
                "max": max(all_color),
                "mean": statistics.mean(all_color),
                "median": statistics.median(all_color)
            }
            
            stats["struct_similarity"] = {
                "min": min(all_struct),
                "max": max(all_struct),
                "mean": statistics.mean(all_struct),
                "median": statistics.median(all_struct)
            }
        
        if self.dissimilar_selections:
            cache_counter = Counter([s[0] for s in self.dissimilar_selections])
            stats["injection_distribution"] = dict(cache_counter)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Stats exported to {output_path}")
        
        return stats


def main():
    """Main entry point"""
    
    # Get log file path
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Default path
        log_path = Path("logs/dream_controller.log")
    
    # Run analysis
    analyzer = CacheLogAnalyzer(log_path)
    analyzer.analyze()
    
    # Optionally export stats
    if "--export" in sys.argv:
        output_path = Path("logs/cache_analysis.json")
        analyzer.export_stats(output_path)


if __name__ == "__main__":
    main()

