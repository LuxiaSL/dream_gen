"""
Cache Log Analyzer - Enhanced Edition
Analyzes dream_controller.log to understand cache behavior and dual-metric performance

Usage:
    python backend/tools/analyze_cache_log.py [log_file]
    
    # Or from backend directory:
    python tools/analyze_cache_log.py ../logs/dream_controller.log
    
    # With JSON export:
    python backend/tools/analyze_cache_log.py --export
    
    # With ASCII visualizations:
    python backend/tools/analyze_cache_log.py --visualize

Output:
    - Cache population statistics
    - Injection patterns and frequencies
    - Similarity distributions (ColorHist and pHash)
    - Convergence detection analysis
    - Timeline of events
    - Temporal pattern analysis
    - Cache utilization metrics
    - Similarity trends over time
    - Optimization recommendations
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import statistics
import json


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
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal spacing of injections"""
        temporal = {
            "cache_injection_gaps": [],
            "seed_injection_gaps": [],
            "cache_avg_gap": 0,
            "seed_avg_gap": 0,
            "cache_median_gap": 0,
            "seed_median_gap": 0,
        }
        
        # Calculate gaps between cache injections
        if len(self.cache_injections) > 1:
            for i in range(1, len(self.cache_injections)):
                gap = self.cache_injections[i] - self.cache_injections[i-1]
                temporal["cache_injection_gaps"].append(gap)
            
            temporal["cache_avg_gap"] = statistics.mean(temporal["cache_injection_gaps"])
            temporal["cache_median_gap"] = statistics.median(temporal["cache_injection_gaps"])
        
        # Calculate gaps between seed injections
        if len(self.seed_injections) > 1:
            for i in range(1, len(self.seed_injections)):
                gap = self.seed_injections[i] - self.seed_injections[i-1]
                temporal["seed_injection_gaps"].append(gap)
            
            temporal["seed_avg_gap"] = statistics.mean(temporal["seed_injection_gaps"])
            temporal["seed_median_gap"] = statistics.median(temporal["seed_injection_gaps"])
        
        return temporal
    
    def analyze_cache_utilization(self) -> Dict:
        """Analyze which cache frames are actually used"""
        utilization = {
            "total_cached": len(set(cache_id for cache_id, _ in self.cache_additions)),
            "total_used": 0,
            "usage_rate": 0.0,
            "unused_frames": [],
            "most_used": None,
            "least_used": None,
        }
        
        if not self.cache_additions:
            return utilization
        
        # Get all cache frame IDs
        all_cache_ids = set(cache_id for cache_id, _ in self.cache_additions)
        
        # Get used cache frame IDs
        if self.dissimilar_selections:
            used_cache_ids = set(s[0] for s in self.dissimilar_selections)
            utilization["total_used"] = len(used_cache_ids)
            utilization["usage_rate"] = len(used_cache_ids) / len(all_cache_ids)
            utilization["unused_frames"] = sorted(list(all_cache_ids - used_cache_ids))
            
            # Find most/least used
            usage_counter = Counter(s[0] for s in self.dissimilar_selections)
            if usage_counter:
                utilization["most_used"] = usage_counter.most_common(1)[0]
                utilization["least_used"] = usage_counter.most_common()[-1]
        
        return utilization
    
    def analyze_similarity_trends(self) -> Dict:
        """Analyze how similarity metrics trend over time"""
        trends = {
            "color_trend": "stable",
            "struct_trend": "stable",
            "color_first_half_mean": 0.0,
            "color_second_half_mean": 0.0,
            "struct_first_half_mean": 0.0,
            "struct_second_half_mean": 0.0,
        }
        
        if len(self.diversity_checks) < 10:
            return trends
        
        # Split into halves
        midpoint = len(self.diversity_checks) // 2
        first_half = self.diversity_checks[:midpoint]
        second_half = self.diversity_checks[midpoint:]
        
        # Color similarity trend
        color_first = statistics.mean([c[0] for c in first_half])
        color_second = statistics.mean([c[0] for c in second_half])
        trends["color_first_half_mean"] = color_first
        trends["color_second_half_mean"] = color_second
        
        # Determine trend (lower similarity = more diverse = positive)
        color_delta = color_second - color_first
        if abs(color_delta) < 0.05:
            trends["color_trend"] = "stable"
        elif color_delta < 0:
            trends["color_trend"] = "increasing diversity" 
        else:
            trends["color_trend"] = "decreasing diversity"
        
        # Struct similarity trend
        struct_first = statistics.mean([c[1] for c in first_half])
        struct_second = statistics.mean([c[1] for c in second_half])
        trends["struct_first_half_mean"] = struct_first
        trends["struct_second_half_mean"] = struct_second
        
        struct_delta = struct_second - struct_first
        if abs(struct_delta) < 0.03:
            trends["struct_trend"] = "stable"
        elif struct_delta < 0:
            trends["struct_trend"] = "increasing diversity"
        else:
            trends["struct_trend"] = "decreasing diversity"
        
        return trends
    
    def analyze_frame_age_patterns(self) -> Dict:
        """Analyze whether old or new frames are being selected"""
        age_patterns = {
            "avg_frame_age": 0.0,
            "age_preference": "unknown",
            "ages": [],
        }
        
        if not self.dissimilar_selections or not self.cache_additions:
            return age_patterns
        
        # Build cache ID to creation order mapping
        cache_order = {}
        for idx, (cache_id, _) in enumerate(self.cache_additions):
            cache_order[cache_id] = idx
        
        # Calculate age for each injection (how many frames ago was it cached?)
        for cache_id, _, _, _ in self.dissimilar_selections:
            if cache_id in cache_order:
                # Age = total cache size at injection - when it was cached
                age = len(self.cache_additions) - cache_order[cache_id]
                age_patterns["ages"].append(age)
        
        if age_patterns["ages"]:
            avg_age = statistics.mean(age_patterns["ages"])
            age_patterns["avg_frame_age"] = avg_age
            
            # Determine preference
            total_possible_age = len(self.cache_additions)
            age_ratio = avg_age / total_possible_age if total_possible_age > 0 else 0
            
            if age_ratio < 0.3:
                age_patterns["age_preference"] = "recent frames (fresh)"
            elif age_ratio > 0.7:
                age_patterns["age_preference"] = "old frames (vintage)"
            else:
                age_patterns["age_preference"] = "mixed (balanced)"
        
        return age_patterns
    
    def generate_ascii_histogram(self, values: List[float], title: str, bins: int = 20, width: int = 50) -> str:
        """Generate ASCII histogram"""
        if not values:
            return f"{title}: No data\n"
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val == 0:
            return f"{title}: All values are {min_val:.3f}\n"
        
        # Create bins
        bin_size = range_val / bins
        hist_bins = [0] * bins
        
        for val in values:
            bin_idx = int((val - min_val) / bin_size)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist_bins[bin_idx] += 1
        
        # Generate ASCII representation
        max_count = max(hist_bins)
        result = [f"\n{title}:"]
        result.append(f"Range: [{min_val:.3f}, {max_val:.3f}]  Mean: {statistics.mean(values):.3f}")
        result.append("-" * (width + 20))
        
        for i, count in enumerate(hist_bins):
            bin_start = min_val + (i * bin_size)
            bin_end = bin_start + bin_size
            bar_length = int((count / max_count) * width) if max_count > 0 else 0
            bar = "█" * bar_length
            result.append(f"{bin_start:6.3f}-{bin_end:6.3f} │{bar} {count}")
        
        return "\n".join(result)
    
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
        
        # Temporal pattern analysis
        temporal = self.analyze_temporal_patterns()
        if temporal["cache_injection_gaps"] or temporal["seed_injection_gaps"]:
            print(f"\n⏱️  TEMPORAL INJECTION PATTERNS")
            
            if temporal["cache_injection_gaps"]:
                print(f"  Cache injection spacing:")
                print(f"    {'Average gap:':<20} {temporal['cache_avg_gap']:.1f} keyframes")
                print(f"    {'Median gap:':<20} {temporal['cache_median_gap']:.1f} keyframes")
                print(f"    {'Gap range:':<20} {min(temporal['cache_injection_gaps'])}-{max(temporal['cache_injection_gaps'])} keyframes")
                
                # Check for regularity
                if len(temporal["cache_injection_gaps"]) > 2:
                    gap_std = statistics.stdev(temporal["cache_injection_gaps"])
                    if gap_std < 3:
                        print(f"    {'Pattern:':<20} ✓ Regular spacing (low variance)")
                    else:
                        print(f"    {'Pattern:':<20} ⚠ Irregular spacing (high variance: {gap_std:.1f})")
            
            if temporal["seed_injection_gaps"]:
                print(f"\n  Seed injection spacing:")
                print(f"    {'Average gap:':<20} {temporal['seed_avg_gap']:.1f} keyframes")
                print(f"    {'Median gap:':<20} {temporal['seed_median_gap']:.1f} keyframes")
                print(f"    {'Gap range:':<20} {min(temporal['seed_injection_gaps'])}-{max(temporal['seed_injection_gaps'])} keyframes")
        
        # Cache utilization analysis
        utilization = self.analyze_cache_utilization()
        if utilization["total_cached"] > 0:
            print(f"\n📈 CACHE UTILIZATION EFFICIENCY")
            print(f"  {'Total frames cached:':<25} {utilization['total_cached']}")
            print(f"  {'Frames actually used:':<25} {utilization['total_used']}")
            print(f"  {'Utilization rate:':<25} {utilization['usage_rate']:.1%}")
            
            if utilization["usage_rate"] < 0.2 and utilization["total_cached"] > 10:
                print(f"  ⚠️  Low utilization - only {utilization['usage_rate']:.0%} of cache is being used")
                print(f"      → Consider reducing max_size or widening dissimilarity_range")
            elif utilization["usage_rate"] > 0.8:
                print(f"  ✓ High utilization - cache is being used effectively")
            
            if utilization["unused_frames"] and len(utilization["unused_frames"]) <= 10:
                print(f"\n  Unused cache frames: {', '.join(utilization['unused_frames'])}")
        
        # Similarity trends
        trends = self.analyze_similarity_trends()
        if trends and len(self.diversity_checks) >= 10:
            print(f"\n📊 SIMILARITY TRENDS OVER TIME")
            print(f"  ColorHist similarity:")
            print(f"    {'First half mean:':<20} {trends['color_first_half_mean']:.3f}")
            print(f"    {'Second half mean:':<20} {trends['color_second_half_mean']:.3f}")
            print(f"    {'Trend:':<20} {trends['color_trend']}")
            
            print(f"\n  pHash-8 similarity:")
            print(f"    {'First half mean:':<20} {trends['struct_first_half_mean']:.3f}")
            print(f"    {'Second half mean:':<20} {trends['struct_second_half_mean']:.3f}")
            print(f"    {'Trend:':<20} {trends['struct_trend']}")
            
            # Warnings for concerning trends
            if "decreasing" in trends["color_trend"] or "decreasing" in trends["struct_trend"]:
                print(f"\n  ⚠️  Diversity is decreasing over time - system may be converging")
                print(f"      → Check convergence events and consider forcing more injections")
            elif "increasing" in trends["color_trend"] and "increasing" in trends["struct_trend"]:
                print(f"\n  ✓ Diversity is increasing - system is exploring aesthetic space well")
        
        # Frame age analysis
        age_patterns = self.analyze_frame_age_patterns()
        if age_patterns["ages"]:
            print(f"\n🕐 INJECTION FRAME AGE ANALYSIS")
            print(f"  {'Average frame age:':<25} {age_patterns['avg_frame_age']:.1f} frames old")
            print(f"  {'Age preference:':<25} {age_patterns['age_preference']}")
            
            if "vintage" in age_patterns["age_preference"]:
                print(f"  → System prefers older cache frames (potentially good for long-term diversity)")
            elif "fresh" in age_patterns["age_preference"]:
                print(f"  → System prefers recent cache frames (may limit diversity range)")
        
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
    
    def print_visualizations(self):
        """Print ASCII histogram visualizations of similarity distributions"""
        print("\n" + "="*70)
        print("📊 SIMILARITY DISTRIBUTION VISUALIZATIONS")
        print("="*70)
        
        if self.diversity_checks:
            all_color = [c[0] for c in self.diversity_checks]
            all_struct = [c[1] for c in self.diversity_checks]
            
            print(self.generate_ascii_histogram(all_color, "ColorHist Similarity Distribution", bins=15, width=40))
            print(self.generate_ascii_histogram(all_struct, "pHash-8 Similarity Distribution", bins=15, width=40))
            
            # Separate distributions for diverse vs redundant
            diverse = [c for c in self.diversity_checks if c[2]]
            redundant = [c for c in self.diversity_checks if not c[2]]
            
            if diverse and redundant:
                print("\n" + "-"*70)
                diverse_color = [c[0] for c in diverse]
                redundant_color = [c[0] for c in redundant]
                
                print(self.generate_ascii_histogram(diverse_color, "Diverse Frames - ColorHist", bins=12, width=35))
                print(self.generate_ascii_histogram(redundant_color, "Redundant Frames - ColorHist", bins=12, width=35))
        
        print("\n" + "="*70)
    
    def export_stats(self, output_path: Optional[Path] = None):
        """Export enhanced statistics to JSON file"""
        
        stats = {
            "summary": {
                "total_keyframes": len(self.keyframes_generated),
                "generated": sum(1 for _, t in self.keyframes_generated if t == "generated"),
                "cache_injections": len(self.cache_injections),
                "seed_injections": len(self.seed_injections),
                "cache_size": len(self.cache_additions),
            },
            "diversity_checks": {
                "total": len(self.diversity_checks),
                "diverse": sum(1 for _, _, d in self.diversity_checks if d),
                "redundant": sum(1 for _, _, d in self.diversity_checks if not d),
                "acceptance_rate": sum(1 for _, _, d in self.diversity_checks if d) / len(self.diversity_checks) if self.diversity_checks else 0,
            }
        }
        
        # Similarity distributions
        if self.diversity_checks:
            all_color = [c[0] for c in self.diversity_checks]
            all_struct = [c[1] for c in self.diversity_checks]
            
            stats["color_similarity"] = {
                "min": min(all_color),
                "max": max(all_color),
                "mean": statistics.mean(all_color),
                "median": statistics.median(all_color),
                "stdev": statistics.stdev(all_color) if len(all_color) > 1 else 0,
            }
            
            stats["struct_similarity"] = {
                "min": min(all_struct),
                "max": max(all_struct),
                "mean": statistics.mean(all_struct),
                "median": statistics.median(all_struct),
                "stdev": statistics.stdev(all_struct) if len(all_struct) > 1 else 0,
            }
        
        # Injection distribution
        if self.dissimilar_selections:
            cache_counter = Counter([s[0] for s in self.dissimilar_selections])
            stats["injection_distribution"] = dict(cache_counter)
        
        # Enhanced analytics
        stats["temporal_patterns"] = self.analyze_temporal_patterns()
        stats["cache_utilization"] = self.analyze_cache_utilization()
        stats["similarity_trends"] = self.analyze_similarity_trends()
        stats["frame_age_patterns"] = self.analyze_frame_age_patterns()
        
        # Timeline data
        stats["timeline"] = {
            "cache_injections": self.cache_injections,
            "seed_injections": self.seed_injections,
            "keyframe_types": [{"keyframe": kf, "type": t} for kf, t in self.keyframes_generated],
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Enhanced stats exported to {output_path}")
        
        return stats


def main():
    """Main entry point"""
    
    # Parse arguments
    args = sys.argv[1:]
    log_path = None
    enable_viz = False
    enable_export = False
    
    for arg in args:
        if arg == "--visualize" or arg == "-v":
            enable_viz = True
        elif arg == "--export" or arg == "-e":
            enable_export = True
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            print("\nOptions:")
            print("  --visualize, -v    Show ASCII histogram visualizations")
            print("  --export, -e       Export detailed stats to JSON")
            print("  --help, -h         Show this help message")
            sys.exit(0)
        elif not arg.startswith("--"):
            log_path = Path(arg)
    
    # Default log path
    if log_path is None:
        log_path = Path("logs/dream_controller.log")
    
    # Run analysis
    analyzer = CacheLogAnalyzer(log_path)
    analyzer.analyze()
    
    # Optional visualizations
    if enable_viz:
        analyzer.print_visualizations()
    
    # Optional export
    if enable_export:
        output_path = Path("logs/cache_analysis.json")
        analyzer.export_stats(output_path)


if __name__ == "__main__":
    main()

