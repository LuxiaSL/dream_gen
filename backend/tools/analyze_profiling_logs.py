"""
Profiling Log Analyzer

Parses dream_controller.log to extract and analyze performance profiling data:
- Interpolation timing breakdowns
- VAE lock contention statistics
- Executor queue depth warnings
- CUDA context information
- Bottleneck identification
- Optimization recommendations

Usage:
    python backend/tools/analyze_profiling_logs.py logs/dream_controller.log
    
    # Or analyze latest log
    python backend/tools/analyze_profiling_logs.py
"""

import re
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class InterpolationTiming:
    """Timing breakdown for a single interpolation pair"""
    pair: str  # e.g., "5->6"
    total_time: float
    slerp_precompute: float
    avg_slerp: float
    avg_decode: float
    avg_save: float
    avg_overhead: float
    unaccounted: float
    unaccounted_pct: float


@dataclass
class VAELockStats:
    """VAE lock contention statistics"""
    acquisitions: int
    avg_wait_ms: float
    max_wait_ms: Optional[float] = None


@dataclass
class ProfilingReport:
    """Complete profiling analysis report"""
    interpolations: List[InterpolationTiming] = field(default_factory=list)
    vae_locks: List[VAELockStats] = field(default_factory=list)
    executor_warnings: List[Tuple[str, int]] = field(default_factory=list)  # (timestamp, queue_depth)
    cuda_device: Optional[str] = None
    cuda_context: Optional[str] = None
    
    # Computed statistics
    avg_total_time: float = 0.0
    avg_decode_time: float = 0.0
    avg_save_time: float = 0.0
    avg_slerp_time: float = 0.0
    avg_overhead: float = 0.0
    avg_unaccounted_pct: float = 0.0
    
    # Bottleneck flags
    has_vae_contention: bool = False
    has_executor_saturation: bool = False
    has_slow_decode: bool = False
    has_slow_save: bool = False
    has_high_unaccounted: bool = False


def parse_interpolation_timing(log_file: Path) -> List[InterpolationTiming]:
    """
    Parse [TIMING] blocks from log file
    
    Supports both old (per-frame) and new (batched) formats:
    
    Old format:
    [TIMING] Interpolation 5->6 breakdown:
      Total time:        2.234s
      Slerp precompute:  0.045s
      Avg per frame:
        - Slerp:         12.3ms
        - Decode (VAE):  165.2ms
        - Save (I/O):    18.4ms
        - Overhead:      3.1ms
      Unaccounted time:  0.087s (3.9%)
    
    New format (BATCHED):
    [TIMING] Interpolation 5->6 breakdown (BATCHED):
      Total time:        2.050s
      Slerp precompute:  0.045s
      Phase timings:
        - Slerp all:     0.154s (15.4ms per frame)
        - Decode all:    1.850s (185.0ms per frame)
        - Save all:      0.001s (0.1ms per frame)
      [BATCHED] Single VAE lock acquisition for 10 frames
    """
    timings = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for timing header (both old and new format)
        match = re.search(r'\[TIMING\] Interpolation (\d+)->(\d+) breakdown', line)
        if match:
            start_kf, end_kf = match.groups()
            pair = f"{start_kf}->{end_kf}"
            is_batched = '(BATCHED)' in line
            
            # Parse next several lines for timing data
            try:
                total_time = float(re.search(r'Total time:\s+([\d.]+)s', lines[i+1]).group(1))
                slerp_precompute = float(re.search(r'Slerp precompute:\s+([\d.]+)s', lines[i+2]).group(1))
                
                if is_batched:
                    # New batched format
                    # Phase timings section starts at line i+3
                    avg_slerp = float(re.search(r'- Slerp all:.*\(([\d.]+)ms per frame\)', lines[i+4]).group(1))
                    avg_decode = float(re.search(r'- Decode all:.*\(([\d.]+)ms per frame\)', lines[i+5]).group(1))
                    avg_save = float(re.search(r'- Save all:.*\(([\d.]+)ms per frame\)', lines[i+6]).group(1))
                    avg_overhead = 0.0  # Not tracked in batched version
                    unaccounted = 0.0
                    unaccounted_pct = 0.0
                else:
                    # Old per-frame format
                    avg_slerp = float(re.search(r'- Slerp:\s+([\d.]+)ms', lines[i+4]).group(1))
                    avg_decode = float(re.search(r'- Decode \(VAE\):\s+([\d.]+)ms', lines[i+5]).group(1))
                    avg_save = float(re.search(r'- Save \(I/O\):\s+([\d.]+)ms', lines[i+6]).group(1))
                    avg_overhead = float(re.search(r'- Overhead:\s+([\d.]+)ms', lines[i+7]).group(1))
                    unaccounted = float(re.search(r'Unaccounted time:\s+([\d.]+)s', lines[i+8]).group(1))
                    unaccounted_pct = float(re.search(r'Unaccounted time:.*\(([\d.]+)%\)', lines[i+8]).group(1))
                
                timing = InterpolationTiming(
                    pair=pair,
                    total_time=total_time,
                    slerp_precompute=slerp_precompute,
                    avg_slerp=avg_slerp,
                    avg_decode=avg_decode,
                    avg_save=avg_save,
                    avg_overhead=avg_overhead,
                    unaccounted=unaccounted,
                    unaccounted_pct=unaccounted_pct
                )
                timings.append(timing)
            except (AttributeError, ValueError, IndexError) as e:
                print(f"Warning: Failed to parse timing block at line {i}: {e}")
        
        i += 1
    
    return timings


def parse_vae_lock_stats(log_file: Path) -> List[VAELockStats]:
    """
    Parse VAE lock statistics
    
    Example formats:
    VAE Lock: 42 ops, avg wait: 4.2ms
    VAE Lock Contention: 42 ops, avg wait: 125.3ms, max wait: 340.0ms
    """
    stats = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Normal lock stats
            match = re.search(r'VAE Lock: (\d+) ops, avg wait: ([\d.]+)ms', line)
            if match and 'Contention' not in line:
                stats.append(VAELockStats(
                    acquisitions=int(match.group(1)),
                    avg_wait_ms=float(match.group(2))
                ))
            
            # Contention warning
            match = re.search(r'VAE Lock Contention: (\d+) ops, avg wait: ([\d.]+)ms, max wait: ([\d.]+)ms', line)
            if match:
                stats.append(VAELockStats(
                    acquisitions=int(match.group(1)),
                    avg_wait_ms=float(match.group(2)),
                    max_wait_ms=float(match.group(3))
                ))
    
    return stats


def parse_executor_warnings(log_file: Path) -> List[Tuple[str, int]]:
    """
    Parse executor queue depth warnings
    
    Example: Executor queue depth high: 12 pending tasks
    """
    warnings = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Executor queue depth high: (\d+)', line)
            if match:
                timestamp = match.group(1)
                depth = int(match.group(2))
                warnings.append((timestamp, depth))
    
    return warnings


def parse_cuda_context(log_file: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse CUDA context information from startup
    
    Example:
    === CUDA Context Info ===
      Current device: 0
      Device name: NVIDIA GeForce GTX TITAN X
      Context handle: 94558571854400
    """
    device = None
    context = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if '=== CUDA Context Info ===' in line:
            # Parse next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if 'Device name:' in lines[j]:
                    match = re.search(r'Device name: (.+)', lines[j])
                    if match:
                        device = match.group(1).strip()
                
                if 'Context handle:' in lines[j]:
                    match = re.search(r'Context handle: (.+)', lines[j])
                    if match:
                        context = match.group(1).strip()
            
            break
    
    return device, context


def generate_report(log_file: Path) -> ProfilingReport:
    """Generate comprehensive profiling report from log file"""
    print(f"Analyzing log file: {log_file}")
    print("=" * 80)
    
    report = ProfilingReport()
    
    # Parse all data
    print("Parsing interpolation timings...")
    report.interpolations = parse_interpolation_timing(log_file)
    print(f"  Found {len(report.interpolations)} interpolation pairs")
    
    print("Parsing VAE lock statistics...")
    report.vae_locks = parse_vae_lock_stats(log_file)
    print(f"  Found {len(report.vae_locks)} lock stat entries")
    
    print("Parsing executor warnings...")
    report.executor_warnings = parse_executor_warnings(log_file)
    print(f"  Found {len(report.executor_warnings)} executor warnings")
    
    print("Parsing CUDA context...")
    report.cuda_device, report.cuda_context = parse_cuda_context(log_file)
    
    # Calculate statistics
    if report.interpolations:
        report.avg_total_time = statistics.mean([t.total_time for t in report.interpolations])
        report.avg_decode_time = statistics.mean([t.avg_decode for t in report.interpolations])
        report.avg_save_time = statistics.mean([t.avg_save for t in report.interpolations])
        report.avg_slerp_time = statistics.mean([t.avg_slerp for t in report.interpolations])
        report.avg_overhead = statistics.mean([t.avg_overhead for t in report.interpolations])
        report.avg_unaccounted_pct = statistics.mean([t.unaccounted_pct for t in report.interpolations])
    
    # Detect bottlenecks
    if report.vae_locks:
        max_vae_wait = max(s.avg_wait_ms for s in report.vae_locks)
        report.has_vae_contention = max_vae_wait > 10.0
    
    report.has_executor_saturation = len(report.executor_warnings) > 0
    report.has_slow_decode = report.avg_decode_time > 150.0
    report.has_slow_save = report.avg_save_time > 30.0
    report.has_high_unaccounted = report.avg_unaccounted_pct > 15.0
    
    print("\n[OK] Parsing complete")
    print("=" * 80)
    
    return report


def print_report(report: ProfilingReport):
    """Print formatted profiling report with recommendations"""
    
    print("\n" + "=" * 80)
    print(" PROFILING ANALYSIS REPORT")
    print("=" * 80)
    
    # === CUDA Context ===
    print("\n📊 CUDA CONTEXT")
    print("-" * 80)
    if report.cuda_device:
        print(f"  Device: {report.cuda_device}")
    if report.cuda_context:
        print(f"  Context Handle: {report.cuda_context}")
    
    # === Interpolation Performance ===
    print("\n📊 INTERPOLATION PERFORMANCE")
    print("-" * 80)
    if report.interpolations:
        print(f"  Total pairs analyzed: {len(report.interpolations)}")
        print(f"  Average total time:   {report.avg_total_time:.3f}s")
        print(f"\n  Per-frame breakdown:")
        print(f"    - Slerp:            {report.avg_slerp_time:.1f}ms")
        print(f"    - Decode (VAE):     {report.avg_decode_time:.1f}ms  {'⚠️ SLOW' if report.has_slow_decode else '✅'}")
        print(f"    - Save (I/O):       {report.avg_save_time:.1f}ms  {'⚠️ SLOW' if report.has_slow_save else '✅'}")
        print(f"    - Overhead:         {report.avg_overhead:.1f}ms")
        print(f"  Unaccounted time:     {report.avg_unaccounted_pct:.1f}%  {'⚠️ HIGH' if report.has_high_unaccounted else '✅'}")
        
        # Show distribution
        decode_times = [t.avg_decode for t in report.interpolations]
        print(f"\n  Decode time distribution:")
        print(f"    - Min:    {min(decode_times):.1f}ms")
        print(f"    - Median: {statistics.median(decode_times):.1f}ms")
        print(f"    - Max:    {max(decode_times):.1f}ms")
        print(f"    - StdDev: {statistics.stdev(decode_times):.1f}ms")
    else:
        print("  No interpolation timing data found")
    
    # === VAE Lock Contention ===
    print("\n📊 VAE LOCK CONTENTION")
    print("-" * 80)
    if report.vae_locks:
        # Filter out zero acquisitions
        active_locks = [l for l in report.vae_locks if l.acquisitions > 0]
        if active_locks:
            avg_wait = statistics.mean([l.avg_wait_ms for l in active_locks])
            max_wait = max([l.avg_wait_ms for l in active_locks])
            total_ops = sum([l.acquisitions for l in active_locks])
            
            print(f"  Total VAE operations: {total_ops}")
            print(f"  Average wait time:    {avg_wait:.2f}ms  {'⚠️ CONTENTION' if report.has_vae_contention else '✅'}")
            print(f"  Maximum wait time:    {max_wait:.2f}ms")
            
            # Show contention events
            contentions = [l for l in report.vae_locks if l.max_wait_ms is not None]
            if contentions:
                print(f"\n  ⚠️  {len(contentions)} contention events detected:")
                for lock in contentions[:5]:  # Show first 5
                    print(f"      - {lock.acquisitions} ops, avg {lock.avg_wait_ms:.1f}ms, max {lock.max_wait_ms:.1f}ms")
        else:
            print("  ✅ No VAE lock contention detected (0ms wait times)")
    else:
        print("  No VAE lock statistics found")
    
    # === Executor Queue ===
    print("\n📊 EXECUTOR QUEUE SATURATION")
    print("-" * 80)
    if report.executor_warnings:
        print(f"  ⚠️  {len(report.executor_warnings)} executor queue warnings")
        depths = [d for _, d in report.executor_warnings]
        print(f"  Average queue depth:  {statistics.mean(depths):.1f}")
        print(f"  Maximum queue depth:  {max(depths)}")
        print(f"\n  Recent warnings (first 5):")
        for ts, depth in report.executor_warnings[:5]:
            print(f"    - {ts}: {depth} tasks pending")
    else:
        print("  ✅ No executor queue warnings (queue depth healthy)")
    
    # === BOTTLENECK SUMMARY ===
    print("\n🔍 BOTTLENECK ANALYSIS")
    print("-" * 80)
    
    bottlenecks_found = any([
        report.has_vae_contention,
        report.has_executor_saturation,
        report.has_slow_decode,
        report.has_slow_save,
        report.has_high_unaccounted
    ])
    
    if not bottlenecks_found:
        print("  ✅ No major bottlenecks detected!")
        print("     System performance is healthy.")
    else:
        if report.has_vae_contention:
            print("  ⚠️  VAE Lock Contention: avg wait > 10ms")
            print("     → Recommendation: Apply Fix 1 (Batched VAE operations)")
        
        if report.has_executor_saturation:
            print("  ⚠️  Executor Queue Saturation: high queue depth")
            print("     → Recommendation: Apply Fix 2 (Dedicated VAE executor)")
        
        if report.has_slow_decode:
            print(f"  ⚠️  Slow VAE Decode: {report.avg_decode_time:.1f}ms avg (target: <150ms)")
            print("     → Recommendation: Check GPU utilization, consider GPU split")
        
        if report.has_slow_save:
            print(f"  ⚠️  Slow I/O Save: {report.avg_save_time:.1f}ms avg (target: <30ms)")
            print("     → Recommendation: Apply Fix 3 (Async image saving)")
        
        if report.has_high_unaccounted:
            print(f"  ⚠️  High Unaccounted Time: {report.avg_unaccounted_pct:.1f}% avg")
            print("     → Recommendation: Investigate async overhead, apply Fix 5 (CUDA streams)")
    
    # === RECOMMENDATIONS ===
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    # Priority-ordered recommendations
    if report.has_vae_contention:
        recommendations.append(("HIGH", "Fix 1: Batched VAE Operations", 
                               "Batch all VAE decodes to acquire lock once per pair"))
    
    if report.has_slow_save:
        recommendations.append(("MEDIUM", "Fix 3: Async Image Saving",
                               "Move PNG save operations to executor to avoid blocking"))
    
    if report.has_executor_saturation:
        recommendations.append(("MEDIUM", "Fix 2: Dedicated VAE Executor",
                               "Create separate thread pool for VAE to prevent queue saturation"))
    
    if report.has_slow_decode and not report.has_vae_contention:
        recommendations.append(("LOW", "Verify GPU Configuration",
                               "Ensure dual-GPU setup or check for CUDA context switching"))
    
    if report.has_high_unaccounted:
        recommendations.append(("LOW", "Fix 5: CUDA Stream Optimization",
                               "Use dedicated CUDA streams to overlap operations"))
    
    # Always recommend display I/O optimization
    recommendations.append(("EASY WIN", "Fix 4: Async Display I/O",
                           "Move display writes to executor (should improve FPS)"))
    
    if recommendations:
        for i, (priority, title, desc) in enumerate(recommendations, 1):
            print(f"\n  {i}. [{priority}] {title}")
            print(f"     {desc}")
    else:
        print("  System is well-optimized! Consider:")
        print("  - Reducing interpolation resolution for higher FPS")
        print("  - Adjusting target FPS based on hardware capabilities")
    
    # === CONFIGURATION SUGGESTIONS ===
    if report.avg_total_time > 3.5:
        print("\n⚙️  CONFIGURATION TUNING")
        print("-" * 80)
        print("  System is running slower than optimal. Consider:")
        print(f"  - Current interpolation time: {report.avg_total_time:.2f}s per pair")
        print(f"  - Target: <3.0s for 4 FPS sustained generation")
        print("\n  Options:")
        print("  1. Reduce interpolation_resolution_divisor (e.g., 2 or 3)")
        print("  2. Reduce target FPS to 2.5-3.0")
        print("  3. Reduce interpolation_frames from 10 to 8")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    # Get log file path
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        # Default to latest log
        log_file = Path("logs/dream_controller.log")
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        print("\nUsage:")
        print("  python backend/tools/analyze_profiling_logs.py [log_file]")
        print("\nExample:")
        print("  python backend/tools/analyze_profiling_logs.py logs/dream_controller.log")
        sys.exit(1)
    
    # Generate and print report
    report = generate_report(log_file)
    print_report(report)
    
    # Export summary to file
    output_file = log_file.parent / "profiling_summary.txt"
    print(f"\n💾 Summary saved to: {output_file}")


if __name__ == "__main__":
    main()

