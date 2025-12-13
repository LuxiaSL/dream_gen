"""
Performance Statistics Collector

Centralized collection and analysis of pipeline performance metrics for
understanding FPS headroom and identifying bottlenecks.

Usage:
    from utils.perf_stats import PerfStats, get_perf_stats
    
    # Record events
    perf = get_perf_stats()
    perf.record_keyframe_time(2.5)
    perf.record_interpolation_batch(15, 3.2)
    perf.record_frame_push(0.045)
    perf.record_display_frame()
    
    # Get analysis
    analysis = perf.analyze_fps_headroom()
    print(f"Achieved FPS: {analysis['achieved_fps']:.2f}")
    print(f"Bottleneck: {analysis['bottleneck']}")
    print(f"Max Theoretical FPS: {analysis['max_theoretical_fps']:.2f}")
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from statistics import mean, median, stdev
import threading

logger = logging.getLogger(__name__)

# Singleton instance
_perf_stats_instance = None
_lock = threading.Lock()


def get_perf_stats() -> 'PerfStats':
    """Get the singleton PerfStats instance"""
    global _perf_stats_instance
    if _perf_stats_instance is None:
        with _lock:
            if _perf_stats_instance is None:
                _perf_stats_instance = PerfStats()
    return _perf_stats_instance


@dataclass
class TimingSeries:
    """Track timing samples with statistics"""
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    total_count: int = 0
    total_sum: float = 0.0
    
    def add(self, value: float) -> None:
        """Add a timing sample (in seconds)"""
        self.samples.append(value)
        self.total_count += 1
        self.total_sum += value
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics for recent samples"""
        if not self.samples:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "total_count": self.total_count,
                "total_sum_s": self.total_sum,
            }
        
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        
        return {
            "count": n,
            "avg_ms": mean(self.samples) * 1000,
            "min_ms": min(self.samples) * 1000,
            "max_ms": max(self.samples) * 1000,
            "p50_ms": sorted_samples[n // 2] * 1000,
            "p95_ms": sorted_samples[int(n * 0.95)] * 1000 if n >= 20 else sorted_samples[-1] * 1000,
            "std_ms": stdev(self.samples) * 1000 if n > 1 else 0.0,
            "total_count": self.total_count,
            "total_sum_s": self.total_sum,
        }


class PerfStats:
    """
    Centralized performance statistics collector
    
    Tracks timing for each pipeline stage and calculates FPS headroom.
    """
    
    def __init__(self, window_seconds: float = 60.0):
        """
        Initialize performance stats collector
        
        Args:
            window_seconds: Time window for rate calculations
        """
        self.window_seconds = window_seconds
        self.start_time = time.time()
        
        # === Timing Series for Each Stage ===
        self.keyframe_times = TimingSeries()      # ComfyUI generation
        self.interpolation_times = TimingSeries()  # Per-batch interpolation
        self.decode_times = TimingSeries()         # VAE decode per frame
        self.push_times = TimingSeries()           # Frame push to VPS
        self.display_times = TimingSeries()        # Time between displays
        
        # === Frame Counters with Timestamps ===
        self.display_timestamps: Deque[float] = deque(maxlen=500)
        self.keyframe_timestamps: Deque[float] = deque(maxlen=100)
        self.push_timestamps: Deque[float] = deque(maxlen=500)
        
        # === Buffer Tracking ===
        self.buffer_samples: Deque[tuple] = deque(maxlen=300)  # (timestamp, seconds_buffered)
        
        # === Worker Idle Time ===
        self.keyframe_worker_busy_time = 0.0
        self.interpolation_worker_busy_time = 0.0
        self.last_keyframe_start: Optional[float] = None
        self.last_interpolation_start: Optional[float] = None
        
        # === Batch Sizes ===
        self.interpolation_batch_sizes: Deque[int] = deque(maxlen=50)
        
        logger.info("PerfStats collector initialized")
    
    # === Recording Methods ===
    
    def record_keyframe_time(self, duration_s: float) -> None:
        """Record a keyframe generation time"""
        self.keyframe_times.add(duration_s)
        self.keyframe_timestamps.append(time.time())
        self.keyframe_worker_busy_time += duration_s
    
    def record_interpolation_batch(self, frame_count: int, duration_s: float) -> None:
        """Record an interpolation batch completion"""
        self.interpolation_times.add(duration_s)
        self.interpolation_batch_sizes.append(frame_count)
        self.interpolation_worker_busy_time += duration_s
    
    def record_decode_time(self, duration_s: float) -> None:
        """Record a single VAE decode time"""
        self.decode_times.add(duration_s)
    
    def record_frame_push(self, duration_s: float) -> None:
        """Record a frame push time"""
        self.push_times.add(duration_s)
        self.push_timestamps.append(time.time())
    
    def record_display_frame(self) -> None:
        """Record that a frame was displayed"""
        now = time.time()
        if self.display_timestamps:
            interval = now - self.display_timestamps[-1]
            self.display_times.add(interval)
        self.display_timestamps.append(now)
    
    def record_buffer_status(self, seconds_buffered: float) -> None:
        """Record current buffer level"""
        self.buffer_samples.append((time.time(), seconds_buffered))
    
    # === Analysis Methods ===
    
    def get_achieved_fps(self, window_seconds: Optional[float] = None) -> float:
        """
        Calculate actual achieved display FPS over recent window
        
        Args:
            window_seconds: Time window to analyze (default: self.window_seconds)
        
        Returns:
            Frames per second achieved
        """
        window = window_seconds or self.window_seconds
        now = time.time()
        cutoff = now - window
        
        recent_frames = sum(1 for ts in self.display_timestamps if ts > cutoff)
        
        if recent_frames < 2:
            return 0.0
        
        return recent_frames / window
    
    def get_push_fps(self, window_seconds: Optional[float] = None) -> float:
        """Calculate frames pushed per second (cloud throughput)"""
        window = window_seconds or self.window_seconds
        now = time.time()
        cutoff = now - window
        
        recent_pushes = sum(1 for ts in self.push_timestamps if ts > cutoff)
        
        if recent_pushes < 2:
            return 0.0
        
        return recent_pushes / window
    
    def get_buffer_trend(self) -> Dict[str, float]:
        """
        Analyze buffer trend over recent samples
        
        Returns:
            Dict with trend analysis:
            - current: Current buffer level
            - avg_10s: Average over last 10 seconds
            - trend_per_second: Change rate (positive = growing)
            - time_to_empty: Seconds until buffer depleted (if shrinking)
        """
        if len(self.buffer_samples) < 2:
            return {
                "current": 0.0,
                "avg_10s": 0.0,
                "trend_per_second": 0.0,
                "time_to_empty": float('inf'),
            }
        
        now = time.time()
        current = self.buffer_samples[-1][1]
        
        # Average over last 10 seconds
        recent = [(ts, val) for ts, val in self.buffer_samples if ts > now - 10]
        avg_10s = mean([val for _, val in recent]) if recent else current
        
        # Calculate trend (linear regression would be better, but simple diff works)
        if len(recent) >= 2:
            first_ts, first_val = recent[0]
            last_ts, last_val = recent[-1]
            dt = last_ts - first_ts
            if dt > 0:
                trend = (last_val - first_val) / dt
            else:
                trend = 0.0
        else:
            trend = 0.0
        
        # Time to empty
        if trend < 0 and current > 0:
            time_to_empty = current / abs(trend)
        else:
            time_to_empty = float('inf')
        
        return {
            "current": current,
            "avg_10s": avg_10s,
            "trend_per_second": trend,
            "time_to_empty": time_to_empty,
        }
    
    def analyze_fps_headroom(self) -> Dict:
        """
        Comprehensive FPS headroom analysis
        
        Returns analysis including:
        - achieved_fps: Actual display FPS
        - target_fps: Configured target
        - bottleneck: Which stage is limiting
        - max_theoretical_fps: If bottleneck removed
        - headroom_percentage: How much faster we could go
        - stage_times: Breakdown by pipeline stage
        """
        elapsed = time.time() - self.start_time
        
        # Get timing stats for each stage
        kf_stats = self.keyframe_times.get_stats()
        interp_stats = self.interpolation_times.get_stats()
        decode_stats = self.decode_times.get_stats()
        push_stats = self.push_times.get_stats()
        display_stats = self.display_times.get_stats()
        
        # Calculate achieved FPS
        achieved_fps = self.get_achieved_fps(30.0)
        push_fps = self.get_push_fps(30.0)
        
        # Buffer trend
        buffer_trend = self.get_buffer_trend()
        
        # Average batch size
        avg_batch_size = mean(self.interpolation_batch_sizes) if self.interpolation_batch_sizes else 10
        
        # Calculate production rate
        # Keyframe + interpolation batch = avg_batch_size + 1 frames
        # Time = keyframe_time + interpolation_batch_time
        avg_keyframe_time = kf_stats['avg_ms'] / 1000 if kf_stats['count'] > 0 else 2.0
        avg_interp_batch_time = interp_stats['avg_ms'] / 1000 if interp_stats['count'] > 0 else 3.0
        
        frames_per_cycle = avg_batch_size + 1
        time_per_cycle = avg_keyframe_time + avg_interp_batch_time
        production_fps = frames_per_cycle / time_per_cycle if time_per_cycle > 0 else 0
        
        # Identify bottleneck
        bottlenecks = []
        
        # Check if production is behind display
        if buffer_trend['trend_per_second'] < -0.1:
            bottlenecks.append("PRODUCTION")
        
        # Check which stage is slowest
        stage_times = {
            "keyframe_gen": avg_keyframe_time,
            "interpolation_batch": avg_interp_batch_time,
            "avg_decode": decode_stats['avg_ms'] / 1000,
            "avg_push": push_stats['avg_ms'] / 1000,
        }
        slowest_stage = max(stage_times, key=stage_times.get)
        
        if slowest_stage == "keyframe_gen" and avg_keyframe_time > 2.0:
            bottlenecks.append("KEYFRAME_GEN")
        if slowest_stage == "interpolation_batch" and avg_interp_batch_time > 3.0:
            bottlenecks.append("INTERPOLATION")
        if decode_stats['avg_ms'] > 200:
            bottlenecks.append("VAE_DECODE")
        if push_stats['avg_ms'] > 100:
            bottlenecks.append("NETWORK_PUSH")
        
        # Calculate theoretical max FPS
        # If we could produce instantly, display rate would be limited by push rate
        # If we could push instantly, display rate would be limited by production rate
        max_theoretical_fps = min(production_fps * 1.5, push_fps * 1.5) if push_fps > 0 else production_fps * 1.5
        
        # Headroom
        headroom_pct = ((max_theoretical_fps - achieved_fps) / achieved_fps * 100) if achieved_fps > 0 else 0
        
        return {
            "runtime_seconds": elapsed,
            
            # FPS metrics
            "achieved_fps": achieved_fps,
            "push_fps": push_fps,
            "production_fps": production_fps,
            "max_theoretical_fps": max_theoretical_fps,
            "headroom_percentage": headroom_pct,
            
            # Buffer health
            "buffer_current": buffer_trend['current'],
            "buffer_trend": buffer_trend['trend_per_second'],
            "time_to_buffer_empty": buffer_trend['time_to_empty'],
            "buffer_healthy": buffer_trend['trend_per_second'] >= -0.1,
            
            # Bottleneck identification
            "bottleneck": bottlenecks[0] if bottlenecks else "NONE",
            "all_bottlenecks": bottlenecks,
            
            # Stage breakdown (ms)
            "stage_times_ms": {
                "keyframe_gen_avg": kf_stats['avg_ms'],
                "keyframe_gen_p95": kf_stats['p95_ms'],
                "interpolation_batch_avg": interp_stats['avg_ms'],
                "interpolation_batch_p95": interp_stats['p95_ms'],
                "vae_decode_avg": decode_stats['avg_ms'],
                "vae_decode_p95": decode_stats['p95_ms'],
                "frame_push_avg": push_stats['avg_ms'],
                "frame_push_p95": push_stats['p95_ms'],
                "display_interval_avg": display_stats['avg_ms'],
            },
            
            # Frame counts
            "total_keyframes": kf_stats['total_count'],
            "total_interpolation_batches": interp_stats['total_count'],
            "total_frames_pushed": len(self.push_timestamps),
            "total_frames_displayed": len(self.display_timestamps),
            "avg_interpolation_batch_size": avg_batch_size,
        }
    
    def log_summary(self) -> None:
        """Log a performance summary"""
        analysis = self.analyze_fps_headroom()
        
        logger.info("=" * 70)
        logger.info("[PERF_SUMMARY] Pipeline Performance Analysis")
        logger.info("=" * 70)
        logger.info(f"  Runtime: {analysis['runtime_seconds']:.0f}s")
        logger.info("")
        logger.info("  FPS Metrics:")
        logger.info(f"    Achieved Display FPS:    {analysis['achieved_fps']:.2f}")
        logger.info(f"    Frame Push FPS:          {analysis['push_fps']:.2f}")
        logger.info(f"    Production FPS:          {analysis['production_fps']:.2f}")
        logger.info(f"    Max Theoretical FPS:     {analysis['max_theoretical_fps']:.2f}")
        logger.info(f"    Headroom:                {analysis['headroom_percentage']:.0f}%")
        logger.info("")
        logger.info("  Buffer Health:")
        logger.info(f"    Current:                 {analysis['buffer_current']:.1f}s")
        logger.info(f"    Trend:                   {analysis['buffer_trend']:+.2f}s/s")
        logger.info(f"    Status:                  {'✅ HEALTHY' if analysis['buffer_healthy'] else '⚠️ DRAINING'}")
        if not analysis['buffer_healthy']:
            logger.info(f"    Time to Empty:           {analysis['time_to_buffer_empty']:.0f}s")
        logger.info("")
        logger.info(f"  Bottleneck: {analysis['bottleneck']}")
        if analysis['all_bottlenecks']:
            logger.info(f"    All issues: {', '.join(analysis['all_bottlenecks'])}")
        logger.info("")
        logger.info("  Stage Timings (avg / p95):")
        st = analysis['stage_times_ms']
        logger.info(f"    Keyframe Gen:            {st['keyframe_gen_avg']:.0f}ms / {st['keyframe_gen_p95']:.0f}ms")
        logger.info(f"    Interpolation Batch:     {st['interpolation_batch_avg']:.0f}ms / {st['interpolation_batch_p95']:.0f}ms")
        logger.info(f"    VAE Decode:              {st['vae_decode_avg']:.0f}ms / {st['vae_decode_p95']:.0f}ms")
        logger.info(f"    Frame Push:              {st['frame_push_avg']:.0f}ms / {st['frame_push_p95']:.0f}ms")
        logger.info("")
        logger.info("  Totals:")
        logger.info(f"    Keyframes:               {analysis['total_keyframes']}")
        logger.info(f"    Interpolation Batches:   {analysis['total_interpolation_batches']}")
        logger.info(f"    Frames Pushed:           {analysis['total_frames_pushed']}")
        logger.info(f"    Frames Displayed:        {analysis['total_frames_displayed']}")
        logger.info("=" * 70)
    
    def reset(self) -> None:
        """Reset all statistics"""
        self.__init__(self.window_seconds)
        logger.info("PerfStats reset")

