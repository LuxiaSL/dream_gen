"""
Mode Collapse Detector

Real-time detection of mode collapse using dual-metric embedding analysis.
Tracks embedding history and convergence trends to identify when the
generative feedback loop is converging toward a single aesthetic.

Key Features:
- Track dual-metric embedding history (configurable, default 100 frames)
- Calculate color AND structural similarity trends independently
- Detect convergence delta for BOTH metrics (recent vs early frames)
- Baseline comparison mode: compare against established healthy baseline
- Template-aware warmup period: no triggers while establishing baseline
- OR logic: trigger if EITHER metric shows convergence
- Return action recommendations (none/scale_injection/force_cache)
- Performance: ~5ms per frame (numpy operations only)
"""

import logging
from collections import deque
from typing import Dict, Optional, List, Literal, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModeCollapseDetector:
    """
    Real-time mode collapse detection using dual-metric embedding analysis
    
    Tracks:
    - Dual-metric embedding history (configurable, default 100 frames)
    - Color AND structural similarity trends independently
    - Convergence delta for BOTH metrics (baseline vs recent frames)
    - Template-aware warmup with baseline recording
    - OR logic: converging if EITHER metric exceeds threshold
    
    Baseline Mode:
    - During warmup period, frames are recorded as the "healthy baseline"
    - After warmup, comparison is against this baseline (not just early frames)
    - reset_for_template() clears history and starts fresh baseline recording
    
    Performance: ~5ms per frame (numpy operations only)
    
    Usage:
        detector = ModeCollapseDetector(similarity_manager=manager)
        
        # After each keyframe generation:
        result = detector.analyze_frame(embedding)
        
        if result["action"] == "scale_injection":
            # Scale injection probability based on convergence
            pass
        elif result["action"] == "force_cache":
            # Force cache injection (severe convergence)
            pass
        
        # On template switch:
        detector.reset_for_template("new_template_id")
    """
    
    def __init__(
        self,
        similarity_manager=None,
        history_size: int = 100,
        detection_window: int = 50,
        color_convergence_threshold: float = 0.15,
        color_force_cache_threshold: float = 0.30,
        struct_convergence_threshold: float = 0.08,
        struct_force_cache_threshold: float = 0.15,
        convergence_mode: str = "absolute",
        log_stats: bool = True,
        warmup_frames: int = 50,
        use_baseline_comparison: bool = True
    ):
        """
        Initialize mode collapse detector
        
        Args:
            similarity_manager: DualMetricSimilarityManager instance (required for dual-metric)
            history_size: Number of recent embeddings to track (default: 100)
            detection_window: Window size for convergence comparison (default: 50)
            color_convergence_threshold: Color delta threshold for scaling injection
            color_force_cache_threshold: Color delta threshold for forcing cache
            struct_convergence_threshold: Structural delta threshold for scaling injection
            struct_force_cache_threshold: Structural delta threshold for forcing cache
            convergence_mode: "absolute" or "percentage" based detection
            log_stats: Whether to log detailed convergence statistics
            warmup_frames: Frames to establish baseline before detection activates (default: 50)
            use_baseline_comparison: Compare against baseline instead of just early frames
        """
        self.similarity_manager = similarity_manager
        self.history_size = history_size
        self.detection_window = detection_window
        self.convergence_mode = convergence_mode
        self.log_stats = log_stats
        
        # Dual-metric thresholds
        self.color_convergence_threshold = color_convergence_threshold
        self.color_force_cache_threshold = color_force_cache_threshold
        self.struct_convergence_threshold = struct_convergence_threshold
        self.struct_force_cache_threshold = struct_force_cache_threshold
        
        # Warmup and baseline configuration
        self.warmup_frames = warmup_frames
        self.use_baseline_comparison = use_baseline_comparison
        
        # Embedding history (fixed size circular buffer)
        # Stores Dict[str, Any] for dual-metric embeddings {'color': ..., 'struct': ...}
        self.embedding_history: deque = deque(maxlen=history_size)
        
        # Separate similarity histories for dual-metric
        self.color_similarity_history: deque = deque(maxlen=history_size * 2)
        self.struct_similarity_history: deque = deque(maxlen=history_size * 2)
        
        # Warmup and baseline tracking
        self._frames_analyzed = 0
        self._in_warmup = True
        self._current_template_id: Optional[str] = None
        
        # Baseline statistics (captured during warmup)
        # These represent the "healthy" state for comparison
        self._baseline_color_mean: Optional[float] = None
        self._baseline_color_std: Optional[float] = None
        self._baseline_struct_mean: Optional[float] = None
        self._baseline_struct_std: Optional[float] = None
        self._baseline_color_similarities: List[float] = []
        self._baseline_struct_similarities: List[float] = []
        
        logger.info(
            f"ModeCollapseDetector initialized (dual-metric mode, "
            f"history_size={history_size}, detection_window={detection_window}, "
            f"warmup_frames={warmup_frames}, baseline_comparison={use_baseline_comparison})"
        )
        logger.info(
            f"  Thresholds: color={color_convergence_threshold}/{color_force_cache_threshold}, "
            f"struct={struct_convergence_threshold}/{struct_force_cache_threshold}"
        )
    
    def analyze_frame(self, embedding: Any) -> Dict[str, any]:
        """
        Analyze single frame for collapse indicators
        
        Handles warmup period: during warmup, frames are recorded to establish
        a baseline but no collapse triggers fire. After warmup, comparison is
        made against the baseline (or early frames if baseline comparison disabled).
        
        Args:
            embedding: Dual-metric embedding {'color': ndarray, 'struct': str}
        
        Returns:
            Dictionary with analysis results:
            For dual-metric:
            {
                "status": "ok" | "converging" | "collapsed" | "warmup",
                "avg_color_similarity": float,
                "avg_struct_similarity": float,
                "color_delta": float,
                "struct_delta": float,
                "convergence_delta": float,  # Max of color/struct deltas
                "action": "none" | "scale_injection" | "force_cache",
                "scaled_injection_probability": float,
                "trigger_reason": str,  # Which metric(s) triggered
                "in_warmup": bool,
                "frames_analyzed": int,
                "warmup_progress": float  # 0.0 to 1.0
            }
        """
        # Add to history
        self.embedding_history.append(embedding)
        self._frames_analyzed += 1
        
        # Base result structure
        base_result = {
            "in_warmup": self._in_warmup,
            "frames_analyzed": self._frames_analyzed,
            "warmup_progress": min(1.0, self._frames_analyzed / self.warmup_frames) if self.warmup_frames > 0 else 1.0
        }
        
        # Need at least 2 frames to calculate similarity
        if len(self.embedding_history) < 2:
            return {
                **base_result,
                "status": "warmup" if self._in_warmup else "ok",
                "action": "none",
                "avg_similarity": 0.0,
                "avg_color_similarity": 0.0,
                "avg_struct_similarity": 0.0,
                "variance": 0.0,
                "convergence_delta": 0.0,
                "color_delta": 0.0,
                "struct_delta": 0.0,
                "scaled_injection_probability": 0.0,
                "trigger_reason": "insufficient history",
                "should_force_mutation": False
            }
        
        # Calculate current similarities and record to history
        result = self._analyze_dual_metric()
        result.update(base_result)
        
        # Handle warmup period
        if self._in_warmup:
            # Record similarities for baseline calculation
            if result["avg_color_similarity"] > 0:
                self._baseline_color_similarities.append(result["avg_color_similarity"])
            if result["avg_struct_similarity"] > 0:
                self._baseline_struct_similarities.append(result["avg_struct_similarity"])
            
            # Check if warmup complete
            if self._frames_analyzed >= self.warmup_frames:
                self._finalize_baseline()
                self._in_warmup = False
                logger.info(
                    f"[WARMUP_COMPLETE] Baseline established after {self._frames_analyzed} frames. "
                    f"Color baseline: {self._baseline_color_mean:.4f} ± {self._baseline_color_std:.4f}, "
                    f"Struct baseline: {self._baseline_struct_mean:.4f} ± {self._baseline_struct_std:.4f}"
                )
            
            # During warmup, never trigger actions
            result["status"] = "warmup"
            result["action"] = "none"
            result["scaled_injection_probability"] = 0.0
            result["trigger_reason"] = f"warmup ({self._frames_analyzed}/{self.warmup_frames})"
        
        return result
    
    def _finalize_baseline(self) -> None:
        """Calculate baseline statistics from warmup period similarities"""
        if self._baseline_color_similarities:
            self._baseline_color_mean = float(np.mean(self._baseline_color_similarities))
            self._baseline_color_std = float(np.std(self._baseline_color_similarities))
        else:
            self._baseline_color_mean = 0.0
            self._baseline_color_std = 0.1  # Default spread
        
        if self._baseline_struct_similarities:
            self._baseline_struct_mean = float(np.mean(self._baseline_struct_similarities))
            self._baseline_struct_std = float(np.std(self._baseline_struct_similarities))
        else:
            self._baseline_struct_mean = 0.0
            self._baseline_struct_std = 0.1
        
        logger.debug(
            f"Baseline finalized: color={self._baseline_color_mean:.4f}±{self._baseline_color_std:.4f}, "
            f"struct={self._baseline_struct_mean:.4f}±{self._baseline_struct_std:.4f}"
        )
    
    def _analyze_dual_metric(self) -> Dict[str, any]:
        """
        Analyze using dual-metric (ColorHist + pHash-8) with OR logic
        
        Comparison modes:
        - Baseline comparison (default): Compare recent window against established baseline
        - Early-frame comparison: Compare recent window against early window (legacy)
        """
        
        # Calculate recent pairwise similarities for BOTH metrics
        recent = list(self.embedding_history)[-10:]
        color_sims = []
        struct_sims = []
        
        for i in range(len(recent) - 1):
            if isinstance(recent[i], dict) and isinstance(recent[i+1], dict):
                color_sim = self.similarity_manager.get_color_similarity(recent[i], recent[i+1])
                struct_sim = self.similarity_manager.get_struct_similarity(recent[i], recent[i+1])
                color_sims.append(color_sim)
                struct_sims.append(struct_sim)
        
        avg_color_sim = float(np.mean(color_sims)) if color_sims else 0.0
        avg_struct_sim = float(np.mean(struct_sims)) if struct_sims else 0.0
        
        # Track similarity over time
        self.color_similarity_history.append(avg_color_sim)
        self.struct_similarity_history.append(avg_struct_sim)
        
        # Detect convergence for BOTH metrics
        color_delta = 0.0
        struct_delta = 0.0
        status: Literal["ok", "converging", "collapsed"] = "ok"
        action: Literal["none", "scale_injection", "force_cache"] = "none"
        scaled_injection_probability = 0.0
        trigger_reason = "none"
        should_force_mutation = False  # Initialize outside the history check block
        
        # Determine required history length based on detection window
        # Need at least detection_window * 2 samples, or 40 minimum for backwards compat
        min_history = max(40, self.detection_window * 2)
        
        if len(self.color_similarity_history) >= min_history and len(self.struct_similarity_history) >= min_history:
            # Get recent averages (using detection_window)
            recent_color_avg = float(np.mean(list(self.color_similarity_history)[-self.detection_window:]))
            recent_struct_avg = float(np.mean(list(self.struct_similarity_history)[-self.detection_window:]))
            
            # Get comparison baseline
            if self.use_baseline_comparison and self._baseline_color_mean is not None:
                # Compare against established baseline from warmup period
                baseline_color = self._baseline_color_mean
                baseline_struct = self._baseline_struct_mean
                comparison_source = "baseline"
            else:
                # Compare against early frames (legacy behavior)
                baseline_color = float(np.mean(list(self.color_similarity_history)[:self.detection_window]))
                baseline_struct = float(np.mean(list(self.struct_similarity_history)[:self.detection_window]))
                comparison_source = "early_frames"
            
            color_delta = recent_color_avg - baseline_color
            struct_delta = recent_struct_avg - baseline_struct
            
            # Log calibration stats if enabled
            if self.log_stats:
                logger.debug(
                    f"[CALIBRATION] ({comparison_source}) "
                    f"COLOR: baseline={baseline_color:.4f}, recent={recent_color_avg:.4f}, delta={color_delta:.4f} | "
                    f"STRUCT: baseline={baseline_struct:.4f}, recent={recent_struct_avg:.4f}, delta={struct_delta:.4f}"
                )
            
            # OR LOGIC: Check if EITHER metric exceeds its threshold
            # Thresholds hierarchy (from plan):
            #   convergence_threshold → force_mutation (soft intervention)
            #   scaled midpoint → scale_injection + force_mutation (medium)
            #   force_cache_threshold → force_cache + force_mutation (hard)
            color_force = color_delta > self.color_force_cache_threshold
            color_scale = color_delta > self.color_convergence_threshold
            struct_force = struct_delta > self.struct_force_cache_threshold
            struct_scale = struct_delta > self.struct_convergence_threshold
            
            # Track whether mutation should be forced (at ANY convergence level)
            should_force_mutation = False
            
            # Determine action based on OR logic (most severe action wins)
            if color_force or struct_force:
                status = "collapsed"
                action = "force_cache"
                scaled_injection_probability = 1.0
                should_force_mutation = True  # Mutation accompanies hard intervention
                reasons = []
                if color_force:
                    reasons.append(f"COLOR={color_delta:.4f}>{self.color_force_cache_threshold:.4f}")
                if struct_force:
                    reasons.append(f"STRUCT={struct_delta:.4f}>{self.struct_force_cache_threshold:.4f}")
                trigger_reason = " AND ".join(reasons)
                logger.warning(f"[COLLAPSE] Severe convergence! {trigger_reason} -> forcing cache (100%) + mutation")
            
            elif color_scale or struct_scale:
                status = "converging"
                action = "scale_injection"
                should_force_mutation = True  # Mutation accompanies medium intervention
                
                # Calculate scaling factor for each metric
                color_progress = 0.0
                if color_scale:
                    color_range = self.color_force_cache_threshold - self.color_convergence_threshold
                    color_progress = (color_delta - self.color_convergence_threshold) / color_range if color_range > 0 else 0.0
                
                struct_progress = 0.0
                if struct_scale:
                    struct_range = self.struct_force_cache_threshold - self.struct_convergence_threshold
                    struct_progress = (struct_delta - self.struct_convergence_threshold) / struct_range if struct_range > 0 else 0.0
                
                # OR logic: Use maximum progress (most severe metric)
                scaled_injection_probability = min(1.0, max(color_progress, struct_progress))
                
                reasons = []
                if color_scale:
                    reasons.append(f"COLOR={color_delta:.4f}")
                if struct_scale:
                    reasons.append(f"STRUCT={struct_delta:.4f}")
                trigger_reason = " OR ".join(reasons)
                
                logger.info(
                    f"[CONVERGING] {trigger_reason} -> scaling to {scaled_injection_probability:.0%} + mutation"
                )
            else:
                status = "ok"
                action = "none"
                scaled_injection_probability = 0.0
                should_force_mutation = False
                trigger_reason = "no convergence"
        
        return {
            "status": status,
            "avg_similarity": max(avg_color_sim, avg_struct_sim),  # For compatibility
            "avg_color_similarity": avg_color_sim,
            "avg_struct_similarity": avg_struct_sim,
            "variance": 0.0,  # Not used in dual-metric
            "convergence_delta": max(abs(color_delta), abs(struct_delta)),  # Max delta
            "color_delta": color_delta,
            "struct_delta": struct_delta,
            "action": action,
            "scaled_injection_probability": scaled_injection_probability,
            "trigger_reason": trigger_reason,
            "should_force_mutation": should_force_mutation  # NEW: trigger mutation at any convergence level
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics for logging
        
        Returns:
            Dictionary with dual-metric statistics including warmup and baseline info
        """
        base_stats = {
            "frames_analyzed": self._frames_analyzed,
            "in_warmup": self._in_warmup,
            "warmup_frames": self.warmup_frames,
            "warmup_progress": min(1.0, self._frames_analyzed / self.warmup_frames) if self.warmup_frames > 0 else 1.0,
            "current_template_id": self._current_template_id,
            "use_baseline_comparison": self.use_baseline_comparison,
            "baseline_color_mean": self._baseline_color_mean,
            "baseline_color_std": self._baseline_color_std,
            "baseline_struct_mean": self._baseline_struct_mean,
            "baseline_struct_std": self._baseline_struct_std,
        }
        
        if not self.color_similarity_history:
            return {
                **base_stats,
                "recent_avg_color_similarity": 0.0,
                "recent_avg_struct_similarity": 0.0,
                "overall_avg_color_similarity": 0.0,
                "overall_avg_struct_similarity": 0.0,
                "color_variance": 0.0,
                "struct_variance": 0.0,
            }
        
        recent_color = list(self.color_similarity_history)[-10:]
        recent_struct = list(self.struct_similarity_history)[-10:] if self.struct_similarity_history else []
        
        return {
            **base_stats,
            "recent_avg_color_similarity": float(np.mean(recent_color)) if recent_color else 0.0,
            "recent_avg_struct_similarity": float(np.mean(recent_struct)) if recent_struct else 0.0,
            "overall_avg_color_similarity": float(np.mean(self.color_similarity_history)),
            "overall_avg_struct_similarity": float(np.mean(self.struct_similarity_history)) if self.struct_similarity_history else 0.0,
            "color_variance": float(np.var(self.color_similarity_history)),
            "struct_variance": float(np.var(self.struct_similarity_history)) if self.struct_similarity_history else 0.0,
        }
    
    def reset(self) -> None:
        """
        Reset detector state
        
        Clears all histories and baseline, re-enters warmup period.
        Useful for starting a fresh analysis after major changes.
        Call this after successful injections to prevent feedback loops.
        """
        self.embedding_history.clear()
        self.color_similarity_history.clear()
        self.struct_similarity_history.clear()
        
        # Reset warmup and baseline state
        self._frames_analyzed = 0
        self._in_warmup = True
        self._baseline_color_mean = None
        self._baseline_color_std = None
        self._baseline_struct_mean = None
        self._baseline_struct_std = None
        self._baseline_color_similarities = []
        self._baseline_struct_similarities = []
        
        logger.info("ModeCollapseDetector reset (all histories and baseline cleared, re-entering warmup)")
    
    def reset_for_template(self, template_id: Optional[str] = None) -> None:
        """
        Reset detector for a new template context
        
        Called at template switch boundaries to establish a fresh baseline
        for the new aesthetic. Clears all history and re-enters warmup.
        
        This is the preferred method for template-aware systems. It:
        1. Clears all embedding and similarity history
        2. Clears baseline statistics
        3. Re-enters warmup mode to establish new baseline
        4. Records the template ID for debugging/logging
        
        Args:
            template_id: Optional identifier for the new template (for logging)
        """
        old_template = self._current_template_id
        self._current_template_id = template_id
        
        # Full reset of all state
        self.embedding_history.clear()
        self.color_similarity_history.clear()
        self.struct_similarity_history.clear()
        
        # Reset warmup and baseline
        self._frames_analyzed = 0
        self._in_warmup = True
        self._baseline_color_mean = None
        self._baseline_color_std = None
        self._baseline_struct_mean = None
        self._baseline_struct_std = None
        self._baseline_color_similarities = []
        self._baseline_struct_similarities = []
        
        if old_template:
            logger.info(
                f"ModeCollapseDetector reset for template switch: "
                f"'{old_template}' -> '{template_id}' (entering warmup for {self.warmup_frames} frames)"
            )
        else:
            logger.info(
                f"ModeCollapseDetector initialized for template '{template_id}' "
                f"(entering warmup for {self.warmup_frames} frames)"
            )
    
    def set_warmup(self, frames: int) -> None:
        """
        Set warmup period duration
        
        Can be called to adjust warmup length for specific templates
        that need more or less time to establish baseline.
        
        Args:
            frames: Number of frames for warmup period
        """
        self.warmup_frames = frames
        logger.debug(f"Warmup period set to {frames} frames")
    
    def in_warmup(self) -> bool:
        """
        Check if currently in warmup period
        
        Returns:
            True if detector is in warmup (no triggers fire)
        """
        return self._in_warmup
    
    def record_baseline(self, embedding: Any) -> None:
        """
        Explicitly record an embedding as part of the baseline
        
        Alternative to automatic baseline recording during warmup.
        Use this for manual baseline population.
        
        Args:
            embedding: Dual-metric embedding to add to baseline
        """
        # Add to history and calculate similarities
        self.embedding_history.append(embedding)
        
        if len(self.embedding_history) >= 2:
            recent = list(self.embedding_history)[-2:]
            if isinstance(recent[0], dict) and isinstance(recent[1], dict):
                color_sim = self.similarity_manager.get_color_similarity(recent[0], recent[1])
                struct_sim = self.similarity_manager.get_struct_similarity(recent[0], recent[1])
                self._baseline_color_similarities.append(color_sim)
                self._baseline_struct_similarities.append(struct_sim)
        
        self._frames_analyzed += 1
    
    def partial_reset(self, keep_recent: int = 5):
        """
        Partial reset: keep only recent frames
        
        Less aggressive than full reset. Useful after injections to break
        convergence signal but maintain some history context.
        
        Note: Partial reset does NOT clear baseline or re-enter warmup.
        The established baseline is preserved for continued comparison.
        
        Args:
            keep_recent: Number of recent frames to keep
        """
        if len(self.embedding_history) > keep_recent:
            recent_embeddings = list(self.embedding_history)[-keep_recent:]
            self.embedding_history.clear()
            self.embedding_history.extend(recent_embeddings)
            
            # Also trim similarity histories
            if self.color_similarity_history:
                recent_color = list(self.color_similarity_history)[-keep_recent:]
                self.color_similarity_history.clear()
                self.color_similarity_history.extend(recent_color)
            
            if self.struct_similarity_history:
                recent_struct = list(self.struct_similarity_history)[-keep_recent:]
                self.struct_similarity_history.clear()
                self.struct_similarity_history.extend(recent_struct)
            
            # Note: We preserve _frames_analyzed and baseline here
            # Only trim the rolling histories used for comparison
            
            logger.info(f"ModeCollapseDetector partial reset (kept {keep_recent} recent frames, baseline preserved)")


# Mock similarity manager for testing
class _MockSimilarityManager:
    """Mock similarity manager for standalone testing"""
    
    def get_color_similarity(self, emb1: Dict, emb2: Dict) -> float:
        """Calculate mock color similarity from embedding dicts"""
        if 'color' in emb1 and 'color' in emb2:
            c1, c2 = np.array(emb1['color']), np.array(emb2['color'])
            return float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8))
        return 0.5
    
    def get_struct_similarity(self, emb1: Dict, emb2: Dict) -> float:
        """Calculate mock structural similarity"""
        if 'struct' in emb1 and 'struct' in emb2:
            s1, s2 = np.array(emb1['struct']), np.array(emb2['struct'])
            return float(np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8))
        return 0.5


def test_collapse_detector() -> bool:
    """Test mode collapse detector with mock data"""
    print("=" * 60)
    print("Testing ModeCollapseDetector...")
    print("=" * 60)
    
    try:
        # Create detector with mock similarity manager
        print("\n1. Creating detector with template-aware settings...")
        mock_sim = _MockSimilarityManager()
        detector = ModeCollapseDetector(
            similarity_manager=mock_sim,
            history_size=100,
            detection_window=50,
            warmup_frames=20,  # Shorter warmup for test
            use_baseline_comparison=True
        )
        print("✓ Detector created")
        print(f"   history_size={detector.history_size}, detection_window={detector.detection_window}")
        print(f"   warmup_frames={detector.warmup_frames}, use_baseline_comparison={detector.use_baseline_comparison}")
        
        # Test warmup period
        print("\n2. Testing warmup period...")
        np.random.seed(42)
        
        # Generate diverse embeddings during warmup
        for i in range(20):
            embedding = {
                'color': np.random.randn(64).tolist(),
                'struct': np.random.randn(64).tolist()
            }
            result = detector.analyze_frame(embedding)
        
        stats = detector.get_stats()
        print(f"✓ After warmup ({stats['frames_analyzed']} frames):")
        print(f"   In warmup: {stats['in_warmup']}")
        print(f"   Baseline color mean: {stats['baseline_color_mean']:.4f}" if stats['baseline_color_mean'] else "   Baseline: not yet set")
        
        if stats['in_warmup']:
            print("✗ Should have exited warmup after 20 frames")
            return False
        
        print(f"✓ Warmup complete, baseline established")
        
        # Test reset_for_template
        print("\n3. Testing reset_for_template()...")
        detector.reset_for_template("test_template_v2")
        
        stats = detector.get_stats()
        print(f"   Template ID: {stats['current_template_id']}")
        print(f"   In warmup: {stats['in_warmup']}")
        print(f"   Frames analyzed: {stats['frames_analyzed']}")
        
        if not stats['in_warmup']:
            print("✗ Should re-enter warmup after reset_for_template")
            return False
        if stats['frames_analyzed'] != 0:
            print("✗ Frames analyzed should reset to 0")
            return False
        
        print("✓ reset_for_template works correctly")
        
        # Test convergence detection after warmup
        print("\n4. Testing convergence detection...")
        
        # Complete warmup with diverse frames
        for i in range(25):
            embedding = {
                'color': np.random.randn(64).tolist(),
                'struct': np.random.randn(64).tolist()
            }
            result = detector.analyze_frame(embedding)
        
        # Now generate converging frames
        base_color = np.random.randn(64)
        base_struct = np.random.randn(64)
        
        for i in range(80):
            # Reduce noise over time (simulate convergence)
            noise_factor = max(0.1, 1.0 - (i / 80.0) * 0.9)
            embedding = {
                'color': (base_color + np.random.randn(64) * noise_factor).tolist(),
                'struct': (base_struct + np.random.randn(64) * noise_factor).tolist()
            }
            result = detector.analyze_frame(embedding)
        
        stats = detector.get_stats()
        print(f"✓ After {stats['frames_analyzed']} frames:")
        print(f"   Status: {result['status']}")
        print(f"   Action: {result['action']}")
        print(f"   Color delta: {result['color_delta']:.4f}")
        print(f"   Struct delta: {result['struct_delta']:.4f}")
        
        # Test in_warmup() helper
        print("\n5. Testing helper methods...")
        print(f"   in_warmup(): {detector.in_warmup()}")
        
        detector.set_warmup(30)
        print(f"   set_warmup(30): warmup_frames={detector.warmup_frames}")
        
        # Test get_stats with all new fields
        print("\n6. Testing get_stats()...")
        stats = detector.get_stats()
        required_keys = [
            'frames_analyzed',
            'in_warmup',
            'warmup_frames',
            'warmup_progress',
            'current_template_id',
            'use_baseline_comparison',
            'baseline_color_mean',
            'baseline_struct_mean',
            'recent_avg_color_similarity',
            'recent_avg_struct_similarity'
        ]
        for key in required_keys:
            if key not in stats:
                print(f"✗ Missing stat key: {key}")
                return False
        
        print(f"✓ All stats present:")
        for key, value in sorted(stats.items()):
            if isinstance(value, float) and value is not None:
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("ModeCollapseDetector test PASSED ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_collapse_detector()
    exit(0 if success else 1)

