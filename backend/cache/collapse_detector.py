"""
Mode Collapse Detector

Real-time detection of mode collapse using dual-metric embedding analysis.
Tracks embedding history and convergence trends to identify when the
generative feedback loop is converging toward a single aesthetic.

Key Features:
- Track dual-metric embedding history (last 50 frames)
- Calculate color AND structural similarity trends independently
- Detect convergence delta for BOTH metrics (recent vs early frames)
- OR logic: trigger if EITHER metric shows convergence
- Return action recommendations (none/scale_injection/force_cache)
- Performance: ~5ms per frame (numpy operations only)
"""

import logging
from collections import deque
from typing import Dict, Optional, List, Literal, Any

import numpy as np

logger = logging.getLogger(__name__)


class ModeCollapseDetector:
    """
    Real-time mode collapse detection using dual-metric embedding analysis
    
    Tracks:
    - Dual-metric embedding history (last 50 frames)
    - Color AND structural similarity trends independently
    - Convergence delta for BOTH metrics (early vs recent frames)
    - OR logic: converging if EITHER metric exceeds threshold
    
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
    """
    
    def __init__(
        self,
        similarity_manager=None,
        history_size: int = 50,
        detection_window: int = 20,
        color_convergence_threshold: float = 0.15,
        color_force_cache_threshold: float = 0.30,
        struct_convergence_threshold: float = 0.08,
        struct_force_cache_threshold: float = 0.15,
        convergence_mode: str = "absolute",
        log_stats: bool = True
    ):
        """
        Initialize mode collapse detector
        
        Args:
            similarity_manager: DualMetricSimilarityManager instance (required for dual-metric)
            history_size: Number of recent embeddings to track
            detection_window: Window size for convergence comparison
            color_convergence_threshold: Color delta threshold for scaling injection
            color_force_cache_threshold: Color delta threshold for forcing cache
            struct_convergence_threshold: Structural delta threshold for scaling injection
            struct_force_cache_threshold: Structural delta threshold for forcing cache
            convergence_mode: "absolute" or "percentage" based detection
            log_stats: Whether to log detailed convergence statistics
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
        
        # Embedding history (fixed size circular buffer)
        # Stores Dict[str, Any] for dual-metric embeddings {'color': ..., 'struct': ...}
        self.embedding_history: deque = deque(maxlen=history_size)
        
        # Separate similarity histories for dual-metric
        self.color_similarity_history: deque = deque(maxlen=100)
        self.struct_similarity_history: deque = deque(maxlen=100)
        
        logger.info(
            f"ModeCollapseDetector initialized (dual-metric mode, "
            f"history_size={history_size}, "
            f"color_thresh={color_convergence_threshold}/{color_force_cache_threshold}, "
            f"struct_thresh={struct_convergence_threshold}/{struct_force_cache_threshold})"
        )
    
    def analyze_frame(self, embedding: Any) -> Dict[str, any]:
        """
        Analyze single frame for collapse indicators
        
        Args:
            embedding: Dual-metric embedding {'color': ndarray, 'struct': str}
        
        Returns:
            Dictionary with analysis results:
            For dual-metric:
            {
                "status": "ok" | "converging" | "collapsed",
                "avg_color_similarity": float,
                "avg_struct_similarity": float,
                "color_delta": float,
                "struct_delta": float,
                "convergence_delta": float,  # Max of color/struct deltas
                "action": "none" | "scale_injection" | "force_cache",
                "scaled_injection_probability": float,
                "trigger_reason": str  # Which metric(s) triggered
            }
        """
        # Add to history
        self.embedding_history.append(embedding)
        
        # Need at least 2 frames to calculate similarity
        if len(self.embedding_history) < 2:
            return {
                "status": "ok",
                "action": "none",
                "avg_similarity": 0.0,
                "avg_color_similarity": 0.0,
                "avg_struct_similarity": 0.0,
                "variance": 0.0,
                "convergence_delta": 0.0,
                "color_delta": 0.0,
                "struct_delta": 0.0,
                "scaled_injection_probability": 0.0,
                "trigger_reason": "insufficient history"
            }
        
        # Analyze with dual-metric system
        return self._analyze_dual_metric()
    
    def _analyze_dual_metric(self) -> Dict[str, any]:
        """Analyze using dual-metric (ColorHist + pHash-8) with OR logic"""
        
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
        
        # Detect convergence (compare recent to early) for BOTH metrics
        color_delta = 0.0
        struct_delta = 0.0
        status: Literal["ok", "converging", "collapsed"] = "ok"
        action: Literal["none", "scale_injection", "force_cache"] = "none"
        scaled_injection_probability = 0.0
        trigger_reason = "none"
        
        if len(self.color_similarity_history) >= 40 and len(self.struct_similarity_history) >= 40:
            # Compare recent window to early window for COLOR
            recent_color_avg = float(np.mean(list(self.color_similarity_history)[-20:]))
            early_color_avg = float(np.mean(list(self.color_similarity_history)[:20]))
            color_delta = recent_color_avg - early_color_avg
            
            # Compare recent window to early window for STRUCTURE
            recent_struct_avg = float(np.mean(list(self.struct_similarity_history)[-20:]))
            early_struct_avg = float(np.mean(list(self.struct_similarity_history)[:20]))
            struct_delta = recent_struct_avg - early_struct_avg
            
            # Log calibration stats if enabled
            if self.log_stats:
                logger.debug(
                    f"[CALIBRATION] COLOR: early={early_color_avg:.4f}, recent={recent_color_avg:.4f}, delta={color_delta:.4f} | "
                    f"STRUCT: early={early_struct_avg:.4f}, recent={recent_struct_avg:.4f}, delta={struct_delta:.4f}"
                )
            
            # OR LOGIC: Check if EITHER metric exceeds its threshold
            color_force = color_delta > self.color_force_cache_threshold
            color_scale = color_delta > self.color_convergence_threshold
            struct_force = struct_delta > self.struct_force_cache_threshold
            struct_scale = struct_delta > self.struct_convergence_threshold
            
            # Determine action based on OR logic (most severe action wins)
            if color_force or struct_force:
                status = "collapsed"
                action = "force_cache"
                scaled_injection_probability = 1.0
                reasons = []
                if color_force:
                    reasons.append(f"COLOR={color_delta:.4f}>{self.color_force_cache_threshold:.4f}")
                if struct_force:
                    reasons.append(f"STRUCT={struct_delta:.4f}>{self.struct_force_cache_threshold:.4f}")
                trigger_reason = " AND ".join(reasons)
                logger.warning(f"[COLLAPSE] Severe convergence! {trigger_reason} -> forcing cache (100%)")
            
            elif color_scale or struct_scale:
                status = "converging"
                action = "scale_injection"
                
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
                    f"[CONVERGING] {trigger_reason} -> scaling to {scaled_injection_probability:.0%}"
                )
            else:
                status = "ok"
                action = "none"
                scaled_injection_probability = 0.0
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
            "trigger_reason": trigger_reason
        }
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get detector statistics for logging
        
        Returns:
            Dictionary with dual-metric statistics:
            {
                "recent_avg_color_similarity": float,
                "recent_avg_struct_similarity": float,
                "overall_avg_color_similarity": float,
                "overall_avg_struct_similarity": float,
                "color_variance": float,
                "struct_variance": float,
                "frames_analyzed": int
            }
        """
        if self.color_similarity_history:
            if not self.color_similarity_history:
                return {
                    "recent_avg_color_similarity": 0.0,
                    "recent_avg_struct_similarity": 0.0,
                    "overall_avg_color_similarity": 0.0,
                    "overall_avg_struct_similarity": 0.0,
                    "color_variance": 0.0,
                    "struct_variance": 0.0,
                    "frames_analyzed": 0
                }
            
            recent_color = list(self.color_similarity_history)[-10:]
            recent_struct = list(self.struct_similarity_history)[-10:]
            
            return {
                "recent_avg_color_similarity": float(np.mean(recent_color)) if recent_color else 0.0,
                "recent_avg_struct_similarity": float(np.mean(recent_struct)) if recent_struct else 0.0,
                "overall_avg_color_similarity": float(np.mean(self.color_similarity_history)),
                "overall_avg_struct_similarity": float(np.mean(self.struct_similarity_history)),
                "color_variance": float(np.var(self.color_similarity_history)),
                "struct_variance": float(np.var(self.struct_similarity_history)),
                "frames_analyzed": len(self.embedding_history)
            }
        
        return {
            "recent_avg_color_similarity": 0.0,
            "recent_avg_struct_similarity": 0.0,
            "overall_avg_color_similarity": 0.0,
            "overall_avg_struct_similarity": 0.0,
            "color_variance": 0.0,
            "struct_variance": 0.0,
            "frames_analyzed": 0
        }
    
    def reset(self) -> None:
        """
        Reset detector state
        
        Useful for starting a fresh analysis after major changes.
        Call this after successful injections to prevent feedback loops.
        """
        self.embedding_history.clear()
        self.color_similarity_history.clear()
        self.struct_similarity_history.clear()
        logger.info("ModeCollapseDetector reset (all histories cleared)")
    
    def partial_reset(self, keep_recent: int = 5):
        """
        Partial reset: keep only recent frames
        
        Less aggressive than full reset. Useful after injections to break
        convergence signal but maintain some history context.
        
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
            
            
            logger.info(f"ModeCollapseDetector partial reset (kept {keep_recent} recent frames)")


# Test function
def test_collapse_detector() -> bool:
    """Test mode collapse detector with mock data"""
    print("=" * 60)
    print("Testing ModeCollapseDetector...")
    print("=" * 60)
    
    try:
        # Create detector
        print("\n1. Creating detector...")
        detector = ModeCollapseDetector(
            history_size=50,
            detection_window=20,
            convergence_threshold=0.1
        )
        print("✓ Detector created")
        
        # Test with diverse embeddings (should stay OK)
        print("\n2. Testing with diverse embeddings...")
        np.random.seed(42)
        for i in range(50):
            # Generate diverse random embeddings
            embedding = np.random.randn(512)
            embedding = embedding / np.linalg.norm(embedding)
            result = detector.analyze_frame(embedding)
        
        stats = detector.get_stats()
        print(f"✓ After 50 diverse frames:")
        print(f"   Status: {result['status']}")
        print(f"   Avg similarity: {stats['recent_avg_similarity']:.3f}")
        print(f"   Convergence delta: {result['convergence_delta']:.3f}")
        
        if result['status'] != 'ok':
            print("✗ Expected OK status for diverse embeddings")
            return False
        
        # Test with converging embeddings (should detect collapse)
        print("\n3. Testing with converging embeddings...")
        detector.reset()
        
        # Generate embeddings that gradually converge
        base = np.random.randn(512)
        base = base / np.linalg.norm(base)
        
        for i in range(50):
            # Mix with decreasing noise (simulates convergence)
            noise_factor = 1.0 - (i / 50.0) * 0.8  # Reduce noise over time
            embedding = base + np.random.randn(512) * noise_factor
            embedding = embedding / np.linalg.norm(embedding)
            result = detector.analyze_frame(embedding)
        
        stats = detector.get_stats()
        print(f"✓ After 50 converging frames:")
        print(f"   Status: {result['status']}")
        print(f"   Action: {result['action']}")
        print(f"   Avg similarity: {stats['recent_avg_similarity']:.3f}")
        print(f"   Convergence delta: {result['convergence_delta']:.3f}")
        
        if result['action'] not in ['scale_injection', 'force_cache']:
            print("✗ Expected action for converging embeddings")
            return False
        
        # Test stats
        print("\n4. Testing stats...")
        stats = detector.get_stats()
        required_keys = [
            'recent_avg_similarity',
            'overall_avg_similarity',
            'similarity_variance',
            'frames_analyzed'
        ]
        for key in required_keys:
            if key not in stats:
                print(f"✗ Missing stat key: {key}")
                return False
        
        print(f"✓ All stats present:")
        for key, value in stats.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        
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

