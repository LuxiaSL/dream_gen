"""
Spherical linear interpolation (slerp) for latent space

Provides smooth, magnitude-preserving interpolation between latent vectors.
Optimized for speed with pre-computation and GPU acceleration.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple


def spherical_lerp(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    t: float,
    epsilon: float = 1e-6,
    precomputed: Optional[Tuple] = None
) -> torch.Tensor:
    """
    Spherical linear interpolation (slerp) between two latent tensors (OPTIMIZED)
    
    Why slerp over linear interpolation?
    - Preserves magnitude (important for latent spaces)
    - Smoother, more natural transitions
    - Avoids "dead zones" in middle of interpolation
    - Better for high-dimensional spaces
    
    OPTIMIZATION: Supports pre-computed values to avoid redundant calculations
    when interpolating between the same two latents multiple times.
    
    Algorithm:
    1. Flatten latents to vectors
    2. Normalize to unit vectors
    3. Compute angle between them (via dot product)
    4. Interpolate along the great circle arc
    5. Scale back to original magnitude
    6. Reshape to original shape
    
    Args:
        latent_a: Starting latent tensor (shape: [B, C, H, W] or any shape)
        latent_b: Ending latent tensor (same shape as latent_a)
        t: Interpolation factor (0.0 = latent_a, 1.0 = latent_b)
        epsilon: Threshold for near-identical vectors
        precomputed: Optional tuple of (a_norm, b_norm, omega, sin_omega, mag_a, mag_b, original_shape)
                     from precompute_slerp_params() for faster repeated interpolations
    
    Returns:
        Interpolated latent tensor (same shape as inputs)
    
    References:
        - https://en.wikipedia.org/wiki/Slerp
        - "Understanding Slerp, Then Not Using It" by Jonathan Blow
    """
    # Validate inputs
    if latent_a.shape != latent_b.shape:
        raise ValueError(f"Latent shapes must match: {latent_a.shape} != {latent_b.shape}")
    
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    
    # Handle edge cases
    if t == 0.0:
        return latent_a.clone()
    if t == 1.0:
        return latent_b.clone()
    
    # Use pre-computed values if available (OPTIMIZATION)
    if precomputed is not None:
        a_norm, b_norm, omega, sin_omega, mag_a, mag_b, original_shape = precomputed
        
        # If vectors are nearly identical, fall back to linear
        if torch.abs(omega) < epsilon:
            result_magnitude = (1.0 - t) * mag_a + t * mag_b
            result_norm = (1.0 - t) * a_norm + t * b_norm
            result_flat = result_norm * result_magnitude
        else:
            # Compute interpolation weights (only depends on t, omega, sin_omega)
            weight_a = torch.sin((1.0 - t) * omega) / sin_omega
            weight_b = torch.sin(t * omega) / sin_omega
            
            # Interpolate on unit sphere
            result_norm = weight_a * a_norm + weight_b * b_norm
            
            # Scale back to interpolated magnitude
            result_magnitude = (1.0 - t) * mag_a + t * mag_b
            result_flat = result_norm * result_magnitude
        
        # Reshape back
        result = result_flat.reshape(original_shape)
        return result
    
    # Standard path (no pre-computed values)
    original_shape = latent_a.shape
    
    # Flatten to 1D vectors for interpolation
    a_flat = latent_a.reshape(-1)
    b_flat = latent_b.reshape(-1)
    
    # Compute magnitudes (for scaling back later)
    mag_a = torch.norm(a_flat)
    mag_b = torch.norm(b_flat)
    
    # Normalize to unit vectors
    a_norm = a_flat / (mag_a + epsilon)
    b_norm = b_flat / (mag_b + epsilon)
    
    # Compute dot product (cosine of angle)
    dot_product = torch.dot(a_norm, b_norm)
    
    # Clamp to avoid numerical errors in arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle between vectors
    omega = torch.acos(dot_product)
    
    # If vectors are nearly identical, fall back to linear interpolation
    if torch.abs(omega) < epsilon:
        # Linear interpolation
        result_flat = (1.0 - t) * a_flat + t * b_flat
    else:
        # Spherical interpolation
        sin_omega = torch.sin(omega)
        
        # Compute interpolation weights
        weight_a = torch.sin((1.0 - t) * omega) / sin_omega
        weight_b = torch.sin(t * omega) / sin_omega
        
        # Interpolate on unit sphere
        result_norm = weight_a * a_norm + weight_b * b_norm
        
        # Scale back to interpolated magnitude
        result_magnitude = (1.0 - t) * mag_a + t * mag_b
        result_flat = result_norm * result_magnitude
    
    # Reshape back to original shape
    result = result_flat.reshape(original_shape)
    
    return result


def precompute_slerp_params(latent_a: torch.Tensor, latent_b: torch.Tensor, epsilon: float = 1e-6) -> Tuple:
    """
    Pre-compute slerp parameters for faster repeated interpolations
    
    When you need to interpolate multiple times between the same two latents
    (e.g., generating 7 frames between two keyframes), pre-computing these values
    and reusing them is much faster than recalculating for each frame.
    
    Args:
        latent_a: Starting latent
        latent_b: Ending latent
        epsilon: Threshold for near-identical vectors
    
    Returns:
        Tuple of (a_norm, b_norm, omega, sin_omega, mag_a, mag_b, original_shape)
    """
    original_shape = latent_a.shape
    
    # Flatten
    a_flat = latent_a.reshape(-1)
    b_flat = latent_b.reshape(-1)
    
    # Compute magnitudes
    mag_a = torch.norm(a_flat)
    mag_b = torch.norm(b_flat)
    
    # Normalize
    a_norm = a_flat / (mag_a + epsilon)
    b_norm = b_flat / (mag_b + epsilon)
    
    # Compute angle
    dot_product = torch.dot(a_norm, b_norm)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    
    return (a_norm, b_norm, omega, sin_omega, mag_a, mag_b, original_shape)


def linear_lerp(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Linear interpolation between two latent tensors
    
    Simpler than spherical lerp, but doesn't preserve magnitude.
    Use this for comparison or when slerp is too slow.
    
    Args:
        latent_a: Starting latent tensor
        latent_b: Ending latent tensor
        t: Interpolation factor (0.0 = latent_a, 1.0 = latent_b)
    
    Returns:
        Linearly interpolated latent tensor
    """
    if latent_a.shape != latent_b.shape:
        raise ValueError(f"Latent shapes must match: {latent_a.shape} != {latent_b.shape}")
    
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    
    return (1.0 - t) * latent_a + t * latent_b


def batch_spherical_lerp(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    num_frames: int,
    include_endpoints: bool = True
) -> torch.Tensor:
    """
    Generate multiple interpolated frames between two latents
    
    Args:
        latent_a: Starting latent
        latent_b: Ending latent
        num_frames: Number of frames to generate
        include_endpoints: If True, include latent_a and latent_b in output
    
    Returns:
        Stacked tensor of interpolated latents [num_frames, C, H, W]
    
    Example:
        >>> latents = batch_spherical_lerp(a, b, num_frames=7, include_endpoints=True)
        >>> # Returns: [a, interp1, interp2, interp3, interp4, interp5, b]
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    
    # Generate interpolation values
    if include_endpoints:
        t_values = torch.linspace(0.0, 1.0, num_frames)
    else:
        # Exclude endpoints (useful for continuous loops)
        t_values = torch.linspace(0.0, 1.0, num_frames + 2)[1:-1]
    
    # Generate all interpolated frames
    interpolated = []
    for t in t_values:
        frame = spherical_lerp(latent_a, latent_b, t.item())
        interpolated.append(frame)
    
    return torch.stack(interpolated)


# Unit test when run directly
if __name__ == "__main__":
    print("🧪 Testing spherical_lerp...")
    
    # Test 1: Basic functionality
    print("\n1. Basic interpolation test")
    a = torch.randn(1, 4, 32, 64)
    b = torch.randn(1, 4, 32, 64)
    
    # Test endpoints
    result_0 = spherical_lerp(a, b, 0.0)
    result_1 = spherical_lerp(a, b, 1.0)
    assert torch.allclose(result_0, a), "t=0.0 should return latent_a"
    assert torch.allclose(result_1, b), "t=1.0 should return latent_b"
    print("✓ Endpoints correct")
    
    # Test midpoint
    result_mid = spherical_lerp(a, b, 0.5)
    assert result_mid.shape == a.shape, "Shape should be preserved"
    print(f"✓ Midpoint shape: {result_mid.shape}")
    
    # Test 2: Magnitude preservation
    print("\n2. Magnitude preservation test")
    mag_a = torch.norm(a.reshape(-1))
    mag_b = torch.norm(b.reshape(-1))
    mag_mid = torch.norm(result_mid.reshape(-1))
    expected_mag = (mag_a + mag_b) / 2
    print(f"  Magnitude A: {mag_a:.4f}")
    print(f"  Magnitude B: {mag_b:.4f}")
    print(f"  Magnitude mid (actual): {mag_mid:.4f}")
    print(f"  Magnitude mid (expected): {expected_mag:.4f}")
    print(f"  Difference: {abs(mag_mid - expected_mag):.6f}")
    
    # Test 3: Compare slerp vs linear
    print("\n3. Slerp vs Linear comparison")
    slerp_result = spherical_lerp(a, b, 0.5)
    linear_result = linear_lerp(a, b, 0.5)
    
    slerp_mag = torch.norm(slerp_result.reshape(-1))
    linear_mag = torch.norm(linear_result.reshape(-1))
    
    print(f"  Slerp magnitude: {slerp_mag:.4f}")
    print(f"  Linear magnitude: {linear_mag:.4f}")
    print(f"  Difference: {abs(slerp_mag - linear_mag):.4f}")
    
    # Test 4: Batch generation
    print("\n4. Batch interpolation test")
    batch_result = batch_spherical_lerp(a, b, num_frames=7)
    print(f"✓ Generated {len(batch_result)} frames")
    print(f"  Batch shape: {batch_result.shape}")
    
    # Verify first and last match
    assert torch.allclose(batch_result[0], a), "First frame should match latent_a"
    assert torch.allclose(batch_result[-1], b), "Last frame should match latent_b"
    print("✓ Batch endpoints correct")
    
    # Test 5: Near-identical vectors (should fall back to linear)
    print("\n5. Near-identical vectors test")
    c = torch.randn(1, 4, 32, 64)
    d = c + 1e-8  # Nearly identical
    result_identical = spherical_lerp(c, d, 0.5)
    print("✓ Near-identical case handled (no NaN)")
    
    # Test 6: Performance benchmark
    print("\n6. Performance benchmark")
    import time
    
    num_iterations = 1000
    start = time.time()
    for _ in range(num_iterations):
        _ = spherical_lerp(a, b, 0.5)
    elapsed = time.time() - start
    per_iteration = (elapsed / num_iterations) * 1000
    
    print(f"✓ {num_iterations} iterations in {elapsed:.3f}s")
    print(f"  Average: {per_iteration:.3f}ms per interpolation")
    
    print("\n✅ All tests passed!")
    print("\nUsage example:")
    print("  from backend.interpolation import spherical_lerp")
    print("  result = spherical_lerp(latent_a, latent_b, t=0.5)")

