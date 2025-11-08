"""
Latent space interpolation module

Provides smooth transitions between images via latent space operations:
- Spherical linear interpolation (slerp)
- VAE encoding/decoding
- Hybrid generation (interpolation + img2img)
"""

from .spherical_lerp import spherical_lerp, linear_lerp
from .latent_encoder import LatentEncoder
from .hybrid_generator import HybridGenerator

__all__ = [
    'spherical_lerp',
    'linear_lerp',
    'LatentEncoder',
    'HybridGenerator',
]

