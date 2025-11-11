"""
Latent space interpolation module

Provides smooth transitions between images via latent space operations:
- Spherical linear interpolation (slerp)
- VAE encoding/decoding
"""

from .spherical_lerp import spherical_lerp, linear_lerp
from .latent_encoder import LatentEncoder

__all__ = [
    'spherical_lerp',
    'linear_lerp',
    'LatentEncoder',
]

