"""
Fresh Frame Generation Package

Provides pre-generation of fresh frames during idle periods for seamless
template switches. When the system detects it needs a seed injection, the
fresh frame buffer provides an immediately-available frame generated with
a new template, eliminating visible generation delay.
"""

from .buffer import FreshFrameBuffer

__all__ = ['FreshFrameBuffer']

