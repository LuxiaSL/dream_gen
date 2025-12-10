"""
Dream Window Cloud Module

Optional cloud integration for pushing frames to a VPS WebSocket endpoint.
This module is only loaded when cloud.enabled is True in config.yaml.

Components:
- websocket_client: Persistent WebSocket connection to VPS
- frame_pusher: WebP encoding and frame transmission
- state_sync: Periodic state snapshots for resume capability

Usage:
    When cloud.enabled: false (default), Dream Window operates in standalone
    mode with Rainmeter output only. When cloud.enabled: true, frames are
    additionally pushed to the configured VPS endpoint.
"""

from .websocket_client import VPSWebSocketClient
from .frame_pusher import CloudFramePusher
from .state_sync import CloudStateSync

__all__ = [
    "VPSWebSocketClient",
    "CloudFramePusher",
    "CloudStateSync",
]

