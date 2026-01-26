"""
Dream Window Cloud Module

Optional cloud integration for pushing frames to a VPS WebSocket endpoint.
This module is only loaded when cloud.enabled is True in config.yaml.

Components:
- websocket_client: Persistent WebSocket connection to VPS with self-healing
- frame_pusher: WebP encoding and frame transmission
- state_sync: Periodic state snapshots for resume capability

Self-healing features:
- Automatic reconnection with exponential backoff
- Proactive health monitoring via ping/pong tracking
- Frame queue for buffering during brief disconnects
- Fallback URL support for alternative VPS endpoints
- Connection lifecycle callbacks for system coordination
- Circuit breaker pattern to prevent hammering dead servers

Usage:
    When cloud.enabled: false (default), Dream Window operates in standalone
    mode with Rainmeter output only. When cloud.enabled: true, frames are
    additionally pushed to the configured VPS endpoint.
"""

from .websocket_client import VPSWebSocketClient, ConnectionState, MessageType, ControlType
from .frame_pusher import CloudFramePusher
from .state_sync import CloudStateSync

__all__ = [
    "VPSWebSocketClient",
    "ConnectionState",
    "MessageType",
    "ControlType",
    "CloudFramePusher",
    "CloudStateSync",
]

