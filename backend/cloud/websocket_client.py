"""
VPS WebSocket Client

Maintains a persistent WebSocket connection to the VPS for:
- Pushing frames (binary)
- Pushing state snapshots (binary)
- Receiving control messages (pause, resume, shutdown)
- Heartbeat to keep connection alive

The connection is resilient to brief network interruptions with
automatic reconnection and exponential backoff.
"""

import asyncio
import logging
import time
import os
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Binary message type bytes for GPU → VPS communication"""
    FRAME = 0x01
    STATE = 0x02
    HEARTBEAT = 0x03
    STATUS = 0x04


class ControlType(IntEnum):
    """Binary message type bytes for VPS → GPU communication"""
    PAUSE = 0x10
    RESUME = 0x11
    SAVE_STATE = 0x12
    SHUTDOWN = 0x13
    LOAD_STATE = 0x14


@dataclass
class ConnectionStats:
    """Statistics for the WebSocket connection"""
    connected: bool = False
    connect_time: Optional[float] = None
    disconnect_time: Optional[float] = None
    reconnect_attempts: int = 0
    messages_sent: int = 0
    bytes_sent: int = 0
    messages_received: int = 0
    last_heartbeat: Optional[float] = None


class VPSWebSocketClient:
    """
    WebSocket client for GPU → VPS communication
    
    Handles connection lifecycle, message sending, and control message receiving.
    Designed to be used with asyncio event loop.
    """
    
    def __init__(self, config: dict):
        """
        Initialize WebSocket client
        
        Args:
            config: Cloud configuration dict containing:
                - vps_websocket_url: WebSocket endpoint URL
                - auth_token: Authentication token (optional)
        """
        self.url = config.get('vps_websocket_url', 'ws://localhost:8000/ws/gpu')
        self.auth_token = config.get('auth_token') or os.environ.get('DREAM_GEN_AUTH_TOKEN')
        
        # Connection settings
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.heartbeat_interval = 30.0
        
        # State
        self._websocket = None
        self._connected = False
        self._should_run = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        
        # Callbacks for control messages
        self._on_pause: Optional[Callable[[], Awaitable[None]]] = None
        self._on_resume: Optional[Callable[[], Awaitable[None]]] = None
        self._on_save_state: Optional[Callable[[], Awaitable[None]]] = None
        self._on_shutdown: Optional[Callable[[], Awaitable[None]]] = None
        self._on_load_state: Optional[Callable[[bytes], Awaitable[None]]] = None
        
        # Statistics
        self.stats = ConnectionStats()
        
        logger.info(f"VPS WebSocket client initialized: {self.url}")
    
    @property
    def connected(self) -> bool:
        """Whether currently connected to VPS"""
        return self._connected and self._websocket is not None
    
    def set_callbacks(
        self,
        on_pause: Optional[Callable[[], Awaitable[None]]] = None,
        on_resume: Optional[Callable[[], Awaitable[None]]] = None,
        on_save_state: Optional[Callable[[], Awaitable[None]]] = None,
        on_shutdown: Optional[Callable[[], Awaitable[None]]] = None,
        on_load_state: Optional[Callable[[bytes], Awaitable[None]]] = None,
    ) -> None:
        """Set callbacks for control messages from VPS"""
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_save_state = on_save_state
        self._on_shutdown = on_shutdown
        self._on_load_state = on_load_state
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to VPS
        
        Returns:
            True if connected successfully
        """
        try:
            import websockets
            
            # Build headers with auth if provided
            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            logger.info(f"Connecting to VPS: {self.url}")
            
            self._websocket = await websockets.connect(
                self.url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            
            self._connected = True
            self.stats.connected = True
            self.stats.connect_time = time.time()
            self.stats.reconnect_attempts = 0
            
            logger.info("Connected to VPS successfully")
            
            # Start background tasks
            self._should_run = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to VPS: {e}")
            self._connected = False
            self.stats.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from VPS"""
        self._should_run = False
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._receive_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        self._websocket = None
        self._connected = False
        self.stats.connected = False
        self.stats.disconnect_time = time.time()
        
        logger.info("Disconnected from VPS")
    
    async def send_frame(self, frame_data: bytes) -> bool:
        """
        Send a frame to VPS
        
        Args:
            frame_data: WebP-encoded frame bytes
        
        Returns:
            True if sent successfully
        """
        return await self._send_binary(MessageType.FRAME, frame_data)
    
    async def send_state(self, state_data: bytes) -> bool:
        """
        Send state snapshot to VPS
        
        Args:
            state_data: Serialized state bytes (msgpack)
        
        Returns:
            True if sent successfully
        """
        return await self._send_binary(MessageType.STATE, state_data)
    
    async def send_status(self, status_json: bytes) -> bool:
        """
        Send status update to VPS
        
        Args:
            status_json: JSON-encoded status bytes
        
        Returns:
            True if sent successfully
        """
        return await self._send_binary(MessageType.STATUS, status_json)
    
    async def _send_binary(self, msg_type: MessageType, payload: bytes) -> bool:
        """Send a binary message with type prefix"""
        if not self.connected:
            return False
        
        try:
            message = bytes([msg_type]) + payload
            await self._websocket.send(message)
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self._connected = False
            self._schedule_reconnect()
            return False
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to keep connection alive"""
        while self._should_run:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.connected:
                    # Send heartbeat with timestamp
                    timestamp = int(time.time() * 1000).to_bytes(8, 'big')
                    await self._send_binary(MessageType.HEARTBEAT, timestamp)
                    self.stats.last_heartbeat = time.time()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
    
    async def _receive_loop(self) -> None:
        """Receive and handle control messages from VPS"""
        while self._should_run:
            try:
                if not self.connected:
                    await asyncio.sleep(1)
                    continue
                
                message = await self._websocket.recv()
                self.stats.messages_received += 1
                
                if isinstance(message, bytes) and len(message) > 0:
                    await self._handle_control_message(message)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._connected = False
                self._schedule_reconnect()
                await asyncio.sleep(1)
    
    async def _handle_control_message(self, message: bytes) -> None:
        """Handle a control message from VPS"""
        msg_type = message[0]
        payload = message[1:] if len(message) > 1 else b''
        
        try:
            if msg_type == ControlType.PAUSE:
                logger.info("Received PAUSE command from VPS")
                if self._on_pause:
                    await self._on_pause()
            
            elif msg_type == ControlType.RESUME:
                logger.info("Received RESUME command from VPS")
                if self._on_resume:
                    await self._on_resume()
            
            elif msg_type == ControlType.SAVE_STATE:
                logger.info("Received SAVE_STATE command from VPS")
                if self._on_save_state:
                    await self._on_save_state()
            
            elif msg_type == ControlType.SHUTDOWN:
                logger.info("Received SHUTDOWN command from VPS")
                if self._on_shutdown:
                    await self._on_shutdown()
            
            elif msg_type == ControlType.LOAD_STATE:
                logger.info(f"Received LOAD_STATE command from VPS ({len(payload)} bytes)")
                if self._on_load_state:
                    await self._on_load_state(payload)
            
            else:
                logger.warning(f"Unknown control message type: {msg_type}")
        
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
    
    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt"""
        if self._reconnect_task and not self._reconnect_task.done():
            return  # Already scheduled
        
        if self._should_run:
            self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff"""
        delay = self.reconnect_delay
        
        while self._should_run and not self.connected:
            self.stats.reconnect_attempts += 1
            logger.info(f"Reconnection attempt {self.stats.reconnect_attempts} in {delay:.1f}s...")
            
            await asyncio.sleep(delay)
            
            if await self.connect():
                break
            
            # Exponential backoff
            delay = min(delay * 2, self.max_reconnect_delay)
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        uptime = None
        if self.stats.connect_time:
            uptime = time.time() - self.stats.connect_time
        
        return {
            "connected": self.connected,
            "uptime_seconds": round(uptime, 1) if uptime else None,
            "reconnect_attempts": self.stats.reconnect_attempts,
            "messages_sent": self.stats.messages_sent,
            "bytes_sent": self.stats.bytes_sent,
            "messages_received": self.stats.messages_received,
            "last_heartbeat_age": round(time.time() - self.stats.last_heartbeat, 1) if self.stats.last_heartbeat else None,
        }

