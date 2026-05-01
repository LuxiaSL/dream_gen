"""
VPS WebSocket Client

Maintains a persistent WebSocket connection to the VPS for:
- Pushing frames (binary)
- Pushing state snapshots (binary)
- Receiving control messages (pause, resume, shutdown)
- Heartbeat to keep connection alive

Self-healing features:
- Automatic reconnection with exponential backoff
- Proactive health monitoring via ping/pong tracking
- Frame queue for buffering during brief disconnects
- Fallback URL support for alternative VPS endpoints
- Connection lifecycle callbacks for system coordination
- Circuit breaker pattern to prevent hammering dead servers
"""

import asyncio
import logging
import time
import os
from collections import deque
from typing import Optional, Callable, Awaitable, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum, Enum

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


class ConnectionState(Enum):
    """Connection lifecycle states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"  # Circuit breaker tripped


@dataclass
class ConnectionStats:
    """Statistics for the WebSocket connection"""
    connected: bool = False
    connect_time: Optional[float] = None
    disconnect_time: Optional[float] = None
    reconnect_attempts: int = 0
    total_reconnects: int = 0  # Lifetime reconnection count
    messages_sent: int = 0
    bytes_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0  # Messages dropped due to queue overflow
    messages_queued: int = 0  # Messages sent from queue after reconnect
    last_heartbeat: Optional[float] = None
    last_pong_received: Optional[float] = None
    consecutive_failures: int = 0  # For circuit breaker
    current_url_index: int = 0  # Which URL we're currently using


@dataclass  
class QueuedMessage:
    """A message waiting to be sent after reconnection"""
    msg_type: MessageType
    payload: bytes
    metadata: Optional[bytes] = None  # For frame messages with metadata
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more important (keyframes > interpolations)


class VPSWebSocketClient:
    """
    WebSocket client for GPU → VPS communication with self-healing
    
    Features:
    - Automatic reconnection with exponential backoff
    - Proactive health monitoring (ping timeout detection)
    - Frame queue for buffering during brief disconnects
    - Fallback URL support for alternative VPS endpoints
    - Connection lifecycle callbacks for system coordination
    - Circuit breaker pattern to prevent hammering dead servers
    
    Usage:
        client = VPSWebSocketClient(config)
        client.set_lifecycle_callbacks(
            on_connected=my_connected_handler,
            on_disconnected=my_disconnected_handler,
        )
        await client.connect()
    """
    
    # Circuit breaker settings
    CIRCUIT_BREAKER_THRESHOLD = 5  # Consecutive failures before tripping
    CIRCUIT_BREAKER_RESET_TIME = 60.0  # Seconds before trying again after trip
    
    # Health monitoring settings
    # NOTE: The websockets library already handles ping/pong with ping_timeout.
    # This custom health monitor is a backup for detecting silent failures.
    # It must be set HIGH to avoid false positives during warmup phase when
    # no application-level messages are flowing (only websocket-level pings).
    PING_TIMEOUT = 20.0  # Websocket library's ping timeout
    HEALTH_CHECK_INTERVAL = 15.0  # Seconds between health checks
    # Only trigger reconnect after this many seconds of silence.
    # Set high to avoid false positives during warmup/idle periods.
    HEALTH_DEAD_THRESHOLD = 0  # Disabled — websockets library handles ping/pong natively.
    # The old 90s threshold killed the connection every 90s because the VPS
    # never sends application-level messages back (one-way frame stream).
    # Set to 0 to disable; set to e.g. 300 to re-enable with a safe margin.
    
    # Message queue settings
    MAX_QUEUE_SIZE = 100  # Max messages to buffer during disconnect
    MAX_QUEUE_AGE = 30.0  # Seconds - drop queued messages older than this
    
    def __init__(self, config: dict):
        """
        Initialize WebSocket client
        
        Args:
            config: Cloud configuration dict containing:
                - vps_websocket_url: Primary WebSocket endpoint URL
                - vps_websocket_urls: List of fallback URLs (optional)
                - auth_token: Authentication token (optional)
                - reconnect_delay: Initial reconnect delay in seconds (default: 1.0)
                - max_reconnect_delay: Maximum reconnect delay (default: 60.0)
                - heartbeat_interval: Seconds between heartbeats (default: 30.0)
                - queue_frames_on_disconnect: Buffer frames during disconnect (default: True)
        """
        # URL configuration - support single URL or list of fallbacks
        primary_url = config.get('vps_websocket_url', 'ws://localhost:8000/ws/gpu')
        fallback_urls = config.get('vps_websocket_urls', [])
        
        # Build URL list: primary first, then fallbacks
        self.urls: List[str] = [primary_url]
        if fallback_urls:
            self.urls.extend([u for u in fallback_urls if u != primary_url])
        
        self.auth_token = config.get('auth_token') or os.environ.get('DREAM_GEN_AUTH_TOKEN')
        
        # Connection settings
        self.reconnect_delay = config.get('reconnect_delay', 1.0)
        self.max_reconnect_delay = config.get('max_reconnect_delay', 60.0)
        self.heartbeat_interval = config.get('heartbeat_interval', 30.0)
        self.queue_on_disconnect = config.get('queue_frames_on_disconnect', True)
        
        # State
        self._websocket = None
        self._connected = False
        self._should_run = False
        self._state = ConnectionState.DISCONNECTED
        self._current_url_index = 0
        self._circuit_breaker_tripped_at: Optional[float] = None
        
        # Tasks
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._queue_processor_task: Optional[asyncio.Task] = None
        
        # Message queue for buffering during disconnects
        self._message_queue: deque[QueuedMessage] = deque(maxlen=self.MAX_QUEUE_SIZE)
        self._queue_lock = asyncio.Lock()
        
        # Connection lock to prevent concurrent connect attempts
        self._connect_lock = asyncio.Lock()
        
        # Callbacks for control messages
        self._on_pause: Optional[Callable[[], Awaitable[None]]] = None
        self._on_resume: Optional[Callable[[], Awaitable[None]]] = None
        self._on_save_state: Optional[Callable[[], Awaitable[None]]] = None
        self._on_shutdown: Optional[Callable[[], Awaitable[None]]] = None
        self._on_load_state: Optional[Callable[[bytes], Awaitable[None]]] = None
        
        # Lifecycle callbacks for system coordination
        self._on_connected: Optional[Callable[[], Awaitable[None]]] = None
        self._on_disconnected: Optional[Callable[[], Awaitable[None]]] = None
        self._on_reconnecting: Optional[Callable[[int, str], Awaitable[None]]] = None
        self._on_reconnected: Optional[Callable[[], Awaitable[None]]] = None
        
        # Statistics
        self.stats = ConnectionStats()
        
        logger.info(f"VPS WebSocket client initialized")
        logger.info(f"  Primary URL: {self.urls[0]}")
        if len(self.urls) > 1:
            logger.info(f"  Fallback URLs: {self.urls[1:]}")
        logger.info(f"  Queue on disconnect: {self.queue_on_disconnect}")
    
    @property
    def connected(self) -> bool:
        """Whether currently connected to VPS"""
        return self._connected and self._websocket is not None
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state"""
        return self._state
    
    @property
    def current_url(self) -> str:
        """Currently active URL"""
        return self.urls[self._current_url_index]
    
    @property
    def queue_size(self) -> int:
        """Number of messages waiting in queue"""
        return len(self._message_queue)
    
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
    
    def set_lifecycle_callbacks(
        self,
        on_connected: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnected: Optional[Callable[[], Awaitable[None]]] = None,
        on_reconnecting: Optional[Callable[[int, str], Awaitable[None]]] = None,
        on_reconnected: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """
        Set callbacks for connection lifecycle events
        
        Args:
            on_connected: Called when initial connection succeeds
            on_disconnected: Called when connection is lost
            on_reconnecting: Called before each reconnection attempt (attempt_num, url)
            on_reconnected: Called when reconnection succeeds
        """
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        logger.debug("Lifecycle callbacks configured")
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker allows connection attempt
        
        Returns:
            True if connection attempt is allowed
        """
        if self._circuit_breaker_tripped_at is None:
            return True
        
        elapsed = time.time() - self._circuit_breaker_tripped_at
        if elapsed >= self.CIRCUIT_BREAKER_RESET_TIME:
            logger.info(f"Circuit breaker reset after {elapsed:.0f}s - allowing connection attempt")
            self._circuit_breaker_tripped_at = None
            self.stats.consecutive_failures = 0
            return True
        
        remaining = self.CIRCUIT_BREAKER_RESET_TIME - elapsed
        logger.warning(f"Circuit breaker tripped - waiting {remaining:.0f}s before retry")
        return False
    
    def _trip_circuit_breaker(self) -> None:
        """Trip the circuit breaker after too many failures"""
        self._circuit_breaker_tripped_at = time.time()
        self._state = ConnectionState.FAILED
        logger.error(
            f"Circuit breaker TRIPPED after {self.stats.consecutive_failures} consecutive failures. "
            f"Will retry in {self.CIRCUIT_BREAKER_RESET_TIME}s"
        )
    
    def _rotate_url(self) -> str:
        """
        Rotate to next URL in the list
        
        Returns:
            The new URL to try
        """
        if len(self.urls) > 1:
            self._current_url_index = (self._current_url_index + 1) % len(self.urls)
            self.stats.current_url_index = self._current_url_index
            logger.info(f"Rotating to fallback URL: {self.current_url}")
        return self.current_url
    
    async def connect(self, url: Optional[str] = None) -> bool:
        """
        Establish WebSocket connection to VPS
        
        Args:
            url: Specific URL to connect to (uses current_url if None)
        
        Returns:
            True if connected successfully
        """
        # Prevent concurrent connection attempts
        async with self._connect_lock:
            # Already connected? Return success
            if self._connected and self._websocket is not None:
                logger.debug("Already connected, skipping connect")
                return True
            
            # Check circuit breaker
            if not self._check_circuit_breaker():
                return False
            
            target_url = url or self.current_url
            
            try:
                import websockets
                
                self._state = ConnectionState.CONNECTING

                # Cancel old background tasks before opening new connection
                # (prevents "cannot call recv" race with stale receive loop)
                for task_name in ('_receive_task', '_heartbeat_task', '_health_monitor_task'):
                    old_task = getattr(self, task_name, None)
                    if old_task and not old_task.done():
                        old_task.cancel()
                        try:
                            await old_task
                        except (asyncio.CancelledError, Exception):
                            pass

                # Close old socket if still open
                if self._websocket:
                    try:
                        await self._websocket.close()
                    except Exception:
                        pass
                    self._websocket = None

                # Build headers with auth if provided
                headers = {}
                if self.auth_token:
                    headers['Authorization'] = f'Bearer {self.auth_token}'

                logger.info(f"Connecting to VPS: {target_url}")

                self._websocket = await websockets.connect(
                    target_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=self.PING_TIMEOUT,
                    close_timeout=5,
                )
                
                self._connected = True
                self._state = ConnectionState.CONNECTED
                self.stats.connected = True
                self.stats.connect_time = time.time()
                self.stats.consecutive_failures = 0  # Reset on success
                self.stats.last_pong_received = time.time()  # Initialize pong tracking
                
                logger.info("Connected to VPS successfully")
                
                # Start background tasks (only if not already running from a previous connect)
                self._should_run = True

                def _task_alive(t):
                    return t is not None and not t.done()

                if not _task_alive(self._heartbeat_task):
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                if not _task_alive(self._receive_task):
                    self._receive_task = asyncio.create_task(self._receive_loop())
                if not _task_alive(self._health_monitor_task):
                    self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
                
                # Start queue processor if we have queued messages
                if self._message_queue and self.queue_on_disconnect:
                    self._queue_processor_task = asyncio.create_task(self._process_queue())
                
                # Notify via lifecycle callback
                if self._on_connected and self.stats.total_reconnects == 0:
                    try:
                        await self._on_connected()
                    except Exception as e:
                        logger.warning(f"on_connected callback error: {e}")
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to connect to VPS ({target_url}): {e}")
                self._connected = False
                self._state = ConnectionState.DISCONNECTED
                self.stats.connected = False
                self.stats.consecutive_failures += 1
                
                # Check if we should trip the circuit breaker
                if self.stats.consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self._trip_circuit_breaker()
                
                return False
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from VPS"""
        self._should_run = False
        
        # Cancel all background tasks
        tasks_to_cancel = [
            self._heartbeat_task, 
            self._receive_task, 
            self._reconnect_task,
            self._health_monitor_task,
            self._queue_processor_task,
        ]
        for task in tasks_to_cancel:
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
        self._state = ConnectionState.DISCONNECTED
        self.stats.connected = False
        self.stats.disconnect_time = time.time()
        
        # Clear message queue on graceful disconnect
        self._message_queue.clear()
        
        logger.info("Disconnected from VPS")
    
    async def _handle_connection_lost(self) -> None:
        """
        Handle unexpected connection loss
        
        Called when send/receive fails. Triggers lifecycle callback
        and schedules reconnection.
        
        IMPORTANT: This method properly closes the old WebSocket before
        scheduling reconnection. This ensures the VPS knows the old
        connection is dead and will accept the new one.
        """
        if not self._connected:
            return  # Already handled
        
        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        self.stats.connected = False
        self.stats.disconnect_time = time.time()
        
        logger.warning("Connection to VPS lost unexpectedly")
        
        # CRITICAL: Close the old WebSocket so VPS cleans up its reference.
        # Without this, the VPS may still think the old connection is alive
        # and reject our reconnection attempt with "GPU already connected".
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing old WebSocket (expected if already dead): {e}")
            self._websocket = None
        
        # Cancel background tasks that depend on the old connection
        # They will be restarted when we reconnect
        for task in [self._heartbeat_task, self._receive_task, self._health_monitor_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        self._heartbeat_task = None
        self._receive_task = None
        self._health_monitor_task = None
        
        # Notify via lifecycle callback
        if self._on_disconnected:
            try:
                await self._on_disconnected()
            except Exception as e:
                logger.warning(f"on_disconnected callback error: {e}")
        
        # Schedule reconnection
        self._schedule_reconnect()
    
    async def send_frame(self, frame_data: bytes, priority: int = 0) -> bool:
        """
        Send a frame to VPS (legacy format without metadata)
        
        Args:
            frame_data: WebP-encoded frame bytes
            priority: Queue priority if disconnected (higher = more important)
        
        Returns:
            True if sent successfully (or queued if disconnected)
        """
        return await self._send_binary(MessageType.FRAME, frame_data, priority=priority)
    
    async def send_frame_with_metadata(
        self,
        frame_data: bytes,
        metadata_bytes: bytes,
        priority: int = 0,
    ) -> bool:
        """
        Send a frame to VPS with metadata (v2 format)
        
        Message format:
            0x01 | metadata_len (4 bytes BE) | JSON metadata | WebP data
        
        Args:
            frame_data: WebP-encoded frame bytes
            metadata_bytes: JSON metadata encoded as UTF-8 bytes
            priority: Queue priority if disconnected (higher = more important)
        
        Returns:
            True if sent successfully (or queued if disconnected)
        """
        if not self.connected:
            # Queue the message if enabled
            if self.queue_on_disconnect:
                return await self._queue_message(
                    MessageType.FRAME, frame_data, metadata=metadata_bytes, priority=priority
                )
            logger.warning("Cannot send frame: not connected and queueing disabled")
            return False
        
        try:
            t_start = time.time()
            
            # Build message: type + metadata_len + metadata + frame
            metadata_len = len(metadata_bytes)
            message = (
                bytes([MessageType.FRAME]) +
                metadata_len.to_bytes(4, 'big') +
                metadata_bytes +
                frame_data
            )
            
            await self._websocket.send(message)
            send_time_ms = (time.time() - t_start) * 1000
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message)
            
            # Log slow sends (potential network issue)
            if send_time_ms > 50:
                logger.warning(
                    f"[PERF] Slow WS send: {send_time_ms:.1f}ms for "
                    f"{len(frame_data)/1024:.1f}KB frame + {len(metadata_bytes)}B metadata"
                )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send frame with metadata: {e}")
            await self._handle_connection_lost()
            
            # Try to queue the frame
            if self.queue_on_disconnect:
                return await self._queue_message(
                    MessageType.FRAME, frame_data, metadata=metadata_bytes, priority=priority
                )
            return False
    
    async def send_state(self, state_data: bytes) -> bool:
        """
        Send state snapshot to VPS
        
        Args:
            state_data: Serialized state bytes (msgpack)
        
        Returns:
            True if sent successfully
        """
        # State messages are high priority
        return await self._send_binary(MessageType.STATE, state_data, priority=10)
    
    async def send_status(self, status_json: bytes) -> bool:
        """
        Send status update to VPS
        
        Args:
            status_json: JSON-encoded status bytes
        
        Returns:
            True if sent successfully
        """
        return await self._send_binary(MessageType.STATUS, status_json, priority=5)
    
    async def _send_binary(self, msg_type: MessageType, payload: bytes, priority: int = 0) -> bool:
        """
        Send a binary message with type prefix
        
        Args:
            msg_type: Message type
            payload: Message payload
            priority: Queue priority if disconnected
        
        Returns:
            True if sent successfully (or queued)
        """
        if not self.connected:
            # Queue the message if enabled
            if self.queue_on_disconnect and msg_type == MessageType.FRAME:
                return await self._queue_message(msg_type, payload, priority=priority)
            return False
        
        try:
            t_start = time.time()
            message = bytes([msg_type]) + payload
            await self._websocket.send(message)
            send_time_ms = (time.time() - t_start) * 1000
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message)
            
            # Log slow sends (potential network issue)
            if send_time_ms > 50:
                logger.warning(
                    f"[PERF] Slow WS send: {send_time_ms:.1f}ms for "
                    f"{len(payload)/1024:.1f}KB (type={msg_type})"
                )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            await self._handle_connection_lost()
            
            # Try to queue the message
            if self.queue_on_disconnect and msg_type == MessageType.FRAME:
                return await self._queue_message(msg_type, payload, priority=priority)
            return False
    
    async def _queue_message(
        self, 
        msg_type: MessageType, 
        payload: bytes, 
        metadata: Optional[bytes] = None,
        priority: int = 0
    ) -> bool:
        """
        Queue a message for later sending when reconnected
        
        Args:
            msg_type: Message type
            payload: Message payload
            metadata: Optional metadata (for frame messages)
            priority: Priority level (higher = more important)
        
        Returns:
            True if queued successfully
        """
        async with self._queue_lock:
            if len(self._message_queue) >= self.MAX_QUEUE_SIZE:
                # Queue is full - drop lowest priority message
                self.stats.messages_dropped += 1
                logger.warning(
                    f"Message queue full ({self.MAX_QUEUE_SIZE}), dropping oldest message"
                )
            
            msg = QueuedMessage(
                msg_type=msg_type,
                payload=payload,
                metadata=metadata,
                priority=priority,
            )
            self._message_queue.append(msg)
            
            if len(self._message_queue) == 1:
                logger.info("Started queueing frames due to disconnect")
            elif len(self._message_queue) % 10 == 0:
                logger.info(f"Queue size: {len(self._message_queue)} messages")
            
            return True
    
    async def _process_queue(self) -> None:
        """
        Process queued messages after reconnection
        
        Sends queued messages in priority order, dropping stale ones.
        If processing fails partway through, remaining messages are re-queued.
        """
        if not self._message_queue:
            return
        
        logger.info(f"Processing {len(self._message_queue)} queued messages...")
        
        sent = 0
        dropped = 0
        now = time.time()
        
        async with self._queue_lock:
            # Sort by priority (highest first) then by timestamp (oldest first)
            sorted_queue = sorted(
                self._message_queue, 
                key=lambda m: (-m.priority, m.timestamp)
            )
            self._message_queue.clear()
        
        # Track which messages we've processed (by index)
        processed_idx = 0
        
        try:
            for idx, msg in enumerate(sorted_queue):
                processed_idx = idx
                
                # Skip stale messages
                age = now - msg.timestamp
                if age > self.MAX_QUEUE_AGE:
                    dropped += 1
                    continue
                
                # Send the message
                try:
                    if msg.metadata:
                        success = await self.send_frame_with_metadata(
                            msg.payload, msg.metadata, priority=msg.priority
                        )
                    else:
                        success = await self._send_binary(msg.msg_type, msg.payload)
                    
                    if success:
                        sent += 1
                        self.stats.messages_queued += 1
                    else:
                        # Send failed - re-queue this message and all remaining
                        remaining = sorted_queue[idx:]
                        async with self._queue_lock:
                            for remaining_msg in remaining:
                                self._message_queue.append(remaining_msg)
                        logger.warning(
                            f"Queue processing interrupted - re-queued {len(remaining)} messages"
                        )
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to send queued message: {e}")
                    # Re-queue this message and all remaining
                    remaining = sorted_queue[idx:]
                    async with self._queue_lock:
                        for remaining_msg in remaining:
                            self._message_queue.append(remaining_msg)
                    logger.warning(
                        f"Queue processing error - re-queued {len(remaining)} messages"
                    )
                    break
                    
        except Exception as e:
            # Unexpected error - re-queue all unprocessed messages
            logger.error(f"Unexpected error in queue processing: {e}")
            remaining = sorted_queue[processed_idx:]
            async with self._queue_lock:
                for remaining_msg in remaining:
                    self._message_queue.append(remaining_msg)
            logger.warning(f"Re-queued {len(remaining)} unprocessed messages")
        
        if sent > 0 or dropped > 0:
            logger.info(f"Queue processed: {sent} sent, {dropped} dropped (stale)")
    
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
                if not self.connected or not self._websocket:
                    await asyncio.sleep(1)
                    continue

                # Capture the socket we're receiving on — if it changes
                # (reconnect swapped it), exit and let a new loop take over
                current_ws = self._websocket

                message = await current_ws.recv()

                # Check we're still the active socket after await
                if self._websocket is not current_ws:
                    logger.debug("Receive loop: socket was replaced during recv, exiting")
                    break

                self.stats.messages_received += 1
                self.stats.last_pong_received = time.time()

                if isinstance(message, bytes) and len(message) > 0:
                    await self._handle_control_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Only handle if we're still the active connection
                if self._websocket is not None:
                    logger.error(f"Receive error: {e}")
                    await self._handle_connection_lost()
                break  # Exit loop — reconnect will start a fresh one
    
    async def _health_monitor_loop(self) -> None:
        """
        Proactive health monitoring
        
        This is a BACKUP monitor for detecting silent connection failures.
        The websockets library already handles ping/pong with its own timeout,
        but this catches edge cases where the connection goes silent without
        a clean disconnect.
        
        IMPORTANT: The threshold is set high (90s) to avoid false positives
        during warmup/idle periods when no application-level messages flow.
        During warmup, the GPU is generating frames locally and hasn't started
        sending them yet, so VPS has nothing to send back.
        """
        while self._should_run:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                
                if not self.connected:
                    continue
                
                # Check if we've received any message recently
                # This tracks application-level messages, not websocket pings
                if self.stats.last_pong_received and self.HEALTH_DEAD_THRESHOLD > 0:
                    silence_duration = time.time() - self.stats.last_pong_received

                    if silence_duration > self.HEALTH_DEAD_THRESHOLD:
                        logger.warning(
                            f"Connection appears dead - no activity for {silence_duration:.0f}s. "
                            f"Triggering reconnection..."
                        )
                        await self._handle_connection_lost()
                    elif silence_duration > self.HEALTH_DEAD_THRESHOLD * 0.5:
                        # Warn at 50% threshold
                        logger.info(
                            f"No VPS messages for {silence_duration:.0f}s "
                            f"(threshold: {self.HEALTH_DEAD_THRESHOLD:.0f}s)"
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health monitor error: {e}")
    
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
            self._state = ConnectionState.RECONNECTING
            self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """
        Attempt to reconnect with exponential backoff and URL rotation
        
        Features:
        - Exponential backoff (1s → 60s max)
        - URL rotation through fallbacks
        - Lifecycle callbacks
        - Circuit breaker integration
        """
        delay = self.reconnect_delay
        url_attempts = 0  # Track attempts per URL for rotation
        
        while self._should_run and not self.connected:
            # Check circuit breaker
            if not self._check_circuit_breaker():
                await asyncio.sleep(self.CIRCUIT_BREAKER_RESET_TIME / 2)
                continue
            
            self.stats.reconnect_attempts += 1
            url_attempts += 1
            
            # Notify via lifecycle callback
            if self._on_reconnecting:
                try:
                    await self._on_reconnecting(
                        self.stats.reconnect_attempts, 
                        self.current_url
                    )
                except Exception as e:
                    logger.warning(f"on_reconnecting callback error: {e}")
            
            logger.info(
                f"Reconnection attempt {self.stats.reconnect_attempts} "
                f"to {self.current_url} in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)
            
            if await self.connect():
                # Success!
                self.stats.total_reconnects += 1
                logger.info(
                    f"Reconnected successfully after {self.stats.reconnect_attempts} attempts"
                )
                
                # Notify via lifecycle callback
                if self._on_reconnected:
                    try:
                        await self._on_reconnected()
                    except Exception as e:
                        logger.warning(f"on_reconnected callback error: {e}")
                
                # Reset for next time
                self.stats.reconnect_attempts = 0
                break
            
            # Rotate URL after a few failed attempts to the same URL
            if url_attempts >= 2 and len(self.urls) > 1:
                self._rotate_url()
                url_attempts = 0
                # Reset delay when switching URLs
                delay = self.reconnect_delay
            else:
                # Exponential backoff
                delay = min(delay * 2, self.max_reconnect_delay)
    
    def get_stats(self) -> dict:
        """Get comprehensive connection statistics"""
        uptime = None
        if self.stats.connect_time and self.connected:
            uptime = time.time() - self.stats.connect_time
        
        pong_age = None
        if self.stats.last_pong_received:
            pong_age = time.time() - self.stats.last_pong_received
        
        return {
            "connected": self.connected,
            "state": self._state.value,
            "current_url": self.current_url,
            "uptime_seconds": round(uptime, 1) if uptime else None,
            "reconnect_attempts": self.stats.reconnect_attempts,
            "total_reconnects": self.stats.total_reconnects,
            "consecutive_failures": self.stats.consecutive_failures,
            "circuit_breaker_tripped": self._circuit_breaker_tripped_at is not None,
            "messages_sent": self.stats.messages_sent,
            "bytes_sent": self.stats.bytes_sent,
            "bytes_sent_mb": round(self.stats.bytes_sent / 1024 / 1024, 2),
            "messages_received": self.stats.messages_received,
            "messages_dropped": self.stats.messages_dropped,
            "messages_queued": self.stats.messages_queued,
            "queue_size": len(self._message_queue),
            "last_heartbeat_age": round(time.time() - self.stats.last_heartbeat, 1) if self.stats.last_heartbeat else None,
            "last_pong_age": round(pong_age, 1) if pong_age else None,
        }
    
    async def force_reconnect(self) -> bool:
        """
        Force an immediate reconnection attempt
        
        Useful for manually triggering reconnection after detecting issues.
        Properly cleans up existing connection and tasks before reconnecting.
        Unlike disconnect(), this preserves the message queue.
        
        Returns:
            True if reconnection succeeded
        """
        logger.info("Force reconnect requested")
        
        # Stop background tasks to prevent duplicates
        self._should_run = False
        
        # Cancel all background tasks (similar to disconnect, but keep queue)
        tasks_to_cancel = [
            self._heartbeat_task,
            self._receive_task,
            self._reconnect_task,
            self._health_monitor_task,
            self._queue_processor_task,
        ]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Reset task references
        self._heartbeat_task = None
        self._receive_task = None
        self._reconnect_task = None
        self._health_monitor_task = None
        self._queue_processor_task = None
        
        # Close existing WebSocket connection
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
        
        self._websocket = None
        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        self.stats.connected = False
        
        # Reset circuit breaker for manual reconnect
        self._circuit_breaker_tripped_at = None
        self.stats.consecutive_failures = 0
        
        # Try to connect (this will set _should_run = True and start new tasks)
        return await self.connect()
    
    def reset_circuit_breaker(self) -> None:
        """
        Manually reset the circuit breaker
        
        Use this when you know the server is back up and want to
        immediately retry connection.
        """
        self._circuit_breaker_tripped_at = None
        self.stats.consecutive_failures = 0
        self._state = ConnectionState.DISCONNECTED
        logger.info("Circuit breaker manually reset")

