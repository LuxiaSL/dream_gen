"""
Dream Window Daemon
===================

Production-ready daemon that manages ComfyUI and the Dream Window generation loop.
Designed for autonomous operation via Rainmeter or standalone.

Features:
- Auto-starts ComfyUI backend (hidden, no console window)
- Launches DreamController generation loop
- Health monitoring with auto-restart on crashes
- Graceful shutdown handling
- Control interface (pause/resume/shutdown via file commands)
- Comprehensive status reporting
- Cross-platform with Windows-optimized subprocess handling

Usage:
    # Standalone (console)
    python daemon.py
    
    # Background mode (no console, Windows)
    pythonw daemon.py
    
    # From Rainmeter (via launch_daemon.ps1)
    Auto-launched when widget loads
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.utils.status_writer import StatusWriter


class DaemonManager:
    """
    Main daemon orchestrator for Dream Window
    
    Responsibilities:
    - Launch and monitor ComfyUI backend
    - Launch and monitor DreamController
    - Auto-restart crashed processes
    - Process control commands (pause/resume/shutdown)
    - Report comprehensive status
    - Handle graceful shutdown
    """
    
    def __init__(self, config_path: str = "backend/config.yaml"):
        """
        Initialize daemon manager
        
        Args:
            config_path: Path to configuration file (relative to project root)
        """
        # Project root is where daemon.py lives
        self.project_root = Path(__file__).parent.resolve()
        
        # Load configuration
        config_file = self.project_root / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Extract daemon config BEFORE setup_logging (it needs this!)
        self.daemon_config = self.config.get('daemon', {})
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("=" * 70)
        self.logger.info("DREAM WINDOW DAEMON STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Config: {config_file}")
        
        # Resolve paths (support both relative and absolute)
        self.comfyui_script = self._resolve_path(
            self.daemon_config.get('comfyui', {}).get('startup_script', '')
        )
        self.control_file = self._resolve_path(
            self.daemon_config.get('control_file', 'output/daemon_control.txt')
        )
        self.pid_file = self._resolve_path(
            self.daemon_config.get('pid_file', 'output/daemon.pid')
        )
        
        # Initialize status writer
        output_dir = self._resolve_path(self.config['system']['output_dir'])
        self.status_writer = StatusWriter(output_dir)
        
        # Process handles
        self.comfyui_process: Optional[subprocess.Popen] = None
        self.controller_process: Optional[subprocess.Popen] = None
        
        # State tracking
        self.running = False
        self.start_time = time.time()
        self.comfyui_restarts = 0
        self.controller_restarts = 0
        self.restart_timestamps: Dict[str, list] = {'comfyui': [], 'controller': []}
        
        # Daemon status
        self.comfyui_status = "stopped"
        self.controller_status = "stopped"
        
        self.logger.info("[OK] Daemon initialization complete")
    
    def _resolve_path(self, path: str) -> Path:
        """
        Resolve path to absolute Path object
        
        Supports:
        - Relative paths (relative to project root)
        - Absolute paths (used as-is)
        - Empty string (returns project root)
        
        Args:
            path: Path string (relative or absolute)
        
        Returns:
            Resolved absolute Path object
        """
        if not path:
            return self.project_root
        
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            return path_obj
        else:
            return (self.project_root / path_obj).resolve()
    
    def _setup_logging(self):
        """Configure daemon logging"""
        log_file = self._resolve_path(
            self.daemon_config.get('log_file', 'logs/daemon.log')
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_level = self.daemon_config.get('log_level', 'INFO')
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def _write_pid_file(self):
        """Write daemon PID to file for external tracking"""
        try:
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"PID file written: {self.pid_file}")
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {e}")
    
    def _remove_pid_file(self):
        """Remove PID file on shutdown"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("PID file removed")
        except Exception as e:
            self.logger.error(f"Failed to remove PID file: {e}")
    
    async def start_comfyui(self) -> bool:
        """
        Start ComfyUI backend process
        
        Features:
        - Hidden console window (Windows)
        - Validates startup script exists
        - Waits for health check before returning
        - Configurable timeout
        - Detects if ComfyUI is already running
        
        Returns:
            True if ComfyUI started successfully
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING COMFYUI BACKEND")
        self.logger.info("=" * 70)
        
        # Check for and clean up orphaned ComfyUI processes first
        self.logger.info("Checking for orphaned ComfyUI processes...")
        killed = self._kill_orphaned_comfyui()
        if killed > 0:
            self.logger.info(f"Cleaned up {killed} orphaned process(es)")
            # Wait a moment for ports to be released
            await asyncio.sleep(2)
        else:
            self.logger.info("No orphaned processes found")
        
        # Check if ComfyUI is already running
        health_url = self.daemon_config.get('comfyui', {}).get(
            'health_check_url',
            'http://127.0.0.1:8188/system_stats'
        )
        
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                self.logger.warning("=" * 70)
                self.logger.warning("⚠️  ComfyUI ALREADY RUNNING!")
                self.logger.warning("=" * 70)
                self.logger.warning("ComfyUI appears to be running from a previous session.")
                self.logger.warning("This usually means the previous daemon did not shut down properly.")
                self.logger.warning("")
                self.logger.warning("Attempting to verify ComfyUI is functional...")
                
                # Try to check queue to verify it's actually responding
                try:
                    base_url = health_url.rsplit('/', 1)[0]  # Get base URL from health_url
                    queue_response = requests.get(f"{base_url}/queue", timeout=2)
                    if queue_response.status_code == 200:
                        queue_data = queue_response.json()
                        running = len(queue_data.get('queue_running', []))
                        pending = len(queue_data.get('queue_pending', []))
                        
                        self.logger.warning(f"ComfyUI queue check: {running} running, {pending} pending")
                        self.logger.warning("")
                        self.logger.warning("CONTINUING with existing ComfyUI instance.")
                        self.logger.warning("If frames don't generate, manually kill ComfyUI:")
                        self.logger.warning("  python kill_comfyui.py")
                        self.logger.warning("=" * 70)
                        
                        self.comfyui_status = "ready"
                        self._update_daemon_status()
                        return True
                    else:
                        self.logger.error("ComfyUI health check passed but queue check failed!")
                        self.logger.error("Killing orphaned instance...")
                        killed = self._kill_orphaned_comfyui()
                        if killed > 0:
                            await asyncio.sleep(2)
                        # Continue with normal startup
                        
                except Exception as e:
                    self.logger.error(f"ComfyUI not responding properly: {e}")
                    self.logger.error("Killing orphaned instance...")
                    killed = self._kill_orphaned_comfyui()
                    if killed > 0:
                        await asyncio.sleep(2)
                    # Continue with normal startup
                    
        except requests.RequestException:
            pass  # ComfyUI not running yet, proceed with startup
        
        # Validate startup script
        if not self.comfyui_script.exists():
            self.logger.error(f"ComfyUI startup script not found: {self.comfyui_script}")
            self.logger.error("Please configure daemon.comfyui.startup_script in config.yaml")
            self.logger.error("")
            self.logger.error("Notes:")
            self.logger.error("  - ComfyUI .bat files handle their own virtual environment")
            self.logger.error("  - Use the .bat file that matches your GPU setup")
            self.logger.error("  - Path should be relative to project root or absolute")
            return False
        
        self.logger.info(f"Script: {self.comfyui_script}")
        self.comfyui_status = "starting"
        self._update_daemon_status()
        
        try:
            # Prepare subprocess arguments
            startup_info = None
            creation_flags = 0
            
            # Windows-specific: Hide console window
            if sys.platform == 'win32':
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup_info.wShowWindow = 0  # SW_HIDE
                creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                
                self.logger.info("Using Windows hidden console mode")
            
            # Determine command based on file extension
            script_str = str(self.comfyui_script)
            if script_str.endswith('.bat') or script_str.endswith('.cmd'):
                # Windows batch file - needs to run from its directory
                script_dir = self.comfyui_script.parent
                cmd = [str(self.comfyui_script)]
                cwd = script_dir
                self.logger.info(f"Working directory: {cwd}")
            elif script_str.endswith('.sh'):
                # Shell script
                cmd = ['bash', script_str]
                cwd = self.project_root
            else:
                # Assume it's a direct command (e.g., "python path/to/main.py")
                cmd = script_str.split()
                cwd = self.project_root
            
            # Launch process
            self.logger.info(f"Launching: {' '.join(cmd)}")
            
            # Don't capture stdout/stderr for .bat files
            # Capturing pipes can cause .bat files to hang or fail silently
            if script_str.endswith('.bat') or script_str.endswith('.cmd'):
                # Let .bat output go to null (don't capture)
                self.comfyui_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startup_info,
                    creationflags=creation_flags if sys.platform == 'win32' else 0,
                )
            else:
                # For non-.bat, capturing is fine
                self.comfyui_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    startupinfo=startup_info,
                    creationflags=creation_flags if sys.platform == 'win32' else 0,
                )
            
            self.logger.info(f"ComfyUI process started (PID: {self.comfyui_process.pid})")
            
            # Wait for ComfyUI to become ready
            health_url = self.daemon_config.get('comfyui', {}).get(
                'health_check_url',
                'http://127.0.0.1:8188/system_stats'
            )
            timeout = self.daemon_config.get('comfyui', {}).get('startup_timeout', 120)
            check_interval = self.daemon_config.get('comfyui', {}).get('health_check_interval', 2)
            
            self.logger.info(f"Waiting for ComfyUI to be FULLY ready (not just web server up)")
            self.logger.info(f"Health check: {health_url}")
            self.logger.info(f"Timeout: {timeout}s")
            
            # Extract base URL for queue checks
            base_url = health_url.rsplit('/', 1)[0]  # e.g., http://127.0.0.1:8188
            
            start_time = time.time()
            web_server_ready = False
            queue_ready = False
            
            while time.time() - start_time < timeout:
                # Check if process died
                if self.comfyui_process.poll() is not None:
                    self.logger.error(f"ComfyUI process died during startup (exit code: {self.comfyui_process.returncode})")
                    self.comfyui_status = "crashed"
                    return False
                
                # Step 1: Check if web server is up
                if not web_server_ready:
                    try:
                        response = requests.get(health_url, timeout=2)
                        if response.status_code == 200:
                            elapsed = time.time() - start_time
                            self.logger.info(f"[1/2] Web server up in {elapsed:.1f}s")
                            web_server_ready = True
                    except requests.RequestException:
                        pass  # Not ready yet
                
                # Step 2: Check if queue endpoint is ready (means ComfyUI fully loaded)
                if web_server_ready and not queue_ready:
                    try:
                        response = requests.get(f"{base_url}/queue", timeout=2)
                        if response.status_code == 200:
                            queue_data = response.json()
                            # Check if queue structure is valid
                            if 'queue_running' in queue_data and 'queue_pending' in queue_data:
                                elapsed = time.time() - start_time
                                self.logger.info(f"[2/2] Queue endpoint ready in {elapsed:.1f}s")
                                queue_ready = True
                                break  # Fully ready!
                    except (requests.RequestException, ValueError, KeyError):
                        pass  # Queue not ready yet
                
                await asyncio.sleep(check_interval)
            
            if web_server_ready and queue_ready:
                elapsed = time.time() - start_time
                self.logger.info(f"✓ ComfyUI FULLY READY in {elapsed:.1f}s")
                self.comfyui_status = "ready"
                self._update_daemon_status()
                return True
            elif web_server_ready:
                self.logger.error(f"ComfyUI web server up but queue endpoint timeout ({timeout}s)")
                self.logger.error("Web server started but ComfyUI may still be loading models")
                self.comfyui_status = "timeout"
                return False
            else:
                self.logger.error(f"ComfyUI health check timeout ({timeout}s)")
                self.logger.error("Check logs and ensure ComfyUI can start properly")
                self.comfyui_status = "timeout"
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to start ComfyUI: {e}", exc_info=True)
            self.comfyui_status = "error"
            return False
    
    async def start_controller(self) -> bool:
        """
        Start DreamController generation loop
        
        Launches as subprocess running backend/main.py with proper venv Python.
        
        Returns:
            True if controller started successfully
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING DREAMCONTROLLER")
        self.logger.info("=" * 70)
        
        self.controller_status = "starting"
        self._update_daemon_status()
        
        try:
            # Get Python executable for controller (may be different from daemon's Python)
            controller_config = self.daemon_config.get('controller', {})
            python_exe_config = controller_config.get('python_executable', 'auto')
            
            if python_exe_config == 'auto':
                # Use same Python as daemon
                python_exe = sys.executable
                self.logger.info(f"Using daemon Python: {python_exe}")
            else:
                # Use configured Python (may be in venv)
                python_exe = self._resolve_path(python_exe_config)
                
                if not python_exe.exists():
                    self.logger.error(f"Configured Python not found: {python_exe}")
                    self.logger.error("Check daemon.controller.python_executable in config.yaml")
                    return False
                
                self.logger.info(f"Using configured Python: {python_exe}")
            
            # Get main script path
            main_script_config = controller_config.get('main_script', 'backend/main.py')
            main_script = self._resolve_path(main_script_config)
            
            if not main_script.exists():
                self.logger.error(f"Controller script not found: {main_script}")
                return False
            
            cmd = [str(python_exe), str(main_script)]
            
            # Windows-specific: Hide console
            startup_info = None
            creation_flags = 0
            
            if sys.platform == 'win32':
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup_info.wShowWindow = 0
                creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            
            # Launch process
            self.logger.info(f"Launching: {' '.join(cmd)}")
            self.controller_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startup_info,
                creationflags=creation_flags if sys.platform == 'win32' else 0,
            )
            
            self.logger.info(f"Controller process started (PID: {self.controller_process.pid})")
            
            # Give it a moment to initialize
            await asyncio.sleep(2)
            
            # Check if it's still alive
            if self.controller_process.poll() is not None:
                self.logger.error(f"Controller died immediately (exit code: {self.controller_process.returncode})")
                self.logger.error("Common causes:")
                self.logger.error("  - Wrong Python version/venv")
                self.logger.error("  - Missing dependencies")
                self.logger.error("  - Config errors in backend/config.yaml")
                self.controller_status = "crashed"
                return False
            
            self.logger.info("[OK] Controller running")
            self.controller_status = "generating"
            self._update_daemon_status()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start controller: {e}", exc_info=True)
            self.controller_status = "error"
            return False
    
    def _can_restart(self, process_name: str) -> bool:
        """
        Check if we can restart a process (rate limiting)
        
        Prevents restart loops by limiting restarts per hour.
        
        Args:
            process_name: 'comfyui' or 'controller'
        
        Returns:
            True if restart is allowed
        """
        max_restarts = self.daemon_config.get('auto_restart', {}).get('max_restarts', 5)
        
        # Clean old timestamps (older than 1 hour)
        current_time = time.time()
        self.restart_timestamps[process_name] = [
            ts for ts in self.restart_timestamps[process_name]
            if current_time - ts < 3600
        ]
        
        # Check if we've hit the limit
        restart_count = len(self.restart_timestamps[process_name])
        
        if restart_count >= max_restarts:
            self.logger.error(
                f"{process_name} has restarted {restart_count} times in the last hour. "
                f"Max restarts ({max_restarts}) reached. Giving up."
            )
            return False
        
        return True
    
    def _record_restart(self, process_name: str):
        """Record a restart attempt"""
        self.restart_timestamps[process_name].append(time.time())
        
        if process_name == 'comfyui':
            self.comfyui_restarts += 1
        elif process_name == 'controller':
            self.controller_restarts += 1
    
    async def monitor_loop(self):
        """
        Main monitoring loop
        
        Continuously checks:
        - Process health (both ComfyUI and controller)
        - Control commands from control file
        - Daemon status updates
        
        Handles:
        - Auto-restart on crashes
        - Control command execution
        - Status reporting
        """
        self.logger.info("=" * 70)
        self.logger.info("MONITORING LOOP STARTED")
        self.logger.info("=" * 70)
        
        status_update_interval = self.daemon_config.get('status_update_interval', 5)
        control_check_interval = self.daemon_config.get('control_check_interval', 2)
        
        last_status_update = 0
        last_control_check = 0
        loop_iterations = 0
        
        self.logger.info(f"Monitoring configuration:")
        self.logger.info(f"  Status updates every {status_update_interval}s")
        self.logger.info(f"  Control checks every {control_check_interval}s")
        self.logger.info(f"  ComfyUI PID: {self.comfyui_process.pid if self.comfyui_process else 'None'}")
        self.logger.info(f"  Controller PID: {self.controller_process.pid if self.controller_process else 'None'}")
        self.logger.info("")
        self.logger.info("Monitoring loop is now running silently.")
        self.logger.info("Logs will only appear for:")
        self.logger.info("  - Process crashes")
        self.logger.info("  - Control commands received")
        self.logger.info("  - Status updates (DEBUG level)")
        self.logger.info("=" * 70)
        
        while self.running:
            try:
                current_time = time.time()
                loop_iterations += 1
                
                # No heartbeat logging - too verbose for production
                # Daemon runs silently unless something needs attention
                
                # Check ComfyUI health
                if self.comfyui_process:
                    poll_result = self.comfyui_process.poll()
                    if poll_result is not None:
                        # Process died
                        self.logger.warning(f"ComfyUI process died (exit code: {poll_result})")
                        self.comfyui_status = "crashed"
                        self._update_daemon_status()
                        
                        # Auto-restart if enabled
                        if self.daemon_config.get('auto_restart', {}).get('comfyui', True):
                            if self._can_restart('comfyui'):
                                restart_delay = self.daemon_config.get('auto_restart', {}).get('restart_delay', 10)
                                self.logger.info(f"Restarting ComfyUI in {restart_delay}s...")
                                await asyncio.sleep(restart_delay)
                                
                                self._record_restart('comfyui')
                                success = await self.start_comfyui()
                                if success:
                                    self.logger.info("[OK] ComfyUI restarted successfully")
                                else:
                                    self.logger.error("Failed to restart ComfyUI")
                        else:
                            self.logger.error("Auto-restart disabled for ComfyUI")
                
                # Check controller health
                if self.controller_process:
                    poll_result = self.controller_process.poll()
                    if poll_result is not None:
                        # Process died
                        self.logger.warning(f"Controller process died (exit code: {poll_result})")
                        self.controller_status = "crashed"
                        self._update_daemon_status()
                        
                        # Auto-restart if enabled
                        if self.daemon_config.get('auto_restart', {}).get('controller', True):
                            if self._can_restart('controller'):
                                restart_delay = self.daemon_config.get('auto_restart', {}).get('restart_delay', 10)
                                self.logger.info(f"Restarting controller in {restart_delay}s...")
                                await asyncio.sleep(restart_delay)
                                
                                self._record_restart('controller')
                                success = await self.start_controller()
                                if success:
                                    self.logger.info("[OK] Controller restarted successfully")
                                else:
                                    self.logger.error("Failed to restart controller")
                        else:
                            self.logger.error("Auto-restart disabled for controller")
                
                # Check for control commands
                if current_time - last_control_check >= control_check_interval:
                    await self._process_control_commands()
                    last_control_check = current_time
                
                # Update daemon status
                if current_time - last_status_update >= status_update_interval:
                    self.logger.debug(f"Updating daemon status (interval: {status_update_interval}s)")
                    self._update_daemon_status()
                    last_status_update = current_time
                
                # Sleep briefly
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        self.logger.info("Monitor loop stopped")
    
    async def _process_control_commands(self):
        """
        Check for and process control commands from control file
        
        Commands:
        - PAUSE: Pause generation (send SIGTERM to controller)
        - RESUME: Resume generation (restart controller)
        - SHUTDOWN: Graceful shutdown of daemon
        - RESTART_COMFYUI: Restart ComfyUI only
        - RESTART_CONTROLLER: Restart controller only
        """
        if not self.control_file.exists():
            return
        
        try:
            # Read command
            with open(self.control_file, 'r') as f:
                command = f.read().strip().upper()
            
            # Delete command file
            self.control_file.unlink()
            
            if not command:
                return
            
            self.logger.info(f"[CONTROL] Received command: {command}")
            
            if command == 'PAUSE':
                await self._handle_pause()
            elif command == 'RESUME':
                await self._handle_resume()
            elif command == 'SHUTDOWN':
                await self._handle_shutdown_command()
            elif command == 'RESTART_COMFYUI':
                await self._handle_restart_comfyui()
            elif command == 'RESTART_CONTROLLER':
                await self._handle_restart_controller()
            else:
                self.logger.warning(f"Unknown command: {command}")
        
        except Exception as e:
            self.logger.error(f"Error processing control command: {e}")
    
    async def _handle_pause(self):
        """Pause generation (stop controller, keep ComfyUI running)"""
        if self.controller_process and self.controller_process.poll() is None:
            self.logger.info("Pausing controller...")
            self._terminate_process(self.controller_process, "Controller")
            self.controller_process = None
            self.controller_status = "paused"
            self._update_daemon_status()
    
    async def _handle_resume(self):
        """Resume generation (restart controller)"""
        if not self.controller_process or self.controller_process.poll() is not None:
            self.logger.info("Resuming controller...")
            success = await self.start_controller()
            if success:
                self.logger.info("[OK] Controller resumed")
            else:
                self.logger.error("Failed to resume controller")
    
    async def _handle_shutdown_command(self):
        """Handle shutdown command"""
        self.logger.info("Shutdown requested via control file")
        self.running = False
    
    async def _handle_restart_comfyui(self):
        """Restart ComfyUI only"""
        self.logger.info("Restarting ComfyUI...")
        if self.comfyui_process:
            self._terminate_process(self.comfyui_process, "ComfyUI")
            self.comfyui_process = None
        
        await asyncio.sleep(2)
        self._record_restart('comfyui')
        success = await self.start_comfyui()
        if success:
            self.logger.info("[OK] ComfyUI restarted")
        else:
            self.logger.error("Failed to restart ComfyUI")
    
    async def _handle_restart_controller(self):
        """Restart controller only"""
        self.logger.info("Restarting controller...")
        if self.controller_process:
            self._terminate_process(self.controller_process, "Controller")
            self.controller_process = None
        
        await asyncio.sleep(2)
        self._record_restart('controller')
        success = await self.start_controller()
        if success:
            self.logger.info("[OK] Controller restarted")
        else:
            self.logger.error("Failed to restart controller")
    
    def _update_daemon_status(self):
        """Update daemon status in status.json"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        self.status_writer.write_daemon_status(
            daemon_status="running" if self.running else "stopping",
            comfyui_status=self.comfyui_status,
            controller_status=self.controller_status,
            daemon_uptime_hours=uptime_hours,
            comfyui_restarts=self.comfyui_restarts,
            controller_restarts=self.controller_restarts,
        )
    
    def _terminate_process(self, process: subprocess.Popen, name: str):
        """
        Gracefully terminate a process
        
        For Windows batch files (.bat/.cmd), this will kill the entire process tree
        since batch files spawn child processes that need to be terminated together.
        
        Args:
            process: Process to terminate
            name: Process name for logging
        """
        if not process or process.poll() is not None:
            return
        
        self.logger.info(f"Terminating {name} (PID: {process.pid})...")
        
        try:
            # Windows-specific: Kill entire process tree
            # Batch files create child processes, so we need taskkill with /T flag
            if sys.platform == 'win32':
                self.logger.debug(f"Terminating process tree for {name}")
                try:
                    # Use taskkill to terminate the entire process tree
                    # /T = terminate all child processes
                    # /F = force termination (no graceful shutdown for .bat spawns)
                    # This is the proper Windows method for batch-spawned processes
                    result = subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        self.logger.debug(f"taskkill successful for {name}")
                    else:
                        # Process might be already dead
                        self.logger.debug(f"taskkill returned {result.returncode}: {result.stderr.strip()}")
                    
                    # Wait for process handle to close
                    try:
                        process.wait(timeout=3)
                        self.logger.info(f"{name} terminated (process tree killed)")
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"{name} handle still open, forcing via Python...")
                        process.kill()
                        process.wait(timeout=2)
                
                except subprocess.TimeoutExpired:
                    self.logger.error(f"taskkill timed out for {name}")
                    # Last resort - kill via Python (may not kill children)
                    process.kill()
                    process.wait(timeout=2)
            
            else:
                # Unix-like: terminate normally (process.terminate() sends SIGTERM)
                process.terminate()
                
                grace_period = 5
                try:
                    process.wait(timeout=grace_period)
                    self.logger.info(f"{name} terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"{name} did not respond to SIGTERM, sending SIGKILL...")
                    process.kill()
                    process.wait(timeout=3)
                    self.logger.info(f"{name} killed")
        
        except Exception as e:
            self.logger.error(f"Error terminating {name}: {e}")
    
    def _kill_orphaned_comfyui(self):
        """
        Kill any orphaned ComfyUI processes running on port 8188
        
        This is a safety measure to clean up processes from previous daemon runs
        that failed to shut down properly.
        
        Returns:
            Number of processes killed
        """
        if sys.platform != 'win32':
            self.logger.debug("Orphan cleanup only implemented for Windows")
            return 0
        
        killed_count = 0
        
        try:
            # Find python processes listening on port 8188 (ComfyUI's default port)
            # Use netstat to find processes using port 8188
            result = subprocess.run(
                ['netstat', '-ano', '-p', 'TCP'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse netstat output looking for :8188
                for line in result.stdout.splitlines():
                    if ':8188' in line and 'LISTENING' in line:
                        # Extract PID (last column)
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                pid = int(parts[-1])
                                self.logger.info(f"Found orphaned ComfyUI process (PID: {pid})")
                                
                                # Kill the process tree
                                subprocess.run(
                                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                                    capture_output=True,
                                    timeout=5,
                                    check=False
                                )
                                killed_count += 1
                                self.logger.info(f"Killed orphaned process tree (PID: {pid})")
                            
                            except (ValueError, subprocess.TimeoutExpired) as e:
                                self.logger.warning(f"Failed to kill orphaned process: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error checking for orphaned ComfyUI processes: {e}")
        
        return killed_count
    
    async def shutdown(self):
        """
        Graceful shutdown of all processes
        
        Order:
        1. Stop controller (gives it time to finish current frame)
        2. Stop ComfyUI
        3. Clean up PID file
        4. Final status update
        """
        self.logger.info("=" * 70)
        self.logger.info("DAEMON SHUTDOWN INITIATED")
        self.logger.info("=" * 70)
        
        self.running = False
        
        # Shutdown controller
        if self.controller_process:
            self.logger.info("Stopping controller...")
            self.controller_status = "stopping"
            self._update_daemon_status()
            
            grace_period = self.daemon_config.get('shutdown', {}).get('controller_grace_period', 10)
            self._terminate_process(self.controller_process, "Controller")
            self.controller_process = None
            self.controller_status = "stopped"
        
        # Shutdown ComfyUI
        if self.comfyui_process:
            self.logger.info("Stopping ComfyUI...")
            self.comfyui_status = "stopping"
            self._update_daemon_status()
            
            grace_period = self.daemon_config.get('shutdown', {}).get('comfyui_grace_period', 30)
            self._terminate_process(self.comfyui_process, "ComfyUI")
            self.comfyui_process = None
            self.comfyui_status = "stopped"
        
        # Clean up PID file
        self._remove_pid_file()
        
        # Final status update
        self._update_daemon_status()
        
        self.logger.info("=" * 70)
        self.logger.info("DAEMON SHUTDOWN COMPLETE")
        self.logger.info("=" * 70)
    
    async def run(self):
        """
        Main entry point for daemon
        
        Sequence:
        1. Write PID file
        2. Start ComfyUI and WAIT for it to be ready
        3. Only after ComfyUI ready, start Controller
        4. Run monitoring loop
        5. Shutdown on exit
        """
        self.running = True
        self._write_pid_file()
        
        try:
            # Start ComfyUI and wait for health check
            self.logger.info("Starting ComfyUI backend...")
            success = await self.start_comfyui()
            if not success:
                self.logger.error("ComfyUI failed to start - aborting daemon")
                self.logger.error("Controller will NOT start until ComfyUI is ready")
                return
            
            self.logger.info("ComfyUI confirmed ready - safe to start controller")
            
            # Now start Controller (only after ComfyUI is verified ready)
            success = await self.start_controller()
            if not success:
                self.logger.error("Controller failed to start - aborting daemon")
                return
            
            self.logger.info("All systems operational - entering monitoring loop")
            
            # Run monitoring loop
            await self.monitor_loop()
            
        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error in daemon: {e}", exc_info=True)
        finally:
            await self.shutdown()


def main():
    """Entry point for daemon"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dream Window Daemon - Autonomous Process Manager"
    )
    parser.add_argument(
        "--config",
        default="backend/config.yaml",
        help="Path to config file (default: backend/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Create and run daemon
    try:
        daemon = DaemonManager(config_path=args.config)
        asyncio.run(daemon.run())
    except Exception as e:
        print(f"Failed to start daemon: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

