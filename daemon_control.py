"""
Dream Window Daemon Control CLI
================================

Command-line interface for controlling the running daemon.

Sends commands to the daemon via control file for:
- Pausing/resuming generation
- Restarting processes
- Graceful shutdown
- Status checking

Usage:
    uv run daemon_control.py pause
    uv run daemon_control.py resume
    uv run daemon_control.py shutdown
    uv run daemon_control.py restart-comfyui
    uv run daemon_control.py restart-controller
    uv run daemon_control.py status
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Optional


def resolve_path(path: str, project_root: Path) -> Path:
    """Resolve path (relative or absolute)"""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    else:
        return (project_root / path_obj).resolve()


def load_config(config_path: str = "backend/config.yaml") -> dict:
    """Load configuration"""
    project_root = Path(__file__).parent.resolve()
    config_file = project_root / config_path
    
    if not config_file.exists():
        print(f"Error: Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f), project_root


def is_daemon_running(pid_file: Path) -> tuple[bool, Optional[int]]:
    """
    Check if daemon is running
    
    Returns:
        (is_running, pid)
    """
    if not pid_file.exists():
        return False, None
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process exists
        if sys.platform == 'win32':
            # Windows: Use tasklist
            import subprocess
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout, pid
        else:
            # Unix: Send signal 0 (doesn't kill, just checks)
            try:
                os.kill(pid, 0)
                return True, pid
            except OSError:
                return False, pid
    except Exception as e:
        print(f"Error checking daemon status: {e}", file=sys.stderr)
        return False, None


def send_command(command: str, control_file: Path) -> bool:
    """
    Send command to daemon via control file
    
    Args:
        command: Command to send (PAUSE, RESUME, SHUTDOWN, etc.)
        control_file: Path to control file
    
    Returns:
        True if command was written successfully
    """
    try:
        control_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(control_file, 'w') as f:
            f.write(command.upper())
        
        return True
    except Exception as e:
        print(f"Error sending command: {e}", file=sys.stderr)
        return False


def read_status(status_file: Path) -> Optional[dict]:
    """Read current status from status.json"""
    try:
        if not status_file.exists():
            return None
        
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading status: {e}", file=sys.stderr)
        return None


def cmd_pause(config: dict, project_root: Path):
    """Pause generation"""
    control_file = resolve_path(config['daemon']['control_file'], project_root)
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    
    running, pid = is_daemon_running(pid_file)
    if not running:
        print("❌ Daemon is not running")
        return 1
    
    print(f"📤 Sending PAUSE command to daemon (PID: {pid})...")
    if send_command('PAUSE', control_file):
        print("✓ Command sent. Generation will pause shortly.")
        return 0
    else:
        print("❌ Failed to send command")
        return 1


def cmd_resume(config: dict, project_root: Path):
    """Resume generation"""
    control_file = resolve_path(config['daemon']['control_file'], project_root)
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    
    running, pid = is_daemon_running(pid_file)
    if not running:
        print("❌ Daemon is not running")
        return 1
    
    print(f"📤 Sending RESUME command to daemon (PID: {pid})...")
    if send_command('RESUME', control_file):
        print("✓ Command sent. Generation will resume shortly.")
        return 0
    else:
        print("❌ Failed to send command")
        return 1


def cmd_shutdown(config: dict, project_root: Path):
    """Shutdown daemon"""
    control_file = resolve_path(config['daemon']['control_file'], project_root)
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    
    running, pid = is_daemon_running(pid_file)
    if not running:
        print("❌ Daemon is not running")
        return 1
    
    print(f"📤 Sending SHUTDOWN command to daemon (PID: {pid})...")
    if send_command('SHUTDOWN', control_file):
        print("✓ Command sent. Daemon will shutdown gracefully.")
        print("   Waiting for shutdown...")
        
        # Wait up to 60s for daemon to stop
        for i in range(60):
            time.sleep(1)
            running, _ = is_daemon_running(pid_file)
            if not running:
                print("✓ Daemon stopped successfully")
                return 0
        
        print("⚠️  Daemon did not stop within 60s")
        return 1
    else:
        print("❌ Failed to send command")
        return 1


def cmd_restart_comfyui(config: dict, project_root: Path):
    """Restart ComfyUI"""
    control_file = resolve_path(config['daemon']['control_file'], project_root)
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    
    running, pid = is_daemon_running(pid_file)
    if not running:
        print("❌ Daemon is not running")
        return 1
    
    print(f"📤 Sending RESTART_COMFYUI command to daemon (PID: {pid})...")
    if send_command('RESTART_COMFYUI', control_file):
        print("✓ Command sent. ComfyUI will restart shortly.")
        return 0
    else:
        print("❌ Failed to send command")
        return 1


def cmd_restart_controller(config: dict, project_root: Path):
    """Restart controller"""
    control_file = resolve_path(config['daemon']['control_file'], project_root)
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    
    running, pid = is_daemon_running(pid_file)
    if not running:
        print("❌ Daemon is not running")
        return 1
    
    print(f"📤 Sending RESTART_CONTROLLER command to daemon (PID: {pid})...")
    if send_command('RESTART_CONTROLLER', control_file):
        print("✓ Command sent. Controller will restart shortly.")
        return 0
    else:
        print("❌ Failed to send command")
        return 1


def cmd_status(config: dict, project_root: Path):
    """Show daemon status"""
    pid_file = resolve_path(config['daemon']['pid_file'], project_root)
    status_file = resolve_path(config['system']['output_dir'], project_root) / "status.json"
    
    print("=" * 60)
    print("DREAM WINDOW DAEMON STATUS")
    print("=" * 60)
    
    # Check if daemon is running
    running, pid = is_daemon_running(pid_file)
    
    if running:
        print(f"✓ Daemon: RUNNING (PID: {pid})")
    else:
        print("❌ Daemon: NOT RUNNING")
        if pid:
            print(f"   (stale PID file: {pid})")
        return 1
    
    # Read detailed status
    status = read_status(status_file)
    if not status:
        print("⚠️  Could not read status.json")
        return 1
    
    print("")
    print("Process Status:")
    print(f"  ComfyUI:    {status.get('comfyui_status', 'unknown').upper()}")
    print(f"  Controller: {status.get('controller_status', 'unknown').upper()}")
    print(f"  Generation: {status.get('status', 'unknown').upper()}")
    
    print("")
    print("Statistics:")
    print(f"  Frames:         {status.get('frame_number', 0)}")
    print(f"  Gen Time:       {status.get('generation_time', 0.0)}s")
    print(f"  Cache Size:     {status.get('cache_size', 0)}")
    print(f"  Current Mode:   {status.get('current_mode', 'unknown')}")
    
    print("")
    print("Uptime:")
    print(f"  Daemon:         {status.get('daemon_uptime_hours', 0.0):.2f}h")
    print(f"  Controller:     {status.get('uptime_hours', 0.0):.2f}h")
    
    print("")
    print("Restarts:")
    print(f"  ComfyUI:        {status.get('comfyui_restarts', 0)}")
    print(f"  Controller:     {status.get('controller_restarts', 0)}")
    
    print("")
    print("=" * 60)
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Dream Window Daemon Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  pause              Pause generation (stop controller, keep ComfyUI running)
  resume             Resume generation (restart controller)
  shutdown           Gracefully shutdown daemon and all processes
  restart-comfyui    Restart ComfyUI backend only
  restart-controller Restart generation controller only
  status             Show detailed daemon status

Examples:
  uv run daemon_control.py pause
  uv run daemon_control.py status
  uv run daemon_control.py shutdown
        """
    )
    
    parser.add_argument(
        'command',
        choices=['pause', 'resume', 'shutdown', 'restart-comfyui', 'restart-controller', 'status'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        default='backend/config.yaml',
        help='Path to config file (default: backend/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config, project_root = load_config(args.config)
    
    # Execute command
    commands = {
        'pause': cmd_pause,
        'resume': cmd_resume,
        'shutdown': cmd_shutdown,
        'restart-comfyui': cmd_restart_comfyui,
        'restart-controller': cmd_restart_controller,
        'status': cmd_status,
    }
    
    cmd_func = commands[args.command]
    exit_code = cmd_func(config, project_root)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

