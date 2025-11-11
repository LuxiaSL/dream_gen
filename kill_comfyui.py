#!/usr/bin/env python3
"""
Emergency ComfyUI Killer
========================

Use this script to kill any ComfyUI processes that failed to shut down properly.

This is particularly useful when the daemon fails to terminate ComfyUI correctly,
leaving orphaned python.exe processes running on port 8188.

Usage:
    uv run kill_comfyui.py
"""

import subprocess
import sys


def kill_comfyui_windows():
    """Kill ComfyUI processes on Windows"""
    killed_count = 0
    
    print("Searching for ComfyUI processes (port 8188)...")
    
    try:
        # Find processes listening on port 8188
        result = subprocess.run(
            ['netstat', '-ano', '-p', 'TCP'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            found_processes = []
            
            # Parse netstat output
            for line in result.stdout.splitlines():
                if ':8188' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            pid = int(parts[-1])
                            if pid not in found_processes:
                                found_processes.append(pid)
                                print(f"  Found ComfyUI process: PID {pid}")
                        except ValueError:
                            pass
            
            if not found_processes:
                print("✓ No ComfyUI processes found")
                return 0
            
            # Kill each process tree
            for pid in found_processes:
                print(f"\nKilling process tree for PID {pid}...")
                try:
                    result = subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(pid)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✓ Killed process tree (PID {pid})")
                        killed_count += 1
                    else:
                        print(f"  ✗ Failed to kill PID {pid}: {result.stderr.strip()}")
                
                except subprocess.TimeoutExpired:
                    print(f"  ✗ Timeout killing PID {pid}")
                except Exception as e:
                    print(f"  ✗ Error killing PID {pid}: {e}")
            
            return killed_count
    
    except Exception as e:
        print(f"Error: {e}")
        return 0


def kill_comfyui_unix():
    """Kill ComfyUI processes on Unix-like systems"""
    print("Unix-like systems not yet implemented")
    print("Use: lsof -i :8188 to find processes")
    print("Then: kill -9 <PID> to terminate")
    return 0


def main():
    print("=" * 70)
    print("ComfyUI Emergency Killer")
    print("=" * 70)
    print()
    
    if sys.platform == 'win32':
        killed = kill_comfyui_windows()
    else:
        killed = kill_comfyui_unix()
    
    print()
    print("=" * 70)
    if killed > 0:
        print(f"✓ Successfully killed {killed} ComfyUI process(es)")
        print()
        print("You can now restart the daemon:")
        print("  uv run daemon.py")
    else:
        print("No processes were killed")
    print("=" * 70)


if __name__ == "__main__":
    main()

