"""
Game Detection Module

Detects when games are running and manages VRAM accordingly.
This prevents VRAM conflicts when memory-hungry games start.

Strategy:
1. When game detected → free VRAM (unload models)
2. When game closes → models reload automatically on next generation
3. Reload penalty: ~15 seconds (acceptable trade-off for stability)
"""

import logging
import psutil
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GameDetector:
    """
    Detect running games and manage system state
    
    Supports multiple detection methods:
    - Process name matching (most reliable)
    - Fullscreen window detection (less reliable)
    - GPU load monitoring (optional)
    """
    
    def __init__(self, config: dict):
        """
        Initialize game detector
        
        Args:
            config: Configuration dictionary with game_detection section
        """
        self.enabled = config.get("game_detection", {}).get("enabled", True)
        self.known_games = config.get("game_detection", {}).get("known_games", [])
        self.method = config.get("game_detection", {}).get("method", "process")
        self.check_interval = config.get("game_detection", {}).get("check_interval", 5.0)
        
        # Normalize game names to lowercase for matching
        self.known_games = [g.lower() for g in self.known_games]
        
        logger.info(f"GameDetector initialized: {len(self.known_games)} known games")
        if self.known_games:
            logger.debug(f"Known games: {', '.join(self.known_games)}")
    
    def is_game_running(self) -> Optional[str]:
        """
        Check if any known game is running
        
        Returns:
            Game executable name if detected, None otherwise
        """
        if not self.enabled:
            return None
        
        if self.method == "process":
            return self._check_process_names()
        elif self.method == "fullscreen":
            return self._check_fullscreen_window()
        else:
            logger.warning(f"Unknown detection method: {self.method}")
            return None
    
    def _check_process_names(self) -> Optional[str]:
        """
        Check if any known game process is running
        
        Returns:
            Game name if found, None otherwise
        """
        try:
            for proc in psutil.process_iter(['name']):
                proc_name = proc.info['name']
                if proc_name:
                    proc_name_lower = proc_name.lower()
                    
                    # Check against known games
                    for game in self.known_games:
                        if game in proc_name_lower:
                            logger.info(f"Game detected: {proc_name}")
                            return proc_name
            
            return None
            
        except psutil.Error as e:
            logger.error(f"Error checking processes: {e}")
            return None
    
    def _check_fullscreen_window(self) -> Optional[str]:
        """
        Check if a fullscreen application is running
        
        This is Windows-specific and less reliable than process detection.
        
        Returns:
            Window title if fullscreen detected, None otherwise
        """
        try:
            # This requires pywin32
            import win32gui  # pyright: ignore[reportMissingModuleSource]
            import win32api  # pyright: ignore[reportMissingModuleSource]
            
            # Get foreground window
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None
            
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Get screen resolution
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            
            # Check if window is fullscreen (with some tolerance)
            is_fullscreen = (
                width >= screen_width - 10 and 
                height >= screen_height - 10
            )
            
            if is_fullscreen:
                # Get window title
                title = win32gui.GetWindowText(hwnd)
                
                # Exclude known non-game fullscreen apps
                excluded = ['rainmeter', 'obs', 'discord']
                if not any(ex in title.lower() for ex in excluded):
                    logger.info(f"Fullscreen application detected: {title}")
                    return title
            
            return None
            
        except ImportError:
            logger.warning("pywin32 not installed - fullscreen detection unavailable")
            return None
        except Exception as e:
            logger.error(f"Error checking fullscreen: {e}")
            return None


# Quick test
def test_game_detector():
    """Test game detector"""
    print("=" * 60)
    print("Testing GameDetector...")
    print("=" * 60)
    
    config = {
        "game_detection": {
            "enabled": True,
            "method": "process",
            "check_interval": 5.0,
            "known_games": [
                "eldenring.exe",
                "cyberpunk2077.exe"
            ]
        }
    }
    
    detector = GameDetector(config)
    
    print("\nChecking for games...")
    game = detector.is_game_running()
    
    if game:
        print(f"[OK] Game detected: {game}")
    else:
        print("[X] No games detected")
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    test_game_detector()

