"""
Prompt Manager
Handles prompt selection and rotation

This ensures variety in generation by rotating through different
aesthetic themes while maintaining coherence.
"""

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt themes and rotation
    
    Features:
    - Cycle through base themes
    - Optional time-based modifiers
    - Random vs sequential rotation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize prompt manager
        
        Args:
            config: Configuration dictionary
        """
        self.base_themes: List[str] = config["prompts"]["base_themes"]
        self.negative: str = config["prompts"]["negative"]
        self.rotation_interval: int = config["prompts"]["rotation_interval"]
        
        # Modifiers
        self.modifiers_enabled: bool = config["prompts"]["modifiers"]["enabled"]
        self.time_based: bool = config["prompts"]["modifiers"]["time_based"]
        
        # State
        self.current_theme_index = 0
        self.frames_on_current_theme = 0
        self.total_frames = 0
        
        logger.info(
            f"PromptManager initialized: {len(self.base_themes)} themes, "
            f"rotation every {self.rotation_interval} frames"
        )

    def get_next_prompt(self) -> str:
        """
        Get next prompt with rotation
        
        Switches theme every rotation_interval frames for variety
        
        Returns:
            Prompt string with optional modifiers
        """
        # Check if should rotate theme
        if self.frames_on_current_theme >= self.rotation_interval:
            self.current_theme_index = (self.current_theme_index + 1) % len(
                self.base_themes
            )
            self.frames_on_current_theme = 0
            logger.info(f"Rotated to theme {self.current_theme_index + 1}")
        
        # Get current theme
        theme = self.base_themes[self.current_theme_index]
        
        # Apply modifiers
        prompt = self._apply_modifiers(theme)
        
        # Update counters
        self.frames_on_current_theme += 1
        self.total_frames += 1
        
        return prompt

    def get_random_prompt(self) -> str:
        """
        Get completely random prompt (ignores rotation)
        
        Useful for cache injection or variation
        
        Returns:
            Random prompt from themes
        """
        theme = random.choice(self.base_themes)
        prompt = self._apply_modifiers(theme)
        return prompt

    def _apply_modifiers(self, base_prompt: str) -> str:
        """
        Apply optional modifiers to base prompt
        
        Args:
            base_prompt: Base theme prompt
        
        Returns:
            Modified prompt
        """
        if not self.modifiers_enabled:
            return base_prompt
        
        modifiers = []
        
        # Time-based modifiers
        if self.time_based:
            time_modifier = self._get_time_modifier()
            if time_modifier:
                modifiers.append(time_modifier)
        
        # Combine
        if modifiers:
            return f"{base_prompt}, {', '.join(modifiers)}"
        return base_prompt

    def _get_time_modifier(self) -> str:
        """
        Get time-of-day based modifier
        
        Returns:
            Time-based modifier string
        """
        from datetime import datetime
        
        hour = datetime.now().hour
        
        if 5 <= hour < 8:
            return "dawn light, morning atmosphere"
        elif 8 <= hour < 12:
            return "bright daylight, clear atmosphere"
        elif 12 <= hour < 17:
            return "afternoon light, warm tones"
        elif 17 <= hour < 20:
            return "twilight, golden hour lighting"
        elif 20 <= hour < 23:
            return "evening atmosphere, deep shadows"
        else:  # 23-5
            return "midnight atmosphere, deep darkness"

    def get_negative_prompt(self) -> str:
        """
        Get negative prompt
        
        Returns:
            Negative prompt string
        """
        return self.negative

    def reset_rotation(self) -> None:
        """Reset rotation state"""
        self.current_theme_index = 0
        self.frames_on_current_theme = 0
        logger.info("Rotation reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get prompt manager statistics
        
        Returns:
            Dictionary with current state
        """
        return {
            "current_theme": self.current_theme_index + 1,
            "total_themes": len(self.base_themes),
            "frames_on_theme": self.frames_on_current_theme,
            "rotation_interval": self.rotation_interval,
            "total_frames": self.total_frames,
        }


# Test function
def test_prompt_manager() -> bool:
    """Test prompt manager"""
    import yaml
    from pathlib import Path
    
    print("=" * 60)
    print("Testing PromptManager...")
    print("=" * 60)
    
    # Load config
    config_path = Path("backend/config.yaml")
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create manager
    print("\n1. Creating prompt manager...")
    manager = PromptManager(config)
    print(f"✓ Manager created: {len(manager.base_themes)} themes")
    
    # Test rotation
    print("\n2. Testing prompt rotation...")
    prompts = []
    for i in range(25):  # Go past rotation interval
        prompt = manager.get_next_prompt()
        prompts.append(prompt)
        if i % 5 == 0:
            stats = manager.get_stats()
            print(
                f"   Frame {i}: Theme {stats['current_theme']}, "
                f"Frames on theme: {stats['frames_on_theme']}"
            )
    
    # Check that rotation happened
    stats = manager.get_stats()
    if stats["current_theme"] > 1:
        print(f"✓ Rotation working: moved to theme {stats['current_theme']}")
    else:
        print("✗ Rotation not working")
        return False
    
    # Test random
    print("\n3. Testing random prompts...")
    random_prompts = [manager.get_random_prompt() for _ in range(5)]
    print(f"✓ Generated {len(random_prompts)} random prompts")
    
    # Test modifiers
    print("\n4. Testing time-based modifiers...")
    if manager.time_based:
        time_mod = manager._get_time_modifier()
        print(f"✓ Time modifier: {time_mod}")
    else:
        print("  (Time modifiers disabled)")
    
    # Test negative prompt
    print("\n5. Testing negative prompt...")
    negative = manager.get_negative_prompt()
    if negative and len(negative) > 0:
        print(f"✓ Negative prompt: {negative[:50]}...")
    else:
        print("✗ Negative prompt empty")
        return False
    
    print("\n" + "=" * 60)
    print("PromptManager test PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    
    success = test_prompt_manager()
    exit(0 if success else 1)

