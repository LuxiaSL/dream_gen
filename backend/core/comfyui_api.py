"""
ComfyUI API Client
Handles all interactions with ComfyUI server

This module provides HTTP and WebSocket communication with ComfyUI:
- Queue workflow execution
- Monitor progress via WebSocket
- Retrieve generated images
- Handle errors and retries
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import requests
import websockets

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """
    Client for ComfyUI API interactions
    
    Responsibilities:
    - HTTP requests to ComfyUI server (queue prompts, get history)
    - WebSocket connection for progress monitoring
    - Queue management
    - Error handling and retries
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        """
        Initialize ComfyUI client
        
        Args:
            base_url: ComfyUI server URL (default: localhost:8188)
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = str(time.time())  # Unique client ID for WebSocket
        self.session: Optional[requests.Session] = None
        self._init_session()

    def _init_session(self) -> None:
        """Initialize HTTP session for connection reuse"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DreamWindow/0.1.0",
            "Accept": "application/json",
        })

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get ComfyUI system status
        
        Returns:
            Dictionary with system information (OS, devices, etc.)
        """
        try:
            response = self.session.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}

    def get_queue(self) -> Dict[str, Any]:
        """
        Get current queue status
        
        Returns:
            Dictionary with queue information (running, pending)
        """
        try:
            response = self.session.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get queue: {e}")
            return {}

    def queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """
        Queue a workflow for generation
        
        Args:
            workflow: ComfyUI workflow JSON (node graph)
        
        Returns:
            prompt_id if successful, None otherwise
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if prompt_id:
                logger.info(f"Queued prompt: {prompt_id}")
            else:
                logger.warning(f"No prompt_id in response: {result}")
            
            return prompt_id
            
        except requests.exceptions.Timeout:
            logger.error("Queue prompt timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error queueing prompt: {e}", exc_info=True)
            return None

    async def wait_for_completion(
        self, 
        prompt_id: str, 
        timeout: float = 60.0
    ) -> bool:
        """
        Wait for a prompt to complete via WebSocket
        
        This connects to ComfyUI's WebSocket endpoint and listens for
        execution messages. Returns when the prompt completes or times out.
        
        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait (seconds)
        
        Returns:
            True if completed successfully, False otherwise
        """
        # Build WebSocket URL
        ws_base = self.base_url.replace("http://", "").replace("https://", "")
        ws_url = f"ws://{ws_base}/ws?clientId={self.client_id}"
        
        start_time = time.time()
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.debug(f"WebSocket connected, waiting for prompt {prompt_id}")
                
                while True:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.error(f"Timeout waiting for prompt {prompt_id} after {elapsed:.1f}s")
                        return False
                    
                    # Receive message with timeout
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0,
                        )
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        # Log progress updates
                        if msg_type == "progress":
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_val = progress_data.get("max", 0)
                            if max_val > 0:
                                pct = (value / max_val) * 100
                                logger.debug(f"Progress: {value}/{max_val} ({pct:.1f}%)")
                        
                        # Check for execution start
                        elif msg_type == "execution_start":
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                logger.debug(f"Execution started for prompt {prompt_id}")
                        
                        # Check for successful completion
                        elif msg_type == "execution_success" or msg_type == "executed":
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                logger.info(f"Prompt {prompt_id} completed successfully")
                                return True
                        
                        # Check for errors
                        elif msg_type == "execution_error":
                            exec_data = data.get("data", {})
                            exec_prompt_id = exec_data.get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                error_msg = exec_data.get("exception_message", "Unknown error")
                                logger.error(f"Prompt {prompt_id} failed: {error_msg}")
                                return False
                    
                    except asyncio.TimeoutError:
                        # No message received in this interval, continue waiting
                        continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode WebSocket message: {e}")
                        continue
        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error waiting for completion: {e}", exc_info=True)
            return False

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get generation history for a specific prompt
        
        Args:
            prompt_id: The prompt ID to query
        
        Returns:
            Dictionary with history information
        """
        try:
            response = self.session.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get history for {prompt_id}: {e}")
            return {}

    def get_output_images(self, prompt_id: str) -> List[str]:
        """
        Get output image filenames from completed prompt
        
        This parses the history to find generated image filenames.
        The images are in ComfyUI's output directory.
        
        Args:
            prompt_id: The completed prompt ID
        
        Returns:
            List of image filenames (relative to ComfyUI output directory)
        """
        history = self.get_history(prompt_id)
        
        if not history or prompt_id not in history:
            logger.warning(f"No history found for prompt {prompt_id}")
            return []
        
        prompt_history = history[prompt_id]
        outputs = prompt_history.get("outputs", {})
        images = []
        
        # Parse outputs from each node
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info.get("filename")
                    if filename:
                        images.append(filename)
                        logger.debug(f"Found output image: {filename}")
        
        if not images:
            logger.warning(f"No images found in outputs for prompt {prompt_id}")
        
        return images

    def get_image_path(self, filename: str, comfyui_dir: Path) -> Optional[Path]:
        """
        Get full path to a generated image
        
        Args:
            filename: Image filename from get_output_images()
            comfyui_dir: Path to ComfyUI installation directory
        
        Returns:
            Full path to image, or None if not found
        """
        output_dir = comfyui_dir / "output"
        image_path = output_dir / filename
        
        if image_path.exists():
            return image_path
        else:
            logger.error(f"Image not found: {image_path}")
            return None

    def interrupt_execution(self) -> bool:
        """
        Interrupt current execution
        
        Returns:
            True if successful
        """
        try:
            response = self.session.post(f"{self.base_url}/interrupt")
            response.raise_for_status()
            logger.info("Execution interrupted")
            return True
        except Exception as e:
            logger.error(f"Failed to interrupt execution: {e}")
            return False

    def clear_queue(self) -> bool:
        """
        Clear the pending queue
        
        Returns:
            True if successful
        """
        try:
            payload = {"clear": True}
            response = self.session.post(f"{self.base_url}/queue", json=payload)
            response.raise_for_status()
            logger.info("Queue cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            return False

    def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """
        Free GPU VRAM by unloading models
        
        This is CRITICAL for game detection! When a game starts, we need to
        unload models from VRAM to avoid conflicts and crashes.
        
        ComfyUI's /free endpoint accepts:
        - unload_models: Unload all loaded models from VRAM
        - free_memory: Run garbage collection and free cached memory
        
        Args:
            unload_models: Whether to unload models from VRAM (default: True)
            free_memory: Whether to free cached memory (default: True)
        
        Returns:
            True if successful
        """
        try:
            payload = {
                "unload_models": unload_models,
                "free_memory": free_memory,
            }
            response = self.session.post(f"{self.base_url}/free", json=payload)
            response.raise_for_status()
            logger.info("Memory freed (models unloaded from VRAM)")
            return True
        except Exception as e:
            logger.error(f"Failed to free memory: {e}")
            logger.warning("This might not be critical - ComfyUI may not have /free endpoint")
            # Don't fail hard - older ComfyUI versions might not have this endpoint
            return False

    def close(self) -> None:
        """Close the HTTP session"""
        if self.session:
            self.session.close()
            logger.debug("HTTP session closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Test function for development
async def test_api() -> bool:
    """
    Test ComfyUI API connection
    
    This should be run with ComfyUI server running to verify connectivity.
    """
    print("=" * 60)
    print("Testing ComfyUI API connection...")
    print("=" * 60)
    
    client = ComfyUIClient()
    
    # Test 1: System stats
    print("\nTest 1: System Stats")
    stats = client.get_system_stats()
    if stats:
        system_info = stats.get("system", {})
        print(f"✓ System stats: OS={system_info.get('os', 'unknown')}")
        print(f"  Python: {system_info.get('python_version', 'unknown')}")
        
        devices = stats.get("devices", [])
        if devices:
            print(f"  Devices: {len(devices)} GPU(s)")
            for i, device in enumerate(devices):
                print(f"    GPU {i}: {device.get('name', 'unknown')}")
    else:
        print("✗ Failed to get system stats (is ComfyUI running?)")
        return False
    
    # Test 2: Queue status
    print("\nTest 2: Queue Status")
    queue = client.get_queue()
    if queue:
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
        print(f"✓ Queue: {running} running, {pending} pending")
    else:
        print("✗ Failed to get queue status")
        return False
    
    print("\n" + "=" * 60)
    print("API connection test PASSED ✓")
    print("=" * 60)
    
    client.close()
    return True


if __name__ == "__main__":
    # Run test when executed directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    success = asyncio.run(test_api())
    exit(0 if success else 1)

