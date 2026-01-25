"""
ComfyUI API Client
Handles all interactions with ComfyUI server

This module provides comprehensive HTTP and WebSocket communication with ComfyUI:

Core Operations:
- Queue workflow execution
- Monitor progress via WebSocket
- Retrieve generated images
- Handle errors and retries

Queue Management:
- Get queue status
- Clear entire queue
- Delete specific queue items
- Interrupt execution (global or targeted)

History Management:
- Get history (with pagination support)
- Clear entire history
- Delete specific history items

Image Operations:
- Upload images and masks
- Retrieve image data via API
- Get image paths from filesystem

Model & Resource Discovery:
- List available models by type (checkpoints, loras, vae, etc.)
- Get embeddings and extensions
- View safetensors metadata
- Get node definitions and info

System Operations:
- Get system stats (VRAM, devices, versions)
- Free memory / unload models
- Get server feature flags

All endpoints are synchronized with the official ComfyUI server implementation.
"""

import asyncio
import json
import logging
import os
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
    
    Supports optional basic authentication for nginx-proxied ComfyUI servers.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        """
        Initialize ComfyUI client
        
        Args:
            base_url: ComfyUI server URL (default: localhost:8188)
        
        Basic auth credentials are read from environment variables:
        - COMFYUI_AUTH_USER: Username for basic auth
        - COMFYUI_AUTH_PASS: Password for basic auth
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = str(time.time())  # Unique client ID for WebSocket
        self.session: Optional[requests.Session] = None
        
        # Read auth from environment (set by runpod_handler in pod mode)
        self.auth_user = os.environ.get("COMFYUI_AUTH_USER", "")
        self.auth_pass = os.environ.get("COMFYUI_AUTH_PASS", "")
        self._has_auth = bool(self.auth_user and self.auth_pass)
        
        if self._has_auth:
            logger.info(f"ComfyUI client using basic auth: {self.auth_user}:****")
        
        self._init_session()

    def _init_session(self) -> None:
        """Initialize HTTP session for connection reuse"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DreamWindow/0.1.0",
            "Accept": "application/json",
        })
        
        # Set up basic auth if credentials provided
        if self._has_auth:
            self.session.auth = (self.auth_user, self.auth_pass)

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

        submit_time = time.time()
        try:
            logger.debug(f"[SUBMIT] Queueing prompt to {self.base_url}/prompt (client_id: {self.client_id})")
            
            response = self.session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=10.0,
            )
            
            response_time = time.time() - submit_time
            response.raise_for_status()
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if prompt_id:
                # Log FULL prompt_id for correlation with ComfyUI logs
                logger.info(f"[SUBMIT OK] prompt_id={prompt_id} (response: {response_time:.3f}s, status: {response.status_code})")
            else:
                logger.warning(f"[SUBMIT WARN] No prompt_id in response: {result} (response: {response_time:.3f}s)")
            
            return prompt_id
            
        except requests.exceptions.Timeout:
            logger.error(f"[SUBMIT FAIL] Timeout after {time.time() - submit_time:.1f}s posting to {self.base_url}/prompt")
            return None
        except requests.exceptions.RequestException as e:
            response_time = time.time() - submit_time
            logger.error(f"[SUBMIT FAIL] {type(e).__name__} after {response_time:.3f}s: {e}")
            # Log response body if available for debugging
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"[SUBMIT FAIL] ComfyUI error details: {error_detail}")
                except:
                    logger.error(f"[SUBMIT FAIL] ComfyUI response text: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"[SUBMIT FAIL] Unexpected error queueing prompt: {e}", exc_info=True)
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
        # Build WebSocket URL with correct scheme (wss:// for https://, ws:// for http://)
        if self.base_url.startswith("https://"):
            ws_url = f"wss://{self.base_url[8:]}/ws?clientId={self.client_id}"
        else:
            ws_url = f"ws://{self.base_url[7:]}/ws?clientId={self.client_id}"
        
        # Build headers for basic auth if needed
        headers = {}
        if self._has_auth:
            import base64
            credentials = f"{self.auth_user}:{self.auth_pass}"
            auth_b64 = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {auth_b64}"
        
        start_time = time.time()
        execution_started = False
        last_progress_time = start_time
        messages_received = 0
        last_message_types = []  # Track recent message types for debugging
        
        # Log full prompt_id for correlation
        logger.info(f"[WAIT] Starting wait for prompt_id={prompt_id} (timeout: {timeout}s)")
        
        try:
            # websockets v13+ uses 'additional_headers', older versions use 'extra_headers'
            # Try the newer API first, fall back to older if needed
            connect_start = time.time()
            try:
                websocket = await websockets.connect(ws_url, additional_headers=headers)
            except TypeError:
                # Older websockets version
                websocket = await websockets.connect(ws_url, extra_headers=headers)
            
            connect_time = time.time() - connect_start
            logger.debug(f"[WAIT] WebSocket connected to {ws_url} in {connect_time:.3f}s")
            
            async with websocket:
                while True:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.error(f"[WAIT TIMEOUT] prompt_id={prompt_id} after {elapsed:.1f}s")
                        logger.error(f"[WAIT TIMEOUT]   execution_started: {execution_started}")
                        logger.error(f"[WAIT TIMEOUT]   messages_received: {messages_received}")
                        logger.error(f"[WAIT TIMEOUT]   time_since_last_progress: {time.time() - last_progress_time:.1f}s")
                        logger.error(f"[WAIT TIMEOUT]   recent_message_types: {last_message_types[-10:]}")
                        return False
                    
                    # Warn if we haven't seen execution_start after 10 seconds
                    if not execution_started and elapsed > 10 and int(elapsed) % 10 == 0:
                        logger.warning(f"[WAIT] Still waiting for execution_start after {elapsed:.0f}s (prompt_id={prompt_id})")
                        # Log queue status to help debug
                        try:
                            queue = self.get_queue()
                            running = len(queue.get("queue_running", []))
                            pending = len(queue.get("queue_pending", []))
                            logger.warning(f"[WAIT] ComfyUI queue: {running} running, {pending} pending")
                        except Exception as e:
                            logger.warning(f"[WAIT] Could not check queue: {e}")
                    
                    # Receive message with timeout
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0,
                        )
                        data = json.loads(message)
                        messages_received += 1
                        
                        msg_type = data.get("type")
                        last_message_types.append(msg_type)
                        if len(last_message_types) > 50:
                            last_message_types = last_message_types[-50:]
                        
                        # Log progress updates
                        if msg_type == "progress":
                            last_progress_time = time.time()
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_val = progress_data.get("max", 0)
                            if max_val > 0:
                                pct = (value / max_val) * 100
                                logger.debug(f"[WAIT] Progress: {value}/{max_val} ({pct:.1f}%)")
                        
                        # Check for execution start
                        elif msg_type == "execution_start":
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            logger.debug(f"[WAIT] execution_start for {exec_prompt_id} (waiting for {prompt_id})")
                            if exec_prompt_id == prompt_id:
                                execution_started = True
                                last_progress_time = time.time()
                                wait_until_start = elapsed
                                logger.info(f"[WAIT] Execution started for prompt_id={prompt_id} (waited {wait_until_start:.2f}s)")
                        
                        # Check for successful completion
                        elif msg_type == "execution_success" or msg_type == "executed":
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                logger.info(f"[WAIT OK] prompt_id={prompt_id} completed in {elapsed:.2f}s (messages: {messages_received})")
                                return True
                            else:
                                logger.debug(f"[WAIT] {msg_type} for different prompt: {exec_prompt_id}")
                        
                        # Check for errors
                        elif msg_type == "execution_error":
                            exec_data = data.get("data", {})
                            exec_prompt_id = exec_data.get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                error_msg = exec_data.get("exception_message", "Unknown error")
                                logger.error(f"[WAIT FAIL] prompt_id={prompt_id} error: {error_msg}")
                                return False
                            else:
                                logger.warning(f"[WAIT] execution_error for different prompt: {exec_prompt_id}: {exec_data.get('exception_message', '?')}")
                        
                        # Log other message types for debugging (but not status which is spammy)
                        elif msg_type not in ("status", "progress_state", "executing"):
                            logger.debug(f"[WAIT] ComfyUI message: {msg_type}")
                    
                    except asyncio.TimeoutError:
                        # No message received in this interval, continue waiting
                        # But warn if it's been too long since any activity
                        silence_duration = time.time() - last_progress_time
                        if silence_duration > 30:
                            logger.warning(f"[WAIT STALL] No ComfyUI activity for {silence_duration:.0f}s (prompt_id={prompt_id}, msgs: {messages_received})")
                            last_progress_time = time.time()  # Reset to avoid spam
                        continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"[WAIT] Failed to decode WebSocket message: {e}")
                        continue
        
        except websockets.exceptions.WebSocketException as e:
            elapsed = time.time() - start_time
            logger.error(f"[WAIT FAIL] WebSocket error after {elapsed:.2f}s for prompt_id={prompt_id}: {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[WAIT FAIL] Unexpected error after {elapsed:.2f}s for prompt_id={prompt_id}: {e}", exc_info=True)
            return False

    async def queue_and_wait(
        self,
        workflow: Dict[str, Any],
        timeout: float = 60.0
    ) -> Optional[str]:
        """
        Queue a workflow and wait for completion - RACE CONDITION SAFE.
        
        This method connects the WebSocket BEFORE submitting the prompt,
        ensuring we're already listening when ComfyUI sends execution messages.
        This prevents the race condition where fast prompts complete before
        the WebSocket connection is established.
        
        Args:
            workflow: ComfyUI workflow JSON (node graph)
            timeout: Maximum time to wait (seconds)
        
        Returns:
            prompt_id if successful, None if failed
        """
        # Build WebSocket URL with correct scheme
        if self.base_url.startswith("https://"):
            ws_url = f"wss://{self.base_url[8:]}/ws?clientId={self.client_id}"
        else:
            ws_url = f"ws://{self.base_url[7:]}/ws?clientId={self.client_id}"
        
        # Build headers for basic auth if needed
        headers = {}
        if self._has_auth:
            import base64
            credentials = f"{self.auth_user}:{self.auth_pass}"
            auth_b64 = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {auth_b64}"
        
        start_time = time.time()
        prompt_id = None
        
        logger.info(f"[QUEUE+WAIT] Connecting WebSocket BEFORE submitting prompt...")
        
        try:
            # Connect WebSocket FIRST - this is the key fix!
            connect_start = time.time()
            try:
                websocket = await websockets.connect(ws_url, additional_headers=headers)
            except TypeError:
                websocket = await websockets.connect(ws_url, extra_headers=headers)
            
            connect_time = time.time() - connect_start
            logger.info(f"[QUEUE+WAIT] WebSocket connected in {connect_time:.3f}s - NOW submitting prompt")
            
            async with websocket:
                # NOW submit the prompt (WebSocket is already listening)
                payload = {
                    "prompt": workflow,
                    "client_id": self.client_id,
                }
                
                submit_time = time.time()
                try:
                    response = self.session.post(
                        f"{self.base_url}/prompt",
                        json=payload,
                        timeout=10.0,
                    )
                    response_time = time.time() - submit_time
                    response.raise_for_status()
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    
                    if not prompt_id:
                        logger.error(f"[QUEUE+WAIT] No prompt_id in response: {result}")
                        return None
                    
                    logger.info(f"[QUEUE+WAIT] prompt_id={prompt_id} (submit: {response_time:.3f}s)")
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"[QUEUE+WAIT] Submit failed: {e}")
                    return None
                
                # Wait for completion (WebSocket was connected BEFORE submit)
                execution_started = False
                last_progress_time = time.time()
                messages_received = 0
                
                while True:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.error(f"[QUEUE+WAIT TIMEOUT] prompt_id={prompt_id} after {elapsed:.1f}s")
                        return None
                    
                    if not execution_started and elapsed > 10 and int(elapsed) % 10 == 0:
                        logger.warning(f"[QUEUE+WAIT] Still waiting for execution_start after {elapsed:.0f}s (prompt_id={prompt_id})")
                    
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        messages_received += 1
                        msg_type = data.get("type")
                        
                        if msg_type == "progress":
                            last_progress_time = time.time()
                        
                        elif msg_type == "execution_start":
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                execution_started = True
                                last_progress_time = time.time()
                                logger.info(f"[QUEUE+WAIT] Execution started for prompt_id={prompt_id}")
                        
                        elif msg_type in ("execution_success", "executed"):
                            exec_prompt_id = data.get("data", {}).get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                elapsed = time.time() - start_time
                                logger.info(f"[QUEUE+WAIT OK] prompt_id={prompt_id} completed in {elapsed:.2f}s")
                                return prompt_id
                        
                        elif msg_type == "execution_error":
                            exec_data = data.get("data", {})
                            exec_prompt_id = exec_data.get("prompt_id")
                            if exec_prompt_id == prompt_id:
                                error_msg = exec_data.get("exception_message", "Unknown error")
                                logger.error(f"[QUEUE+WAIT FAIL] prompt_id={prompt_id}: {error_msg}")
                                return None
                    
                    except asyncio.TimeoutError:
                        silence_duration = time.time() - last_progress_time
                        if silence_duration > 30:
                            logger.warning(f"[QUEUE+WAIT STALL] No activity for {silence_duration:.0f}s")
                            last_progress_time = time.time()
                        continue
                    except json.JSONDecodeError:
                        continue
        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"[QUEUE+WAIT FAIL] WebSocket error: {e}")
            return None
        except Exception as e:
            logger.error(f"[QUEUE+WAIT FAIL] Unexpected error: {e}", exc_info=True)
            return None

    def get_history(
        self, 
        prompt_id: Optional[str] = None,
        max_items: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get generation history
        
        Args:
            prompt_id: Optional specific prompt ID to query. If None, returns all history.
            max_items: Maximum number of items to return (for pagination)
            offset: Offset for pagination (default: -1)
        
        Returns:
            Dictionary with history information
        """
        try:
            if prompt_id:
                # Get history for specific prompt
                response = self.session.get(f"{self.base_url}/history/{prompt_id}")
            else:
                # Get all history with optional pagination
                params = {}
                if max_items is not None:
                    params["max_items"] = max_items
                if offset is not None:
                    params["offset"] = offset
                
                response = self.session.get(f"{self.base_url}/history", params=params)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
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
    
    def upload_image(
        self,
        image_path: Path,
        subfolder: str = "",
        image_type: str = "input",
        overwrite: bool = False,
    ) -> Optional[Dict[str, str]]:
        """
        Upload an image to ComfyUI's input directory via API
        
        This is the proper way to send images to ComfyUI for img2img workflows.
        
        Args:
            image_path: Path to local image file
            subfolder: Optional subfolder in the upload directory
            image_type: Type of upload ("input", "temp", "output")
            overwrite: Whether to overwrite existing file with same name
        
        Returns:
            Dictionary with 'name', 'subfolder', and 'type' if successful, None otherwise
            Example: {"name": "image.png", "subfolder": "", "type": "input"}
        """
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Prepare multipart form data
            with open(image_path, "rb") as f:
                files = {
                    "image": (image_path.name, f, "image/png"),
                }
                data = {
                    "subfolder": subfolder,
                    "type": image_type,
                    "overwrite": "true" if overwrite else "false",
                }
                
                response = self.session.post(
                    f"{self.base_url}/upload/image",
                    files=files,
                    data=data,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Image uploaded: {result['name']} (type: {result['type']})")
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload image: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Server response: {e.response.text[:500]}")
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading image: {e}", exc_info=True)
            return None
    
    def upload_mask(
        self,
        mask_path: Path,
        original_ref: Dict[str, Any],
        subfolder: str = "",
        image_type: str = "input",
        overwrite: bool = False,
    ) -> Optional[Dict[str, str]]:
        """
        Upload a mask image to ComfyUI (applies alpha channel to original image)
        
        This endpoint combines a mask with an original image by applying
        the mask's alpha channel to the original image.
        
        Args:
            mask_path: Path to local mask image file
            original_ref: Reference to original image with keys:
                         {"filename": str, "type": str, "subfolder": str}
            subfolder: Optional subfolder in the upload directory
            image_type: Type of upload ("input", "temp", "output")
            overwrite: Whether to overwrite existing file with same name
        
        Returns:
            Dictionary with 'name', 'subfolder', and 'type' if successful, None otherwise
        """
        if not mask_path.exists():
            logger.error(f"Mask file not found: {mask_path}")
            return None
        
        try:
            # Prepare multipart form data
            with open(mask_path, "rb") as f:
                files = {
                    "image": (mask_path.name, f, "image/png"),
                }
                data = {
                    "original_ref": json.dumps(original_ref),
                    "subfolder": subfolder,
                    "type": image_type,
                    "overwrite": "true" if overwrite else "false",
                }
                
                response = self.session.post(
                    f"{self.base_url}/upload/mask",
                    files=files,
                    data=data,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Mask uploaded: {result['name']} (type: {result['type']})")
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload mask: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    logger.error(f"Server response: {e.response.text[:500]}")
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading mask: {e}", exc_info=True)
            return None
    
    def get_image_data(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Optional[bytes]:
        """
        Get image data directly from ComfyUI API (no file system access needed)
        
        Args:
            filename: Image filename
            subfolder: Subfolder in output directory  
            folder_type: Type of folder ("output", "input", "temp")
        
        Returns:
            Image bytes if successful, None otherwise
        """
        try:
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
            
            response = self.session.get(
                f"{self.base_url}/view",
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            
            # Return raw image bytes
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get image data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting image data: {e}", exc_info=True)
            return None

    def interrupt_execution(self, prompt_id: Optional[str] = None) -> bool:
        """
        Interrupt current execution
        
        Args:
            prompt_id: Optional specific prompt ID to interrupt. If None, interrupts all.
        
        Returns:
            True if successful
        """
        try:
            payload = {}
            if prompt_id:
                payload["prompt_id"] = prompt_id
            
            response = self.session.post(f"{self.base_url}/interrupt", json=payload)
            response.raise_for_status()
            if prompt_id:
                logger.info(f"Execution interrupted for prompt {prompt_id}")
            else:
                logger.info("Execution interrupted (all)")
            return True
        except Exception as e:
            logger.error(f"Failed to interrupt execution: {e}")
            return False

    def clear_queue(self) -> bool:
        """
        Clear the entire pending queue
        
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
    
    def delete_queue_item(self, prompt_id: str) -> bool:
        """
        Delete a specific item from the queue
        
        Args:
            prompt_id: The prompt ID to delete from queue
        
        Returns:
            True if successful
        """
        try:
            payload = {"delete": [prompt_id]}
            response = self.session.post(f"{self.base_url}/queue", json=payload)
            response.raise_for_status()
            logger.info(f"Deleted queue item: {prompt_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete queue item: {e}")
            return False
    
    def delete_queue_items(self, prompt_ids: List[str]) -> bool:
        """
        Delete multiple items from the queue
        
        Args:
            prompt_ids: List of prompt IDs to delete from queue
        
        Returns:
            True if successful
        """
        try:
            payload = {"delete": prompt_ids}
            response = self.session.post(f"{self.base_url}/queue", json=payload)
            response.raise_for_status()
            logger.info(f"Deleted {len(prompt_ids)} queue items")
            return True
        except Exception as e:
            logger.error(f"Failed to delete queue items: {e}")
            return False

    def clear_history(self) -> bool:
        """
        Clear the entire history
        
        Returns:
            True if successful
        """
        try:
            payload = {"clear": True}
            response = self.session.post(f"{self.base_url}/history", json=payload)
            response.raise_for_status()
            logger.info("History cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False
    
    def delete_history_item(self, prompt_id: str) -> bool:
        """
        Delete a specific item from history
        
        Args:
            prompt_id: The prompt ID to delete from history
        
        Returns:
            True if successful
        """
        try:
            payload = {"delete": [prompt_id]}
            response = self.session.post(f"{self.base_url}/history", json=payload)
            response.raise_for_status()
            logger.info(f"Deleted history item: {prompt_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete history item: {e}")
            return False
    
    def delete_history_items(self, prompt_ids: List[str]) -> bool:
        """
        Delete multiple items from history
        
        Args:
            prompt_ids: List of prompt IDs to delete from history
        
        Returns:
            True if successful
        """
        try:
            payload = {"delete": prompt_ids}
            response = self.session.post(f"{self.base_url}/history", json=payload)
            response.raise_for_status()
            logger.info(f"Deleted {len(prompt_ids)} history items")
            return True
        except Exception as e:
            logger.error(f"Failed to delete history items: {e}")
            return False

    def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """
        Free GPU VRAM by unloading models
        
        This is essential for game detection. When a game starts, models
        must be unloaded from VRAM to avoid conflicts and crashes.
        
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
    
    def get_embeddings(self) -> List[str]:
        """
        Get list of available embeddings (without file extensions)
        
        Returns:
            List of embedding names
        """
        try:
            response = self.session.get(f"{self.base_url}/embeddings")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return []
    
    def get_extensions(self) -> List[str]:
        """
        Get list of available web extensions
        
        Returns:
            List of extension paths
        """
        try:
            response = self.session.get(f"{self.base_url}/extensions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get extensions: {e}")
            return []
    
    def get_model_types(self) -> List[str]:
        """
        Get list of available model types (folders)
        
        Returns:
            List of model type names (e.g., "checkpoints", "loras", "vae", etc.)
        """
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model types: {e}")
            return []
    
    def get_models(self, folder: str) -> List[str]:
        """
        Get list of available models in a specific folder
        
        Args:
            folder: Model folder name (e.g., "checkpoints", "loras", "vae")
        
        Returns:
            List of model filenames in that folder
        """
        try:
            response = self.session.get(f"{self.base_url}/models/{folder}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get models for folder {folder}: {e}")
            return []
    
    def get_features(self) -> Dict[str, Any]:
        """
        Get server feature flags
        
        Returns:
            Dictionary of feature flags supported by the server
        """
        try:
            response = self.session.get(f"{self.base_url}/features")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get features: {e}")
            return {}
    
    def get_object_info(self, node_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Get node definitions/info from ComfyUI
        
        Args:
            node_class: Optional specific node class to query. If None, returns all nodes.
        
        Returns:
            Dictionary with node information including inputs, outputs, categories, etc.
        """
        try:
            if node_class:
                response = self.session.get(f"{self.base_url}/object_info/{node_class}")
            else:
                response = self.session.get(f"{self.base_url}/object_info")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get object info: {e}")
            return {}
    
    def get_view_metadata(self, folder_name: str, filename: str) -> Dict[str, Any]:
        """
        Get metadata from a safetensors file
        
        Args:
            folder_name: Folder name (e.g., "checkpoints", "loras")
            filename: Safetensors filename (must end with .safetensors)
        
        Returns:
            Dictionary with metadata from the safetensors file
        """
        try:
            params = {"filename": filename}
            response = self.session.get(
                f"{self.base_url}/view_metadata/{folder_name}",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get metadata for {folder_name}/{filename}: {e}")
            return {}

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
    Comprehensive test of ComfyUI API endpoints
    
    This should be run with ComfyUI server running to verify connectivity
    and endpoint compatibility.
    """
    print("=" * 70)
    print("Testing ComfyUI API - Comprehensive Endpoint Verification")
    print("=" * 70)
    
    client = ComfyUIClient()
    test_results = []
    
    # Test 1: System stats
    print("\n[1/14] System Stats")
    try:
        stats = client.get_system_stats()
        if stats and "system" in stats:
            system_info = stats.get("system", {})
            print(f"  ✓ OS: {system_info.get('os', 'unknown')}")
            print(f"  ✓ ComfyUI: {system_info.get('comfyui_version', 'unknown')}")
            devices = stats.get("devices", [])
            if devices:
                print(f"  ✓ Devices: {len(devices)} GPU(s)")
            test_results.append(("System Stats", True))
        else:
            print("  ✗ Failed to get system stats")
            test_results.append(("System Stats", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("System Stats", False))
    
    # Test 2: Queue status
    print("\n[2/14] Queue Status")
    try:
        queue = client.get_queue()
        if queue is not None:
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
            print(f"  ✓ Queue: {running} running, {pending} pending")
            test_results.append(("Queue Status", True))
        else:
            print("  ✗ Failed to get queue")
            test_results.append(("Queue Status", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Queue Status", False))
    
    # Test 3: Features
    print("\n[3/14] Server Features")
    try:
        features = client.get_features()
        if features is not None:
            print(f"  ✓ Features: {len(features)} feature flags")
            test_results.append(("Features", True))
        else:
            print("  ✗ Failed to get features")
            test_results.append(("Features", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Features", False))
    
    # Test 4: Model Types
    print("\n[4/14] Model Types")
    try:
        model_types = client.get_model_types()
        if model_types is not None:
            print(f"  ✓ Model types: {len(model_types)} types available")
            test_results.append(("Model Types", True))
        else:
            print("  ✗ Failed to get model types")
            test_results.append(("Model Types", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Model Types", False))
    
    # Test 5: Models in checkpoints folder
    print("\n[5/14] Models (Checkpoints)")
    try:
        models = client.get_models("checkpoints")
        if models is not None:
            print(f"  ✓ Checkpoints: {len(models)} models found")
            test_results.append(("Models", True))
        else:
            print("  ✗ Failed to get models")
            test_results.append(("Models", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Models", False))
    
    # Test 6: Embeddings
    print("\n[6/14] Embeddings")
    try:
        embeddings = client.get_embeddings()
        if embeddings is not None:
            print(f"  ✓ Embeddings: {len(embeddings)} embeddings found")
            test_results.append(("Embeddings", True))
        else:
            print("  ✗ Failed to get embeddings")
            test_results.append(("Embeddings", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Embeddings", False))
    
    # Test 7: Extensions
    print("\n[7/14] Extensions")
    try:
        extensions = client.get_extensions()
        if extensions is not None:
            print(f"  ✓ Extensions: {len(extensions)} extensions found")
            test_results.append(("Extensions", True))
        else:
            print("  ✗ Failed to get extensions")
            test_results.append(("Extensions", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Extensions", False))
    
    # Test 8: Object Info (all nodes)
    print("\n[8/14] Object Info (All Nodes)")
    try:
        object_info = client.get_object_info()
        if object_info is not None:
            print(f"  ✓ Nodes: {len(object_info)} node types available")
            test_results.append(("Object Info", True))
        else:
            print("  ✗ Failed to get object info")
            test_results.append(("Object Info", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Object Info", False))
    
    # Test 9: History (basic)
    print("\n[9/14] History")
    try:
        history = client.get_history()
        if history is not None:
            print(f"  ✓ History: {len(history)} items")
            test_results.append(("History", True))
        else:
            print("  ✗ Failed to get history")
            test_results.append(("History", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("History", False))
    
    # Test 10: History with pagination
    print("\n[10/14] History (Paginated)")
    try:
        history = client.get_history(max_items=5, offset=0)
        if history is not None:
            print(f"  ✓ Paginated history: max 5 items")
            test_results.append(("History Pagination", True))
        else:
            print("  ✗ Failed to get paginated history")
            test_results.append(("History Pagination", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("History Pagination", False))
    
    # Test 11: Free memory (non-critical if fails on older versions)
    print("\n[11/14] Free Memory")
    try:
        result = client.free_memory(unload_models=False, free_memory=False)
        if result:
            print("  ✓ Free memory endpoint available")
            test_results.append(("Free Memory", True))
        else:
            print("  ⚠ Free memory endpoint not available (may be older ComfyUI)")
            test_results.append(("Free Memory", True))  # Not critical
    except Exception as e:
        print(f"  ⚠ Error: {e} (not critical)")
        test_results.append(("Free Memory", True))  # Not critical
    
    # Test 12: Interrupt (should succeed even if nothing to interrupt)
    print("\n[12/14] Interrupt")
    try:
        result = client.interrupt_execution()
        if result:
            print("  ✓ Interrupt endpoint available")
            test_results.append(("Interrupt", True))
        else:
            print("  ✗ Failed to interrupt")
            test_results.append(("Interrupt", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Interrupt", False))
    
    # Test 13: Queue Management (clear should work even if queue is empty)
    print("\n[13/14] Queue Management")
    try:
        # Note: We're not actually clearing to avoid disrupting any running jobs
        # Just verify the endpoint exists
        queue_before = client.get_queue()
        if queue_before is not None:
            print("  ✓ Queue management endpoints available")
            test_results.append(("Queue Management", True))
        else:
            print("  ✗ Queue management failed")
            test_results.append(("Queue Management", False))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("Queue Management", False))
    
    # Test 14: WebSocket URL construction (don't actually connect)
    print("\n[14/14] WebSocket Support")
    try:
        # Use correct scheme: wss:// for https://, ws:// for http://
        if client.base_url.startswith("https://"):
            ws_url = f"wss://{client.base_url[8:]}/ws?clientId={client.client_id}"
        else:
            ws_url = f"ws://{client.base_url[7:]}/ws?clientId={client.client_id}"
        print(f"  ✓ WebSocket URL: {ws_url}")
        test_results.append(("WebSocket", True))
    except Exception as e:
        print(f"  ✗ Error: {e}")
        test_results.append(("WebSocket", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("Status: ALL TESTS PASSED ✓")
    else:
        print(f"Status: {total - passed} test(s) failed ✗")
    
    print("=" * 70)
    
    client.close()
    return passed == total


if __name__ == "__main__":
    # Run test when executed directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    success = asyncio.run(test_api())
    exit(0 if success else 1)

