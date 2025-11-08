# ⏱️ Saturday Afternoon Session 1: API Client

**Goal**: Python controller that calls ComfyUI API and generates images

**Duration**: 2 hours

---

## Task 1.1: ComfyUI API Wrapper (45 min)

Create `backend/core/comfyui_api.py`:

```python
"""
ComfyUI API Client
Handles all interactions with ComfyUI server
"""
import requests
import websockets
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for ComfyUI API interactions"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url
        self.client_id = str(time.time())
        
    def get_system_stats(self) -> Dict:
        """Get ComfyUI system status"""
        try:
            response = requests.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_queue(self) -> Dict:
        """Get current queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get queue: {e}")
            return {}
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """
        Queue a workflow for generation
        Returns prompt_id if successful
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            prompt_id = result.get("prompt_id")
            logger.info(f"Queued prompt: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None
    
    async def wait_for_completion(self, prompt_id: str, timeout: float = 60.0) -> bool:
        """
        Wait for a prompt to complete via WebSocket
        Returns True if completed successfully
        """
        ws_url = f"ws://{self.base_url.replace('http://', '')}/ws?clientId={self.client_id}"
        start_time = time.time()
        
        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    if time.time() - start_time > timeout:
                        logger.error(f"Timeout waiting for prompt {prompt_id}")
                        return False
                    
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=1.0
                        )
                        data = json.loads(message)
                        
                        if data.get("type") == "execution_success":
                            if data.get("data", {}).get("prompt_id") == prompt_id:
                                logger.info(f"Prompt {prompt_id} completed")
                                return True
                        
                        if data.get("type") == "execution_error":
                            if data.get("data", {}).get("prompt_id") == prompt_id:
                                logger.error(f"Prompt {prompt_id} failed")
                                return False
                                
                    except asyncio.TimeoutError:
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return False
    
    def get_history(self, prompt_id: str) -> Dict:
        """Get generation history for a specific prompt"""
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {}
    
    def get_output_images(self, prompt_id: str) -> list:
        """
        Get output image paths from completed prompt
        Returns list of image filenames
        """
        history = self.get_history(prompt_id)
        
        if not history or prompt_id not in history:
            return []
        
        outputs = history[prompt_id].get("outputs", {})
        images = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(img["filename"])
        
        return images
```

### Validation

```powershell
# Make sure ComfyUI is running
cd C:\AI\DreamWindow
python -c "from backend.core.comfyui_api import ComfyUIClient; c = ComfyUIClient(); print('✓ Stats:', c.get_system_stats() != {})"
```

**Expected output**: `✓ Stats: True`

---

## Task 1.2: Create `__init__.py` files

Make sure Python can import:

```powershell
# Create __init__ files
echo. > backend\__init__.py
echo. > backend\core\__init__.py
```

---

## ✅ Success Criteria

- [ ] API client created
- [ ] Can connect to ComfyUI
- [ ] System stats retrieved
- [ ] No import errors

---

## Next

Continue to **SAT_02_WORKFLOW_BUILDER.md**

---

**Session 1 of 8 Complete** | ~45 minutes

