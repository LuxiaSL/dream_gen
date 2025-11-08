# 🏗️ Utilities Module

**File operations, system monitoring, and helpers**

---

## 📁 file_ops.py

### Purpose
Safe file I/O that prevents corruption during concurrent reads/writes.

### Key Function: `atomic_write()`

**Why atomic?**
- Prevents Rainmeter from reading half-written files
- Ensures all-or-nothing semantics
- Works reliably on Windows

**Implementation**:
```python
def atomic_write(image: Image.Image, output_path: Path):
    """Write image with atomic rename"""
    import tempfile
    import shutil
    
    # Write to temp in same directory
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        image.save(tmp_file, 'PNG')
        tmp_path = tmp_file.name
    
    # Atomic rename (OS-level operation)
    shutil.move(tmp_path, output_path)
```

**With retry logic**:
```python
def atomic_write_with_retry(image, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            atomic_write(image, output_path)
            return True
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
        except Exception as e:
            logger.error(f"Write failed: {e}")
            return False
    return False
```

---

## 🖥️ system_monitor.py

### Purpose
Monitor system state for pause conditions (game detection).

### Class: SystemMonitor

#### `should_pause()`
Returns True if any condition met:
- Known game process running
- GPU #1 load > threshold
- Fullscreen app detected

#### `_is_game_running()`
Checks for known game processes:
```python
import psutil

for proc in psutil.process_iter(['name']):
    if proc.info['name'].lower() in [g.lower() for g in known_games]:
        return True
```

#### `_is_gpu_busy()`
Checks GPU #1 utilization:
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU #1
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
return util.gpu > threshold  # e.g., 70%
```

#### `_is_fullscreen_app()`
Detects fullscreen windows:
```python
import win32gui
import win32api

hwnd = win32gui.GetForegroundWindow()
rect = win32gui.GetWindowRect(hwnd)
width = rect[2] - rect[0]
height = rect[3] - rect[1]

screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)

return (width >= screen_width * 0.95 and 
        height >= screen_height * 0.95)
```

---

## 📝 prompt_manager.py

### Purpose
Manage prompt rotation and selection.

### Key Methods

#### `get_next_prompt()`
Returns next prompt from rotation:
- Cycles through base_themes
- Adds modifiers (time-based, random)
- Tracks rotation count

#### `get_negative_prompt()`
Returns negative prompt from config

#### Example Usage

```python
manager = PromptManager(config)

# Every frame
prompt = manager.get_next_prompt()  
# "ethereal angel..." for frames 1-20
# "abstract geometry..." for frames 21-40
# etc.

negative = manager.get_negative_prompt()
# "photorealistic, blurry, low quality..."
```

---

## 📊 status_writer.py

### Purpose
Write status.json for Rainmeter to read.

### Status JSON Structure

```json
{
  "frame_number": 234,
  "generation_time": 1.8,
  "status": "live",
  "current_mode": "img2img",
  "current_prompt": "ethereal digital angel...",
  "cache_size": 68,
  "cache_hits": 12,
  "uptime_hours": 4.2,
  "gpu_temp": 65,
  "vram_used_gb": 7.2,
  "paused": false,
  "last_update": "2025-11-08T16:42:33"
}
```

### Key Method

```python
def write_status(status_dict: Dict):
    """Atomically write status JSON"""
    status_path = Path("output/status.json")
    
    # Add timestamp
    status_dict["last_update"] = datetime.now().isoformat()
    
    # Atomic write
    temp_path = status_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(status_dict, f, indent=2)
    
    shutil.move(temp_path, status_path)
```

---

## 🔧 logging_config.py

### Purpose
Centralized logging configuration.

### Setup

```python
def setup_logging(log_level: str = "INFO"):
    """Configure logging for entire application"""
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/dream_controller.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
```

### Usage

```python
# In main.py
from backend.utils.logging_config import setup_logging

setup_logging(config['system']['log_level'])

# In other modules
import logging
logger = logging.getLogger(__name__)

logger.info("Generation started")
logger.error(f"Failed to generate: {error}")
```

---

## 🎯 Design Patterns

| Utility | Pattern | Purpose |
|---------|---------|---------|
| atomic_write | Atomic Operations | Prevent corruption |
| SystemMonitor | Observer | Monitor system state |
| PromptManager | Strategy | Manage prompt selection |
| StatusWriter | Singleton | Centralized status |

---

## 📊 Performance Notes

**File Operations**:
- Atomic write: +10-20ms overhead
- Worth it for reliability
- Critical for concurrent access

**System Monitoring**:
- Process check: ~10ms
- GPU check: ~5ms
- Fullscreen check: ~1ms
- Run every 5 seconds (minimal impact)

**Logging**:
- Negligible overhead with buffering
- File writes async
- Console output can be disabled for production

---

**Next**: [DATA_FLOW.md](DATA_FLOW.md) for complete system data flow diagrams

