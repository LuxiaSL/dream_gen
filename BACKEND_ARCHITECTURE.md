# 🏗️ BACKEND ARCHITECTURE

**Complete Technical Design & Implementation Guide**

This document details the backend Python system architecture, design patterns, class structures, and implementation details.

---

## 📐 System Architecture Overview

### High-Level Design Philosophy

**Modular MVC-Inspired Pattern**:
- **Core**: Business logic (generation, coordination)
- **Cache**: Data persistence and retrieval
- **Interpolation**: Latent space operations  
- **Utils**: Cross-cutting concerns (I/O, logging, monitoring)

**Key Principles**:
1. **Separation of Concerns**: Each module has single responsibility
2. **Dependency Injection**: Config passed down, not global
3. **Error Isolation**: Failures in one component don't crash system
4. **Testability**: Each component can be unit tested
5. **Extensibility**: Easy to add new modes or features

---

## 📁 Directory Structure (Detailed)

```
backend/
├── __init__.py                    # Package init
├── main.py                        # Entry point, main controller
├── config.yaml                    # User configuration
├── requirements.txt               # Python dependencies
│
├── core/                          # Core generation logic
│   ├── __init__.py
│   ├── controller.py              # Main orchestration loop
│   ├── comfyui_api.py             # ComfyUI HTTP/WebSocket client
│   ├── workflow_builder.py        # Generate ComfyUI workflows
│   └── generator.py               # High-level generation interface
│
├── cache/                         # Image caching system
│   ├── __init__.py
│   ├── manager.py                 # Cache CRUD operations
│   ├── aesthetic_matcher.py       # CLIP similarity matching
│   └── database.py                # Metadata persistence (future)
│
├── interpolation/                 # Latent space operations
│   ├── __init__.py
│   ├── spherical_lerp.py          # Slerp implementation
│   ├── latent_encoder.py          # VAE encoding/decoding
│   └── hybrid_generator.py        # Combined interpolation + img2img
│
└── utils/                         # Utilities and helpers
    ├── __init__.py
    ├── file_ops.py                # Atomic file operations
    ├── logging_config.py          # Logging setup
    ├── system_monitor.py          # Game detection, GPU monitoring
    ├── prompt_manager.py          # Prompt rotation and selection
    ├── status_writer.py           # Status JSON writer
    └── frame_buffer.py            # Pre-generation buffer
```

---

## 🎯 Core Module Design

### main.py - Entry Point

**Purpose**: Main controller that orchestrates the entire system

**Class: DreamController**

```python
class DreamController:
    """
    Main controller for Dream Window
    
    Responsibilities:
    - Initialize all subsystems
    - Run main generation loop
    - Handle lifecycle (start/stop/pause)
    - Coordinate between components
    """
    
    def __init__(self, config_path: str):
        """
        Initialize controller
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize subsystems
        self.generator = DreamGenerator(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.status_writer = StatusWriter(self.output_dir)
        self.system_monitor = SystemMonitor(self.config)
        
        # State
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.current_image = None
        self.current_embedding = None
    
    async def run_generation_loop(self, max_frames: int = None):
        """
        Main generation loop
        
        Modes:
        - img2img: Pure feedback loop
        - interpolate: Pure latent interpolation
        - hybrid: Combined approach (recommended)
        """
        pass
    
    def pause(self):
        """Pause generation (e.g., when game detected)"""
        pass
    
    def resume(self):
        """Resume generation"""
        pass
    
    def stop(self):
        """Clean shutdown"""
        pass
```

**Key Methods**:

1. **run_generation_loop()**: Main async loop
   - Checks pause conditions
   - Decides generation mode
   - Coordinates cache injection
   - Writes output frames
   - Updates status

2. **pause()/resume()**: Game detection integration
   - Monitors system state
   - Pauses generation gracefully
   - Resumes from last frame

3. **stop()**: Cleanup
   - Saves cache state
   - Closes connections
   - Writes final logs

**Design Pattern**: **Coordinator Pattern**
- Doesn't do work itself
- Delegates to specialized components
- Manages state transitions

---

### core/generator.py - Generation Interface

**Purpose**: High-level interface for image generation

**Class: DreamGenerator**

```python
class DreamGenerator:
    """
    High-level generation interface
    
    Responsibilities:
    - Abstract ComfyUI details
    - Manage generation modes
    - Handle cache operations
    - Track performance metrics
    """
    
    def __init__(self, config: dict):
        """Initialize generator with dependencies"""
        self.config = config
        
        # Dependencies
        self.client = ComfyUIClient(config['system']['comfyui_url'])
        self.workflow_builder = FluxWorkflowBuilder(config)
        self.cache = CacheManager(config)
        self.aesthetic_matcher = AestheticMatcher()
        
        # State
        self.frame_count = 0
        self.performance_monitor = PerformanceMonitor()
    
    def generate_from_prompt(
        self,
        prompt: str,
        seed: Optional[int] = None
    ) -> Optional[Path]:
        """
        Generate image from text prompt (txt2img)
        
        Returns:
            Path to generated image, or None on failure
        """
        pass
    
    def generate_from_image(
        self,
        image_path: Path,
        prompt: str,
        denoise: float = 0.4,
        seed: Optional[int] = None
    ) -> Optional[Path]:
        """
        Generate image from existing image (img2img)
        
        Args:
            image_path: Source image
            prompt: Generation prompt
            denoise: Strength (0.0-1.0, higher = more change)
            seed: Random seed (None = random)
        
        Returns:
            Path to generated image, or None on failure
        """
        pass
    
    def should_inject_cache(self) -> bool:
        """Decide if should inject cached image"""
        pass
    
    def get_cached_injection(
        self,
        current_embedding: np.ndarray
    ) -> Optional[Path]:
        """Find similar cached image for injection"""
        pass
```

**Generation Flow**:

```
generate_from_image()
    │
    ├──> Build workflow (workflow_builder)
    │
    ├──> Queue prompt (comfyui_api)
    │
    ├──> Wait for completion (async)
    │
    ├──> Retrieve output image
    │
    ├──> Copy to output directory
    │
    ├──> Encode embedding (aesthetic_matcher)
    │
    └──> Add to cache (cache_manager)
```

**Design Pattern**: **Facade Pattern**
- Simple interface for complex subsystem
- Hides ComfyUI complexity
- Provides unified API

---

### core/comfyui_api.py - API Client

**Purpose**: Low-level ComfyUI API interactions

**Class: ComfyUIClient**

```python
class ComfyUIClient:
    """
    ComfyUI API client
    
    Handles:
    - HTTP requests to ComfyUI server
    - WebSocket connection for progress
    - Queue management
    - Error handling and retries
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url
        self.client_id = str(time.time())
        self.session = requests.Session()  # Reuse connections
    
    def get_system_stats(self) -> Dict:
        """GET /system_stats"""
        pass
    
    def get_queue(self) -> Dict:
        """GET /queue"""
        pass
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """
        POST /prompt
        
        Returns:
            prompt_id if successful
        """
        pass
    
    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 60.0
    ) -> bool:
        """
        Wait for prompt completion via WebSocket
        
        Listens to:
        - execution_success
        - execution_error
        - progress updates (for logging)
        
        Returns:
            True if completed successfully
        """
        pass
    
    def get_output_images(self, prompt_id: str) -> List[str]:
        """
        Get output image filenames from history
        
        Parses:
        - /history/{prompt_id}
        - Extracts image filenames from outputs
        
        Returns:
            List of filenames in ComfyUI output directory
        """
        pass
```

**WebSocket Message Handling**:

```python
# WebSocket message types from ComfyUI:
{
    "type": "execution_start",
    "data": {"prompt_id": "..."}
}

{
    "type": "progress",
    "data": {
        "value": 15,  # Current step
        "max": 20     # Total steps
    }
}

{
    "type": "execution_success",
    "data": {"prompt_id": "..."}
}

{
    "type": "execution_error",
    "data": {
        "prompt_id": "...",
        "exception_message": "..."
    }
}
```

**Error Handling Strategy**:
1. Network errors → Retry with exponential backoff
2. Queue full → Wait and retry
3. Generation timeout → Log and skip
4. Server down → Attempt reconnect

**Design Pattern**: **Client Pattern**
- Encapsulates API protocol
- Handles connection lifecycle
- Provides sync/async interfaces

---

### core/workflow_builder.py - Workflow Generation

**Purpose**: Generate ComfyUI workflow JSON dynamically

**Class: FluxWorkflowBuilder**

```python
class FluxWorkflowBuilder:
    """
    Builds ComfyUI workflow JSON structures
    
    Responsibilities:
    - Generate txt2img workflows
    - Generate img2img workflows
    - Support different models (Flux, SD 1.5, etc.)
    - Parameterize generation settings
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = "flux1-schnell.safetensors"
        self.vae_name = "flux_vae.safetensors"
    
    def build_txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 256,
        height: int = 512,
        steps: int = 4,
        cfg: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Build text-to-image workflow
        
        Node structure:
        1. CheckpointLoaderSimple
        2. CLIPTextEncode (positive)
        3. CLIPTextEncode (negative)
        4. EmptyLatentImage
        5. KSampler
        6. VAEDecode
        7. SaveImage
        
        Returns:
            Complete workflow JSON
        """
        pass
    
    def build_img2img(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        denoise: float = 0.4,
        steps: int = 4,
        cfg: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Build image-to-image workflow
        
        Node structure:
        1. CheckpointLoaderSimple
        2. LoadImage
        3. VAEEncode (image → latent)
        4. CLIPTextEncode (positive)
        5. CLIPTextEncode (negative)
        6. KSampler (with latent input)
        7. VAEDecode
        8. SaveImage
        
        Returns:
            Complete workflow JSON
        """
        pass
```

**Workflow JSON Structure**:

```json
{
  "1": {
    "inputs": {
      "ckpt_name": "flux1-schnell.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "2": {
    "inputs": {
      "text": "your prompt here",
      "clip": ["1", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  ...
}
```

**Key Concepts**:
- **Nodes**: Individual operations (load model, sample, etc.)
- **Connections**: `["node_id", output_index]` format
- **Class Types**: Built-in ComfyUI node types
- **Inputs**: Parameters for each node

**Design Pattern**: **Builder Pattern**
- Constructs complex objects step-by-step
- Encapsulates workflow complexity
- Allows different configurations

---

## 💾 Cache Module Design

### cache/manager.py - Cache CRUD Operations

**Purpose**: Store and retrieve generated images with metadata

**Class: CacheManager**

```python
class CacheManager:
    """
    Image cache with metadata
    
    Responsibilities:
    - Store images with embeddings
    - Enforce cache size limits (LRU eviction)
    - Persist metadata to disk
    - Provide query interface
    """
    
    def __init__(self, config: Dict):
        self.cache_dir = Path(config['system']['cache_dir'])
        self.max_size = config['generation']['cache']['max_size']
        
        # In-memory index
        self.entries: Dict[str, CacheEntry] = {}
        
        # Load from disk
        self.load_cache()
    
    def add(
        self,
        image_path: Path,
        prompt: str,
        generation_params: Dict,
        embedding: Optional[List[float]] = None
    ) -> str:
        """
        Add image to cache
        
        Process:
        1. Generate unique cache_id
        2. Copy image to cache directory
        3. Create CacheEntry with metadata
        4. Add to entries dict
        5. Enforce max_size (LRU eviction)
        6. Persist index to disk
        
        Returns:
            cache_id
        """
        pass
    
    def get(self, cache_id: str) -> Optional[CacheEntry]:
        """Retrieve entry by ID"""
        pass
    
    def get_all(self) -> List[CacheEntry]:
        """Get all entries"""
        pass
    
    def get_random(self) -> Optional[CacheEntry]:
        """Get random entry"""
        pass
    
    def _enforce_max_size(self):
        """
        Remove oldest entries if cache exceeds max_size
        
        Strategy:
        - Sort by timestamp
        - Remove oldest first
        - Delete both image file and metadata
        """
        pass
    
    def load_cache(self):
        """Load cache index from disk"""
        pass
    
    def save_cache(self):
        """Persist cache index to disk"""
        pass
```

**CacheEntry Structure**:

```python
@dataclass
class CacheEntry:
    """Single cache entry"""
    cache_id: str
    image_path: Path
    prompt: str
    generation_params: Dict
    embedding: Optional[List[float]]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Serialize for JSON storage"""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Deserialize from JSON"""
```

**Metadata File (cache_index.json)**:

```json
{
  "version": "1.0",
  "last_updated": "2025-11-08T10:30:00",
  "entry_count": 75,
  "entries": [
    {
      "cache_id": "cache_00001",
      "image_path": "cache/images/cache_00001.png",
      "prompt": "ethereal angel...",
      "generation_params": {
        "steps": 4,
        "denoise": 0.4,
        "cfg": 1.0
      },
      "embedding": [0.234, -0.123, ...],
      "timestamp": "2025-11-08T09:15:00"
    },
    ...
  ]
}
```

**Design Pattern**: **Repository Pattern**
- Abstracts data storage
- Provides collection-like interface
- Handles persistence details

---

### cache/aesthetic_matcher.py - Similarity Matching

**Purpose**: Find visually similar images using CLIP embeddings

**Class: AestheticMatcher**

```python
class AestheticMatcher:
    """
    CLIP-based aesthetic similarity matching
    
    Responsibilities:
    - Load CLIP model
    - Encode images to embeddings
    - Compute similarities
    - Find matching cached images
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: CLIPModel = None
        self.processor: CLIPProcessor = None
        self._load_model()
    
    def _load_model(self):
        """
        Load CLIP model
        
        Model: openai/clip-vit-base-patch32
        Size: ~600MB download
        Embedding dim: 512
        """
        pass
    
    def encode_image(self, image_path: Path) -> np.ndarray:
        """
        Encode image to CLIP embedding
        
        Process:
        1. Load and preprocess image
        2. Pass through CLIP vision encoder
        3. Normalize embedding (unit vector)
        
        Returns:
            512-dim normalized embedding
        """
        pass
    
    def encode_images_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Batch encoding for efficiency
        
        ~3x faster than individual encoding
        """
        pass
    
    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity
        
        For normalized vectors:
        similarity = dot(a, b)
        
        Returns:
            Value from -1 (opposite) to 1 (identical)
        """
        return np.dot(embedding1, embedding2)
    
    def find_similar(
        self,
        target_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar images from candidates
        
        Process:
        1. Compute similarity with all candidates
        2. Filter by threshold
        3. Sort by similarity
        4. Return top K
        
        Returns:
            List of (cache_id, similarity) tuples
        """
        pass
    
    def weighted_random_selection(
        self,
        candidates: List[Tuple[str, float]]
    ) -> Optional[str]:
        """
        Select candidate with probability proportional to similarity
        
        Uses softmax-like weighting:
        - Higher similarity = higher probability
        - Still allows variation
        
        Returns:
            Selected cache_id
        """
        pass
```

**Similarity Interpretation**:

```
0.9 - 1.0:  Nearly identical
0.8 - 0.9:  Very similar (same subject, pose)
0.7 - 0.8:  Similar aesthetic (good for injection)
0.6 - 0.7:  Related style
< 0.6:      Different aesthetic
```

**Design Pattern**: **Strategy Pattern**
- Encapsulates matching algorithm
- Could swap CLIP for other models
- Provides pluggable similarity metric

---

## 🔄 Interpolation Module Design

### interpolation/spherical_lerp.py - Latent Interpolation

**Purpose**: Smooth interpolation between latent representations

**Function: spherical_lerp**

```python
def spherical_lerp(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    t: float
) -> torch.Tensor:
    """
    Spherical linear interpolation (slerp) between latents
    
    Why slerp over linear lerp?
    - Preserves magnitude (important for latent spaces)
    - Smoother interpolation
    - Avoids "dead zones" in middle
    
    Args:
        latent_a: Starting latent (shape: [B, C, H, W])
        latent_b: Ending latent
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated latent
    
    Algorithm:
    1. Normalize latents to unit vectors
    2. Compute angle between them
    3. Interpolate along great circle
    4. Scale back to original magnitude
    """
    # Flatten to vectors
    a_flat = latent_a.flatten()
    b_flat = latent_b.flatten()
    
    # Normalize
    a_norm = a_flat / torch.norm(a_flat)
    b_norm = b_flat / torch.norm(b_flat)
    
    # Compute angle
    dot_product = torch.dot(a_norm, b_norm)
    omega = torch.acos(torch.clamp(dot_product, -1, 1))
    
    # Handle near-identical case
    if torch.abs(omega) < 1e-6:
        return (1.0 - t) * latent_a + t * latent_b
    
    # Slerp
    sin_omega = torch.sin(omega)
    result_flat = (
        torch.sin((1.0 - t) * omega) / sin_omega * a_flat +
        torch.sin(t * omega) / sin_omega * b_flat
    )
    
    # Reshape
    result = result_flat.reshape(latent_a.shape)
    
    return result
```

**Visual Comparison**:

```
Linear Interpolation:
A ──────.──────.──────.────── B
        t=0.25  0.5   0.75
        (straight line)

Spherical Interpolation:
A ───╭───.───.───.───╮─── B
      t=0.25 0.5 0.75
      (arc on sphere)
      
Result: Smoother, more natural transitions
```

---

### interpolation/latent_encoder.py - VAE Operations

**Purpose**: Encode images to latent space and decode back

**Class: LatentEncoder**

```python
class LatentEncoder:
    """
    VAE encoding/decoding operations
    
    Responsibilities:
    - Load VAE model
    - Encode images → latents
    - Decode latents → images
    - Batch operations for efficiency
    """
    
    def __init__(self, vae_path: Path):
        self.device = "cuda"
        self.vae = self._load_vae(vae_path)
    
    def _load_vae(self, vae_path: Path):
        """Load VAE model from safetensors"""
        pass
    
    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to latent space
        
        Process:
        1. Preprocess (resize, normalize)
        2. Pass through VAE encoder
        3. Sample from distribution (if variational)
        
        Input: RGB image (256×512)
        Output: Latent tensor (4×32×64 for Flux)
        
        Latent space is ~8x compressed
        """
        pass
    
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to image
        
        Process:
        1. Pass through VAE decoder
        2. Denormalize
        3. Convert to PIL Image
        
        Input: Latent tensor
        Output: RGB image
        """
        pass
    
    def encode_batch(
        self,
        images: List[Image.Image]
    ) -> torch.Tensor:
        """Batch encode for efficiency"""
        pass
    
    def decode_batch(
        self,
        latents: torch.Tensor
    ) -> List[Image.Image]:
        """Batch decode"""
        pass
```

**Usage in Interpolation Pipeline**:

```python
# Interpolation workflow:
encoder = LatentEncoder(vae_path)

# 1. Encode keyframes
latent_a = encoder.encode(image_a)
latent_b = encoder.encode(image_b)

# 2. Interpolate in latent space
interpolated_latents = []
for t in np.linspace(0, 1, num_frames):
    latent_t = spherical_lerp(latent_a, latent_b, t)
    interpolated_latents.append(latent_t)

# 3. Decode all at once
images = encoder.decode_batch(torch.stack(interpolated_latents))
```

---

## 🛠️ Utils Module Design

### utils/file_ops.py - Atomic File Operations

**Purpose**: Safe file I/O that prevents corruption

**Key Functions**:

```python
def atomic_write(image: Image.Image, output_path: Path):
    """
    Write image with atomic rename
    
    Why atomic?
    - Prevents Rainmeter from reading half-written files
    - Ensures all-or-nothing semantics
    - Works reliably on Windows
    
    Process:
    1. Write to temporary file
    2. Atomic rename (OS-level operation)
    3. Clean up
    """
    import tempfile
    import shutil
    
    # Write to temp in same directory (ensures same filesystem)
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        image.save(tmp_file, 'PNG')
        tmp_path = tmp_file.name
    
    # Atomic rename
    shutil.move(tmp_path, output_path)


def atomic_write_with_retry(
    image: Image.Image,
    output_path: Path,
    max_retries: int = 3
) -> bool:
    """
    Atomic write with retry logic
    
    Handles:
    - Permission errors (file locked by Rainmeter)
    - Disk full errors
    - Network drive issues
    """
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

### utils/system_monitor.py - System State Detection

**Purpose**: Monitor system for pause conditions

**Class: SystemMonitor**

```python
class SystemMonitor:
    """
    Monitor system state for pause conditions
    
    Detects:
    - Game processes running
    - High GPU load on gaming GPU
    - Fullscreen applications
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.known_games = config['game_detection']['known_games']
        self.gpu_threshold = config['game_detection']['gpu_threshold']
    
    def should_pause(self) -> bool:
        """
        Check if should pause generation
        
        Returns True if any condition met:
        - Known game process running
        - GPU #1 load > threshold
        - Fullscreen app detected
        """
        if self._is_game_running():
            return True
        
        if self._is_gpu_busy():
            return True
        
        if self._is_fullscreen_app():
            return True
        
        return False
    
    def _is_game_running(self) -> bool:
        """Check for known game processes"""
        import psutil
        
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() in [g.lower() for g in self.known_games]:
                return True
        
        return False
    
    def _is_gpu_busy(self) -> bool:
        """Check GPU #1 utilization"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU #1
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu > self.gpu_threshold
        except:
            return False
    
    def _is_fullscreen_app(self) -> bool:
        """Check if foreground window is fullscreen"""
        import win32gui
        import win32api
        
        hwnd = win32gui.GetForegroundWindow()
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        
        # Get screen resolution
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        # Consider fullscreen if covers > 95% of screen
        return (width >= screen_width * 0.95 and
                height >= screen_height * 0.95)
```

---

## 🎯 Design Patterns Summary

| Pattern | Where Used | Why |
|---------|------------|-----|
| **Facade** | DreamGenerator | Simplify complex subsystem |
| **Builder** | FluxWorkflowBuilder | Construct complex workflows |
| **Repository** | CacheManager | Abstract data storage |
| **Strategy** | AestheticMatcher | Pluggable algorithms |
| **Coordinator** | DreamController | Orchestrate components |
| **Client** | ComfyUIClient | Encapsulate API protocol |
| **Singleton** (lazy) | AestheticMatcher model | Share expensive resources |

---

## 🔄 Data Flow Diagrams

### Generation Flow (img2img mode)

```
DreamController.run_generation_loop()
    │
    ├──> get_next_prompt() [PromptManager]
    │
    └──> generate_from_image() [DreamGenerator]
            │
            ├──> build_img2img() [WorkflowBuilder]
            │
            ├──> queue_prompt() [ComfyUIClient]
            │
            ├──> wait_for_completion() [ComfyUIClient]
            │        └──> WebSocket listening
            │
            ├──> get_output_images() [ComfyUIClient]
            │
            ├──> Copy image to output/
            │
            ├──> encode_image() [AestheticMatcher]
            │
            └──> add() [CacheManager]
                    └──> Save metadata to disk
```

### Cache Injection Flow

```
DreamController.run_generation_loop()
    │
    ├──> should_inject_cache() [DreamGenerator]
    │        └──> Random check vs injection_probability
    │
    └──> get_cached_injection() [DreamGenerator]
            │
            ├──> get_all() [CacheManager]
            │
            ├──> find_similar() [AestheticMatcher]
            │        └──> Compute cosine similarities
            │
            └──> weighted_random_selection() [AestheticMatcher]
                    └──> Return selected cache_id
```

---

## 📊 Performance Considerations

### Bottleneck Analysis

1. **Generation Time** (1-2s)
   - 90% GPU compute (Flux inference)
   - 5% VAE encoding/decoding
   - 5% File I/O

2. **CLIP Encoding** (100-200ms)
   - GPU operation, but lightweight
   - Can be batched for efficiency

3. **File I/O** (50-200ms on HDD)
   - Atomic writes add overhead
   - SSD will reduce to ~10ms

4. **Cache Lookup** (< 1ms)
   - Pure Python, in-memory
   - Negligible impact

### Optimization Strategies

**Implemented**:
- Batch CLIP encoding
- Connection pooling (requests.Session)
- Async WebSocket for non-blocking wait
- In-memory cache index

**Future Optimizations**:
- Torch compile (10-20% speedup)
- Frame buffer (pre-generate ahead)
- Multi-threaded encoding
- GPU-accelerated similarity search

---

## 🧪 Testing Strategy

### Unit Tests (Post-MVP)

```python
# test_generator.py
def test_generation_speed():
    """Verify generation < 3 seconds"""
    
def test_cache_add_and_retrieve():
    """Test cache operations"""
    
def test_aesthetic_similarity():
    """Test CLIP encoding and similarity"""
    
def test_atomic_write():
    """Test file operations don't corrupt"""
```

### Integration Tests

```python
# test_integration.py
def test_full_generation_loop():
    """Test complete workflow end-to-end"""
    
def test_cache_injection():
    """Test cache injection actually happens"""
    
def test_pause_resume():
    """Test pause/resume functionality"""
```

### Performance Tests

```python
# test_performance.py
def test_generation_benchmark():
    """Benchmark 100 generations"""
    
def test_memory_leak():
    """Run for 1 hour, check VRAM"""
    
def test_cache_scalability():
    """Test with 1000+ cached images"""
```

---

## 📝 Code Style Guidelines

**Followed in Implementation**:

1. **Type Hints**: All function signatures
2. **Docstrings**: Google style for all classes/methods
3. **Logging**: Appropriate levels (DEBUG/INFO/WARNING/ERROR)
4. **Error Handling**: Try/except with specific exceptions
5. **Constants**: UPPER_CASE for module-level constants
6. **Private Methods**: Leading underscore (_method_name)

**Example**:

```python
from typing import Optional, List
from pathlib import Path


class ExampleClass:
    """
    Example class demonstrating code style
    
    Attributes:
        config: Configuration dictionary
        state: Current internal state
    """
    
    def __init__(self, config: dict):
        """Initialize with configuration"""
        self.config = config
        self._state = None  # Private attribute
    
    def public_method(self, param: str) -> Optional[int]:
        """
        Public method with type hints
        
        Args:
            param: Description of parameter
        
        Returns:
            Integer result, or None on failure
        
        Raises:
            ValueError: If param is invalid
        """
        try:
            result = self._private_method(param)
            return result
        except Exception as e:
            logger.error(f"Method failed: {e}")
            return None
    
    def _private_method(self, param: str) -> int:
        """Private helper method"""
        if not param:
            raise ValueError("param cannot be empty")
        return len(param)
```

---

## 🔐 Security Considerations

**Current Implementation**:
- No authentication (local only)
- No user input sanitization needed
- File paths validated before use
- No SQL injection risk (no SQL)

**If Exposing Externally** (future):
- Add API key authentication
- Validate/sanitize all prompts
- Rate limiting
- HTTPS for API calls

---

## 🚀 Extension Points

**Easy to Add**:
1. **New Generation Modes**: Add method to DreamGenerator
2. **Different Models**: Add builder class (SD15WorkflowBuilder)
3. **New Similarity Metrics**: Swap AestheticMatcher implementation
4. **Additional Monitors**: Extend SystemMonitor
5. **Output Formats**: Modify file_ops to support video/GIF

**Example: Adding SD 1.5 Support**:

```python
# In workflow_builder.py
class SD15WorkflowBuilder(BaseWorkflowBuilder):
    """Workflow builder for Stable Diffusion 1.5"""
    
    def build_img2img(self, **kwargs) -> Dict:
        # Different node structure for SD 1.5
        pass

# In config.yaml
generation:
  model: "sd15"  # Switch between "flux" and "sd15"

# In generator.py
if config['generation']['model'] == "flux":
    builder = FluxWorkflowBuilder(config)
elif config['generation']['model'] == "sd15":
    builder = SD15WorkflowBuilder(config)
```

---

## 📚 Dependencies Explained

**Core**:
- `torch`: GPU operations, tensor math
- `numpy`: Numerical operations
- `pillow`: Image I/O

**AI/ML**:
- `transformers`: CLIP model
- `huggingface-hub`: Model downloads

**API**:
- `requests`: HTTP client
- `websockets`: WebSocket client

**Config**:
- `pyyaml`: Config file parsing

**System**:
- `psutil`: Process monitoring
- `pywin32`: Windows API

---

**Backend architecture complete!** All components designed for modularity, testability, and extensibility. 🏗️✨
