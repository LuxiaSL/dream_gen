# 🏗️ Core Modules

**Generator, API Client, and Workflow Builder**

---

## 🎯 DreamGenerator (generator.py)

### Purpose
High-level interface for image generation. Abstracts ComfyUI complexity.

### Key Methods

#### `generate_from_prompt(prompt, seed)`
Text-to-image generation:
1. Build txt2img workflow
2. Queue prompt
3. Wait for completion
4. Retrieve and save

#### `generate_from_image(image_path, prompt, denoise, seed)`
Image-to-image generation (PRIMARY METHOD):
1. Build img2img workflow
2. Copy image to ComfyUI input
3. Queue prompt
4. Wait for completion
5. Retrieve output
6. Copy to output directory

### Design Pattern
**Facade Pattern** - Simple interface for complex subsystem

---

## 🌐 ComfyUIClient (comfyui_api.py)

### Purpose
Low-level ComfyUI API interactions via HTTP and WebSocket.

### Key Methods

#### `queue_prompt(workflow)`
Sends workflow JSON to ComfyUI:
- POST to `/prompt`
- Returns `prompt_id`

#### `async wait_for_completion(prompt_id, timeout)`
Waits for generation via WebSocket:
- Connects to `/ws`
- Listens for `execution_success` or `execution_error`
- Times out after specified duration

#### `get_output_images(prompt_id)`
Retrieves generated image filenames:
- GET from `/history/{prompt_id}`
- Parses outputs
- Returns list of filenames

### WebSocket Messages

```json
// Start
{"type": "execution_start", "data": {"prompt_id": "..."}}

// Progress
{"type": "progress", "data": {"value": 2, "max": 4}}

// Success
{"type": "execution_success", "data": {"prompt_id": "..."}}

// Error
{"type": "execution_error", "data": {"prompt_id": "...", "exception_message": "..."}}
```

### Error Handling
- Network errors → Retry with backoff
- Queue full → Wait and retry
- Timeout → Log and skip
- Server down → Reconnect attempt

### Design Pattern
**Client Pattern** - Encapsulates API protocol

---

## 🏗️ FluxWorkflowBuilder (workflow_builder.py)

### Purpose
Generates ComfyUI workflow JSON structures dynamically.

### Key Methods

#### `build_txt2img(prompt, negative, width, height, steps, cfg, seed)`

Creates workflow with nodes:
1. **CheckpointLoaderSimple** - Load Flux model
2. **CLIPTextEncode** (positive) - Encode prompt
3. **CLIPTextEncode** (negative) - Encode negative
4. **EmptyLatentImage** - Create blank latent
5. **KSampler** - Sample/denoise
6. **VAEDecode** - Decode to image
7. **SaveImage** - Save output

#### `build_img2img(image_path, prompt, negative, denoise, steps, cfg, seed)`

Similar but with:
1. **LoadImage** - Load input image
2. **VAEEncode** - Encode to latent
3. **KSampler** - Denoise on latent (denoise parameter key)
4. **VAEDecode** - Back to image
5. **SaveImage** - Save

### Workflow JSON Structure

```json
{
  "1": {
    "inputs": {"ckpt_name": "flux1-schnell.safetensors"},
    "class_type": "CheckpointLoaderSimple"
  },
  "2": {
    "inputs": {
      "text": "prompt here",
      "clip": ["1", 1]  // Connect to node 1, output 1
    },
    "class_type": "CLIPTextEncode"
  },
  // ... more nodes
}
```

### Key Concepts
- **Nodes**: Individual operations
- **Connections**: `["node_id", output_index]` format
- **Class Types**: Built-in ComfyUI types
- **Inputs**: Parameters for each node

### Design Pattern
**Builder Pattern** - Constructs complex objects step-by-step

---

## 🔄 Generation Flow

```
User Request
    │
    ▼
DreamGenerator.generate_from_image()
    │
    ├──> WorkflowBuilder.build_img2img()
    │        └──> Returns workflow JSON
    │
    ├──> ComfyUIClient.queue_prompt(workflow)
    │        └──> Returns prompt_id
    │
    ├──> ComfyUIClient.wait_for_completion(prompt_id)
    │        └──> WebSocket listens until done
    │
    ├──> ComfyUIClient.get_output_images(prompt_id)
    │        └──> Returns ["dream_00123_.png"]
    │
    └──> Copy from ComfyUI/output/ to project/output/
         └──> Return path to generated image
```

---

## 📊 Performance Notes

**Bottlenecks**:
1. Generation Time: 1-2s (GPU compute)
2. WebSocket Wait: Minimal overhead
3. File Copy: 50-200ms (HDD), 10ms (SSD)

**Optimization**:
- Connection pooling (requests.Session)
- Async WebSocket (non-blocking)
- Batch operations where possible

---

## 🧪 Testing

### Unit Tests
- Test workflow JSON structure
- Test API connection
- Mock ComfyUI responses

### Integration Tests
- End-to-end generation
- Error handling
- Timeout scenarios

---

**Next**: [CACHE_SYSTEM.md](CACHE_SYSTEM.md) for cache management details

