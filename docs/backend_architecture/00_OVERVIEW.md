# 🏗️ Backend Architecture Overview

**Complete system design and organization**

---

## 📁 Directory Structure

```
backend/
├── main.py                        # Entry point
├── config.yaml                    # Configuration
│
├── core/                          # Generation logic
│   ├── comfyui_api.py             # API client
│   ├── workflow_builder.py        # Workflow generation
│   └── generator.py               # High-level interface
│
├── cache/                         # Image caching
│   ├── manager.py                 # Cache operations
│   └── aesthetic_matcher.py       # CLIP similarity
│
├── interpolation/                 # Latent operations
│   ├── spherical_lerp.py          # Interpolation
│   ├── latent_encoder.py          # VAE encode/decode
│   └── hybrid_generator.py        # Combined mode
│
└── utils/                         # Utilities
    ├── file_ops.py                # Atomic writes
    ├── prompt_manager.py          # Prompt rotation
    └── status_writer.py           # Status JSON
```

---

## 🎯 Design Philosophy

**Key Principles**:
1. **Separation of Concerns** - Each module has single responsibility
2. **Dependency Injection** - Config passed down, not global
3. **Error Isolation** - Failures don't crash system
4. **Testability** - Each component can be unit tested
5. **Extensibility** - Easy to add new features

---

## 📊 Component Architecture

```
Main Controller (main.py)
    │
    ├──> DreamGenerator (generator.py)
    │        ├──> ComfyUIClient (comfyui_api.py)
    │        ├──> WorkflowBuilder (workflow_builder.py)
    │        └──> CacheManager (cache/manager.py)
    │                 └──> AestheticMatcher (cache/aesthetic_matcher.py)
    │
    ├──> PromptManager (utils/prompt_manager.py)
    │
    └──> StatusWriter (utils/status_writer.py)
```

---

## 🔑 Key Design Patterns

| Pattern | Where Used | Why |
|---------|------------|-----|
| **Facade** | DreamGenerator | Simplify complex subsystem |
| **Builder** | WorkflowBuilder | Construct complex workflows |
| **Repository** | CacheManager | Abstract data storage |
| **Strategy** | AestheticMatcher | Pluggable algorithms |
| **Coordinator** | Main Controller | Orchestrate components |

---

## 📦 Module Details

Each module has its own detailed documentation:

1. **[CORE_MODULES.md](backend_architecture/CORE_MODULES.md)** - Generator, API, Workflow Builder
2. **[CACHE_SYSTEM.md](backend_architecture/CACHE_SYSTEM.md)** - Cache manager and aesthetic matching
3. **[INTERPOLATION.md](backend_architecture/INTERPOLATION.md)** - Latent space operations
4. **[UTILITIES.md](backend_architecture/UTILITIES.md)** - File ops, system monitoring
5. **[DATA_FLOW.md](backend_architecture/DATA_FLOW.md)** - How data moves through system
6. **[PATTERNS.md](backend_architecture/PATTERNS.md)** - Design patterns explained

---

## 🔄 Main Data Flow

```
Main Loop
  │
  ├──> Get prompt (PromptManager)
  │
  ├──> Generate image (DreamGenerator)
  │      ├──> Build workflow (WorkflowBuilder)
  │      ├──> Queue prompt (ComfyUIClient)
  │      ├──> Wait for completion (WebSocket)
  │      └──> Retrieve output
  │
  ├──> Encode embedding (AestheticMatcher)
  │
  ├──> Add to cache (CacheManager)
  │
  ├──> Check cache injection
  │
  └──> Write status (StatusWriter)
```

---

## 🚀 Extension Points

**Easy to Add**:
1. **New Generation Modes** - Add method to DreamGenerator
2. **Different Models** - Add new WorkflowBuilder subclass
3. **New Similarity Metrics** - Swap AestheticMatcher
4. **Additional Monitors** - Extend SystemMonitor
5. **Output Formats** - Modify file_ops

---

## 📝 Code Style

- **Type Hints**: All function signatures
- **Docstrings**: Google style
- **Logging**: Appropriate levels
- **Error Handling**: Specific exceptions
- **Private Methods**: Leading underscore

---

## Next Steps

For detailed implementation of each component, see the individual module documentation files.

---

**Total Lines**: ~3000 lines of Python
**Modules**: 15+ files
**Design**: Modular, testable, extensible

