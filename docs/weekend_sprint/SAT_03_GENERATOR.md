# ⏱️ Saturday Session 3: Generator Class

**Goal**: High-level generation interface

**Duration**: 2 hours

---

## Overview

Create `backend/core/generator.py` - the main interface for image generation.

### Key Components

1. **DreamGenerator class**
2. `generate_from_prompt()` - txt2img
3. `generate_from_image()` - img2img (primary method)
4. File management
5. Error handling

### What It Does

- Orchestrates API client + workflow builder
- Handles image file operations
- Copies generated images to output directory
- Tracks performance metrics

---

## Core Methods

### `generate_from_image()`

Main method for morphing:
1. Build img2img workflow
2. Queue prompt via API
3. Wait for completion
4. Retrieve output
5. Copy to output directory

---

## Implementation Notes

- Use async/await for WebSocket
- Implement retry logic
- Add performance logging
- Atomic file operations

**Full code**: See original WEEKEND_SPRINT.md lines 454-700

---

## Validation

```powershell
cd backend
python -c "from core.generator import DreamGenerator; import yaml; config = yaml.safe_load(open('config.yaml')); g = DreamGenerator(config); print('✓ Generator initialized')"
```

---

## Success Criteria

- [ ] Generator class created
- [ ] Can generate from seed image
- [ ] Images copied to output/
- [ ] Generation time < 3 seconds

---

**Next**: SAT_04_MAIN_LOOP.md

---

**Session 3 of 8** | ~2 hours

