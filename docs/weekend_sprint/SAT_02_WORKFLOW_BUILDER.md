# ⏱️ Saturday Session 2: Workflow Builder

**Goal**: Dynamic ComfyUI workflow generation

**Duration**: 2 hours

---

## Overview

Create `backend/core/workflow_builder.py` that builds ComfyUI workflow JSON structures dynamically.

### Key Components

1. **FluxWorkflowBuilder class**
2. `build_txt2img()` method
3. `build_img2img()` method (primary for morphing)

### What It Does

- Generates ComfyUI node graphs programmatically
- Configures nodes for Flux model
- Sets up sampling parameters
- Handles seed randomization

---

## Implementation

Create the file with:
- Checkpoint loader nodes
- CLIP text encode nodes
- KSampler configuration
- VAE encode/decode
- Image save nodes

**Full code**: See original WEEKEND_SPRINT.md lines 222-448

---

## Validation

```powershell
python backend\core\workflow_builder.py
```

Should create:
- `comfyui_workflows/flux_txt2img.json`
- `comfyui_workflows/flux_img2img.json`

---

## Success Criteria

- [ ] Workflow builder created
- [ ] JSON files generated
- [ ] Valid ComfyUI workflow format

---

**Next**: SAT_03_GENERATOR.md

---

**Session 2 of 8** | ~2 hours

