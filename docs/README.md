# 🌀 Dream Window - Documentation (OUT OF DATE)

**A living AI dream window for your desktop: continuously morphing ethereal technical aesthetics**

---

## What Is This?

Dream Window is a 256×512 pixel desktop widget that displays continuously morphing AI-generated imagery. It creates a "living dream window" using hybrid generation (latent interpolation + img2img + aesthetic cache injection), running on a dedicated GPU with zero gaming impact.

---

## 🚀 Quick Start Paths

### Path 1: First Time Here?
**Goal**: Understand the project vision

1. Read [DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md) (15 min overview)
2. Check [AESTHETIC_SPEC.md](AESTHETIC_SPEC.md) (visual design)
3. Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) (file organization)

### Path 2: Ready to Install?
**Goal**: Set up the environment

1. Start with [setup/01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md)
2. Follow steps 02 through 06 in order
3. Total time: ~2-3 hours

### Path 3: Ready to Build?
**Goal**: Implement the system

1. Review [weekend_sprint/00_OVERVIEW.md](weekend_sprint/00_OVERVIEW.md)
2. Follow SAT_01 through SUN_04 in order
3. Total time: ~15-20 hours

---

## 📚 Documentation Map

### 🎯 Core References (Start Here)
- **[DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md)** - Project vision, architecture, technical decisions
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete file organization reference
- **[AESTHETIC_SPEC.md](AESTHETIC_SPEC.md)** - Visual design specification
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet for daily operations
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Problem solving guide

### 🔧 Setup Guide (6 Parts)
**Location**: `setup/` | **Time**: 2-3 hours

Follow in order:
1. [01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md) - Python & GPU verification
2. [02_COMFYUI_INSTALL.md](setup/02_COMFYUI_INSTALL.md) - ComfyUI installation
3. [03_FLUX_MODEL.md](setup/03_FLUX_MODEL.md) - Flux model download
4. [04_FIRST_TEST.md](setup/04_FIRST_TEST.md) - First generation test
5. [05_PYTHON_ENVIRONMENT.md](setup/05_PYTHON_ENVIRONMENT.md) - Python dependencies
6. [06_PROJECT_STRUCTURE.md](setup/06_PROJECT_STRUCTURE.md) - Project finalization

### ⏱️ Weekend Sprint (8 Parts)
**Location**: `weekend_sprint/` | **Time**: 15-20 hours

**Saturday - Backend**:
1. [SAT_01_API_CLIENT.md](weekend_sprint/SAT_01_API_CLIENT.md) - ComfyUI API wrapper
2. [SAT_02_WORKFLOW_BUILDER.md](weekend_sprint/SAT_02_WORKFLOW_BUILDER.md) - Workflow generation
3. [SAT_03_GENERATOR.md](weekend_sprint/SAT_03_GENERATOR.md) - Generator class
4. [SAT_04_MAIN_LOOP.md](weekend_sprint/SAT_04_MAIN_LOOP.md) - Main generation loop

**Sunday - Integration**:
5. [SUN_01_CACHE_SYSTEM.md](weekend_sprint/SUN_01_CACHE_SYSTEM.md) - Cache manager
6. [SUN_02_AESTHETIC_MATCHING.md](weekend_sprint/SUN_02_AESTHETIC_MATCHING.md) - CLIP similarity
7. [SUN_03_RAINMETER_WIDGET.md](weekend_sprint/SUN_03_RAINMETER_WIDGET.md) - Display widget
8. [SUN_04_FINAL_POLISH.md](weekend_sprint/SUN_04_FINAL_POLISH.md) - Testing & polish

### 🏗️ Backend Architecture (4 Parts)
**Location**: `backend_architecture/` | **When**: Understanding code

1. [00_OVERVIEW.md](backend_architecture/00_OVERVIEW.md) - System architecture
2. [CORE_MODULES.md](backend_architecture/CORE_MODULES.md) - Generator, API, Workflow
3. [CACHE_SYSTEM.md](backend_architecture/CACHE_SYSTEM.md) - Cache & aesthetic matching
4. [UTILITIES.md](backend_architecture/UTILITIES.md) - File ops, system monitoring

### 🖥️ Rainmeter Widget (3 Parts)
**Location**: `rainmeter_widget/` | **When**: Frontend setup

1. [00_OVERVIEW.md](rainmeter_widget/00_OVERVIEW.md) - Widget overview
2. [SETUP.md](rainmeter_widget/SETUP.md) - Installation
3. [CUSTOMIZATION.md](rainmeter_widget/CUSTOMIZATION.md) - Colors, position, effects

---

## 🔍 Find What You Need

| I want to... | Go to... |
|--------------|----------|
| Understand the vision | [DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md) |
| See file organization | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Install everything | [setup/01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md) |
| Build the backend | [weekend_sprint/SAT_01_API_CLIENT.md](weekend_sprint/SAT_01_API_CLIENT.md) |
| Understand the code | [backend_architecture/00_OVERVIEW.md](backend_architecture/00_OVERVIEW.md) |
| Customize the widget | [rainmeter_widget/CUSTOMIZATION.md](rainmeter_widget/CUSTOMIZATION.md) |
| Fix a problem | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Find a command | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |

---

## 🎯 Recommended Reading Order

### For New Users (Understanding)
```
README.md (you are here)
    ↓
DREAM_WINDOW_MASTER.md (vision & architecture)
    ↓
PROJECT_STRUCTURE.md (file organization)
    ↓
AESTHETIC_SPEC.md (visual design)
```

### For Implementation (Building)
```
setup/01_ENVIRONMENT_SETUP.md
    ↓
... follow setup/ in order ...
    ↓
weekend_sprint/00_OVERVIEW.md
    ↓
... follow weekend_sprint/ in order ...
    ↓
MVP Complete! 🎉
```

### For Maintenance (Operating)
```
QUICK_REFERENCE.md (daily commands)
TROUBLESHOOTING.md (when things break)
```

---

## 📊 Documentation Stats

- **Total Files**: 30 organized documents
- **Average File**: ~250 lines (bite-sized)
- **Organization**: Modular by function
- **Best For**: Both humans and AI agents

---

## 💡 How to Use This Documentation

**As a Human**:
1. Start with overview documents
2. Follow step-by-step guides linearly
3. Reference specific docs as needed
4. Bookmark QUICK_REFERENCE.md

**As an AI Agent**:
1. Load README.md first for navigation
2. Load specific modules for tasks
3. Avoid loading entire documentation set
4. Use cross-references for details

---

## 🌟 Ready to Begin?

Choose your path:
- **New to project?** → [DREAM_WINDOW_MASTER.md](DREAM_WINDOW_MASTER.md)
- **Ready to install?** → [setup/01_ENVIRONMENT_SETUP.md](setup/01_ENVIRONMENT_SETUP.md)
- **Ready to code?** → [weekend_sprint/00_OVERVIEW.md](weekend_sprint/00_OVERVIEW.md)
- **Need quick help?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) or [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Let's make this dream window real.** 🌀✨
