# 🖥️ Rainmeter Widget Overview

**Building the Dream Window display frontend**

---

## 📁 File Structure

```
Rainmeter/Skins/DreamWindow/
├── DreamWindow.ini              # Main widget skin
├── Variables.inc                # User configuration
│
└── @Resources/
    ├── Images/
    │   ├── scanlines.png        # CRT effect (optional)
    │   └── glow_overlay.png     # Pulsing glow (optional)
    │
    └── Scripts/
        └── (optional Lua scripts)
```

---

## 🎯 Widget Components

### 1. **Image Display**
- Shows `output/current_frame.png`
- 256×512 pixel viewport
- Crossfade transitions between frames

### 2. **Frame Design**
- Cyan wireframe border
- Corner brackets (L-shaped accents)
- Pulsing glow when generating
- Optional scanline overlay

### 3. **Status Bar**
- Generation count
- Generation time
- Status (LIVE/PAUSED)
- Cache size

### 4. **Animations**
- 750ms crossfade transitions
- 3-second glow pulse cycle
- Smooth alpha blending

---

## 📊 Key Features

### Crossfade System

Uses ActionTimer plugin:
- Previous frame: Alpha 255 → 0 (fade out)
- Current frame: Alpha 0 → 255 (fade in)
- Duration: 750ms
- Steps: 75 (10ms each)
- Result: Smooth, flicker-free transitions

### File Monitoring

FileView plugin watches for changes:
- Path: `#ProjectPath#\output`
- File: `current_frame.png`
- On change → Trigger crossfade

### Status Parsing

WebParser plugin reads `status.json`:
- Frame count
- Generation time
- Status text
- Cache size
- Updates every 1 second

---

## 🎨 Visual Design

### Color Scheme
- **Background**: Dark gray (#1A1A1A) at 85% opacity
- **Border**: Cyan (#00C8FF) primary
- **Accents**: Cyan secondary (#4A90E2)
- **Text**: Gray (#808080)

### Dimensions
- **Widget Total**: 272 × 584 pixels
- **Border Width**: 6 pixels
- **Header**: 24 pixels tall
- **Viewport**: 256 × 512 pixels
- **Footer**: 20 pixels tall

---

## 📦 Module Documentation

Detailed guides for each aspect:

1. **[SETUP.md](rainmeter_widget/SETUP.md)** - Installation and configuration
2. **[MAIN_SKIN.md](rainmeter_widget/MAIN_SKIN.md)** - DreamWindow.ini structure
3. **[ANIMATIONS.md](rainmeter_widget/ANIMATIONS.md)** - Crossfade and effects
4. **[CUSTOMIZATION.md](rainmeter_widget/CUSTOMIZATION.md)** - Colors, position, effects
5. **[TROUBLESHOOTING.md](rainmeter_widget/TROUBLESHOOTING.md)** - Common issues

---

## 🚀 Quick Start

1. Install Rainmeter
2. Create skin directory
3. Copy DreamWindow.ini and Variables.inc
4. Edit ProjectPath in Variables.inc
5. Load skin in Rainmeter

**See [SETUP.md](rainmeter_widget/SETUP.md) for detailed steps**

---

## ⚙️ Configuration

Edit `Variables.inc` to customize:
- Widget position (X, Y)
- Colors (RGB values)
- Animation speed
- Effects (scanlines, glow)
- Update rates

**See [CUSTOMIZATION.md](rainmeter_widget/CUSTOMIZATION.md) for all options**

---

## 📊 Performance

**CPU Usage**: < 2% typical
**Memory**: ~50MB
**GPU**: Uses hardware acceleration

**Optimal for**:
- 24/7 operation
- Multiple monitors
- Low system impact

---

**Next**: See individual module docs for implementation details

