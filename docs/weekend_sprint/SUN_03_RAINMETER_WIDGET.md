# ⏱️ Sunday Session 3: Rainmeter Widget

**Goal**: Beautiful display frontend with crossfades

**Duration**: 3 hours

---

## Overview

Create Rainmeter widget in `rainmeter/DreamWindow/`.

### Key Components

1. **DreamWindow.ini** - Main skin
2. **Variables.inc** - User config
3. **Asset images** (scanlines, glow)
4. **Crossfade animation**

### What It Does

- Displays current_frame.png
- Smooth crossfade transitions
- Status bar with stats
- Frame design with glows
- Optional effects (scanlines)

---

## Widget Structure

```ini
[Rainmeter]
Update=50
HardwareAcceleration=1

[MeasureImageUpdate]
; Watches for file changes
Plugin=FileView
Path=#ProjectPath#\output
File=current_frame.png
OnChangeAction=[!CommandMeasure MeasureCrossfade "Execute 1"]

[ImageMeterCurrent]
Meter=Image
ImageName=#ProjectPath#\output\current_frame.png
W=256
H=512
ImageAlpha=#CurrentAlpha#

[BorderGlow]
Meter=Shape
; Cyan border with glow
Shape=Rectangle ...
```

---

## Key Features

### Crossfade System

Uses ActionTimer to smoothly transition:
- Previous frame alpha: 255 → 0
- Current frame alpha: 0 → 255
- Duration: 750ms
- Steps: 75 (every 10ms)

### Status Display

Reads `output/status.json`:
- Frame count
- Generation time
- Cache size
- Status (LIVE/PAUSED)

---

## Installation

1. Create skin directory:
```powershell
mkdir "$env:USERPROFILE\Documents\Rainmeter\Skins\DreamWindow"
mkdir "$env:USERPROFILE\Documents\Rainmeter\Skins\DreamWindow\@Resources"
```

2. Copy files:
- DreamWindow.ini
- Variables.inc

3. Edit Variables.inc:
```ini
ProjectPath=C:\AI\DreamWindow
```

4. Load in Rainmeter:
- Right-click tray icon → Manage
- Find DreamWindow → Load

---

## Asset Generation (Optional)

**Scanlines texture**:
```python
from PIL import Image, ImageDraw
img = Image.new('RGBA', (256, 512), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
for y in range(0, 512, 2):
    draw.line([(0, y), (256, y)], fill=(255, 255, 255, 25))
img.save('scanlines.png')
```

---

## Validation

1. Widget appears on desktop
2. Image displays
3. Updates when new frame generated
4. Crossfade smooth
5. Status bar shows data

---

## Success Criteria

- [ ] Rainmeter widget created
- [ ] Images displaying
- [ ] Crossfades working
- [ ] Frame design beautiful
- [ ] Status bar functional

---

**Next**: SUN_04_FINAL_POLISH.md

---

**Session 7 of 8** | ~3 hours

