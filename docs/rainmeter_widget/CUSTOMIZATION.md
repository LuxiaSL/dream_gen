# 🖥️ Customization Guide

**Personalizing your Dream Window widget**

---

## 🎨 Changing Colors

Edit `@Resources/Variables.inc`:

### Cyan (Default)
```ini
ColorCyanPrimary=0,200,255,255
ColorCyanSecondary=74,144,226,153
```

### Matrix Green
```ini
ColorCyanPrimary=0,255,65,255
ColorCyanSecondary=0,170,0,153
```

### Warm Red
```ini
ColorCyanPrimary=255,0,64,255
ColorCyanSecondary=200,0,50,153
```

### Pure White
```ini
ColorCyanPrimary=255,255,255,255
ColorCyanSecondary=204,204,204,153
```

### Custom
Use any RGB values:
```ini
ColorCyanPrimary=R,G,B,255
; Example: 100,200,150,255
```

---

## 📐 Changing Position

### Desktop Position
```ini
WindowX=50      # Pixels from left
WindowY=300     # Pixels from top
```

### Common Positions

**Left side:**
```ini
WindowX=50
WindowY=300
```

**Right side (1920px monitor):**
```ini
WindowX=1600
WindowY=400
```

**Center:**
```ini
WindowX=824     # (1920-272)/2
WindowY=248     # (1080-584)/2
```

### Multi-Monitor

**Secondary monitor (1920px primary):**
```ini
WindowX=1970    # 1920 + 50
WindowY=300
```

---

## 🎭 Effects Control

### Scanlines

**Enable:**
```ini
ScanlinesEnabled=1
ScanlinesAlpha=15    # Opacity (0-255)
```

**Disable:**
```ini
ScanlinesEnabled=0
```

### Pulsing Glow

**Adjust speed:**
```ini
PulseSpeed=6         # Default (3 sec cycle)
PulseSpeed=12        # Faster (1.5 sec cycle)
PulseSpeed=3         # Slower (6 sec cycle)
```

**Disable:**
Comment out `[InnerGlowDynamic]` meter in DreamWindow.ini

---

## ⚡ Animation Settings

### Crossfade Duration

```ini
CrossfadeDuration=750    # Default (0.75 seconds)
CrossfadeDuration=1000   # Slower (1 second)
CrossfadeDuration=500    # Faster (0.5 seconds)
```

### Update Rate

**For smoothness:**
```ini
UpdateRate=50      # 20 FPS (default)
UpdateRate=33      # 30 FPS
UpdateRate=16      # 60 FPS (very smooth)
```

**For performance:**
```ini
UpdateRate=100     # 10 FPS (lower CPU)
```

---

## 🖼️ Size Adjustments

### Make Larger

```ini
ViewportWidth=384       # Was 256 (1.5x)
ViewportHeight=768      # Was 512 (1.5x)

; Adjust totals
WidgetWidth=408         # ViewportWidth + borders
WidgetHeight=850        # ViewportHeight + header + footer
```

### Make Smaller

```ini
ViewportWidth=192       # Was 256 (0.75x)
ViewportHeight=384      # Was 512 (0.75x)

WidgetWidth=216
WidgetHeight=466
```

**Note**: Python backend still generates at 256×512. Rainmeter scales display.

---

## 🔧 Behavior Options

### Click-Through Mode

Widget transparent to clicks:
```ini
ClickThrough=1
```

### Always On Top

Widget stays above all windows:
```ini
AlwaysOnTop=1
```

### Stay On Desktop

Widget pinned to desktop layer:
```ini
StayOnDesktop=1
```

### Draggable

Lock position (can't drag):
```ini
Draggable=0
```

---

## 📊 Performance Tuning

### Low CPU Mode

```ini
[Rainmeter]
Update=100           # Less frequent updates

[Variables]
UpdateRate=100
ScanlinesEnabled=0   # Disable effects
HWAcceleration=1     # Ensure enabled

[MeasureStatusFile]
UpdateRate=40        # Check status less often
```

### High Quality Mode

```ini
[Rainmeter]
Update=16            # 60 FPS
HardwareAcceleration=1
AntiAlias=1

[Variables]
UpdateRate=16
ScanlinesEnabled=1
```

---

## 🎨 Custom Border Styles

### Thicker Border

```ini
BorderWidth=10       # Was 6
```

### No Border

Set border shapes alpha to 0:
```ini
; In DreamWindow.ini
[BorderInner]
; Comment out or set alpha to 0
```

### Different Corner Style

Edit corner shapes in DreamWindow.ini:
```ini
[CornerTopLeft]
; Modify line lengths
Shape=Line 0,0,0,24  | StrokeWidth 3 | ...
Shape2=Line 0,0,24,0 | StrokeWidth 3 | ...
```

---

## 🌈 Theme Presets

### Cyberpunk
```ini
ColorCyanPrimary=255,0,255,255      # Magenta
ColorBgDark=10,0,20,230             # Purple-ish dark
```

### Hacker Green
```ini
ColorCyanPrimary=0,255,0,255        # Bright green
ColorBgDark=0,20,0,230              # Dark green tint
```

### Ice Blue
```ini
ColorCyanPrimary=173,216,230,255    # Light blue
ColorCyanSecondary=135,206,250,153  # Sky blue
```

### Blood Red
```ini
ColorCyanPrimary=255,0,0,255        # Bright red
ColorCyanSecondary=139,0,0,153      # Dark red
```

---

## 🔄 Reset to Defaults

If you break something:

1. Delete `Variables.inc`
2. Copy original from documentation
3. Re-edit ProjectPath
4. Refresh skin

---

## 💾 Save Your Customizations

Before experimenting:
```powershell
copy Variables.inc Variables.inc.backup
```

Restore:
```powershell
copy Variables.inc.backup Variables.inc
```

---

## 🎯 Quick Customization Checklist

Common customizations:
- [ ] Set project path
- [ ] Choose position (X, Y)
- [ ] Pick color theme
- [ ] Enable/disable scanlines
- [ ] Adjust animation speed
- [ ] Set behavior (click-through, always on top)
- [ ] Test performance
- [ ] Backup working config

---

**Enjoy your personalized Dream Window!** 🎨✨

