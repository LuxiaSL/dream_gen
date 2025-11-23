# Dream Window Rainmeter Skin

Sleek desktop widget for displaying AI-generated dream frames in real-time.

## 📋 Requirements

1. **Rainmeter** 4.5.x or later - https://www.rainmeter.net/

## 🚀 Quick Install

Run from project root:
```powershell
.\rainmeter_skin\install.ps1
```

Or manually:
1. Install Rainmeter from https://www.rainmeter.net/
2. Copy `rainmeter_skin/` to `%USERPROFILE%\Documents\Rainmeter\Skins\DreamWindow\`
3. Edit `@Resources\Variables.inc` to set your project path
4. Load in Rainmeter Manager

## 📊 Features

- Smooth crossfade transitions between frames
- Live status bar (frame count, gen time, cache, uptime)
- Pulsing cyan glow effect
- Configurable colors, position, scanlines

## 🎨 Customize

Edit `@Resources\Variables.inc`:
```ini
WindowX=50                          # Position
WindowY=300
ColorCyanPrimary=0,200,255,255      # Cyan border
ScanlinesEnabled=0                  # CRT effect
```

### 📐 Changing Display Resolution

To match different generation resolutions from `backend/config.yaml`:

1. **Edit backend config** (`backend/config.yaml`):
   ```yaml
   generation:
     resolution: [512, 256]  # Change to your desired width x height
   ```

2. **Update Rainmeter dimensions** (`@Resources/Variables.inc`):
   ```ini
   ViewportWidth=512      # Match width from backend config
   ViewportHeight=256     # Match height from backend config
   WidgetWidth=528        # ViewportWidth + 16
   WidgetHeight=352       # ViewportHeight + 96
   ```

3. **Refresh Rainmeter skin** (right-click skin → Refresh)

**Common resolutions:**
- `512x256` (default 2:1 cinematic)
- `512x512` (square 1:1)
- `768x512` (3:2 photo aspect)
- `1024x512` (2:1 ultra-wide)

## 🔧 Troubleshooting

**No image?** Check ProjectPath in Variables.inc, verify backend is running and `output/current_frame.png` exists.

**Status not updating?** Backend must be running to generate `status.json`.

**Loading overlay stuck?** Make sure daemon is running. Check `output/status.json` has `"is_buffering": false`.

---

Version 1.0.0 | Part of Dream Window project

