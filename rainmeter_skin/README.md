# Dream Window Rainmeter Skin

Sleek desktop widget for displaying AI-generated dream frames in real-time.

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

## 📖 Full Documentation

See `../docs/rainmeter_widget/` for detailed guides.

## 🔧 Troubleshooting

**No image?** Check ProjectPath in Variables.inc, verify backend is running and `output/current_frame.png` exists.

**Status not updating?** Backend must be running to generate `status.json`.

---

Version 1.0.0 | Part of Dream Window project

