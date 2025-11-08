# 🖥️ Rainmeter Setup Guide

**Installing and configuring the Dream Window widget**

---

## Step 1: Install Rainmeter

Download from: https://www.rainmeter.net/

Run installer, choose default options.

---

## Step 2: Create Skin Directory

```powershell
# Navigate to Rainmeter skins
cd "$env:USERPROFILE\Documents\Rainmeter\Skins"

# Create DreamWindow folder
mkdir DreamWindow
mkdir DreamWindow\@Resources
mkdir DreamWindow\@Resources\Images
```

---

## Step 3: Copy Files

Copy the following files to your DreamWindow directory:

**Required**:
- `DreamWindow.ini` → `DreamWindow\`
- `Variables.inc` → `DreamWindow\@Resources\`

**Optional assets**:
- `scanlines.png` → `DreamWindow\@Resources\Images\`
- `glow_overlay.png` → `DreamWindow\@Resources\Images\`

---

## Step 4: Configure Path

Edit `DreamWindow\@Resources\Variables.inc`:

```ini
[Variables]
ProjectPath=C:\AI\DreamWindow
; ^ Change this to YOUR actual project path
```

**Critical**: Use absolute path, not relative. Use backslashes `\` not forward slashes `/`.

---

## Step 5: Load Skin

1. Right-click Rainmeter tray icon
2. Click "Manage"
3. In left panel, find "DreamWindow"
4. Click "Load" button
5. Widget should appear on desktop

---

## Step 6: Position Widget

### Manually
Drag widget to desired location.

### Via Configuration
Edit `Variables.inc`:
```ini
WindowX=50      # Pixels from left edge
WindowY=300     # Pixels from top
```

---

## Step 7: Verify

**Check that**:
- [ ] Widget visible on desktop
- [ ] Displays image (if Python controller running)
- [ ] Status bar shows frame count
- [ ] Crossfade works on new frames

---

## Troubleshooting

### Widget not appearing
- Check Rainmeter is running (tray icon)
- Verify skin is loaded (Manage → DreamWindow)
- Check error logs (Manage → Log tab)

### Image not displaying
- Verify ProjectPath is correct
- Check `output/current_frame.png` exists
- Right-click widget → Refresh Skin

### Permission errors
- Run Rainmeter as Administrator (once)
- Check file permissions on project folder

---

## Optional: Generate Asset Images

### Scanlines Texture

Create `generate_scanlines.py`:

```python
from PIL import Image, ImageDraw

width, height = 256, 512
img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

for y in range(0, height, 2):
    draw.line([(0, y), (width, y)], fill=(255, 255, 255, 25))

img.save('scanlines.png')
print("✓ Created scanlines.png")
```

Run:
```powershell
python generate_scanlines.py
copy scanlines.png "$env:USERPROFILE\Documents\Rainmeter\Skins\DreamWindow\@Resources\Images\"
```

---

## Configuration Options

See **[CUSTOMIZATION.md](CUSTOMIZATION.md)** for:
- Changing colors
- Adjusting position
- Enabling/disabling effects
- Performance tuning

---

## Next Steps

- **[MAIN_SKIN.md](MAIN_SKIN.md)** - Understanding the INI structure
- **[ANIMATIONS.md](ANIMATIONS.md)** - How crossfades work
- **[CUSTOMIZATION.md](CUSTOMIZATION.md)** - Personalizing the widget

---

**Setup Complete!** Widget should now be displaying your Dream Window 🌀✨

