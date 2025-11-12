# 🎨 AESTHETIC SPECIFICATION - Dream Window

**Visual Design, Frame Mockups, and Prompt Engineering**

Based on analysis of relevant seed images

---

## 🎨 Color Palette (Exact Specifications)

### Monochrome Foundation
```
Pure Black:    #000000
Dark Gray:     #1A1A1A  (frame background)
Medium Gray:   #808080  (overlays)
Light Gray:    #CCCCCC  (highlights)
Pure White:    #FFFFFF  (bright elements)
```

### Accent Colors (Surgical Use)
```
Cyan Primary:    #00C8FF  (main technical accent)
Cyan Secondary:  #4A90E2  (softer highlights)
Red Primary:     #FF0040  (energy accents)
Red Crimson:     #8B0000  (deep reds)
```

---

## 🖼️ Frame Design: "Holographic Data Window"

### Dimensions
- **Total Widget**: 272 × 584 pixels
- **Border Width**: 6 pixels
- **Header**: 24 pixels tall
- **Footer**: 20 pixels tall
- **Viewport**: 256 × 512 pixels (image display)

### Visual Structure

```
┌──┬────────────────────────────────────┬──┐
│  │ ◆ DREAM.WINDOW              [◉] │  │  Header (24px)
└──┴────────────────────────────────────┴──┘
┌──┐                                  ┌──┐
│  │                                  │  │
│  │                                  │  │
│  │      [256×512 VIEWPORT]         │  │  Viewport (512px)
│  │      Morphing AI Images         │  │
│  │                                  │  │
│  │                                  │  │
└──┘                                  └──┘
┌──┬────────────────────────────────────┬──┐
│  │ ▸ GEN:234 ⟲ 1.8s ◉ LIVE        │  │  Footer (20px)
└──┴────────────────────────────────────┴──┘
```

### Frame Components

**1. Border**:
- Base: Dark gray (#1A1A1A) at 85% opacity
- Inner stroke: 1px cyan (#00C8FF) at 60% opacity
- Outer glow: 2px cyan blur at 30% opacity

**2. Corner Accents**:
- L-shaped brackets, 16×16px each corner
- 2px line width, cyan (#00C8FF) at 80% opacity

**3. Header Bar**:
- Background: Dark gray at 90% opacity
- Left: Diamond icon (◆) + "DREAM.WINDOW"
- Right: Status dot (◉ when active, ○ when paused)
- Font: Consolas 9pt, cyan color

**4. Footer Bar** (optional):
- Frame count + generation time + status
- Font: Consolas 7pt, medium gray (#808080)

**5. Inner Glow** (dynamic):
- Pulses during generation (20% → 60% → 20%)
- Cyan secondary color
- Fades to 0% when idle

---

## 🎭 Prompt Engineering

### Base Templates (Cycle Through These)

**Template 1: Ethereal Dissolution**
```
"ethereal digital angel, dissolving into particles, flowing white lines,
technical wireframe overlay, monochrome with cyan accents,
architectural diagrams, high contrast, abstract geometry"
```

**Template 2: Technical Architecture**
```
"abstract geometry, technical wireframe, architectural diagrams,
flowing lines, blueprint aesthetic, monochrome with data corruption,
cyberpunk overlay, grid patterns, high contrast"
```

**Template 3: Glitch Angel**
```
"cyberpunk angel, glitch art aesthetic, digital corruption,
technical overlay, particle dissolution, monochrome with red and cyan accents,
wireframe structure, abstract form, high contrast"
```

**Template 4: Data Stream**
```
"ethereal figure in data stream, technical readouts, flowing particles,
architectural wireframe, monochrome with blue highlights,
digital dissolution, abstract geometry, high contrast"
```

### Negative Prompt (Always Use)
```
"photorealistic, photograph, 3d render, realistic photo, blurry,
low quality, text, watermark, signature, jpeg artifacts,
low contrast, muddy colors, brown tones, warm colors"
```

### Rotation Strategy
- Every 20 frames: Switch to next template
- Random seed each frame
- Cache injection every 15 frames (aesthetic matching)

---

## ⏱️ Animation Timing

### Frame Cycle (4 seconds total)
```
0.0s │ Generation starts, glow pulses on
2.0s │ Generation complete
2.0s │ Crossfade begins (previous → new)
3.5s │ Crossfade complete
4.0s │ Next generation cycle starts
```

### Crossfade Curve
- Duration: 1.5 seconds
- Easing: Ease-in-out
- Previous alpha: 255 → 0
- New alpha: 0 → 255

---

## 🎨 Visual Quality Targets

**Must maintain**:
- High contrast (extreme blacks/whites)
- Monochrome base with sparse color accents
- Technical wireframe elements
- Particle dissolution effects
- Ethereal + technical fusion aesthetic

**Avoid**:
- Warm tones (browns, yellows)
- Photorealism
- Low contrast
- Muddy colors
- Text/watermarks

---

## 🔍 Quality Validation

**After first generation, check**:
- [ ] Maintains high contrast
- [ ] Monochrome with color accents present
- [ ] Technical elements visible
- [ ] Particle effects apparent
- [ ] Matches source aesthetic

**If not matching**:
- Increase prompt weight on key terms
- Reduce denoise strength (0.3 instead of 0.4)
- Add more negative prompts
- Inject seed images more frequently

---

## 💾 Required Assets

**Will need to create**:
1. `border_frame.png` - Frame with transparent center
2. `scanlines.png` - Horizontal line texture (optional)
3. `glow_overlay.png` - Glow texture for animation

Can generate these programmatically or in image editor.

---

## ✅ Design Complete Checklist

- [ ] Frame design matches specification
- [ ] Colors are exact per hex codes
- [ ] Animations are smooth and subtle
- [ ] Generated images match seed aesthetic
- [ ] Widget integrates naturally with desktop
- [ ] Status info is clear and useful
- [ ] Overall effect is "living dream window"
