[Variables]

; ========================================
; PROJECT PATH (REQUIRED - EDIT THIS!)
; ========================================
; Set this to YOUR actual Dream Window project directory
; Use absolute path with backslashes, no trailing backslash
; 
; NOTE: If you ran install.ps1, this was auto-configured!
; If you moved the project, run install.ps1 again or update manually
ProjectPath=C:\Users\luxia\Documents\projects\dream_gen

; ========================================
; WIDGET POSITION
; ========================================
WindowX=50
WindowY=300

; ========================================
; DIMENSIONS - MATCH YOUR BACKEND CONFIG!
; ========================================
; IMPORTANT: These dimensions should match your backend/config.yaml
; The generation resolution in config.yaml is [width, height]
; Set ViewportWidth and ViewportHeight to match your generation resolution
;
; Current backend config: resolution: [512, 256]
; 
; Common resolutions:
;   512x256 (default 2:1 aspect ratio)
;   512x512 (square)
;   768x512 (3:2 wide)
;   1024x512 (2:1 ultra-wide)
;
; After changing these values, adjust WidgetWidth and WidgetHeight:
;   WidgetWidth = ViewportWidth + 16 (padding)
;   WidgetHeight = ViewportHeight + 96 (header + status bar)
;
ViewportWidth=512
ViewportHeight=256
WidgetWidth=528
WidgetHeight=352
BorderWidth=6

; ========================================
; COLORS (RGBA)
; ========================================
ColorCyanPrimary=0,200,255,255
ColorCyanSecondary=74,144,226,153
ColorBgDark=26,26,26,217
ColorTextGray=128,128,128,255

; ========================================
; ANIMATION SETTINGS
; ========================================
CrossfadeDuration=750
UpdateRate=50
PulseSpeed=3

; ========================================
; EFFECTS
; ========================================
ScanlinesEnabled=0
ScanlinesAlpha=15

