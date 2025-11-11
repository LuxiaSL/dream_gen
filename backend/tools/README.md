# Dream Window Testing & Analysis Tools

This directory contains various testing, profiling, and analysis scripts used during development and optimization of the Dream Window interpolation system.

---

## 📊 Analysis Scripts

### `analyze_performance_correlations.py`

**Purpose**: Statistical analysis of interpolation performance correlations

Analyzes performance metrics JSON to find correlations between:
- Upscale methods (bilinear, bicubic, nearest)
- Downsample methods (bilinear, bicubic, lanczos)
- Divisor values (1, 2, 3, 4, 8)
- Performance metrics (FPS, quality, etc.)

**Usage**:
```bash
# Analyze most recent test run
uv run backend/tools/analyze_performance_correlations.py

# Analyze specific metrics file
uv run backend/tools/analyze_performance_correlations.py path/to/performance_metrics.json

# Save report to specific file
uv run backend/tools/analyze_performance_correlations.py --output my_report.txt
```

**Features**:
- Variance analysis (what factors matter most?)
- Stability analysis (which combinations are most consistent?)
- Linear regression (quantify method contributions)
- ANOVA (statistical significance testing)
- Variance decomposition
- Method ranking by divisor
- Combination pattern analysis

**Output**: Comprehensive text report with correlation analysis

---

### `analyze_within_divisor.py`

**Purpose**: Within-divisor performance analysis with visualizations

Controls for divisor variance by analyzing each divisor independently to determine what factors actually matter when divisor is held constant.

**Usage**:
```bash
# Analyze most recent test run
uv run backend/tools/analyze_within_divisor.py

# Analyze specific metrics file
uv run backend/tools/analyze_within_divisor.py path/to/performance_metrics.json

# Combine and analyze ALL test runs
uv run backend/tools/analyze_within_divisor.py --all-runs

# Skip visualization generation
uv run backend/tools/analyze_within_divisor.py --no-viz
```

**Features**:
- Per-divisor method ranking
- Statistical significance testing (ANOVA)
- Cross-divisor summary
- Best combination identification
- Matplotlib visualizations:
  - Performance by method across divisors
  - FPS vs Quality scatter plots
  - Stability heatmaps
  - Per-divisor detailed comparisons

**Output**: 
- Text report: `within_divisor_analysis.txt`
- Visualizations: `performance_analysis.png`, `per_divisor_analysis.png`

---

### `generate_comparison.py`

**Purpose**: Enhanced HTML comparison generator with clean template architecture

Generates interactive HTML pages for visual comparison of interpolation quality tests with charts and animations.

**Usage**:
```bash
# Generate for most recent test
uv run backend/tools/generate_comparison.py output/quality_tests/run_TIMESTAMP

# Generate with custom output path
uv run backend/tools/generate_comparison.py output/quality_tests/run_TIMESTAMP --output my_comparison.html
```

**Features**:
- Interactive tabbed interface
- Real-time animation playback at actual FPS
- Performance vs quality scatter plots
- FPS and quality bar charts
- Visual frame comparison with zoom
- Frame-by-frame navigation
- Divisor filtering
- Responsive design
- Chart.js integration

**Output**: `comparison_enhanced.html` (single-file, self-contained)

---

## 🎬 Animation & Visualization

### `generate_animation.py`

**Purpose**: Generate looping animation from keyframes and interpolations

Creates a WebP or GIF animation from the generated keyframes and interpolation frames. Automatically sequences frames as: keyframe → interpolations → keyframe → interpolations → ... creating a smooth looping animation.

**Usage**:
```bash
# Generate WebP at 5 FPS (default)
uv run backend/tools/generate_animation.py

# Generate GIF at 4 FPS
uv run backend/tools/generate_animation.py --format gif --fps 4

# Custom output path
uv run backend/tools/generate_animation.py --output my_animation.webp

# Use custom input directory
uv run backend/tools/generate_animation.py --input-dir path/to/output

# Disable looping (play once)
uv run backend/tools/generate_animation.py --no-loop

# Fast mode (faster encoding, slightly larger file)
uv run backend/tools/generate_animation.py --fast
```

**Features**:
- Automatically discovers and sequences keyframes + interpolations
- Parses filenames to determine proper frame order
- **Automatic frame resizing** to match config resolution (handles mixed sizes)
- **Parallel processing** for fast image loading (up to 8 threads)
- **Smart resampling** (Lanczos for large downscaling, Bilinear for minor adjustments)
- **Progress indication** during processing
- Loads target resolution from `backend/config.yaml`
- Manual resolution override support
- Supports WebP (better compression) and GIF formats
- Configurable frame rate (1-60 FPS)
- Looping or single-play animations
- Fast mode option for quicker encoding
- Shows animation statistics (duration, file size, frame count)
- Reports size mismatches with warnings

**Frame Sequencing**:
- Keyframes: `keyframe_001.png`, `keyframe_002.png`, etc.
- Interpolations: `001-002_001.png` through `001-002_010.png` (between keyframes 1 and 2)
- Result: `[keyframe_1, interp_1, interp_2, ..., interp_10, keyframe_2, interp_1, ..., keyframe_3, ...]`

**Output**: `output/finished_product.webp` (or `.gif`)

---

## 🧪 Testing Scripts

### `test_quality_comparison.py`

**Purpose**: Comprehensive quality comparison test suite

Tests all combinations of resolution divisors, upscale methods, and downsample methods, generating visual outputs and performance metrics.

**Usage**:
```bash
# Basic test (20 frames per config)
uv run backend/tools/test_quality_comparison.py

# Custom frame count
uv run backend/tools/test_quality_comparison.py --frames 50

# Include fractional divisors (1.5, 2.5, etc.)
uv run backend/tools/test_quality_comparison.py --fractional

# Exclude divisor 8 (focus on div1-4)
uv run backend/tools/test_quality_comparison.py --exclude-div8

# Test specific configs only
uv run backend/tools/test_quality_comparison.py --configs-only div2_bicubicdown_bilinearup div4_bilineardown_bilinearup

# Use custom images
uv run backend/tools/test_quality_comparison.py --image-a path/to/img1.png --image-b path/to/img2.png

# Long-run profiling (repeat test N times)
uv run backend/tools/test_quality_comparison.py --sequences 10
```

**Test Matrix**:
- **Divisors**: 1 (baseline), 2, 3, 4, 8 (optional: 1.5, 2.5, 3.5)
- **Upscale methods**: bilinear, bicubic, nearest
- **Downsample methods**: bilinear, bicubic, lanczos
- **Default configs**: 37 combinations

**Metrics Calculated**:
- FPS (frames per second)
- Frame generation time breakdown
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Quality vs baseline comparison

**Output**:
- `output/quality_tests/run_TIMESTAMP/` directory
- Individual frame sequences per config
- `performance_metrics.json` with complete results
- Auto-generated HTML comparison page

---

### `test_interpolation_speed.py`

**Purpose**: Quick interpolation speed test

Lightweight script to quickly verify optimizations and measure single-cycle performance.

**Usage**:
```bash
# Test at full resolution
uv run backend/tools/test_interpolation_speed.py

# Test with resolution divisor
uv run backend/tools/test_interpolation_speed.py --resolution-divisor 4

# Test with specific upscale method
uv run backend/tools/test_interpolation_speed.py --resolution-divisor 4 --upscale-method bicubic
```

**Tests**:
1. Full interpolation pipeline (7 frames)
2. Pure interpolation speed (decode only)
3. Component breakdown (encode, slerp, decode, save)

**Output**: Console timing report with FPS estimates

---

## 🔬 Profiling Scripts

### `profile_interpolation.py`

**Purpose**: Detailed VAE interpolation performance profiling

Profiles each component of the interpolation pipeline to identify bottlenecks and measure optimization improvements.

**Usage**:
```bash
# Basic profiling (10 iterations per component)
uv run backend/tools/profile_interpolation.py

# Custom iteration count
uv run backend/tools/profile_interpolation.py --iterations 20

# Use custom images
uv run backend/tools/profile_interpolation.py --image-a img1.png --image-b img2.png
```

**Profiles**:
- VAE encode (image → latent)
  - Image preprocessing
  - VAE encoding
- VAE decode (latent → image)
  - VAE decoding
  - Image postprocessing
- Spherical interpolation (slerp)
- Image save operations
- Full pipeline (end-to-end)
- GPU utilization check

**Features**:
- GPU-synchronized timing
- Statistical metrics (avg, std, min, max)
- Memory usage tracking
- Performance breakdown by percentage
- FPS calculation and target comparison

**Output**: Detailed console report with timing breakdowns

---

## 📝 Usage Patterns

### Quick Performance Check
```bash
uv run backend/tools/test_interpolation_speed.py --resolution-divisor 4
```

### Comprehensive Quality Testing
```bash
uv run backend/tools/test_quality_comparison.py --exclude-div8 --frames 30
```

### Statistical Analysis
```bash
uv run backend/tools/analyze_performance_correlations.py
uv run backend/tools/analyze_within_divisor.py --all-runs
```

### Deep Profiling
```bash
uv run backend/tools/profile_interpolation.py --iterations 50
```

---

## 📂 Output Locations

- Test results: `output/quality_tests/run_TIMESTAMP/`
- Visualizations: Saved alongside analysis reports
- HTML comparisons: `comparison_enhanced.html` in test run directory
- JSON metrics: `performance_metrics.json` in test run directory

---

## 🎯 Typical Workflow

### Development & Optimization Workflow
1. **Initial Testing**: Run `test_quality_comparison.py` to generate baseline data
2. **Analysis**: Use `analyze_performance_correlations.py` and `analyze_within_divisor.py` to understand results
3. **Visual Review**: Open generated HTML comparison to visually inspect quality
4. **Optimization**: Use `profile_interpolation.py` to identify bottlenecks
5. **Verification**: Use `test_interpolation_speed.py` for quick checks
6. **Iteration**: Repeat with different configurations

### Animation Generation Workflow
1. **Generate Keyframes**: Create initial keyframes using your generation system
2. **Generate Interpolations**: Run interpolation between consecutive keyframes
3. **Create Animation**: Use `generate_animation.py` to preview the full sequence as a looping animation
4. **Review**: Check the animation to verify smooth transitions and visual quality

---

## 📊 Dependencies

These tools require:
- Core backend modules (`interpolation`, `utils`)
- NumPy, SciPy, scikit-learn (for analysis)
- Matplotlib (for visualizations)
- Pillow (for image operations)
- PyTorch (for VAE operations)

All dependencies are included in the main project requirements.

---

## 🔧 Maintenance Notes

- Scripts use `sys.path.insert` to access parent backend modules
- All scripts are self-contained and can run independently
- Output directories are auto-created as needed
- Scripts handle missing files gracefully with informative errors

---

**Last Updated**: November 2025

