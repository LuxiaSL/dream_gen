#!/bin/bash
# Calibration Benchmark Runner for RunPod
# ========================================
#
# This script sets up and runs the calibration benchmark suite.
# Deploy to a RunPod with ComfyUI pre-configured.
#
# Usage:
#   ./run_calibration.sh broad 500      # 500-frame template survey
#   ./run_calibration.sh deep 500       # 500-frame drift analysis
#   ./run_calibration.sh intervention 100  # 100-frame effect size test
#   ./run_calibration.sh full 1500      # Full suite (1500 frames)
#   ./run_calibration.sh analyze /path  # Analyze existing frames
#
# Environment variables:
#   CALIBRATION_OUTPUT_DIR - Output directory (default: /workspace/calibration)
#   COMFYUI_URL - ComfyUI URL (default: http://127.0.0.1:8188)

set -e

# Configuration
OUTPUT_DIR="${CALIBRATION_OUTPUT_DIR:-/workspace/calibration}"
COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")/backend"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          DREAM GEN CALIBRATION BENCHMARK                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
MODE="${1:-broad}"
NUM_FRAMES="${2:-500}"
TEMPLATE="${3:-}"

# Validate mode
case "$MODE" in
    broad|deep|intervention|full|analyze)
        echo -e "${GREEN}Mode:${NC} $MODE"
        ;;
    *)
        echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
        echo "Valid modes: broad, deep, intervention, full, analyze"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"

# Check ComfyUI connection (skip for analyze mode)
if [ "$MODE" != "analyze" ]; then
    echo -e "${YELLOW}Checking ComfyUI connection...${NC}"
    if ! curl -s --max-time 5 "$COMFYUI_URL/system_stats" > /dev/null 2>&1; then
        echo -e "${RED}Error: Cannot connect to ComfyUI at $COMFYUI_URL${NC}"
        echo "Make sure ComfyUI is running before starting calibration."
        exit 1
    fi
    echo -e "${GREEN}✓ ComfyUI connected${NC}"
fi

# Show sprint information
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}OPTIMIZATION TARGETS (per 500-keyframe sprint):${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo "  Template switches: ~5   (every 100 keyframes)"
echo "  Cache injections:  ~12  (every 33-50 keyframes)"
echo "  Mutations:         ~40  (every 7-18 keyframes)"
echo ""
echo "  500 keyframes ≈ 7500 interpolated frames ≈ 25 min playback"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Build the command
CMD="cd $BACKEND_DIR && uv run python tools/calibration_benchmark.py"
CMD="$CMD $MODE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --config $SCRIPT_DIR/config.calibration.yaml"

if [ "$MODE" = "analyze" ]; then
    if [ -z "$NUM_FRAMES" ] || [ ! -d "$NUM_FRAMES" ]; then
        echo -e "${RED}Error: analyze mode requires a frame directory path${NC}"
        echo "Usage: ./run_calibration.sh analyze /path/to/keyframes"
        exit 1
    fi
    CMD="$CMD --frames $NUM_FRAMES"
else
    CMD="$CMD --num-frames $NUM_FRAMES"
    echo -e "${GREEN}Frames:${NC} $NUM_FRAMES"
fi

if [ -n "$TEMPLATE" ] && [ "$MODE" = "deep" ]; then
    CMD="$CMD --template $TEMPLATE"
    echo -e "${GREEN}Template:${NC} $TEMPLATE"
fi

echo ""
echo -e "${YELLOW}Starting calibration...${NC}"
echo -e "${BLUE}Command:${NC} $CMD"
echo ""

# Run the benchmark
eval "$CMD"

# Results summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  CALIBRATION COMPLETE                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC} $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "To download results:"
echo "  scp runpod:$OUTPUT_DIR/*.json ."
echo ""

