#!/bin/bash
# DreamGen Calibration Entrypoint
# ================================
# Runs the full calibration suite and outputs recommendations.
#
# Expects ComfyUI to be running (either in same pod or separate).
# Waits for ComfyUI to be ready before starting.

set -e

# Configuration from environment
MODE="${CALIBRATION_MODE:-full}"
FRAMES="${CALIBRATION_FRAMES:-1500}"
OUTPUT_DIR="${CALIBRATION_OUTPUT_DIR:-/workspace/calibration}"
COMFYUI="${COMFYUI_URL:-http://127.0.0.1:8188}"
UPLOAD_URL="${UPLOAD_RESULTS_URL:-}"
UPLOAD_TOKEN="${UPLOAD_AUTH_TOKEN:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           DREAM GEN CALIBRATION BENCHMARK SUITE                      ║"
echo "║                                                                      ║"
echo "║   Establishing similarity baselines and threshold recommendations   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Mode:       $MODE"
echo "  Frames:     $FRAMES"
echo "  Output:     $OUTPUT_DIR"
echo "  ComfyUI:    $COMFYUI"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/output" "$OUTPUT_DIR/logs"

# Mark as running
touch "$OUTPUT_DIR/.running"
rm -f "$OUTPUT_DIR/.complete" "$OUTPUT_DIR/.failed"

# Wait for ComfyUI
echo -e "${YELLOW}Waiting for ComfyUI to be ready...${NC}"
MAX_WAIT=300  # 5 minutes
WAITED=0
while ! curl -s --max-time 5 "$COMFYUI/system_stats" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo -e "${RED}ERROR: ComfyUI not ready after ${MAX_WAIT}s${NC}"
        touch "$OUTPUT_DIR/.failed"
        rm -f "$OUTPUT_DIR/.running"
        exit 1
    fi
    echo "  Waiting... (${WAITED}s)"
done
echo -e "${GREEN}✓ ComfyUI is ready${NC}"
echo ""

# Function to run a calibration mode
run_calibration() {
    local mode=$1
    local frames=$2
    local extra_args="${3:-}"
    
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Running: $mode mode ($frames frames)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    
    cd /app
    python -m backend.tools.calibration_benchmark \
        "$mode" \
        --num-frames "$frames" \
        --output-dir "$OUTPUT_DIR" \
        --config /app/calibration/config.calibration.yaml \
        $extra_args \
        2>&1 | tee -a "$OUTPUT_DIR/logs/calibration_${mode}.log"
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ $mode mode complete${NC}"
    else
        echo -e "${RED}✗ $mode mode failed (exit code: $status)${NC}"
    fi
    echo ""
    return $status
}

# Function to upload results
upload_results() {
    if [ -z "$UPLOAD_URL" ]; then
        return 0
    fi
    
    echo -e "${YELLOW}Uploading results to $UPLOAD_URL...${NC}"
    
    # Combine all JSON results
    local combined="$OUTPUT_DIR/calibration_combined.json"
    echo "{" > "$combined"
    echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$combined"
    echo "  \"mode\": \"$MODE\"," >> "$combined"
    echo "  \"frames\": $FRAMES," >> "$combined"
    echo "  \"results\": {" >> "$combined"
    
    local first=true
    for json in "$OUTPUT_DIR"/calibration_*.json; do
        if [ -f "$json" ] && [ "$json" != "$combined" ]; then
            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$combined"
            fi
            local name=$(basename "$json" .json)
            echo "    \"$name\": $(cat "$json")" >> "$combined"
        fi
    done
    
    echo "  }" >> "$combined"
    echo "}" >> "$combined"
    
    # Upload
    local auth_header=""
    if [ -n "$UPLOAD_TOKEN" ]; then
        auth_header="-H \"Authorization: Bearer $UPLOAD_TOKEN\""
    fi
    
    curl -X POST "$UPLOAD_URL" \
        -H "Content-Type: application/json" \
        $auth_header \
        -d @"$combined" \
        && echo -e "${GREEN}✓ Results uploaded${NC}" \
        || echo -e "${RED}✗ Upload failed${NC}"
}

# Run calibration based on mode
START_TIME=$(date +%s)

case "$MODE" in
    broad)
        run_calibration broad "$FRAMES"
        ;;
    
    deep)
        # Run deep on multiple templates for comparison
        FRAMES_PER=$(( FRAMES / 3 ))
        run_calibration deep "$FRAMES_PER" "--template material_study"
        run_calibration deep "$FRAMES_PER" "--template atmospheric_depth"
        run_calibration deep "$FRAMES_PER" "--template textural_macro"
        ;;
    
    intervention)
        run_calibration intervention "$FRAMES"
        ;;
    
    full)
        # Full suite: broad (1/4), deep x3 (1/2), intervention (1/4)
        BROAD_FRAMES=$(( FRAMES / 4 ))
        DEEP_FRAMES=$(( FRAMES / 6 ))  # 3 templates
        INTERVENTION_FRAMES=$(( FRAMES / 4 ))
        
        echo -e "${CYAN}Full calibration suite:${NC}"
        echo "  Broad:        $BROAD_FRAMES frames (all templates)"
        echo "  Deep:         $DEEP_FRAMES frames × 3 templates"
        echo "  Intervention: $INTERVENTION_FRAMES frames"
        echo ""
        
        run_calibration broad "$BROAD_FRAMES"
        
        run_calibration deep "$DEEP_FRAMES" "--template material_study"
        run_calibration deep "$DEEP_FRAMES" "--template atmospheric_depth" 
        run_calibration deep "$DEEP_FRAMES" "--template textural_macro"
        
        run_calibration intervention "$INTERVENTION_FRAMES"
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Valid modes: broad, deep, intervention, full"
        touch "$OUTPUT_DIR/.failed"
        rm -f "$OUTPUT_DIR/.running"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))
DURATION_MIN=$(( DURATION / 60 ))

# Generate summary
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    CALIBRATION COMPLETE                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  Duration:    ${DURATION_MIN} minutes (${DURATION}s)"
echo "  Mode:        $MODE"
echo "  Frames:      $FRAMES"
echo ""
echo -e "${BLUE}Results:${NC}"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""

# Print recommended config from latest result
LATEST_JSON=$(ls -t "$OUTPUT_DIR"/calibration_*.json 2>/dev/null | head -1)
if [ -f "$LATEST_JSON" ]; then
    echo -e "${GREEN}Recommended Configuration:${NC}"
    python3 -c "
import json
with open('$LATEST_JSON') as f:
    data = json.load(f)
rec = data.get('recommended_config', {})
if rec:
    print('cache:')
    print('  color_histogram:')
    for k, v in rec.get('color_histogram', {}).items():
        print(f'    {k}: {v}')
    print('  phash:')
    for k, v in rec.get('phash', {}).items():
        print(f'    {k}: {v}')
else:
    print('  (no recommendations in result)')
"
    echo ""
fi

# Upload if configured
upload_results

# Mark complete
rm -f "$OUTPUT_DIR/.running"
touch "$OUTPUT_DIR/.complete"

echo -e "${GREEN}Calibration complete. Results in: $OUTPUT_DIR${NC}"
echo ""

# Keep container running for result retrieval (optional)
if [ "${KEEP_RUNNING:-false}" = "true" ]; then
    echo "Container staying alive for result retrieval..."
    tail -f /dev/null
fi

