#!/bin/bash
# DreamGen Calibration Entrypoint with Watchdog
# ==============================================
# Runs the calibration suite with automatic restart on failure.
# Maintains state via checkpoints for seamless recovery.
#
# Features:
# - Waits for ComfyUI to be ready before starting
# - Health check endpoint at :8080/health
# - Automatic restart on crash (up to MAX_RETRIES)
# - Checkpoint-based resume after restart
# - Graceful shutdown handling

set -e

# Configuration from environment
MODE="${CALIBRATION_MODE:-full}"
FRAMES="${CALIBRATION_FRAMES:-1500}"
OUTPUT_DIR="${CALIBRATION_OUTPUT_DIR:-/workspace/calibration}"
COMFYUI="${COMFYUI_URL:-http://127.0.0.1:8188}"
UPLOAD_URL="${UPLOAD_RESULTS_URL:-}"
UPLOAD_TOKEN="${UPLOAD_AUTH_TOKEN:-}"
HEALTH_PORT="${HEALTH_CHECK_PORT:-8080}"
MAX_RETRIES="${MAX_RESTART_RETRIES:-3}"
HEALTH_TIMEOUT="${HEALTH_STALE_TIMEOUT:-300}"  # 5 minutes
MAX_DEEP_ROUNDS="${MAX_DEEP_ROUNDS:-3}"  # Limit deep rounds in full mode

# State file for tracking full mode progress (survives restarts)
FULL_STATE_FILE="${OUTPUT_DIR}/.full_mode_state.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# State tracking
RETRY_COUNT=0
CALIBRATION_PID=""

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           DREAM GEN CALIBRATION BENCHMARK SUITE                      ║"
echo "║                                                                      ║"
echo "║   Establishing similarity baselines and threshold recommendations   ║"
echo "║                      (with watchdog + auto-restart)                  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Mode:           $MODE"
echo "  Frames:         $FRAMES"
echo "  Output:         $OUTPUT_DIR"
echo "  ComfyUI:        $COMFYUI"
echo "  Health Port:    $HEALTH_PORT"
echo "  Max Retries:    $MAX_RETRIES"
echo "  Max Deep Rounds: $MAX_DEEP_ROUNDS (full mode only)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/output" "$OUTPUT_DIR/logs"

# Mark as running
touch "$OUTPUT_DIR/.running"
rm -f "$OUTPUT_DIR/.complete" "$OUTPUT_DIR/.failed"

# Graceful shutdown handler
cleanup() {
    echo -e "\n${YELLOW}Received shutdown signal...${NC}"
    if [ -n "$CALIBRATION_PID" ] && kill -0 "$CALIBRATION_PID" 2>/dev/null; then
        echo "Sending SIGTERM to calibration process (PID: $CALIBRATION_PID)"
        kill -TERM "$CALIBRATION_PID" 2>/dev/null || true
        # Wait up to 30 seconds for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$CALIBRATION_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "$CALIBRATION_PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$CALIBRATION_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$OUTPUT_DIR/.running"
    echo -e "${YELLOW}Shutdown complete. Checkpoint preserved for resume.${NC}"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Wait for ComfyUI
wait_for_comfyui() {
    echo -e "${YELLOW}Waiting for ComfyUI to be ready...${NC}"
    local max_wait=300  # 5 minutes
    local waited=0
    
    # Build auth header if credentials provided
    local auth_args=""
    if [ -n "$COMFYUI_AUTH_USER" ] && [ -n "$COMFYUI_AUTH_PASS" ]; then
        auth_args="-u ${COMFYUI_AUTH_USER}:${COMFYUI_AUTH_PASS}"
    fi
    
    while ! curl -sf --max-time 5 $auth_args "$COMFYUI/system_stats" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo -e "${RED}ERROR: ComfyUI not ready after ${max_wait}s${NC}"
            return 1
        fi
        echo "  Waiting... (${waited}s)"
    done
    echo -e "${GREEN}✓ ComfyUI is ready${NC}"
    return 0
}

# Check health endpoint
check_health() {
    local response
    response=$(curl -sf --max-time 5 "http://127.0.0.1:${HEALTH_PORT}/health" 2>/dev/null) || return 1

    # Parse last_activity_seconds_ago from JSON
    local age
    age=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('last_activity_seconds_ago', 9999))" 2>/dev/null) || return 1

    if [ "$age" -gt "$HEALTH_TIMEOUT" ]; then
        echo -e "${RED}Health check: stale (${age}s since last activity)${NC}"
        return 1
    fi
    return 0
}

# ==================== Full Mode State Tracking ====================
# Tracks which phases of "full" mode have completed (survives restarts)

init_full_state() {
    if [ ! -f "$FULL_STATE_FILE" ]; then
        echo -e "${BLUE}Initializing full mode state tracker...${NC}"
        python3 -c "
import json
state = {
    'mode': 'full',
    'started_at': '$(date -Iseconds)',
    'broad_completed': False,
    'deep_rounds_completed': [],
    'deep_rounds_limit': $MAX_DEEP_ROUNDS,
    'intervention_completed': False,
    'total_deep_rounds_ran': 0
}
with open('$FULL_STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print('State initialized')
"
    else
        echo -e "${BLUE}Found existing full mode state file${NC}"
        cat "$FULL_STATE_FILE"
    fi
}

get_full_state() {
    local key=$1
    python3 -c "
import json
with open('$FULL_STATE_FILE') as f:
    state = json.load(f)
print(state.get('$key', ''))
" 2>/dev/null
}

is_deep_round_completed() {
    local template=$1
    python3 -c "
import json
with open('$FULL_STATE_FILE') as f:
    state = json.load(f)
completed = state.get('deep_rounds_completed', [])
print('true' if '$template' in completed else 'false')
" 2>/dev/null
}

get_deep_rounds_count() {
    python3 -c "
import json
with open('$FULL_STATE_FILE') as f:
    state = json.load(f)
print(state.get('total_deep_rounds_ran', 0))
" 2>/dev/null
}

mark_phase_completed() {
    local phase=$1
    local template="${2:-}"

    echo -e "${GREEN}Marking phase completed: $phase ${template}${NC}"
    python3 -c "
import json
with open('$FULL_STATE_FILE') as f:
    state = json.load(f)

if '$phase' == 'broad':
    state['broad_completed'] = True
elif '$phase' == 'deep':
    template = '$template'
    if template and template not in state.get('deep_rounds_completed', []):
        state.setdefault('deep_rounds_completed', []).append(template)
        state['total_deep_rounds_ran'] = state.get('total_deep_rounds_ran', 0) + 1
elif '$phase' == 'intervention':
    state['intervention_completed'] = True

state['last_updated'] = '$(date -Iseconds)'
with open('$FULL_STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
print(f'Updated state: {phase} ${template}')
"
}

check_deep_round_limit() {
    local current_count
    current_count=$(get_deep_rounds_count)

    if [ "$current_count" -ge "$MAX_DEEP_ROUNDS" ]; then
        echo -e "${YELLOW}Deep round limit reached: $current_count / $MAX_DEEP_ROUNDS${NC}"
        return 1
    fi
    echo -e "${BLUE}Deep rounds: $current_count / $MAX_DEEP_ROUNDS${NC}"
    return 0
}

clear_full_state() {
    if [ -f "$FULL_STATE_FILE" ]; then
        echo -e "${YELLOW}Clearing full mode state...${NC}"
        rm -f "$FULL_STATE_FILE"
    fi
}

# Run calibration with a specific mode
run_calibration_mode() {
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
        --health-port "$HEALTH_PORT" \
        --resume \
        $extra_args \
        2>&1 | tee -a "$OUTPUT_DIR/logs/calibration_${mode}.log" &
    
    CALIBRATION_PID=$!
    echo "Calibration started (PID: $CALIBRATION_PID)"
    
    # Wait for process, checking health periodically
    while kill -0 "$CALIBRATION_PID" 2>/dev/null; do
        sleep 30
        
        # Check if process is responsive via health endpoint
        if ! check_health; then
            echo -e "${RED}Process appears stuck (no activity for ${HEALTH_TIMEOUT}s)${NC}"
            # Give it one more chance
            sleep 30
            if ! check_health && kill -0 "$CALIBRATION_PID" 2>/dev/null; then
                echo -e "${RED}Killing stuck process...${NC}"
                kill -9 "$CALIBRATION_PID" 2>/dev/null || true
                return 1
            fi
        fi
    done
    
    # Get exit status
    wait "$CALIBRATION_PID"
    local status=$?
    CALIBRATION_PID=""
    
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ $mode mode complete${NC}"
    else
        echo -e "${RED}✗ $mode mode failed (exit code: $status)${NC}"
    fi
    
    return $status
}

# Main calibration runner with retry logic
run_with_retry() {
    local mode=$1
    local frames=$2
    local extra_args="${3:-}"
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if run_calibration_mode "$mode" "$frames" "$extra_args"; then
            RETRY_COUNT=0  # Reset on success
            return 0
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -e "${YELLOW}Attempt $RETRY_COUNT of $MAX_RETRIES failed${NC}"
        
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo -e "${YELLOW}Waiting 10s before retry (will resume from checkpoint)...${NC}"
            sleep 10
            
            # Re-check ComfyUI is still available
            if ! wait_for_comfyui; then
                echo -e "${RED}ComfyUI not available for retry${NC}"
                return 1
            fi
        fi
    done
    
    echo -e "${RED}Max retries ($MAX_RETRIES) exceeded${NC}"
    return 1
}

# Upload results
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
    echo "  \"retries\": $RETRY_COUNT," >> "$combined"
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

# ==================== Main Execution ====================

# Wait for ComfyUI
if ! wait_for_comfyui; then
    touch "$OUTPUT_DIR/.failed"
    rm -f "$OUTPUT_DIR/.running"
    exit 1
fi
echo ""

# Run calibration based on mode
START_TIME=$(date +%s)
FAILED=false

case "$MODE" in
    broad)
        run_with_retry broad "$FRAMES" || FAILED=true
        ;;
    
    deep)
        FRAMES_PER=$(( FRAMES / 3 ))
        run_with_retry deep "$FRAMES_PER" "--template material_study" || FAILED=true
        run_with_retry deep "$FRAMES_PER" "--template atmospheric_depth" || FAILED=true
        run_with_retry deep "$FRAMES_PER" "--template textural_macro" || FAILED=true
        ;;
    
    intervention)
        run_with_retry intervention "$FRAMES" || FAILED=true
        ;;
    
    full)
        BROAD_FRAMES=$(( FRAMES / 4 ))
        DEEP_FRAMES=$(( FRAMES / 6 ))
        INTERVENTION_FRAMES=$(( FRAMES / 4 ))

        # Initialize state tracking for full mode
        init_full_state

        echo -e "${CYAN}Full calibration suite:${NC}"
        echo "  Broad:        $BROAD_FRAMES frames (all templates)"
        echo "  Deep:         $DEEP_FRAMES frames × up to $MAX_DEEP_ROUNDS templates (limit enforced)"
        echo "  Intervention: $INTERVENTION_FRAMES frames"
        echo ""

        # --- BROAD phase ---
        if [ "$(get_full_state broad_completed)" = "True" ]; then
            echo -e "${GREEN}✓ Broad phase already completed (skipping)${NC}"
        else
            if run_with_retry broad "$BROAD_FRAMES"; then
                mark_phase_completed broad
            else
                FAILED=true
            fi
        fi

        # --- DEEP phases (with limit enforcement) ---
        DEEP_TEMPLATES=("material_study" "atmospheric_depth" "textural_macro")

        for template in "${DEEP_TEMPLATES[@]}"; do
            if [ "$FAILED" = "true" ]; then
                break
            fi

            # Check if already completed
            if [ "$(is_deep_round_completed "$template")" = "true" ]; then
                echo -e "${GREEN}✓ Deep round '$template' already completed (skipping)${NC}"
                continue
            fi

            # Check deep round limit BEFORE running
            if ! check_deep_round_limit; then
                echo -e "${YELLOW}⚠ Skipping deep round '$template' - limit reached${NC}"
                continue
            fi

            echo -e "${BLUE}Running deep round: $template${NC}"
            if run_with_retry deep "$DEEP_FRAMES" "--template $template"; then
                mark_phase_completed deep "$template"
            else
                FAILED=true
            fi
        done

        # Report deep round status
        DEEP_COMPLETED=$(get_deep_rounds_count)
        echo -e "${CYAN}Deep rounds completed: $DEEP_COMPLETED / $MAX_DEEP_ROUNDS limit${NC}"

        # --- INTERVENTION phase ---
        if [ "$FAILED" != "true" ]; then
            if [ "$(get_full_state intervention_completed)" = "True" ]; then
                echo -e "${GREEN}✓ Intervention phase already completed (skipping)${NC}"
            else
                if run_with_retry intervention "$INTERVENTION_FRAMES"; then
                    mark_phase_completed intervention
                else
                    FAILED=true
                fi
            fi
        fi

        # Clear state on successful completion
        if [ "$FAILED" != "true" ]; then
            echo -e "${GREEN}Full calibration suite completed successfully${NC}"
            clear_full_state
        fi
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
if [ "$FAILED" = "true" ]; then
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    CALIBRATION FAILED                                ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    CALIBRATION COMPLETE                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
fi
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  Duration:    ${DURATION_MIN} minutes (${DURATION}s)"
echo "  Mode:        $MODE"
echo "  Frames:      $FRAMES"
echo "  Retries:     $RETRY_COUNT"

# Show deep round tracking info for full mode
if [ "$MODE" = "full" ] && [ -f "$FULL_STATE_FILE" ]; then
    echo ""
    echo -e "${BLUE}Deep Round Tracking:${NC}"
    python3 -c "
import json
try:
    with open('$FULL_STATE_FILE') as f:
        state = json.load(f)
    completed = state.get('deep_rounds_completed', [])
    total_ran = state.get('total_deep_rounds_ran', 0)
    limit = state.get('deep_rounds_limit', $MAX_DEEP_ROUNDS)
    print(f'  Deep rounds ran:    {total_ran} / {limit} limit')
    print(f'  Templates completed: {completed}')
    print(f'  Broad completed:    {state.get(\"broad_completed\", False)}')
    print(f'  Intervention done:  {state.get(\"intervention_completed\", False)}')
except Exception as e:
    print(f'  (error reading state: {e})')
"
fi
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

# Mark complete/failed
rm -f "$OUTPUT_DIR/.running"
if [ "$FAILED" = "true" ]; then
    touch "$OUTPUT_DIR/.failed"
    echo -e "${RED}Calibration failed after $MAX_RETRIES retries${NC}"
else
    touch "$OUTPUT_DIR/.complete"
    echo -e "${GREEN}Calibration complete. Results in: $OUTPUT_DIR${NC}"
fi
echo ""

# Keep container running for result retrieval (optional)
if [ "${KEEP_RUNNING:-false}" = "true" ]; then
    echo "Container staying alive for result retrieval..."
    echo "  Health endpoint: http://localhost:${HEALTH_PORT}/health"
    echo "  Checkpoint: http://localhost:${HEALTH_PORT}/checkpoint"
    tail -f /dev/null
fi
