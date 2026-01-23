#!/bin/bash
# ComfyUI Startup Script (Two-Pod Architecture)
# =============================================
# 1. Creates auth file from environment variables
# 2. Starts nginx reverse proxy
# 3. Registers with VPS for service discovery
# 4. Starts ComfyUI on internal port
#
# Environment Variables:
#   COMFYUI_AUTH_USER  - Basic auth username (required)
#   COMFYUI_AUTH_PASS  - Basic auth password (required)
#   VPS_REGISTER_URL   - VPS registration endpoint (optional)
#   VPS_AUTH_TOKEN     - VPS auth token for registration (optional)
#   RUNPOD_POD_ID      - RunPod pod ID (auto-set by RunPod)

set -e

echo "=========================================="
echo "ComfyUI Startup Script"
echo "=========================================="

# ==================== Auth Setup ====================
if [ -n "$COMFYUI_AUTH_USER" ] && [ -n "$COMFYUI_AUTH_PASS" ]; then
    echo "Creating auth file for user: $COMFYUI_AUTH_USER"
    htpasswd -bc /etc/nginx/.htpasswd "$COMFYUI_AUTH_USER" "$COMFYUI_AUTH_PASS"
else
    echo "WARNING: COMFYUI_AUTH_USER or COMFYUI_AUTH_PASS not set!"
    echo "         ComfyUI will be accessible without authentication."
    # Create a permissive config (allow all)
    echo "" > /etc/nginx/.htpasswd
    # Modify nginx config to not require auth
    sed -i 's/auth_basic "ComfyUI";/#auth_basic "ComfyUI";/' /etc/nginx/sites-available/comfyui
    sed -i 's/auth_basic_user_file/#auth_basic_user_file/' /etc/nginx/sites-available/comfyui
fi

# ==================== Start Nginx ====================
echo "Starting nginx reverse proxy..."
nginx
echo "Nginx started (listening on port 8188)"

# ==================== Start ComfyUI ====================
echo "Starting ComfyUI on internal port 8189..."
cd /app/ComfyUI

# Start ComfyUI in background
python main.py \
    --listen 127.0.0.1 \
    --port 8189 \
    --highvram \
    --cuda-malloc \
    --preview-method none &

COMFYUI_PID=$!
echo "ComfyUI started (PID: $COMFYUI_PID)"

# ==================== Wait for ComfyUI Ready ====================
echo "Waiting for ComfyUI to be ready..."
STARTUP_TIMEOUT=120
ELAPSED=0

while [ $ELAPSED -lt $STARTUP_TIMEOUT ]; do
    # Check internal port (no auth needed for internal)
    if curl -sf http://127.0.0.1:8189/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is ready! (took ${ELAPSED}s)"
        break
    fi
    
    # Check if ComfyUI process died
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
        echo "ERROR: ComfyUI process died!"
        exit 1
    fi
    
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "  Still waiting for ComfyUI... (${ELAPSED}s)"
    fi
done

if [ $ELAPSED -ge $STARTUP_TIMEOUT ]; then
    echo "ERROR: ComfyUI failed to start within ${STARTUP_TIMEOUT}s"
    exit 1
fi

# ==================== Register with VPS ====================
if [ -n "$VPS_REGISTER_URL" ]; then
    echo "Registering with VPS..."
    
    # Get public IP
    PUBLIC_IP=$(curl -sf ifconfig.me || curl -sf icanhazip.com || curl -sf ipinfo.io/ip || echo "")
    
    if [ -z "$PUBLIC_IP" ]; then
        echo "WARNING: Could not determine public IP, skipping VPS registration"
    else
        echo "Public IP: $PUBLIC_IP"
        
        # Build registration payload
        PAYLOAD=$(cat <<EOF
{
    "ip": "$PUBLIC_IP",
    "port": 8188,
    "auth_user": "${COMFYUI_AUTH_USER:-}",
    "auth_pass": "${COMFYUI_AUTH_PASS:-}",
    "pod_id": "${RUNPOD_POD_ID:-}"
}
EOF
)
        
        # Register with VPS
        REGISTER_RESPONSE=$(curl -sf -X POST "$VPS_REGISTER_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${VPS_AUTH_TOKEN:-}" \
            -d "$PAYLOAD" 2>&1) || true
        
        if [ -n "$REGISTER_RESPONSE" ]; then
            echo "VPS registration response: $REGISTER_RESPONSE"
        else
            echo "WARNING: VPS registration failed (VPS may be unreachable)"
        fi
    fi
else
    echo "VPS_REGISTER_URL not set, skipping VPS registration"
fi

# ==================== Keep Running ====================
echo "=========================================="
echo "ComfyUI is running!"
echo "  External port: 8188 (nginx with auth)"
echo "  Internal port: 8189 (ComfyUI direct)"
echo "=========================================="

# Wait for ComfyUI process (keeps container running)
wait $COMFYUI_PID

