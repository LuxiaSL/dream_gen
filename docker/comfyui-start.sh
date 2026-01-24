#!/bin/bash
# ComfyUI Startup Script (Two-Pod Architecture)
# =============================================
# 1. Optionally fetches secrets from admin panel (bootstrap mode)
# 2. Creates auth file from environment variables
# 3. Starts nginx reverse proxy
# 4. Registers with VPS for service discovery
# 5. Starts ComfyUI on internal port
#
# Environment Variables (Direct Mode - provide all secrets):
#   COMFYUI_AUTH_USER  - Basic auth username (required)
#   COMFYUI_AUTH_PASS  - Basic auth password (required)
#   VPS_REGISTER_URL   - VPS registration endpoint (optional)
#   VPS_AUTH_TOKEN     - VPS auth token for registration (optional)
#   RUNPOD_POD_ID      - RunPod pod ID (auto-set by RunPod)
#
# Environment Variables (Bootstrap Mode - fetch secrets from admin):
#   ADMIN_PANEL_URL    - Admin panel base URL (e.g., https://admin.aetherawi.red)
#   POD_BOOTSTRAP_TOKEN - Bootstrap token to authenticate with admin panel
#
# Bootstrap mode is used when creating pods automatically via lifecycle management.
# The pod only needs the admin URL and a bootstrap token; it fetches other secrets.

set -e

echo "=========================================="
echo "ComfyUI Startup Script"
echo "=========================================="

# ==================== Bootstrap Mode ====================
# If ADMIN_PANEL_URL and POD_BOOTSTRAP_TOKEN are set, fetch secrets from admin
if [ -n "$ADMIN_PANEL_URL" ] && [ -n "$POD_BOOTSTRAP_TOKEN" ]; then
    echo "Bootstrap mode: Fetching secrets from admin panel..."
    echo "Admin URL: $ADMIN_PANEL_URL"
    
    # Fetch secrets from admin panel
    SECRETS_URL="${ADMIN_PANEL_URL}/api/dreams/secrets/comfyui?token=${POD_BOOTSTRAP_TOKEN}"
    
    SECRETS_RESPONSE=$(curl -sf "$SECRETS_URL" 2>&1) || {
        echo "ERROR: Failed to fetch secrets from admin panel"
        echo "Response: $SECRETS_RESPONSE"
        echo "Falling back to environment variables..."
    }
    
    if [ -n "$SECRETS_RESPONSE" ] && echo "$SECRETS_RESPONSE" | grep -q "comfyui_auth_user"; then
        echo "Successfully retrieved secrets from admin panel"
        
        # Parse JSON response (simple extraction)
        # Note: This uses basic tools available in most containers
        # For more robust parsing, you could add jq to the image
        
        # Extract values using grep and sed
        FETCHED_AUTH_USER=$(echo "$SECRETS_RESPONSE" | grep -o '"comfyui_auth_user":"[^"]*"' | sed 's/.*:"\([^"]*\)"/\1/')
        FETCHED_AUTH_PASS=$(echo "$SECRETS_RESPONSE" | grep -o '"comfyui_auth_pass":"[^"]*"' | sed 's/.*:"\([^"]*\)"/\1/')
        FETCHED_VPS_TOKEN=$(echo "$SECRETS_RESPONSE" | grep -o '"vps_auth_token":"[^"]*"' | sed 's/.*:"\([^"]*\)"/\1/')
        FETCHED_VPS_REGISTER=$(echo "$SECRETS_RESPONSE" | grep -o '"vps_register_url":"[^"]*"' | sed 's/.*:"\([^"]*\)"/\1/')
        
        # Override env vars with fetched values (if not empty)
        [ -n "$FETCHED_AUTH_USER" ] && export COMFYUI_AUTH_USER="$FETCHED_AUTH_USER"
        [ -n "$FETCHED_AUTH_PASS" ] && export COMFYUI_AUTH_PASS="$FETCHED_AUTH_PASS"
        [ -n "$FETCHED_VPS_TOKEN" ] && export VPS_AUTH_TOKEN="$FETCHED_VPS_TOKEN"
        [ -n "$FETCHED_VPS_REGISTER" ] && export VPS_REGISTER_URL="$FETCHED_VPS_REGISTER"
        
        echo "Secrets loaded: AUTH_USER=${COMFYUI_AUTH_USER}, VPS_REGISTER=${VPS_REGISTER_URL}"
    fi
else
    echo "Direct mode: Using environment variables for configuration"
fi

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
    sed -i 's/auth_basic "ComfyUI";/#auth_basic "Comfyui";/' /etc/nginx/sites-available/comfyui
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
    
    # Determine the ComfyUI endpoint URL
    # On RunPod, we MUST use the proxy URL format: https://{pod_id}-{port}.proxy.runpod.net
    # Direct IP:port does NOT work on RunPod due to their networking
    
    if [ -n "$RUNPOD_POD_ID" ]; then
        # Running on RunPod - use proxy URL
        COMFYUI_URL="https://${RUNPOD_POD_ID}-8188.proxy.runpod.net"
        echo "RunPod detected (pod: $RUNPOD_POD_ID)"
        echo "Using proxy URL: $COMFYUI_URL"
    else
        # Not on RunPod - use direct IP:port
        PUBLIC_IP=$(curl -sf ifconfig.me || curl -sf icanhazip.com || curl -sf ipinfo.io/ip || echo "")
        if [ -z "$PUBLIC_IP" ]; then
            echo "WARNING: Could not determine public IP, skipping VPS registration"
            COMFYUI_URL=""
        else
            COMFYUI_URL="http://${PUBLIC_IP}:8188"
            echo "Public IP: $PUBLIC_IP"
            echo "Using direct URL: $COMFYUI_URL"
        fi
    fi
    
    if [ -n "$COMFYUI_URL" ]; then
        # Build registration payload with full URL
        # Note: We include both the URL and the legacy ip/port fields for compatibility
        PAYLOAD=$(cat <<EOF
{
    "url": "$COMFYUI_URL",
    "ip": "${RUNPOD_POD_ID:-$PUBLIC_IP}",
    "port": 8188,
    "auth_user": "${COMFYUI_AUTH_USER:-}",
    "auth_pass": "${COMFYUI_AUTH_PASS:-}",
    "pod_id": "${RUNPOD_POD_ID:-}"
}
EOF
)
        
        echo "Sending registration to $VPS_REGISTER_URL..."
        
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

