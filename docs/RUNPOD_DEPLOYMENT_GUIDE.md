# RunPod Deployment Guide for Dream Window

> **Complete step-by-step guide** for deploying Dream Window on RunPod Serverless with Flashboot, including VPS configuration and end-to-end testing.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Overview](#overview)
3. [Part 1: VPS Setup (æthera)](#part-1-vps-setup-æthera)
4. [Part 2: RunPod Account Setup](#part-2-runpod-account-setup)
5. [Part 3: Docker Image](#part-3-docker-image)
6. [Part 4: RunPod Endpoint](#part-4-runpod-endpoint)
7. [Part 5: Environment Variables](#part-5-environment-variables)
8. [Part 6: Testing](#part-6-testing)
9. [Part 7: Troubleshooting](#part-7-troubleshooting)
10. [Cost Management](#cost-management)

---

## Prerequisites

Before starting, ensure you have:

- [ ] A DigitalOcean VPS (or similar) running æthera
- [ ] A domain pointing to your VPS (e.g., `aetherawi.red`)
- [ ] SSL certificate (Let's Encrypt via Certbot)
- [ ] Docker installed locally
- [ ] Docker Hub account (or other container registry)
- [ ] RunPod account (sign up at [runpod.io](https://runpod.io))
- [ ] ~$10 RunPod credits to start

---

## Overview

The deployment has three components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Browser] ←──WebSocket──→ [VPS/æthera] ←──WebSocket──→ [RunPod]│
│                                                                  │
│  1. Browser visits /dreams                                       │
│  2. VPS detects viewer, starts RunPod job                       │
│  3. RunPod GPU connects back to VPS via WebSocket               │
│  4. GPU streams frames to VPS, VPS broadcasts to browsers       │
│  5. When viewers leave, VPS stops RunPod job                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 0: Local Testing (Before Deployment!)

Before deploying to production, test everything locally:

### 0.1 Start æthera Locally

```bash
# Terminal 1: Start the server
cd aethera-mono/aethera
export DREAM_GEN_AUTH_TOKEN="test-token-for-local-dev"
uv run python -m aethera.main
```

### 0.2 Run the Test Script

```bash
# Terminal 2: Run tests
cd aethera-mono
export DREAM_GEN_AUTH_TOKEN="test-token-for-local-dev"
python test_dreams_local.py
```

This tests:
- ✅ REST API endpoints (`/api/dreams/status`, `/current`, `/embed`)
- ✅ Viewer WebSocket (`/ws/dreams`)
- ✅ GPU WebSocket with authentication (`/ws/gpu`)

### 0.3 Simulate GPU Streaming

```bash
# Terminal 2: Stream test frames
export DREAM_GEN_AUTH_TOKEN="test-token-for-local-dev"
python test_dreams_local.py --stream --frames=50

# Terminal 3: Open browser
# Go to http://localhost:8000/dreams
```

You should see colored test frames with frame numbers appearing in the browser!

### 0.4 What This Tests

| Component | Tested Locally | Needs Production |
|-----------|----------------|------------------|
| Dreams viewer page | ✅ | - |
| WebSocket hub | ✅ | - |
| Frame caching | ✅ | - |
| GPU authentication | ✅ | - |
| Rate limiting | ✅ | - |
| RunPod API calls | ❌ | ✅ |
| Nginx/SSL | ❌ | ✅ |
| Real frame generation | ❌ | ✅ |

---

## Part 1: VPS Setup (æthera)

Your setup uses **Docker + GitHub Actions** for deployment. Most changes are automatic!

### 1.1 What's Already Handled

When you push to `main`, the GitHub workflow automatically:
1. Pulls latest code to `/opt/aethera`
2. Builds Docker image with Dreams module included
3. Restarts container with your `.env` file

**No manual deployment needed!** Just push to main.

### 1.2 Configure Environment Variables (One-Time Setup)

SSH to your VPS and add Dreams-specific variables to `/opt/aethera/.env`:

```bash
# SSH to VPS
ssh your-user@your-vps

# Edit .env file
nano /opt/aethera/.env

# Add these lines (keep existing vars like AETHERA_TRIPCODE_SALT):

# Generate a secure shared secret (32+ characters)
# Use: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
DREAM_GEN_AUTH_TOKEN="your-super-secret-token-here"

# RunPod API Key (get from RunPod dashboard → Settings → API Keys)
RUNPOD_API_KEY="your-runpod-api-key"

# RunPod Endpoint ID (get after creating endpoint in Part 4)
RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

**Generating a secure token:**

```bash
# On your local machine or VPS
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Example output: kB7xV2mN9pQ4wR1tY8uZ3cX6fA0hJ5sL
```

**Important:** Save this token! You'll need the same value for RunPod.

### 1.3 Caddy Configuration (Already Set Up!)

You're using Caddy as your reverse proxy. **Good news:** Caddy handles WebSockets automatically!

**Verify your Caddyfile includes aetherawi.red:**

```bash
# SSH to VPS
ssh your-user@your-vps

# Check Caddy config
sudo cat /etc/caddy/Caddyfile
```

It should have something like:

```caddyfile
aetherawi.red {
    reverse_proxy localhost:8000
}
```

**That's it!** Caddy will:
- ✅ Auto-manage SSL certificates
- ✅ Proxy WebSocket connections automatically
- ✅ Handle `/ws/dreams` and `/ws/gpu` without extra config

If you need to reload after any changes:

```bash
sudo systemctl reload caddy
```

### 1.4 Deploy the Changes

**Option A: Push to GitHub (Recommended)**

The GitHub workflow will automatically deploy:

```bash
# On your local machine
cd aethera-mono/aethera
git add .
git commit -m "feat: add Dreams module environment config"
git push origin main
# GitHub Actions will deploy automatically!
```

**Option B: Manual restart on VPS**

If you just added env vars and want to restart:

```bash
# SSH to VPS
ssh your-user@your-vps

# Restart container to pick up new .env
cd /opt/aethera
docker restart aethera

# Check it's running
docker logs aethera --tail 20
```

### 1.5 Verify VPS is Ready

Test the dreams endpoint:

```bash
# Check status endpoint
curl https://aetherawi.red/api/dreams/status

# Expected response:
# {"status":"idle","gpu":{"active":false},...}
```

---

## Part 2: RunPod Account Setup

### 2.1 Create Account

1. Go to [runpod.io](https://runpod.io)
2. Sign up with email or OAuth
3. Verify your email

### 2.2 Add Credits

1. Click on your profile → **Billing**
2. Add $10-20 to start (pay-as-you-go)
3. Serverless charges per-second, so $10 goes far

### 2.3 Get API Key

1. Go to **Settings** → **API Keys**
2. Click **Create API Key**
3. Name it "dream-window" or similar
4. **Copy the key immediately** (won't be shown again!)
5. Save it as `RUNPOD_API_KEY` in your VPS `.env`

---

## Part 3: Docker Image

### 3.1 Build the Docker Image

On your local machine (with Docker installed):

```bash
# Clone dream_gen if not already present
cd ~/projects
git clone https://github.com/LuxiaSL/dream_gen.git
cd dream_gen

# Build the cloud Docker image
docker build -t dreamwindow:latest -f docker/Dockerfile.cloud .
```

**Build time:** ~10-15 minutes (downloads ComfyUI, models, etc.)

### 3.2 Tag and Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag with your username
docker tag dreamwindow:latest yourusername/dreamwindow:latest

# Push
docker push yourusername/dreamwindow:latest
```

**Alternative: Push to RunPod Registry**

RunPod also has its own registry, but Docker Hub is simpler to start.

### 3.3 Verify Image

```bash
# Check image is public
curl -s https://hub.docker.com/v2/repositories/yourusername/dreamwindow/tags | jq '.results[0].name'
# Should show: "latest"
```

---

## Part 4: RunPod Endpoint

### 4.1 Create Serverless Endpoint

1. Go to RunPod dashboard
2. Click **Serverless** in the left menu
3. Click **+ New Endpoint**

### 4.2 Configure Endpoint

Fill in the form:

| Setting | Value | Notes |
|---------|-------|-------|
| **Name** | `dream-window` | Any descriptive name |
| **Container Image** | `yourusername/dreamwindow:latest` | Your Docker Hub image |
| **Container Disk** | 20 GB | For models and temp files |
| **GPU Type** | RTX 3060 (12GB) or RTX 3070 | 12GB VRAM recommended |
| **Max Workers** | 1 | Only need one for single stream |
| **Idle Timeout** | 30 seconds | Matches VPS grace period |
| **Active Workers** | 0 | Scale to zero when idle |
| **Min Workers** | 0 | Truly serverless |

### 4.3 Environment Variables (RunPod Side)

Add these environment variables in the endpoint config:

```
VPS_WEBSOCKET_URL=wss://aetherawi.red/ws/gpu
DREAM_GEN_AUTH_TOKEN=your-super-secret-token-here
LOG_LEVEL=INFO
```

**Important:** The `DREAM_GEN_AUTH_TOKEN` must match what's on your VPS!

### 4.4 Advanced Settings

Under "Advanced":

- **Execution Timeout:** 0 (no timeout, runs indefinitely)
- **Startup Delay:** 0
- **Entrypoint:** Leave blank (uses Dockerfile CMD)

### 4.5 Create Endpoint

Click **Create Endpoint**.

### 4.6 Copy Endpoint ID

After creation, you'll see your endpoint with an ID like:

```
abc123xyz456
```

**Copy this ID** and add it to your VPS `.env`:

```bash
RUNPOD_ENDPOINT_ID="abc123xyz456"
```

### 4.7 Enable Flashboot (Optional but Recommended)

Flashboot pre-loads your Docker image layers for faster cold starts.

1. Go to your endpoint settings
2. Find **Flashboot**
3. Enable it
4. Wait ~10 minutes for the snapshot to be created

**Result:** Cold starts drop from 3-5 minutes to 30-60 seconds!

---

## Part 5: Environment Variables

### Summary of All Environment Variables

**VPS (.env):**

```bash
# Required
DREAM_GEN_AUTH_TOKEN="your-shared-secret"
RUNPOD_API_KEY="your-runpod-api-key"
RUNPOD_ENDPOINT_ID="your-endpoint-id"

# Optional
RUNPOD_WEBHOOK_URL=""  # For status callbacks
```

**RunPod Endpoint:**

```bash
# Required
VPS_WEBSOCKET_URL="wss://yourdomain.com/ws/gpu"
DREAM_GEN_AUTH_TOKEN="your-shared-secret"

# Optional
LOG_LEVEL="INFO"
```

### Restart Services After Changes

On VPS:

```bash
sudo systemctl restart aethera
# or
pkill -f "aethera.main" && cd /var/www/aethera && uv run python -m aethera.main &
```

---

## Part 6: Testing

### 6.1 Test GPU WebSocket Authentication

```bash
# On VPS, test that auth works
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Authorization: Bearer your-shared-secret" \
  https://aetherawi.red/ws/gpu

# Should NOT immediately close (if it does, auth failed)
```

### 6.2 Test RunPod Endpoint Manually

Using RunPod's test feature:

1. Go to your endpoint in RunPod dashboard
2. Click **Test** or **Run**
3. Input JSON:
   ```json
   {
     "type": "health"
   }
   ```
4. Should return:
   ```json
   {
     "status": "healthy",
     "message": "Dream Window handler ready"
   }
   ```

### 6.3 Test Full Flow

1. Open `https://yourdomain.com/dreams` in browser
2. Watch the loading animation
3. Check RunPod dashboard → you should see a worker spinning up
4. After 30-60 seconds, frames should start appearing
5. Close the browser tab
6. Wait 30 seconds
7. Check RunPod dashboard → worker should stop

### 6.4 Monitor Logs

**VPS Logs:**

```bash
# If using systemd
journalctl -u aethera -f

# Or check application logs
tail -f /var/www/aethera/logs/*.log
```

**RunPod Logs:**

1. Go to your endpoint in RunPod dashboard
2. Click **Logs**
3. Select the worker instance
4. View real-time logs

---

## Part 7: Troubleshooting

### Problem: GPU never connects

**Symptoms:** Loading animation never ends

**Check:**
1. Is RunPod job starting? (Check RunPod dashboard)
2. Is VPS WebSocket reachable? (`curl wss://...`)
3. Do auth tokens match?

**Debug:**
```bash
# On VPS, check aethera logs
grep "GPU" /var/log/aethera/*.log

# In RunPod logs, look for connection errors
```

### Problem: Frames not appearing

**Symptoms:** Connected but canvas stays blank

**Check:**
1. Is ComfyUI running inside container?
2. Are models loaded?
3. Is WebSocket receiving frames?

**Debug:**
```python
# In browser console
dreamViewer.ws.onmessage = (e) => console.log("Received:", e.data);
```

### Problem: GPU starts but stops immediately

**Symptoms:** Worker spins up then terminates

**Check:**
1. Is idle timeout too short?
2. Is there an error in RunPod logs?
3. Is VPS rejecting the WebSocket connection?

**Fix:**
- Increase idle timeout to 60+ seconds
- Check auth token matches
- Check Nginx WebSocket config

### Problem: High costs

**Symptoms:** Bill higher than expected

**Check:**
1. Are workers actually stopping when idle?
2. Is idle timeout configured correctly?
3. Are there multiple workers running?

**Fix:**
- Set max workers to 1
- Reduce idle timeout
- Check for stuck jobs in RunPod dashboard

---

## Cost Management

### Expected Costs

| Usage Pattern | Hours/Day | Monthly Cost |
|---------------|-----------|--------------|
| Demo (occasional) | 1 hr | ~$4 |
| Regular | 4 hrs | ~$15 |
| Popular | 8 hrs | ~$30 |
| Heavy | 16 hrs | ~$60 |

### Cost Optimization Tips

1. **Set billing alerts** in RunPod dashboard
2. **Use idle timeout** (30 seconds is good)
3. **Monitor worker count** — should never exceed 1
4. **Use Flashboot** — reduces cold start time, no extra cost
5. **Check for stuck jobs** periodically

### Billing Alerts

1. Go to RunPod → Settings → Billing
2. Set up alert at $30, $50, $70
3. Get email notifications before hitting budget

---

## Quick Reference

### Useful Commands

```bash
# Check VPS dreams status
curl https://yourdomain.com/api/dreams/status | jq

# Test RunPod endpoint health
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"type": "health"}}'

# Force stop all RunPod jobs
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/purge-queue \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"

# Check RunPod worker status
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" | jq
```

### Key Files

| File | Purpose |
|------|---------|
| `aethera/dreams/gpu_manager.py` | RunPod API integration |
| `aethera/api/dreams.py` | WebSocket endpoints |
| `dream_gen/docker/Dockerfile.cloud` | GPU container image |
| `dream_gen/backend/cloud/runpod_handler.py` | RunPod entry point |

### Support

- **RunPod Discord:** [discord.gg/runpod](https://discord.gg/runpod)
- **Dream Window Issues:** GitHub Issues
- **Author:** @luxia on Discord

---

*Last updated: 2024-12-10*

