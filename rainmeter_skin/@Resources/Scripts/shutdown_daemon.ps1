# Dream Window - Daemon Shutdown Script for Rainmeter
# ============================================
# 
# This script is called by Rainmeter when the DreamWindow skin closes.
# It gracefully shuts down the daemon using the daemon_control.py script.

param(
    [string]$ProjectPath = "C:\Users\[username_here]\Documents\projects\dream_gen"
)

# Change directory to project root
try {
    Set-Location $ProjectPath
    Write-Host "Shutting down daemon from: $ProjectPath"
} catch {
    Write-Host "ERROR: Could not access project directory: $ProjectPath"
    exit 1
}

# Check if uv is available
$uvPath = Get-Command uv -ErrorAction SilentlyContinue

if ($uvPath) {
    Write-Host "Using uv to run shutdown command..."
    $result = uv run daemon_control.py shutdown
} else {
    Write-Host "uv not found, trying python directly..."
    $result = python daemon_control.py shutdown
}

Write-Host "Shutdown command completed"
Write-Host $result

exit 0