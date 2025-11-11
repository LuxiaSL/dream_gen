# Dream Window Daemon Launcher for Rainmeter
# ============================================
# 
# This script is called by Rainmeter when the DreamWindow skin loads.
# It checks if the daemon is already running, and launches it if not.
#
# Features:
# - Checks for existing daemon via PID file
# - Validates process is actually running (not just stale PID)
# - Launches daemon in background (no console window)
# - Uses pythonw.exe for truly hidden execution
# - Returns status to Rainmeter for display

param(
    [string]$ProjectPath = "C:\Users\luxia\Documents\projects\dream_gen",
    [string]$ConfigPath = "backend\config.yaml"
)

# ============================================
# CONFIGURATION
# ============================================

# Change directory to project root
try {
    Set-Location $ProjectPath
    Write-Host "Project Path: $ProjectPath"
} catch {
    Write-Host "ERROR: Could not access project directory: $ProjectPath"
    Write-Host "       Please configure ProjectPath in Variables.inc"
    exit 1
}

# Resolve config file
$ConfigFile = Join-Path $ProjectPath $ConfigPath
if (!(Test-Path $ConfigFile)) {
    Write-Host "ERROR: Config file not found: $ConfigFile"
    exit 1
}

# Read config to get PID file location
try {
    # Simple YAML parsing for pid_file (assumes format: "pid_file: path")
    $ConfigContent = Get-Content $ConfigFile -Raw
    if ($ConfigContent -match 'pid_file:\s*"?([^"\r\n]+)"?') {
        $PidFilePath = $matches[1]
    } else {
        # Default if not found in config
        $PidFilePath = "output\daemon.pid"
    }
    
    $PidFile = Join-Path $ProjectPath $PidFilePath
    Write-Host "PID File: $PidFile"
} catch {
    Write-Host "WARNING: Could not parse config, using default PID file location"
    $PidFile = Join-Path $ProjectPath "output\daemon.pid"
}

# ============================================
# CHECK IF DAEMON IS ALREADY RUNNING
# ============================================

function Test-DaemonRunning {
    param([string]$PidFilePath)
    
    if (!(Test-Path $PidFilePath)) {
        return $false
    }
    
    try {
        $PID = Get-Content $PidFilePath -Raw | ForEach-Object { $_.Trim() }
        
        # Check if process with this PID exists
        $Process = Get-Process -Id $PID -ErrorAction SilentlyContinue
        
        if ($Process) {
            Write-Host "✓ Daemon already running (PID: $PID)"
            return $true
        } else {
            Write-Host "  Stale PID file found (process $PID not running)"
            # Clean up stale PID file
            Remove-Item $PidFilePath -Force -ErrorAction SilentlyContinue
            return $false
        }
    } catch {
        Write-Host "  Error checking PID file: $_"
        return $false
    }
}

if (Test-DaemonRunning -PidFilePath $PidFile) {
    Write-Host "  No action needed - daemon is operational"
    exit 0
}

# ============================================
# FIND PYTHON EXECUTABLE
# ============================================

Write-Host ""
Write-Host "Daemon not running - starting..."

# Look for pythonw.exe (hidden) or python.exe (console)
$PythonExe = $null

# Check common locations
$PythonLocations = @(
    # Virtual environment
    (Join-Path $ProjectPath ".venv\Scripts\pythonw.exe"),
    (Join-Path $ProjectPath ".venv\Scripts\python.exe"),
    # System Python
    (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source,
    (Get-Command python.exe -ErrorAction SilentlyContinue).Source
)

foreach ($Location in $PythonLocations) {
    if ($Location -and (Test-Path $Location)) {
        $PythonExe = $Location
        Write-Host "Found Python: $PythonExe"
        break
    }
}

if (!$PythonExe) {
    Write-Host "ERROR: Python not found"
    Write-Host "       Install Python or activate virtual environment"
    exit 1
}

# Prefer pythonw.exe for truly hidden execution
if ($PythonExe -like "*python.exe") {
    $PythonwPath = $PythonExe -replace "python\.exe$", "pythonw.exe"
    if (Test-Path $PythonwPath) {
        $PythonExe = $PythonwPath
        Write-Host "Using pythonw.exe for hidden execution"
    }
}

# ============================================
# LAUNCH DAEMON
# ============================================

$DaemonScript = Join-Path $ProjectPath "daemon.py"

if (!(Test-Path $DaemonScript)) {
    Write-Host "ERROR: Daemon script not found: $DaemonScript"
    exit 1
}

Write-Host "Launching daemon..."
Write-Host "  Script: $DaemonScript"
Write-Host "  Config: $ConfigFile"

try {
    # Launch daemon as background process (no window)
    $ProcessStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $ProcessStartInfo.FileName = $PythonExe
    $ProcessStartInfo.Arguments = "`"$DaemonScript`" --config `"$ConfigPath`""
    $ProcessStartInfo.WorkingDirectory = $ProjectPath
    $ProcessStartInfo.UseShellExecute = $false
    $ProcessStartInfo.CreateNoWindow = $true
    $ProcessStartInfo.RedirectStandardOutput = $false
    $ProcessStartInfo.RedirectStandardError = $false
    
    $Process = [System.Diagnostics.Process]::Start($ProcessStartInfo)
    
    if ($Process) {
        Write-Host "✓ Daemon launched successfully"
        Write-Host "  Process ID: $($Process.Id)"
        
        # Wait a moment for daemon to initialize
        Start-Sleep -Seconds 2
        
        # Verify daemon is still running and PID file was created
        if (Test-Path $PidFile) {
            Write-Host "✓ Daemon started and PID file created"
            Write-Host ""
            Write-Host "Dream Window is now running!"
            Write-Host "  - ComfyUI will start automatically"
            Write-Host "  - Generation will begin shortly"
            Write-Host "  - Check widget for live status"
            exit 0
        } else {
            Write-Host "WARNING: Daemon launched but PID file not found"
            Write-Host "         Daemon may have failed to start properly"
            Write-Host "         Check logs: logs\daemon.log"
            exit 1
        }
    } else {
        Write-Host "ERROR: Failed to launch daemon process"
        exit 1
    }
} catch {
    Write-Host "ERROR: Exception launching daemon: $_"
    Write-Host "       Check that Python and all dependencies are installed"
    exit 1
}

