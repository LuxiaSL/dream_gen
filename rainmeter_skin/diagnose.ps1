# Dream Window v2.1 - Diagnostic Script
# Run this to check if everything is configured correctly

$projectPath = "C:\Users\luxia\Documents\projects\dream_gen"
$rainmeterPath = "$env:USERPROFILE\Documents\Rainmeter\Skins\DreamWindow"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DREAM WINDOW v2.1 DIAGNOSTICS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check 1: Project files exist
Write-Host "1. Checking project files..." -ForegroundColor Yellow

$checks = @{
    "daemon.py" = "$projectPath\daemon.py"
    "daemon_control.py" = "$projectPath\daemon_control.py"
    "status.json" = "$projectPath\output\status.json"
    "config.yaml" = "$projectPath\backend\config.yaml"
}

foreach ($name in $checks.Keys) {
    $path = $checks[$name]
    if (Test-Path $path) {
        Write-Host "  ✓ $name exists" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $name MISSING" -ForegroundColor Red
    }
}

Write-Host ""

# Check 2: Rainmeter skin
Write-Host "2. Checking Rainmeter skin..." -ForegroundColor Yellow

if (Test-Path "$rainmeterPath\DreamWindow.ini") {
    Write-Host "  ✓ DreamWindow.ini exists" -ForegroundColor Green
    
    # Check if it's v2.1
    $content = Get-Content "$rainmeterPath\DreamWindow.ini" -Raw
    if ($content -match "Version=2\.1") {
        Write-Host "  ✓ Version 2.1 detected" -ForegroundColor Green
    } elseif ($content -match "Version") {
        Write-Host "  ⚠  Older version detected - update needed" -ForegroundColor Yellow
    }
    
    # Check for OnCloseAction
    if ($content -match "OnCloseAction") {
        Write-Host "  ✓ Shutdown support enabled" -ForegroundColor Green
    } else {
        Write-Host "  ✗ No shutdown support found" -ForegroundColor Red
    }
} else {
    Write-Host "  ✗ DreamWindow.ini NOT FOUND" -ForegroundColor Red
    Write-Host "    Expected at: $rainmeterPath\DreamWindow.ini" -ForegroundColor Gray
}

Write-Host ""

# Check 3: status.json validity
Write-Host "3. Checking status.json..." -ForegroundColor Yellow

$statusPath = "$projectPath\output\status.json"
if (Test-Path $statusPath) {
    try {
        $status = Get-Content $statusPath -Raw | ConvertFrom-Json
        Write-Host "  ✓ Valid JSON" -ForegroundColor Green
        
        # Check key fields
        $fields = @("frame_number", "is_buffering", "comfyui_status", "daemon_status")
        foreach ($field in $fields) {
            if ($null -ne $status.$field) {
                Write-Host "  ✓ $field = $($status.$field)" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $field MISSING" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  ✗ Invalid JSON!" -ForegroundColor Red
        Write-Host "    Error: $_" -ForegroundColor Gray
    }
} else {
    Write-Host "  ✗ status.json not found" -ForegroundColor Red
    Write-Host "    Daemon may not be running" -ForegroundColor Gray
}

Write-Host ""

# Check 4: Daemon status
Write-Host "4. Checking daemon status..." -ForegroundColor Yellow

Push-Location $projectPath
try {
    $result = python daemon_control.py status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Daemon is running" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Daemon is not running" -ForegroundColor Red
    }
} catch {
    Write-Host "  ✗ Could not check status" -ForegroundColor Red
    Write-Host "    Error: $_" -ForegroundColor Gray
}
Pop-Location

Write-Host ""

# Check 5: Python environment
Write-Host "5. Checking Python environment..." -ForegroundColor Yellow

if (Test-Path "$projectPath\.venv") {
    Write-Host "  ✓ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ No virtual environment found" -ForegroundColor Red
}

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found in PATH" -ForegroundColor Red
}

Write-Host ""

# Check 6: Rainmeter running
Write-Host "6. Checking Rainmeter..." -ForegroundColor Yellow

$rainmeterProcess = Get-Process Rainmeter -ErrorAction SilentlyContinue
if ($rainmeterProcess) {
    Write-Host "  ✓ Rainmeter is running (PID: $($rainmeterProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "  ✗ Rainmeter is not running" -ForegroundColor Red
}

Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "If all checks passed:" -ForegroundColor Green
Write-Host "  • Refresh your Rainmeter skin" -ForegroundColor White
Write-Host "  • Loading overlay should appear" -ForegroundColor White
Write-Host "  • Status bar should show live data" -ForegroundColor White
Write-Host "  • Overlay should disappear when buffering completes" -ForegroundColor White

Write-Host ""
Write-Host "If checks failed:" -ForegroundColor Red
Write-Host "  • Fix the issues shown above" -ForegroundColor White
Write-Host "  • Re-run this script to verify" -ForegroundColor White
Write-Host "  • Check Rainmeter log for errors" -ForegroundColor White

Write-Host ""
Write-Host "To view detailed logs:" -ForegroundColor Yellow
Write-Host "  • Right-click Rainmeter tray icon → Manage → Open log" -ForegroundColor White

Write-Host ""
Write-Host "To test shutdown:" -ForegroundColor Yellow
Write-Host "  • Right-click skin → Unload skin" -ForegroundColor White
Write-Host "  • Check if daemon stops (run: python daemon_control.py status)" -ForegroundColor White

Write-Host ""
Write-Host "Press Enter to exit..." -ForegroundColor Gray
Read-Host