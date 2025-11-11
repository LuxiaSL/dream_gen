# Dream Window - Rainmeter Installation Script
# Automatically copies the skin to Rainmeter's Skins directory

Write-Host "🎨 Dream Window Rainmeter Installer" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Get Rainmeter skins directory
$rainmeterSkins = "$env:USERPROFILE\Documents\Rainmeter\Skins"

# Check if Rainmeter is installed
if (!(Test-Path $rainmeterSkins)) {
    Write-Host "❌ Rainmeter not found!" -ForegroundColor Red
    Write-Host "   Install from: https://www.rainmeter.net/" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Found Rainmeter at: $rainmeterSkins" -ForegroundColor Green
Write-Host ""

# Get current project directory
$projectDir = Get-Location
$sourceDir = Join-Path $projectDir "rainmeter_skin"
$targetDir = Join-Path $rainmeterSkins "DreamWindow"

# Check if source exists
if (!(Test-Path $sourceDir)) {
    Write-Host "❌ Source directory not found: $sourceDir" -ForegroundColor Red
    exit 1
}

# Check if already installed
if (Test-Path $targetDir) {
    Write-Host "⚠️  DreamWindow skin already exists at:" -ForegroundColor Yellow
    Write-Host "   $targetDir" -ForegroundColor Gray
    Write-Host ""
    $overwrite = Read-Host "   Overwrite? (y/N)"
    
    if ($overwrite -ne "y" -and $overwrite -ne "Y") {
        Write-Host "   Installation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Backup existing if it exists
    if (Test-Path $targetDir) {
        $backupDir = "$targetDir.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Write-Host "   Backing up to: $backupDir" -ForegroundColor Gray
        Move-Item $targetDir $backupDir
    }
}

# Copy files
Write-Host "📦 Installing DreamWindow skin..." -ForegroundColor Cyan

try {
    # Copy entire directory structure
    Copy-Item -Path $sourceDir -Destination $targetDir -Recurse -Force
    
    # Update ProjectPath in Variables.inc to current location
    $variablesFile = Join-Path $targetDir "@Resources\Variables.inc"
    if (Test-Path $variablesFile) {
        $content = Get-Content $variablesFile -Raw
        $content = $content -replace 'ProjectPath=.*', "ProjectPath=$projectDir"
        Set-Content -Path $variablesFile -Value $content -NoNewline
        Write-Host "   ✓ Updated ProjectPath to: $projectDir" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "✅ Installation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 Installed to: $targetDir" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Right-click Rainmeter tray icon" -ForegroundColor White
    Write-Host "  2. Click 'Manage'" -ForegroundColor White
    Write-Host "  3. Find 'DreamWindow' in left panel" -ForegroundColor White
    Write-Host "  4. Click 'DreamWindow.ini' → 'Load'" -ForegroundColor White
    Write-Host ""
    Write-Host "Customize:" -ForegroundColor Cyan
    Write-Host "  Edit: $targetDir\@Resources\Variables.inc" -ForegroundColor Gray
    Write-Host ""
    
    # Ask if user wants to open Rainmeter
    $openRainmeter = Read-Host "Open Rainmeter Manager now? (Y/n)"
    if ($openRainmeter -ne "n" -and $openRainmeter -ne "N") {
        Start-Process "rainmeter" -ArgumentList "!Manage"
    }
    
} catch {
    Write-Host "❌ Installation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Press Enter to exit..." -ForegroundColor Gray
Read-Host

