$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "chrono-compositor"
$Conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"

if (!(Test-Path -LiteralPath $Conda)) {
  throw "Conda not found at $Conda"
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONNOUSERSITE = "1"

$existing = & $Conda env list | Select-String -Pattern "^\s*$EnvName\s"
if ($existing) {
  Write-Host "Conda env '$EnvName' already exists. Installing/updating dependencies..."
  & $Conda env update -n $EnvName -f "$ProjectRoot\environment.yml" --prune
} else {
  Write-Host "Creating Conda env '$EnvName'..."
  & $Conda env create -f "$ProjectRoot\environment.yml"
}

Write-Host "Downloading SAM2.1 checkpoint if needed..."
& $Conda run -n $EnvName python "$ProjectRoot\scripts\download_models.py"

Write-Host "Smoke testing environment..."
& $Conda run -n $EnvName python "$ProjectRoot\scripts\smoke_test.py"

Write-Host "Setup complete."
