param(
  [ValidateSet("yes", "no")]
  [string]$Localhost = "yes",
  [int]$Port = $(if ($env:CHRONO_PORT) { [int]$env:CHRONO_PORT } else { 8766 })
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "chrono-compositor"
$Conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$HostAddress = if ($Localhost -eq "yes") { "127.0.0.1" } else { "0.0.0.0" }

if (!(Test-Path -LiteralPath $Conda)) {
  throw "Conda not found at $Conda"
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONNOUSERSITE = "1"
Write-Host "Starting Chrono Compositor on http://$HostAddress`:$Port"
& $Conda run -n $EnvName python -m uvicorn app.main:app --host $HostAddress --port $Port
