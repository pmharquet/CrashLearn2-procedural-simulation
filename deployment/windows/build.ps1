$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Créez .venv et installez le projet avec pip install -e ".[freeze]".'
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m PyInstaller --noconfirm deployment/windows/CrashLearn.spec
    if ($LASTEXITCODE -ne 0) { throw 'Construction échouée.' }
    Write-Host "Exécutable : $projectRoot/dist/CrashLearn/CrashLearn.exe"
} finally {
    Pop-Location
}
