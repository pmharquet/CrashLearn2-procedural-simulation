$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Installez les dépendances dans .venv avant de construire l'exécutable."
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m pip install 'pyinstaller>=6.10,<7'
    if ($LASTEXITCODE -ne 0) { throw 'Installation de PyInstaller échouée.' }
    & $pythonPath -m PyInstaller --noconfirm CrashLearn.spec
    if ($LASTEXITCODE -ne 0) { throw 'Construction échouée.' }
    Write-Host "Exécutable : $projectRoot\dist\CrashLearn\CrashLearn.exe"
} finally {
    Pop-Location
}
