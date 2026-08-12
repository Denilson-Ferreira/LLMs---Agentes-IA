$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $projectDir "..\..\..")).Path
$localPython = Join-Path $projectDir ".venv\Scripts\python.exe"
$repoPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $localPython) {
    $python = $localPython
} elseif (Test-Path -LiteralPath $repoPython) {
    $python = $repoPython
} else {
    Write-Error "Ambiente virtual não encontrado. Consulte o README.md para instalar as dependências."
}

Push-Location $projectDir
try {
    & $python .\main.py
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $exitCode
