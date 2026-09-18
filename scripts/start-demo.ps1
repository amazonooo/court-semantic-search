$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw 'Ollama is not installed. Install it from https://ollama.com/download, then run this script again.'
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python virtual environment.' }
}
& $venvPython -m pip install -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install Python dependencies.' }

try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null
} catch {
    Start-Process -FilePath $ollama.Source -ArgumentList 'serve' -WindowStyle Hidden
    $ready = $false
    for ($attempt = 0; $attempt -lt 15; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw 'Ollama did not start on port 11434.' }
}

$model = 'qwen3:4b'
$installed = & $ollama.Source list
if ($LASTEXITCODE -ne 0) { throw 'Could not list Ollama models.' }
if (-not ($installed | Select-String -SimpleMatch $model)) {
    Write-Host "Downloading $model for the first demo run..."
    & $ollama.Source pull $model
    if ($LASTEXITCODE -ne 0) { throw "Could not download $model." }
}

$env:COURT_PROVIDER = 'mock'
$env:LLM_PROVIDER = 'ollama'
$env:OLLAMA_MODEL = $model
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'

Write-Host 'Demo: http://127.0.0.1:8000/demo'
& $venvPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
