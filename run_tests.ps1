# One-command PowerShell QA evaluation test runner for soup-cli (v0.72.4)
$ErrorActionPreference = "Stop"

if (-not (Test-Path "raw_logs")) {
    New-Item -ItemType Directory -Path "raw_logs" | Out-Null
}

$PytestExe = ".\.venv\Scripts\pytest.exe"
if (-not (Test-Path $PytestExe)) {
    $PytestExe = "pytest"
}

Write-Host "=================================================================="
Write-Host "Running Soup-CLI v0.72.4 Full QA Suite (99 Tests)"
Write-Host "=================================================================="

& $PytestExe tests/test_config_fuzzing.py tests/test_silent_failures.py tests/test_reproducibility.py -v --tb=short 2>&1 | Tee-Object -FilePath "raw_logs\full_test_suite_run.log"
exit $LASTEXITCODE
