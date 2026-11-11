#!/usr/bin/env bash
# One-command QA evaluation test runner for soup-cli (v0.72.4)
# Runs all 3 pytest suites (Config Fuzzing, Silent Failures, Reproducibility)
# and preserves unfiltered execution logs in raw_logs/.

set -euo pipefail

mkdir -p raw_logs

if [ -x ".venv/Scripts/pytest.exe" ]; then
    PYTEST=".venv/Scripts/pytest.exe"
elif [ -x ".venv/bin/pytest" ]; then
    PYTEST=".venv/bin/pytest"
else
    PYTEST="pytest"
fi

echo "=================================================================="
echo "Running Soup-CLI v0.72.4 Full QA Suite (99 Tests)"
echo "=================================================================="

"$PYTEST" tests/test_config_fuzzing.py tests/test_silent_failures.py tests/test_reproducibility.py -v --tb=short 2>&1 | tee raw_logs/full_test_suite_run.log
