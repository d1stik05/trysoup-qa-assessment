# `soup-cli` (`v0.72.4`) — Comprehensive QA & ML Systems Evaluation

**Release Readiness Verdict:** **`DON'T SHIP`** (`3 Critical`, `4 High`, `1 Medium` verified defects)

This repository contains the complete grey-box and empirical Google Colab **NVIDIA Tesla T4 GPU** QA evaluation of `soup-cli v0.72.4` (`pip install "soup-cli[train]==0.72.4"`). All claims, bug reports, and reproducibility analyses are backed by untouched raw execution logs in [`./raw_logs/`](./raw_logs/).

---

## Quick Start (One-Command Test Execution)

### 1. Install Dependencies
```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Run the Full Automated QA Suite (`99 passed`)
```bash
# Bash (Linux / macOS / Git Bash):
bash run_tests.sh

# PowerShell (Windows):
.\run_tests.ps1

# Or directly via pytest:
pytest
```

### 3. Re-Run Google Colab Tesla T4 GPU Experiments (Optional)
The untouched Colab T4 GPU logs ([`./raw_logs/03_colab_t4_experiments.log`](./raw_logs/03_colab_t4_experiments.log) and [`./raw_logs/03_colab_t4_results.json`](./raw_logs/03_colab_t4_results.json)) are already included in `raw_logs/` and verified by [`./tests/test_reproducibility.py`](./tests/test_reproducibility.py). To re-execute on a live Colab T4 GPU:
```bash
pip install "soup-cli[train]==0.72.4"
python scripts/run_colab_t4_experiments.py
```

---

## Repository Structure

```text
.
├── .agent/skills/                         # Custom QA enforcement skills (Hypothesis, Silent Failures, Bug Reporting)
├── docs/
│   ├── 01_test_strategy.md                # Pre-test 1-page QA strategy (written strictly BEFORE running tests)
│   ├── 02_config_fuzz_matrix.md           # 90-case generative Config & CLI Fuzzing Matrix (Clear Error / Traceback / Silent Acceptance)
│   ├── 03_reproducibility_analysis.md     # Tesla T4 GPU & CPU seed reproducibility & per-tensor LoRA L2 divergence analysis
│   ├── 04_silent_failures.md              # Catalog of 9 silent failure modes + explicit test-suite blind spots
│   ├── FINAL_REPORT.md                    # <= 2-page Executive Release Readiness Verdict ("DON'T SHIP") + AI disclosure
│   └── ai_usage_log.md                    # Complete transparency log (AI generated vs. human/Colab T4 verified)
├── bug_reports/
│   ├── BUG-001.md                         # [Critical] Pydantic schemas lack extra="forbid" -> silent hyperparameter replacement
│   ├── BUG-002.md                         # [Critical] Missing TrainingConfig.seed & unseeded LoRA A init (L2=4.73, seeds={})
│   ├── BUG-003.md                         # [Critical] Loss watchdog ignores NaN & exits 0 ("Training Complete!") on 1e15 LR abort
│   ├── BUG-004.md                         # [High] soup data validate & soup train --dry-run exit 0 on 0 train samples
│   ├── BUG-005.md                         # [High] Unbounded schema fields, batch_size:0 ZeroDivisionError, lr:1e-30 zero update
│   ├── BUG-006.md                         # [High] soup plan bypasses SoupConfig validation and writes soup.tfstate (exit 0)
│   ├── BUG-007.md                         # [High] False-positive 14 GB Hardware-Fit gate blocks <10 MB models on 16 GB T4 GPU
│   └── BUG-008.md                         # [Medium] Unhandled TypeError & ScannerError tracebacks in load_config()
├── configs/                               # Minimal standalone YAML reproducers for BUG-001 through BUG-008
├── tests/
│   ├── test_config_fuzzing.py             # 90 pytest + Hypothesis fuzzing tests across all 6 input classes
│   ├── test_silent_failures.py            # 6 grey-box & CLI silent-failure tests
│   └── test_reproducibility.py            # 3 seed-wiring, CPU bitwise determinism, and Colab T4 GPU metric parser tests
├── scripts/
│   ├── inspect_soup.py                    # Phase 0 grey-box reconnaissance & CLI help dumper
│   └── run_colab_t4_experiments.py        # Self-contained Google Colab Tesla T4 GPU experiment harness
├── raw_logs/                              # Untouched raw execution logs (including preserved exploratory/failed runs)
├── run_tests.sh                           # One-command Bash test runner
├── run_tests.ps1                          # One-command PowerShell test runner
├── requirements.txt                       # Pinned environment dependencies
└── pyproject.toml                         # Pytest & project configuration
```

---

## Summary of Key Empirical Findings

| Evaluation Phase | Test Suite / Script | Result | Key Empirical Takeaway |
| :--- | :--- | :---: | :--- |
| **Phase 1: Strategy** | [`./docs/01_test_strategy.md`](./docs/01_test_strategy.md) | Complete | Prioritized P0 silent model corruption over P1 tracebacks and P2 UX errors prior to testing. |
| **Phase 2: Config & CLI Fuzzing** | [`./tests/test_config_fuzzing.py`](./tests/test_config_fuzzing.py) | `90 passed` | **30 Clear Error (33.3%)**, **5 Traceback (5.6%)**, **55 Silent Acceptance (61.1%)**. |
| **Phase 3: Reproducibility (T4 GPU)** | [`./tests/test_reproducibility.py`](./tests/test_reproducibility.py) | `3 passed` | Loss curve diff $\le 0.0006$, but `lora_A` is initialized unseeded before `TrainingArguments` (**$L_2 = 4.7295$**). |
| **Phase 3/4: Silent Failures** | [`./tests/test_silent_failures.py`](./tests/test_silent_failures.py) | `6 passed` | Confirmed `rc=0` on ignored typos (`lora_r`, `learning_rate`), `lr=1e15` watchdog abort, and `lr=1e-30` zero update. |
| **Phase 4: Final Verdict** | [`./docs/FINAL_REPORT.md`](./docs/FINAL_REPORT.md) | **`DON'T SHIP`** | 3 Critical, 4 High, 1 Medium defects documented in [`./bug_reports/`](./bug_reports/). |
