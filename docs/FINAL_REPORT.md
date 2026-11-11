# Soup-CLI (`v0.72.4`) Release Candidate QA Evaluation — Final Report

**Release Readiness Verdict:** **`DON'T SHIP`** (`3 Critical`, `4 High`, `1 Medium` verified defects across local CPU & Google Colab Tesla T4 GPU)

`soup-cli v0.72.4` is **not ready for production release**. While its CLI UX and positive-path SFT training loop are ergonomic, our grey-box audit and empirical Google Colab Tesla T4 GPU tests uncovered three release-blocking (`Critical`) silent failures:
1. **Silent Hyperparameter Replacement ([BUG-001](../bug_reports/BUG-001.md)):** None of the Pydantic config models (`SoupConfig`, `TrainingConfig`, `DataConfig`, `LoraConfig`) set `extra="forbid"`. Misspelled keys (`epchs: 99`, `lora_r: 4`, `taks`) and standard HuggingFace aliases (`learning_rate: 0.0`, `max_steps`, `seed`) are silently discarded with `exit_code = 0`, training models with unintended defaults (`lr=2e-5`, `r=64`, `epochs=3`).
2. **Broken Seed Reproducibility & Empty Compliance Receipt ([BUG-002](../bug_reports/BUG-002.md)):** `TrainingConfig` omits a `seed` field (`--repro-receipt` writes `"seeds": {}`), and `SFTTrainerWrapper.setup()` calls `get_peft_model()` **before** `TrainingArguments` sets `seed=42`. Consequently, LoRA `lora_A` weights are initialized from unseeded OS entropy on every run, causing two identical runs (`seed=42` vs `seed=42`) to diverge by **`global_l2_diff = 4.7295`**.
3. **Broken Loss Watchdog Safety Gate ([BUG-003](../bug_reports/BUG-003.md)):** `SoupTrainerCallback.on_log` checks `loss > self.watchdog_threshold`, which evaluates to `False` on `NaN` (`float('nan') > 3.0 is False`). Worse, when the watchdog *does* halt training at Step 1/15 (`lr: 1e15`, weights exploded to $2.26 \times 10^{16}$), `soup train` still saves the corrupted checkpoint, marks the run `"completed"`, prints `"Training Complete!"`, and **exits with code `0`**.

---

## 1. Summary Across the 5 QA Evaluation Areas

### Area 1: Pre-Test Strategy ([`./01_test_strategy.md`](./01_test_strategy.md))
Written strictly **before** test execution after grey-box inspection of `soup_cli` ([`../raw_logs/00_env_and_cli_help.log`](../raw_logs/00_env_and_cli_help.log)). Prioritized silent model corruption (`P0`) over unhandled crashes (`P1`) and UX errors (`P2`), mapping out 6 generative fuzzing classes, dual-run GPU reproducibility tolerances, and active silent-failure probes.

### Area 2: Generative Config & CLI Fuzzing ([`./02_config_fuzz_matrix.md`](./02_config_fuzz_matrix.md), [`../tests/test_config_fuzzing.py`](../tests/test_config_fuzzing.py))
Executed **90 deterministic + Hypothesis property-based test cases** across all 6 required input classes ([`../raw_logs/02_config_fuzzing.log`](../raw_logs/02_config_fuzzing.log), [`../raw_logs/02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl)):
- **Behavior Breakdown (90 cases):** **`30 Clear Error` (33.3%)**, **`5 Traceback` (5.6%)**, **`55 Silent Acceptance` (61.1%)**.
- Exposed unhandled `ZeroDivisionError` in `soup doctor` (`batch_size: 0`), unhandled `TypeError` / `ScannerError` in `load_config()` ([BUG-008](../bug_reports/BUG-008.md)), and complete schema validation bypass in `soup plan` ([BUG-006](../bug_reports/BUG-006.md)).

### Area 3: Seed Reproducibility on Tesla T4 GPU ([`./03_reproducibility_analysis.md`](./03_reproducibility_analysis.md), [`../tests/test_reproducibility.py`](../tests/test_reproducibility.py))
Executed 15-step SFT runs on Google Colab **NVIDIA Tesla T4 GPU** ([`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log), [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json)):
- **Identical Seed (`42` vs `42`):** Step-1 loss matches (`10.3557`) and max step-by-step loss diff is **`0.0006`** (`6.0e-4`), but saved adapter weights diverge by **`global_l2_diff = 4.7295`** (`max_abs_diff = 0.4617`). Per-tensor decomposition proved that `lora_A` diverges by `2.30–2.46` per layer ($\sqrt{\|A_1\|_2^2 + \|A_2\|_2^2}$, orthogonal unseeded Kaiming init), whereas zero-initialized `lora_B` diverges by only `0.021–0.043`.
- **Disconnected Seed (`42` vs `999`):** `training.seed: 999` is ignored; `--repro-receipt` outputs `"seeds": {}`; adapter $L_2$ diff is **`4.6256`**.

### Area 4: Silent Failures & Blind Spots ([`./04_silent_failures.md`](./04_silent_failures.md), [`../tests/test_silent_failures.py`](../tests/test_silent_failures.py))
Confirmed **9 distinct silent failure modes** across local CPU ([`../raw_logs/03_silent_failures_local.log`](../raw_logs/03_silent_failures_local.log)) and Tesla T4 GPU, including `lr: 1e-30` exiting `0` with `lora_B_global_l2_norm == 0.0` (100% untrained adapter), `soup data validate` and `soup train --dry-run` exiting `0` on `0 train samples` ([BUG-004](../bug_reports/BUG-004.md)), and `SoupTrainerCallback` overwriting the final step metric in SQLite and Rich TUI with `Loss: 0.0000, LR: 0.00e+00, GPU: 0.0/14.6 GB`. Documented 6 explicit test-suite blind spots (7B+ long-horizon forgetting, `{% generation %}` mask alignment, multi-GPU FSDP2/DDP, NF4 quantization drift, non-SFT trainers, semantic leakage).

### Area 5: Defect Triage & Hardware Gate Readiness (`../bug_reports/BUG-001..008.md`)

| Bug ID | Severity | Classification | Summary of Verified Defect | Raw Log Evidence |
| :--- | :---: | :---: | :--- | :--- |
| [BUG-001](../bug_reports/BUG-001.md) | **Critical** | `Silent Acceptance` | Pydantic schemas lack `extra="forbid"`; typos (`epchs`, `lora_r`) & `learning_rate` silently ignored (`rc=0`). | [`03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) |
| [BUG-002](../bug_reports/BUG-002.md) | **Critical** | `Silent Acceptance` | `TrainingConfig` lacks `seed`; `get_peft_model` runs unseeded before `TrainingArguments` ($L_2=4.73$, `"seeds": {}`). | [`03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) |
| [BUG-003](../bug_reports/BUG-003.md) | **Critical** | `Silent Acceptance` | Loss watchdog misses `NaN`; watchdog abort on `lr: 1e15` saves $10^{16}$ weights and exits `0` (`Training Complete!`). | [`03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) |
| [BUG-004](../bug_reports/BUG-004.md) | **High** | `Silent Acceptance` | `soup data validate` & `soup train --dry-run` exit `0` on `0 train samples` / corrupted JSONL / wrong columns. | [`03_silent_failures_cases.jsonl`](../raw_logs/03_silent_failures_cases.jsonl) |
| [BUG-005](../bug_reports/BUG-005.md) | **High** | `Silent / Traceback` | Unbounded `lora.r: -8`, `lr: .inf`, `epochs: true`; `batch_size: 0` -> `ZeroDivisionError`; `lr: 1e-30` -> zero update. | [`02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl) |
| [BUG-006](../bug_reports/BUG-006.md) | **High** | `Silent Acceptance` | `soup plan` bypasses `SoupConfig` validation, writing `soup.tfstate` (`rc=0`) on invalid tasks and negative LR. | [`02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl) |
| [BUG-007](../bug_reports/BUG-007.md) | **High** | `Clear Error` (Bug) | False-positive Hardware-Fit gate assumes `7.0`B (`14 GB`) for `<10 MB` models on 16 GB T4; default `bf16=True` on T4. | [`03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) |
| [BUG-008](../bug_reports/BUG-008.md) | **Medium** | `Traceback` | Unhandled `TypeError` & `ScannerError` in `load_config()` when YAML root is a list/scalar or contains tabs. | [`02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl) |

---

## 2. What Surprised Us and What Still Concerns Us

**What Surprised Us:** Two empirical findings on the Tesla T4 GPU were genuinely surprising. First, in **Exp 1A (`seed=42` vs `seed=42`)**, the step-by-step loss curves matched to within `0.0006` (`10.3557 -> 10.3570` vs `10.3571`), which initially looked like near-perfect GPU determinism—until we inspected `adapter_model.safetensors` and discovered a massive **`4.7295` $L_2$ weight divergence**. Tracing `soup_cli/trainer/sft.py` revealed why: `get_peft_model()` runs at line 949 *before* `TrainingArguments` sets `seed=42` at line 471, so `lora_A` was initialized completely unseeded while the DataLoader batch order *was* seeded! Anyone checking only loss curves would have missed that the saved adapters were in orthogonal subspaces. Second, in **SF-GPU-2 (`lr: 1e15`)**, we saw the `Loss Watchdog` banner fire at Step 1/15 (`Epoch 0.2/3`) and print `Training will stop.`—and then watched `soup train` immediately print `Training Complete!`, mark the SQLite run `"completed"`, save adapter weights exploded to $2.26 \times 10^{16}$, and exit with code `0`.

**What Still Concerns Us:** Beyond the 8 filed bugs, we remain concerned by the **illusion of validation** across `soup-cli`'s pre-flight tooling (`soup config validate`, `soup data validate`, `soup train --dry-run`, `soup plan`, and `--repro-receipt`). Every single one of those safety/compliance commands returned green checkmarks (`exit 0`) on broken inputs during our audit, while the actual Hardware-Fit gate blocked a 4 MB model on a 16 GB GPU ([BUG-007](../bug_reports/BUG-007.md)). Additionally, TRL's warning during T4 training (`return_assistant_tokens_mask==True but chat template does not contain {% generation %} keyword`) suggests that `train_on_inputs: false` may silently fall back to computing loss over user prompt tokens when using built-in templates like `chatml`.

---

## 3. AI Tools Disclosure

We used **Antigravity (Google DeepMind agentic coding assistant)** throughout this evaluation, governed by 3 custom workspace skills (`.agent/skills/hypothesis-cli-fuzzer`, `ml-silent-failure-auditor`, `strict-qa-bug-reporter`) and logged in full in [`./ai_usage_log.md`](./ai_usage_log.md):
- **What AI Generated:** Grey-box source inspection script (`../scripts/inspect_soup.py`), pre-test strategy ([`./01_test_strategy.md`](./01_test_strategy.md)), Hypothesis/pytest suites ([`../tests/test_config_fuzzing.py`](../tests/test_config_fuzzing.py), [`../tests/test_silent_failures.py`](../tests/test_silent_failures.py), [`../tests/test_reproducibility.py`](../tests/test_reproducibility.py)), Colab T4 runner (`../scripts/run_colab_t4_experiments.py`), and structured documentation.
- **What AI Got Wrong & Human/Empirical Execution Caught:** (1) Initial AI test drafts assumed `soup-cli` printed tracebacks by default (`--verbose` is required), assumed `_write_repro_receipt` took 2 args instead of 3, and assumed `runs` table had a `started_at` column (`sqlite3.OperationalError: no such column: started_at` caught on Colab T4; fixed to `created_at`). (2) Real Colab T4 execution exposed three hardware/runtime gates that static code reading missed: Colab installing `soup-cli 0.77.2` unless pinned to `==0.72.4`, the false-positive 14 GB Hardware-Fit VRAM gate requiring `--allow-oom-attempt`, default `bf16=True` requiring `auto_mixed_precision: true` on Tesla T4 (`cc=7.5`), and `tiny-random-LlamaForCausalLM` requiring explicit `data.chat_template: chatml`. All failed and exploratory logs are preserved untouched in `../raw_logs/`.
