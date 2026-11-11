# 04. Silent Failures Audit & Test Suite Blind Spots

## 1. Executive Summary

Following the `.agent/skills/ml-silent-failure-auditor/SKILL.md` protocol, we actively probed `soup-cli v0.72.4` across both local CPU ([`../tests/test_silent_failures.py`](../tests/test_silent_failures.py)) and Google Colab **NVIDIA Tesla T4 GPU** (`../scripts/run_colab_t4_experiments.py`) for **Silent Failures**—scenarios where the CLI exits with status `0` (`"success"`) or reports `"Config valid. Ready to train!"` / `"Training Complete!"` while silently ignoring user configuration, corrupting model weights, bypassing safety watchdogs, or training on zero samples.

Every finding below is backed by untouched raw execution logs in:
- [`../raw_logs/03_silent_failures_local.log`](../raw_logs/03_silent_failures_local.log) & [`../raw_logs/03_silent_failures_cases.jsonl`](../raw_logs/03_silent_failures_cases.jsonl)
- [`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) & [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json)
- [`../raw_logs/02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl)

---

## 2. Catalog of Verified Silent Failures

| ID | Failure Mode | Command / Component | Exit Code | Actual Silent Behavior | Bug Report |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **SF-01 / SF-GPU-1** | Misspelled / Alias Hyperparameters | `soup config validate`, `soup train` | **`0`** | `learning_rate: 0.0`, `epchs: 99`, `lora_r: 4` are silently dropped; trains 3 epochs at `lr=2e-5`, `r=64`. | [BUG-001](../bug_reports/BUG-001.md) |
| **SF-02 / Exp 1B** | Ignored `training.seed` & Empty Compliance Receipt | `soup train --repro-receipt` | **`0`** | `training.seed: 999` is dropped; `--repro-receipt` writes `"seeds": {}`; `lora_A` initialized unseeded ($L_2 = 4.73$). | [BUG-002](../bug_reports/BUG-002.md) |
| **SF-03** | 0-Byte Empty or 100% Corrupted JSONL Dataset | `soup data validate` | **`0`** | Prints `Total rows: 0` / `Valid rows: 0 / 5` on malformed JSONL and exits `0` instead of non-zero. | [BUG-004](../bug_reports/BUG-004.md) |
| **SF-04 / SF-GPU-4** | `--dry-run` Passes on 0 Samples & Mismatched Columns | `soup train --dry-run` | **`0`** | Prints `Data OK: 0 train samples` and `Config valid. Ready to train!`, then crashes during full `soup train`. | [BUG-004](../bug_reports/BUG-004.md) |
| **SF-05** | Single-Row Dataset Truncated to `0` Train Rows | `soup train --dry-run` (`val_split: 0.2`) | **`0`** | `int(1 * 0.8) == 0` puts 0 rows in `train` and 1 row in `val`, exiting `0` (`Data OK: 0 train samples`). | [BUG-004](../bug_reports/BUG-004.md) |
| **SF-06A** | Loss Watchdog Fails to Detect `NaN` Loss | `SoupTrainerCallback.on_log` | **`0`** | `loss > self.watchdog_threshold` evaluates `float('nan') > 3.0` as `False`, resetting `_watchdog_counter = 0`. | [BUG-003](../bug_reports/BUG-003.md) |
| **SF-06B / SF-GPU-2** | Exploding LR (`1e15`) + Watchdog Abort Exits `0` | `soup train` (`loss_watchdog: true`) | **`0`** | Watchdog stops training at Step 1/15 with weights exploded to $2.26 \times 10^{16}$, yet prints `Training Complete!` and exits `0`. | [BUG-003](../bug_reports/BUG-003.md) |
| **SF-07 / SF-GPU-3** | Degenerate `lr: 1e-30` Produces 100% Untrained Adapter | `soup train` (`lr: 1e-30`) | **`0`** | Runs all 15 steps, prints `Training Complete!`, and saves adapter with `lora_B_global_l2_norm == 0.0` ($\Delta W = 0$). | [BUG-005](../bug_reports/BUG-005.md) |
| **SF-08** | `soup plan` Bypasses `SoupConfig` Validation | `soup plan` | **`0`** | Accepts `task: "completely_invalid_task_name"`, `epochs: -100`, `lr: -5.0`, and writes `soup.tfstate` (`exit 0`). | [BUG-006](../bug_reports/BUG-006.md) |
| **SF-09** | Corrupted Final Step Telemetry (`loss=0.0`, `GPU=0.0 GB`) | `SoupTrainerCallback.on_log` | **`0`** | Appends phantom final step metric (`loss: 0.0, lr: 0.0, grad_norm: 0.0`) to SQLite `metrics` and Rich live panel. | [BUG-003](../bug_reports/BUG-003.md) |

---

## 3. Deep-Dive Analysis & Empirical Log Evidence

### 3.1 SF-01 / SF-GPU-1: Silently Ignored Misspelled & Alias Hyperparameters
- **Root Cause (`soup_cli/config/schema.py`):** None of `SoupConfig`, `DataConfig`, `TrainingConfig`, `LoraConfig`, or `EvalConfig` configure `model_config = ConfigDict(extra="forbid")`. Common HuggingFace aliases (`learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `max_steps`, `seed`) and typos (`epchs`, `lora_r`, `taks`) are silently discarded.
- **Colab T4 GPU Evidence ([`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) lines 341–406, [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) lines 1025–1211):**
  When `soup train` was invoked with `learning_rate: 0.0`, `epchs: 99`, and `lora_r: 4`, it exited `0` (`run_20261111_112103_9939e658`) after running **3 epochs** (15 steps) at default **`learning_rate: 2e-05`** and **`LoRA: r=64, alpha=16`** (`8,192 trainable` parameters instead of `512`).

### 3.2 SF-06A & SF-06B / SF-GPU-2: Dual Failure in Loss Watchdog (`NaN` Blindness & Exit `0` on Abort)
- **Part A — `NaN` Blindness (`soup_cli/monitoring/callback.py:233`):**
  ```python
  if self.loss_watchdog and loss > 0:
      if loss > self.watchdog_threshold:
          self._watchdog_counter += 1
  ```
  In IEEE 754 floating-point arithmetic, `float("nan") > 0` and `float("nan") > 3.0` both evaluate to **`False`**. As proven in `test_sf06_loss_watchdog_ignores_nan_loss_due_to_ieee754_comparison` ([`../raw_logs/03_silent_failures_cases.jsonl`](../raw_logs/03_silent_failures_cases.jsonl) line 6), feeding 5 consecutive steps of `loss: float("nan")` leaves `_watchdog_counter == 0` and `should_training_stop == False`.
- **Part B — Exit Code `0` & `"Training Complete!"` After Watchdog Abort on T4 GPU ([`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) lines 408–466, [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) lines 1212–1258):**
  When configured with `lr: 1e15` and `loss_watchdog: true` (`threshold: 3.0, patience: 1`), Step 1 took an optimizer step with `learning_rate: 1000000000000000.0`, exploding the adapter weights to:
  - `lora_A_global_l2_norm = 32,834,075,941,553.297` ($3.28 \times 10^{13}$)
  - `lora_B_global_l2_norm = 22,604,958,881,415,044.0` ($2.26 \times 10^{16}$)
  The watchdog fired at Step 1/15 (`Epoch 0.2/3, 7%`): `Loss watchdog triggered! Loss 10.3557 exceeded threshold 3.0... Training will stop.` However, `soup train` never checked whether the watchdog aborted training—it saved the corrupted $10^{16}$-norm adapter checkpoint, marked SQLite run `run_20261111_112127_99d38e4b` as `"status": "completed"`, printed `Training Complete!`, and **exited with code `0`**!

### 3.3 SF-07 / SF-GPU-3: Degenerate Learning Rate (`lr: 1e-30`) Produces 100% Zero Adapter Update
- **Colab T4 GPU Evidence ([`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) lines 468–533, [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) lines 1259–1445):**
  In FP16/FP32 arithmetic, $10^{-30}$ underflows below machine epsilon relative to the parameter scale. `soup train` ran all 15 steps (`run_20261111_112150_89e897ba`), printed `Training Complete!`, and exited `0`. Inspecting `adapter_model.safetensors` confirmed **`lora_B_global_l2_norm = 0.0`** across all 4 `lora_B` tensors. Because $\Delta W = \frac{\alpha}{r} B A$ and $B = \mathbf{0}$, the trained adapter is a mathematical no-op ($\Delta W = \mathbf{0}$), yet no warning is emitted when `lora_B` remains all zeros after training.

### 3.4 SF-03, SF-04, SF-05 & SF-GPU-4: Zero-Sample Datasets & `--dry-run` Validation Gaps
- **Local & GPU Evidence ([`../raw_logs/03_silent_failures_cases.jsonl`](../raw_logs/03_silent_failures_cases.jsonl) lines 3–5, [`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) lines 535–659):**
  - `load_raw_data()` (`soup_cli/data/loader.py:182`) catches `json.JSONDecodeError` and skips corrupted lines. When all 5 lines of a JSONL file are corrupted, `soup data validate` prints `Valid rows: 0 / 5` and exits `0`.
  - When a JSONL file has `{"question": "...", "answer": "..."}` rows under `format: chatml`, `soup train --dry-run` prints `Data OK: 0 train samples` and `Config valid. Ready to train!` (`exit 0`), and also skips checking whether the model tokenizer has a `chat_template`. When the user then runs full `soup train`, it loads the model onto the GPU and crashes mid-setup with `ValueError: Column name ['messages'] not in the dataset. Current columns in the dataset: []` (`soup_cli/trainer/sft.py:342`).
  - When a 1-row JSONL file is split with default `val_split: 0.2`, `split_idx = int(1 * 0.8) = 0`, producing `0` train samples and `1` val sample while exiting `0`.

### 3.5 SF-09: Corrupted Final Step Telemetry in SQLite & TUI (`loss: 0.0`, `GPU: 0.0/14.6 GB`)
- **Colab T4 GPU Evidence ([`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) lines 166–175, 1193–1201):**
  At step 15 of every completed run, HuggingFace `Trainer` emits a final summary dictionary `{'train_runtime': ..., 'train_loss': 10.361, 'epoch': 3.0}` that does not contain the key `'loss'`. `SoupTrainerCallback.on_log` (`soup_cli/monitoring/callback.py:212`) executes `loss = logs.get("loss", 0.0)` and logs a second step-15 row into SQLite `metrics` with `"loss": 0.0, "lr": 0.0, "grad_norm": 0.0`, overwriting the Rich live panel to display `Loss: 0.0000  LR: 0.00e+00  GPU: 0.0/14.6 GB`.

---

## 4. Blind Spots: What Our Test Suite Would NOT Catch

In accordance with strict QA engineering transparency, we explicitly document the **blind spots** that our 99 automated local tests and 10 Google Colab T4 GPU probes do **not** catch:

1. **Long-Horizon Semantic Degradation & Catastrophic Forgetting on 7B–70B Models:**
   Our T4 GPU experiments used a 1.03M-parameter synthetic Llama model (`trl-internal-testing/tiny-random-LlamaForCausalLM`) over 15 steps to isolate numerical determinism, gradient flow, and adapter weight norms. Our suite will **not** catch subtle representation collapse, catastrophic forgetting, or EOS-token generation loops that only manifest after 1,000+ steps on real 7B/8B/70B pretrained weights.
2. **Subtle Chat-Template Token Masking & Label Alignment Bugs:**
   During T4 runs, TRL emitted the warning: `return_assistant_tokens_mask==True but chat template does not contain {% generation %} keyword` ([`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) line 110). While training succeeded and updated weights, our tests do not verify whether loss is being erroneously computed over system/user prompt tokens when `train_on_inputs: false` is combined with templates lacking `{% generation %}` tags.
3. **Multi-GPU / Multi-Node Distributed Synchronization Bugs (DDP, DeepSpeed, FSDP2):**
   All GPU tests ran on a single NVIDIA Tesla T4 (`world_size=1`). Our test suite cannot detect NCCL collective deadlocks, un-averaged gradient all-reduce bugs, FSDP2 sharded state-dict corruption on save/resume, or rank-0-only file race conditions in `soup sweep`.
4. **4-Bit (`bitsandbytes` NF4) & 8-Bit Quantization Dequantization Drift:**
   Because our T4 reproducibility baseline ran with `quantization: none` (`fp16`) to establish clean weight-diff bounds, our suite does not test `bitsandbytes` NF4 paged optimizer states, double-quantization numerical drift, or `soup merge` adapter dequantization fidelity on 4-bit base models.
5. **Non-SFT Trainer Loops (`dpo`, `grpo`, `ppo`, `orpo`, `kto`, `simpo`, `ipo`, `emb`, `pretrain`):**
   While we grey-box inspected `pretrain.py` and `sft.py` and fuzzed `SoupConfig` across all task types, our live T4 GPU execution exercised only `task: sft`. Reward-model advantage NaN propagation in `grpo`/`ppo` or reference-model log-prob mismatches in `dpo`/`kto` remain untested on GPU.
6. **Semantic Data Leakage & Near-Duplicate Contamination:**
   While we tested exact duplicate rows (`SF-03`), empty files, and broken schemas, our suite will not detect semantic train-into-eval contamination (e.g., paraphrased benchmark test questions leaked into the training JSONL).
