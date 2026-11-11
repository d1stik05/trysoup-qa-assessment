# Phase 1: Test Strategy (`soup-cli` v0.72.4 Release Candidate)

> **Order-of-Operations Attestation:** Written **before** executing any config fuzzing, reproducibility, or silent-failure test suites. Based strictly on Phase 0 CLI discovery (`raw_logs/00_env_and_cli_help.log`) and grey-box static inspection of `soup_cli` source layout.

## 1. Objective & Product Risk Hypothesis
`soup-cli` (`v0.72.4`) is an all-in-one LLM post-training CLI spanning 70+ commands, a 6,385-line Pydantic configuration schema (`soup_cli/config/schema.py`), 25+ trainers (`sft`, `dpo`, `grpo`, `ppo`, `kto`, `orpo`, `simpo`, `ipo`, etc.), dataset loaders/validators, and a release gate (`soup ship`).
**Core QA Question:** *Can the current version be released, or can a run finish with exit code `0` ("success" / `SHIP`) while producing an invalid, un-trained, or corrupted result?*

## 2. What We Test First and Why (Execution Order)
We order test execution from fast, high-breadth static/CLI surfaces to deeper, compute-heavy training and artifact verification:

1. **Stage 1 — Smoke & CLI Entry-Point Hygiene (Seconds):**
   - *Target:* Top-level `soup` invocation, `--help`, `version` vs `--version`, `soup init`, `soup doctor`, `soup train --dry-run`, `soup data validate`, `soup plan`, and `soup ship`.
   - *Why first:* If CLI option parsing or config loading crashes or returns inconsistent exit codes (`0` vs `1`/`2`/`3`), all downstream automation and CI gating are unreliable.
2. **Stage 2 — Generative Config & CLI Boundary Fuzzing (`tests/test_config_fuzzing.py`) (Minutes, Local CPU):**
   - *Target:* `SoupConfig` (`LoraConfig`, `DataConfig`, `TrainingConfig`, `EvalConfig`, `ShipConfig`) and CLI flags across all 6 mandatory input classes using `hypothesis` + `pytest`:
     1. Typos in keys (root, `data`, `training`, `lora`, `eval`)
     2. Wrong data types (`str` for `int`/`float`, `bool` for `int`, `list` for `dict`, `null`)
     3. Out-of-range values (`lr <= 0`, `epochs <= 0`, `val_split > 0.5`, `lora.r <= 0`, `NaN`/`Inf`)
     4. Empty / missing fields (empty YAML, `null` required fields, missing `base` or `data.train`)
     5. Conflicting parameters (e.g., `use_dora` + `use_vera`, `load_in_8bit` + `load_in_16bit`, incompatible task/format/backend combinations, CLI `--wandb` + `--tracker`)
     6. Very long strings (10,000+ chars in `base`, `data.train`, `output`, `experiment_name`, `chat_template`)
   - *Why second:* Pydantic v2 models default to `extra='ignore'` unless `model_config = ConfigDict(extra='forbid')` is explicitly set on every nested model. If unknown/typo keys are silently ignored, users believe they changed `learning_rate` or `lora_r` while `soup` silently trains with defaults.
3. **Stage 3 — Training Pipeline Silent Failures (`tests/test_silent_failures.py`) (Minutes, Local + T4 GPU):**
   - *Target:* End-to-end and `--dry-run` pipelines (`soup train`, `soup data`, `soup ship`, dataset loaders, and trainer wrappers) probed for cases where exit code is `0` despite broken state:
     - Silent dropping of misspelled YAML keys (`learning_rate` instead of `lr`, `lora_r` instead of `lora.r`).
     - Corrupt/degenerate JSONL datasets (100% malformed lines skipped with only a warning, wrong column names filtered to 0 rows, duplicate rows, `val_split` truncation).
     - Degenerate training dynamics (`NaN`/`Inf` loss propagation, zero-gradient / frozen adapter weights where saved checkpoint diff $\|\Delta \theta\|_2 = 0$, or `soup ship` reporting `SHIP` on degenerate/forgetting evidence).
   - *Why third:* This directly addresses the highest-scoring product risk: a run that finishes "successfully" (`exit 0`) whose artifact is unusable.
4. **Stage 4 — Seed Reproducibility & Numerical Determinism (`tests/test_reproducibility.py` & `scripts/run_colab_t4_experiments.py`):**
   - *Target:* Two identical training runs (`seed=42`) compared across step-by-step loss curves, eval metrics, and saved LoRA adapter tensors ($\|\theta_1 - \theta_2\|_2$ and $\max |\theta_1 - \theta_2|$).
   - *Why fourth:* Requires executing full forward/backward/optimizer steps and isolating true seed bugs (unseeded RNGs, missing seed wiring in `TrainingConfig` / `SFTTrainerWrapper`) from hardware-level CUDA/T4 floating-point non-determinism.

## 3. Risk Prioritization Matrix
| Priority | Failure Class | Definition & Impact | Target Action |
| :--- | :--- | :--- | :--- |
| **P0 (Critical)** | **Silent Failure / False Success** | CLI exits `0` (or `soup ship` exits `0`) when config keys are typo'd/ignored, dataset rows are silently dropped to empty/tiny sets, seed is ignored, weights do not update, or `NaN` loss is saved as a valid checkpoint. | Block release (`DON'T SHIP`); file P0/High bug report with minimal reproducer. |
| **P1 (High)** | **Unhandled Traceback on User Input** | Raw Python exception/stack trace (`TypeError`, `KeyError`, `ZeroDivisionError`, `AttributeError`) on malformed YAML, bad CLI flags, or edge-case data instead of a clean validation message. | File High/Medium bug report; require graceful error + non-zero exit. |
| **P2 (Medium)** | **Inconsistent Exit Codes / Gate Bypass** | Validation errors exiting `0`, or CLI usability traps (e.g., standard flags like `--version` failing or conflicting flags accepted). | Document in fuzz matrix and bug reports. |
| **P3 (Low)** | **Cosmetic / UI Formatting** | Minor Rich panel wrapping, help-text typos, or log verbosity quirks that do not affect artifacts or exit codes. | Note briefly; do not block release. |

## 4. What We Deliberately DO NOT Test (and Why)
To maximize defect yield within the 4–6 hour evaluation budget and free Colab T4 (16 GB VRAM) hardware envelope, we explicitly exclude:
1. **Multi-Node / Multi-GPU Distributed Runtime (`FSDP`, `DeepSpeed`, `ring_attention`):** Cannot be faithfully executed on a single T4 GPU; we test their config/CLI validation boundaries only.
2. **Large Scale Models (7B–70B+) & Long-Horizon Convergence:** Running 7B+ models or thousands of steps tests base-model capability rather than `soup-cli` software correctness. We use a tiny causal LM (`trl-internal-testing/tiny-random-LlamaForCausalLM` / `hf-internal-testing/tiny-random-gpt2`) on synthetic JSONL data so runs finish in seconds/minutes while exercising the exact same code paths.
3. **External Cloud / Paid APIs & Live Hub Push:** We do not test live `soup push` to HuggingFace Hub, Modal cloud execution, or paid OpenAI/Anthropic judge APIs to avoid external network flakiness and credential dependencies.
4. **Hardware-Specific Kernels Unavailable on T4/Windows:** Apple Silicon `mlx` backend, H100/B100 `fp8` CUTLASS kernels, and interactive Web UI (`soup ui` / `soup tui`) visual rendering.
