# Part 2: Generative Config & CLI Fuzzing Matrix (`soup-cli v0.72.4`)

**Execution Suite:** [`tests/test_config_fuzzing.py`](../tests/test_config_fuzzing.py) (`pytest` + `hypothesis`)  
**Raw Execution Logs:** [`raw_logs/02_config_fuzzing.log`](../raw_logs/02_config_fuzzing.log) & [`raw_logs/02_config_fuzzing_cases.jsonl`](../raw_logs/02_config_fuzzing_cases.jsonl)  
*(Initial exploratory runs preserved untouched in [`raw_logs/02_config_fuzzing_run1_failed.log`](../raw_logs/02_config_fuzzing_run1_failed.log) and [`raw_logs/02_config_fuzzing_run2_failed.log`](../raw_logs/02_config_fuzzing_run2_failed.log)).*

---

## 1. Executive Summary of Fuzzing Outcomes

Across **90 automated test executions** (59 deterministic boundary cases + 31 Hypothesis property-based fuzzing strategies generating >70 randomized examples across all 6 required input classes), every execution was strictly classified by exit code and output stream into:
- **`Clear Error`**: Graceful validation message + non-zero exit code (`rc != 0`), no unhandled Python traceback.
- **`Traceback`**: Unhandled Python exception with raw stack trace (`Traceback (most recent call last):`).
- **`Silent Acceptance`**: Exit code `0` (`rc == 0`) with no error or warning despite broken, misspelled, out-of-range, conflicting, or 10,000+ character input.

| Input Class | Total Cases | `Clear Error` | `Traceback` | `Silent Acceptance` (Defect) | Primary Root Cause in `soup_cli` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Typos in keys** | 15 + 20 Hypothesis | 2 (CLI flags) | 0 | **13 + 20 Hypothesis** | `SoupConfig`, `TrainingConfig`, `DataConfig`, `LoraConfig` omit `model_config = ConfigDict(extra="forbid")` in `soup_cli/config/schema.py` |
| **2. Wrong data types** | 15 + 20 Hypothesis | 6 + 20 Hypothesis | **4** | **5** | `load_config()` does `SoupConfig(**raw)` without `isinstance(raw, dict)` (`TypeError`); missing `_reject_bool` guards on `epochs`, `batch_size`, `lora.r`, `save_steps` |
| **3. Out-of-range values** | 22 + 20 Hypothesis | 8 | 0 | **14 + 20 Hypothesis** | Missing `ge=1` / `[0,1)` bounds on `batch_size`, `lora.r`, `lora.alpha`, `lora.dropout`, `save_steps`, `logging_steps`; `.inf` allowed on `lr`, `weight_decay`, `max_grad_norm` |
| **4. Empty / missing fields** | 16 | 10 | **1** | **5** | `ScannerError` uncaught on tab-whitespace YAML; empty strings `""` accepted for `base`, `data.train`, `output`, and `[]` for `lora.target_modules` |
| **5. Conflicting parameters** | 18 | 12 | 0 | **6** | `soup plan` bypasses `SoupConfig` validation completely; `task: dpo` + `format: alpaca` uncaught; `unfrozen_parameters` + `lora.r/alpha` silently ignores LoRA settings |
| **6. Very long values (10k+)** | 7 + 10 Hypothesis | 1 | **1** | **5 + 10 Hypothesis** | `base`, `data.train`, `output`, `experiment_name`, `chat_template` lack `max_length` caps in `SoupConfig`, causing downstream `OSError` / `FileNotFoundError` tracebacks |

---

## 2. Input Class -> Behavior Assessment Matrix

### Class 1: Typos in Keys (YAML Schema & CLI Flags)

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class1_typo_root_typo_baze` | `load_config` | Root key `baze: "some-model"` alongside `base` | `0` | **`Silent Acceptance`** | Silently ignored (`SoupConfig` lacks `extra="forbid"`). |
| `class1_typo_root_typo_taks` | `load_config` | Root key `taks: "dpo"` (typo for `task`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.task` silently defaults to `"sft"`. |
| `class1_typo_training_typo_learning_rate` | `load_config` | `training.learning_rate: 1e-3` (alias for `lr`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.training.lr` stays default `2e-5`. |
| `class1_typo_training_typo_learing_rate` | `load_config` | `training.learing_rate: 5e-4` | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.training.lr` stays default `2e-5`. |
| `class1_typo_training_typo_epchs` | `load_config` | `training.epchs: 10` (typo for `epochs`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.training.epochs` stays default `3`. |
| `class1_typo_training_typo_seed` | `load_config` | `training.seed: 42` | `0` | **`Silent Acceptance`** | **Critical:** `sft.py:585` & `train.py:1668` call `getattr(tcfg, "seed", None)`, but `TrainingConfig` has no `seed` field, so `seed` is silently dropped! |
| `class1_typo_training_typo_max_steps` | `load_config` | `training.max_steps: 50` | `0` | **`Silent Acceptance`** | Silently ignored (`TrainingConfig` has no `max_steps`). |
| `class1_typo_lora_typo_lora_r` | `load_config` | `training.lora.lora_r: 128` (instead of `r`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.training.lora.r` stays default `64`. |
| `class1_typo_lora_typo_lora_alpha` | `load_config` | `training.lora.lora_alpha: 64` (instead of `alpha`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.training.lora.alpha` stays default `16`. |
| `class1_typo_data_typo_train_file` | `load_config` | `data.train_file: "other.jsonl"` | `0` | **`Silent Acceptance`** | Silently ignored (`DataConfig` lacks `extra="forbid"`). |
| `class1_typo_data_typo_max_seq_length` | `load_config` | `data.max_seq_length: 512` (instead of `max_length`) | `0` | **`Silent Acceptance`** | Silently ignored; `cfg.data.max_length` stays `2048`. |
| `class1_hypo_unknown_key_*` (20x) | `load_config` | Hypothesis arbitrary keys at root/`training`/`data`/`lora` | `0` | **`Silent Acceptance`** | 100% of arbitrary unknown YAML keys are silently ignored. |
| `class1_cli_train_dry_run_with_yaml_typos` | `soup train --dry-run` | `learning_rate: 999.0`, `epchs: -999`, `lora_r: -999` | `0` | **`Silent Acceptance`** | Prints `Config valid. Ready to train!` (`rc=0`). |
| `class1_cli_typo_epocs` | `soup train` | `soup train --epocs 5` | `2` | **`Clear Error`** | Typer rejects unknown CLI option: `No such option: --epocs`. |
| `class1_cli_flag_dash_version` | `soup` | `soup --version` | `2` | **`Clear Error`** | `No such option: --version` (Soup uses subcommand `soup version`). |

---

### Class 2: Wrong Data Types

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class2_type_yaml_root_list` | `load_config` | Top-level YAML list: `- item1\n- item2` | `1` | **`Traceback`** | Uncaught `TypeError: SoupConfig() argument after ** must be a mapping, not list` (`loader.py:24`). |
| `class2_type_yaml_root_scalar_str` | `load_config` | Bare string YAML: `just_a_string_not_a_dict` | `1` | **`Traceback`** | Uncaught `TypeError: SoupConfig() argument after ** must be a mapping, not str`. |
| `class2_type_yaml_root_scalar_int` | `load_config` | Bare integer YAML: `12345` | `1` | **`Traceback`** | Uncaught `TypeError: SoupConfig() argument after ** must be a mapping, not int`. |
| `class2_cli_train_yaml_list_default` | `soup train` | `soup train --config list.yaml --dry-run` | `1` | **`Clear Error`** | Global handler in `cli.py:835` masks `TypeError` as 1-line `Error: TypeError: ... Run with --verbose`. |
| `class2_cli_train_yaml_list_verbose_traceback` | `soup --verbose train` | `soup --verbose train --config list.yaml --dry-run` | `1` | **`Traceback`** | Emits full unhandled `TypeError` stack trace from `loader.py:24`. |
| `class2_type_epochs_string` | `load_config` | `training.epochs: "three"` | `1` | **`Clear Error`** | Pydantic validation table: `Input should be a valid integer`. |
| `class2_type_lr_list` | `load_config` | `training.lr: [1e-4, 2e-4]` | `1` | **`Clear Error`** | Pydantic validation table: `Input should be a valid number`. |
| `class2_type_data_is_list` | `load_config` | `data: ["train.jsonl"]` | `1` | **`Clear Error`** | Pydantic validation table: `Input should be a valid dictionary`. |
| `class2_type_lora_is_string` | `load_config` | `training.lora: "invalid_lora"` | `1` | **`Clear Error`** | Pydantic validation table: `Input should be a valid dictionary`. |
| `class2_hypo_wrong_type_epochs` (20x) | `load_config` | Hypothesis lists/dicts/non-numeric strings for `epochs` | `1` | **`Clear Error`** | All non-coercible types cleanly rejected by Pydantic. |
| `class2_type_epochs_bool_true` | `load_config` | `training.epochs: true` | `0` | **`Silent Acceptance`** | Coerced to `epochs=1` (missing `_reject_bool` validator on `epochs`). |
| `class2_type_lora_r_bool_true` | `load_config` | `training.lora.r: true` | `0` | **`Silent Acceptance`** | Coerced to `r=1` (missing `_reject_bool` validator on `LoraConfig.r`). |
| `class2_type_batch_size_bool_true` | `load_config` | `training.batch_size: true` | `0` | **`Silent Acceptance`** | Coerced to `batch_size=1` (missing `_reject_bool` on `batch_size`). |
| `class2_type_save_steps_bool_false` | `load_config` | `training.save_steps: false` | `0` | **`Silent Acceptance`** | Coerced to `save_steps=0` (missing `_reject_bool` & `ge=1` on `save_steps`). |
| `class2_type_numeric_string_coercion` | `load_config` | `epochs: "5"`, `lr: "0.0001"` | `0` | **`Silent Acceptance`** | Lax Pydantic mode coerces numeric strings to `int`/`float`. |

---

### Class 3: Out-of-Range Values

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class3_range_epochs_zero` | `load_config` | `training.epochs: 0` | `1` | **`Clear Error`** | Rejected (`Field(default=3, ge=1)`). |
| `class3_range_epochs_negative` | `load_config` | `training.epochs: -5` | `1` | **`Clear Error`** | Rejected (`Input should be greater than or equal to 1`). |
| `class3_range_lr_zero` | `load_config` | `training.lr: 0.0` | `1` | **`Clear Error`** | Rejected (`Field(default=2e-5, gt=0)`). |
| `class3_range_lr_negative` | `load_config` | `training.lr: -1e-4` | `1` | **`Clear Error`** | Rejected (`Input should be greater than 0`). |
| `class3_range_lr_nan` | `load_config` | `training.lr: .nan` | `1` | **`Clear Error`** | Rejected (`NaN > 0` is `False`). |
| `class3_range_val_split_over_half` | `load_config` | `data.val_split: 0.8` | `1` | **`Clear Error`** | Rejected (`Field(default=0.1, ge=0.0, le=0.5)`). |
| `class3_range_val_split_negative` | `load_config` | `data.val_split: -0.1` | `1` | **`Clear Error`** | Rejected (`Input should be >= 0`). |
| `class3_range_max_length_too_small` | `load_config` | `data.max_length: 16` | `1` | **`Clear Error`** | Rejected (`Field(default=2048, ge=64)`). |
| `class3_range_lr_inf` | `load_config` | `training.lr: .inf` | `0` | **`Silent Acceptance`** | **`float('inf') > 0` is `True`**; lacks `allow_inf_nan=False` or upper bound! |
| `class3_range_weight_decay_inf` | `load_config` | `training.weight_decay: .inf` | `0` | **`Silent Acceptance`** | `.inf` accepted (`ge=0.0` without `allow_inf_nan=False`). |
| `class3_range_max_grad_norm_inf` | `load_config` | `training.max_grad_norm: .inf` | `0` | **`Silent Acceptance`** | `.inf` accepted (`gt=0` without `allow_inf_nan=False`). |
| `class3_range_batch_size_zero` | `load_config` | `training.batch_size: 0` | `0` | **`Silent Acceptance`** | **Critical:** `Union[int, Literal["auto"]]` has no `ge=1` bound; causes `ZeroDivisionError` in `sft.py:361`! |
| `class3_range_batch_size_negative` | `load_config` | `training.batch_size: -4` | `0` | **`Silent Acceptance`** | Accepted by schema (`batch_size: -4`). |
| `class3_range_lora_r_zero` | `load_config` | `training.lora.r: 0` | `0` | **`Silent Acceptance`** | Accepted by `LoraConfig` (`r: int = 64` has no `ge=1` bound); crashes later inside `peft` after model download. |
| `class3_range_lora_r_negative` | `load_config` | `training.lora.r: -8` | `0` | **`Silent Acceptance`** | Accepted by `LoraConfig` (`r: -8`). |
| `class3_range_lora_alpha_negative` | `load_config` | `training.lora.alpha: -16` | `0` | **`Silent Acceptance`** | Accepted by `LoraConfig` (`alpha: int = 16` has no `ge=1` bound). |
| `class3_range_lora_dropout_negative` | `load_config` | `training.lora.dropout: -0.5` | `0` | **`Silent Acceptance`** | Accepted by `LoraConfig` (`dropout: float = 0.05` has no `ge=0, lt=1` bound). |
| `class3_range_lora_dropout_over_one` | `load_config` | `training.lora.dropout: 5.0` | `0` | **`Silent Acceptance`** | Accepted by `LoraConfig` (`dropout: 5.0`). |
| `class3_range_save_steps_zero` | `load_config` | `training.save_steps: 0` | `0` | **`Silent Acceptance`** | Accepted by `TrainingConfig` (`save_steps: int = 100` has no `ge=1`). |
| `class3_range_save_steps_negative` | `load_config` | `training.save_steps: -10` | `0` | **`Silent Acceptance`** | Accepted by `TrainingConfig` (`save_steps: -10`). |
| `class3_range_logging_steps_zero` | `load_config` | `training.logging_steps: 0` | `0` | **`Silent Acceptance`** | Accepted by `TrainingConfig` (`logging_steps: int = 10` has no `ge=1`). |
| `class3_range_logging_steps_negative` | `load_config` | `training.logging_steps: -5` | `0` | **`Silent Acceptance`** | Accepted by `TrainingConfig` (`logging_steps: -5`). |
| `class3_hypo_unbounded_lora_bs` (20x) | `load_config` | Hypothesis `batch_size <= 0`, `lora.r <= 0`, `dropout` outside `[0,1)` | `0` | **`Silent Acceptance`** | All 20 generated out-of-range combinations silently pass `SoupConfig`. |

---

### Class 4: Empty / Missing Fields

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class4_empty_empty_file` | `load_config` | 0-byte empty `soup.yaml` | `1` | **`Clear Error`** | `Config error: Config file is empty` (`loader.py:19`). |
| `class4_empty_spaces_only_file` | `load_config` | `"   \n  \n"` (spaces and newlines only) | `1` | **`Clear Error`** | `yaml.safe_load` returns `None` -> `Config file is empty`. |
| `class4_empty_tab_whitespace_file` | `load_config` | `"   \n\t  \n"` (contains tab character `\t`) | `1` | **`Traceback`** | **Uncaught `yaml.scanner.ScannerError`:** `found character '\t' that cannot start any token` (`loader.py:17` does not catch `yaml.YAMLError`!). |
| `class4_empty_comments_only_file` | `load_config` | `"# Only a comment\n"` | `1` | **`Clear Error`** | `Config error: Config file is empty`. |
| `class4_empty_empty_dict` | `load_config` | `"{}\n"` | `1` | **`Clear Error`** | Reports missing required fields `base` and `data`. |
| `class4_empty_missing_base` | `load_config` | Valid config with `base` key omitted | `1` | **`Clear Error`** | Reports `base: Field required`. |
| `class4_empty_missing_data` | `load_config` | Valid config with `data` key omitted | `1` | **`Clear Error`** | Reports `data: Field required`. |
| `class4_empty_missing_data_train` | `load_config` | `data: {format: alpaca}` (`train` omitted) | `1` | **`Clear Error`** | Reports `data.train: Field required`. |
| `class4_empty_null_base` | `load_config` | `base: null` | `1` | **`Clear Error`** | Reports `base: Input should be a valid string`. |
| `class4_empty_null_training` | `load_config` | `training: null` | `1` | **`Clear Error`** | Reports `training: Input should be a valid dictionary`. |
| `class4_empty_empty_string_base` | `load_config` | `base: ""` (empty string) | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` (`base: str` lacks `min_length=1`!). |
| `class4_empty_whitespace_string_base` | `load_config` | `base: "   "` (whitespace string) | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` (no `.strip()` / non-empty validator). |
| `class4_empty_empty_string_data_train` | `load_config` | `data.train: ""` | `0` | **`Silent Acceptance`** | Accepted by `DataConfig` (`train: str` lacks `min_length=1`). |
| `class4_empty_empty_string_output` | `load_config` | `output: ""` | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` (`output: str` lacks `min_length=1`). |
| `class4_empty_empty_list_target_modules` | `load_config` | `training.lora.target_modules: []` | `0` | **`Silent Acceptance`** | Empty list `[]` accepted by `LoraConfig`. |
| `class4_cli_dry_run_empty_base_string` | `soup train --dry-run` | `base: ""` with valid `data.train` | `0` | **`Silent Acceptance`** | **Critical:** `soup train --dry-run` exits `0` and prints `Config valid. Ready to train!` when `base: ""`! |

---

### Class 5: Conflicting Parameters

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class5_conflict_dora_and_vera` | `load_config` | `use_dora: true` + `use_vera: true` | `1` | **`Clear Error`** | Caught by `_validate_peft_combinations` (`schema.py:926`). |
| `class5_conflict_dora_and_pissa` | `load_config` | `use_dora: true` + `init_strategy: "pissa"` | `1` | **`Clear Error`** | Caught by `_validate_peft_combinations` (`schema.py:933`). |
| `class5_conflict_olora_and_random_init` | `load_config` | `use_olora: true` + `init_strategy: "random"` | `1` | **`Clear Error`** | Caught by `_validate_peft_combinations` (`schema.py:953`). |
| `class5_conflict_vera_and_rank_pattern` | `load_config` | `use_vera: true` + `rank_pattern: {q_proj: 8}` | `1` | **`Clear Error`** | Caught by `_validate_peft_combinations` (`schema.py:964`). |
| `class5_conflict_root_and_training_lora` | `load_config` | Both root `lora:` and `training.lora:` set | `1` | **`Clear Error`** | Caught by `_hoist_top_level_lora` (`schema.py:4820`). |
| `class5_conflict_load_in_8bit_and_16bit` | `load_config` | `load_in_8bit: true` + `load_in_16bit: true` | `1` | **`Clear Error`** | Caught by `_normalize_quantization_aliases` (`schema.py:4463`). |
| `class5_conflict_eval_gate_enabled_no_suite` | `load_config` | `eval_gate: {enabled: true, suite: null}` | `1` | **`Clear Error`** | Caught by `EvalGateConfig` validator (`schema.py:1945`). |
| `class5_conflict_reward_hack_halt_no_detector` | `load_config` | `reward_hack_halt: true`, `reward_hack_detector: null` | `1` | **`Clear Error`** | Caught by `_validate_reward_hack` (`schema.py:4517`). |
| `class5_conflict_mlx_with_multipack` | `load_config` | `backend: "mlx"` + `training.multipack: true` | `1` | **`Clear Error`** | Caught by `_validate_multipack_backend` (`schema.py:4884`). |
| `class5_conflict_unfrozen_params_with_dora` | `load_config` | `unfrozen_parameters: ["layers.0.*"]` + `use_dora: true` | `1` | **`Clear Error`** | Caught by `_validate_unfrozen_parameters` (`schema.py:4531`). |
| `class5_cli_wandb_and_tensorboard` | `soup train` | `soup train --wandb --tensorboard --dry-run` | `1` | **`Clear Error`** | Caught in `train.py:258` (`Cannot use both --wandb and --tensorboard`). |
| `class5_cli_hf_resume_without_push_as` | `soup train` | `soup train --hf-resume --dry-run` | `1` | **`Clear Error`** | Caught in `train.py:293` (`--hf-resume requires --push-as`). |
| `class5_conflict_unfrozen_params_with_custom_lora` | `load_config` | `unfrozen_parameters: ["layers.0.*"]` + `lora: {r: 32, alpha: 64}` | `0` | **`Silent Acceptance`** | `_validate_unfrozen_parameters` only checks boolean flags (`use_dora`, etc.), so custom `lora.r`/`alpha` are **silently ignored** when Spectrum unfreezing disables LoRA! |
| `class5_conflict_dpo_task_with_alpaca_format` | `load_config` | `task: "dpo"` + `data.format: "alpaca"` | `0` | **`Silent Acceptance`** | DPO requires `chosen`/`rejected` pairs, incompatible with `alpaca` (`instruction`/`output`), yet `SoupConfig` has no task-vs-format cross-validator. |
| `class5_conflict_kto_task_with_sharegpt_format` | `load_config` | `task: "kto"` + `data.format: "sharegpt"` | `0` | **`Silent Acceptance`** | KTO requires `completion`/`label`, incompatible with `sharegpt`, yet accepted by `SoupConfig`. |
| `class5_conflict_save_steps_gt_total_steps` | `load_config` | `epochs: 1`, `save_steps: 999999`, `logging_steps: 999999` | `0` | **`Silent Acceptance`** | Accepted without warning even when `save_steps` / `logging_steps` exceed total steps by 100,000x. |
| `class5_conflict_loftq_bits_with_random_init` | `load_config` | `init_strategy: "random"`, `loftq_bits: 2`, `loftq_iter: 5` | `0` | **`Silent Acceptance`** | Dead knobs `loftq_bits`/`loftq_iter` silently ignored unless `init_strategy == "loftq"`. |
| `class5_cli_soup_plan_bypasses_schema_validation` | `soup plan` | `task: "completely_invalid_task_name"`, `epochs: -100`, `lr: -5.0`, `use_dora: true` + `use_vera: true` | `0` | **`Silent Acceptance`** | **Critical Gate Bug:** `soup plan` (`commands/plan.py:64-73`) loads raw YAML without calling `SoupConfig`, exits `0`, prints a cost/VRAM plan, and writes `soup.tfstate`! |

---

### Class 6: Very Long Values (10,000+ Characters)

| Case ID | Surface | Input / Mutation | Exit Code | Behavior Assessment | Exact Observation & Root Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `class6_long_unfrozen_param_10k_chars` | `load_config` | `training.unfrozen_parameters: ["a" * 12000]` | `1` | **`Clear Error`** | Caught by `_validate_unfrozen_parameters` (`exceeds 512 chars`). |
| `class6_long_base_10k_chars` | `load_config` | `base: "a" * 12000` | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` (whereas `soup plan` caps `base` at 512 chars!). |
| `class6_long_data_train_10k_chars` | `load_config` | `data.train: "./" + "a" * 12000 + ".jsonl"` | `0` | **`Silent Acceptance`** | Accepted by `DataConfig` without path-length validation. |
| `class6_long_output_10k_chars` | `load_config` | `output: "./" + "a" * 12000` | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` without path-length validation. |
| `class6_long_experiment_name_10k_chars` | `load_config` | `experiment_name: "a" * 12000` | `0` | **`Silent Acceptance`** | Accepted by `SoupConfig` without length cap. |
| `class6_long_chat_template_10k_chars` | `load_config` | `data.chat_template: "a" * 12000` | `0` | **`Silent Acceptance`** | Accepted by `DataConfig`. |
| `class6_hypo_10k_len_*` (10x) | `load_config` | Hypothesis 10,000–15,000 char strings in `base`, `experiment_name`, `output` | `0` | **`Silent Acceptance`** | All 10 Hypothesis 10k–15k char payloads pass `SoupConfig`. |
| `class6_cli_dry_run_10k_data_train_path` | `soup --verbose train` | `data.train` set to 10,500-char path in `--dry-run` | `1` | **`Traceback`** | Passes `load_config()` (`Silent Acceptance`), then crashes in `Path.exists()` / `load_dataset()` with unhandled `OSError` / `FileNotFoundError` traceback. |
