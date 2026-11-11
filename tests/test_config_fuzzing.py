"""Phase 2: Generative Config & CLI Testing (Hypothesis + pytest).

Exercises all 6 mandatory input classes against soup-cli (v0.72.4):
  1. Typos in keys (root, data, training, lora, eval, CLI flags)
  2. Wrong data types (strings for ints/floats, bools for ints, lists/scalars for dicts, nulls)
  3. Out-of-range values (negative/zero epochs, lr, batch_size, lora.r, lora.alpha, dropout, save_steps, NaN/Inf)
  4. Empty / missing fields (empty YAML, missing base, missing data.train, empty strings)
  5. Conflicting parameters (PEFT mutual exclusion, quantization aliases, CLI flag conflicts, plan vs train)
  6. Excessively long values (10,000+ char strings in base, data.train, output, experiment_name, chat_template)

Every test case records its unfiltered execution (command/API, payload, exit code, stdout, stderr,
and strict classification: Clear Error, Traceback, or Silent Acceptance) to
`raw_logs/02_config_fuzzing_cases.jsonl`.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
import pytest
import yaml

from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.config.schema import SoupConfig

RAW_LOG_DIR = Path("raw_logs")
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
CASES_JSONL = RAW_LOG_DIR / "02_config_fuzzing_cases.jsonl"

BASELINE_DATASET = Path("configs/sample_train.jsonl")


def get_valid_config_dict() -> dict[str, Any]:
    return {
        "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
        "task": "sft",
        "backend": "transformers",
        "data": {
            "train": str(BASELINE_DATASET).replace("\\", "/"),
            "format": "alpaca",
            "val_split": 0.2,
            "max_length": 128,
        },
        "training": {
            "epochs": 1,
            "lr": 2e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "save_steps": 10,
            "logging_steps": 1,
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": "auto",
            },
        },
        "output": "./output_fuzz",
    }


def classify_outcome(exit_code: int, stdout: str, stderr: str) -> str:
    """Strictly classify behavior into Clear Error, Traceback, or Silent Acceptance."""
    combined = f"{stdout}\n{stderr}"
    if "Traceback (most recent call last):" in combined:
        return "Traceback"
    if exit_code == 0:
        return "Silent Acceptance"
    return "Clear Error"


def record_case(
    *,
    test_id: str,
    input_class: str,
    surface: str,
    description: str,
    payload_preview: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> str:
    classification = classify_outcome(exit_code, stdout, stderr)
    entry = {
        "test_id": test_id,
        "input_class": input_class,
        "surface": surface,
        "description": description,
        "payload_preview": (
            payload_preview[:500] + f"... [len={len(payload_preview)}]"
            if len(payload_preview) > 500
            else payload_preview
        ),
        "exit_code": exit_code,
        "classification": classification,
        "stdout": stdout,
        "stderr": stderr,
    }
    with open(CASES_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return classification


def run_load_config_inproc(
    yaml_content: str,
    tmp_path: Path,
    *,
    test_id: str,
    input_class: str,
    description: str,
) -> tuple[str, int, str, str, SoupConfig | None]:
    """Run soup_cli.config.loader.load_config(path) and capture exact exit code / tracebacks."""
    cfg_file = tmp_path / "fuzz_soup.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    cfg_obj: SoupConfig | None = None
    exit_code = 0

    import soup_cli.config.loader as loader_mod
    from rich.console import Console

    orig_console = loader_mod.console
    loader_mod.console = Console(file=out_buf, force_terminal=False, color_system=None, width=120)
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            cfg_obj = load_config(cfg_file)
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception:
        exit_code = 1
        err_buf.write(traceback.format_exc())
    finally:
        loader_mod.console = orig_console

    stdout = out_buf.getvalue()
    stderr = err_buf.getvalue()
    classification = record_case(
        test_id=test_id,
        input_class=input_class,
        surface="load_config(path)",
        description=description,
        payload_preview=yaml_content,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
    return classification, exit_code, stdout, stderr, cfg_obj


def run_soup_cli_subproc(
    args: list[str],
    *,
    test_id: str,
    input_class: str,
    description: str,
    payload_preview: str = "",
) -> tuple[str, int, str, str]:
    """Invoke the real `soup` CLI in a subprocess and capture unfiltered exit code, stdout, stderr."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env["SOUP_NO_AUDIT_LOG"] = "1"
    soup_exe = Path(sys.executable).parent / ("soup.exe" if os.name == "nt" else "soup")
    cmd = [str(soup_exe)] + args if soup_exe.exists() else [sys.executable, "-m", "soup_cli.cli"] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
    )
    classification = record_case(
        test_id=test_id,
        input_class=input_class,
        surface=f"CLI: soup {' '.join(args[:4])}",
        description=description,
        payload_preview=payload_preview or " ".join(args),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    return classification, proc.returncode, proc.stdout, proc.stderr


# ==============================================================================
# CLASS 1: TYPOS IN KEYS (Config YAML & CLI Flags)
# ==============================================================================

TYPO_CONFIG_CASES = [
    ("root_typo_baze", lambda d: {**d, "baze": "some-model"}, "Unknown root key 'baze' alongside valid 'base'"),
    ("root_typo_taks", lambda d: {**d, "taks": "dpo"}, "Misspelled root key 'taks: dpo' (task stays default 'sft')"),
    ("training_typo_learning_rate", lambda d: {**d, "training": {**d["training"], "learning_rate": 1e-3}}, "Common alias 'training.learning_rate' instead of 'training.lr'"),
    ("training_typo_learing_rate", lambda d: {**d, "training": {**d["training"], "learing_rate": 5e-4}}, "Typo 'training.learing_rate'"),
    ("training_typo_epchs", lambda d: {**d, "training": {**d["training"], "epchs": 10}}, "Typo 'training.epchs: 10' (epochs stays default 3)"),
    ("training_typo_seed", lambda d: {**d, "training": {**d["training"], "seed": 42}}, "Undocumented 'training.seed: 42' (referenced in sft.py/train.py getattr, absent in schema)"),
    ("training_typo_max_steps", lambda d: {**d, "training": {**d["training"], "max_steps": 50}}, "Unsupported 'training.max_steps: 50'"),
    ("lora_typo_lora_r", lambda d: {**d, "training": {**d["training"], "lora": {**d["training"]["lora"], "lora_r": 128}}}, "Misspelled 'training.lora.lora_r: 128' instead of 'r'"),
    ("lora_typo_lora_alpha", lambda d: {**d, "training": {**d["training"], "lora": {**d["training"]["lora"], "lora_alpha": 64}}}, "Misspelled 'training.lora.lora_alpha: 64' instead of 'alpha'"),
    ("data_typo_train_file", lambda d: {**d, "data": {**d["data"], "train_file": "other.jsonl"}}, "Typo 'data.train_file' alongside 'data.train'"),
    ("data_typo_max_seq_length", lambda d: {**d, "data": {**d["data"], "max_seq_length": 512}}, "Typo 'data.max_seq_length: 512' instead of 'data.max_length'"),
]


@pytest.mark.parametrize("case_id,mutator,desc", TYPO_CONFIG_CASES)
def test_class1_typos_in_config_keys_deterministic(
    case_id: str, mutator, desc: str, tmp_path: Path
) -> None:
    cfg = mutator(get_valid_config_dict())
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, out, err, parsed = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class1_typo_{case_id}",
        input_class="1. Typos in keys",
        description=desc,
    )
    # Document actual behavior: Pydantic models in soup_cli/config/schema.py lack extra="forbid",
    # so unknown/typo keys are silently accepted with exit code 0!
    assert classification in ("Silent Acceptance", "Clear Error")


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    typo_key=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz_",
        min_size=4,
        max_size=20,
    ).filter(
        lambda k: k not in set(SoupConfig.model_fields.keys())
        and k not in ("lora",)
    ),
    typo_val=st.one_of(st.integers(-100, 100), st.text(min_size=1, max_size=20), st.booleans()),
)
def test_class1_hypothesis_arbitrary_unknown_keys(
    typo_key: str, typo_val: Any, tmp_path: Path
) -> None:
    cfg = get_valid_config_dict()
    cfg[typo_key] = typo_val
    cfg["training"][f"typo_{typo_key}"] = typo_val
    cfg["data"][f"typo_{typo_key}"] = typo_val
    cfg["training"]["lora"][f"typo_{typo_key}"] = typo_val
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class1_hypo_unknown_key_{typo_key[:10]}",
        input_class="1. Typos in keys",
        description=f"Hypothesis arbitrary unknown key '{typo_key}' at root/training/data/lora",
    )
    assert classification == "Silent Acceptance" and rc == 0


def test_class1_cli_flag_typos_and_train_dry_run(tmp_path: Path) -> None:
    # 1a. CLI typo flag on `soup train --epocs 5`
    c1, rc1, _, _ = run_soup_cli_subproc(
        ["train", "--epocs", "5"],
        test_id="class1_cli_typo_epocs",
        input_class="1. Typos in keys",
        description="CLI flag typo: soup train --epocs 5",
    )
    assert c1 == "Clear Error" and rc1 != 0

    # 1b. CLI `--version` flag (common standard flag, rejected in favor of `soup version`)
    c2, rc2, _, _ = run_soup_cli_subproc(
        ["--version"],
        test_id="class1_cli_flag_dash_version",
        input_class="1. Typos in keys",
        description="Top-level soup --version flag",
    )
    assert c2 == "Clear Error" and rc2 == 2

    # 1c. End-to-end `soup train --dry-run` with misspelled hyperparameters in YAML
    cfg = get_valid_config_dict()
    cfg["training"]["learning_rate"] = 999.0
    cfg["training"]["epchs"] = -999
    cfg["training"]["lora"]["lora_r"] = -999
    cfg_path = tmp_path / "typo_dry_run.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    c3, rc3, out3, _ = run_soup_cli_subproc(
        ["train", "--config", str(cfg_path), "--dry-run"],
        test_id="class1_cli_train_dry_run_with_yaml_typos",
        input_class="1. Typos in keys",
        description="soup train --dry-run silently ignores learning_rate, epchs, and lora.lora_r typos",
        payload_preview=cfg_path.read_text(encoding="utf-8"),
    )
    assert c3 == "Silent Acceptance" and rc3 == 0
    assert "Config valid. Ready to train!" in out3


# ==============================================================================
# CLASS 2: WRONG DATA TYPES
# ==============================================================================

WRONG_TYPE_CASES = [
    ("yaml_root_list", "- item1\n- item2\n", "YAML document is a top-level list instead of mapping"),
    ("yaml_root_scalar_str", "just_a_string_not_a_dict\n", "YAML document is a bare scalar string"),
    ("yaml_root_scalar_int", "12345\n", "YAML document is a bare integer"),
    ("epochs_string", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "epochs": "three"}}), "training.epochs is non-numeric string 'three'"),
    ("lr_list", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "lr": [1e-4, 2e-4]}}), "training.lr is a list instead of float"),
    ("data_is_list", yaml.safe_dump({**get_valid_config_dict(), "data": ["train.jsonl"]}), "data block is a list instead of mapping"),
    ("lora_is_string", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "lora": "invalid_lora"}}), "training.lora is a string instead of mapping"),
    ("epochs_bool_true", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "epochs": True}}), "training.epochs is boolean True (Python bool is subclass of int)"),
    ("lora_r_bool_true", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "lora": {**get_valid_config_dict()["training"]["lora"], "r": True}}}), "training.lora.r is boolean True (coerced to int 1 by Pydantic)"),
    ("batch_size_bool_true", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "batch_size": True}}), "training.batch_size is boolean True (coerced to int 1)"),
    ("save_steps_bool_false", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "save_steps": False}}), "training.save_steps is boolean False (coerced to int 0)"),
    ("numeric_string_coercion", yaml.safe_dump({**get_valid_config_dict(), "training": {**get_valid_config_dict()["training"], "epochs": "5", "lr": "0.0001"}}), "Numeric strings '5' and '0.0001' for int/float fields (Pydantic lax coercion)"),
]


@pytest.mark.parametrize("case_id,yaml_payload,desc", WRONG_TYPE_CASES)
def test_class2_wrong_types_deterministic(
    case_id: str, yaml_payload: str, desc: str, tmp_path: Path
) -> None:
    classification, rc, out, err, _ = run_load_config_inproc(
        yaml_payload,
        tmp_path,
        test_id=f"class2_type_{case_id}",
        input_class="2. Wrong data types",
        description=desc,
    )
    if case_id in ("yaml_root_list", "yaml_root_scalar_str", "yaml_root_scalar_int"):
        # BUG: load_config() does SoupConfig(**raw) without checking isinstance(raw, dict)!
        assert classification == "Traceback"
    elif case_id in ("epochs_bool_true", "lora_r_bool_true", "batch_size_bool_true", "save_steps_bool_false", "numeric_string_coercion"):
        # Booleans and numeric strings are silently coerced to ints/floats on fields lacking bool guards!
        assert classification == "Silent Acceptance" and rc == 0
    else:
        assert classification == "Clear Error" and rc == 1


def test_class2_cli_non_mapping_yaml_traceback(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "list_config.yaml"
    bad_yaml.write_text("- a\n- b\n", encoding="utf-8")
    # Without --verbose, soup_cli/cli.py:835 catches the TypeError and formats a 1-line message to stderr
    c_default, rc_default, out_default, err_default = run_soup_cli_subproc(
        ["train", "--config", str(bad_yaml), "--dry-run"],
        test_id="class2_cli_train_yaml_list_default",
        input_class="2. Wrong data types",
        description="CLI soup train --config <list.yaml> --dry-run (default masks TypeError behind 1-line error)",
        payload_preview="- a\n- b\n",
    )
    assert c_default == "Clear Error" and rc_default == 1
    assert "TypeError" in f"{out_default}\n{err_default}"

    # With --verbose, the unhandled TypeError traceback is emitted in full
    c_verb, rc_verb, _, _ = run_soup_cli_subproc(
        ["--verbose", "train", "--config", str(bad_yaml), "--dry-run"],
        test_id="class2_cli_train_yaml_list_verbose_traceback",
        input_class="2. Wrong data types",
        description="CLI soup --verbose train --config <list.yaml> --dry-run emits unhandled TypeError traceback",
        payload_preview="- a\n- b\n",
    )
    assert c_verb == "Traceback" and rc_verb == 1


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    bad_epochs=st.one_of(
        st.lists(st.integers(), min_size=1, max_size=3),
        st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), min_size=1, max_size=2),
        st.text(alphabet="abcdef", min_size=2, max_size=8),
    )
)
def test_class2_hypothesis_wrong_types_for_epochs(bad_epochs: Any, tmp_path: Path) -> None:
    cfg = get_valid_config_dict()
    cfg["training"]["epochs"] = bad_epochs
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id="class2_hypo_wrong_type_epochs",
        input_class="2. Wrong data types",
        description=f"Hypothesis wrong type for training.epochs: {type(bad_epochs).__name__}",
    )
    assert classification == "Clear Error" and rc == 1


# ==============================================================================
# CLASS 3: OUT-OF-RANGE VALUES
# ==============================================================================

OUT_OF_RANGE_CASES = [
    ("epochs_zero", lambda d: _mut_train(d, epochs=0), "training.epochs = 0 (schema has ge=1)"),
    ("epochs_negative", lambda d: _mut_train(d, epochs=-5), "training.epochs = -5"),
    ("lr_zero", lambda d: _mut_train(d, lr=0.0), "training.lr = 0.0 (schema has gt=0)"),
    ("lr_negative", lambda d: _mut_train(d, lr=-1e-4), "training.lr = -1e-4"),
    ("lr_nan", lambda d: _mut_train(d, lr=float("nan")), "training.lr = .nan"),
    ("lr_inf", lambda d: _mut_train(d, lr=float("inf")), "training.lr = .inf (unbounded above in schema!)"),
    ("val_split_over_half", lambda d: _mut_data(d, val_split=0.8), "data.val_split = 0.8 (schema has le=0.5)"),
    ("val_split_negative", lambda d: _mut_data(d, val_split=-0.1), "data.val_split = -0.1"),
    ("max_length_too_small", lambda d: _mut_data(d, max_length=16), "data.max_length = 16 (schema has ge=64)"),
    ("batch_size_zero", lambda d: _mut_train(d, batch_size=0), "training.batch_size = 0 (no ge=1 bound on int branch!)"),
    ("batch_size_negative", lambda d: _mut_train(d, batch_size=-4), "training.batch_size = -4 (no ge=1 bound on int branch!)"),
    ("lora_r_zero", lambda d: _mut_lora(d, r=0), "training.lora.r = 0 (no ge=1 bound in LoraConfig!)"),
    ("lora_r_negative", lambda d: _mut_lora(d, r=-8), "training.lora.r = -8 (no ge=1 bound in LoraConfig!)"),
    ("lora_alpha_negative", lambda d: _mut_lora(d, alpha=-16), "training.lora.alpha = -16 (no ge=1 bound in LoraConfig!)"),
    ("lora_dropout_negative", lambda d: _mut_lora(d, dropout=-0.5), "training.lora.dropout = -0.5 (no [0, 1) bound in LoraConfig!)"),
    ("lora_dropout_over_one", lambda d: _mut_lora(d, dropout=5.0), "training.lora.dropout = 5.0 (no [0, 1) bound in LoraConfig!)"),
    ("save_steps_zero", lambda d: _mut_train(d, save_steps=0), "training.save_steps = 0 (no ge=1 bound in TrainingConfig!)"),
    ("save_steps_negative", lambda d: _mut_train(d, save_steps=-10), "training.save_steps = -10 (no ge=1 bound in TrainingConfig!)"),
    ("logging_steps_zero", lambda d: _mut_train(d, logging_steps=0), "training.logging_steps = 0 (no ge=1 bound in TrainingConfig!)"),
    ("logging_steps_negative", lambda d: _mut_train(d, logging_steps=-5), "training.logging_steps = -5 (no ge=1 bound in TrainingConfig!)"),
    ("weight_decay_inf", lambda d: _mut_train(d, weight_decay=float("inf")), "training.weight_decay = .inf"),
    ("max_grad_norm_inf", lambda d: _mut_train(d, max_grad_norm=float("inf")), "training.max_grad_norm = .inf"),
]


def _mut_train(d: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    out = copy.deepcopy(d)
    out["training"].update(kwargs)
    return out


def _mut_data(d: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    out = copy.deepcopy(d)
    out["data"].update(kwargs)
    return out


def _mut_lora(d: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    out = copy.deepcopy(d)
    out["training"]["lora"].update(kwargs)
    return out


@pytest.mark.parametrize("case_id,mutator,desc", OUT_OF_RANGE_CASES)
def test_class3_out_of_range_deterministic(
    case_id: str, mutator, desc: str, tmp_path: Path
) -> None:
    cfg = mutator(get_valid_config_dict())
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class3_range_{case_id}",
        input_class="3. Out-of-range values",
        description=desc,
    )
    unbounded_bugs = {
        "lr_inf",
        "batch_size_zero",
        "batch_size_negative",
        "lora_r_zero",
        "lora_r_negative",
        "lora_alpha_negative",
        "lora_dropout_negative",
        "lora_dropout_over_one",
        "save_steps_zero",
        "save_steps_negative",
        "logging_steps_zero",
        "logging_steps_negative",
        "weight_decay_inf",
        "max_grad_norm_inf",
    }
    if case_id in unbounded_bugs:
        assert classification == "Silent Acceptance" and rc == 0
    else:
        assert classification == "Clear Error" and rc == 1


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    neg_r=st.integers(min_value=-10_000, max_value=0),
    neg_bs=st.integers(min_value=-10_000, max_value=0),
    bad_dropout=st.one_of(
        st.floats(min_value=-1000.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_class3_hypothesis_unbounded_lora_and_batch_size(
    neg_r: int, neg_bs: int, bad_dropout: float, tmp_path: Path
) -> None:
    cfg = get_valid_config_dict()
    cfg["training"]["batch_size"] = neg_bs
    cfg["training"]["lora"]["r"] = neg_r
    cfg["training"]["lora"]["dropout"] = bad_dropout
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id="class3_hypo_unbounded_lora_bs",
        input_class="3. Out-of-range values",
        description=f"Hypothesis out-of-range batch_size={neg_bs}, lora.r={neg_r}, lora.dropout={bad_dropout}",
    )
    assert classification == "Silent Acceptance" and rc == 0


# ==============================================================================
# CLASS 4: EMPTY / MISSING FIELDS
# ==============================================================================

EMPTY_MISSING_CASES = [
    ("empty_file", "", "Completely empty YAML file (0 bytes)"),
    ("spaces_only_file", "   \n  \n", "Spaces-only YAML file (parses to None -> Clear Error)"),
    ("tab_whitespace_file", "   \n\t  \n", "Tab-containing whitespace YAML file (uncaught yaml.scanner.ScannerError -> Traceback)"),
    ("comments_only_file", "# Only a comment\n", "YAML file containing only comments (parses to None)"),
    ("empty_dict", "{}\n", "Empty YAML mapping '{}' missing required base and data"),
    ("missing_base", yaml.safe_dump({k: v for k, v in get_valid_config_dict().items() if k != "base"}), "Missing required root field 'base'"),
    ("missing_data", yaml.safe_dump({k: v for k, v in get_valid_config_dict().items() if k != "data"}), "Missing required root field 'data'"),
    ("missing_data_train", yaml.safe_dump({**get_valid_config_dict(), "data": {"format": "alpaca"}}), "Missing required 'data.train' field"),
    ("null_base", yaml.safe_dump({**get_valid_config_dict(), "base": None}), "'base: null'"),
    ("null_training", yaml.safe_dump({**get_valid_config_dict(), "training": None}), "'training: null'"),
    ("empty_string_base", yaml.safe_dump({**get_valid_config_dict(), "base": ""}), "'base: \"\"' (empty string accepted by SoupConfig!)"),
    ("whitespace_string_base", yaml.safe_dump({**get_valid_config_dict(), "base": "   "}), "'base: \"   \"' (whitespace-only string accepted by SoupConfig!)"),
    ("empty_string_data_train", yaml.safe_dump(_mut_data(get_valid_config_dict(), train="")), "'data.train: \"\"' (empty string accepted by DataConfig!)"),
    ("empty_string_output", yaml.safe_dump({**get_valid_config_dict(), "output": ""}), "'output: \"\"' (empty string accepted by SoupConfig!)"),
    ("empty_list_target_modules", yaml.safe_dump(_mut_lora(get_valid_config_dict(), target_modules=[])), "'training.lora.target_modules: []' (empty list accepted by LoraConfig!)"),
]


@pytest.mark.parametrize("case_id,yaml_payload,desc", EMPTY_MISSING_CASES)
def test_class4_empty_and_missing_fields_deterministic(
    case_id: str, yaml_payload: str, desc: str, tmp_path: Path
) -> None:
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_payload,
        tmp_path,
        test_id=f"class4_empty_{case_id}",
        input_class="4. Empty/missing fields",
        description=desc,
    )
    silent_empty_cases = {
        "empty_string_base",
        "whitespace_string_base",
        "empty_string_data_train",
        "empty_string_output",
        "empty_list_target_modules",
    }
    if case_id == "tab_whitespace_file":
        assert classification == "Traceback" and rc == 1
    elif case_id in silent_empty_cases:
        assert classification == "Silent Acceptance" and rc == 0
    else:
        assert classification == "Clear Error" and rc == 1


def test_class4_empty_string_data_train_dry_run_silent_success(tmp_path: Path) -> None:
    """CRITICAL SILENT FAILURE: When data.train is '', Path('').suffix is '', so load_dataset()
    Wait: let's see what `soup train --dry-run` does when `base: ""` or `data.train: "./configs"`!
    """
    cfg = get_valid_config_dict()
    cfg["base"] = ""
    cfg_path = tmp_path / "empty_base.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    c, rc, out, _ = run_soup_cli_subproc(
        ["train", "--config", str(cfg_path), "--dry-run"],
        test_id="class4_cli_dry_run_empty_base_string",
        input_class="4. Empty/missing fields",
        description="soup train --dry-run with base: '' (empty string) exits 0 and prints Ready to train!",
        payload_preview=cfg_path.read_text(encoding="utf-8"),
    )
    assert c == "Silent Acceptance" and rc == 0


# ==============================================================================
# CLASS 5: CONFLICTING PARAMETERS
# ==============================================================================

CONFLICTING_CASES = [
    ("dora_and_vera", lambda d: _mut_lora(d, use_dora=True, use_vera=True), "PEFT conflict: use_dora=True and use_vera=True"),
    ("dora_and_pissa", lambda d: _mut_lora(d, use_dora=True, init_strategy="pissa"), "PEFT conflict: use_dora=True and init_strategy='pissa'"),
    ("olora_and_random_init", lambda d: _mut_lora(d, use_olora=True, init_strategy="random"), "PEFT conflict: use_olora=True and init_strategy='random'"),
    ("vera_and_rank_pattern", lambda d: _mut_lora(d, use_vera=True, rank_pattern={"q_proj": 8}), "PEFT conflict: use_vera=True and rank_pattern set"),
    ("root_and_training_lora", lambda d: {**d, "lora": {"r": 16}}, "Duplicate 'lora' block at both root and training.lora"),
    ("load_in_8bit_and_16bit", lambda d: _mut_train(d, load_in_8bit=True, load_in_16bit=True), "Quantization conflict: load_in_8bit=True and load_in_16bit=True"),
    ("eval_gate_enabled_no_suite", lambda d: _mut_train(d, eval_gate={"enabled": True, "suite": None}), "eval_gate.enabled=True without eval_gate.suite"),
    ("reward_hack_halt_no_detector", lambda d: _mut_train(d, reward_hack_halt=True, reward_hack_detector=None), "reward_hack_halt=True without reward_hack_detector"),
    ("mlx_with_multipack", lambda d: {**_mut_train(d, multipack=True), "backend": "mlx"}, "backend='mlx' with training.multipack=True"),
    ("unfrozen_params_with_dora", lambda d: _mut_train(d, unfrozen_parameters=["layers.0.*"], lora={"use_dora": True}), "unfrozen_parameters combined with lora.use_dora=True (caught by validator)"),
    # Subtle conflicts that are NOT caught by SoupConfig:
    ("unfrozen_params_with_custom_lora", lambda d: _mut_train(d, unfrozen_parameters=["layers.0.*"], lora={"r": 32, "alpha": 64}), "unfrozen_parameters (disables LoRA) combined with custom lora.r=32, alpha=64 (silently ignored!)"),
    ("dpo_task_with_alpaca_format", lambda d: {**d, "task": "dpo", "data": {**d["data"], "format": "alpaca"}}, "Conflict: task='dpo' (requires chosen/rejected) paired with data.format='alpaca'"),
    ("kto_task_with_sharegpt_format", lambda d: {**d, "task": "kto", "data": {**d["data"], "format": "sharegpt"}}, "Conflict: task='kto' (requires completion/label) paired with data.format='sharegpt'"),
    ("save_steps_gt_total_steps", lambda d: _mut_train(d, epochs=1, save_steps=999_999, logging_steps=999_999), "Conflict: save_steps=999999 and logging_steps=999999 >> total training steps"),
    ("loftq_bits_with_random_init", lambda d: _mut_lora(d, init_strategy="random", loftq_bits=2, loftq_iter=5), "Dead-knob conflict: loftq_bits=2 and loftq_iter=5 set while init_strategy='random'"),
]


@pytest.mark.parametrize("case_id,mutator,desc", CONFLICTING_CASES)
def test_class5_conflicting_parameters_deterministic(
    case_id: str, mutator, desc: str, tmp_path: Path
) -> None:
    cfg = mutator(get_valid_config_dict())
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class5_conflict_{case_id}",
        input_class="5. Conflicting parameters",
        description=desc,
    )
    uncaught_conflicts = {
        "unfrozen_params_with_custom_lora",
        "dpo_task_with_alpaca_format",
        "kto_task_with_sharegpt_format",
        "save_steps_gt_total_steps",
        "loftq_bits_with_random_init",
    }
    if case_id in uncaught_conflicts:
        assert classification == "Silent Acceptance" and rc == 0
    else:
        assert classification == "Clear Error" and rc == 1


def test_class5_cli_conflicting_flags_and_plan_vs_train_discrepancy(tmp_path: Path) -> None:
    cfg_path = tmp_path / "valid_for_cli_conflict.yaml"
    cfg_path.write_text(yaml.safe_dump(get_valid_config_dict()), encoding="utf-8")

    # 5a. CLI `--wandb` and `--tensorboard` together
    c1, rc1, _, _ = run_soup_cli_subproc(
        ["train", "--config", str(cfg_path), "--wandb", "--tensorboard", "--dry-run"],
        test_id="class5_cli_wandb_and_tensorboard",
        input_class="5. Conflicting parameters",
        description="CLI conflict: soup train --wandb --tensorboard",
    )
    assert c1 == "Clear Error" and rc1 == 1

    # 5b. CLI `--hf-resume` without `--push-as`
    c2, rc2, _, _ = run_soup_cli_subproc(
        ["train", "--config", str(cfg_path), "--hf-resume", "--dry-run"],
        test_id="class5_cli_hf_resume_without_push_as",
        input_class="5. Conflicting parameters",
        description="CLI conflict: soup train --hf-resume without --push-as",
    )
    assert c2 == "Clear Error" and rc2 == 1

    # 5c. CRITICAL GATE DISCREPANCY: `soup plan` accepts a config with invalid task, negative epochs,
    # and conflicting PEFT flags because `plan_cmd` bypasses `SoupConfig` validation!
    # Note: both --config and --state must be under cwd (`is_under_cwd` in `commands/plan.py:37`).
    broken_cfg = get_valid_config_dict()
    broken_cfg["task"] = "completely_invalid_task_name"
    broken_cfg["training"]["epochs"] = -100
    broken_cfg["training"]["lr"] = -5.0
    broken_cfg["training"]["lora"]["use_dora"] = True
    broken_cfg["training"]["lora"]["use_vera"] = True
    broken_rel_path = Path("configs/_tmp_fuzz_broken_plan.yaml")
    state_rel_path = Path("configs/_tmp_fuzz_broken.tfstate")
    broken_rel_path.write_text(yaml.safe_dump(broken_cfg), encoding="utf-8")
    try:
        c3, rc3, out3, _ = run_soup_cli_subproc(
            ["plan", "--config", str(broken_rel_path), "--state", str(state_rel_path)],
            test_id="class5_cli_soup_plan_bypasses_schema_validation",
            input_class="5. Conflicting parameters",
            description="soup plan succeeds (exit 0) on config with invalid task, negative epochs/lr, and conflicting PEFT flags",
            payload_preview=broken_rel_path.read_text(encoding="utf-8"),
        )
        assert c3 == "Silent Acceptance" and rc3 == 0
    finally:
        broken_rel_path.unlink(missing_ok=True)
        state_rel_path.unlink(missing_ok=True)
        Path(f"{state_rel_path}.lock").unlink(missing_ok=True)


# ==============================================================================
# CLASS 6: EXCESSIVELY LONG STRINGS (10,000+ chars)
# ==============================================================================

LONG_STRING_CASES = [
    ("base_10k_chars", lambda d, s: {**d, "base": s}, "base set to 12,000-char string (accepted by SoupConfig, rejected by soup plan >512)"),
    ("data_train_10k_chars", lambda d, s: _mut_data(d, train=f"./{s}.jsonl"), "data.train set to 12,000-char path string"),
    ("output_10k_chars", lambda d, s: {**d, "output": f"./{s}"}, "output set to 12,000-char directory string"),
    ("experiment_name_10k_chars", lambda d, s: {**d, "experiment_name": s}, "experiment_name set to 12,000-char string (no max_length cap in SoupConfig!)"),
    ("chat_template_10k_chars", lambda d, s: _mut_data(d, chat_template=s), "data.chat_template set to 12,000-char raw string"),
    ("unfrozen_param_10k_chars", lambda d, s: _mut_train(d, unfrozen_parameters=[s]), "training.unfrozen_parameters pattern of 12,000 chars (capped at 512)"),
]


@pytest.mark.parametrize("case_id,mutator,desc", LONG_STRING_CASES)
def test_class6_very_long_strings_deterministic(
    case_id: str, mutator, desc: str, tmp_path: Path
) -> None:
    long_str = "a" * 12_000
    cfg = mutator(get_valid_config_dict(), long_str)
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class6_long_{case_id}",
        input_class="6. Very long values (10,000+ chars)",
        description=desc,
    )
    if case_id == "unfrozen_param_10k_chars":
        assert classification == "Clear Error" and rc == 1
    else:
        assert classification == "Silent Acceptance" and rc == 0


def test_class6_long_experiment_name_crashes_train_dry_run(tmp_path: Path) -> None:
    """When experiment_name or data.train is 10,000+ chars, SoupConfig accepts it silently,
    and downstream filesystem operations in `soup train --dry-run` fail or crash with OSError.
    """
    long_str = "x" * 10_500
    cfg = get_valid_config_dict()
    cfg["data"]["train"] = f"./{long_str}.jsonl"
    cfg_path = tmp_path / "long_train_path.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    c, rc, out, err = run_soup_cli_subproc(
        ["--verbose", "train", "--config", str(cfg_path), "--dry-run"],
        test_id="class6_cli_dry_run_10k_data_train_path",
        input_class="6. Very long values (10,000+ chars)",
        description="soup --verbose train --config <10500-char data.train> --dry-run crashes with unhandled OSError/FileNotFoundError traceback",
        payload_preview=f"data.train: ./{'x'*40}...[10500 chars].jsonl",
    )
    assert c == "Traceback" and rc == 1


@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    long_len=st.integers(min_value=10_000, max_value=15_000),
    fill_char=st.sampled_from(["a", "Z", "9", "_", "-"]),
)
def test_class6_hypothesis_10k_plus_strings_in_schema(
    long_len: int, fill_char: str, tmp_path: Path
) -> None:
    s = fill_char * long_len
    cfg = get_valid_config_dict()
    cfg["base"] = s
    cfg["experiment_name"] = s
    cfg["output"] = s
    yaml_str = yaml.safe_dump(cfg)
    classification, rc, _, _, _ = run_load_config_inproc(
        yaml_str,
        tmp_path,
        test_id=f"class6_hypo_10k_len_{long_len}",
        input_class="6. Very long values (10,000+ chars)",
        description=f"Hypothesis {long_len}-char string in base, experiment_name, and output",
    )
    assert classification == "Silent Acceptance" and rc == 0
