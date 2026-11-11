"""Phase 3: Local Silent Failure Probes (Automated Pytest Suite).

Verifies grey-box silent failures in `soup-cli` (v0.72.4) where invalid inputs, corrupted datasets,
or broken training states exit with status code 0 ("success") or silently ignore user settings:
  1. Ignored misspelled/alias config keys (`learning_rate`, `epchs`, `lora_r`, `max_steps`) -> exit 0
  2. Missing `seed` in `TrainingConfig` -> `training.seed` in `soup.yaml` is silently dropped,
     `getattr(cfg.training, "seed", None)` is always `None`, and `--repro-receipt` writes `"seeds": {}`
  3. `soup data validate` exits 0 even when 100% of rows are invalid or dataset is empty
  4. `soup train --dry-run` exits 0 ("Data OK: 0 train samples", "Config valid. Ready to train!")
     when 100% of JSONL lines are malformed JSON or have mismatched column names (`question`/`answer`
     instead of `instruction`/`output`)
  5. Single-row dataset with `val_split: 0.2` truncates `train` split to 0 rows (`int(1 * 0.8) == 0`)
     while `soup train --dry-run` still exits 0
  6. `SoupTrainerCallback` loss watchdog fails to trigger on `NaN` loss because `float('nan') > threshold`
     evaluates to `False` in Python (`monitoring/callback.py:232`)
  7. `soup plan` exits 0 and writes state on completely invalid configs (`task: invalid`, `epochs: -100`)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from soup_cli.config.loader import load_config
from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.data.loader import load_dataset
from soup_cli.monitoring.callback import SoupTrainerCallback
from soup_cli.monitoring.display import TrainingDisplay

RAW_LOG_DIR = Path("raw_logs")
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
SILENT_FAILURES_JSONL = RAW_LOG_DIR / "03_silent_failures_cases.jsonl"


def record_silent_probe(
    *,
    probe_id: str,
    title: str,
    command_or_api: str,
    exit_code: int,
    silent_failure_confirmed: bool,
    details: dict[str, Any],
) -> None:
    entry = {
        "probe_id": probe_id,
        "title": title,
        "command_or_api": command_or_api,
        "exit_code": exit_code,
        "silent_failure_confirmed": silent_failure_confirmed,
        "details": details,
    }
    with open(SILENT_FAILURES_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_soup_subproc(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    env["SOUP_NO_AUDIT_LOG"] = "1"
    soup_exe = Path(sys.executable).parent / ("soup.exe" if os.name == "nt" else "soup")
    cmd = [str(soup_exe)] + args if soup_exe.exists() else [sys.executable, "-m", "soup_cli.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
    )


def test_sf01_silent_ignored_typo_and_alias_hyperparameters(tmp_path: Path) -> None:
    """SF-01: User specifies `learning_rate: 0.0`, `epchs: 50`, `max_steps: 10`, `lora_r: 4` in soup.yaml.
    Because `SoupConfig`, `TrainingConfig`, and `LoraConfig` omit `extra='forbid'`, all four keys are
    silently discarded and `soup train --dry-run` exits 0 using default `lr=2e-5`, `epochs=3`, `r=64`.
    """
    cfg_dict = {
        "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
        "task": "sft",
        "data": {
            "train": "configs/sample_train.jsonl",
            "format": "alpaca",
        },
        "training": {
            "learning_rate": 0.0,   # Ignored alias! Default lr=2e-5 used instead.
            "epchs": 50,            # Ignored typo! Default epochs=3 used instead.
            "max_steps": 10,        # Ignored key!
            "lora": {
                "lora_r": 4,        # Ignored alias! Default r=64 used instead.
            },
        },
    }
    cfg_path = tmp_path / "sf01_typo_hyperparams.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    parsed = load_config(cfg_path)
    proc = run_soup_subproc(["train", "--config", str(cfg_path), "--dry-run"])

    record_silent_probe(
        probe_id="SF-01",
        title="Misspelled / alias hyperparameters silently ignored with exit code 0",
        command_or_api=f"soup train --config {cfg_path.name} --dry-run",
        exit_code=proc.returncode,
        silent_failure_confirmed=(
            proc.returncode == 0
            and parsed.training.lr == 2e-5
            and parsed.training.epochs == 3
            and parsed.training.lora.r == 64
        ),
        details={
            "effective_lr": parsed.training.lr,
            "effective_epochs": parsed.training.epochs,
            "effective_lora_r": parsed.training.lora.r,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
    )
    assert proc.returncode == 0
    assert parsed.training.lr == 2e-5
    assert parsed.training.epochs == 3
    assert parsed.training.lora.r == 64
    assert "Config valid. Ready to train!" in proc.stdout


def test_sf02_seed_field_missing_from_schema_and_repro_receipt_empty(tmp_path: Path) -> None:
    """SF-02: `soup_cli/commands/train.py:1668`, `sft.py:585`, `pretrain.py:217`, and `stream_setup.py:305`
    all call `getattr(tcfg, 'seed', None)`, and `soup train --help` advertises `--repro-receipt` as
    recording seeds (`seeds, git SHA, config SHA-256`).
    However, `TrainingConfig` in `soup_cli/config/schema.py` has NO `seed` field!
    Setting `training.seed: 12345` in `soup.yaml` is silently dropped, `getattr(cfg.training, 'seed', None)`
    returns `None`, and `_write_repro_receipt` writes `"seeds": {}`.
    """
    from soup_cli.commands.train import _write_repro_receipt

    cfg_dict = {
        "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
        "task": "sft",
        "data": {"train": "configs/sample_train.jsonl", "format": "alpaca"},
        "training": {"epochs": 1, "seed": 12345},
    }
    cfg_path = tmp_path / "sf02_seed.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    parsed = load_config(cfg_path)
    seed_attr = getattr(parsed.training, "seed", None)

    receipt_rel = Path("configs/_tmp_sf02_receipt.json")
    try:
        _write_repro_receipt(str(receipt_rel), "run_sf02_test", parsed)
        receipt_data = json.loads(receipt_rel.read_text(encoding="utf-8"))
    finally:
        receipt_rel.unlink(missing_ok=True)

    record_silent_probe(
        probe_id="SF-02",
        title="training.seed silently dropped by TrainingConfig; --repro-receipt writes empty seeds {}",
        command_or_api="load_config() + _write_repro_receipt()",
        exit_code=0,
        silent_failure_confirmed=("seed" not in TrainingConfig.model_fields and seed_attr is None and receipt_data.get("seeds") == {}),
        details={
            "seed_in_TrainingConfig_fields": "seed" in TrainingConfig.model_fields,
            "getattr_training_seed": seed_attr,
            "repro_receipt_seeds": receipt_data.get("seeds"),
            "repro_receipt_full": receipt_data,
        },
    )
    assert "seed" not in TrainingConfig.model_fields
    assert seed_attr is None
    assert receipt_data["seeds"] == {}


def test_sf03_data_validate_exits_zero_on_empty_and_invalid_datasets(tmp_path: Path) -> None:
    """SF-03: `soup data validate` prints `Issues found: ! Dataset is empty` or
    `! 5 rows missing required keys: {'output', 'instruction'}` and `0/5 rows valid`,
    yet exits with returncode 0 (`commands/data.py:106-116` never raises `typer.Exit(1)`).
    """
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    proc_empty = run_soup_subproc(["data", "validate", str(empty_jsonl), "--format", "alpaca"])

    wrong_cols_jsonl = tmp_path / "wrong_cols.jsonl"
    wrong_cols_jsonl.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "answer": f"A{i}"}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    proc_wrong = run_soup_subproc(["data", "validate", str(wrong_cols_jsonl), "--format", "alpaca"])

    record_silent_probe(
        probe_id="SF-03",
        title="soup data validate exits 0 when dataset is empty or 0/N rows are valid",
        command_or_api="soup data validate <empty_or_invalid.jsonl> --format alpaca",
        exit_code=proc_wrong.returncode,
        silent_failure_confirmed=(proc_empty.returncode == 0 and proc_wrong.returncode == 0),
        details={
            "empty_returncode": proc_empty.returncode,
            "empty_stdout": proc_empty.stdout,
            "wrong_cols_returncode": proc_wrong.returncode,
            "wrong_cols_stdout": proc_wrong.stdout,
        },
    )
    assert proc_empty.returncode == 0
    assert "Dataset is empty" in proc_empty.stdout
    assert proc_wrong.returncode == 0
    assert "0/5 rows valid" in proc_wrong.stdout


def test_sf04_corrupted_jsonl_and_wrong_columns_pass_train_dry_run_with_0_samples(tmp_path: Path) -> None:
    """SF-04: When 100% of lines in `train.jsonl` are invalid JSON or have wrong column names
    (e.g., `question`/`answer` when `format: alpaca` expects `instruction`/`output`),
    `load_dataset()` skips every row and returns `{'train': [], 'val': []}` (`0` rows).
    `soup train --dry-run` prints `Data OK: 0 train samples` and `Config valid. Ready to train!`
    and exits with status code 0!
    """
    corrupt_jsonl = tmp_path / "all_corrupt.jsonl"
    corrupt_jsonl.write_text("{not valid json line 1\n{broken json line 2\n", encoding="utf-8")

    wrong_cols_jsonl = tmp_path / "mismatched_columns.jsonl"
    wrong_cols_jsonl.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "answer": f"A{i}"}) for i in range(10)) + "\n",
        encoding="utf-8",
    )

    cfg_corrupt = tmp_path / "cfg_corrupt.yaml"
    cfg_corrupt.write_text(
        yaml.safe_dump(
            {
                "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
                "task": "sft",
                "data": {"train": str(corrupt_jsonl), "format": "alpaca"},
            }
        ),
        encoding="utf-8",
    )

    cfg_wrong_cols = tmp_path / "cfg_wrong_cols.yaml"
    cfg_wrong_cols.write_text(
        yaml.safe_dump(
            {
                "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
                "task": "sft",
                "data": {"train": str(wrong_cols_jsonl), "format": "alpaca"},
            }
        ),
        encoding="utf-8",
    )

    proc_corrupt = run_soup_subproc(["train", "--config", str(cfg_corrupt), "--dry-run"])
    proc_wrong = run_soup_subproc(["train", "--config", str(cfg_wrong_cols), "--dry-run"])

    record_silent_probe(
        probe_id="SF-04",
        title="100% corrupted JSONL or mismatched columns yield 0 train samples but pass soup train --dry-run (exit 0)",
        command_or_api="soup train --config <corrupt_or_wrong_cols.yaml> --dry-run",
        exit_code=proc_wrong.returncode,
        silent_failure_confirmed=(
            proc_corrupt.returncode == 0
            and "Data OK: 0 train samples" in proc_corrupt.stdout
            and proc_wrong.returncode == 0
            and "Data OK: 0 train samples" in proc_wrong.stdout
        ),
        details={
            "corrupt_returncode": proc_corrupt.returncode,
            "corrupt_stdout": proc_corrupt.stdout,
            "wrong_cols_returncode": proc_wrong.returncode,
            "wrong_cols_stdout": proc_wrong.stdout,
        },
    )
    assert proc_corrupt.returncode == 0
    assert "Data OK: 0 train samples" in proc_corrupt.stdout
    assert proc_wrong.returncode == 0
    assert "Data OK: 0 train samples" in proc_wrong.stdout


def test_sf05_single_row_dataset_val_split_truncates_train_to_zero_rows(tmp_path: Path) -> None:
    """SF-05: In `soup_cli/data/loader.py:180` (`_finalize`), `split_idx = int(len(raw_data) * (1 - val_split))`.
    When `len(raw_data) == 1` and `val_split == 0.2`, `split_idx = int(1 * 0.8) = 0`, placing 0 rows in
    `train` and 1 row in `val`! `soup train --dry-run` still prints `Data OK: 0 train samples, 1 val samples`
    and exits 0.
    """
    one_row_jsonl = tmp_path / "one_row.jsonl"
    one_row_jsonl.write_text(
        json.dumps({"instruction": "Say hi", "input": "", "output": "Hello!"}) + "\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "one_row_cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "base": "trl-internal-testing/tiny-random-LlamaForCausalLM",
                "task": "sft",
                "data": {"train": str(one_row_jsonl), "format": "alpaca", "val_split": 0.2},
            }
        ),
        encoding="utf-8",
    )
    parsed = load_config(cfg_path)
    ds = load_dataset(parsed.data)
    proc = run_soup_subproc(["train", "--config", str(cfg_path), "--dry-run"])

    record_silent_probe(
        probe_id="SF-05",
        title="1-row dataset with val_split=0.2 truncates train split to 0 rows and exits 0 in --dry-run",
        command_or_api="load_dataset() + soup train --dry-run",
        exit_code=proc.returncode,
        silent_failure_confirmed=(len(ds["train"]) == 0 and len(ds["val"]) == 1 and proc.returncode == 0),
        details={
            "train_len": len(ds["train"]),
            "val_len": len(ds["val"]),
            "stdout": proc.stdout,
        },
    )
    assert len(ds["train"]) == 0
    assert len(ds["val"]) == 1
    assert proc.returncode == 0


def test_sf06_loss_watchdog_ignores_nan_loss_due_to_ieee754_comparison() -> None:
    """SF-06: In `soup_cli/monitoring/callback.py:232`, `SoupTrainerCallback.on_log` checks:
        `if float(loss) > self.loss_watchdog_threshold:`
    In IEEE-754 floating point, `float('nan') > threshold` is ALWAYS `False`!
    Therefore, even when a user explicitly enables `loss_watchdog: true`, a `NaN` training loss
    NEVER increments `_loss_spike_count` and NEVER sets `control.should_training_stop = True`.
    """
    cfg = SoupConfig(
        base="some-model",
        data={"train": "configs/sample_train.jsonl"},
    )
    display = TrainingDisplay(cfg, device_name="cpu")
    cb = SoupTrainerCallback(
        display=display,
        loss_watchdog=True,
        loss_watchdog_threshold=3.0,
        loss_watchdog_patience=2,
    )

    class DummyState:
        global_step = 10
        epoch = 1.0

    class DummyControl:
        should_training_stop = False

    state = DummyState()
    control = DummyControl()

    # Feed 5 consecutive NaN loss logs — watchdog should stop after patience=2, but never triggers!
    for step in range(1, 6):
        state.global_step = step
        cb.on_log(args=None, state=state, control=control, logs={"loss": float("nan")})

    record_silent_probe(
        probe_id="SF-06",
        title="SoupTrainerCallback loss_watchdog fails to detect NaN loss (float('nan') > threshold is False)",
        command_or_api="SoupTrainerCallback(loss_watchdog=True, patience=2).on_log(logs={'loss': nan})",
        exit_code=0,
        silent_failure_confirmed=(cb._watchdog_counter == 0 and cb._watchdog_fired is False and control.should_training_stop is False),
        details={
            "steps_with_nan_loss": 5,
            "watchdog_patience": 2,
            "final_watchdog_counter": cb._watchdog_counter,
            "watchdog_fired": cb._watchdog_fired,
            "should_training_stop": control.should_training_stop,
        },
    )
    assert cb._watchdog_counter == 0
    assert cb._watchdog_fired is False
    assert control.should_training_stop is False
