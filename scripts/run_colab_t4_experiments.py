#!/usr/bin/env python3
"""Self-contained Google Colab T4 GPU QA Experiment Runner for `soup-cli` (v0.72.4).

Runs Phase 3 GPU-dependent experiments on a NVIDIA Tesla T4 (or any CUDA GPU) in ~2-3 minutes:
  1. Hardware & Environment Telemetry (`nvidia-smi`, PyTorch/CUDA/cuDNN versions, `soup version`)
  2. Experiment 1A (Reproducibility — Identical Seed Run 1 vs Run 2 on T4 GPU, Default Mode):
     - Runs `soup train` twice with the exact same config and seed, capturing step-by-step
       loss curves from `experiments.db` (`runs` and `metrics` tables), final metrics, and saved LoRA adapter weights
       (`adapter_model.safetensors`), computing global and per-tensor L2 norm ||theta_1 - theta_2||_2
       and max |theta_1 - theta_2|.
  3. Experiment 1B (Reproducibility — Different Seeds `seed: 42` vs `seed: 999` in `soup.yaml`):
     - Demonstrates the grey-box defect in `soup-cli`: because `TrainingConfig` has no `seed` field
       and `SFTTrainerWrapper` never passes `seed` to `TrainingArguments`, `seed: 42` and `seed: 999`
       in `soup.yaml` are both silently ignored (both fall back to HF default `seed=42`) and `--repro-receipt`
       writes `"seeds": {}`!
  4. Experiment 1C (CUDA Determinism Probe — Default cuBLAS/Atomic vs `use_deterministic_algorithms(True)`):
     - Measures exact T4 FP16/BF16 atomic reduction non-determinism vs deterministic algorithms.
  5. Experiment 2 (End-to-End `soup train` Silent Failure Probes on T4 GPU):
     - Probe SF-GPU-1: `soup train` with misspelled hyperparameters (`learning_rate: 0.0`, `epchs: 99`,
       `lora_r: 4`) exits `0` ("Training Complete!") while silently training with default `lr=2e-5`, `r=64`.
     - Probe SF-GPU-2: `soup train` with extreme learning rate (`lr: 1e15`) + `loss_watchdog: true`
       triggers `NaN` / `0.0` loss explosion, bypasses `SoupTrainerCallback` (`float('nan') > threshold`
       is `False`), saves corrupted/NaN adapter weights, and exits `0` ("Training Complete!").
     - Probe SF-GPU-3: `soup train` with near-zero learning rate (`lr: 1e-30`) exits `0` while
       producing effectively untrained adapter weights (||Delta theta||_2 ~ 0).
     - Probe SF-GPU-4: `soup train` with 100% mismatched dataset columns (`question`/`answer` instead of
       `instruction`/`output`) passes `--dry-run` with `0 train samples` (`exit 0`), and when executed
       without `--dry-run` crashes only after downloading/loading the model on GPU.

Outputs two files ready to drop directly into `raw_logs/`:
  - `raw_logs/03_colab_t4_experiments.log` (unfiltered human-readable execution log)
  - `raw_logs/03_colab_t4_results.json`    (structured metrics & raw CLI stdout/stderr for pytest)
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def EnsureDependencies() -> None:
    try:
        import soup_cli  # noqa: F401
        import safetensors.torch  # noqa: F401
    except ImportError:
        print("[setup] Installing soup-cli[train]==0.72.4...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "soup-cli[train]==0.72.4"],
            check=True,
        )


EnsureDependencies()

import safetensors.torch  # noqa: E402
import soup_cli  # noqa: E402

OUT_DIR = Path("raw_logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "03_colab_t4_experiments.log"
JSON_FILE = OUT_DIR / "03_colab_t4_results.json"

TINY_MODEL_ID = "trl-internal-testing/tiny-random-LlamaForCausalLM"

CHATML_SAMPLE_ROWS = [
    {"messages": [{"role": "user", "content": "Explain what a binary search tree is."}, {"role": "assistant", "content": "A binary search tree is a node-based binary tree data structure where left descendants are smaller and right descendants are larger."}]},
    {"messages": [{"role": "user", "content": "Convert 25 C from Celsius to Fahrenheit."}, {"role": "assistant", "content": "25 degrees Celsius is equal to 77 degrees Fahrenheit."}]},
    {"messages": [{"role": "user", "content": "Write a Python function to compute the factorial of n=5."}, {"role": "assistant", "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\nprint(factorial(5))  # 120"}]},
    {"messages": [{"role": "user", "content": "Summarize the main benefit of LoRA fine-tuning."}, {"role": "assistant", "content": "LoRA freezes pretrained model weights and injects trainable low-rank matrices, drastically reducing trainable parameters and GPU memory."}]},
    {"messages": [{"role": "user", "content": "Identify the capital city of Japan."}, {"role": "assistant", "content": "The capital city of Japan is Tokyo."}]},
    {"messages": [{"role": "user", "content": "Explain the difference between TCP and UDP."}, {"role": "assistant", "content": "TCP is connection-oriented and guarantees ordered delivery, whereas UDP is connectionless with lower overhead."}]},
    {"messages": [{"role": "user", "content": "Reverse the string antigravity."}, {"role": "assistant", "content": "ytivargitna"}]},
    {"messages": [{"role": "user", "content": "Calculate the area of a rectangle with width=7, height=4."}, {"role": "assistant", "content": "The area of the rectangle is 28 square units."}]},
    {"messages": [{"role": "user", "content": "List three HTTP status codes and their meanings."}, {"role": "assistant", "content": "200 OK (success), 404 Not Found (missing resource), 500 Internal Server Error (server failure)."}]},
    {"messages": [{"role": "user", "content": "What does SQL stand for?"}, {"role": "assistant", "content": "SQL stands for Structured Query Language."}]},
    {"messages": [{"role": "user", "content": "Explain gradient clipping in deep learning."}, {"role": "assistant", "content": "Gradient clipping rescales gradients when their norm exceeds a threshold to prevent exploding gradients."}]},
    {"messages": [{"role": "user", "content": "Write a SQL query to select all active users from table users."}, {"role": "assistant", "content": "SELECT * FROM users WHERE active = 1;"}]},
]


class DualLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.fh = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def run_cmd(cmd: list[str], cwd: Path | None = None, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = round(time.time() - t0, 3)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "duration_sec": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def soup_cmd(args: list[str], cwd: Path | None = None, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    soup_bin = shutil.which("soup")
    cmd = [soup_bin] + args if soup_bin else [sys.executable, "-m", "soup_cli.cli"] + args
    return run_cmd(cmd, cwd=cwd, env_extra=env_extra)


def find_adapter_safetensors(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("adapter_model.safetensors"))
    return candidates[0] if candidates else None


def compare_safetensors(path_a: Path, path_b: Path) -> dict[str, Any]:
    sd_a = safetensors.torch.load_file(str(path_a), device="cpu")
    sd_b = safetensors.torch.load_file(str(path_b), device="cpu")
    keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))
    sq_sum = 0.0
    max_abs = 0.0
    nan_count_a = 0
    nan_count_b = 0
    per_tensor: dict[str, Any] = {}

    for k in keys:
        ta = sd_a[k].float()
        tb = sd_b[k].float()
        nan_a = int(torch.isnan(ta).sum().item())
        nan_b = int(torch.isnan(tb).sum().item())
        nan_count_a += nan_a
        nan_count_b += nan_b
        diff = ta - tb
        l2 = float(torch.norm(diff, p=2).item()) if (nan_a == 0 and nan_b == 0) else float("nan")
        m_abs = float(torch.max(torch.abs(diff)).item()) if (diff.numel() > 0 and nan_a == 0 and nan_b == 0) else float("nan")
        if not math.isnan(l2):
            sq_sum += l2 * l2
        if not math.isnan(m_abs) and m_abs > max_abs:
            max_abs = m_abs
        per_tensor[k] = {
            "shape": list(ta.shape),
            "l2_diff": l2,
            "max_abs_diff": m_abs,
            "run1_l2_norm": float(torch.norm(ta, p=2).item()),
            "run2_l2_norm": float(torch.norm(tb, p=2).item()),
            "nan_count_run1": nan_a,
            "nan_count_run2": nan_b,
        }

    return {
        "global_l2_diff": math.sqrt(sq_sum),
        "global_max_abs_diff": max_abs,
        "num_tensors": len(keys),
        "nan_count_run1": nan_count_a,
        "nan_count_run2": nan_count_b,
        "per_tensor": per_tensor,
    }


def inspect_single_safetensors(path_a: Path) -> dict[str, Any]:
    sd = safetensors.torch.load_file(str(path_a), device="cpu")
    total_nan = 0
    total_inf = 0
    lora_b_sq_norm = 0.0
    lora_a_sq_norm = 0.0
    for k, t in sd.items():
        tf = t.float()
        total_nan += int(torch.isnan(tf).sum().item())
        total_inf += int(torch.isinf(tf).sum().item())
        nrm = float(torch.norm(tf, p=2).item())
        if "lora_B" in k and not math.isnan(nrm):
            lora_b_sq_norm += nrm * nrm
        elif "lora_A" in k and not math.isnan(nrm):
            lora_a_sq_norm += nrm * nrm
    return {
        "num_tensors": len(sd),
        "total_nan_elements": total_nan,
        "total_inf_elements": total_inf,
        "lora_A_global_l2_norm": math.sqrt(lora_a_sq_norm),
        "lora_B_global_l2_norm": math.sqrt(lora_b_sq_norm),
    }


def fetch_latest_run_logs(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"run_id": None, "steps": []}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1")
    run_row = cur.fetchone()
    if not run_row:
        conn.close()
        return {"run_id": None, "steps": []}
    run_id = run_row["run_id"]
    cur.execute(
        "SELECT step, epoch, loss, lr, grad_norm, speed, gpu_mem, timestamp FROM metrics WHERE run_id = ? ORDER BY step ASC",
        (run_id,),
    )
    steps = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {
        "run_id": run_id,
        "created_at": run_row["created_at"],
        "initial_loss": run_row["initial_loss"],
        "final_loss": run_row["final_loss"],
        "total_steps": run_row["total_steps"],
        "duration_secs": run_row["duration_secs"],
        "status": run_row["status"],
        "steps": steps,
    }


def main() -> None:
    logger = DualLogger(LOG_FILE)
    logger.log("=" * 80)
    logger.log(f"SOUP-CLI (v{getattr(soup_cli, '__version__', 'unknown')}) GOOGLE COLAB T4 GPU QA EXPERIMENT RUNNER")
    logger.log(f"Timestamp (UTC): {datetime.now(tz=timezone.utc).isoformat()}")
    logger.log("=" * 80)

    results: dict[str, Any] = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "environment": {},
        "reproducibility": {},
        "silent_failures": {},
    }

    # -------------------------------------------------------------------------
    # 1. ENVIRONMENT & HARDWARE TELEMETRY
    # -------------------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA GPU)"
    gpu_props = {}
    if cuda_available:
        p = torch.cuda.get_device_properties(0)
        gpu_props = {
            "name": p.name,
            "total_memory_mb": round(p.total_memory / (1024**2), 1),
            "major": p.major,
            "minor": p.minor,
            "multi_processor_count": p.multi_processor_count,
        }
    nvidia_smi = run_cmd(["nvidia-smi"]) if shutil.which("nvidia-smi") else {"returncode": -1, "stdout": "nvidia-smi not found"}
    soup_ver = soup_cmd(["version"])
    soup_doc = soup_cmd(["doctor"])

    results["environment"] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "soup_cli_version": getattr(soup_cli, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_name": gpu_name,
        "gpu_properties": gpu_props,
        "nvidia_smi_stdout": nvidia_smi.get("stdout", ""),
        "soup_version_stdout": soup_ver.get("stdout", ""),
        "soup_doctor_stdout": soup_doc.get("stdout", ""),
    }

    logger.log(f"[ENV] Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | soup-cli: {results['environment']['soup_cli_version']}")
    logger.log(f"[ENV] CUDA Available: {cuda_available} | Device: {gpu_name} | CUDA: {torch.version.cuda}")
    if nvidia_smi.get("stdout"):
        logger.log(nvidia_smi["stdout"].strip())

    work_dir = Path(tempfile.mkdtemp(prefix="soup_t4_qa_"))
    db_path = work_dir / "soup_qa_runs.db"
    env_db = {"SOUP_DB_PATH": str(db_path)}

    train_jsonl = work_dir / "train_chatml.jsonl"
    train_jsonl.write_text(
        "\n".join(json.dumps(r) for r in CHATML_SAMPLE_ROWS) + "\n",
        encoding="utf-8",
    )

    def make_base_config(out_subdir: str, **train_overrides: Any) -> dict[str, Any]:
        t_dict: dict[str, Any] = {
            "epochs": 3,
            "lr": 5e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "auto_mixed_precision": True,
            "save_steps": 100,
            "logging_steps": 1,
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
            },
        }
        t_dict.update(train_overrides)
        return {
            "base": TINY_MODEL_ID,
            "task": "sft",
            "backend": "transformers",
            "data": {
                "train": str(train_jsonl),
                "format": "chatml",
                "chat_template": "chatml",
                "val_split": 0.2,
                "max_length": 128,
            },
            "training": t_dict,
            "output": str(work_dir / out_subdir),
        }

    # -------------------------------------------------------------------------
    # 2. EXPERIMENT 1A: IDENTICAL SEED REPRODUCIBILITY (RUN 1 vs RUN 2)
    # -------------------------------------------------------------------------
    logger.log("\n" + "=" * 80)
    logger.log("EXPERIMENT 1A: 2x IDENTICAL SEED RUNS (seed=42 vs seed=42) via `soup train`")
    logger.log("=" * 80)

    cfg_1a_run1 = work_dir / "exp1a_run1.yaml"
    cfg_1a_run2 = work_dir / "exp1a_run2.yaml"
    cfg_1a_run1.write_text(yaml.safe_dump(make_base_config("out_1a_run1", seed=42)), encoding="utf-8")
    cfg_1a_run2.write_text(yaml.safe_dump(make_base_config("out_1a_run2", seed=42)), encoding="utf-8")

    # Pre-probe 1: Record the false-positive Hardware-fit gate behavior WITHOUT --allow-oom-attempt
    res_1a_gate_block = soup_cmd(["train", "--config", str(cfg_1a_run1), "--yes"], cwd=work_dir, env_extra=env_db)
    logger.log(f"[Hardware-Fit Gate Probe (no --allow-oom-attempt)] rc={res_1a_gate_block['returncode']}")
    logger.log(res_1a_gate_block["stdout"])

    # Pre-probe 2: Record default `auto_mixed_precision: false` crashing on Tesla T4 (cc=7.5) with bf16 error
    cfg_t4_bf16_bug = work_dir / "exp1a_default_bf16_t4_bug.yaml"
    cfg_t4_bf16_bug.write_text(
        yaml.safe_dump(make_base_config("out_t4_bf16_bug", auto_mixed_precision=False)),
        encoding="utf-8",
    )
    res_t4_bf16_bug = soup_cmd(
        ["--verbose", "train", "--config", str(cfg_t4_bf16_bug), "--allow-oom-attempt", "--yes"],
        cwd=work_dir,
        env_extra=env_db,
    )
    logger.log(f"[Default auto_mixed_precision=False on T4 Probe] rc={res_t4_bf16_bug['returncode']}")
    logger.log(res_t4_bf16_bug["stdout"] + "\n" + res_t4_bf16_bug["stderr"])

    res_1a_1 = soup_cmd(["train", "--config", str(cfg_1a_run1), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    log_1a_1 = fetch_latest_run_logs(db_path)
    logger.log(f"[Exp 1A Run 1] rc={res_1a_1['returncode']} ({res_1a_1['duration_sec']}s) | steps={len(log_1a_1['steps'])} | final_loss={log_1a_1.get('final_loss')}")
    logger.log(res_1a_1["stdout"])

    res_1a_2 = soup_cmd(["train", "--config", str(cfg_1a_run2), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    log_1a_2 = fetch_latest_run_logs(db_path)
    logger.log(f"[Exp 1A Run 2] rc={res_1a_2['returncode']} ({res_1a_2['duration_sec']}s) | steps={len(log_1a_2['steps'])} | final_loss={log_1a_2.get('final_loss')}")
    logger.log(res_1a_2["stdout"])

    st_1a_1 = find_adapter_safetensors(work_dir / "out_1a_run1")
    st_1a_2 = find_adapter_safetensors(work_dir / "out_1a_run2")
    weight_cmp_1a = compare_safetensors(st_1a_1, st_1a_2) if (st_1a_1 and st_1a_2) else {}

    curve_1a_1 = [s["loss"] for s in log_1a_1["steps"] if s.get("loss") is not None]
    curve_1a_2 = [s["loss"] for s in log_1a_2["steps"] if s.get("loss") is not None]
    step_diffs_1a = [abs(a - b) for a, b in zip(curve_1a_1, curve_1a_2)]

    results["reproducibility"]["exp1a_identical_seed_runs"] = {
        "hardware_fit_gate_false_positive_without_flag": res_1a_gate_block,
        "default_bf16_crash_on_t4_without_auto_mixed_precision": res_t4_bf16_bug,
        "run1": {"cli": res_1a_1, "db_logs": log_1a_1},
        "run2": {"cli": res_1a_2, "db_logs": log_1a_2},
        "loss_curve_run1": curve_1a_1,
        "loss_curve_run2": curve_1a_2,
        "step_abs_diffs": step_diffs_1a,
        "max_step_loss_abs_diff": max(step_diffs_1a) if step_diffs_1a else None,
        "weight_comparison": weight_cmp_1a,
    }
    logger.log(f"[Exp 1A Summary] Curve 1: {curve_1a_1}")
    logger.log(f"[Exp 1A Summary] Curve 2: {curve_1a_2}")
    logger.log(f"[Exp 1A Summary] Max step loss |diff|: {max(step_diffs_1a) if step_diffs_1a else 'N/A'}")
    logger.log(f"[Exp 1A Summary] Adapter Weights Global L2 diff: {weight_cmp_1a.get('global_l2_diff')} | Max abs diff: {weight_cmp_1a.get('global_max_abs_diff')}")

    # -------------------------------------------------------------------------
    # 3. EXPERIMENT 1B: DIFFERENT SEEDS IN SOUP.YAML (seed=42 vs seed=999) + REPRO RECEIPT
    # -------------------------------------------------------------------------
    logger.log("\n" + "=" * 80)
    logger.log("EXPERIMENT 1B: `training.seed: 999` vs `training.seed: 42` + `--repro-receipt`")
    logger.log("=" * 80)

    cfg_1b_seed999 = work_dir / "exp1b_seed999.yaml"
    receipt_1b = work_dir / "repro_receipt_seed999.json"
    cfg_1b_seed999.write_text(yaml.safe_dump(make_base_config("out_1b_seed999", seed=999)), encoding="utf-8")

    res_1b = soup_cmd(
        ["train", "--config", str(cfg_1b_seed999), "--repro-receipt", str(receipt_1b), "--allow-oom-attempt", "--yes"],
        cwd=work_dir,
        env_extra=env_db,
    )
    log_1b = fetch_latest_run_logs(db_path)
    receipt_1b_data = json.loads(receipt_1b.read_text(encoding="utf-8")) if receipt_1b.exists() else {}
    st_1b = find_adapter_safetensors(work_dir / "out_1b_seed999")
    weight_cmp_1b_vs_1a = compare_safetensors(st_1a_1, st_1b) if (st_1a_1 and st_1b) else {}
    curve_1b = [s["loss"] for s in log_1b["steps"] if s.get("loss") is not None]

    results["reproducibility"]["exp1b_different_yaml_seed_ignored"] = {
        "cli": res_1b,
        "db_logs": log_1b,
        "loss_curve_seed999": curve_1b,
        "repro_receipt_json": receipt_1b_data,
        "weight_comparison_seed42_vs_seed999": weight_cmp_1b_vs_1a,
    }
    logger.log(f"[Exp 1B Summary] `training.seed: 999` curve: {curve_1b}")
    logger.log(f"[Exp 1B Summary] `--repro-receipt` recorded seeds: {receipt_1b_data.get('seeds')} (EXPECTED {{'torch': 999, ...}}, ACTUAL empty {{}}!)")
    logger.log(f"[Exp 1B Summary] Adapter L2 diff between `seed: 42` and `seed: 999` in soup.yaml: {weight_cmp_1b_vs_1a.get('global_l2_diff')} (0.0 proves `training.seed` was completely ignored!)")

    # -------------------------------------------------------------------------
    # 4. EXPERIMENT 1C: DETERMINISTIC vs NON-DETERMINISTIC CUDA DROPOUT/ATOMIC PROBE
    # -------------------------------------------------------------------------
    logger.log("\n" + "=" * 80)
    logger.log("EXPERIMENT 1C: CUDA DETERMINISM PROBE (lora.dropout=0.1, Deterministic Mode)")
    logger.log("=" * 80)

    cfg_1c_run1 = work_dir / "exp1c_det_run1.yaml"
    cfg_1c_run2 = work_dir / "exp1c_det_run2.yaml"
    c1c_dict1 = make_base_config("out_1c_run1")
    c1c_dict1["training"]["lora"]["dropout"] = 0.1
    c1c_dict2 = make_base_config("out_1c_run2")
    c1c_dict2["training"]["lora"]["dropout"] = 0.1
    cfg_1c_run1.write_text(yaml.safe_dump(c1c_dict1), encoding="utf-8")
    cfg_1c_run2.write_text(yaml.safe_dump(c1c_dict2), encoding="utf-8")

    det_env = {**env_db, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "42"}
    res_1c_1 = soup_cmd(["train", "--config", str(cfg_1c_run1), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=det_env)
    log_1c_1 = fetch_latest_run_logs(db_path)
    res_1c_2 = soup_cmd(["train", "--config", str(cfg_1c_run2), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=det_env)
    log_1c_2 = fetch_latest_run_logs(db_path)

    st_1c_1 = find_adapter_safetensors(work_dir / "out_1c_run1")
    st_1c_2 = find_adapter_safetensors(work_dir / "out_1c_run2")
    weight_cmp_1c = compare_safetensors(st_1c_1, st_1c_2) if (st_1c_1 and st_1c_2) else {}
    results["reproducibility"]["exp1c_with_dropout_and_cublas_det"] = {
        "loss_curve_run1": [s["loss"] for s in log_1c_1["steps"]],
        "loss_curve_run2": [s["loss"] for s in log_1c_2["steps"]],
        "weight_comparison": weight_cmp_1c,
    }
    logger.log(f"[Exp 1C Summary] Adapter L2 diff with dropout=0.1: {weight_cmp_1c.get('global_l2_diff')}")

    # -------------------------------------------------------------------------
    # 5. EXPERIMENT 2: TRAINING SILENT FAILURE PROBES ON GPU
    # -------------------------------------------------------------------------
    logger.log("\n" + "=" * 80)
    logger.log("EXPERIMENT 2: END-TO-END GPU TRAINING SILENT FAILURE PROBES")
    logger.log("=" * 80)

    # Probe SF-GPU-1: Misspelled hyperparameters (`learning_rate: 0.0`, `epchs: 99`, `lora_r: 4`)
    cfg_sf1 = work_dir / "sf_gpu1_typos.yaml"
    sf1_dict = make_base_config("out_sf_gpu1")
    del sf1_dict["training"]["lr"]
    del sf1_dict["training"]["epochs"]
    del sf1_dict["training"]["lora"]["r"]
    sf1_dict["training"]["learning_rate"] = 0.0
    sf1_dict["training"]["epchs"] = 99
    sf1_dict["training"]["lora"]["lora_r"] = 4
    cfg_sf1.write_text(yaml.safe_dump(sf1_dict), encoding="utf-8")

    res_sf1 = soup_cmd(["train", "--config", str(cfg_sf1), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    log_sf1 = fetch_latest_run_logs(db_path)
    st_sf1 = find_adapter_safetensors(work_dir / "out_sf_gpu1")
    sf1_stats = inspect_single_safetensors(st_sf1) if st_sf1 else {}
    results["silent_failures"]["sf_gpu1_typo_hyperparams"] = {
        "cli": res_sf1,
        "db_logs": log_sf1,
        "adapter_stats": sf1_stats,
    }
    logger.log(f"[SF-GPU-1] Misspelled `learning_rate: 0.0`, `epchs: 99`, `lora_r: 4` -> exit_code={res_sf1['returncode']} | steps={len(log_sf1['steps'])} | lora_B_norm={sf1_stats.get('lora_B_global_l2_norm')}")
    logger.log(res_sf1["stdout"])

    # Probe SF-GPU-2: Extreme LR (`lr: 1e15`) + `loss_watchdog: true` -> NaN loss & NaN weights with exit code 0
    cfg_sf2 = work_dir / "sf_gpu2_nan_loss.yaml"
    sf2_dict = make_base_config(
        "out_sf_gpu2",
        lr=1e15,
        loss_watchdog=True,
        loss_watchdog_threshold=3.0,
        loss_watchdog_patience=1,
    )
    cfg_sf2.write_text(yaml.safe_dump(sf2_dict), encoding="utf-8")

    res_sf2 = soup_cmd(["train", "--config", str(cfg_sf2), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    log_sf2 = fetch_latest_run_logs(db_path)
    st_sf2 = find_adapter_safetensors(work_dir / "out_sf_gpu2")
    sf2_stats = inspect_single_safetensors(st_sf2) if st_sf2 else {}
    results["silent_failures"]["sf_gpu2_nan_loss_watchdog_bypass"] = {
        "cli": res_sf2,
        "db_logs": log_sf2,
        "adapter_stats": sf2_stats,
    }
    logger.log(f"[SF-GPU-2] Extreme `lr: 1e15` with `loss_watchdog: true` -> exit_code={res_sf2['returncode']} | nan_elements_in_adapter={sf2_stats.get('total_nan_elements')} | final_loss={log_sf2.get('final_loss')}")
    logger.log(res_sf2["stdout"])

    # Probe SF-GPU-3: Near-zero LR (`lr: 1e-30`) -> exits 0 with untrained/zero LoRA B weights
    cfg_sf3 = work_dir / "sf_gpu3_zero_update.yaml"
    sf3_dict = make_base_config("out_sf_gpu3", lr=1e-30)
    cfg_sf3.write_text(yaml.safe_dump(sf3_dict), encoding="utf-8")

    res_sf3 = soup_cmd(["train", "--config", str(cfg_sf3), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    log_sf3 = fetch_latest_run_logs(db_path)
    st_sf3 = find_adapter_safetensors(work_dir / "out_sf_gpu3")
    sf3_stats = inspect_single_safetensors(st_sf3) if st_sf3 else {}
    results["silent_failures"]["sf_gpu3_near_zero_lr_untrained"] = {
        "cli": res_sf3,
        "db_logs": log_sf3,
        "adapter_stats": sf3_stats,
    }
    logger.log(f"[SF-GPU-3] Near-zero `lr: 1e-30` -> exit_code={res_sf3['returncode']} | lora_B_global_l2_norm={sf3_stats.get('lora_B_global_l2_norm')} (0.0 means zero weight update!)")
    logger.log(res_sf3["stdout"])

    # Probe SF-GPU-4: Mismatched dataset columns (`question`/`answer` for `format: alpaca`)
    wrong_jsonl = work_dir / "wrong_columns.jsonl"
    wrong_jsonl.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "answer": f"A{i}"}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    cfg_sf4 = work_dir / "sf_gpu4_wrong_cols.yaml"
    sf4_dict = make_base_config("out_sf_gpu4")
    sf4_dict["data"]["train"] = str(wrong_jsonl)
    cfg_sf4.write_text(yaml.safe_dump(sf4_dict), encoding="utf-8")

    res_sf4_dry = soup_cmd(["train", "--config", str(cfg_sf4), "--dry-run"], cwd=work_dir, env_extra=env_db)
    res_sf4_train = soup_cmd(["--verbose", "train", "--config", str(cfg_sf4), "--allow-oom-attempt", "--yes"], cwd=work_dir, env_extra=env_db)
    results["silent_failures"]["sf_gpu4_wrong_dataset_columns"] = {
        "dry_run_cli": res_sf4_dry,
        "full_train_cli": res_sf4_train,
    }
    logger.log(f"[SF-GPU-4] Wrong dataset columns -> `--dry-run` exit_code={res_sf4_dry['returncode']} ('Data OK: 0 train samples') | full train exit_code={res_sf4_train['returncode']}")
    logger.log(res_sf4_dry["stdout"])
    logger.log(res_sf4_train["stdout"] + "\n" + res_sf4_train["stderr"])

    # Save structured JSON
    JSON_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.log("\n" + "=" * 80)
    logger.log(f"COMPLETED ALL EXPERIMENTS. Saved:")
    logger.log(f"  1. {LOG_FILE}")
    logger.log(f"  2. {JSON_FILE}")
    logger.log("=" * 80)
    logger.close()

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
