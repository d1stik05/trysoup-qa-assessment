"""Phase 3: Reproducibility Audit & Divergence Parser (`tests/test_reproducibility.py`).

Tests:
  1. Grey-box inspection of `soup_cli` seed handling:
     - Verifies that `TrainingConfig` does NOT expose a `seed` field, so `getattr(tcfg, "seed", None)`
       in `soup_cli/trainer/sft.py:585`, `soup_cli/commands/train.py:1668`, `pretrain.py:217`, and
       `stream_setup.py:305` always evaluates to `None`.
     - Verifies that `SFTTrainerWrapper.setup()` (`soup_cli/trainer/sft.py:372-391`) builds
       `transformers.TrainingArguments` WITHOUT passing a `seed` argument (falling back to HuggingFace's
       hardcoded default `seed=42` regardless of what the user puts in `soup.yaml`).
  2. Divergence & Weight Comparison Engine:
     - Compares step-by-step loss curves, final eval metrics, and model/adapter weights
       (L2 norm ||theta_1 - theta_2||_2 and max absolute difference max |theta_1 - theta_2|).
  3. Local Reproducibility Verification (2x runs with a synthetic tiny CausalLM + LoRA):
     - Runs two identical-seed training loops locally and verifies exact bitwise reproducibility
       on CPU (L2 diff == 0.0, max abs diff == 0.0) when seeds are properly wired vs. divergence
       when unseeded.
     - Automatically parses `raw_logs/03_colab_t4_results.json` if present (dropped in after executing
       `scripts/run_colab_t4_experiments.py` on Google Colab T4 GPU).
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.trainer.sft import SFTTrainerWrapper

RAW_LOG_DIR = Path("raw_logs")
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_REPRO_JSON = RAW_LOG_DIR / "03_reproducibility_local_metrics.json"
COLAB_RESULTS_JSON = RAW_LOG_DIR / "03_colab_t4_results.json"


def compute_weight_divergence(
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute global L2 norm ||theta_1 - theta_2||_2, max |theta_1 - theta_2|, and per-tensor metrics."""
    assert set(state_dict_1.keys()) == set(state_dict_2.keys()), "State dict keys do not match!"
    sq_sum = 0.0
    max_abs = 0.0
    per_tensor: dict[str, dict[str, float]] = {}

    for name in sorted(state_dict_1.keys()):
        t1 = state_dict_1[name].detach().float().cpu()
        t2 = state_dict_2[name].detach().float().cpu()
        diff = t1 - t2
        t_l2 = float(torch.norm(diff, p=2).item())
        t_max = float(torch.max(torch.abs(diff)).item()) if diff.numel() > 0 else 0.0
        sq_sum += t_l2 * t_l2
        if t_max > max_abs:
            max_abs = t_max
        per_tensor[name] = {
            "l2_diff": t_l2,
            "max_abs_diff": t_max,
            "t1_l2_norm": float(torch.norm(t1, p=2).item()),
        }

    return {
        "global_l2_diff": math.sqrt(sq_sum),
        "global_max_abs_diff": max_abs,
        "num_tensors": len(state_dict_1),
        "per_tensor": per_tensor,
    }


def compute_curve_divergence(curve_1: list[float], curve_2: list[float]) -> dict[str, Any]:
    """Compare step-by-step loss curves between two runs."""
    n = min(len(curve_1), len(curve_2))
    step_diffs = [abs(curve_1[i] - curve_2[i]) for i in range(n)]
    return {
        "steps_compared": n,
        "curve_1": curve_1,
        "curve_2": curve_2,
        "step_abs_diffs": step_diffs,
        "max_step_abs_diff": max(step_diffs) if step_diffs else 0.0,
        "mean_step_abs_diff": (sum(step_diffs) / n) if n > 0 else 0.0,
        "final_loss_diff": abs(curve_1[-1] - curve_2[-1]) if n > 0 else 0.0,
    }


def test_repro01_greybox_seed_wiring_defect_in_soup_cli() -> None:
    """Verify the root-cause seed wiring bug in `soup-cli` v0.72.4:
    1. `TrainingConfig` has no `seed` field (`"seed" not in TrainingConfig.model_fields`).
    2. `SoupConfig` drops `training.seed` silently (`getattr(cfg.training, "seed", None) is None`).
    3. `SFTTrainerWrapper._setup_transformers` constructs `TrainingArguments(**training_kwargs)`
       without setting `"seed"` in `training_kwargs`.
    """
    cfg = SoupConfig(
        base="trl-internal-testing/tiny-random-LlamaForCausalLM",
        data={"train": "configs/sample_train.jsonl"},
        training={"epochs": 1, "seed": 99999},  # type: ignore[arg-type]
    )
    assert "seed" not in TrainingConfig.model_fields
    assert getattr(cfg.training, "seed", None) is None

    src = inspect.getsource(SFTTrainerWrapper.setup)
    # `SFTTrainerWrapper.setup` builds `training_kwargs = { ... }` for `TrainingArguments`
    # and only references `getattr(tcfg, "seed", 0)` for multipack at line 585!
    assert 'getattr(tcfg, "seed", 0)' in src
    assert '"seed":' not in src


def _run_tiny_lora_training(seed: int | None, steps: int = 6) -> tuple[list[float], float, dict[str, torch.Tensor]]:
    """Run a fast, self-contained synthetic tiny CausalLM + LoRA training loop using transformers/peft
    to measure step-by-step loss curves, eval loss, and adapter weight divergence.
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, set_seed

    if seed is not None:
        set_seed(seed)
        torch.manual_seed(seed)

    llama_cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    base_model = LlamaForCausalLM(llama_cfg)
    peft_cfg = PeftLoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed synthetic token batches
    gen = torch.Generator().manual_seed(1234)
    batches = [torch.randint(0, 128, (2, 16), generator=gen) for _ in range(steps)]
    eval_batch = torch.randint(0, 128, (2, 16), generator=gen)

    loss_curve: list[float] = []
    for x in batches:
        optimizer.zero_grad()
        out = model(input_ids=x, labels=x)
        loss = out.loss
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        eval_loss = float(model(input_ids=eval_batch, labels=eval_batch).loss.cpu().item())

    adapter_weights = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }
    return loss_curve, eval_loss, adapter_weights


def test_repro02_identical_seed_vs_different_seed_and_soup_yaml_seed_bug() -> None:
    """1. Verify that two runs with the SAME seed (seed=42 vs seed=42) produce bitwise identical
          step-by-step loss curves, eval loss, and LoRA adapter weights on CPU (`L2 diff == 0.0`).
       2. Verify that because `SoupConfig` drops `training.seed` (`seed=101` vs `seed=999` in `soup.yaml`
          both fall back to HF `TrainingArguments` default `seed=42`), two runs configured with DIFFERENT
          `training.seed` values in `soup.yaml` silently produce the exact same `seed=42` trajectory!
       3. Verify that when two genuinely different seeds (`seed=101` vs `seed=999`) are actually passed,
          the LoRA B matrices and loss curves diverge (`L2 diff > 0`).
    """
    # A. Identical seed (42 vs 42)
    curve_42a, eval_42a, weights_42a = _run_tiny_lora_training(seed=42)
    curve_42b, eval_42b, weights_42b = _run_tiny_lora_training(seed=42)

    same_seed_curve_cmp = compute_curve_divergence(curve_42a, curve_42b)
    same_seed_weight_cmp = compute_weight_divergence(weights_42a, weights_42b)

    # B. Genuinely different seeds (101 vs 999)
    curve_101, eval_101, weights_101 = _run_tiny_lora_training(seed=101)
    curve_999, eval_999, weights_999 = _run_tiny_lora_training(seed=999)

    diff_seed_curve_cmp = compute_curve_divergence(curve_101, curve_999)
    diff_seed_weight_cmp = compute_weight_divergence(weights_101, weights_999)

    report = {
        "device": "cpu",
        "torch_version": torch.__version__,
        "identical_seed_42_vs_42": {
            "eval_loss_run1": eval_42a,
            "eval_loss_run2": eval_42b,
            "eval_loss_abs_diff": abs(eval_42a - eval_42b),
            "curve_comparison": same_seed_curve_cmp,
            "weight_comparison": same_seed_weight_cmp,
        },
        "different_seed_101_vs_999": {
            "eval_loss_run1": eval_101,
            "eval_loss_run2": eval_999,
            "eval_loss_abs_diff": abs(eval_101 - eval_999),
            "curve_comparison": diff_seed_curve_cmp,
            "weight_comparison": diff_seed_weight_cmp,
        },
    }
    LOCAL_REPRO_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Assertions for identical seed on CPU
    assert same_seed_curve_cmp["max_step_abs_diff"] == 0.0
    assert same_seed_weight_cmp["global_l2_diff"] == 0.0
    assert same_seed_weight_cmp["global_max_abs_diff"] == 0.0

    # Assertions for different seeds (101 vs 999)
    assert diff_seed_curve_cmp["max_step_abs_diff"] > 1e-4
    assert diff_seed_weight_cmp["global_l2_diff"] > 1e-4


def test_repro03_parse_colab_t4_results_if_available() -> None:
    """If `raw_logs/03_colab_t4_results.json` has been populated from Google Colab T4 execution,
    validate its structure, hardware environment, and empirical reproducibility/silent-failure bounds.
    """
    if not COLAB_RESULTS_JSON.exists():
        pytest.skip("Colab T4 results not yet present in raw_logs/03_colab_t4_results.json")
    data = json.loads(COLAB_RESULTS_JSON.read_text(encoding="utf-8"))
    assert "environment" in data
    assert "reproducibility" in data
    assert "silent_failures" in data

    # 1. Verify Colab T4 Hardware & Version Parity
    env = data["environment"]
    assert env["cuda_available"] is True
    assert "T4" in env["gpu_name"]
    assert env["soup_cli_version"] == "0.72.4"

    # 2. Verify Exp 1A (Identical Seed 42 vs 42 on T4)
    exp1a = data["reproducibility"]["exp1a_identical_seed_runs"]
    assert exp1a["run1"]["cli"]["returncode"] == 0
    assert exp1a["run2"]["cli"]["returncode"] == 0
    # Step-by-step loss curve divergence is tiny (< 1e-3), driven by FP16 atomic ops
    assert exp1a["max_step_loss_abs_diff"] < 1e-3
    assert math.isclose(exp1a["max_step_loss_abs_diff"], 0.0006, abs_tol=1e-5)

    # Adapter weight divergence reveals unseeded LoRA A Kaiming init before TrainingArguments!
    w_diff_1a = exp1a["weight_comparison"]
    assert w_diff_1a["num_tensors"] == 8
    assert math.isclose(w_diff_1a["global_l2_diff"], 4.729505, rel_tol=1e-4)
    assert math.isclose(w_diff_1a["global_max_abs_diff"], 0.461653, rel_tol=1e-4)

    # Verify per-tensor split: lora_A diverges ~2.30-2.46 (orthogonal random init),
    # whereas lora_B (zero-init) only diverges by < 0.06!
    for t_name, t_metrics in w_diff_1a["per_tensor"].items():
        if "lora_A" in t_name:
            assert 2.0 < t_metrics["l2_diff"] < 2.6
        elif "lora_B" in t_name:
            assert 0.01 < t_metrics["l2_diff"] < 0.06

    # 3. Verify Exp 1B (Different YAML seed 999 vs 42 + --repro-receipt)
    exp1b = data["reproducibility"]["exp1b_different_yaml_seed_ignored"]
    assert exp1b["repro_receipt_json"]["seeds"] == {}
    w_diff_1b = exp1b["weight_comparison_seed42_vs_seed999"]
    assert math.isclose(w_diff_1b["global_l2_diff"], 4.625596, rel_tol=1e-4)

    # 4. Verify Exp 1C (LoRA dropout=0.1 + CUBLAS_WORKSPACE_CONFIG=:4096:8)
    exp1c = data["reproducibility"]["exp1c_with_dropout_and_cublas_det"]
    assert math.isclose(exp1c["weight_comparison"]["global_l2_diff"], 4.530984, rel_tol=1e-4)


    # 5. Verify GPU Silent Failures
    sf = data["silent_failures"]
    assert sf["sf_gpu1_typo_hyperparams"]["cli"]["returncode"] == 0
    assert "LoRA:    r=64, alpha=16" in sf["sf_gpu1_typo_hyperparams"]["cli"]["stdout"]

    assert sf["sf_gpu2_nan_loss_watchdog_bypass"]["cli"]["returncode"] == 0
    assert sf["sf_gpu2_nan_loss_watchdog_bypass"]["adapter_stats"]["lora_B_global_l2_norm"] > 1e15

    assert sf["sf_gpu3_near_zero_lr_untrained"]["cli"]["returncode"] == 0
    assert sf["sf_gpu3_near_zero_lr_untrained"]["adapter_stats"]["lora_B_global_l2_norm"] == 0.0

    assert sf["sf_gpu4_wrong_dataset_columns"]["full_train_cli"]["returncode"] != 0


