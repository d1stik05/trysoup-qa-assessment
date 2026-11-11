# 03. Seed Reproducibility & GPU Non-Determinism Analysis

## 1. Executive Summary & Reproducibility Verdict

**Verdict:** **`FAIL` (Broken Seed Wiring + Unseeded LoRA Initialization in `soup-cli v0.72.4`)**

An audit of `soup-cli v0.72.4` across both local CPU (`PyTorch 2.15.0+cpu`, `Python 3.14.2`) and Google Colab **NVIDIA Tesla T4 GPU** (`PyTorch 2.14.1+cu130`, `CUDA 13.0`, `cuDNN 92800`, Compute Capability `7.5`) uncovered **two compounding reproducibility defects** in the training pipeline:

1. **Unseeded LoRA Weight Initialization (`soup_cli/trainer/sft.py:219, 471, 949`):** During `SFTTrainerWrapper.setup()`, `self._setup_transformers()` calls `get_peft_model(self.model, lora_config)` at line 949 **before** `transformers.TrainingArguments` is instantiated at line 471. Because HuggingFace's `TrainingArguments.__post_init__` is what calls `set_seed(self.seed)` (default `42`), PEFT initializes every `lora_A.weight` matrix via Kaiming uniform sampling using an **unseeded OS-entropy PyTorch RNG**. Consequently, two identical runs (`seed: 42` vs `seed: 42`) initialize orthogonal random `lora_A` matrices, yielding a massive **Global Adapter $L_2$ Weight Difference of $\|\theta_1 - \theta_2\|_2 = 4.729505$** ($\max |\theta_1 - \theta_2| = 0.461653$), even though their step-by-step loss curves match to within $\max_t |L_1(t) - L_2(t)| = 0.0006$ (`0.0058%`).
2. **Silently Ignored `training.seed` & Empty Compliance Receipt (`soup_cli/config/schema.py:2875`, `soup_cli/commands/train.py:1668`):** `TrainingConfig` omits a `seed` field entirely and lacks `extra="forbid"`. Any `training.seed` specified in `soup.yaml` is silently discarded during Pydantic parsing, `getattr(cfg.training, "seed", None)` always evaluates to `None`, and the SR 11-7 compliance flag `--repro-receipt` writes `"seeds": {}`.

---

## 2. Experimental Setup & Hardware Telemetry

All raw logs and machine-readable JSON outputs are preserved untouched in:
- [`../raw_logs/03_colab_t4_experiments.log`](../raw_logs/03_colab_t4_experiments.log) (666 lines, unfiltered CLI output from Google Colab T4)
- [`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json) (1,463 lines, step-by-step metrics and per-tensor $L_2$ / $L_\infty$ diffs)
- [`../raw_logs/03_reproducibility_local.log`](../raw_logs/03_reproducibility_local.log) & [`../raw_logs/03_reproducibility_local_metrics.json`](../raw_logs/03_reproducibility_local_metrics.json)

| Parameter | Google Colab T4 GPU Environment | Local Baseline Environment |
| :--- | :--- | :--- |
| **OS / Platform** | `Linux-6.6.113+-x86_64-with-glibc2.35` | `Windows-11-10.0.26200-SP0` |
| **Python / `soup-cli`** | `Python 3.13.16` / `soup-cli 0.72.4` | `Python 3.14.2` / `soup-cli 0.72.4` |
| **PyTorch / CUDA / cuDNN** | `2.14.1+cu130` / `CUDA 13.0` / `cuDNN 92800` | `2.15.0+cpu` / `N/A` |
| **Accelerator** | `NVIDIA Tesla T4` (15,360 MiB, `cc=7.5`, Driver `580.82.07`) | CPU (`x86_64`) |
| **Base Model & Task** | `trl-internal-testing/tiny-random-LlamaForCausalLM` (`sft`) | `LlamaForCausalLM` (`vocab=128, hidden=32, layers=2`) |
| **LoRA Config** | `r=8, alpha=16, target_modules=[q_proj, v_proj]` (4,096 params) | `r=4, alpha=8, target_modules=[q_proj, v_proj]` |
| **Training Schedule** | `epochs=3, batch_size=2, lr=1e-3, logging_steps=1` (15 steps) | `steps=6, batch_size=2, lr=1e-3` |
| **Required T4 Workarounds** | `auto_mixed_precision: true` (FP16), `chat_template: chatml`, `--allow-oom-attempt` | Direct PyTorch/PEFT harness |

---

## 3. Step-by-Step Loss Curve Comparison (Tesla T4 GPU)

We executed four 15-step SFT training runs on Tesla T4 ([`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json)):
- **Exp 1A Run 1 (`seed=42`):** `run_20261111_111908_0366aef8` (`lora.dropout: 0.0`)
- **Exp 1A Run 2 (`seed=42`):** `run_20261111_111931_5eed7d3c` (`lora.dropout: 0.0`)
- **Exp 1B (`seed=999`):** `run_20261111_111954_6607c7fe` (`training.seed: 999` in `soup.yaml`, `lora.dropout: 0.0`)
- **Exp 1C (`dropout=0.1` + cuBLAS det):** `run_20261111_112017_024e91f1` (`run1`) vs `run_20261111_112040_5857f7f0` (`run2`) with `CUBLAS_WORKSPACE_CONFIG=:4096:8`

| Step | Epoch | Exp 1A Run 1 (`seed=42`) | Exp 1A Run 2 (`seed=42`) | $\|\Delta L_{\text{1A}}\|$ (`42` vs `42`) | Exp 1B (`seed=999`) | $\|\Delta L_{\text{1B}}\|$ (`42` vs `999`) | Grad Norm (Run 1 / Run 2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.2 | `10.3557` | `10.3557` | **`0.0000`** | `10.3557` | `0.0000` | `0.2683` / `0.2618` |
| **2** | 0.4 | `10.3712` | `10.3712` | **`0.0000`** | `10.3713` | `0.0001` | `0.2738` / `0.2660` |
| **3** | 0.6 | `10.3694` | `10.3693` | `0.0001` | `10.3694` | `0.0000` | `0.2757` / `0.2776` |
| **4** | 0.8 | `10.3547` | `10.3546` | `0.0001` | `10.3549` | `0.0002` | `0.3054` / `0.3075` |
| **5** | 1.0 | `10.3646` | `10.3646` | **`0.0000`** | `10.3647` | `0.0001` | `0.2611` / `0.2663` |
| **6** | 1.2 | `10.3662` | `10.3662` | **`0.0000`** | `10.3663` | `0.0001` | `0.2673` / `0.2747` |
| **7** | 1.4 | `10.3574` | `10.3574` | **`0.0000`** | `10.3578` | `0.0004` | `0.3084` / `0.3095` |
| **8** | 1.6 | `10.3596` | `10.3601` | `0.0005` | `10.3601` | `0.0005` | `0.2747` / `0.2677` |
| **9** | 1.8 | `10.3679` | `10.3676` | `0.0003` | `10.3680` | `0.0001` | `0.2646` / `0.2734` |
| **10** | 2.0 | `10.3594` | `10.3595` | `0.0001` | `10.3599` | `0.0005` | `0.2896` / `0.2921` |
| **11** | 2.2 | `10.3587` | `10.3587` | **`0.0000`** | `10.3592` | `0.0005` | `0.2848` / `0.2816` |
| **12** | 2.4 | `10.3538` | `10.3543` | `0.0005` | `10.3547` | `0.0009` | `0.3183` / `0.3145` |
| **13** | 2.6 | `10.3610` | `10.3616` | **`0.0006`** | `10.3616` | `0.0006` | `0.2757` / `0.2683` |
| **14** | 2.8 | `10.3586` | `10.3591` | `0.0005` | `10.3594` | `0.0008` | `0.2842` / `0.2877` |
| **15** | 3.0 | `10.3570` | `10.3571` | `0.0001` | `10.3575` | `0.0005` | `0.2920` / `0.2925` |
| **Mean / Summary** | **3.0** | `10.361012` | `10.361198` | **Max: `0.0006`** | `10.361372` | **Max: `0.0009`** | — |

### Why Step 1 Loss Is Identical (`10.3557`) While Step 1 `grad_norm` Diverges (`0.2683` vs `0.2618`)
At Step 1, LoRA initializes $B_0 = \mathbf{0}$ and $A_0 \sim \mathcal{U}(-\sqrt{3/d_{\text{in}}}, \sqrt{3/d_{\text{in}}})$. Because the LoRA forward contribution at Step 1 is $\Delta W_0 x = B_0 A_0 x = \mathbf{0}$, the Step 1 forward pass depends **only** on the frozen base model weights and the Step 1 mini-batch order (which *is* seeded by `TrainingArguments(seed=42)`). Thus, Step 1 loss is identical (`10.3557`) across all runs!
However, during the Step 1 backward pass, the gradient with respect to $B$ is:
$$\frac{\partial \mathcal{L}}{\partial B_0} = \left(\frac{\partial \mathcal{L}}{\partial y}\right) (A_0 x)^\top$$
Because $A_0$ was initialized **before** `TrainingArguments` set the random seed, $A_0^{(\text{run1})} \neq A_0^{(\text{run2})}$, causing $\nabla_{B_0} \mathcal{L}$ (and therefore Step 1 `grad_norm`: `0.2683` vs `0.2618`) to diverge immediately on Step 1!

---

## 4. Per-Tensor Adapter Weight Divergence Analysis (`adapter_model.safetensors`)

Inspecting the 8 saved LoRA tensors in `adapter_model.safetensors` ([`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json)) exposes the exact structural split between `lora_A` (`shape [8, 16]`) and `lora_B` (`shape [16, 8]`):

| Adapter Tensor Name | Shape | Run 1 $\|\theta_1\|_2$ | Run 2 $\|\theta_2\|_2$ | Exp 1A (`42` vs `42`) $\|\theta_1 - \theta_2\|_2$ | Exp 1A (`42` vs `42`) $\max |\theta_1 - \theta_2|$ | Exp 1B (`42` vs `999`) $\|\theta_1 - \theta_2\|_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `layers.0.self_attn.q_proj.lora_A.weight` | `[8, 16]` | `1.595567` | `1.668018` | **`2.299807`** | `0.453445` | `2.369325` |
| `layers.0.self_attn.q_proj.lora_B.weight` | `[16, 8]` | `0.023532` | `0.020952` | **`0.028923`** | `0.006378` | `0.029873` |
| `layers.0.self_attn.v_proj.lora_A.weight` | `[8, 16]` | `1.696827` | `1.592306` | **`2.463119`** | `0.456767` | `2.288940` |
| `layers.0.self_attn.v_proj.lora_B.weight` | `[16, 8]` | `0.030945` | `0.028632` | **`0.043038`** | `0.007419` | `0.039897` |
| `layers.1.self_attn.q_proj.lora_A.weight` | `[8, 16]` | `1.678950` | `1.519465` | **`2.313760`** | `0.461653` | `2.226681` |
| `layers.1.self_attn.q_proj.lora_B.weight` | `[16, 8]` | `0.015660` | `0.014933` | **`0.021614`** | `0.005801` | `0.020937` |
| `layers.1.self_attn.v_proj.lora_A.weight` | `[8, 16]` | `1.679803` | `1.665963` | **`2.378283`** | `0.449438` | `2.363506` |
| `layers.1.self_attn.v_proj.lora_B.weight` | `[16, 8]` | `0.023121` | `0.025216` | **`0.032750`** | `0.006560` | `0.033297` |
| **Global Across All 8 Tensors** | **1,024** | — | — | **`4.729505`** | **`0.461653`** | **`4.625596`** |

### Mathematical Proof of Orthogonal Unseeded Initialization
For two independent zero-mean random matrices $A_1, A_2 \in \mathbb{R}^{r \times d_{\text{in}}}$ with $\mathbb{E}[\langle A_1, A_2 \rangle_F] = 0$, the expected Frobenius ($L_2$) distance is:
$$\mathbb{E}\left[\|A_1 - A_2\|_F\right] \approx \sqrt{\|A_1\|_F^2 + \|A_2\|_F^2}$$
Plugging in the empirical norms from Layer 0 `q_proj.lora_A` ($\|A_1\|_2 = 1.5956$, $\|A_2\|_2 = 1.6680$):
$$\sqrt{1.5956^2 + 1.6680^2} = \sqrt{2.5459 + 2.7822} = \mathbf{2.3083} \quad (\text{Observed: } \mathbf{2.2998})$$
Summing across all four `lora_A` tensors ($\sqrt{2.2998^2 + 2.4631^2 + 2.3138^2 + 2.3783^2} = \mathbf{4.7291}$) accounts for **99.99%** of the total global $L_2$ divergence (`4.729505`).

Furthermore, comparing **Exp 1A** (`seed=42` vs `seed=42`, global $L_2 = 4.7295$) against **Exp 1B** (`seed=42` vs `seed=999`, global $L_2 = 4.6256$) and **Exp 1C** (`dropout=0.1` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`, global $L_2 = 4.5310$) proves that changing `training.seed` in `soup.yaml` or setting `CUBLAS_WORKSPACE_CONFIG` has zero effect on adapter weight determinism until `soup-cli` seeds PyTorch **prior** to calling `get_peft_model()`.

### Local CPU Controlled Baseline ([`../tests/test_reproducibility.py`](../tests/test_reproducibility.py))
To confirm that calling `set_seed(seed)` before `get_peft_model()` restores exact determinism, [`../tests/test_reproducibility.py`](../tests/test_reproducibility.py) ([`../raw_logs/03_reproducibility_local_metrics.json`](../raw_logs/03_reproducibility_local_metrics.json)) runs two seeded 6-step loops (`seed=42` vs `seed=42`) and two distinct-seed loops (`seed=101` vs `seed=999`):
- **Properly Seeded (`seed=42` vs `seed=42`):** `max_step_abs_diff = 0.0`, `eval_loss_abs_diff = 0.0`, `global_l2_diff = 0.0`, `global_max_abs_diff = 0.0`.
- **Distinct Seeds (`seed=101` vs `seed=999`):** `max_step_abs_diff = 0.14568`, `eval_loss_abs_diff = 0.00297`, `global_l2_diff = 2.02267`.

---

## 5. Three-Layer Root Cause Analysis

1. **Defect Layer 1 — Unseeded PEFT/Model Initialization (`soup_cli/trainer/sft.py:219, 471, 949`):**
   `SFTTrainerWrapper.setup()` executes `self._setup_transformers(cfg, tcfg)` at line 219 (which loads the model and calls `get_peft_model()` at line 949) **before** creating `TrainingArguments(**training_kwargs)` at line 471. Because `transformers.set_seed()` is only invoked inside `TrainingArguments.__post_init__()`, all weight initialization in `_setup_transformers()` is completely unseeded.
2. **Defect Layer 2 — Missing `TrainingConfig.seed` Field (`soup_cli/config/schema.py:2875`):**
   `TrainingConfig` defines 65+ fields but omits `seed: Optional[int] = 42`. Meanwhile, `soup_cli/commands/train.py:1668` checks `getattr(cfg.training, "seed", None)` when generating `--repro-receipt`, which always evaluates to `None` and writes `"seeds": {}` ([`../raw_logs/03_colab_t4_results.json`](../raw_logs/03_colab_t4_results.json)).
3. **Hardware Layer 3 — CUDA / Tesla T4 FP16 Non-Determinism:**
   Even when `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set (Exp 1C), `soup-cli` never calls `torch.use_deterministic_algorithms(True)` or sets `torch.backends.cudnn.deterministic = True`. Combined with the unseeded `lora_A` projections, FP16 Tensor Core reductions introduce a bounded trajectory drift of $\max_t |\Delta L(t)| = 6 \times 10^{-4}$.

---

## 6. Justified Reproducibility Acceptance Tolerances

Based on our theoretical derivations and empirical measurements across CPU and Tesla T4 GPU, we establish the following release-gating tolerances (once [BUG-002](../bug_reports/BUG-002.md) is fixed by adding `seed: int = 42` to `TrainingConfig` and calling `transformers.set_seed(tcfg.seed)` at the top of `SFTTrainerWrapper.setup()`):

| Execution Mode | Metric | Acceptance Tolerance | Empirical Value (`v0.72.4`) | Gate Status |
| :--- | :--- | :--- | :--- | :---: |
| **CPU Deterministic** (`seed=42` vs `42`) | Step Loss $\|\Delta L\|_\infty$ | $\le 10^{-7}$ (`0.0` bitwise) | `0.000000` (when seeded pre-init) | **PASS** (with fix) |
| **CPU Deterministic** (`seed=42` vs `42`) | Adapter Weight $\|\theta_1 - \theta_2\|_2$ | $\le 10^{-6}$ (`0.0` bitwise) | `0.000000` (when seeded pre-init) | **PASS** (with fix) |
| **T4 GPU FP16** (`seed=42` vs `42`, `dropout=0`) | Step Loss $\max_t |L_1(t) - L_2(t)|$ | $\le 1.0 \times 10^{-3}$ | `0.000600` (`6.0e-4`) | **PASS** |
| **T4 GPU FP16** (`seed=42` vs `42`, `dropout=0`) | Final Eval / Train Loss $|\Delta L_{\text{final}}|$ | $\le 5.0 \times 10^{-4}$ | `0.000186` (`1.86e-4`) | **PASS** |
| **T4 GPU FP16** (`seed=42` vs `42`, `dropout=0`) | Adapter Weight $\|\theta_1 - \theta_2\|_2$ | $\le 1.0 \times 10^{-2}$ | **`4.729505`** (`lora_A` unseeded!) | **FAIL ([BUG-002](../bug_reports/BUG-002.md))** |
| **Compliance Receipt** (`--repro-receipt`) | `receipt["seeds"]["training"]` | Must equal configured `training.seed` | **`{}` (empty dict)** | **FAIL ([BUG-002](../bug_reports/BUG-002.md))** |
