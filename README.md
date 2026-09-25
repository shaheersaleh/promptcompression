# Trajectory-Aware Reinforcement Learning for Discrete Prompt Compression

Official implementation of **Trajectory-Aware Reinforcement Learning for Discrete Prompt Compression**. 

This repository provides an upstream-downstream reinforcement learning framework that trains an **XLM-RoBERTa (560M)** policy network to perform hard discrete token pruning ($m \in \{0, 1\}^N$). The policy is supervised by dense, step-by-step Kullback-Leibler divergence ($D_{\mathrm{KL}}$) computed over the causal answer generation trajectory of a completely frozen downstream **Microsoft Phi-4 (14B, 4-bit NF4)** language model.

---

## Architecture Overview

```
Raw Prompt X ──► [ Upstream XLM-RoBERTa (560M) ] ──► Discrete Hard Mask m ∈ {0, 1}^N
                        ▲                                      │
                        │ Policy Gradient (REINFORCE)          ▼
                        │ (NCCL All-Reduce across 4 GPUs)  Pruned Prompt M = {x_i | m_i = 1}
                        │                                      │
                        └────── Dense Trajectory D_KL ─────────┴──► [ Downstream Phi-4 (14B, Frozen) ]
                                (Evaluated over t = 1 ... T)         Target Evaluation: P(y_t | y_<t, M)
```

### Key Technical Pillars
1. **Discrete Hard Binary Masking:** The upstream editor emits discrete binary actions $m_i \in \{0, 1\}$ directly via REINFORCE with a self-critical baseline ($A = R(\mathbf{m}_{\text{sampled}}) - R(\mathbf{m}_{\text{greedy}})$). Pruned prompts are evaluated as physical sub-strings, eliminating the continuous test-time distribution collapse inherent to soft Gumbel-Softmax masks.
2. **Dense Trajectory Supervision ($t = 1 \dots T$):** Replaces uninformative $t=0$ boundary supervision with multi-step causal relative entropy:
   $$D_{\mathrm{KL}}^{\mathrm{traj}} = \frac{1}{T} \sum_{t=1}^T D_{\mathrm{KL}}\Big( P(Y_t \mid X) \,\parallel\, P(Y_t \mid M) \Big)$$
3. **Single-Pass Causal Slicing:** Evaluates all $T$ answer positions simultaneously in a single forward pass through downstream Phi-4, cutting evaluation latency from $1.20\,\mathrm{s}$ down to $0.25\,\mathrm{s}$ ($4.8\times$ acceleration).
4. **Distributed 4-GPU All-Reduce:** Scales exploration using PyTorch DistributedDataParallel (DDP). Each GPU evaluates an independent sentence batch, synchronizing continuous policy gradients across GPUs via NCCL All-Reduce:
   $$\bar{\mathbf{g}} = \frac{1}{K} \sum_{k=0}^{K-1} \nabla_\theta \mathcal{L}_k(\theta), \quad \theta \leftarrow \theta - \eta \bar{\mathbf{g}}$$
   *(Binary masks are never averaged; only parameter gradients are synchronized).*

---

## Repository File Manifest

| File | Purpose | Input / Dependencies | Output Produced |
| :--- | :--- | :--- | :--- |
| `compare_all.py` | Master evaluation aggregator across all runs | The 4 `.jsonl` audit files | Formatted terminal table, `evaluation_results.md`, `evaluation_results.csv` |
| `audit_results_ddp.jsonl` | Audit metrics for Traj RL 4-GPU (300 steps) | Model checkpoint evaluation | Per-sample token retention, step-wise $D_{\mathrm{KL}}$ drift, EM/F1 |
| `audit_results_ddp_full.jsonl` | Audit metrics for Traj RL 87k (21.9k steps) | Model checkpoint evaluation | High-compression saturation metrics & drift trajectory |
| `audit_results_trajectory_rl.jsonl` | Audit metrics for Traj RL 1-GPU (300 steps) | Model checkpoint evaluation | Single-GPU baseline metrics |
| `comprehensive_audit_results.jsonl` | Audit metrics for Token-0 Baseline (300 steps) | Model checkpoint evaluation | Sparse reward baseline metrics |
| `distribution_comparison.png` | Trajectory divergence visualization | Audit logs | Generated multi-step drift curves ($t = 1 \dots 10$) |
| `prepare_full_squad.py` | Preprocesses full SQuAD train/val splits | Hugging Face datasets (`squad`) | `data/train_full.jsonl` (161 MB) |
| `data/train.jsonl` | Sample training subset (356 KB) | None (pre-packaged) | Standalone prompt-response pairs for rapid testing |
| `train_trajectory_reinforce_ddp.py` | Distributed 4-GPU REINFORCE training engine | PyTorch DDP, Phi-4, dataset | Checkpoints in `checkpoints/`, loss logs |
| `train_trajectory_reinforce.py` | Single-GPU baseline training engine | Single GPU, Phi-4, dataset | Checkpoints in `checkpoints/` |
| `train_multigpu.sh` | Cluster launcher for 4-GPU DDP training | Slurm environment / bash | Spawns 4 worker processes via `torchrun` |
| `run_comprehensive_audit.py` | Evaluates checkpoints on answer trajectory $D_{\mathrm{KL}}$ | Model checkpoint, Phi-4 | Detailed per-sample JSONL audit log |
| `run_audit_full.sh` | Batch audit runner script | Slurm cluster / bash | Automated execution of `run_comprehensive_audit.py` |
| `requirements.txt` | Frozen environment specifications | pip | Reproducible dependencies |
| `slides.tex` / `slides.pdf` | Beamer defense presentation | pdflatex | 11-slide oral presentation deck |

---

## Reproduction Guide: Step-by-Step

### 1. Instant Benchmark Reproduction (No GPU Required)
You can verify the macro benchmark tables and step-wise drift comparisons directly from the audited evaluations without running multi-hour GPU jobs:

```bash
python compare_all.py
```

* **Files Read:** `comprehensive_audit_results.jsonl`, `audit_results_trajectory_rl.jsonl`, `audit_results_ddp.jsonl`, `audit_results_ddp_full.jsonl`.
* **Outputs Generated:**
  * Outputs the complete 4-way comparison table to stdout.
  * Writes `evaluation_results.md` and `evaluation_results.csv` to disk.

---

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/shaheersaleh/promptcompression.git
cd promptcompression

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

*Requirements:* Python $\ge 3.10$, CUDA $\ge 12.0$, PyTorch $\ge 2.2.0$ compiled with NCCL backend support.

---

### 3. Dataset Preparation

* **Quick Sanity Testing:** The pre-packaged `data/train.jsonl` (356 KB) requires no download and enables instant verification of the training pipeline.
* **Full-Scale Scaling Run (87.6k samples):** To reconstruct the full 161 MB dataset used for large-scale training:
  ```bash
  python prepare_full_squad.py
  ```
  *Output:* Creates `data/train_full.jsonl`.

---

### 4. Distributed Multi-GPU Training (4x GPUs)

To train the upstream XLM-RoBERTa policy with trajectory-aware REINFORCE across 4 GPUs:

```bash
# Option A: Via Slurm cluster script
bash train_multigpu.sh

# Option B: Directly via PyTorch torchrun
torchrun --nproc_per_node=4 train_trajectory_reinforce_ddp.py \
    --data_path data/train.jsonl \
    --batch_size 4 \
    --lr 1e-5 \
    --beta 0.5 \
    --epochs 3 \
    --output_dir checkpoints/
```

* **Hardware Target:** 4x GPUs with $\ge 24\,\text{GB}$ VRAM (NVIDIA A100, RTX 6000 Ada, or H100).
* **Execution Flow:** 4 worker processes concurrently sample independent prompts. Downstream Microsoft Phi-4 (14B, 4-bit NF4) executes strictly in inference mode (`torch.no_grad()`). Gradients are averaged across devices via NCCL All-Reduce before parameter updates.
* **Outputs Generated:** Checkpoints saved to `checkpoints/checkpoint_epoch_*/`.

---

### 5. Checkpoint Auditing & Evaluation

To evaluate step-by-step causal divergence against downstream Phi-4 on a trained checkpoint:

```bash
# Option A: Via batch evaluation script
bash run_audit_full.sh

# Option B: Direct Python execution
python run_comprehensive_audit.py \
    --checkpoint_path checkpoints/checkpoint_epoch_2 \
    --eval_data data/train.jsonl \
    --output_file audit_results_ddp.jsonl
```

* **Outputs Generated:** A structured JSONL audit file containing exact drop rates, downstream Exact Match (EM), token F1, and step-level divergence metrics ($D_{\mathrm{KL}}$ for steps $t = 1 \dots 10$).

---

## Macro Benchmark Results

The following table compares the baseline method against single-GPU, 4-GPU DDP, and full-scale 87k scaling configurations evaluated against downstream Microsoft Phi-4 (14B NF4):

| Metric | Token-0 Base | Traj RL (1-GPU) | Traj RL (4-GPU) | Traj RL (87k Full) |
| :--- | :---: | :---: | :---: | :---: |
| **Supervision Horizon** | Token 0 only | Full Sequence | Full Sequence | Full Sequence |
| **Training Budget** | 300 steps | 300 steps | 300 steps | 21,900 steps |
| **Mean Token Drop Rate** | **74.14%** | 20.77% | **46.94% (Pareto)** | **75.93% (Saturated)** |
| **Downstream F1** | 0.5991 | **0.7978** | **0.7542** | 0.5772 |
| **Downstream Exact Match** | 0.0667 (6.7%) | **0.4667 (46.7%)** | **0.3333 (33.3%)** | 0.0667 (6.7%) |
| **Gain over Baseline** | Baseline | **7.0$\times$ Multiplier** | **5.0$\times$ Multiplier** | 1.0$\times$ (Cliff) |
| Step $t=1$ Drift (Syntax) | 0.7086 | 0.1347 | 0.6905 | 1.2411 |
| Step $t=4$ Drift (Entity Token) | 1.5340 | 0.4017 | 0.9309 | **0.2412 (Best)** |
| Step $t=10$ Drift (Terminal Token) | 1.1162 | **0.0658** | 0.3560 | 1.2952 |

---

## Key Experimental Findings

### 1. The Token-0 Boundary Trap
Prior prompt compression approaches penalize distribution divergence exclusively at token index $t=0$. Because the opening token of an answer is almost always syntactic boilerplate (*"The"*, *"In"*, *"According"*), the mutual information between prompt entities and $Y_0$ is near zero ($I(X_{\text{entity}}; Y_0 \mid \text{Prompt}) \approx 0$). The policy exploits this loophole by deleting entity anchors, creating an illusion of high compression ($74.14\%$ drop) while collapsing downstream Exact Match to **$6.67\%$**.

### 2. The 4-GPU Pareto Sweet Spot
Evaluating trajectories across distributed workers (4-GPU DDP) yields the optimal operational balance:
* **Token Drop Rate:** Halves prompt context length (**$46.94\%$** reduction).
* **Factual Integrity:** Retains **$75.42\%$ F1** and **$33.33\%$ Exact Match** ($5.0\times$ higher factual retention than baseline).
* **Drift Suppression:** Reduces terminal token drift at $t=10$ by **$68.1\%$** ($1.116 \to 0.356$).

### 3. Policy Saturation & The Hallucination Cliff (87k Run)
Running unconstrained policy optimization over 87,599 samples (21.9k steps) with a fixed length penalty ($\beta = 0.5$) revealed an empirical boundary:
* The policy achieved the lowest entity drift at $t=4$ ($D_{\mathrm{KL}} = \mathbf{0.2412}$), showing deep entity awareness.
* However, chasing the linear length reward caused the drop rate to drift to **$75.93\%$**, triggering a severe hallucination cliff where downstream Exact Match collapsed back to $6.7\%$.
* **Takeaway:** Length incentives must be constrained via early stopping near step 1,500 or dynamic $\beta$-annealing to permanently lock in the $47\%$ drop Pareto sweet spot.

---

## Compiling Thesis Defense Slides

The repository includes the 11-slide Beamer presentation deck (`slides.tex`):

```bash
# Compile LaTeX slides to PDF
pdflatex slides.tex
pdflatex slides.tex  # Second pass resolves layout metrics and slide numbers
```

*Output:* `slides.pdf` (an unclipped presentation covering motivation, mathematical formulation, distributed architecture, and empirical findings).

---

## Citation & Author

**Muhammad Shaheer Saleh**  
Computer Science Graduate Studies  
Research focus: Reinforcement Learning, Prompt Optimization, LLM Interpretability.
