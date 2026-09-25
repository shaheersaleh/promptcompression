# Trajectory-Aware Reinforcement Learning for Discrete Prompt Compression

Reinforcement learning framework for upstream discrete prompt pruning using an **XLM-RoBERTa (560M)** editor evaluated against a frozen downstream **Microsoft Phi-4 (14B, 4-bit NF4)** language model.

## Method Highlights
- **Discrete Hard Masking**: Enforces strict binary token masks ($m \in \{0, 1\}^N$) directly via REINFORCE, completely avoiding test-time continuous distribution collapse.
- **Dense Trajectory Divergence**: Supervises upstream pruning using step-by-step $D_{\mathrm{KL}}$ distributions across target answer tokens ($t = 1 \dots T$) via single-pass causal evaluation.
- **Distributed All-Reduce**: Scales exploration across 4 GPUs via PyTorch DDP; synchronizes policy gradients via NCCL All-Reduce rather than mask averaging.

## Macro Benchmark Results

| Model / Budget | Drop Rate | Downstream F1 | Exact Match (EM) | Step $t=4$ Drift (Entity) | Step $t=10$ Drift (Terminal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Token-0 Baseline** (300 steps) | 74.14% | 0.5991 | 6.7% | 1.5340 | 1.1162 |
| **Traj RL (1-GPU)** (300 steps) | 20.77% | 0.7978 | 46.7% | 0.4017 | 0.0658 |
| **Traj RL (4-GPU)** (300 steps) | **46.94%** | **0.7542** | **33.3%** | **0.9309** | **0.3560** |
| **Traj RL Full (87k)** (21.9k steps) | 75.93% | 0.5772 | 6.7% | **0.2412** | 1.2952 |

## Reproduction

```bash
# 1. Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Distributed 4-GPU Training
bash train_multigpu.sh

# 3. Macro Audit & Benchmark Comparison
bash run_audit_full.sh
python compare_all.py
