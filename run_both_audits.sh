#!/bin/bash
#SBATCH --job-name=eval_both_audits
#SBATCH --output=logs/eval_both-%j.out
#SBATCH --error=logs/eval_both-%j.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e
mkdir -p logs
source venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -d "/project/6112831/shaheer/.cache/huggingface/hub" ]; then
    export HF_HOME=/project/6112831/shaheer/.cache/huggingface
fi

echo "=========================================================="
echo "Phase 1: Auditing Baseline Checkpoint (checkpoint_epoch_1)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

python -u run_comprehensive_audit.py \
    --data_path data/train.jsonl \
    --output_file comprehensive_audit_results.jsonl \
    --checkpoint checkpoints/checkpoint_epoch_1 \
    --num_samples 30 \
    --threshold 0.5

echo "=========================================================="
echo "Phase 2: Auditing Trajectory RL Checkpoint (checkpoint_final)"
echo "Time: $(date)"
echo "=========================================================="

python -u run_comprehensive_audit.py \
    --data_path data/train.jsonl \
    --output_file audit_results_trajectory_rl.jsonl \
    --checkpoint checkpoints/trajectory_rl/checkpoint_final \
    --num_samples 30 \
    --threshold 0.5

echo "=========================================================="
echo "Phase 3: Computing Head-to-Head Comparison"
echo "=========================================================="

python compare_audits.py comprehensive_audit_results.jsonl audit_results_trajectory_rl.jsonl

echo "=========================================================="
echo "All audits and comparisons completed at: $(date)"
echo "=========================================================="
