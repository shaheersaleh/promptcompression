#!/bin/bash
#SBATCH --job-name=audit_traj_rl
#SBATCH --output=logs/audit_traj-%j.out
#SBATCH --error=logs/audit_traj-%j.err
#SBATCH --time=01:30:00
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
echo "Auditing Trajectory RL Checkpoint on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

python -u run_comprehensive_audit.py \
    --data_path data/train.jsonl \
    --output_file audit_results_trajectory_rl.jsonl \
    --checkpoint checkpoints/trajectory_rl/checkpoint_final \
    --num_samples 30 \
    --threshold 0.5

echo "=========================================================="
echo "Audit finished at: $(date)"
echo "=========================================================="
