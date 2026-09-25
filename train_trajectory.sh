#!/bin/bash
#SBATCH --job-name=train_traj_rl
#SBATCH --output=logs/train_traj-%j.out
#SBATCH --error=logs/train_traj-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1

set -e
mkdir -p logs checkpoints/trajectory_rl
source venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# If models were cached in /project, use that path. Otherwise keep the default ~/.cache
if [ -d "/project/6112831/shaheer/.cache/huggingface/hub" ]; then
    export HF_HOME=/project/6112831/shaheer/.cache/huggingface
fi

echo "=========================================================="
echo "Starting Trajectory REINFORCE on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

python -u train_trajectory_reinforce.py \
    --data_path data/train.jsonl \
    --base_checkpoint checkpoints/checkpoint_epoch_1 \
    --eval_model microsoft/phi-4 \
    --output_dir checkpoints/trajectory_rl \
    --lr 2e-6 \
    --beta 0.5 \
    --max_samples 500 \
    --max_answer_tokens 24

echo "=========================================================="
echo "Training finished successfully at: $(date)"
echo "=========================================================="
