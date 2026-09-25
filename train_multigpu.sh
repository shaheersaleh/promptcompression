#!/bin/bash
#SBATCH --job-name=traj_rl_4gpu
#SBATCH --output=logs/4gpu-%j.out
#SBATCH --error=logs/4gpu-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

set -e
mkdir -p logs checkpoints/trajectory_rl_ddp

# Use full absolute path to the virtualenv activation script
source /project/6112831/shaheer/phi4_compression_rl/venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

# Detect cache directory
if [ -d "/project/6112831/shaheer/.cache/huggingface/hub" ]; then
    export HF_HOME=/project/6112831/shaheer/.cache/huggingface
fi

echo "=========================================================="
echo "Starting 4-GPU Distributed Trajectory RL"
echo "Host: $(hostname) | Slurm Job ID: $SLURM_JOB_ID"
echo "Active Python: $(which python)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================================="

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_trajectory_reinforce_ddp.py \
        --data_path data/train.jsonl \
        --base_checkpoint checkpoints/checkpoint_epoch_1 \
        --eval_model microsoft/phi-4 \
        --output_dir checkpoints/trajectory_rl_ddp \
        --lr 2e-6 \
        --beta 0.5 \
        --max_answer_tokens 24 \
        --log_interval 10 \
        --save_interval 250

echo "=========================================================="
echo "Job finished successfully at: $(date)"
echo "=========================================================="
