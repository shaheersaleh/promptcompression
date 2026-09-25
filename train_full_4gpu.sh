#!/bin/bash
#SBATCH --job-name=traj_squad_4gpu
#SBATCH --output=logs/full_4gpu-%j.out
#SBATCH --error=logs/full_4gpu-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

set -e
mkdir -p logs checkpoints/trajectory_rl_ddp_full
source /project/6112831/shaheer/phi4_compression_rl/venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4

# HuggingFace Cache
if [ -d "/project/6112831/shaheer/.cache/huggingface/hub" ]; then
    export HF_HOME=/project/6112831/shaheer/.cache/huggingface
fi

echo "=========================================================="
echo "Starting Full SQuAD Trajectory RL across 4 GPUs"
echo "Host: $(hostname) | Slurm Job ID: $SLURM_JOB_ID"
echo "Total Samples: $(wc -l < data/train_full.jsonl)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================================="

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_trajectory_reinforce_ddp.py \
        --data_path data/train_full.jsonl \
        --base_checkpoint checkpoints/checkpoint_epoch_1 \
        --eval_model microsoft/phi-4 \
        --output_dir checkpoints/trajectory_rl_ddp_full \
        --lr 2e-6 \
        --beta 0.5 \
        --max_answer_tokens 24 \
        --log_interval 50 \
        --save_interval 1000

echo "=========================================================="
echo "Full dataset run completed successfully at: $(date)"
echo "=========================================================="
