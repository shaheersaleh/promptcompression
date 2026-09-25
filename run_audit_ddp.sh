#!/bin/bash
#SBATCH --job-name=audit_ddp
#SBATCH --output=logs/audit_ddp-%j.out
#SBATCH --error=logs/audit_ddp-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e
mkdir -p logs
source /project/6112831/shaheer/phi4_compression_rl/venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -d "/project/6112831/shaheer/.cache/huggingface/hub" ]; then
    export HF_HOME=/project/6112831/shaheer/.cache/huggingface
fi

echo "=========================================================="
echo "Auditing 4-GPU DDP Checkpoint on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

python -u run_comprehensive_audit.py \
    --data_path data/train.jsonl \
    --output_file audit_results_ddp.jsonl \
    --checkpoint checkpoints/trajectory_rl_ddp/checkpoint_final \
    --num_samples 30 \
    --threshold 0.5

echo "=========================================================="
echo "Comparing All Three Models: Baseline vs. 1-GPU vs. 4-GPU DDP"
echo "=========================================================="

python compare_audits.py comprehensive_audit_results.jsonl audit_results_ddp.jsonl

echo "=========================================================="
echo "Finished at: $(date)"
echo "=========================================================="
