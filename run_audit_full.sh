#!/bin/bash
#SBATCH --job-name=audit_87k
#SBATCH --output=logs/audit_full-%j.out
#SBATCH --error=logs/audit_full-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
echo "Auditing Full 87k SQuAD DDP Checkpoint on $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start Time: $(date)"
echo "=========================================================="

python -u run_comprehensive_audit.py \
    --data_path data/train.jsonl \
    --output_file audit_results_ddp_full.jsonl \
    --checkpoint checkpoints/trajectory_rl_ddp_full/checkpoint_final \
    --num_samples 30 \
    --threshold 0.5

echo "=========================================================="
echo "Audit complete! Output saved to audit_results_ddp_full.jsonl"
echo "=========================================================="
