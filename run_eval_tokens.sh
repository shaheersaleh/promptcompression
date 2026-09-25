#!/bin/bash
#SBATCH --job-name=eval_tokens
#SBATCH --account=def-sadafsk
#SBATCH --partition=livi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.out
#SBATCH --error=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.err

module load cuda/12.4.1 arch/avx2 gcc/12.3.0 python/3.10.14

export WORKDIR=/project/6112831/$USER/phi4_compression_rl
export HF_HOME=$WORKDIR/cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

source $WORKDIR/venv/bin/activate

python $WORKDIR/src/eval_token_distributions.py \
    --phi4_id microsoft/phi-4 \
    --checkpoint_dir $WORKDIR/checkpoints/checkpoint_epoch_1 \
    --test_data $WORKDIR/data/train.jsonl \
    --sample_idx 0 \
    --max_eval_tokens 16 \
    --output_csv $WORKDIR/token_distribution_trace.csv
