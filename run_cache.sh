#!/bin/bash
#SBATCH --job-name=phi4_cache
#SBATCH --account=def-sadafsk
#SBATCH --partition=livi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.out
#SBATCH --error=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.err

module load cuda/12.4.1 arch/avx2 gcc/12.3.0 python/3.10.14

export WORKDIR=/project/6112831/$USER/phi4_compression_rl
export HF_HOME=$WORKDIR/cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/cache/huggingface

source $WORKDIR/venv/bin/activate

python $WORKDIR/src/cache_phi4_logits.py \
    --model_id microsoft/phi-4 \
    --data_path $WORKDIR/data/train.jsonl \
    --output_path $WORKDIR/data/cached_phi4_top100.pt \
    --top_k 100 \
    --max_prompt_len 2048
