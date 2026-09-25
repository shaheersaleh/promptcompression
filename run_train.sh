#!/bin/bash
#SBATCH --job-name=phi4_compress_rl
#SBATCH --account=def-sadafsk
#SBATCH --partition=livi
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.out
#SBATCH --error=/project/6112831/%u/phi4_compression_rl/logs/%x-%j.err

module load cuda/12.4.1 arch/avx2 gcc/12.3.0 python/3.10.14

export WORKDIR=/project/6112831/$USER/phi4_compression_rl
export HF_HOME=$WORKDIR/cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $WORKDIR/venv/bin/activate

python $WORKDIR/src/train_compressor_rl.py \
    --phi4_id microsoft/phi-4 \
    --encoder_id microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --cached_targets $WORKDIR/data/cached_phi4_top100.pt \
    --output_dir $WORKDIR/checkpoints \
    --epochs 3 \
    --lr 1e-5 \
    --beta 1.5 \
    --max_len 1024
