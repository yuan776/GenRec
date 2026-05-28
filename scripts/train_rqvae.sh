#!/bin/bash
# Train RQVAE for Semantic ID generation
# Usage: bash scripts/train_rqvae.sh [split] [device]

SPLIT=${1:-beauty}
DEVICE=${2:-cuda}

echo "Training RQVAE on Amazon ${SPLIT}..."

python src/trainers/rqvae_trainer.py \
    --split ${SPLIT} \
    --data_dir ./data \
    --embedding_dim 64 \
    --rqvae_dim 256 \
    --num_codebooks 3 \
    --codebook_size 256 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --device ${DEVICE} \
    --save_dir ./checkpoints

echo "RQVAE training complete. Item codes saved to ./checkpoints/rqvae_${SPLIT}/item_codes.npy"
