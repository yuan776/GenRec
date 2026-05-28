#!/bin/bash
# Train GenRec with Page-wise NTP SFT
# Usage: bash scripts/train_genrec.sh [split] [model_path] [device]

SPLIT=${1:-beauty}
MODEL_PATH=${2:-Qwen/Qwen2.5-0.5B}
DEVICE=${3:-cuda}

echo "Training GenRec on Amazon ${SPLIT} with model ${MODEL_PATH}..."

python src/trainers/genrec_trainer.py \
    --split ${SPLIT} \
    --data_dir ./data \
    --model_path ${MODEL_PATH} \
    --num_codebooks 3 \
    --codebook_size 256 \
    --use_token_merger \
    --page_size 3 \
    --max_seq_len 50 \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --gradient_checkpointing \
    --beam_width 20 \
    --eval_topk 10 \
    --eval_every 5 \
    --device ${DEVICE} \
    --save_dir ./checkpoints

echo "GenRec training complete."
