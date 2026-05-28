#!/bin/bash
# Evaluate a trained GenRec model
# Usage: bash scripts/evaluate.sh [split] [device]

SPLIT=${1:-beauty}
DEVICE=${2:-cuda}

echo "Evaluating GenRec on Amazon ${SPLIT}..."

python -c "
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.genrec import GenRec
from src.data.amazon import AmazonDataset
from src.data.genrec_dataset import GenRecEvalDataset, collate_eval
from src.trainers.genrec_trainer import evaluate

split = '${SPLIT}'
device = '${DEVICE}'

# Load dataset
dataset = AmazonDataset(data_dir='./data', split=split)
train_seqs, val_targets, test_targets = dataset.get_splits()

# Load item codes
codes_path = f'./checkpoints/rqvae_{split}/item_codes.npy'
item_codes = np.load(codes_path)

# Create test dataset
test_dataset = GenRecEvalDataset(
    user_sequences=train_seqs,
    targets=test_targets,
    item_codes=item_codes,
    num_codebooks=3,
    max_seq_len=50,
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_eval)

# Load model
model_path = f'./checkpoints/genrec_{split}'
model = GenRec(
    pretrained_path=model_path,
    num_codebooks=3,
    codebook_size=256,
    use_token_merger=True,
)
model = model.to(device)

# Evaluate
metrics = evaluate(model, test_loader, item_codes, beam_width=20, topk=10, device=device)
print()
print('=' * 50)
print(f'Test Results on Amazon {split}:')
for name, val in metrics.items():
    print(f'  {name}: {val:.4f}')
print('=' * 50)
"
