"""
GenRec SFT Trainer - Page-wise Next Token Prediction training for GenRec.
Trains the decoder-only model with Token Merger on page-wise supervision.
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from typing import Optional, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.genrec import GenRec
from src.data.amazon import AmazonDataset
from src.data.genrec_dataset import GenRecSFTDataset, GenRecEvalDataset, collate_sft, collate_eval
from src.modules.metrics import TopKAccumulator


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup."""
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: GenRec,
    eval_loader: DataLoader,
    item_codes: np.ndarray,
    beam_width: int = 20,
    topk: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model with constrained beam search.
    Computes Recall@K and NDCG@K.
    """
    model.eval()
    accumulator = TopKAccumulator(ks=[5, 10])

    # Build reverse mapping: code tuple -> item_id
    code_to_item = {}
    for item_id in range(1, len(item_codes)):
        code_tuple = tuple(item_codes[item_id])
        if code_tuple not in code_to_item:
            code_to_item[code_tuple] = item_id

    num_hallucinations = 0
    num_total = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_codes = batch["input_codes"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_items = batch["target_items"].numpy()

            B = input_codes.shape[0]

            try:
                sem_ids, scores = model.generate_beam(
                    input_codes, input_lengths,
                    beam_width=beam_width, topk=topk
                )
            except Exception as e:
                print(f"Beam search error: {e}")
                continue

            # Convert predicted SIDs to item IDs
            for b in range(B):
                predictions = []
                for k in range(min(topk, sem_ids.shape[1])):
                    code_tuple = tuple(sem_ids[b, k].cpu().numpy())
                    item_id = code_to_item.get(code_tuple, -1)
                    if item_id == -1:
                        num_hallucinations += 1
                    else:
                        predictions.append(item_id)
                    num_total += 1

                accumulator.update(predictions, target_items[b])

    metrics = accumulator.compute()
    if num_total > 0:
        metrics["HaR"] = num_hallucinations / num_total
    return metrics


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_genrec(
    split: str = "beauty",
    data_dir: str = "./data",
    model_path: str = "Qwen/Qwen2.5-0.5B",
    codes_path: Optional[str] = None,
    num_codebooks: int = 3,
    codebook_size: int = 256,
    use_token_merger: bool = True,
    page_size: int = 3,
    max_seq_len: int = 50,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.01,
    gradient_checkpointing: bool = True,
    beam_width: int = 20,
    eval_topk: int = 10,
    eval_every: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "./checkpoints",
):
    """Train GenRec with Page-wise NTP SFT."""
    print(f"Training GenRec on Amazon {split}")
    print(f"Model: {model_path}")
    print(f"Token Merger: {use_token_merger}")
    print(f"Page Size: {page_size}")
    print(f"Device: {device}")

    # Load dataset
    dataset = AmazonDataset(data_dir=data_dir, split=split, max_seq_len=max_seq_len)
    train_seqs, val_targets, test_targets = dataset.get_splits()
    print(f"Dataset: {dataset.num_users} users, {dataset.num_items} items")
    print(f"Train sequences: {len(train_seqs)}")

    # Load item codes
    if codes_path is None:
        codes_path = os.path.join(save_dir, f"rqvae_{split}", "item_codes.npy")

    if os.path.exists(codes_path):
        item_codes = np.load(codes_path)
        print(f"Loaded item codes from {codes_path}")
    else:
        print(f"Item codes not found at {codes_path}. Generating random codes for demo...")
        item_codes = np.random.randint(0, codebook_size, size=(dataset.num_items + 1, num_codebooks))
        item_codes[0] = 0  # padding

    # Create datasets
    train_dataset = GenRecSFTDataset(
        user_sequences=train_seqs,
        item_codes=item_codes,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
    )
    print(f"Training samples: {len(train_dataset)}")

    val_dataset = GenRecEvalDataset(
        user_sequences=train_seqs,
        targets=val_targets,
        item_codes=item_codes,
        num_codebooks=num_codebooks,
        max_seq_len=max_seq_len,
    )
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sft, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=0,
    )

    # Create model
    print(f"Loading model from {model_path}...")
    model = GenRec(
        pretrained_path=model_path,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        use_token_merger=use_token_merger,
    ).to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_recall = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            input_codes = batch["input_codes"].to(device)
            target_codes = batch["target_codes"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            # Forward pass with Page-wise NTP loss
            outputs = model.forward_sft(
                input_codes=input_codes,
                target_codes=target_codes,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )

            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Evaluation
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            print("  Running evaluation...")
            metrics = evaluate(
                model, val_loader, item_codes,
                beam_width=beam_width, topk=eval_topk, device=device
            )
            print(f"  Metrics: {metrics}")

            recall_10 = metrics.get("Recall@10", 0.0)
            if recall_10 > best_recall:
                best_recall = recall_10
                save_path = os.path.join(save_dir, f"genrec_{split}")
                model.save_pretrained(save_path)
                print(f"  Saved best model (Recall@10={best_recall:.4f})")

    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    test_dataset = GenRecEvalDataset(
        user_sequences=train_seqs,
        targets=test_targets,
        item_codes=item_codes,
        num_codebooks=num_codebooks,
        max_seq_len=max_seq_len,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_eval, num_workers=0,
    )
    test_metrics = evaluate(
        model, test_loader, item_codes,
        beam_width=beam_width, topk=eval_topk, device=device
    )
    print(f"Test Results: {test_metrics}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GenRec with Page-wise NTP SFT")
    parser.add_argument("--split", type=str, default="beauty", choices=["beauty", "sports", "toys"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model path. Use 'sshleifer/tiny-gpt2' for quick CPU testing.")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: use tiny model, small data, few epochs for local testing")
    parser.add_argument("--codes_path", type=str, default=None)
    parser.add_argument("--num_codebooks", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--use_token_merger", action="store_true", default=True)
    parser.add_argument("--no_token_merger", dest="use_token_merger", action="store_false")
    parser.add_argument("--page_size", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda/mps/cpu. Auto-detected if not specified.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    if args.device is None:
        args.device = get_device()
    print(f"Using device: {args.device}")

    # Demo mode overrides for quick local testing
    if args.demo:
        print("=== DEMO MODE: using tiny model and reduced settings for local CPU testing ===")
        args.model_path = "sshleifer/tiny-gpt2"
        args.epochs = 3
        args.batch_size = 4
        args.max_seq_len = 10
        args.page_size = 2
        args.beam_width = 5
        args.eval_topk = 5
        args.eval_every = 1

    train_genrec(
        split=args.split,
        data_dir=args.data_dir,
        model_path=args.model_path,
        codes_path=args.codes_path,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        use_token_merger=args.use_token_merger,
        page_size=args.page_size,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        beam_width=args.beam_width,
        eval_topk=args.eval_topk,
        eval_every=args.eval_every,
        device=args.device,
        save_dir=args.save_dir,
    )
