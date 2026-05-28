"""
RQVAE Trainer - Trains Residual Quantized VAE for Semantic ID generation.
Produces item_codes mapping for use in GenRec training.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.rqvae import RQVAE
from src.data.amazon import AmazonDataset


def train_rqvae(
    split: str = "beauty",
    data_dir: str = "./data",
    embedding_dim: int = 64,
    rqvae_dim: int = 256,
    num_codebooks: int = 3,
    codebook_size: int = 256,
    hidden_dims: Optional[list] = None,
    commitment_cost: float = 0.25,
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    kmeans_init: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "./checkpoints",
):
    """Train RQVAE and save item codes."""
    if hidden_dims is None:
        hidden_dims = [512, 256]

    print(f"Training RQVAE on Amazon {split}")
    print(f"Device: {device}")

    # Load dataset
    dataset = AmazonDataset(data_dir=data_dir, split=split)
    print(f"Loaded {dataset.num_items} items")

    # Get item embeddings (in practice, these come from a pretrained model)
    item_embeddings = dataset.get_item_embeddings(embedding_dim=embedding_dim)
    # Remove padding row (index 0)
    train_embeddings = torch.FloatTensor(item_embeddings[1:])  # [num_items, embed_dim]

    # Create model
    model = RQVAE(
        input_dim=embedding_dim,
        embedding_dim=rqvae_dim,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        hidden_dims=hidden_dims,
        commitment_cost=commitment_cost,
    ).to(device)

    # K-means initialization
    if kmeans_init:
        print("Initializing codebooks with K-means...")
        init_data = train_embeddings[:min(10000, len(train_embeddings))].to(device)
        model.init_codebook_with_kmeans(init_data)
        print("K-means initialization done")

    # DataLoader
    train_dataset = TensorDataset(train_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # early stopping patience
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for (batch_embeddings,) in pbar:
            batch_embeddings = batch_embeddings.to(device)

            recon, indices, commit_loss, recon_loss = model(batch_embeddings)
            loss = recon_loss + commit_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "recon": f"{recon_loss.item():.4f}",
                "commit": f"{commit_loss.item():.4f}",
            })

        scheduler.step()

        avg_recon = total_recon_loss / num_batches
        avg_commit = total_commit_loss / num_batches
        avg_loss = avg_recon + avg_commit

        print(f"Epoch {epoch+1}/{epochs} | Recon: {avg_recon:.4f} | Commit: {avg_commit:.4f} | Total: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_path = os.path.join(save_dir, f"rqvae_{split}")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Load best model for code generation
    best_model_path = os.path.join(save_dir, f"rqvae_{split}", "model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model (loss={best_loss:.4f})")

    # Generate and save item codes
    print("\nGenerating item codes...")
    model.eval()
    all_codes = []
    with torch.no_grad():
        for i in range(0, len(train_embeddings), batch_size):
            batch = train_embeddings[i:i+batch_size].to(device)
            codes = model.get_codes(batch)
            all_codes.append(codes.cpu())

    all_codes = torch.cat(all_codes, dim=0).numpy()  # [num_items, num_codebooks]

    # Add padding row at index 0
    item_codes = np.zeros((dataset.num_items + 1, num_codebooks), dtype=np.int64)
    item_codes[1:] = all_codes

    codes_path = os.path.join(save_dir, f"rqvae_{split}", "item_codes.npy")
    np.save(codes_path, item_codes)
    print(f"Saved item codes to {codes_path}")
    print(f"Code distribution stats per codebook:")
    for c in range(num_codebooks):
        unique = len(np.unique(all_codes[:, c]))
        print(f"  Codebook {c}: {unique}/{codebook_size} codes used")

    # Check collision rate
    code_tuples = [tuple(row) for row in all_codes]
    unique_tuples = len(set(code_tuples))
    collision_rate = 1.0 - unique_tuples / len(code_tuples)
    print(f"Collision rate: {collision_rate:.4f} ({len(code_tuples) - unique_tuples} collisions)")

    return item_codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RQVAE for Semantic ID generation")
    parser.add_argument("--split", type=str, default="beauty", choices=["beauty", "sports", "toys"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--rqvae_dim", type=int, default=256)
    parser.add_argument("--num_codebooks", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda/mps/cpu. Auto-detected if not specified.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    train_rqvae(
        split=args.split,
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        rqvae_dim=args.rqvae_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )
