"""
Residual Quantized VAE (RQ-VAE) for Semantic ID generation.
Maps item embeddings to hierarchical discrete codes.
Reference: https://github.com/phonism/genrec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with EMA codebook updates."""

    def __init__(self, codebook_size: int, embed_dim: int, commitment_cost: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D] input vectors
        Returns:
            quantized: [B, D] quantized vectors
            indices: [B] codebook indices
            loss: scalar commitment loss
        """
        # Compute distances to codebook entries
        distances = (
            z.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=-1)
            - 2 * z @ self.embedding.weight.t()
        )

        # Find nearest codebook entry
        indices = distances.argmin(dim=-1)  # [B]
        quantized = self.embedding(indices)  # [B, D]

        # EMA codebook update (training only)
        if self.training:
            encodings = F.one_hot(indices, self.codebook_size).float()  # [B, K]
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            embed_sum = encodings.t() @ z  # [K, D]
            self.ema_embed.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            )
            self.embedding.weight.data.copy_(self.ema_embed / cluster_size.unsqueeze(-1))

            # Dead code revival: reinitialize unused codes from batch
            dead_mask = self.cluster_size < 1.0
            num_dead = dead_mask.sum().item()
            if num_dead > 0 and z.size(0) > 0:
                # Pick random samples from batch to replace dead codes
                num_replace = min(num_dead, z.size(0))
                rand_indices = torch.randperm(z.size(0), device=z.device)[:num_replace]
                dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_replace]
                self.embedding.weight.data[dead_indices] = z[rand_indices].detach()
                self.ema_embed.data[dead_indices] = z[rand_indices].detach()
                self.cluster_size.data[dead_indices] = 1.0

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss


class RQVAE(nn.Module):
    """
    Residual Quantized VAE for mapping item embeddings to multi-level semantic IDs.

    Each item embedding is encoded through an encoder, then quantized through
    multiple codebooks in sequence. Each subsequent codebook quantizes the
    residual from the previous level.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        hidden_dims: Optional[List[int]] = None,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # Encoder: input_dim -> embedding_dim
        if hidden_dims is None:
            hidden_dims = [512, 256]
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: embedding_dim -> input_dim
        decoder_layers = []
        prev_dim = embedding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Residual quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
            for _ in range(num_codebooks)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction."""
        return self.decoder(z)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Residual quantization through multiple codebooks.

        Args:
            z: [B, D] encoded vectors
        Returns:
            quantized_sum: [B, D] sum of all quantized vectors
            all_indices: [B, num_codebooks] codebook indices at each level
            total_loss: scalar total commitment loss
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)

        for quantizer in self.quantizers:
            quantized, indices, loss = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized.detach()
            all_indices.append(indices)
            total_loss = total_loss + loss

        all_indices = torch.stack(all_indices, dim=-1)  # [B, num_codebooks]
        return quantized_sum, all_indices, total_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: [B, input_dim] item embeddings
        Returns:
            recon: [B, input_dim] reconstructed embeddings
            indices: [B, num_codebooks] semantic IDs
            commit_loss: scalar commitment loss
            recon_loss: scalar reconstruction loss
        """
        z = self.encode(x)
        quantized, indices, commit_loss = self.quantize(z)
        recon = self.decode(quantized)

        recon_loss = F.mse_loss(recon, x)

        return recon, indices, commit_loss, recon_loss

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get semantic ID codes for items (inference only).

        Args:
            x: [B, input_dim] item embeddings
        Returns:
            indices: [B, num_codebooks] semantic IDs
        """
        z = self.encode(x)
        _, indices, _ = self.quantize(z)
        return indices

    def init_codebook_with_kmeans(self, data: torch.Tensor, n_iter: int = 20):
        """
        Initialize codebook embeddings using K-means clustering.

        Args:
            data: [N, input_dim] training data
            n_iter: number of K-means iterations
        """
        with torch.no_grad():
            z = self.encode(data)
            residual = z

            for quantizer in self.quantizers:
                # Simple K-means initialization
                indices = torch.randperm(z.size(0))[:quantizer.codebook_size]
                centroids = residual[indices].clone()

                for _ in range(n_iter):
                    distances = (
                        residual.pow(2).sum(-1, keepdim=True)
                        + centroids.pow(2).sum(-1)
                        - 2 * residual @ centroids.t()
                    )
                    assignments = distances.argmin(dim=-1)

                    for k in range(quantizer.codebook_size):
                        mask = assignments == k
                        if mask.sum() > 0:
                            centroids[k] = residual[mask].mean(dim=0)

                quantizer.embedding.weight.data.copy_(centroids)
                quantizer.ema_embed.data.copy_(centroids)

                # Update residual
                quantized = quantizer.embedding(assignments)
                residual = residual - quantized
