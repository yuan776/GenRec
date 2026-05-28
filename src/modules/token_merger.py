"""Token Merger: compresses multi-token Semantic IDs into a single latent vector."""
import torch
from torch import nn

class TokenMerger(nn.Module):
    """
    Linear Token Merger from the GenRec paper.
    Compresses num_codes SID embeddings per item into a single hidden vector.
    h_vi = Linear(Concat(e(si^1), e(si^2), e(si^3)))
    """
    def __init__(self, embed_dim: int, num_codes: int = 3):
        super().__init__()
        self.num_codes = num_codes
        self.proj = nn.Linear(embed_dim * num_codes, embed_dim, bias=False)

    def forward(self, sid_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sid_embeddings: [batch, num_items, num_codes, embed_dim]
        Returns:
            merged: [batch, num_items, embed_dim]
        """
        batch, num_items, num_codes, embed_dim = sid_embeddings.shape
        # Concat along last dim: [batch, num_items, num_codes * embed_dim]
        concatenated = sid_embeddings.view(batch, num_items, num_codes * embed_dim)
        # Project to single vector
        merged = self.proj(concatenated)
        return merged
