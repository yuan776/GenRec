"""Evaluation metrics for recommendation: Recall@K and NDCG@K."""
import numpy as np
from typing import List, Dict
import math

def compute_recall_at_k(predictions: List[int], ground_truth: int, k: int) -> float:
    """Recall@K: 1 if ground_truth in top-k predictions, else 0."""
    return 1.0 if ground_truth in predictions[:k] else 0.0

def compute_ndcg_at_k(predictions: List[int], ground_truth: int, k: int) -> float:
    """NDCG@K for single ground truth item."""
    for i, pred in enumerate(predictions[:k]):
        if pred == ground_truth:
            return 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1, log2(rank+1)
    return 0.0

class TopKAccumulator:
    """Accumulates Recall@K and NDCG@K metrics over batches."""
    def __init__(self, ks: List[int] = [5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.recall = {k: 0.0 for k in self.ks}
        self.ndcg = {k: 0.0 for k in self.ks}
        self.count = 0

    def update(self, predictions: List[int], ground_truth: int):
        """Update with one sample's predictions and ground truth."""
        self.count += 1
        for k in self.ks:
            self.recall[k] += compute_recall_at_k(predictions, ground_truth, k)
            self.ndcg[k] += compute_ndcg_at_k(predictions, ground_truth, k)

    def update_batch(self, batch_predictions: List[List[int]], batch_ground_truths: List[int]):
        """Update with a batch of predictions."""
        for preds, gt in zip(batch_predictions, batch_ground_truths):
            self.update(preds, gt)

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        if self.count == 0:
            return {}
        results = {}
        for k in self.ks:
            results[f"Recall@{k}"] = self.recall[k] / self.count
            results[f"NDCG@{k}"] = self.ndcg[k] / self.count
        return results

    def __str__(self) -> str:
        metrics = self.compute()
        parts = [f"{name}: {val:.4f}" for name, val in metrics.items()]
        return " | ".join(parts)
