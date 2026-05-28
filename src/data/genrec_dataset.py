"""
GenRec Page-wise SFT Dataset and Evaluation Dataset.
Implements the Page-wise NTP training format from the GenRec paper.
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np


class GenRecSFTDataset(Dataset):
    """
    Page-wise SFT dataset for GenRec training.

    Training format:
    - Input (prompt): User history items as SID tokens, merged via Token Merger
    - Target (response): A "page" of next items as full SID tokens

    For academic datasets without real page data, we use the last `page_size`
    items as a synthetic page target.
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        item_codes: np.ndarray,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        max_seq_len: int = 50,
        page_size: int = 3,
    ):
        """
        Args:
            user_sequences: user_id -> list of item_ids (training portion)
            item_codes: [num_items+1, num_codebooks] array of semantic IDs
            num_codebooks: number of codebook levels
            codebook_size: size of each codebook
            max_seq_len: max number of items in input history
            page_size: number of items per page (target)
        """
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.item_codes = item_codes  # [num_items+1, num_codebooks]

        # Build training samples: each sample is (history, page_targets)
        self.samples: List[Tuple[List[int], List[int]]] = []
        for user_id, seq in user_sequences.items():
            if len(seq) < page_size + 1:
                continue
            # Create multiple training samples with sliding window
            for end_idx in range(page_size, len(seq)):
                start_idx = max(0, end_idx - page_size - max_seq_len)
                history = seq[start_idx:end_idx - page_size]
                page_items = seq[end_idx - page_size:end_idx]
                if len(history) > 0:
                    self.samples.append((history[-max_seq_len:], page_items))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns tokenized prompt and response for SFT.

        Returns dict with:
            - input_codes: [seq_len, num_codebooks] SID codes for input items
            - target_codes: [page_size, num_codebooks] SID codes for target items
            - input_length: number of input items
            - target_length: number of target items
        """
        history, page_items = self.samples[idx]

        # Get SID codes for history items
        input_codes = np.array([self.item_codes[item_id] for item_id in history])
        # Get SID codes for page target items
        target_codes = np.array([self.item_codes[item_id] for item_id in page_items])

        return {
            "input_codes": torch.LongTensor(input_codes),    # [hist_len, num_codebooks]
            "target_codes": torch.LongTensor(target_codes),  # [page_size, num_codebooks]
            "input_length": len(history),
            "target_length": len(page_items),
        }


class GenRecEvalDataset(Dataset):
    """
    Evaluation dataset for GenRec (point-wise, single next item prediction).

    Uses leave-one-out protocol: predict the held-out item given history.
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        targets: Dict[int, int],
        item_codes: np.ndarray,
        num_codebooks: int = 3,
        max_seq_len: int = 50,
    ):
        """
        Args:
            user_sequences: user_id -> training sequence
            targets: user_id -> target item_id
            item_codes: [num_items+1, num_codebooks] semantic IDs
            num_codebooks: number of codebook levels
            max_seq_len: max input sequence length
        """
        self.num_codebooks = num_codebooks
        self.max_seq_len = max_seq_len
        self.item_codes = item_codes

        self.samples: List[Tuple[List[int], int]] = []
        for user_id in user_sequences:
            if user_id in targets:
                seq = user_sequences[user_id][-max_seq_len:]
                self.samples.append((seq, targets[user_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            - input_codes: [seq_len, num_codebooks] SID codes for input
            - target_item: int, the ground truth item ID
            - target_codes: [num_codebooks] SID codes for target item
        """
        history, target_item = self.samples[idx]

        input_codes = np.array([self.item_codes[item_id] for item_id in history])
        target_codes = self.item_codes[target_item]

        return {
            "input_codes": torch.LongTensor(input_codes),
            "target_item": target_item,
            "target_codes": torch.LongTensor(target_codes),
            "input_length": len(history),
        }


def collate_sft(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for SFT training. Pads sequences to max length in batch.
    """
    max_input_len = max(item["input_length"] for item in batch)
    max_target_len = max(item["target_length"] for item in batch)
    num_codebooks = batch[0]["input_codes"].shape[-1]

    batch_size = len(batch)
    input_codes = torch.zeros(batch_size, max_input_len, num_codebooks, dtype=torch.long)
    target_codes = torch.zeros(batch_size, max_target_len, num_codebooks, dtype=torch.long)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        il = item["input_length"]
        tl = item["target_length"]
        input_codes[i, :il] = item["input_codes"]
        target_codes[i, :tl] = item["target_codes"]
        input_lengths[i] = il
        target_lengths[i] = tl

    return {
        "input_codes": input_codes,
        "target_codes": target_codes,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }


def collate_eval(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for evaluation."""
    max_input_len = max(item["input_length"] for item in batch)
    num_codebooks = batch[0]["input_codes"].shape[-1]

    batch_size = len(batch)
    input_codes = torch.zeros(batch_size, max_input_len, num_codebooks, dtype=torch.long)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_items = torch.zeros(batch_size, dtype=torch.long)
    target_codes = torch.zeros(batch_size, num_codebooks, dtype=torch.long)

    for i, item in enumerate(batch):
        il = item["input_length"]
        input_codes[i, :il] = item["input_codes"]
        input_lengths[i] = il
        target_items[i] = item["target_item"]
        target_codes[i] = item["target_codes"]

    return {
        "input_codes": input_codes,
        "input_lengths": input_lengths,
        "target_items": target_items,
        "target_codes": target_codes,
    }
