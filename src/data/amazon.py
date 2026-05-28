"""
Amazon 2014 dataset loading and preprocessing.
Supports Beauty, Sports, Toys with 5-core filtering and leave-one-out split.
"""
import os
import json
import gzip
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from urllib.request import urlretrieve
from tqdm import tqdm


AMAZON_URLS = {
    "beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    "sports": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
    "toys": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
}


class AmazonDataset:
    """
    Amazon 2014 dataset with 5-core filtering and leave-one-out evaluation.

    Attributes:
        user_sequences: Dict[int, List[int]] - user_id -> list of item_ids (chronological)
        num_users: int
        num_items: int
        item_id_map: Dict[str, int] - original item ASIN -> internal item ID
    """

    def __init__(self, data_dir: str = "./data", split: str = "beauty", max_seq_len: int = 50):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len

        self.user_sequences: Dict[int, List[int]] = {}
        self.num_users = 0
        self.num_items = 0
        self.item_id_map: Dict[str, int] = {}
        self.user_id_map: Dict[str, int] = {}

        self._load_or_download()

    def _load_or_download(self):
        """Load preprocessed data or download and preprocess."""
        cache_path = os.path.join(self.data_dir, f"amazon_{self.split}_processed.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.user_sequences = data["user_sequences"]
            self.num_users = data["num_users"]
            self.num_items = data["num_items"]
            self.item_id_map = data["item_id_map"]
            self.user_id_map = data["user_id_map"]
            return

        # Download raw data
        os.makedirs(self.data_dir, exist_ok=True)
        raw_path = os.path.join(self.data_dir, f"reviews_{self.split}_5.json.gz")

        if not os.path.exists(raw_path):
            url = AMAZON_URLS[self.split]
            print(f"Downloading {self.split} dataset from {url}...")
            urlretrieve(url, raw_path)

        # Parse and preprocess
        self._preprocess(raw_path, cache_path)

    def _preprocess(self, raw_path: str, cache_path: str):
        """Parse raw JSON, apply 5-core filtering, build sequences."""
        # Parse reviews
        interactions = []
        with gzip.open(raw_path, "rt", encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                user = review["reviewerID"]
                item = review["asin"]
                time = review["unixReviewTime"]
                interactions.append((user, item, time))

        # Sort by time
        interactions.sort(key=lambda x: x[2])

        # Build user->items and item->users
        user_items = defaultdict(list)
        for user, item, time in interactions:
            user_items[user].append((item, time))

        # 5-core filtering (iterative)
        for _ in range(10):  # iterate until stable
            item_count = defaultdict(int)
            for user, items in user_items.items():
                for item, _ in items:
                    item_count[item] += 1

            # Remove items with < 5 interactions
            valid_items = {item for item, count in item_count.items() if count >= 5}
            new_user_items = {}
            for user, items in user_items.items():
                filtered = [(item, t) for item, t in items if item in valid_items]
                if len(filtered) >= 5:
                    new_user_items[user] = filtered
            user_items = new_user_items

        # Build ID mappings
        all_items = set()
        for user, items in user_items.items():
            for item, _ in items:
                all_items.add(item)

        self.item_id_map = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}  # 1-indexed
        self.user_id_map = {user: idx for idx, user in enumerate(sorted(user_items.keys()))}

        # Build sequences (sorted by time)
        self.user_sequences = {}
        for user, items in user_items.items():
            items_sorted = sorted(items, key=lambda x: x[1])
            seq = [self.item_id_map[item] for item, _ in items_sorted]
            self.user_sequences[self.user_id_map[user]] = seq

        self.num_users = len(self.user_sequences)
        self.num_items = len(self.item_id_map)

        # Save cache
        with open(cache_path, "wb") as f:
            pickle.dump({
                "user_sequences": self.user_sequences,
                "num_users": self.num_users,
                "num_items": self.num_items,
                "item_id_map": self.item_id_map,
                "user_id_map": self.user_id_map,
            }, f)

        print(f"Preprocessed {self.split}: {self.num_users} users, {self.num_items} items")

    def get_splits(self) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
        """
        Leave-one-out split: last item for test, second-to-last for validation.

        Returns:
            train_seqs: user_id -> training sequence (all but last 2)
            val_targets: user_id -> validation item
            test_targets: user_id -> test item
        """
        train_seqs = {}
        val_targets = {}
        test_targets = {}

        for user_id, seq in self.user_sequences.items():
            if len(seq) < 3:
                continue
            train_seqs[user_id] = seq[:-2]
            val_targets[user_id] = seq[-2]
            test_targets[user_id] = seq[-1]

        return train_seqs, val_targets, test_targets

    def get_item_embeddings(self, embedding_dim: int = 64) -> np.ndarray:
        """
        Generate item embeddings using SVD on user-item interaction matrix.
        This captures collaborative filtering signal for meaningful RQVAE codes.

        Args:
            embedding_dim: dimension of item embeddings
        Returns:
            embeddings: [num_items + 1, embedding_dim] array (index 0 is padding)
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        # Build user-item interaction matrix
        rows, cols = [], []
        for uid, items in self.user_sequences.items():
            for iid in items:
                rows.append(uid)  # user IDs are 0-indexed
                cols.append(iid - 1)  # item IDs are 1-indexed, convert to 0-indexed

        data = np.ones(len(rows), dtype=np.float32)
        interaction_matrix = csr_matrix(
            (data, (rows, cols)), shape=(self.num_users, self.num_items)
        )

        # SVD - use min of requested dim and matrix rank limit
        k = min(embedding_dim, min(self.num_users, self.num_items) - 1)
        _, s, Vt = svds(interaction_matrix, k=k)

        # Item embeddings = V * sqrt(S) for balanced scaling
        item_vecs = (Vt.T * np.sqrt(s)).astype(np.float32)  # [num_items, k]

        # Pad to embedding_dim if k < embedding_dim
        if k < embedding_dim:
            pad = np.zeros((self.num_items, embedding_dim - k), dtype=np.float32)
            item_vecs = np.concatenate([item_vecs, pad], axis=1)

        # Normalize to unit sphere for better quantization
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        item_vecs = item_vecs / norms

        # Add padding row at index 0
        embeddings = np.zeros((self.num_items + 1, embedding_dim), dtype=np.float32)
        embeddings[1:] = item_vecs

        return embeddings
