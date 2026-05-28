# JDGenRec: Implementing the GenRec Paper (Simplified)

## Goal

Implement the **GenRec** algorithm from the paper:
> *"GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation"*
> (Zou et al., SIGIR 2026) — [arXiv:2604.14878](https://arxiv.org/abs/2604.14878)

**Scope**: We implement the two core SFT-stage innovations (Page-wise NTP + Token Merger) but **skip the GRPO-SR reinforcement learning stage** to simplify the task.

---

## Paper Summary

GenRec is a decoder-only generative recommendation system deployed on JD.com. It represents items as 3-level Semantic IDs (via RQ-VAE) and autoregressively generates the next item(s) a user will interact with.

### Key Innovations (what we implement)

| Component | Description |
|-----------|-------------|
| **Semantic IDs via RQ-VAE** | Items are encoded into 3-token hierarchical codes `(s^1, s^2, s^3)` using Residual Quantized VAE on item embeddings |
| **Page-wise NTP (PW-NTP)** | Instead of predicting one item at a time, the model is trained to predict an entire "page" of interacted items in one sequence, resolving the one-to-many label ambiguity |
| **Token Merger** | A linear layer that compresses the 3 SID embeddings of each item into 1 merged vector on the **prefilling** (input) side, reducing prompt length by ~2× while keeping full-resolution decoding |
| **Point-wise Beam Search** | At inference, standard beam search generates one item at a time (asymmetric with list-wise training) |

### What we skip

| Component | Reason |
|-----------|--------|
| **GRPO-SR** | Reinforcement learning with reward model, hybrid rewards, and NLL regularization — requires a separate reward model and complex RL infrastructure |

---

## Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                      GenRec Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User History: [v1, v2, ..., vn]                              │
│       ↓                                                       │
│  Each item → SID(vi) = {si^1, si^2, si^3}                    │
│       ↓                                                       │
│  ┌─────────────────────────────────┐                          │
│  │ TOKEN MERGER (Prefill Side)     │                          │
│  │ h_vi = Linear(Concat(e(si^1),  │                          │
│  │        e(si^2), e(si^3)))       │                          │
│  │ 3 tokens → 1 merged token      │                          │
│  └─────────────────────────────────┘                          │
│       ↓                                                       │
│  Compressed Prompt: [h_v1, <sep>, h_v2, <sep>, ..., h_vn]    │
│       ↓                                                       │
│  ┌─────────────────────────────────┐                          │
│  │ DECODER-ONLY TRANSFORMER       │                          │
│  │ (Qwen2.5 backbone)             │                          │
│  └─────────────────────────────────┘                          │
│       ↓                                                       │
│  Decode (full resolution): s^1, s^2, s^3, <sep>, s^1, ...    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Training**: Page-wise NTP loss over entire page of items (ordered > clicked > exposed)
**Inference**: Point-wise beam search generating one item at a time

---

## Implementation Plan

We base the implementation on the [phonism/genrec](https://github.com/phonism/genrec) repository's patterns (LCRec model + trainer structure) and adapt for the GenRec paper.

### Phase 1: Semantic ID Generation (RQVAE)

- **Reuse**: `genrec/models/rqvae.py` and `genrec/trainers/rqvae_trainer.py` from the reference repo
- **Purpose**: Train an RQVAE to convert item embeddings into 3-level discrete codes
- **Output**: A mapping `item_id → (code_1, code_2, code_3)` with configurable codebook size (e.g., 256 per level)

### Phase 2: Data Pipeline

- **Dataset**: Amazon 2014 (Beauty, Sports, Toys) with 5-core filtering, leave-one-out split
- **Training data format (Page-wise)**:
  - Input (prompt): User history as merged SID tokens separated by `<sep>`
  - Target (response): Multiple next items' SIDs concatenated (simulating a "page")
  - For academic datasets without real page data, we group consecutive items as synthetic pages (e.g., last K items as a "page")
- **Evaluation data format (Point-wise)**:
  - Input: User history
  - Target: Single next item (standard leave-one-out)

### Phase 3: Model — GenRec with Token Merger

- **Backbone**: Qwen2.5 (1.5B or smaller) decoder-only LLM
- **Modifications on top of LCRec-style model**:
  1. **Codebook token embedding**: Add `num_codebooks × codebook_size` special tokens to tokenizer
  2. **Token Merger module**: `nn.Linear(3 * hidden_dim, hidden_dim)` that merges 3 SID embeddings into 1 on the input side
  3. **Asymmetric forward pass**:
     - Prefilling: Apply token merger to compress input item SIDs
     - Decoding: Generate full-resolution SID tokens autoregressively
  4. **Special tokens**: `<sep>` between items, `<page>` separator if needed

### Phase 4: Training (Page-wise NTP SFT)

- **Loss**: Standard cross-entropy on the response portion (page of SIDs)
- **Optimizer**: AdamW with linear warmup + cosine decay
- **Training regime**:
  - Gradient checkpointing for memory efficiency
  - bf16 mixed precision
  - configurable batch size and max sequence length

### Phase 5: Evaluation

- **Constrained beam search** over valid SID vocabulary (same as LCRec)
- **Metrics**: Recall@K, NDCG@K (K=5, 10) with full-item-set ranking
- **Hallucination Rate**: Percentage of generated SIDs that don't map to real items

---

## Reasoning & Design Decisions

### Why base on LCRec from genrec repo?

LCRec already implements the core pattern we need:
- Qwen2 backbone with codebook tokens for item SIDs
- SFT training with autoregressive loss
- Constrained beam search for evaluation
- Amazon dataset pipeline with RQVAE semantic IDs

GenRec is essentially an **enhanced LCRec** with:
1. Page-wise training targets (multi-item supervision)
2. Token merger for input compression
3. RL alignment (which we skip)

### Why Token Merger is a simple Linear layer?

The paper explicitly states:
> `h_vi = Linear(Concat(e(si^1), e(si^2), e(si^3)))`

This is elegant because:
- It's parameter-efficient (only one linear projection)
- It preserves information from all 3 SID levels
- It's applied only at input/prefill time, keeping decoding unchanged

### Why Page-wise training with Point-wise inference?

The paper argues this asymmetry is intentional:
- **Training**: List-wise supervision gives denser gradients and resolves the one-to-many ambiguity (same input → multiple valid outputs)
- **Inference**: Point-wise beam search is compatible with production serving and standard evaluation protocols

### Synthetic Page Construction for Academic Datasets

JD.com has real pagination data, but Amazon datasets don't. We simulate pages by:
- Taking the last `page_size` (e.g., 3-5) items from the sequence as a "page"
- Ordering them by interaction intensity if metadata available, otherwise chronologically
- This still provides the denser gradient benefit even without real page data

### Handling RQVAE Codebook Collapse

**Problem**: During RQVAE training, codebook collapse occurs when only 1-2 out of 256 codes are used per codebook, resulting in ~99.97% collision rate. This makes Semantic IDs useless for recommendation since nearly all items map to the same code.

**Root Causes Identified**:
1. **Random item embeddings lack structure** — Standard normal vectors in 64D are roughly equidistant from each other. After the encoder projects them to latent space, they all map to the same region, so a single codebook entry "wins" every assignment.
2. **Dead codes never recover** — Once a codebook entry stops receiving assignments, the EMA update with decay=0.99 causes it to drift further from the data distribution, making it permanently unused.

**Solution 1: SVD-based Collaborative Filtering Embeddings**

Instead of random embeddings, we compute item representations via truncated SVD on the user-item interaction matrix:
```python
U, S, Vt = svds(interaction_matrix, k=embedding_dim)
item_embeddings = Vt.T * sqrt(S)  # [num_items, dim]
item_embeddings = normalize(item_embeddings)  # unit sphere
```

**Why this works**:
- SVD captures the latent collaborative filtering structure: items bought by similar users get similar embeddings
- The resulting vectors have natural cluster structure (items form communities/categories)
- Unit-sphere normalization ensures all items are at equal distance from origin, preventing a single centroid from dominating
- With real structure, K-means initialization produces well-separated centroids, and the residual quantization at each level captures increasingly fine-grained distinctions

**Why this method over alternatives**:
| Method | Pros | Cons |
|--------|------|------|
| Random embeddings | Simple | No structure → collapse |
| Pretrained LLM (paper uses Qwen2.5-VL) | Best quality | Requires GPU, model download, slow |
| Word2Vec/Item2Vec on sequences | Good structure | Needs tuning, slower than SVD |
| **SVD on interaction matrix** | **Fast, captures real signal, no extra models** | Loses content features |

For a research prototype, SVD provides the best quality/complexity tradeoff. The paper's production system uses Qwen2.5-VL multimodal embeddings, which could be substituted later.

**Solution 2: Balanced K-means Initialization**

Standard K-means can itself produce degenerate initial centroids where some clusters are empty. Balanced K-means adds a capacity constraint: each centroid is assigned at most ⌈N/K⌉ points.

```python
# For each iteration:
capacity = ceil(N / K)
# Sort points by confidence (closest distance to preferred centroid)
# Assign greedily: each point goes to nearest centroid with remaining capacity
```

Combined with **K-means++ seeding** (initialize centroids spread apart by distance-weighted sampling), this guarantees:
- Every codebook entry starts with ~N/K=47 assigned items (for 12K items, 256 codes)
- No empty centroids at initialization → EMA updates maintain utilization
- Residual quantization at levels 2 and 3 also starts balanced (each level quantizes a well-distributed residual)

**Why balanced over standard K-means**: Standard K-means tends toward power-law distributions — a few large clusters dominate. For quantization codebooks, uniform utilization is critical because:
- Maximum information capacity: 256 codes × 3 levels = 256³ = 16.7M unique IDs (only achievable if all codes are used)
- The LLM needs to discriminate between codes equally — if 90% of items share one code, the model learns to always predict that code

**Solution 3: Dead Code Revival**

During EMA codebook updates, we detect codes with `cluster_size < 1.0` (effectively unused) and reinitialize them:
```python
dead_mask = self.cluster_size < 1.0
dead_indices = dead_mask.nonzero()[0][:num_replace]
# Replace dead codes with random samples from current batch
self.embedding.weight.data[dead_indices] = z[rand_indices].detach()
```

**Why this works**:
- Ensures full codebook utilization throughout training — no wasted capacity
- Reinitialized codes are placed directly in the data manifold (from batch samples), so they immediately start receiving assignments
- Combined with EMA updates, the revived codes quickly specialize to their local region
- This is a standard technique from VQ-VAE-2 (Razavi et al., 2019) and SoundStream (Zeghidour et al., 2021)

**All three mechanisms work together**:
1. SVD embeddings → items have real cluster structure in input space
2. Balanced K-means → codebook starts with uniform utilization
3. Dead code revival → maintains utilization if codes drift during training

**Expected Results After Fix**:
- Codebook utilization: 200-256 codes used per codebook (was 1-2)
- Collision rate: < 0.30 (was 0.9997)
- Each item gets a nearly unique 3-level code, enabling the LLM to distinguish between items

---

## Project Structure

```
JDGenRec/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── config/
│   ├── base.gin               # Base configuration
│   └── genrec/
│       ├── rqvae.gin          # RQVAE config
│       └── genrec.gin         # GenRec SFT config
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rqvae.py           # RQVAE (borrowed from genrec repo)
│   │   └── genrec.py          # GenRec model with Token Merger
│   ├── data/
│   │   ├── __init__.py
│   │   ├── amazon.py          # Amazon dataset loading & preprocessing
│   │   └── genrec_dataset.py  # Page-wise SFT dataset
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── token_merger.py    # Token Merger linear projection
│   │   ├── metrics.py         # Recall@K, NDCG@K
│   │   └── beam_search.py     # Constrained beam search
│   └── trainers/
│       ├── __init__.py
│       ├── rqvae_trainer.py   # RQVAE training script
│       └── genrec_trainer.py  # GenRec SFT training script
└── scripts/
    ├── train_rqvae.sh         # Train RQVAE
    ├── train_genrec.sh        # Train GenRec
    └── evaluate.sh            # Run evaluation
```

---

## Key Differences from Reference Repo (phonism/genrec)

| Aspect | phonism/genrec (LCRec) | Our Implementation (GenRec) |
|--------|------------------------|----------------------------|
| Training target | Single next item | Page of items (multi-item) |
| Input representation | Raw SID tokens (3 per item) | Merged SID tokens (1 per item via Linear) |
| Decoding | Standard beam search | Same (constrained beam search) |
| RL alignment | Not present | Skipped (GRPO-SR omitted) |
| Architecture | Qwen2.5 + codebook tokens | Same + Token Merger layer |

---

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- transformers (HuggingFace)
- gin-config
- wandb (optional, for logging)
- numpy, scipy

---

## References

- [GenRec Paper](https://arxiv.org/abs/2604.14878) — Zou et al., SIGIR 2026
- [phonism/genrec](https://github.com/phonism/genrec) — Reference implementation repo
- [TIGER](https://arxiv.org/abs/2305.05065) — Rajput et al., 2023
- [LC-Rec](https://arxiv.org/abs/2311.09049) — Zheng et al., 2024
- [Qwen2.5](https://arxiv.org/abs/2412.15115) — Backbone LLM
