"""
GenRec: A Preference-Oriented Generative Framework for Recommendation.
Decoder-only architecture with Token Merger and Page-wise NTP.
Reference: arXiv:2604.14878 (Zou et al., SIGIR 2026)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    DynamicCache,
)

from src.modules.token_merger import TokenMerger


class GenRec(nn.Module):
    """
    GenRec model with Qwen2.5 backbone, Token Merger, and Page-wise NTP support.

    Architecture:
    - Items are represented as 3-level Semantic IDs (SIDs) via special tokens
    - Token Merger compresses 3 SID embeddings per item into 1 on the input side
    - Full-resolution SID tokens are generated autoregressively on the output side
    - Trained with Page-wise NTP (multi-item targets per sample)
    """

    def __init__(
        self,
        pretrained_path: str,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        use_token_merger: bool = True,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.use_token_merger = use_token_merger

        # Load tokenizer and LLM backbone
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_path, trust_remote_code=True
        )
        # Use bf16 on Ampere+ (compute capability >= 8.0), fp16 on T4/V100, float32 on CPU
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_path, trust_remote_code=True, torch_dtype=dtype
        )

        # Add codebook special tokens: <C0_0>, <C0_1>, ..., <C2_255>
        self._add_codebook_tokens()

        # Add structural special tokens
        special_tokens = ["<sep>", "<page>", "<bos_rec>", "<eos_rec>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Store special token IDs
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids("<sep>")
        self.page_token_id = self.tokenizer.convert_tokens_to_ids("<page>")
        self.bos_rec_id = self.tokenizer.convert_tokens_to_ids("<bos_rec>")
        self.eos_rec_id = self.tokenizer.convert_tokens_to_ids("<eos_rec>")

        # Build codebook token ID lookup: codebook_token_ids[c][k] = token_id for <Cc_k>
        self.codebook_token_ids = []
        for c in range(num_codebooks):
            level_ids = []
            for k in range(codebook_size):
                token = f"<C{c}_{k}>"
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                level_ids.append(token_id)
            self.codebook_token_ids.append(level_ids)

        # Token Merger: compresses 3 SID embeddings into 1 for input
        hidden_size = self.model.config.hidden_size
        if use_token_merger:
            self.token_merger = TokenMerger(embed_dim=hidden_size, num_codes=num_codebooks).to(dtype)

        # Build position mask for constrained beam search
        self._build_position_mask()

    def _add_codebook_tokens(self):
        """Add codebook tokens <Cc_k> to the tokenizer."""
        for c in range(self.num_codebooks):
            for k in range(self.codebook_size):
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": [f"<C{c}_{k}>"]}
                )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _build_position_mask(self):
        """Build position masks for constrained beam search decoding."""
        vocab_size = len(self.tokenizer)
        # num_codebooks steps for SID tokens + 1 step for EOS/sep
        num_steps = self.num_codebooks + 1

        position_mask = torch.zeros(num_steps, vocab_size, dtype=torch.bool)

        # Steps 0..num_codebooks-1: only allow tokens from the corresponding codebook level
        for step in range(self.num_codebooks):
            for token_id in self.codebook_token_ids[step]:
                position_mask[step, token_id] = True

        # Final step: allow separator or EOS
        position_mask[self.num_codebooks, self.sep_token_id] = True
        position_mask[self.num_codebooks, self.eos_rec_id] = True
        if self.tokenizer.eos_token_id is not None:
            position_mask[self.num_codebooks, self.tokenizer.eos_token_id] = True

        self.register_buffer("position_mask", position_mask)

        # Build code_map: maps token_id -> code value for each codebook
        code_map = torch.zeros(self.num_codebooks, vocab_size, dtype=torch.long)
        for c in range(self.num_codebooks):
            for k in range(self.codebook_size):
                token_id = self.codebook_token_ids[c][k]
                code_map[c, token_id] = k
        self.register_buffer("code_map", code_map)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing_enable()

    def get_input_embeddings(self):
        """Get the embedding layer of the backbone model."""
        return self.model.get_input_embeddings()

    def codes_to_token_ids(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert semantic ID codes to token IDs.

        Args:
            codes: [..., num_codebooks] integer codes (0-indexed per codebook)
        Returns:
            token_ids: [..., num_codebooks] corresponding token IDs
        """
        shape = codes.shape
        token_ids = torch.zeros_like(codes)
        for c in range(self.num_codebooks):
            # Map code value to token ID for codebook c
            level_ids = torch.tensor(self.codebook_token_ids[c], device=codes.device)
            token_ids[..., c] = level_ids[codes[..., c]]
        return token_ids

    def build_sft_inputs(
        self,
        input_codes: torch.Tensor,
        target_codes: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build model inputs for Page-wise NTP SFT training.

        Constructs the full sequence:
        [bos_rec] [merged_item1] <sep> [merged_item2] <sep> ... <page> [sid1_1 sid1_2 sid1_3] <sep> [sid2_1 ...] <eos_rec>

        The loss is only computed on the target portion (after <page>).

        Args:
            input_codes: [B, max_hist, num_codebooks] input item SID codes
            target_codes: [B, max_page, num_codebooks] target item SID codes
            input_lengths: [B] actual input lengths
            target_lengths: [B] actual target lengths
        Returns:
            input_ids: [B, total_seq_len] token IDs
            attention_mask: [B, total_seq_len]
            labels: [B, total_seq_len] with -100 for non-target positions
        """
        B = input_codes.shape[0]
        device = input_codes.device

        # Convert codes to token IDs
        input_token_ids = self.codes_to_token_ids(input_codes)  # [B, hist, 3]
        target_token_ids = self.codes_to_token_ids(target_codes)  # [B, page, 3]

        embed_layer = self.get_input_embeddings()

        # Build sequences per sample
        all_input_ids = []
        all_labels = []

        for b in range(B):
            hist_len = input_lengths[b].item()
            tgt_len = target_lengths[b].item()

            # --- Prompt portion (input history) ---
            if self.use_token_merger:
                # Get embeddings for history SID tokens and merge them
                hist_token_ids_b = input_token_ids[b, :hist_len]  # [hist_len, 3]
                hist_embeds = embed_layer(hist_token_ids_b)  # [hist_len, 3, hidden]
                # Merge: [hist_len, 3, hidden] -> reshape for token merger
                merged = self.token_merger(hist_embeds.unsqueeze(0))  # [1, hist_len, hidden]
                # For token merger path, we need to handle embeddings differently
                # We'll store a flag and handle in forward pass
                prompt_ids = [self.bos_rec_id]
                for i in range(hist_len):
                    # Each merged item represented by a placeholder
                    # We use the first SID token as placeholder (will be replaced by merged embed)
                    prompt_ids.append(input_token_ids[b, i, 0].item())
                    if i < hist_len - 1:
                        prompt_ids.append(self.sep_token_id)
                prompt_ids.append(self.page_token_id)
            else:
                # Without token merger: use full SID tokens in prompt
                prompt_ids = [self.bos_rec_id]
                for i in range(hist_len):
                    for c in range(self.num_codebooks):
                        prompt_ids.append(input_token_ids[b, i, c].item())
                    if i < hist_len - 1:
                        prompt_ids.append(self.sep_token_id)
                prompt_ids.append(self.page_token_id)

            # --- Target portion (page items as full SID tokens) ---
            target_ids = []
            for i in range(tgt_len):
                for c in range(self.num_codebooks):
                    target_ids.append(target_token_ids[b, i, c].item())
                if i < tgt_len - 1:
                    target_ids.append(self.sep_token_id)
            target_ids.append(self.eos_rec_id)

            # Combine
            full_ids = prompt_ids + target_ids
            # Labels: -100 for prompt, actual IDs for target
            labels = [-100] * len(prompt_ids) + target_ids

            all_input_ids.append(torch.tensor(full_ids, dtype=torch.long, device=device))
            all_labels.append(torch.tensor(labels, dtype=torch.long, device=device))

        # Pad to same length
        max_len = max(ids.size(0) for ids in all_input_ids)
        input_ids = torch.full((B, max_len), self.tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)

        for b in range(B):
            seq_len = all_input_ids[b].size(0)
            input_ids[b, :seq_len] = all_input_ids[b]
            attention_mask[b, :seq_len] = 1
            labels[b, :seq_len] = all_labels[b]

        return input_ids, attention_mask, labels

    def build_eval_inputs(
        self,
        input_codes: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build model inputs for evaluation (point-wise beam search).

        Sequence: [bos_rec] [items with sep] <page>
        Then beam search generates one item's SID tokens.

        Args:
            input_codes: [B, max_hist, num_codebooks]
            input_lengths: [B]
        Returns:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
        """
        B = input_codes.shape[0]
        device = input_codes.device
        input_token_ids = self.codes_to_token_ids(input_codes)

        all_input_ids = []
        for b in range(B):
            hist_len = input_lengths[b].item()
            if self.use_token_merger:
                ids = [self.bos_rec_id]
                for i in range(hist_len):
                    ids.append(input_token_ids[b, i, 0].item())
                    if i < hist_len - 1:
                        ids.append(self.sep_token_id)
                ids.append(self.page_token_id)
            else:
                ids = [self.bos_rec_id]
                for i in range(hist_len):
                    for c in range(self.num_codebooks):
                        ids.append(input_token_ids[b, i, c].item())
                    if i < hist_len - 1:
                        ids.append(self.sep_token_id)
                ids.append(self.page_token_id)
            all_input_ids.append(torch.tensor(ids, dtype=torch.long, device=device))

        max_len = max(ids.size(0) for ids in all_input_ids)
        input_ids = torch.full((B, max_len), self.tokenizer.pad_token_id or 0,
                               dtype=torch.long, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            seq_len = all_input_ids[b].size(0)
            input_ids[b, :seq_len] = all_input_ids[b]
            attention_mask[b, :seq_len] = 1

        return input_ids, attention_mask

    def forward_sft(
        self,
        input_codes: torch.Tensor,
        target_codes: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Page-wise NTP SFT training.

        Returns:
            dict with 'loss' and 'logits'
        """
        input_ids, attention_mask, labels = self.build_sft_inputs(
            input_codes, target_codes, input_lengths, target_lengths
        )

        if self.use_token_merger:
            # For token merger: we need custom embedding injection
            # Get standard embeddings first
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Replace history item positions with merged embeddings
            B = input_codes.shape[0]
            for b in range(B):
                hist_len = input_lengths[b].item()
                input_token_ids_b = self.codes_to_token_ids(
                    input_codes[b, :hist_len].unsqueeze(0)
                )  # [1, hist_len, 3]
                # Get embeddings for all 3 SID tokens per item
                sid_embeds = self.get_input_embeddings()(
                    input_token_ids_b.squeeze(0)
                )  # [hist_len, 3, hidden]
                # Apply token merger
                merged = self.token_merger(sid_embeds.unsqueeze(0)).squeeze(0)  # [hist_len, hidden]

                # Replace positions in inputs_embeds
                # Position mapping: [bos_rec, item0, sep, item1, sep, ..., page_token, ...]
                pos = 1  # start after bos_rec
                for i in range(hist_len):
                    inputs_embeds[b, pos] = merged[i]
                    pos += 1
                    if i < hist_len - 1:
                        pos += 1  # skip sep

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

        return {"loss": outputs.loss, "logits": outputs.logits}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Standard forward pass (for non-merged inference)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate_beam(
        self,
        input_codes: torch.Tensor,
        input_lengths: torch.Tensor,
        beam_width: int = 20,
        topk: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate item predictions using constrained beam search.

        Args:
            input_codes: [B, max_hist, num_codebooks]
            input_lengths: [B]
            beam_width: number of beams
            topk: number of results to return
        Returns:
            sem_ids: [B, topk, num_codebooks] predicted semantic IDs
            scores: [B, topk] log-probability scores
        """
        from src.modules.beam_search import constrained_beam_search

        input_ids, attention_mask = self.build_eval_inputs(input_codes, input_lengths)

        if self.use_token_merger:
            # Inject merged embeddings for beam search
            inputs_embeds = self.get_input_embeddings()(input_ids)
            B = input_codes.shape[0]
            for b in range(B):
                hist_len = input_lengths[b].item()
                input_token_ids_b = self.codes_to_token_ids(
                    input_codes[b, :hist_len].unsqueeze(0)
                )
                sid_embeds = self.get_input_embeddings()(
                    input_token_ids_b.squeeze(0)
                )
                merged = self.token_merger(sid_embeds.unsqueeze(0)).squeeze(0)
                pos = 1
                for i in range(hist_len):
                    inputs_embeds[b, pos] = merged[i]
                    pos += 1
                    if i < hist_len - 1:
                        pos += 1

            # Run prefill with embeddings to get KV cache
            out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

            # Continue with constrained beam search from here
            return self._beam_search_from_logits(
                logits, past_kv, attention_mask, beam_width, topk
            )
        else:
            return constrained_beam_search(
                self.model, input_ids, attention_mask,
                self.position_mask, self.code_map,
                self.num_codebooks, beam_width, topk
            )

    def _beam_search_from_logits(
        self,
        first_logits: torch.Tensor,
        past_kv,
        attention_mask: torch.Tensor,
        beam_width: int,
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constrained beam search starting from pre-computed first logits."""
        B = first_logits.shape[0]
        L = attention_mask.shape[1]
        device = first_logits.device
        num_steps = self.num_codebooks + 1

        # Step 0
        mask_0 = self.position_mask[0].unsqueeze(0)
        first_logits = first_logits.masked_fill(~mask_0, float('-inf'))
        log_probs = F.log_softmax(first_logits, dim=-1)
        topk_scores, topk_ids = log_probs.topk(beam_width, dim=-1)

        beam_scores = topk_scores
        beam_tokens = topk_ids.unsqueeze(-1)

        # Expand for beams
        from src.modules.beam_search import _expand_past_kv, _reorder_past_kv
        past_kv = _expand_past_kv(past_kv, beam_width)
        attn_mask = attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(B * beam_width, L)

        for step in range(1, num_steps):
            next_input = beam_tokens[:, :, -1].reshape(B * beam_width, 1)
            attn_mask = torch.cat([attn_mask, torch.ones(B * beam_width, 1, device=device, dtype=attn_mask.dtype)], dim=-1)

            out = self.model(input_ids=next_input, attention_mask=attn_mask, past_key_values=past_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

            mask_s = self.position_mask[step].unsqueeze(0)
            logits = logits.masked_fill(~mask_s, float('-inf'))
            log_probs = F.log_softmax(logits, dim=-1).view(B, beam_width, -1)

            if step < self.num_codebooks:
                candidate_scores = beam_scores.unsqueeze(-1) + log_probs
                candidate_scores = candidate_scores.view(B, -1)
                top_scores, top_indices = candidate_scores.topk(beam_width, dim=-1)

                beam_idx = top_indices // log_probs.size(-1)
                token_idx = top_indices % log_probs.size(-1)

                beam_scores = top_scores
                prev_tokens = torch.gather(
                    beam_tokens, 1,
                    beam_idx.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
                )
                beam_tokens = torch.cat([prev_tokens, token_idx.unsqueeze(-1)], dim=-1)

                reorder_idx = (torch.arange(B, device=device).unsqueeze(1) * beam_width + beam_idx).view(-1)
                past_kv = _reorder_past_kv(past_kv, reorder_idx)
                attn_mask = attn_mask[reorder_idx]
            else:
                allowed_mask = self.position_mask[step]
                eos_token_idx = allowed_mask.nonzero(as_tuple=False)[0, 0]
                eos_log_prob = log_probs[:, :, eos_token_idx]
                beam_scores = beam_scores + eos_log_prob

        # Extract semantic IDs
        sem_ids = torch.zeros(B, beam_width, self.num_codebooks, dtype=torch.long, device=device)
        for c in range(self.num_codebooks):
            token_ids = beam_tokens[:, :, c]
            sem_ids[:, :, c] = self.code_map[c][token_ids]

        sorted_idx = beam_scores.argsort(dim=-1, descending=True)[:, :topk]
        sem_ids = torch.gather(sem_ids, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, self.num_codebooks))
        scores = torch.gather(beam_scores, 1, sorted_idx)

        return sem_ids, scores

    def save_pretrained(self, save_dir: str, **kwargs):
        """Save model, tokenizer, and token merger weights."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir, **kwargs)
        self.tokenizer.save_pretrained(save_dir)
        if self.use_token_merger:
            torch.save(self.token_merger.state_dict(),
                       os.path.join(save_dir, "token_merger.pt"))

    def load_pretrained(self, load_dir: str):
        """Load model from checkpoint."""
        import os
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            load_dir, torch_dtype=dtype, trust_remote_code=True
        )
        if self.use_token_merger:
            merger_path = os.path.join(load_dir, "token_merger.pt")
            if os.path.exists(merger_path):
                self.token_merger.load_state_dict(torch.load(merger_path, map_location="cpu"))
            self.token_merger = self.token_merger.to(dtype)
