"""Constrained beam search for generating Semantic ID sequences."""
import torch
from torch.nn import functional as F
from typing import Optional, Tuple, List
from transformers import DynamicCache

def constrained_beam_search(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_mask: torch.Tensor,
    code_map: torch.Tensor,
    num_codebooks: int,
    beam_width: int = 20,
    topk: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrained beam search that only generates valid SID token combinations.

    Args:
        model: The causal LM model
        input_ids: [B, L] input token IDs
        attention_mask: [B, L] attention mask
        position_mask: [num_steps, vocab_size] boolean mask of allowed tokens at each step
        code_map: [num_codebooks, vocab_size] maps token IDs to code values
        num_codebooks: number of codebook levels (e.g., 3)
        beam_width: number of beams to maintain
        topk: number of top results to return

    Returns:
        sem_ids: [B, topk, num_codebooks] predicted semantic IDs
        scores: [B, topk] log-probability scores
    """
    B, L = input_ids.shape
    device = input_ids.device
    num_steps = num_codebooks + 1  # codebook tokens + EOS

    position_mask = position_mask.to(device)
    code_map = code_map.to(device)

    # Prefill: get KV cache + first logits
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = out.logits[:, -1, :]  # [B, V]
    past_kv = out.past_key_values

    # Apply step-0 constraint
    mask_0 = position_mask[0].unsqueeze(0)
    logits = logits.masked_fill(~mask_0, float('-inf'))
    log_probs = F.log_softmax(logits, dim=-1)

    # Initial beam selection
    topk_scores, topk_ids = log_probs.topk(beam_width, dim=-1)
    beam_scores = topk_scores  # [B, beam_width]
    beam_tokens = topk_ids.unsqueeze(-1)  # [B, beam_width, 1]

    # Expand KV cache for beams
    past_kv = _expand_past_kv(past_kv, beam_width)
    attn_mask = attention_mask.unsqueeze(1).expand(-1, beam_width, -1).reshape(B * beam_width, L)

    # Autoregressive decoding steps
    for step in range(1, num_steps):
        next_input = beam_tokens[:, :, -1].reshape(B * beam_width, 1)
        attn_mask = torch.cat([attn_mask, torch.ones(B * beam_width, 1, device=device, dtype=attn_mask.dtype)], dim=-1)

        out = model(input_ids=next_input, attention_mask=attn_mask, past_key_values=past_kv, use_cache=True)
        logits = out.logits[:, -1, :]
        past_kv = out.past_key_values

        # Apply step constraint
        mask_s = position_mask[step].unsqueeze(0)
        logits = logits.masked_fill(~mask_s, float('-inf'))
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(B, beam_width, -1)

        if step < num_codebooks:
            # Expand beams
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

            # Reorder KV cache
            reorder_idx = (torch.arange(B, device=device).unsqueeze(1) * beam_width + beam_idx).view(-1)
            past_kv = _reorder_past_kv(past_kv, reorder_idx)
            attn_mask = attn_mask[reorder_idx]
        else:
            # Final EOS step
            allowed_mask = position_mask[step]
            eos_token_idx = allowed_mask.nonzero(as_tuple=False)[0, 0]
            eos_log_prob = log_probs[:, :, eos_token_idx]
            beam_scores = beam_scores + eos_log_prob

    # Extract semantic IDs from beam tokens
    sem_ids = torch.zeros(B, beam_width, num_codebooks, dtype=torch.long, device=device)
    for c in range(num_codebooks):
        token_ids = beam_tokens[:, :, c]
        sem_ids[:, :, c] = code_map[c][token_ids]

    # Sort by score and take topk
    sorted_idx = beam_scores.argsort(dim=-1, descending=True)
    sorted_idx_k = sorted_idx[:, :topk]
    sem_ids = torch.gather(sem_ids, 1, sorted_idx_k.unsqueeze(-1).expand(-1, -1, num_codebooks))
    scores = torch.gather(beam_scores, 1, sorted_idx_k)

    return sem_ids, scores


def _expand_past_kv(past_kv, beam_width):
    """Expand KV cache from [B, ...] to [B*beam_width, ...]."""
    if isinstance(past_kv, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx in range(len(past_kv)):
            key = past_kv.key_cache[layer_idx]
            value = past_kv.value_cache[layer_idx]
            new_key = key.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(-1, *key.shape[1:])
            new_value = value.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(-1, *value.shape[1:])
            new_cache.update(new_key, new_value, layer_idx)
        return new_cache
    expanded = []
    for layer_kv in past_kv:
        expanded.append(tuple(
            t.unsqueeze(1).expand(-1, beam_width, -1, -1, -1)
             .reshape(t.size(0) * beam_width, *t.shape[1:])
            for t in layer_kv
        ))
    return tuple(expanded)


def _reorder_past_kv(past_kv, reorder_idx):
    """Reorder KV cache according to beam selection indices."""
    if isinstance(past_kv, DynamicCache):
        past_kv.reorder_cache(reorder_idx)
        return past_kv
    return tuple(
        tuple(t.index_select(0, reorder_idx) for t in layer_kv)
        for layer_kv in past_kv
    )
