import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        use_snapkv = (
            context.is_prefill and context.snapkv_enabled and context.slot_mapping is not None
            and context.slot_mapping.numel() and context.block_tables is None
            and k_cache.numel() and v_cache.numel()
        )
        if use_snapkv:
            keep_mask = self.run_snapkv_selection(q, k, context)
            compact_slots = self.build_compact_slot_mapping(context, keep_mask)
            if compact_slots.numel():
                store_kvcache(k[keep_mask], v[keep_mask], k_cache, v_cache, compact_slots)
        else:
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            kv_k, kv_v = (k_cache, v_cache) if context.block_tables is not None else (k, v)
            o = flash_attn_varlen_func(q, kv_k, kv_v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def run_snapkv_selection(self, q: torch.Tensor, k: torch.Tensor, context) -> torch.Tensor:
        limit = context.snapkv_limit
        if limit is None or limit <= 0:
            return torch.ones(k.shape[0], dtype=torch.bool, device=k.device)
        cu = context.cu_seqlens_q
        device = k.device
        H = self.num_heads
        Hkv = self.num_kv_heads
        scale = self.scale
        mask = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
        # Process each sequence independently to avoid huge attention matrices
        for i in range(cu.numel() - 1):
            start = cu[i].item()
            end = cu[i + 1].item()
            L = end - start
            if L <= limit:
                mask[start:end] = True
                continue
            sample_q = context.snapkv_sample_queries or 128
            sample_q = min(sample_q, L)
            # Take the last `sample_q` queries for importance estimation
            q_seq = q[end - sample_q:end]                  # [sample_q, H, D]
            k_seq = k[start:end]                           # [L, Hkv, D]
            # Expand KV heads to match Q heads if needed (GQA)
            if Hkv != H:
                assert H % Hkv == 0
                rep = H // Hkv
                k_seq = k_seq.repeat_interleave(rep, dim=1)  # [L, H, D]
            # Compute causal attention scores over keys for sampled queries
            # qh: [H, sample_q, D], kh: [H, L, D]
            qh = q_seq.transpose(0, 1)
            kh = k_seq.transpose(0, 1)
            # logits: [H, sample_q, L]
            logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
            # Build causal mask: keys allowed for query at absolute pos p_q
            # Absolute positions of sampled queries: [L - sample_q, ..., L - 1]
            abs_q = torch.arange(L - sample_q, L, device=device)
            abs_k = torch.arange(L, device=device)
            allowed = (abs_k.unsqueeze(0) <= abs_q.unsqueeze(1))  # [sample_q, L]
            logits.masked_fill_(~allowed.unsqueeze(0), float("-inf"))
            attn = torch.softmax(logits, dim=-1)  # softmax over keys
            # Importance per key: sum over heads and sampled queries
            importance = attn.sum(dim=(0, 1))  # [L]
            # Select top-`limit` keys to keep
            topk = torch.topk(importance, k=limit, largest=True)
            keep_idx = topk.indices.sort().values
            seq_mask = mask[start:end]
            seq_mask[keep_idx] = True
        return mask

    def build_compact_slot_mapping(self, context, keep_mask: torch.Tensor) -> torch.Tensor:
        cu = context.cu_seqlens_q
        slot_mapping = context.slot_mapping
        compact_slots = []
        for i in range(cu.numel() - 1):
            start = cu[i].item()
            end = cu[i + 1].item()
            num_kept = int(keep_mask[start:end].sum().item())
            if num_kept == 0:
                continue
            compact_slots.append(slot_mapping[start:start + num_kept])
        if compact_slots:
            return torch.cat(compact_slots)
        return torch.empty(0, device=slot_mapping.device, dtype=slot_mapping.dtype)
