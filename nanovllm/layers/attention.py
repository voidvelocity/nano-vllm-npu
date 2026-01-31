from typing import Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.context import get_context


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    """
    The k_cache and v_cache is global cache that store all cached k,v, instead of current `key` and `value`.

    key:          (N, num_kv_heads, head_dim)
    value:        (N, num_kv_heads, head_dim)
    k_cache:      (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache:      (num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping: (N,)
    """
    assert key.ndim == 3
    assert value.ndim == 3
    assert k_cache.ndim == 4
    assert v_cache.ndim == 4
    assert slot_mapping.ndim == 1

    N, num_kv_heads, head_dim = key.shape
    block_size = k_cache.shape[1]

    for i in range(N):
        slot = int(slot_mapping[i])
        if slot < 0:
            continue

        block_id = slot // block_size
        offset = slot % block_size

        k_cache[block_id, offset].copy_(key[i])
        v_cache[block_id, offset].copy_(value[i])


def _scaled_dot_product_attention(q, k, v, scale, causal):
    """
    q: (Tq, Hq=16, D)
    k: (Tk, Hk=8, D)
    v: (Tk, Hv=8, D)
    """
    # print(f"---- in _scaled_dot_product_attention: {q.shape=} {k.shape=} {v.shape=}")
    # q.shape=torch.Size([4096, 16, 128]) k.shape=torch.Size([4096, 8, 128]) v.shape=torch.Size([4096, 8, 128])

    Tq, Hq, D = q.shape
    Tk, Hk, _ = k.shape
    Tv, Hv, _ = v.shape
    H = Hk                  # base is Hk
    # ---- GQA / MQA handling ----
    assert Hq % H == 0, f"Hq={Hq} must be divisible by Hk={Hk}"
    assert H == Hv

    group_size = Hq // H

    q = q.reshape(Tq, H, group_size, D)        # [Tq, H, g, D]
    k = k.reshape(Tk, H, 1, D)                 # [Tk, H, 1, D]
    v = v.reshape(Tk, H, 1, D)                 # [Tv, H, 1, D]

    # scores: [Tq, H, g, D] @ [Tk, H, 1(broadcast to group_size), D] -> [H, g, Tq, Tk]
    scores = torch.einsum("qhgd,khgd->hgqk", q, k) * scale

    if causal:
        # Offset needed when Tq != Tk (decode case)
        offset = Tk - Tq
        mask = torch.triu(
            torch.ones(Tq, Tk, device=q.device, dtype=torch.bool),
            diagonal=1 + offset,
        )
        """mask
        [[False,  True,  True,  ...,  True,  True,  True],
        [False, False,  True,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        ...,
        [False, False, False,  ..., False,  True,  True],
        [False, False, False,  ..., False, False,  True],
        [False, False, False,  ..., False, False, False]]
        """
        scores.masked_fill_(mask[None, None, :, :], float("-inf"))
        # print(f"{mask=}")

    # attn: [H, g, Tq, Tk]
    attn = torch.softmax(scores, dim=-1)

    # out: [H, g, Tq, Tk] @ [Tv(==Tk), H, 1(broadcast to group_size), D] -> [Tq, H, g, D]
    out = torch.einsum("hgqk,khgd->qhgd", attn, v)

    # [Tq, H, g, D] -> [Tq, Hq(=H*g), D]
    out = out.reshape(Tq, Hq, D)
    return out


def _gather_kv_from_cache(k_cache, v_cache, block_table, seqlen, num_kv_heads, head_dim):
    """
    ⚠️ Slow but correct reference implementation.

    k_cache, v_cache:
        (num_blocks, block_size, num_kv_heads, head_dim)

    block_table:
        (num_blocks_used,)  # token blocks in order

    Returns:
        k: (seqlen, num_kv_heads, head_dim)
        v: (seqlen, num_kv_heads, head_dim)
    """
    assert k_cache.ndim == 4
    assert block_table is not None

    block_size = k_cache.shape[1]
    device = k_cache.device

    # print(f"=== {num_kv_heads=}  {k_cache.shape=}  {v_cache.shape=}")
    k_out = torch.empty(
        (seqlen, num_kv_heads, head_dim),
        device=device,
        dtype=k_cache.dtype,
    )
    v_out = torch.empty_like(k_out)

    token_idx = 0
    for block_id in block_table.tolist():
        for offset in range(block_size):
            if token_idx >= seqlen:
                return k_out, v_out

            k_out[token_idx].copy_(k_cache[block_id, offset])
            v_out[token_idx].copy_(v_cache[block_id, offset])
            token_idx += 1

    return k_out, v_out


def attention_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """
    This function is to keep interface same with `from flash_attn import flash_attn_varlen_func`.
    But in our code, some parameter we don't care and just ignore them, in details:
    ❌ ignore dropout
    ❌ ignore alibi
    ❌ ignore window_size
    ❌ ignore softcap
    ❌ ignore rotary
    ❌ ignore return_attn_probs / return_softmax_lse

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    Return:
        out: (total, nheads, headdim).
    """
    assert not return_attn_probs
    assert softcap == 0.0

    total_q, nheads, headdim = q.shape
    scale = softmax_scale or (1.0 / headdim ** 0.5)

    batch_size = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)

    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        k_start, k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        qb = q[q_start:q_end]   # [Lq, Hq, D]
        kb = k[k_start:k_end]   # [Lk, Hk, D]
        vb = v[k_start:k_end]   # [Lv, Hv, D]

        # print(f"---- in attention_varlen_func {q.shape=} {qb.shape=}")

        out[q_start:q_end] = _scaled_dot_product_attention(
            qb, kb, vb, scale, causal
        )

    return out


def attention_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """
    This function is to keep interface same with `from flash_attn import flash_attn_varlen_func`.
    But in our code, some parameter we don't care and just ignore them, in details:
    ❌ ignore dropout
    ❌ ignore alibi
    ❌ ignore window_size
    ❌ ignore softcap
    ❌ ignore rotary
    ❌ ignore return_attn_probs / return_softmax_lse
    ❌ ignore k and v

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    assert k is None and v is None
    assert not return_softmax_lse
    assert softcap == 0.0

    B, T, Hq, D = q.shape   # `T` should be 1
    Hk = k_cache.size(2)
    scale = softmax_scale or (1.0 / D ** 0.5)

    out = torch.empty_like(q)

    for b in range(B):
        seqlen = int(cache_seqlens[b])
        qb = q[b]  # (T=1, H, D)

        # Gather KV
        kb, vb = _gather_kv_from_cache(
            k_cache, v_cache, block_table[b], seqlen, Hk, D
        )

        # (T=1, H, D)
        out_b = _scaled_dot_product_attention(
            qb, kb, vb, scale, causal
        )
        out[b, 0] = out_b[0]

    return out


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
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = attention_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = attention_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
