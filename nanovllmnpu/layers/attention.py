from typing import Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllmnpu.utils.context import get_context


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


@triton.jit
def kv_cache_gather_kernel(
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,   # [num_blocks_used]
    k_out_ptr,         # [seqlen, Hk, D]
    v_out_ptr,

    stride_cache_b,
    stride_cache_s,
    stride_cache_h,
    stride_cache_d,

    stride_out_t,
    stride_out_h,
    stride_out_d,

    block_size: tl.constexpr,
    Hk: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)  # token index

    block_idx = pid // block_size
    offset    = pid % block_size

    block_id = tl.load(block_table_ptr + block_idx)

    idx = tl.arange(0, Hk * D)
    h = idx // D
    d = idx % D

    cache_offset = (
        block_id * stride_cache_b
        + offset * stride_cache_s
        + h * stride_cache_h
        + d * stride_cache_d
    )

    k = tl.load(k_cache_ptr + cache_offset)
    v = tl.load(v_cache_ptr + cache_offset)

    out_offset = (
        pid * stride_out_t
        + h * stride_out_h
        + d * stride_out_d
    )

    tl.store(k_out_ptr + out_offset, k)
    tl.store(v_out_ptr + out_offset, v)


def kv_cache_gather(k_cache, v_cache, block_table, seqlen):
    Hk = k_cache.shape[2]
    D  = k_cache.shape[3]

    k_out = torch.empty((seqlen, Hk, D), device=k_cache.device, dtype=k_cache.dtype)
    v_out = torch.empty_like(k_out)

    grid = (seqlen,)

    kv_cache_gather_kernel[grid](
        k_cache, v_cache, block_table,
        k_out, v_out,
        k_cache.stride(0), k_cache.stride(1),
        k_cache.stride(2), k_cache.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        block_size=k_cache.shape[1],
        Hk=Hk,
        D=D,
    )

    return k_out, v_out


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

        kb, vb = kv_cache_gather(
            k_cache,
            v_cache,
            block_table[b],
            seqlen,
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
