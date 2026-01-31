from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # rotary_dim == head_size == head_dim
    # x.shape:   [batch_size*seq_len, num_(kv)_heads, head_dim]
    # x1.shape:  [batch_size*seq_len, num_(kv)_heads, head_dim/2]
    # x2.shape:  [batch_size*seq_len, num_(kv)_heads, head_dim/2]
    # cos.shape: [batch_size*seq_len, 1, head_dim/2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # return shape: [batch_size*seq_len, 1, head_dim]
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs shape=[max_position_embeddings, rotary_dim//2]  rotary_dim == head_size == head_dim
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # cache shape=[max_position_embeddings, 1, rotary_dim]  rotary_dim == head_size == head_dim
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # self.cos_sin_cahce: [max_position_embeddings, 1, head_dim]
        # cos_sin:            [batch_size*seq_len, 1, head_dim]
        # cos:                [batch_size*seq_len, 1, head_dim/2]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # query.shape:        [batch_size*seq_len, num_heads, head_dim]
        # key.shape:          [batch_size*seq_len, num_kv_heads, head_dim]
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
