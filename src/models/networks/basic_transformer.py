from typing import *
from torch import Tensor, LongTensor

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    def __init__(self,
        query_dim: int, context_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimenson ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        h = self.n_heads

        q = self.to_q(x)  # (b, n, d*h)
        if mask is not None:
            q = q * mask.unsqueeze(-1)

        # If context is not provided, use self-attention
        if context is None:
            context = x
            context_mask = mask

        k = self.to_k(context)  # (b, m, d*h)
        v = self.to_v(context)  # (b, m, d*h)
        if context_mask is not None:
            k = k * context_mask.unsqueeze(-1)
            v = v * context_mask.unsqueeze(-1)

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))  # (b*h, n or m, d)

        sim: Tensor = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale  # (b*h, n, m)

        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(-1)  # (b, 1, m, 1)
            attn_mask = attn_mask.expand(-1, q.shape[1], -1, h)  # (b, n, m, h)
            attn_mask = rearrange(attn_mask, "b n m h -> (b h) n m").bool()  # (b*h, n, m)
            sim = sim.masked_fill(~attn_mask, float(-1e9))
        attn = sim.softmax(dim=-1)  # (b*h, n, m)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)  # (b*h, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (b, n, d*h)
        out = self.to_out(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*2)

    def forward(self, x: Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self,
        dim: int, dim_out: Optional[int]=None,
        mult=4, gated=False, dropout=0.
    ):
        super().__init__()
        hidden_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU()
        ) if not gated else GEGLU(dim, hidden_dim)

        self.mlp = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x: Tensor):
        return self.mlp(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self,
        dim: int, attn_dim: int,
        context_dim: Optional[int]=None,
        n_heads=8, gated_ff=True, dropout=0.,
        **kwargs
    ):
        super().__init__()

        self.attn = Attention(dim, None, n_heads, attn_dim, dropout)
        self.ff = FeedForward(dim, gated=gated_ff, dropout=dropout)

        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        if context_dim is not None:
            self.cross_attn = Attention(dim, context_dim, n_heads, attn_dim, dropout)
            self.ca_norm = nn.LayerNorm(dim)

    def forward(self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        # 1. Self-attention
        x_norm = self.attn_norm(x)
        x = self.attn(x_norm, None, mask) + x

        # 2. (Optional) Cross-attention
        if context is not None:
            x_norm = self.ca_norm(x)
            x = self.cross_attn(x_norm, context, mask, context_mask) + x

        # 3. Feed-forward MLPs
        x_norm = self.ff_norm(x)
        x = self.ff(x_norm) + x

        return x
