from typing import Optional, cast
from torch import Tensor, LongTensor

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from .basic_transformer import *
from .triplet_decoder import TripletDecoder
from .triplet_predictor import TripletPredictor

class SelfAttention(nn.Module):
    def __init__(self,
        dim: int, attn_dim: int, context_dim: Optional[int]=None,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        if context_dim is None:
            context_dim = attn_dim

        self.scale = head_dim ** -0.5
        self.n_heads = n_heads

        self.to_q = nn.Linear(dim, attn_dim, bias=False)
        self.to_k = nn.Linear(dim, attn_dim, bias=False)
        self.to_v = nn.Linear(dim, attn_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.Dropout(dropout)
        )

    def forward(self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None,
        context_mask: Optional[LongTensor]=None,
        truncate: bool=False,
    ):

        if len(x.shape) == 4:
            x = rearrange(x, 'b n a d -> (b n) a d')
            if mask is not None:
                mask = cast(LongTensor, rearrange(mask, 'b n -> (b n)').unsqueeze(-1).to(torch.long))

        h = self.n_heads

        q = self.to_q(x)  # (b, n, d)
        if mask is not None:
            if truncate:
                q[:, 1:] = q[:, 1:] * mask.unsqueeze(-1)
            else:
                q = q * mask.unsqueeze(-1)

        # If context is not provided, use self-attention
        if context is None:
            context = x
            context_mask = mask

        k = self.to_k(context)  # (b, m, d)
        v = self.to_v(context)  # (b, m, d)
        if context_mask is not None:
            if truncate:
                k[:, 1:] = k[:, 1:] * context_mask.unsqueeze(-1)
                v[:, 1:] = v[:, 1:] * context_mask.unsqueeze(-1)
            else:
                k = k * context_mask.unsqueeze(-1)
                v = v * context_mask.unsqueeze(-1)

        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> (b h) n d", h=h), (q, k, v))  # (b*h, n or m, d)

        sim: Tensor = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale  # (b*h, n, m)

        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, 21, 1)
            attn_mask = attn_mask.expand(-1, context_mask.shape[1], -1, h)  # (B, 21, 21, h)
            attn_mask = rearrange(attn_mask, "b n m h -> (b h) n m").bool()  # (B*h, 21, 21)
            if truncate:
                full_attn_mask = torch.ones((sim.shape[0], sim.shape[1], sim.shape[2]), dtype=torch.bool, device=sim.device)
                full_attn_mask[:, 1:, 1:] = attn_mask
                sim = sim.masked_fill(~full_attn_mask, -1e9)
            else:
                sim = sim.masked_fill(~attn_mask, -1e9)

        attn = sim.softmax(dim=-1)  # (b*h, n, m)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)  # (b*h, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (b, n, d*h)
        out = self.to_out(out)
        if mask is not None:
            if truncate:
                out[:, 1:] = out[:, 1:] * mask.unsqueeze(-1)
            else:   
                out = out * mask.unsqueeze(-1)

        return out

class CrossAttention(nn.Module):
    def __init__(self,
        query_dim: int, context_dim: int,
        n_heads=8, hidden_dim=512, dropout=0.
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({n_heads})"
        head_dim = hidden_dim // n_heads

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
        context: Tensor,
        mask: Optional[LongTensor]=None, context_mask: Optional[LongTensor]=None
    ):
        h = self.n_heads

        q = self.to_q(x)  # (b, n, d*h)
        if mask is not None:
            if q.shape[1] == mask.shape[1] + 1: # TODO: temporary workaround.
                q[:, 1:] = q[:, 1:] * mask.unsqueeze(-1)
            else:
                q = q * mask.unsqueeze(-1)

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
            sim = sim.masked_fill(~attn_mask, -1e9)
        attn = sim.softmax(dim=-1)  # (b*h, n, m)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)  # (b*h, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)  # (b, n, d*h)
        out = self.to_out(out)
        if mask is not None:
            if out.shape[1] == mask.shape[1] + 1: # TODO: same as above.
                out[:, 1:] = out[:, 1:] * mask.unsqueeze(-1)
            else:
                out = out * mask.unsqueeze(-1)

        return out

class DecoderBlock(nn.Module):
    def __init__(self,
        dim: int,
        context_dim: Optional[int]=None,
        n_sa=8,
        dropout=0.,
    ):
        super().__init__()
        self.dim = dim
        self.n_sa = n_sa
        self.dropout = dropout

        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(
            dim=dim,
            attn_dim=dim,
            n_heads=n_sa,
            hidden_dim=dim,
            dropout=dropout
        )
        
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim if context_dim is not None else dim,
            n_heads=n_sa,
            hidden_dim=dim,
            dropout=dropout
        )
        
        self.ff_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            mult=4,
            dropout=dropout
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None,
        context_mask: Optional[LongTensor]=None,
    ):
        x_norm = self.self_attn_norm(x)
        self_attn_out = self.self_attn(
            x=x_norm,
            context=None,
            mask=mask
        )
        x = x + self_attn_out
        
        if context is not None:
            x_norm = self.cross_attn_norm(x)
            cross_attn_out = self.cross_attn(
                x=x_norm,
                context=context,
                mask=mask,
                context_mask=context_mask
            )
            x = x + cross_attn_out
        
        x = x + self.ffn(self.ff_norm(x))
        
        return x


class MTransformer(nn.Module):
    def __init__(self,
        dim: int,
        context_dim: Optional[int]=None,
        n_heads=8,
        dropout=0.,
        max_obj_num: int=21,
        scene_dec_layers: int=3, 
        triplet_decoder_layers: int=2,
        max_num_rel: int=4,
        n_predicate_types: int=10,
        obj_class_num: int=24,
        triplet_context: Optional[str]=None,
        **kwargs
    ):
        super().__init__()

        # Position embeddings for encoder and decoder
        self.dec_embed = nn.Parameter(torch.randn(max_obj_num, dim))
        nn.init.normal_(self.dec_embed, mean=0.0, std=0.02)

        self.downproj = nn.Sequential(
            Rearrange('b n t d -> b n (t d)'),
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.scene_decoder = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                context_dim=context_dim,
                n_sa=n_heads,
                dropout=dropout,
            ) for _ in range(scene_dec_layers)
        ])

        self.triplet_context = triplet_context
        if triplet_context == "text":
            triplet_context_dim = context_dim
        else:
            triplet_context_dim = dim
        # Use Triplet Decoder
        self.triplet_decoder = TripletDecoder(
            dim=dim,
            attn_dim=dim,
            context_dim=triplet_context_dim,
            max_num_rel=max_num_rel,
            n_heads=n_heads,
            n_layers=triplet_decoder_layers,
            dropout=dropout
        )
        
        self.triplet_predictor = TripletPredictor(
            dim=dim,
            n_class_types=obj_class_num,
            n_predicate_types=n_predicate_types
        )
        
        # Cross-attention for translation prediction using triplet features
        self.triplet_cross_attn = BasicTransformerBlock(
            dim=dim,
            attn_dim=dim,
            context_dim=dim,
            n_heads=n_heads,
            dropout=dropout
        )

        self.proj_xo = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

        self.proj_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 3),
            nn.SiLU(),
        )
        
        self.proj_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 3),
            nn.SiLU(),
        )
        
        self.proj_r = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        
        
    def forward(
        self,
        x: Tensor,
        text_emb: Optional[Tensor]=None,
        pad_mask: Optional[LongTensor]=None,
    ):
        x_scene = self.downproj(x)

        x_dec = x_scene + self.dec_embed.unsqueeze(0)
        
        # Scene decoder
        for decoder_block in self.scene_decoder:
            x_dec = decoder_block(
                x=x_dec,
                context=text_emb,
                mask=pad_mask,
            )

        xo_out = self.proj_xo(x_dec)                  # (B,N,D)
        s_out = self.proj_s(x_dec)
        r_out = self.proj_r(x_dec)

        # Use Triplet Decoder
        if self.triplet_context == "text":
            triplet_features = self.triplet_decoder(text_emb)
        else:
            triplet_features = self.triplet_decoder(x_dec, pad_mask)
        triplet_logits = self.triplet_predictor(triplet_features)
        
        # Use cross-attention between triplet_features and x_dec for translation prediction
        # query: x_dec, key/value: triplet_features
        t_out = self.triplet_cross_attn(
            x=x_dec,           # query: (B, N, D)
            context=triplet_features,  # key/value: (B, max_num_rel, D)
            mask=pad_mask,     # mask for x_dec
            context_mask=None  # all slots in triplet_features are valid
        )

        t_out = self.proj_t(t_out)

        return xo_out, t_out, s_out, r_out, triplet_logits
