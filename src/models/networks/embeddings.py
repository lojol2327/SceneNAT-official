import math

import torch
from torch import nn


def get_1d_sincos_encode(steps: torch.Tensor, emb_dim: int, max_period: int=10000) -> torch.Tensor:
    """Get sinusoidal encodings for a batch of timesteps/positions."""
    assert steps.dim() == 1, f"Parameter `steps` must be a 1D tensor, but got {steps.dim()}D."

    half_dim = emb_dim // 2
    emb = torch.exp(- math.log(max_period) *\
        torch.arange(0, half_dim, device=steps.device).float() / half_dim)
    emb = steps[:, None].float() * emb[None, :]  # (num_steps, half_dim)

    # Concat sine and cosine encodings
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # (num_steps, emb_dim)

    # Zero padding
    if emb_dim % 2 == 1: emb = nn.functional.pad(emb, (0, 1))
    assert emb.shape == (steps.shape[0], emb_dim)

    return emb