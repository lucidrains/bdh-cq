from __future__ import annotations

import torch
from torch import arange, is_tensor, Tensor
from torch.nn import Module, Parameter
from torch.amp import autocast

from einops import einsum, rearrange, repeat

from torch_einops_utils.shape import size

# helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

# applying rotary to qk, partial by default (freqs length determines how many dims rotate)

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0
):
    dtype = t.dtype

    if freqs.ndim == 2:
        seq_len = size(t, '... [n] d')
        freqs = freqs[-seq_len:]

    rot_dim = size(freqs, '... [d]')
    end_index = start_index + rot_dim

    feature_dim = size(t, '... [d]')

    assert rot_dim <= feature_dim, f'feature dimension {feature_dim} is not of sufficient size to rotate in all the positions {rot_dim}'

    # split t into three parts: left, middle (to be rotated), and right

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    t_rotated = (t_middle * freqs.cos()) + (rotate_half(t_middle) * freqs.sin())

    out = torch.cat((t_left, t_rotated, t_right), dim = -1)

    return out.type(dtype)

# rotary embedding module, yields the interleaved freqs per position

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        *,
        theta = 10000.,
        learned_freq = False
    ):
        super().__init__()

        freqs = 1. / (theta ** (arange(0, dim, 2)[:(dim // 2)].float() / dim))

        self.freqs = Parameter(freqs, requires_grad = learned_freq)

    @property
    def device(self):
        return self.freqs.device

    def forward(
        self,
        pos_or_seq_len: Tensor | int,
        offset = 0
    ):
        # get positions depending on input

        if is_tensor(pos_or_seq_len):
            pos = pos_or_seq_len
        else:
            seq_len = pos_or_seq_len
            pos = arange(seq_len, device = self.device, dtype = self.freqs.dtype)

        pos = pos + offset

        # freqs, each frequency repeated for the cos/sin pair

        freqs = einsum(pos.type(self.freqs.dtype), self.freqs, '... i, j -> ... i j')
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return freqs
