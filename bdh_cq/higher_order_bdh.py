from __future__ import annotations

import torch
from torch import einsum
from torch.nn import Module, ModuleList, Parameter, ReLU

from einops.layers.torch import Rearrange

from bdh_cq.bdh_cq import (
    LinearNoBias,
    LayerNormNoParams,
    default,
    exists
)
from bdh_cq.rotary import apply_rotary_emb

# order-2 poly attention (https://arxiv.org/abs/2602.02422), adapted for linear attention

class HigherOrderBDHLayer(Module):
    """Two-hop poly attention over each root's visible prefix.

    Forward returns a complete (S3, z3, keys, messages, masses) cache whose
    key/message histories grow with context. Independently computed caches
    cannot be merged. Legacy two-tensor caches must be recreated by re-ingesting
    the context; parameter shapes are unchanged.
    """

    def __init__(
        self,
        dim,
        *,
        heads,
        dim_queries_keys,
        qk_activation = ReLU(),
        ff_activation = ReLU(),
        include_pass1_mass = True,
        omit_self = True,
        eps = 1e-6
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads

        self.to_qks = ModuleList([LinearNoBias(dim, dim_inner_qk) for _ in range(3)])

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.qk_activation = qk_activation

        self.include_pass1_mass = include_pass1_mass
        self.omit_self = omit_self
        self.eps = eps

        self.post_attn_norm = LayerNormNoParams(dim)
        self.post_ff_norm = LayerNormNoParams(dim)

        # the feedforward part, gates from the root (pass-2 query) stream

        self.proj_up = Parameter(torch.randn(heads, dim, dim_queries_keys) * 0.02)

        self.ff_act = ff_activation

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.proj_out = LinearNoBias(dim_queries_keys * heads, dim)

    def forward(
        self,
        tokens,
        memories = None,
        rotary_emb = None,
        return_memories = False
    ):
        # the three sparse qk streams, relu activated - the exact same features as BDH

        eps = self.eps

        q1, q2, q3 = map(
            lambda to_qk: self.split_heads(self.qk_activation(to_qk(tokens))),
            self.to_qks
        )

        # the root stream also gates the ff

        gates = q1

        # relative positions - applied to the queries and keys only, never the values

        if exists(rotary_emb):
            q1, q2, q3 = (apply_rotary_emb(rotary_emb, t) for t in (q1, q2, q3))

        # the values are the tokens, no projection

        v3 = tokens

        batch, heads, seq_len, dim_qk = q2.shape
        device = tokens.device

        if exists(memories) and len(memories) != 5:
            raise ValueError('legacy HigherOrder memory is incomplete; re-ingest the context to recreate it')

        if exists(memories):
            S3, z3, past_keys, past_messages, past_masses = memories
        else:
            S3 = q3.new_zeros(batch, heads, dim_qk, tokens.shape[-1])
            z3 = q3.new_zeros(batch, heads, dim_qk)

        # Seed new intermediates from past leaves; old messages already cover that prefix.

        keys = q2
        messages = einsum('b h n d, b h d e -> b h n e', q2, S3)
        masses = einsum('b h n d, b h d -> b h n', q2, z3)

        if exists(memories):
            keys = torch.cat((past_keys, keys), dim = -2)
            messages = torch.cat((past_messages, messages), dim = -2)
            masses = torch.cat((past_masses, masses), dim = -1)

        past_len = keys.shape[-2] - seq_len
        root_positions = torch.arange(seq_len, device = device) + past_len
        key_positions = torch.arange(keys.shape[-2], device = device)

        pass1 = einsum('b h j d, b h k d -> b h j k', keys, q3)
        if self.omit_self:
            pass1 = pass1.masked_fill(key_positions[:, None] == root_positions[None, :], 0.)

        pass2 = einsum('b h i d, b h j d -> b h i j', q1, keys)
        intermediate_mask = key_positions[None, :] <= root_positions[:, None] - int(self.omit_self)
        pass2 = pass2.masked_fill(~intermediate_mask, 0.)

        # Each intermediate may read any leaf in the root's prefix, including the root itself.
        # ponytail: cubic chunk paths / serial normalized scans need a fused prefix kernel for large-chunk training.

        if self.include_pass1_mass or seq_len == 0:
            leaf_paths = einsum('b h i j, b h j k -> b h i k', pass2, pass1)
            leaf_mask = torch.ones(seq_len, seq_len, device = device, dtype = torch.bool).tril()
            leaf_paths = leaf_paths.masked_fill(~leaf_mask, 0.)

            num = einsum('b h i j, b h j e -> b h i e', pass2, messages)
            num = num + einsum('b h i k, b k e -> b h i e', leaf_paths, v3)
            den = einsum('b h i j, b h j -> b h i', pass2, masses) + leaf_paths.sum(dim = -1)

            if return_memories:
                messages = messages + einsum('b h j k, b k e -> b h j e', pass1, v3)
                masses = masses + pass1.sum(dim = -1)
        else:
            # Per-prefix normalization is nonlinear; update raw messages without self subtraction.

            numerators, denominators = [], []
            for index in range(seq_len):
                scores = pass1[..., index]
                messages = messages + scores[..., None] * v3[:, None, index:index + 1]
                masses = masses + scores
                normalizers = masses.clamp_min(eps)
                weights = pass2[:, :, index]

                numerators.append(einsum('b h j, b h j e -> b h e', weights, messages / normalizers[..., None]))
                denominators.append(einsum('b h j, b h j -> b h', weights, normalizers))

            num = torch.stack(numerators, dim = -2)
            den = torch.stack(denominators, dim = -1)

        out = num / (den[..., None] + eps)

        # fully masked query rows (all keys masked) should output zero

        out = torch.where(den[..., None] != 0., out, torch.zeros_like(out))

        # post attn norm

        attn_out = self.post_attn_norm(out)

        # the interesting ff glu variant - the root sparse input gates the projection

        projected = einsum('b h n d, h d e -> b h n e', attn_out, self.proj_up)

        projected = self.ff_act(projected * gates)

        out = self.merge_heads(projected)

        out = self.proj_out(out)

        out = self.post_ff_norm(out)

        # Return complete state: existing intermediates now also contain this chunk's leaves.

        if not return_memories:
            return out

        S3 = S3 + einsum('b h n d, b n e -> b h d e', q3, v3)
        z3 = z3 + q3.sum(dim = -2)
        memories = (S3, z3, keys, messages, masses)

        return out, memories

    # Forward already incorporated the previous cache; frozen writes skip this replacement.

    @staticmethod
    def combine_memories(new_memory, prev_memory):
        if len(new_memory) != 5 or (exists(prev_memory) and len(prev_memory) != 5):
            raise ValueError('legacy HigherOrder memory is incomplete; re-ingest the context to recreate it')

        return new_memory
