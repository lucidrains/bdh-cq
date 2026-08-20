from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, einsum, cat, stack, is_tensor, Tensor, zeros
from torch.nn import Module, Embedding, Linear, LayerNorm, RMSNorm, Parameter
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# constants

Memory = namedtuple('Memory', ('tokens_seen', 'embeds', 'fast_weight_memories'))

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(v):
    return v[0]

def last(v):
    return v[-1]

def divisible_by(n, d):
    return (n % d) == 0

def pop_if_len_one(returns):
    return returns[0] if len(returns) == 1 else returns

def LinearNoBias(dim, dim_out):
    return Linear(dim, dim_out, bias = False)

def LayerNormNoParams(dim):
    return LayerNorm(dim, elementwise_affine = False)

def top_k(logits, thres = 0.9):
    num_keep = max(1, int((1 - thres) * logits.shape[-1]))
    kth_largest = logits.topk(num_keep, dim = -1).values[..., -1:]
    return logits.masked_fill(logits < kth_largest, -float('inf'))

# attention residual depth bias
# each latent key is biased on its sim by its distance from the end of reasoning

def compute_attn_residual_depth_bias(
    num_keys,
    *,
    bias_schedule,
    depth,
    total_reasoning_iterations
):
    total_latents = depth * total_reasoning_iterations
    device = bias_schedule.device

    # no reasoning cycles, nothing to be away from the end of

    if total_latents == 0:
        return zeros(num_keys, device = device)

    # biases are indexed by distance from the end of reasoning, one per cycle, repeated across the depths,
    # curtailed to the cycles that will exist, earlier cycles zero padded on the left, and the tail excised
    # by the depth still to go before the end

    if bias_schedule.numel() > total_reasoning_iterations:
        bias_schedule = bias_schedule[-total_reasoning_iterations:]

    schedule = repeat(bias_schedule, 'd -> (d depth)', depth = depth)

    if schedule.numel() < total_latents:
        schedule = cat((zeros(total_latents - schedule.numel(), device = device), schedule))

    num_latents = num_keys - 1

    if num_latents > total_latents:
        # latents written after reasoning concludes are the most final, keep the maximum bias

        schedule = cat((schedule, repeat(schedule[-1:], '1 -> n', n = num_latents - total_latents)))
    else:
        schedule = schedule[:num_latents]

    # the seed latents precede reasoning, no bias

    return cat((zeros(1, device = device), schedule))

# residual

class AttentionResidual(Module):
    # attention over the depth axis with a learnable pseudo-query, replacing the identity residual - Kimi team https://arxiv.org/abs/2603.15031

    def __init__(
        self,
        dim,
        *,
        depth = 1,
        depth_bias_distance = 0
    ):
        super().__init__()
        self.query = Parameter(torch.randn(depth, dim) * 0.02)
        self.key_rmsnorm = RMSNorm(dim)

        # bias on the sim by distance from the end of reasoning, off at 0

        self.has_depth_bias_distance = depth_bias_distance > 0
        if self.has_depth_bias_distance:
            self.depth_bias = Parameter(torch.randn(depth_bias_distance) * 0.02)

    def forward(
        self,
        query,
        keys_values: list[Tensor],
        layer_index = 0,
        depth = 1,
        total_reasoning_iterations = 1
    ):
        keys_values = list(keys_values)

        assert all(hidden.shape == query.shape for hidden in keys_values), f'layer hiddens must all match the query shape ({query.shape}), got {[h.shape for h in keys_values]}'

        past_layers = rearrange(keys_values, 'l b n d -> b n l d')

        queries, keys = self.query[layer_index], self.key_rmsnorm(past_layers)

        sim = einsum('d, b n l d -> b n l', queries, keys)

        # latent aware of its distance from the end of reasoning

        if self.has_depth_bias_distance:
            depth_bias = compute_attn_residual_depth_bias(
                len(keys_values),
                bias_schedule = self.depth_bias,
                depth = depth,
                total_reasoning_iterations = total_reasoning_iterations
            )

            sim = sim + depth_bias

        attn = sim.softmax(dim = -1)

        return einsum('b n l, b n l d -> b n d', attn, past_layers)

# classes

class BDHBlock(Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        dim_queries_keys,
        qk_activation = nn.ReLU(),
        ff_activation = nn.ReLU(),
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads

        self.to_qk = LinearNoBias(dim, dim_inner_qk)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.qk_activation = qk_activation

        self.post_attn_norm = LayerNormNoParams(dim)

        self.post_ff_norm = LayerNormNoParams(dim)

        # the feedforward part

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
        device = tokens.device

        # queries and keys, relu activated

        sparse_input = self.qk_activation(self.to_qk(tokens))

        # split heads

        q = k = ff_gates = self.split_heads(sparse_input)

        # the values are the tokens

        v = tokens

        # relative positions

        if exists(rotary_emb):
            q, k = (apply_rotary_emb(rotary_emb, t) for t in (q, k))

        # linear attention, omitting attention to self

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).tril(-1) # omit self, seen in Reformer shared qk attention years ago

        attn = sim.masked_fill(~causal_mask, 0.)

        # they directly aggregate on the tokens as the values, no projection

        agg = einsum('b h i j, b j d -> b h i d', attn, v)

        # past memories

        if exists(memories):
            retrieved = einsum('b h n d, b h d e -> b h n e', q, memories)
            agg = agg + retrieved

        # post attn norm

        attn_out = self.post_attn_norm(agg)

        # the interesting ff glu variant

        projected = einsum('b h n d, h d e -> b h n e', attn_out, self.proj_up)

        # they use the projected sparse input itself (q, k) as the gates

        projected = self.ff_act(projected * ff_gates)

        out = self.merge_heads(projected)

        out = self.proj_out(out)

        out = self.post_ff_norm(out)

        # maybe return memories

        if not return_memories:
            return out

        memories = einsum('b h n d, b n e -> b h d e', k, v)

        return out, memories

# main bdh

class BDH(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth = 8,
        heads = 4,
        dim_qk_heads = 32_768, # their neurons is the dim_qk * heads
        attn_residual = False,
        attn_residual_tied = True,
        attn_residual_depth_bias_distance = 0,
    ):
        super().__init__()
        assert divisible_by(dim_qk_heads, heads)
        dim_qk = dim_qk_heads // heads

        self.dim = dim
        self.token_embed = Embedding(num_tokens, dim)

        self.rope = RotaryEmbedding(dim_qk // 2)
        self.depth = depth

        self.post_embed_norm = LayerNormNoParams(dim)

        self.block = BDHBlock(
            dim,
            heads = heads,
            dim_queries_keys = dim_qk
        )

        self.post_norm = LayerNormNoParams(dim)

        self.to_logits = LinearNoBias(dim, num_tokens)

        # attention residual, defined once - a single pseudo-query attends over all depth x step latent hiddens,
        # or a distinct pseudo-query per depth when untied

        self.attn_residual_tied = attn_residual_tied

        self.attn_residual = AttentionResidual(dim, depth = depth if attn_residual and not attn_residual_tied else 1, depth_bias_distance = attn_residual_depth_bias_distance) if attn_residual else None

    def forward(
        self,
        tokens_or_ids,
        memories = None,
        return_memory = False,
        return_logits = True,
        update_memory = True,
        return_per_pass_hiddens = False,
        all_block_outputs = None,
        total_reasoning_iterations = 1
    ):
        device = tokens_or_ids.device

        # attention residual wired in through the flag on init

        attention_residual = self.attn_residual

        # the input can be tokens, from last forward, for recurrent latent reasoning

        tokens = tokens_or_ids if tokens_or_ids.is_floating_point() else None

        # usual token embed if the input is not floating point

        if not exists(tokens):

            tokens = self.token_embed(tokens_or_ids)

            tokens = self.post_embed_norm(tokens)

        # variables

        seq_len, depth = tokens.shape[-2], self.depth

        # the initial token embeddings can be attention residual-ed

        if exists(attention_residual):
            if is_tensor(all_block_outputs):
                all_block_outputs = [all_block_outputs]

            all_block_outputs = default(all_block_outputs, [tokens])

        # destruct memories

        tokens_seen = 0

        if exists(memories):
            tokens_seen, _, memories = memories

        # positions

        seq = torch.arange(seq_len, device = device) + tokens_seen

        pos_emb = self.rope(seq)

        # memories

        memories = iter(default(memories, (None,) * depth))
        next_memories = []

        per_pass_hiddens = []

        # layers

        for layer_index in range(depth):
            prev_memory = next(memories, None)

            # bdh layer forward

            block_out, layer_memory = self.block(tokens, memories = prev_memory, rotary_emb = pos_emb, return_memories = True)

            # residual

            if exists(attention_residual):
                all_block_outputs.append(block_out)

                # the attention readout over the block outputs replaces the identity residual (paper eq. 5)

                query_index = 0 if self.attn_residual_tied else layer_index

                readout = attention_residual(
                    tokens,
                    all_block_outputs,
                    layer_index = query_index,
                    depth = depth,
                    total_reasoning_iterations = total_reasoning_iterations
                )

                tokens = readout
            else:
                # normal identity residual

                tokens = tokens + block_out

            # for latent reasoning steps being able to attention residual back across reasoning steps
            # in my experiments, much more stable at 8 reasoning steps, the usual identity way collapses, sans knowing their secretive latent transition function.

            if return_per_pass_hiddens:
                per_pass_hiddens.append(tokens)

            # the memory update can be frozen with `update_memory` (section 3.3)

            if update_memory:
                next_memory = layer_memory + default(prev_memory, 0.)
            else:
                next_memory = prev_memory

            next_memories.append(next_memory)

        # post norm, applied once

        tokens = self.post_norm(tokens)

        # readout

        logits = self.to_logits(tokens) if return_logits else None

        # return

        returns = (logits,)

        if return_memory:
            returns += (Memory(tokens_seen + seq_len, tokens, next_memories),)

        if return_per_pass_hiddens:
            returns += (per_pass_hiddens,)

        return pop_if_len_one(returns)

# reasoning wrapper for interleaved parallel token ingestion and recurrent latent reasoning

class BDHReasoningWrapper(Module):
    def __init__(
        self,
        bdh: BDH,
        ignore_index = -1,
        latent_step_embed = False
    ):
        super().__init__()
        self.bdh = bdh
        self.ignore_index = ignore_index

        # optional learned embedding injected at the very start of each latent reasoning step

        self.latent_step_embed = Parameter(zeros(bdh.dim)) if latent_step_embed else None

    def forward(
        self,
        *args,
        memories: Memory | None = None,
        return_loss = False,
        return_memory = False,
        update_memory = True,
        update_latent_memory = True,
        update_memory_per_stage: list[bool] | None = None,
        weight: Tensor | None = None
    ):
        # allow for passing a single list or tuple of inputs

        if len(args) == 1 and isinstance(first(args), (list, tuple)):
            args = first(args)

        # per-stage memory update flags zip with the stages and override the
        # two bool flags, so they never conflict - must cover every stage

        if exists(update_memory_per_stage):
            assert len(update_memory_per_stage) == len(args), 'update_memory_per_stage must have one flag per stage'
            assert all(isinstance(flag, bool) for flag in update_memory_per_stage)

        # loop through parallel tokens and latent reasoning steps

        logits = None
        last_tensor = None

        latent_logits = []
        latent_labels = []
        num_labeled_latents = 0

        # the block outputs are aggregated across the reasoning chain and re-fed to the attention residual during
        # the latent steps, seeded with the initial token embeddings (paper: v_0 = h_1), which persist as the first key

        all_block_outputs = None

        # attention residual latents are aware of their distance from the end of reasoning

        total_reasoning_iterations = sum(stage for stage in args if isinstance(stage, int))

        for stage_index, item in enumerate(args):

            # per-stage flag wins when given; the bool flags are the per-kind default

            stage_update = update_memory_per_stage[stage_index] if exists(update_memory_per_stage) else None

            # latent reasoning step, each step projected to predict the first token of the next segment

            if isinstance(item, int):
                assert exists(memories), 'must ingest tokens before latent reasoning'

                # the query's last hidden, already conditioned on the prompt memory (eq. 2, E_theta)

                latent = memories.embeds[..., -1:, :]

                # seed the aggregate with the initial token embeddings, once

                if not exists(all_block_outputs):
                    all_block_outputs = [latent]

                update = default(stage_update, update_latent_memory)

                for _ in range(item):
                    if exists(self.latent_step_embed):
                        latent = latent + self.latent_step_embed

                    _, memories = self.bdh(latent, memories = memories, return_memory = True, return_logits = False, update_memory = update, all_block_outputs = all_block_outputs, total_reasoning_iterations = total_reasoning_iterations)

                    latent = memories.embeds

                    if return_loss:
                        latent_logits.append(self.bdh.to_logits(latent))

            # parallel tokens

            elif is_tensor(item):
                last_tensor = item

                if return_loss:
                    # every latent token predicts the first token of this next segment

                    num_unlabeled = len(latent_logits) - num_labeled_latents

                    if not item.is_floating_point():
                        latent_labels.append(repeat(item[:, :1], 'b 1 -> b n', n = num_unlabeled))

                    num_labeled_latents = len(latent_logits)

                # depth bias applies only to latent steps, never parallel token passes

                update = default(stage_update, update_memory)

                logits, memories = self.bdh(item, memories = memories, update_memory = update, return_memory = True, total_reasoning_iterations = 0)

        # return

        if not return_loss:
            returns = (logits,)

            if return_memory:
                returns += (memories,)

            return pop_if_len_one(returns)

        assert exists(logits), 'a tensor stage must follow the latent reasoning'

        # never leave latent reasoning dangling at the very end

        assert not isinstance(last(args), int), 'latent reasoning cannot be the final stage'

        # every latent token predicts the first token of the next segment; each answer position predicts the next answer token

        all_logits = logits[:, :-1]
        labels = last_tensor[:, 1:]

        if latent_logits:
            latent_logits = cat(latent_logits, dim = 1)
            latent_labels = cat(latent_labels, dim = 1)

            all_logits = cat((latent_logits, all_logits), dim = 1)
            labels = cat((latent_labels, labels), dim = 1)

        # loss

        loss = F.cross_entropy(
            rearrange(all_logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index,
            weight = weight
        )

        # returns

        returns = (loss,)

        if return_memory:
            returns += (logits, memories)

        return pop_if_len_one(returns)

    def generate(
        self,
        *args,
        memories = None,
        num_tokens = None,
        stop_token = None,
        temperature = 1.,
        filter_thres = 0.9,
        update_memory = True,
        update_latent_memory = True,
        update_memory_per_stage: list[bool] | None = None,
        return_memory = False
    ):
        # decode an answer autoregressively after the given stages - the same interleaving as forward,
        # tensors ingested, ints latent reasoning steps - each generated token fed back in, seeded from
        # the projection of the last latent position, which at training predicts the first answer token.
        # stops early on `stop_token` when given. single sequence only

        assert exists(num_tokens) or exists(stop_token), 'either num_tokens or stop_token must be given'
        assert exists(memories) or len(args) > 0, 'must ingest tokens or pass memories before generating'

        device = next(self.parameters()).device

        # run the stages - any interleaving of prompt tensors and latent steps

        _, memories = self(
            *args,
            memories = memories,
            return_memory = True,
            update_memory = update_memory,
            update_latent_memory = update_latent_memory,
            update_memory_per_stage = update_memory_per_stage
        )

        # the seed: the last latent position projects the first answer token at training time

        latent = memories.embeds[..., -1:, :]
        logits = self.bdh.to_logits(latent)

        # decode one token at a time, feeding each back in, until num_tokens
        # are generated or the stop token is sampled

        tokens = []

        while not exists(num_tokens) or len(tokens) < num_tokens:
            if temperature == 0:
                token = logits[:, -1].argmax(-1).item()
            else:
                token = logits[:, -1] / temperature

                if filter_thres < 1.:
                    token = top_k(token, filter_thres)

                token = token.softmax(-1).multinomial(1).item()

            tokens.append(token)

            if exists(stop_token) and token == stop_token:
                break

            token_embeds = self.bdh.token_embed(torch.tensor([[token]], device = device))
            logits, memories = self(
                token_embeds,
                memories = memories,
                return_memory = True,
                update_memory = update_memory,
                update_latent_memory = update_latent_memory
            )

        # returns

        returns = (tokens,)

        if return_memory:
            returns += (memories,)

        return pop_if_len_one(returns)

# quick test

if __name__ == '__main__':

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)
