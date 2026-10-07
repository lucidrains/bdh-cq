import torch
import pytest

from bdh_cq.bdh_cq import (BDH, BDHBlock, BDHReasoningWrapper,
                           compute_attn_residual_depth_bias, exists)
from bdh_cq.higher_order_bdh import HigherOrderBDHLayer
from bdh_cq.rotary import RotaryEmbedding, apply_rotary_emb

MODEL_KWARGS = dict(
    dim = 512,
    num_tokens = 16,
    dim_qk_heads = 2048,
    depth = 2
)

# the higher order layer slots into bdh in place of the order-1 layer, sharing
# the outer machinery - every e2e test below runs against both

BLOCK_CLASSES = [BDHBlock, HigherOrderBDHLayer]

param = pytest.mark.parametrize('block_cls', BLOCK_CLASSES, ids = ['BDH', 'HigherOrder'])

def make_model(block_cls = BDHBlock, **overrides):
    return BDH(**{**MODEL_KWARGS, 'block_cls': block_cls, **overrides})

def make_wrapper(block_cls = BDHBlock):
    return BDHReasoningWrapper(make_model(block_cls))

def memory_equal(a, b):
    # order-1 memories are tensors, order-2 a tuple per layer

    if isinstance(a, tuple):
        return all(torch.equal(x, y) for x, y in zip(a, b))

    return torch.equal(a, b)

def rand_ids(shape, num = 16):
    return torch.randint(0, num, shape)

def make_stages(stage_spec):
    # tensor stages by shape, ints passed through as latent reasoning steps

    return [rand_ids(item) if isinstance(item, tuple) else item for item in stage_spec]

@param
def test_bdh_cq(block_cls):
    model = make_model(block_cls, num_tokens = 256)

    ids = rand_ids((2, 1024), 256)

    logits, memories = model(ids, return_memory = True)

    assert logits.shape == (2, 1024, 256)
    assert model(ids, memories = memories).shape == logits.shape

@param
def test_bdh_cq_latent_reasoning(block_cls):
    # raw model, latent reasoning loop with the memory writes frozen

    model = make_model(block_cls)

    _, memories = model(rand_ids((1, 50)), return_memory = True)

    latent = memories.embeds[..., -1:, :]

    for _ in range(8):
        _, memories = model(latent, memories = memories, return_memory = True, return_logits = False, update_memory = False)
        latent = memories.embeds

    assert model(rand_ids((1, 100)), memories = memories).shape == (1, 100, 16)

# e2e - tensor stages ingested, int stages latent reasoning, any interleaving

@param
@pytest.mark.parametrize(('stage_spec', 'logits_shape', 'seen'), [
    ([(2, 20), 8, (2, 30)], (2, 30, 16), 58),
    ([(1, 10), 2, (1, 15), 4, (1, 20)], (1, 20, 16), 51),
])
def test_bdh_reasoning_wrapper(block_cls, stage_spec, logits_shape, seen):
    wrapper = make_wrapper(block_cls)

    logits, memories = wrapper(*make_stages(stage_spec), return_memory = True)

    assert logits.shape == logits_shape
    assert memories.tokens_seen == seen

    # a single list of stages works too

    assert wrapper(make_stages(stage_spec)).shape == logits_shape

@param
def test_bdh_reasoning_wrapper_trailing_latent(block_cls):
    # a run ending on latent reasoning has no logits of its own

    wrapper = make_wrapper(block_cls)

    logits, _ = wrapper(rand_ids((1, 10)), 2, return_memory = True)

    assert logits is None

    # and memories alone are enough to generate from

    _, memories = wrapper(rand_ids((1, 10)), return_memory = True)
    tokens = wrapper.generate(memories = memories, num_tokens = 3)

    assert len(tokens) == 3

# e2e - every latent step predicts the first token of the next segment, every answer position the next answer token

@param
@pytest.mark.parametrize(('stage_spec', 'seen', 'rejected'), [
    ([(2, 20), 0, (2, 30)], 50, False),
    ([(2, 20), 8, (2, 30)], 58, False),
    ([(2, 10), 8, (2, 15)], 33, False),
    ([(2, 10), 4, (2, 15), 5, (2, 20)], 54, False),
    ([(2, 20), 8], None, True),
])
def test_bdh_reasoning_wrapper_loss(block_cls, stage_spec, seen, rejected):
    wrapper = make_wrapper(block_cls)

    if rejected:
        with pytest.raises(AssertionError):
            wrapper(make_stages(stage_spec), return_loss = True)
        return

    loss, _, memories = wrapper(make_stages(stage_spec), return_loss = True, return_memory = True)

    assert memories.tokens_seen == seen

    loss.backward()

# e2e - autoregressive decode seeded from the last latent projection

@param
@pytest.mark.parametrize(('stage_spec', 'num_tokens', 'stop_token'), [
    ([(1, 10), 2, (1, 15), 4], 8, None),
    ([(1, 10)], 100, 0),
])
def test_bdh_reasoning_wrapper_generate(block_cls, stage_spec, num_tokens, stop_token):
    wrapper = make_wrapper(block_cls)

    tokens = wrapper.generate(make_stages(stage_spec), num_tokens = num_tokens, stop_token = stop_token)

    assert all(isinstance(token, int) for token in tokens)
    assert len(tokens) == num_tokens or tokens[-1] == stop_token

    # generate, think, generate - latent steps interleaved with the answer itself

    first, memories = wrapper.generate(rand_ids((1, 10)), 2, num_tokens = 5, return_memory = True)
    middle, memories = wrapper.generate(3, memories = memories, num_tokens = 5, return_memory = True)
    last, memories = wrapper.generate(3, memories = memories, num_tokens = 5, return_memory = True)

    assert memories.tokens_seen == 33
    assert all(0 <= token < 16 for token in first + middle + last)

@param
def test_bdh_reasoning_wrapper_update_memory(block_cls):
    wrapper = make_wrapper(block_cls)

    prompts, answers = rand_ids((2, 20)), rand_ids((2, 30))

    # update_latent_memory freezes only the latent writes

    _, base = wrapper(prompts, return_memory = True)
    _, frozen = wrapper(4, memories = base, return_memory = True, update_latent_memory = False)

    assert all(memory_equal(f, b) for f, b in zip(frozen.fast_weight_memories, base.fast_weight_memories))

    # per-stage flags zip with the stages and override the two bools

    logits, _ = wrapper(prompts, 4, answers, return_memory = True, update_memory = False, update_latent_memory = False, update_memory_per_stage = [True] * 3)
    logits_default, _ = wrapper(prompts, 4, answers, return_memory = True)

    assert torch.equal(logits, logits_default)

    # a frozen parallel stage writes nothing

    _, mem = wrapper(prompts, return_memory = True, update_memory_per_stage = [False])
    assert all(not exists(m) for m in mem.fast_weight_memories)

    # the spec must cover every stage

    with pytest.raises(AssertionError):
        wrapper(prompts, 4, answers, update_memory_per_stage = [True, True])

def test_apply_rotary_emb_partial():
    # only the dims covered by the freqs rotate, the rest of the qk dims are untouched

    freqs = RotaryEmbedding(8)(torch.arange(16))

    t = torch.randn(2, 3, 16, 64)

    rotated = apply_rotary_emb(freqs, t)

    assert rotated.shape == t.shape
    assert not torch.allclose(rotated[..., :8], t[..., :8])
    assert torch.equal(rotated[..., 8:], t[..., 8:])

    # start_index lifts the rotation to a later slice of the qk dims

    rotated = apply_rotary_emb(freqs, t, start_index = 16)

    assert torch.equal(rotated[..., :16], t[..., :16])
    assert not torch.allclose(rotated[..., 16:24], t[..., 16:24])
    assert torch.equal(rotated[..., 24:], t[..., 24:])

    # rotary needs the qk dims, and only rotates at position-dependent freqs

    assert torch.equal(rotated[:, :, 0, 16:24], t[:, :, 0, 16:24])

@param
def test_partial_rotary(block_cls):
    # dim_qk is huge, a small slice of it is position-encoded

    model = make_model(block_cls, dim_qk_heads = 512, rotary_dim = 16)

    assert model.rope.freqs.shape == (8,)

    logits, memories = model(rand_ids((2, 20)), return_memory = True)

    assert logits.shape == (2, 20, 16)
    assert memories.embeds.shape == (2, 20, 512)

    loss = logits.float().mean()
    loss.backward()

    assert exists(model.block.proj_up.grad)

    # rotary_dim 0 disables position embeddings entirely

    model = make_model(block_cls, dim_qk_heads = 512, rotary_dim = 0)

    assert not exists(model.rope)
    assert model(rand_ids((2, 20))).shape == (2, 20, 16)

@param
def test_attn_residual_recycling(block_cls):
    # alphafold2 style recycling - attend over the previous pass's per-layer hiddens

    model = make_model(block_cls, attn_residual = True)

    tokens = rand_ids((1, 10))

    logits, _, hiddens = model(tokens, return_memory = True, return_per_pass_hiddens = True)
    recycled = model(tokens, all_block_outputs = hiddens)

    assert logits.shape == recycled.shape == (1, 10, 16)

    # mismatched sequence length rejected

    with pytest.raises(AssertionError):
        model(tokens[:, :-1], all_block_outputs = hiddens)

@param
def test_attn_residual_depth_bias_wiring(block_cls):
    # latent hiddens aware of their distance from the end of reasoning

    prompts, answers = rand_ids((2, 20)), rand_ids((2, 30))

    model = make_model(block_cls, depth = 4, attn_residual = True, attn_residual_depth_bias_distance = 2)
    loss = BDHReasoningWrapper(model)(prompts, 3, answers, return_loss = True)
    loss.backward()

    assert model.attn_residual.has_depth_bias_distance
    assert exists(model.attn_residual.depth_bias.grad)

    # off at a distance of 0

    model = make_model(block_cls, depth = 4, attn_residual = True)
    loss = BDHReasoningWrapper(model)(prompts, 3, answers, return_loss = True)
    loss.backward()

    assert not model.attn_residual.has_depth_bias_distance
    assert not hasattr(model.attn_residual, 'depth_bias')

@pytest.mark.parametrize(('num_keys', 'depth', 'total_reasoning_iterations', 'bias_schedule', 'expected'), [
    # canonical example - 3 reasoning cycles of depth 4, 2 biases, on the last reasoning step at the last layer

    (13, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]),

    # 2 depths away from the end of the last reasoning step - the tail is excised

    (11, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1]),

    # on the first reasoning step, beyond the designated distance, all biases excised

    (5, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0]),

    # fewer reasoning cycles than biases, curtailed to those closest to the end

    (5, 4, 1, [0.5, 1.0], [0, 1, 1, 1, 1]),

    # latents written after reasoning concludes keep the maximum bias

    (15, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]),

    # no reasoning cycles, no bias anywhere

    (5, 4, 0, [0.5, 1.0], [0, 0, 0, 0, 0]),
])
def test_compute_attn_residual_depth_bias(num_keys, depth, total_reasoning_iterations, bias_schedule, expected):

    bias = compute_attn_residual_depth_bias(
        num_keys,
        bias_schedule = torch.tensor(bias_schedule),
        depth = depth,
        total_reasoning_iterations = total_reasoning_iterations
    )

    assert bias.shape == (num_keys,)
    assert torch.allclose(bias, torch.tensor(expected, dtype = torch.float32))
