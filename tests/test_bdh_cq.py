import torch
import pytest

from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper, compute_attn_residual_depth_bias

MODEL_KWARGS = dict(
    dim = 512,
    num_tokens = 16,
    dim_qk_heads = 2048,
    depth = 2
)

def make_model(**overrides):
    return BDH(**{**MODEL_KWARGS, **overrides})

@pytest.fixture
def wrapper():
    return BDHReasoningWrapper(make_model())

def rand_ids(shape, num = 16):
    return torch.randint(0, num, shape)

def make_stages(stage_spec):
    # tensor stages by shape, ints passed through as latent reasoning steps

    return [rand_ids(item) if isinstance(item, tuple) else item for item in stage_spec]

def test_bdh_cq():
    model = make_model(num_tokens = 256)

    ids = rand_ids((2, 1024), 256)

    logits, memories = model(ids, return_memory = True)

    assert logits.shape == (2, 1024, 256)
    assert model(ids, memories = memories).shape == logits.shape

def test_bdh_cq_latent_reasoning():
    # raw model, latent reasoning loop with the memory writes frozen

    model = make_model()

    _, memories = model(rand_ids((1, 50)), return_memory = True)

    latent = memories.embeds[..., -1:, :]

    for _ in range(8):
        _, memories = model(latent, memories = memories, return_memory = True, return_logits = False, update_memory = False)
        latent = memories.embeds

    assert model(rand_ids((1, 100)), memories = memories).shape == (1, 100, 16)

# e2e - tensor stages ingested, int stages latent reasoning, any interleaving

@pytest.mark.parametrize(('stage_spec', 'logits_shape', 'seen'), [
    ([(2, 20), 8, (2, 30)], (2, 30, 16), 58),
    ([(1, 10), 2, (1, 15), 4, (1, 20)], (1, 20, 16), 51),
])
def test_bdh_reasoning_wrapper(wrapper, stage_spec, logits_shape, seen):

    logits, memories = wrapper(*make_stages(stage_spec), return_memory = True)

    assert logits.shape == logits_shape
    assert memories.tokens_seen == seen

    # a single list of stages works too

    assert wrapper(make_stages(stage_spec)).shape == logits_shape

# e2e - every latent step predicts the first token of the next segment, every answer position the next answer token

@pytest.mark.parametrize(('stage_spec', 'seen', 'rejected'), [
    ([(2, 20), 0, (2, 30)], 50, False),
    ([(2, 20), 8, (2, 30)], 58, False),
    ([(2, 10), 8, (2, 15)], 33, False),
    ([(2, 10), 4, (2, 15), 5, (2, 20)], 54, False),
    ([(2, 20), 8], None, True),
])
def test_bdh_reasoning_wrapper_loss(wrapper, stage_spec, seen, rejected):

    if rejected:
        with pytest.raises(AssertionError):
            wrapper(make_stages(stage_spec), return_loss = True)
        return

    loss, _, memories = wrapper(make_stages(stage_spec), return_loss = True, return_memory = True)

    assert memories.tokens_seen == seen

    loss.backward()

# e2e - autoregressive decode seeded from the last latent projection

@pytest.mark.parametrize(('stage_spec', 'num_tokens', 'stop_token'), [
    ([(1, 10), 2, (1, 15), 4], 8, None),
    ([(1, 10)], 100, 0),
])
def test_bdh_reasoning_wrapper_generate(wrapper, stage_spec, num_tokens, stop_token):

    tokens = wrapper.generate(make_stages(stage_spec), num_tokens = num_tokens, stop_token = stop_token)

    assert all(isinstance(token, int) for token in tokens)
    assert len(tokens) == num_tokens or tokens[-1] == stop_token

    # generate, think, generate - latent steps interleaved with the answer itself

    first, memories = wrapper.generate(rand_ids((1, 10)), 2, num_tokens = 5, return_memory = True)
    middle, memories = wrapper.generate(3, memories = memories, num_tokens = 5, return_memory = True)
    last, memories = wrapper.generate(3, memories = memories, num_tokens = 5, return_memory = True)

    assert memories.tokens_seen == 33
    assert all(0 <= token < 16 for token in first + middle + last)

def test_bdh_reasoning_wrapper_update_memory(wrapper):

    prompts, answers = rand_ids((2, 20)), rand_ids((2, 30))

    # update_latent_memory freezes only the latent writes

    _, base = wrapper(prompts, return_memory = True)
    _, frozen = wrapper(4, memories = base, return_memory = True, update_latent_memory = False)

    assert all(torch.equal(f, b) for f, b in zip(frozen.fast_weight_memories, base.fast_weight_memories))

    # per-stage flags zip with the stages and override the two bools

    logits, _ = wrapper(prompts, 4, answers, return_memory = True, update_memory = False, update_latent_memory = False, update_memory_per_stage = [True] * 3)
    logits_default, _ = wrapper(prompts, 4, answers, return_memory = True)

    assert torch.equal(logits, logits_default)

    # a frozen parallel stage writes nothing

    _, mem = wrapper(prompts, return_memory = True, update_memory_per_stage = [False])
    assert all(m is None for m in mem.fast_weight_memories)

    # the spec must cover every stage

    with pytest.raises(AssertionError):
        wrapper(prompts, 4, answers, update_memory_per_stage = [True, True])

def test_bdh_attn_residual_recycling():
    # alphafold2 style recycling - attend over the previous pass's per-layer hiddens

    model = make_model(attn_residual = True)

    tokens = rand_ids((1, 10))

    logits, _, hiddens = model(tokens, return_memory = True, return_per_pass_hiddens = True)
    recycled = model(tokens, all_block_outputs = hiddens)

    assert logits.shape == recycled.shape == (1, 10, 16)

    # mismatched sequence length rejected

    with pytest.raises(AssertionError):
        model(tokens[:, :-1], all_block_outputs = hiddens)

def test_bdh_attn_residual_depth_bias_wiring():
    # latent hiddens aware of their distance from the end of reasoning

    prompts, answers = rand_ids((2, 20)), rand_ids((2, 30))

    model = make_model(depth = 4, attn_residual = True, attn_residual_depth_bias_distance = 2)
    loss = BDHReasoningWrapper(model)(prompts, 3, answers, return_loss = True)
    loss.backward()

    assert model.attn_residual.has_depth_bias_distance
    assert model.attn_residual.depth_bias.grad is not None

    # off at a distance of 0

    model = make_model(depth = 4, attn_residual = True)
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
