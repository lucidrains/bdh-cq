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

def test_bdh_cq():

    model = BDH(
        dim = 512,
        num_tokens = 256
    )

    ids = torch.randint(0, 256, (2, 1024))

    logits = model(ids)

    assert logits.shape == (2, 1024, 256)

    logits, memories = model(ids, return_memory = True)
    logits2 = model(ids, memories = memories)

    assert logits2.shape == logits.shape

def test_bdh_cq_latent_reasoning():

    model = BDH(
        dim = 512,
        num_tokens = 16
    )

    prompts = torch.randint(0, 16, (1, 50))
    answers = torch.randint(0, 16, (1, 100))

    _, memories = model(prompts, return_memory = True)

    latent = memories.embeds[..., -1:, :]
    for _ in range(8):
        _, memories = model(latent, memories = memories, return_memory = True, return_logits = False, update_memory = False)
        latent = memories.embeds

    answer_logits = model(answers, memories = memories)

    assert answer_logits.shape == (1, 100, 16)

def test_bdh_reasoning_wrapper(wrapper):

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    answer_logits = wrapper(prompts, 8, answers)
    assert answer_logits.shape == (2, 30, 16)

    answer_logits = wrapper([prompts, 8, answers])
    assert answer_logits.shape == (2, 30, 16)

    # arbitrary stages: parallel -> latent -> parallel -> latent -> parallel

    p1 = torch.randint(0, 16, (1, 10))
    p2 = torch.randint(0, 16, (1, 15))
    ans = torch.randint(0, 16, (1, 20))

    ans_logits, memories = wrapper(p1, 2, p2, 4, ans, return_memory = True)

    assert ans_logits.shape == (1, 20, 16)
    assert memories.tokens_seen == (10 + 2 + 15 + 4 + 20)

def test_bdh_reasoning_wrapper_return_loss(wrapper):

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    loss, logits, memories = wrapper(prompts, 8, answers, return_loss = True, return_memory = True)

    assert logits.shape == (2, 30, 16)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_without_latent(wrapper):

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    loss = wrapper(prompts, 0, answers, return_loss = True)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_predicts_next_segment_first_token(wrapper):

    # no answer targets: every latent token still predicts the first token
    # of the next tensor segment

    p1 = torch.randint(0, 16, (2, 20))
    p2 = torch.randint(0, 16, (2, 30))

    loss = wrapper(p1, 8, p2, return_loss = True)

    loss.backward()

def test_bdh_reasoning_wrapper_loss_interleaved(wrapper):

    # p1, 4 latent, p2, 5 latent, answer: each latent section predicts the
    # first token of the segment that follows it

    p1 = torch.randint(0, 16, (2, 10))
    p2 = torch.randint(0, 16, (2, 15))
    answer = torch.randint(0, 16, (2, 20))

    loss, logits, memories = wrapper(p1, 4, p2, 5, answer, return_loss = True, return_memory = True)

    assert memories.tokens_seen == (10 + 4 + 15 + 5 + 20)
    assert logits.shape == (2, 20, 16)

    loss.backward()

def test_bdh_reasoning_wrapper_trailing_latent_rejected(wrapper):

    prompts = torch.randint(0, 16, (2, 20))

    with pytest.raises(AssertionError):
        wrapper(prompts, 8, return_loss = True)

def test_bdh_attn_residual_recycling():

    # pass the same sequence in again, attended over the previous pass's per-layer hiddens, alphafold2 style recycling

    model = make_model(attn_residual = True)

    tokens = torch.randint(0, 16, (1, 10))

    logits, _, per_pass_hiddens = model(tokens, return_memory = True, return_per_pass_hiddens = True)
    recycled = model(tokens, all_block_outputs = per_pass_hiddens)

    assert logits.shape == (1, 10, 16)
    assert recycled.shape == (1, 10, 16)

    # mismatched sequence length must be rejected

    with pytest.raises(AssertionError):
        model(tokens[:, :-1], all_block_outputs = per_pass_hiddens)

@pytest.mark.parametrize(('num_keys', 'depth', 'total_reasoning_iterations', 'bias_schedule', 'expected'), [
    # canonical example: 3 reasoning cycles of depth 4, 2 biases, on the last reasoning step at the last layer
    # the 2 biases are repeated by the depth (8), the earlier cycle zero padded on the left (12)

    (13, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]),

    # same, but 2 depths away from the end of the last reasoning step - the tail is excised

    (11, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1]),

    # on the first reasoning step, beyond the designated distance, all biases excised

    (5, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0]),

    # fewer reasoning cycles than biases, curtailed to those closest to the end

    (5, 4, 1, [0.5, 1.0], [0, 1, 1, 1, 1]),

    # latents written after reasoning concludes keep the maximum bias

    (15, 4, 3, [0.5, 1.0], [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]),

    # as many biases as reasoning cycles, no zero padding needed

    (7, 2, 3, [0.1, 0.5, 1.0], [0, 0.1, 0.1, 0.5, 0.5, 1, 1]),

    # a single bias for the last reasoning cycle only

    (5, 2, 2, [0.7], [0, 0, 0, 0.7, 0.7]),

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

def test_attn_residual_depth_bias_wiring():

    # the reasoning wrapper tracks the iteration and the total so the attention residual
    # latents are aware of their distance from the end of reasoning

    model = make_model(
        depth = 4,
        attn_residual = True,
        attn_residual_depth_bias_distance = 2
    )

    wrapper = BDHReasoningWrapper(model)

    prompts = torch.randint(0, 16, (2, 20))
    answers = torch.randint(0, 16, (2, 30))

    loss = wrapper(prompts, 3, answers, return_loss = True)

    loss.backward()

    assert model.attn_residual.has_depth_bias_distance
    assert model.attn_residual.depth_bias.grad is not None

    # distance of 0 keeps the attention residual bias off entirely

    model = make_model(
        depth = 4,
        attn_residual = True
    )

    wrapper = BDHReasoningWrapper(model)

    loss = wrapper(prompts, 3, answers, return_loss = True)

    loss.backward()

    assert not model.attn_residual.has_depth_bias_distance
    assert not hasattr(model.attn_residual, 'depth_bias')

def test_bdh_reasoning_wrapper_generate():

    # any interleaving of prompt tensors and latent steps, then an
    # autoregressively decoded answer - paper sec. 3.3 / fig. 7

    wrapper = BDHReasoningWrapper(make_model())

    first_prompt = torch.randint(0, 16, (1, 10))
    second_prompt = torch.randint(0, 16, (1, 15))

    # greedy decoding to a fixed number of tokens

    tokens = wrapper.generate(
        first_prompt,
        2,
        second_prompt,
        4,
        num_tokens = 8
    )
    assert isinstance(tokens, list)
    assert len(tokens) == 8
    assert all(isinstance(token, int) for token in tokens)

    # stop early on the stop token, or decode the full length otherwise

    tokens = wrapper.generate(
        first_prompt,
        num_tokens = 100,
        stop_token = 0
    )
    assert len(tokens) == 100 or tokens[-1] == 0

    # memories can be returned and passed back in, so latent steps can be
    # interleaved with the answer itself: generate, think, generate

    first_answer, memories = wrapper.generate(
        first_prompt,
        2,
        num_tokens = 5,
        return_memory = True
    )
    middle_answer, memories = wrapper.generate(
        3,
        memories = memories,
        num_tokens = 5,
        return_memory = True
    )
    last_answer, memories = wrapper.generate(
        3,
        memories = memories,
        num_tokens = 5,
        return_memory = True
    )

    assert memories.tokens_seen == 10 + 2 + 5 + 3 + 5 + 3 + 5
    assert len(first_answer) == len(middle_answer) == len(last_answer) == 5

    # token ids stay in the vocabulary

    assert all(0 <= token < 16 for token in first_answer + middle_answer + last_answer)
