import pytest
import torch

from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper, compute_attn_residual_depth_bias
from bdh_cq.rotary import RotaryEmbedding, apply_rotary_emb


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(('num_keys', 'iterations', 'bias_values', 'expected'), [
    (5, 0, [0.5, 1.0], [0, 0, 0, 0, 0]),
    (5, 1, [0.5, 1.0], [0, 1, 1, 1, 1]),
    (9, 2, [1.0], [0, 0, 0, 0, 0, 1, 1, 1, 1]),
    (7, 1, [1.0], [0, 1, 1, 1, 1, 1, 1]),
])
def test_depth_bias_preserves_schedule_dtype(dtype, num_keys, iterations, bias_values, expected):
    schedule = torch.tensor(bias_values, dtype = dtype, requires_grad = True)
    bias = compute_attn_residual_depth_bias(
        num_keys,
        bias_schedule = schedule,
        depth = 4,
        total_reasoning_iterations = iterations
    )

    assert bias.dtype == dtype
    assert bias.device == schedule.device
    torch.testing.assert_close(bias, torch.tensor(expected, dtype = dtype))
    if iterations:
        bias.sum().backward()
        assert schedule.grad is not None
        assert torch.isfinite(schedule.grad).all()


@pytest.mark.parametrize('use_length', [False, True])
def test_bfloat16_rotary_keeps_adjacent_integer_positions(use_length):
    rope = RotaryEmbedding(8).bfloat16()
    positions = torch.tensor([256, 257])
    phases = rope(2, offset = 256) if use_length else rope(positions)
    expected = (positions.float()[:, None] * rope.freqs.float()).repeat_interleave(2, dim = -1)

    assert phases.dtype == torch.float32
    torch.testing.assert_close(phases, expected)
    assert not torch.equal(phases[0], phases[1])

    # identical activations at adjacent positions must rotate differently.
    activations = torch.ones(1, 1, 2, 8, dtype = torch.bfloat16)
    rotated = apply_rotary_emb(phases, activations)
    assert rotated.dtype == activations.dtype
    assert not torch.equal(rotated[..., 0, :], rotated[..., 1, :])


@pytest.mark.parametrize('use_length', [False, True])
def test_float16_rotary_stays_finite_at_large_integer_positions(use_length):
    rope = RotaryEmbedding(8).half()
    positions = torch.tensor([65520, 65521, 65522])
    phases = rope(3, offset = 65520) if use_length else rope(positions)
    expected = (positions.float()[:, None] * rope.freqs.float()).repeat_interleave(2, dim = -1)

    assert phases.dtype == torch.float32
    assert torch.isfinite(phases).all()
    torch.testing.assert_close(phases, expected)

    activations = torch.ones(1, 1, 3, 8, dtype = torch.float16)
    rotated = apply_rotary_emb(phases, activations)
    assert rotated.dtype == activations.dtype
    assert torch.isfinite(rotated).all()
    assert not torch.equal(rotated[..., 0, :], rotated[..., 1, :])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rotary_partial_rotation_and_learned_frequency_gradients(dtype):
    torch.manual_seed(0)
    rope = RotaryEmbedding(8, learned_freq = True).to(dtype = dtype)
    positions = torch.tensor([0, 7, 256, 257])
    phases = rope(positions)
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    expected_phases = (positions.to(compute_dtype)[:, None] * rope.freqs.to(compute_dtype)).repeat_interleave(2, dim = -1)
    activations = torch.randn(1, 2, 4, 16, dtype = dtype, requires_grad = True)

    # Independent pairwise rotation reference, with the native model's rounded
    # frequency values rather than the values before model.half()/bfloat16().
    pairs = activations[..., 4:12].to(compute_dtype).reshape(1, 2, 4, 4, 2)
    phase_pairs = expected_phases[..., ::2]
    first = pairs[..., 0] * phase_pairs.cos() - pairs[..., 1] * phase_pairs.sin()
    second = pairs[..., 1] * phase_pairs.cos() + pairs[..., 0] * phase_pairs.sin()
    expected_middle = torch.stack((first, second), dim = -1).flatten(-2).to(dtype)
    rotated = apply_rotary_emb(phases, activations, start_index = 4)

    assert phases.dtype == compute_dtype
    assert rotated.dtype == dtype
    torch.testing.assert_close(phases, expected_phases)
    torch.testing.assert_close(rotated[..., 4:12], expected_middle)
    assert torch.equal(rotated[..., :4], activations[..., :4])
    assert torch.equal(rotated[..., 12:], activations[..., 12:])

    weights = torch.arange(16, dtype = torch.float32)
    (rotated.float() * weights).mean().backward()
    assert rope.freqs.grad is not None
    assert torch.isfinite(rope.freqs.grad).all()
    assert rope.freqs.grad.abs().max() > 0
    assert activations.grad is not None
    assert torch.isfinite(activations.grad).all()


def test_rotary_preserves_double_positions_and_activation_math():
    positions = torch.tensor([256.125, 257.25], dtype = torch.float64)
    rope = RotaryEmbedding(4)
    phases = rope(positions)
    expected = (positions[:, None] * rope.freqs.double()).repeat_interleave(2, dim = -1)
    assert phases.dtype == torch.float64
    torch.testing.assert_close(phases, expected, rtol = 0, atol = 0)

    activations = torch.randn(1, 1, 2, 4, dtype = torch.float64)
    rotated = apply_rotary_emb(phases, activations)
    pairs = activations.reshape(1, 1, 2, 2, 2)
    phase_pairs = expected[..., ::2]
    first = pairs[..., 0] * phase_pairs.cos() - pairs[..., 1] * phase_pairs.sin()
    second = pairs[..., 1] * phase_pairs.cos() + pairs[..., 0] * phase_pairs.sin()
    reference = torch.stack((first, second), dim = -1).flatten(-2)
    assert rotated.dtype == torch.float64
    torch.testing.assert_close(rotated, reference, rtol = 0, atol = 0)


def test_double_rotary_keeps_large_fractional_length_offset():
    rope = RotaryEmbedding(8).double()
    offset = 16777216.125
    positions = torch.arange(3, dtype = torch.float64) + offset
    expected = (positions[:, None] * rope.freqs).repeat_interleave(2, dim = -1)
    tensor_phases = rope(positions)
    length_phases = rope(3, offset = offset)

    assert length_phases.dtype == torch.float64
    torch.testing.assert_close(tensor_phases, expected, rtol = 0, atol = 0)
    torch.testing.assert_close(length_phases, expected, rtol = 0, atol = 0)


def test_rotary_phase_precision_under_cpu_autocast():
    rope = RotaryEmbedding(8).bfloat16()
    positions = torch.tensor([256, 257])
    with torch.autocast('cpu', dtype = torch.bfloat16):
        phases = rope(positions)
        rotated = apply_rotary_emb(phases, torch.ones(1, 1, 2, 8, dtype = torch.bfloat16))

    assert phases.dtype == torch.float32
    assert not torch.equal(phases[0], phases[1])
    assert torch.isfinite(rotated).all()
    assert not torch.equal(rotated[..., 0, :], rotated[..., 1, :])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_native_low_precision_biased_model_forward_backward(dtype):
    torch.manual_seed(0)
    wrapper = BDHReasoningWrapper(BDH(
        dim = 16,
        num_tokens = 8,
        depth = 2,
        heads = 2,
        dim_qk_heads = 32,
        rotary_dim = 8,
        attn_residual = True,
        attn_residual_depth_bias_distance = 2
    )).to(dtype = dtype)
    prompts = torch.randint(0, 8, (1, 6))
    answers = torch.randint(0, 8, (1, 4))

    # This deliberately uses native .half()/.bfloat16(), outside autocast.
    loss, logits, memory = wrapper(prompts, 3, answers, return_loss = True, return_memory = True)
    assert logits.dtype == dtype
    assert memory.embeds.dtype == dtype
    assert torch.isfinite(loss)
    assert torch.isfinite(logits).all()
    loss.backward()
    gradients = [param.grad for param in wrapper.parameters() if param.requires_grad]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
