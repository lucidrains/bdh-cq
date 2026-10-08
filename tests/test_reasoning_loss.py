import pytest
import torch
import torch.nn.functional as F

from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper
from bdh_cq.icq import CLASS_WEIGHTS, train_loss
from bdh_cq.tasks import TASKS


def make_wrapper(ignore_index = -1, num_tokens = 8):
    torch.manual_seed(0)
    return BDHReasoningWrapper(
        BDH(dim = 16, num_tokens = num_tokens, depth = 2, heads = 1,
            dim_qk_heads = 16, rotary_dim = 4),
        ignore_index = ignore_index
    )


@pytest.mark.parametrize('context', ['inline', 'cached_zero', 'cached_direct'])
@pytest.mark.parametrize('answer_length', [1, 3])
@pytest.mark.parametrize('packing', ['args', 'list', 'tuple'])
def test_zero_effort_supervises_first_answer(context, answer_length, packing):
    wrapper = make_wrapper()
    prompt = torch.tensor([[1, 2, 3]])
    answer = torch.tensor([[4, 5, 6]])[:, :answer_length]
    weight = torch.tensor([1., 1., 1., 1., 0.5, 2., 3., 1.])
    prompt_logits, memory = wrapper(prompt, return_memory = True)
    answer_logits = wrapper(answer, memories = memory)
    expected_logits = torch.cat((prompt_logits[:, -1:], answer_logits[:, :-1]), dim = 1)
    expected = F.cross_entropy(expected_logits.transpose(1, 2), answer, weight = weight)

    stages, kwargs = ([prompt, 0, answer], {}) if context == 'inline' else (
        [0, answer] if context == 'cached_zero' else [answer], {'memories': memory}
    )
    args = [tuple(stages)] if packing == 'tuple' else [stages] if packing == 'list' else stages
    loss, logits, returned = wrapper(*args, **kwargs, return_loss = True, return_memory = True, weight = weight)

    torch.testing.assert_close(loss, expected)
    assert logits.shape == (1, answer_length, 8)
    assert returned.tokens_seen == prompt.shape[1] + answer_length
    loss.backward()
    assert wrapper.bdh.token_embed.weight.grad[3].abs().sum() > 0
    assert wrapper.bdh.to_logits.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('steps', [0, 2])
def test_zero_stage_separators_do_not_duplicate_boundary_targets(steps):
    wrapper = make_wrapper()
    prompt, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    projections = []
    handle = wrapper.bdh.to_logits.register_forward_hook(lambda module, inputs, output: projections.append(output))
    loss = wrapper(prompt, 0, steps, 0, answer, return_loss = True)
    handle.remove()
    if steps:
        expected_logits = torch.cat((*projections[1:1 + steps], projections[-1][:, :-1]), dim = 1)
        labels = torch.cat((answer[:, :1].expand(-1, steps), answer[:, 1:]), dim = 1)
    else:
        expected_logits = torch.cat((projections[0][:, -1:], projections[-1][:, :-1]), dim = 1)
        labels = answer
    expected = F.cross_entropy(expected_logits.transpose(1, 2), labels)
    torch.testing.assert_close(loss, expected)


def test_ordinary_prompt_chunks_keep_final_only_shifted_loss():
    wrapper = make_wrapper()
    first, last = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    loss, logits, _ = wrapper(first, last, return_loss = True, return_memory = True)
    expected = F.cross_entropy(logits[:, :-1].transpose(1, 2), last[:, 1:])
    torch.testing.assert_close(loss, expected)


def test_zero_effort_intermediate_segment_remains_prompt_context():
    wrapper = make_wrapper()
    first, middle, last = torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), torch.tensor([[5, 6]])
    loss, logits, _ = wrapper(first, 0, middle, last, return_loss = True, return_memory = True)
    expected = F.cross_entropy(logits[:, :-1].transpose(1, 2), last[:, 1:])
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize('kind', ['single_token', 'all_ignored', 'empty'])
def test_no_valid_targets_returns_differentiable_zero(kind):
    wrapper = make_wrapper(ignore_index = 0)
    ids = torch.tensor([[4]]) if kind == 'single_token' else torch.tensor([[0, 0, 0]])
    if kind == 'empty':
        ids = ids[:, :0]
    loss = wrapper(ids, return_loss = True)
    assert loss.item() == 0.
    assert loss.requires_grad
    loss.backward()
    assert wrapper.bdh.to_logits.weight.grad is not None
    assert torch.isfinite(wrapper.bdh.to_logits.weight.grad).all()
    assert wrapper.bdh.to_logits.weight.grad.count_nonzero() == 0


def test_ignored_boundary_and_answer_targets_return_zero():
    wrapper = make_wrapper(ignore_index = 0)
    loss = wrapper(torch.tensor([[1, 2, 3]]), 0, torch.tensor([[0, 0]]), return_loss = True)
    assert loss.item() == 0.
    loss.backward()
    assert torch.isfinite(wrapper.bdh.token_embed.weight.grad).all()


def test_empty_cached_context_does_not_invent_boundary_prediction():
    wrapper = make_wrapper()
    _, memory = wrapper(torch.empty((1, 0), dtype = torch.long), return_memory = True)
    loss = wrapper(torch.tensor([[4]]), memories = memory, return_loss = True)
    assert loss.item() == 0.
    loss.backward()


def test_float_prompt_can_condition_zero_effort_answer():
    wrapper = make_wrapper()
    prompt = wrapper.bdh.token_embed(torch.tensor([[1, 2, 3]]))
    loss = wrapper(prompt, 0, torch.tensor([[4]]), return_loss = True)
    assert torch.isfinite(loss)
    loss.backward()


def test_unsupervised_float_context_after_zero_stage_preserves_shifted_loss():
    wrapper = make_wrapper()
    prompt, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])
    context = wrapper.bdh.token_embed(torch.tensor([[6, 7]]))
    loss, logits, _ = wrapper(prompt, 0, context, answer, return_loss = True, return_memory = True)
    expected = F.cross_entropy(logits[:, :-1].transpose(1, 2), answer[:, 1:])
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(wrapper.bdh.to_logits.weight.grad).all()


@pytest.mark.parametrize('float_target', ['final', 'after_reasoning'])
def test_float_supervised_stages_require_integer_targets(float_target):
    wrapper = make_wrapper()
    prompt, answer = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    floats = wrapper.bdh.token_embed(answer)
    stages = [prompt, 0, floats] if float_target == 'final' else [prompt, 1, floats, answer]
    with pytest.raises(ValueError, match = 'integer token IDs'):
        wrapper(stages, return_loss = True)


def test_icq_zero_effort_training_is_finite_and_backpropagates():
    wrapper = make_wrapper(num_tokens = 14)
    task = TASKS['order'](size = 3).generate(seed = 0)
    loss = train_loss(wrapper, task, 0, class_weights = CLASS_WEIGHTS)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(wrapper.bdh.to_logits.weight.grad).all()
