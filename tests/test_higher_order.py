from copy import deepcopy

import pytest
import torch

from bdh_cq import BDH, HigherOrderBDHLayer, RotaryEmbedding, apply_rotary_emb


def make_layer(**kwargs):
    torch.manual_seed(35)
    return HigherOrderBDHLayer(6, heads=2, dim_queries_keys=4, **kwargs).double()


def frequencies(length, rotary_dim):
    return RotaryEmbedding(rotary_dim).double()(length) if rotary_dim else None


def dense_prefix_reference(layer, tokens, rotary_emb):
    """Original two-hop operator evaluated independently at each visible prefix."""
    q1, q2, q3 = [layer.split_heads(layer.qk_activation(projection(tokens)))
                  for projection in layer.to_qks]
    if rotary_emb is not None:
        q1, q2, q3 = [apply_rotary_emb(rotary_emb, q) for q in (q1, q2, q3)]

    outputs = []
    for index in range(tokens.shape[1]):
        keys2, keys3 = q2[:, :, :index + 1], q3[:, :, :index + 1]
        pass1 = torch.einsum('bhjd,bhkd->bhjk', keys2, keys3)
        pass2 = torch.einsum('bhd,bhjd->bhj', q1[:, :, index], keys2)
        if layer.omit_self:
            pass1 = pass1.masked_fill(torch.eye(index + 1, dtype=torch.bool), 0.)
            pass2 = pass2.masked_fill(torch.arange(index + 1) == index, 0.)

        messages = torch.einsum('bhjk,bke->bhje', pass1, tokens[:, :index + 1])
        mass = pass1.sum(dim=-1)
        if not layer.include_pass1_mass:
            mass = mass.clamp_min(layer.eps)
            messages = messages / mass[..., None]

        numerator = torch.einsum('bhj,bhje->bhe', pass2, messages)
        denominator = (pass2 * mass).sum(dim=-1)
        aggregate = numerator / (denominator[..., None] + layer.eps)
        aggregate = torch.where(denominator[..., None] != 0., aggregate, torch.zeros_like(aggregate))
        outputs.append(aggregate)

    return torch.stack(outputs, dim=-2)


@pytest.mark.parametrize('include_mass', [True, False])
@pytest.mark.parametrize('omit_self', [True, False])
@pytest.mark.parametrize('rotary_dim', [0, 4])
def test_higher_order_matches_visible_prefix_poly_attention(include_mass, omit_self, rotary_dim):
    layer = make_layer(include_pass1_mass=include_mass, omit_self=omit_self)
    tokens = torch.randn(2, 7, 6, dtype=torch.double)
    rotary = frequencies(tokens.shape[1], rotary_dim)
    seen = []
    handle = layer.post_attn_norm.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    layer(tokens, rotary_emb=rotary)
    handle.remove()
    expected = dense_prefix_reference(layer, tokens, rotary)
    torch.testing.assert_close(seen[0], expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('include_mass', [True, False])
@pytest.mark.parametrize('omit_self', [True, False])
@pytest.mark.parametrize('chunk_size', [1, 3])
def test_higher_order_chunked_outputs_and_memory(include_mass, omit_self, chunk_size):
    layer = make_layer(include_pass1_mass=include_mass, omit_self=omit_self)
    tokens = torch.randn(2, 7, 6, dtype=torch.double)
    rotary = frequencies(tokens.shape[1], 4)
    full, full_memory = layer(tokens, rotary_emb=rotary, return_memories=True)
    q2, q3 = [apply_rotary_emb(rotary, layer.split_heads(layer.qk_activation(projection(tokens))))
              for projection in layer.to_qks[1:]]
    pass1 = torch.einsum('bhjd,bhkd->bhjk', q2, q3)
    if omit_self:
        pass1 = pass1.masked_fill(torch.eye(tokens.shape[1], dtype=torch.bool), 0.)
    expected_memory = (torch.einsum('bhnd,bne->bhde', q3, tokens), q3.sum(dim=-2), q2,
                       torch.einsum('bhjk,bke->bhje', pass1, tokens), pass1.sum(dim=-1))
    for actual_state, expected_state in zip(full_memory, expected_memory):
        torch.testing.assert_close(actual_state, expected_state, rtol=1e-8, atol=1e-9)
    memory = None
    chunks = []
    for start in range(0, tokens.shape[1], chunk_size):
        out, updated_memory = layer(tokens[:, start:start + chunk_size], memories=memory,
                                    rotary_emb=rotary[start:start + chunk_size], return_memories=True)
        memory = layer.combine_memories(updated_memory, memory)
        chunks.append(out)

    torch.testing.assert_close(torch.cat(chunks, dim=1), full, rtol=1e-8, atol=1e-9)
    assert len(memory) == len(full_memory)
    for chunked_state, full_state in zip(memory, full_memory):
        torch.testing.assert_close(chunked_state, full_state, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('include_mass', [True, False])
@pytest.mark.parametrize('omit_self', [True, False])
def test_higher_order_no_future_outputs_or_input_gradients(include_mass, omit_self):
    layer = make_layer(include_pass1_mass=include_mass, omit_self=omit_self)
    tokens = torch.randn(2, 7, 6, dtype=torch.double, requires_grad=True)
    changed = tokens.detach().clone()
    changed[:, 3:] += 4.
    original = layer(tokens)
    altered = layer(changed)
    torch.testing.assert_close(original[:, :3], altered[:, :3], rtol=0., atol=0.)
    gradient, = torch.autograd.grad(original[:, :3].square().sum(), tokens)
    assert torch.count_nonzero(gradient[:, 3:]) == 0


@pytest.mark.parametrize('include_mass', [True, False])
@pytest.mark.parametrize('omit_self', [True, False])
def test_higher_order_model_chunked_gradients(include_mass, omit_self):
    torch.manual_seed(1729)
    full_model = BDH(dim=12, num_tokens=16, depth=2, heads=2,
                     dim_qk_heads=16, rotary_dim=4, block_cls=HigherOrderBDHLayer).double()
    full_model.block.include_pass1_mass = include_mass
    full_model.block.omit_self = omit_self
    chunked_model = deepcopy(full_model)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
    full, full_memory = full_model(ids, return_memory=True)
    full.square().mean().backward()
    memory = None
    logits = []
    for chunk in ids.split(1, dim=1):
        out, memory = chunked_model(chunk, memories=memory, return_memory=True)
        logits.append(out)
    chunked = torch.cat(logits, dim=1)
    chunked.square().mean().backward()

    torch.testing.assert_close(chunked, full, rtol=1e-8, atol=1e-9)
    for name, parameter in full_model.named_parameters():
        gradient = dict(chunked_model.named_parameters())[name].grad
        if parameter.grad is None:
            assert gradient is None
            continue
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, parameter.grad, rtol=1e-6, atol=1e-8)
    for expected_layer, actual_layer in zip(full_memory.fast_weight_memories, memory.fast_weight_memories):
        for expected_state, actual_state in zip(expected_layer, actual_layer):
            torch.testing.assert_close(actual_state, expected_state, rtol=1e-8, atol=1e-9)


def test_higher_order_singleton_reads_historical_intermediates():
    layer = HigherOrderBDHLayer(4, heads=1, dim_queries_keys=1).double()
    with torch.no_grad():
        for projection in layer.to_qks:
            projection.weight.fill_(1.)
    query = torch.tensor([[[.2, .4, .1, 1.]]], dtype=torch.double)
    contexts = [torch.tensor([[[1., 2., 3., 4.], [4., 1., 7., 2.]]], dtype=torch.double),
                torch.tensor([[[7., 1., 1., 2.], [1., 7., 2., 1.]]], dtype=torch.double)]
    seen = []
    handle = layer.post_attn_norm.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    for context in contexts:
        _, memory = layer(context, return_memories=True)
        layer(query, memories=memory)
    handle.remove()
    assert torch.count_nonzero(seen[1]) > 0
    assert not torch.allclose(seen[1], seen[3])


def test_higher_order_rejects_incomplete_legacy_memory():
    layer = make_layer()
    legacy_memory = (torch.zeros(2, 2, 4, 6, dtype=torch.double),
                     torch.zeros(2, 2, 4, dtype=torch.double))
    tokens = torch.randn(2, 3, 6, dtype=torch.double)
    with pytest.raises(ValueError, match='legacy|recreate|re-ingest'):
        layer(tokens, memories=legacy_memory)
    _, modern_memory = layer(tokens, return_memories=True)
    with pytest.raises(ValueError, match='legacy|recreate|re-ingest'):
        layer.combine_memories(modern_memory, legacy_memory)


@pytest.mark.parametrize('include_mass', [True, False])
def test_higher_order_float32_sparse_empty_messages_are_exactly_zero(include_mass):
    layer = HigherOrderBDHLayer(4, heads=1, dim_queries_keys=4,
                               include_pass1_mass=include_mass)
    with torch.no_grad():
        layer.to_qks[0].weight.fill_(.179)
        layer.to_qks[1].weight.copy_(torch.diag(torch.tensor([.137, .311, .517, .733])))
        layer.to_qks[2].weight.copy_(torch.diag(torch.tensor([.193, .367, .563, .797])))
    tokens = torch.diag(torch.tensor([.113, .257, .419, .683]))[None]
    seen = []
    handle = layer.post_attn_norm.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    full, _ = layer(tokens, return_memories=True)
    memory = None
    chunked = []
    for chunk in tokens.split(1, dim=1):
        out, updated = layer(chunk, memories=memory, return_memories=True)
        memory = layer.combine_memories(updated, memory)
        chunked.append(out)
    handle.remove()

    for aggregate in seen:
        assert torch.isfinite(aggregate).all()
        assert torch.count_nonzero(aggregate) == 0
    assert torch.count_nonzero(full) == 0
    torch.testing.assert_close(torch.cat(chunked, dim=1), full, rtol=0., atol=0.)
