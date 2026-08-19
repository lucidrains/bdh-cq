import numpy as np
import pytest
import torch

from bdh_cq.bdh_cq import BDHReasoningWrapper
from bdh_cq.icq import (make_model, encode_grid, encode_output, decode_grid,
                        task_prompt, task_answer, answer_length, task_at_level,
                        ingest, ingest_hiddens, generate_answer, solve,
                        cell_stats, train_loss, CLASS_WEIGHTS, ROW, IN, OUT, EOS)
from bdh_cq.tasks import TASKS

# a tiny model so the whole suite runs in seconds

TINY = dict(dim = 64, depth = 2, dim_qk_heads = 512)


@pytest.fixture
def wrapper():
    return BDHReasoningWrapper(make_model(**TINY))


def test_encode_decode_roundtrip():
    # grids serialize to tokens (markers + rows) and back, for every family

    for family in TASKS.values():
        task = family(size = 3).generate(seed = 0)

        for _, input_grid, output_grid in task["train"]:
            assert np.array_equal(decode_grid(encode_grid(input_grid)), input_grid)
            assert np.array_equal(decode_grid(encode_output(output_grid)), output_grid)

            tokens = encode_output(output_grid)
            assert tokens[0] == OUT
            assert tokens[-1] == EOS
            assert ROW in tokens


def test_task_prompt_answer():
    # the prompt is the demos followed by the query input, and the answer
    # is the query output grid serialized with a terminator

    task = TASKS["order"]().generate(seed = 0)

    prompt = task_prompt(task)

    assert prompt[-1] != EOS
    assert prompt.count(IN) == 4  # 3 demos + query input

    answer = task_answer(task)
    assert answer == encode_output(task["test"][0][2])
    assert len(answer) == answer_length(task)


def test_task_at_level():
    # demos are drawn from the easy levels, the query from the held-out one

    task = task_at_level(TASKS["nesting"], seed = 0, level = 3)

    demo_levels = [level for level, _, _ in task["train"]]
    query_levels = [level for level, _, _ in task["test"]]

    assert all(level <= 3 for level in demo_levels)
    assert query_levels == [3]


def test_cell_stats():
    predicted = np.array([[1, 2], [3, 4]])
    target = np.array([[1, 2], [3, 0]])

    assert cell_stats(predicted, target) == (3, 4, True)

    # wrong dimensions are not counted as correct cells

    assert cell_stats(np.zeros((2, 2)), np.zeros((3, 3)))[2] is False


def test_ingest_and_hiddens(wrapper):
    task = TASKS["propagation"]().generate(seed = 0)
    prompt = task_prompt(task)

    memories = ingest(wrapper, prompt)
    assert memories.tokens_seen == len(prompt)

    # ingest_hiddens additionally returns the hidden states and per-position
    # logits used for the next-token targets over the whole prompt

    memories, hiddens, logits = ingest_hiddens(wrapper, prompt)
    assert memories.tokens_seen == len(prompt)
    assert hiddens.shape == (1, len(prompt), 64)
    assert logits.shape == (1, len(prompt), 14)


def test_generate_answer_protocol(wrapper):
    # the answer is decoded autoregressively to exactly the length determined
    # by the query input grid, stopping early on the terminator

    task = TASKS["copy"]().generate(seed = 0)
    length = answer_length(task)

    tokens = generate_answer(wrapper, task, 4)
    assert isinstance(tokens, list)
    assert len(tokens) <= length
    assert all(0 <= token < 14 for token in tokens)

    # sampling with temperature, and the same memory reused to sweep
    # reasoning efforts without re-reading the demos

    tokens = generate_answer(wrapper, task, 2, temperature = 1.)
    assert isinstance(tokens, list)

    memories = ingest(wrapper, task_prompt(task))
    tokens_low_effort = generate_answer(wrapper, task, 1, memories = memories)
    tokens_high_effort = generate_answer(wrapper, task, 8, memories = memories)

    assert len(tokens_low_effort) <= length
    assert len(tokens_high_effort) <= length


def test_solve_returns_grid(wrapper):
    task = TASKS["propagation"]().generate(seed = 0)
    predicted = solve(wrapper, task, 4)

    assert predicted.ndim == 2
    assert predicted.dtype == np.int64


def test_train_loss(wrapper):
    # one full training step: prompt next-token targets, plus the latent
    # steps each predicting the first answer token, plus the answer

    task = TASKS["order"]().generate(seed = 0)

    loss = train_loss(wrapper, task, 4, class_weights = CLASS_WEIGHTS)
    loss.backward()

    assert loss.ndim == 0
    assert loss.item() >= 0
    assert all(
        param.grad is not None
        for param in wrapper.parameters()
        if param.requires_grad
    )


def test_train_loss_stages():
    # any interleaving of prompt, latent, prompt, latent, answer is trainable
    # through the wrapper's forward (stages passed as one list)

    wrapper = BDHReasoningWrapper(make_model(**TINY))

    first_prompt = torch.randint(0, 14, (1, 10))
    second_prompt = torch.randint(0, 14, (1, 15))
    answer = torch.randint(0, 14, (1, 20))

    loss, logits, memories = wrapper(
        first_prompt, 2, second_prompt, 4, answer,
        return_loss = True, return_memory = True
    )

    assert logits.shape == (1, 20, 14)
    assert memories.tokens_seen == 10 + 2 + 15 + 4 + 20

    loss.backward()
