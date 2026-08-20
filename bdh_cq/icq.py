"""In-context learning for the ARC-style task families in tasks.py.

Grids serialize over the color palette (tokens 0-9) with row (10), input
(11), output (12) and end-of-output (13) markers. The prompt ingests into
the BDH memory in chunks, the query iterates `steps` latent reasoning
steps, and the answer grid decodes autoregressively, seeded from the last
latent step's projection - the same protocol at training and inference,
so the only free parameter at inference is the number of latent steps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from bdh_cq.bdh_cq import BDH, BDHReasoningWrapper, Memory, exists
from bdh_cq.tasks import Task

# tokens

NUM_TOKENS = 14
ROW, IN, OUT, EOS = 10, 11, 12, 13

# model

MODEL_KWARGS = dict(
    dim = 384,
    depth = 4,
    heads = 4,
    dim_qk_heads = 2048
)

CHUNK_SIZE = 128

# rare tokens carry the answer structure; weight them up so the model
# does not collapse into predicting background zeros

CLASS_WEIGHTS = torch.tensor([0.5] + [3.0] * 9 + [2.0, 1.0, 2.0, 3.0])


def device_of(module) -> torch.device:
    return next(module.parameters()).device


def make_model(**overrides) -> BDH:
    return BDH(num_tokens = NUM_TOKENS, **{**MODEL_KWARGS, **overrides})


# serialization

def encode_grid(grid: np.ndarray, marker: int = IN) -> list[int]:
    # marker, rows joined by the row separator

    tokens = [marker]
    for row_index, row in enumerate(grid):
        if row_index:
            tokens.append(ROW)
        tokens.extend(int(value) for value in row)
    return tokens


def encode_output(grid: np.ndarray) -> list[int]:
    return encode_grid(grid, marker = OUT) + [EOS]


def decode_grid(tokens: list[int]) -> np.ndarray:
    # generated tokens back to a grid: stop at <eos>, skip markers,
    # ragged rows padded with zeros

    grid, row = [], []
    for token in tokens:
        if token == EOS:
            break
        if 0 <= token < 10:
            row.append(token)
        elif token == ROW:
            if row or grid:
                grid.append(row)
            row = []
    if row or not grid:
        grid.append(row)
    width = max((len(r) for r in grid), default = 0)
    grid = [r + [0] * (width - len(r)) for r in grid]
    return np.array(grid, dtype = np.int64)


# prompt

def task_prompt(task: Task, n_demos: int = 3) -> list[int]:
    # demos (input, output) pairs then the query input grid

    tokens: list[int] = []
    for _, input_grid, output_grid in task["train"][:n_demos]:
        tokens += encode_grid(input_grid) + encode_output(output_grid)
    query_input = task["test"][0][1]
    tokens += encode_grid(query_input)
    return tokens


def task_answer(task: Task) -> list[int]:
    return encode_output(task["test"][0][2])


def answer_length(task: Task) -> int:
    # all four families keep the query input's shape, so the answer
    # length is fully determined by the query input grid

    return len(encode_output(np.zeros_like(task["test"][0][1])))


def task_at_level(
    cls,
    seed: int,
    level: int,
    n_demos: int = 3,
    n_tests: int = 1,
    size: int | None = None
) -> Task:
    # demos from the easy range, query at exactly `level`

    task = cls(size = size) if size is not None else cls()
    return task.generate(seed, n_demos, n_tests, test_levels = (level, level))


# ingest

def _chunks(ids: list[int], chunk_size: int):
    for index in range(0, len(ids), chunk_size):
        yield torch.tensor(ids[index:index + chunk_size], dtype = torch.long).unsqueeze(0)


def ingest(
    wrapper: BDHReasoningWrapper,
    ids: list[int],
    memories: Memory | None = None,
    chunk_size: int = CHUNK_SIZE,
    update_memory: bool = True
) -> Memory:
    # run a token sequence through the model in chunks, carrying memory;
    # update_memory = False ablates the query's conditioning on the prompt (eq. 1, 2)

    device = device_of(wrapper)
    for chunk in _chunks(ids, chunk_size):
        _, memories = wrapper(chunk.to(device), memories = memories, update_memory = update_memory, return_memory = True)
    return memories


def ingest_hiddens(
    wrapper: BDHReasoningWrapper,
    ids: list[int],
    chunk_size: int = CHUNK_SIZE,
    update_memory: bool = True
) -> tuple[Memory, Tensor, Tensor]:
    # ingest in chunks, collecting hiddens and per-chunk logits for the
    # next-token targets over the whole prompt

    device = device_of(wrapper)
    memories, hiddens, logits = None, [], []

    for chunk in _chunks(ids, chunk_size):
        chunk_logits, memories = wrapper(chunk.to(device), memories = memories, update_memory = update_memory, return_memory = True)
        hiddens.append(memories.embeds)
        logits.append(chunk_logits)

    return memories, torch.cat(hiddens, dim = 1), torch.cat(logits, dim = 1)


# training

def train_loss(
    wrapper: BDHReasoningWrapper,
    task: Task,
    steps: int,
    class_weights: Tensor | None = None,
    update_memory: bool = True,
    update_latent_memory: bool = True
) -> Tensor:
    # prompt next-token targets (demos included), plus the wrapper loss:
    # every latent step predicts the first answer token, every answer
    # position the next answer token

    device = device_of(wrapper)
    prompt_ids = task_prompt(task)
    answer = task_answer(task)

    memories, _, prompt_logits = ingest_hiddens(wrapper, prompt_ids, update_memory = update_memory)

    # next-token targets over the prompt, ending with <out>

    prompt_targets = torch.tensor(prompt_ids[1:] + [OUT], dtype = torch.long, device = device)

    # latent reasoning, then the answer teacher-forced as the next segment

    answer_tokens = torch.tensor(answer, dtype = torch.long, device = device).unsqueeze(0)
    weight = class_weights.to(device) if exists(class_weights) else None

    loss, _, _ = wrapper(
        steps,
        answer_tokens,
        memories = memories,
        return_loss = True,
        return_memory = True,
        weight = weight,
        update_latent_memory = update_latent_memory
    )

    mask = prompt_targets != IN
    return F.cross_entropy(
        prompt_logits[0][mask],
        prompt_targets[mask],
        weight = weight
    ) + loss


# inference

@torch.no_grad()
def generate_answer(
    wrapper: BDHReasoningWrapper,
    task: Task,
    steps: int,
    memories: Memory | None = None,
    update_memory: bool = True,
    update_latent_memory: bool = True,
    temperature: float = 0.
) -> list[int]:
    # decode the query output at the given latent effort, seeded from the
    # last latent step's projection, generated to exactly the answer
    # length; pass a pre-ingested memory to sweep efforts without
    # re-reading the demos

    if memories is None:
        memories = ingest(wrapper, task_prompt(task))

    length = answer_length(task)

    tokens = wrapper.generate(
        steps,
        memories = memories,
        num_tokens = length,
        stop_token = EOS,
        temperature = temperature,
        update_memory = update_memory,
        update_latent_memory = update_latent_memory
    )

    # the final position is the terminator; strip the model's guess when
    # it never emitted <eos>, so the grid still decodes to the expected shape

    if tokens and tokens[-1] != EOS:
        tokens = tokens[:-1]
    return tokens


@torch.no_grad()
def solve(
    wrapper: BDHReasoningWrapper,
    task: Task,
    steps: int,
    memories: Memory | None = None
) -> np.ndarray:
    # predicted query output grid at the given latent effort

    return decode_grid(generate_answer(wrapper, task, steps, memories = memories))


def cell_stats(pred: np.ndarray, target: np.ndarray):
    # (correct cells, total cells, dimensions valid)

    if pred.shape != target.shape:
        return 0, 0, False
    total = target.size
    correct = int((pred == target).sum())
    return correct, total, True
