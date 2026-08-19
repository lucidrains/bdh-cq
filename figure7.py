"""Replicate the main result of the BDH-CQ paper (figure 7 / table 5):
training with the latent-effort schedule of section 7, then sweeping the
number of latent reasoning steps at inference and watching accuracy climb.

Usage:

    python figure7.py
    python figure7.py --device mps --steps 1600
"""

import random

import fire
import torch

from bdh_cq.bdh_cq import BDHReasoningWrapper
from bdh_cq.icq import (make_model, train_loss, task_at_level, task_prompt,
                        ingest, generate_answer, decode_grid, cell_stats,
                        CLASS_WEIGHTS)
from bdh_cq.tasks import TASKS

# the reasoning efforts swept at inference (figure 7)

REASONING_STEPS_SWEEP = [1, 2, 4, 6, 8]

# training draws the latent effort uniformly over this range (section 7),
# so every inference effort is in distribution

MAX_REASONING_STEPS = 8

# small-scale regime - the working replication's scale (small grids, dense
# color content, mild extrapolation), so the whole demo runs in minutes

SIZES = {
    "propagation": 5,
    "copy": 2,
    "order": 4,
    "nesting": 3
}

LEVELS = {
    "propagation": [2, 3, 4],
    "copy": [2, 3],
    "order": [3, 4],
    "nesting": [2, 3]
}


def run(
    device: str = "cpu",
    family: str = "order",
    steps: int = 800,
    seed: int = 3
):
    torch.manual_seed(seed)
    random.seed(seed)

    if device == "cpu":
        torch.set_num_threads(8)

    # model, with the attention residual as the depth-residual readout
    # (section 3.3), the latent trajectory aware of its distance from
    # the end of reasoning

    wrapper = BDHReasoningWrapper(make_model(
        dim = 256,
        depth = 4,
        dim_qk_heads = 1024,
        attn_residual = True,
        attn_residual_depth_bias_distance = 1
    )).to(device)

    # train with the latent effort drawn uniformly over 0..MAX_REASONING_STEPS
    # each step, so every reasoning effort is in distribution at inference

    opt = torch.optim.AdamW(wrapper.parameters(), lr = 1e-3, weight_decay = 0.1)
    rng = random.Random(seed)

    for step in range(steps):
        task_family = rng.choice(list(TASKS.values()))
        task = task_family(
            size = SIZES[task_family.name]
        ).generate(seed = rng.randrange(2 ** 31))

        reasoning_steps = rng.randint(0, MAX_REASONING_STEPS)

        loss = train_loss(wrapper, task, reasoning_steps, class_weights = CLASS_WEIGHTS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)

        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    wrapper.eval()

    # sweep the number of reasoning steps on held-out tasks, all above
    # the difficulty levels of the demonstrations

    task_family = TASKS[family]

    with torch.no_grad():
        exact_match = {steps: 0 for steps in REASONING_STEPS_SWEEP}
        correct_cells = {steps: 0 for steps in REASONING_STEPS_SWEEP}
        total_cells = {steps: 0 for steps in REASONING_STEPS_SWEEP}
        num_outputs = 0

        for level in LEVELS[family]:
            for task_index in range(4):
                task = task_at_level(
                    task_family,
                    seed = 1_000_000 + task_index * 10_000 + level,
                    level = level,
                    n_tests = 2,
                    size = SIZES[family]
                )

                memories = ingest(wrapper, task_prompt(task))

                for _, _, target in task["test"]:
                    num_outputs += 1

                    for reasoning_steps in REASONING_STEPS_SWEEP:
                        predicted = generate_answer(
                            wrapper,
                            task,
                            reasoning_steps,
                            memories = memories
                        )
                        predicted = decode_grid(predicted)

                        exact_match[reasoning_steps] += (
                            predicted.shape == target.shape and
                            bool((predicted == target).all())
                        )
                        correct, total, _ = cell_stats(predicted, target)
                        correct_cells[reasoning_steps] += correct
                        total_cells[reasoning_steps] += total

    # figure 7: more reasoning steps, more accuracy

    print()
    print(f"== {family}, {num_outputs} held-out outputs at each reasoning step")

    print(f"{'steps':<7}" + "".join(
        f"{reasoning_steps:>8}" for reasoning_steps in REASONING_STEPS_SWEEP
    ))

    print(f"{'exact':<7}" + "".join(
        f"{exact_match[reasoning_steps]}/{num_outputs:<6}"
        for reasoning_steps in REASONING_STEPS_SWEEP
    ))

    print(f"{'cells':<7}" + "".join(
        f"{correct_cells[reasoning_steps] / max(1, total_cells[reasoning_steps]) * 100:7.1f}%"
        for reasoning_steps in REASONING_STEPS_SWEEP
    ))

    cell_accuracy = [
        correct_cells[reasoning_steps] / max(1, total_cells[reasoning_steps])
        for reasoning_steps in REASONING_STEPS_SWEEP
    ]

    print("monotone in R:", all(b >= a for a, b in zip(cell_accuracy, cell_accuracy[1:])))


if __name__ == "__main__":
    fire.Fire(run)
