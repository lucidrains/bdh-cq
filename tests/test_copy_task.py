import numpy as np

from bdh_cq.tasks import Copy, GRAY, draw_motif


def solve_copy_from_input(inp, k, ni, nj):
    # the source motif is the one block holding color and no gray - paste it on every gray anchor

    blocks = {
        (i, j): inp[i * k:(i + 1) * k, j * k:(j + 1) * k]
        for i in range(ni)
        for j in range(nj)
    }

    sources = [
        block for block in blocks.values()
        if (block != 0).any() and not (block == GRAY).any()
    ]

    assert len(sources) == 1
    motif = sources[0]

    out = inp.copy()

    for i, j in blocks:
        if inp[i * k, j * k] == GRAY:
            draw_motif(out, (i * k, j * k), motif)

    return out


def test_copy_output_follows_from_input():
    # every rendered output is recoverable from its input alone

    for size in (None, 2):
        for seed in range(200):
            task = Copy(size = size).generate(seed = seed)
            params = task["params"]

            for _, inp, out in task["train"] + task["test"]:
                assert np.array_equal(solve_copy_from_input(inp, params["k"], params["ni"], params["nj"]), out)
