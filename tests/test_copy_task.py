import numpy as np

from bdh_cq.tasks import Copy, GRAY, draw_motif


def solve_from_input(inp, k, ni, nj):
    # the one lattice block holding colour (and no gray) is the motif - paste it on every gray anchor
    blocks = [(i, j) for i in range(ni) for j in range(nj)
              if (inp[i * k:(i + 1) * k, j * k:(j + 1) * k] != 0).any()
              and not (inp[i * k:(i + 1) * k, j * k:(j + 1) * k] == GRAY).any()]
    assert len(blocks) == 1
    si, sj = blocks[0]
    motif = inp[si * k:(si + 1) * k, sj * k:(sj + 1) * k]
    out = inp.copy()
    for i in range(ni):
        for j in range(nj):
            if inp[i * k, j * k] == GRAY:
                draw_motif(out, (i * k, j * k), motif)
    return out


def test_copy_output_follows_from_input():
    for size in (None, 2):
        for seed in range(200):
            task = Copy(size = size).generate(seed = seed)
            p = task['params']
            for _, inp, out in task['train'] + task['test']:
                assert np.array_equal(solve_from_input(inp, p['k'], p['ni'], p['nj']), out)
