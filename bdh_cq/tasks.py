"""ARC-style task families from the BDH-CQ paper (sec 6.2, fig 5):
propagation, copy, order, nesting.

Each family samples a fixed layout once, then renders (input, output)
grids at a difficulty level. Demos draw from an easy range, held-out
inputs a harder one, so the same generator covers training and
extrapolation. Deterministic given a seed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator

# constants

GRAY = 5
COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def randint(rng: Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))

def make_grid(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), np.int64)

def draw_motif(grid: np.ndarray, pos: tuple[int, int], motif: np.ndarray) -> None:
    i, j = pos
    grid[i:i + motif.shape[0], j:j + motif.shape[1]] = motif

def draw_border(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    grid[r, c:c + w] = color
    grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = color
    grid[r:r + h, c + w - 1] = color

# base task

class Task(ABC):
    """task dicts hold {"name", "params", "train", "test"}, where examples
    are (level, input, output) grid tuples"""

    name: str
    demo_levels: tuple[int, int]
    test_levels: tuple[int, int]

    def __init__(self, size: int | None = None):
        self.size = size

    @abstractmethod
    def sample(self, rng: Generator) -> dict[str, Any]:
        ...

    @abstractmethod
    def render(self, rng: Generator, level: int, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ...

    def generate(
        self,
        seed: int = 0,
        n_demos: int = 3,
        n_tests: int = 2,
        test_levels: tuple[int, int] | None = None
    ) -> Task:
        rng = np.random.default_rng(seed)
        params = self.sample(rng)

        train = []
        for _ in range(n_demos):
            level = randint(rng, *self.demo_levels)
            train.append((level, *self.render(rng, level, params)))

        test = []
        for _ in range(n_tests):
            level = randint(rng, *(test_levels or self.test_levels))
            test.append((level, *self.render(rng, level, params)))

        return dict(name = self.name, params = params, train = train, test = test)

# task families

class Propagation(Task):
    """seed bar extends to the right boundary, level = gap to the edge"""

    name = 'propagation'
    demo_levels = (1, 3)
    test_levels = (2, 8)

    def __init__(self, size: int | None = None):
        super().__init__(size)

        if exists(size):
            self.demo_levels = (1, 2)
            self.test_levels = (2, 4)

    def sample(self, rng):
        if self.size:
            height = width = self.size
            h = randint(rng, 2, min(3, height - 1))
        else:
            height = randint(rng, 6, 10)
            width = randint(rng, 10, 12)
            h = randint(rng, 2, min(4, height - 1))

        return dict(
            height = height,
            width = width,
            h = h,
            r0 = randint(rng, 0, height - h),
            color = int(rng.choice(COLORS))
        )

    def render(self, rng, gap, params):
        height, width, h, r0, color = params.values()
        rows = slice(r0, r0 + h)
        col = width - 1 - gap

        inp = make_grid(height, width)
        inp[rows, col] = color

        out = inp.copy()
        out[rows, col:] = color

        return inp, out

class Copy(Task):
    """motif copied to every gray anchor, level = number of anchors"""

    name = 'copy'
    demo_levels = (1, 2)
    test_levels = (2, 4)

    def __init__(self, size: int | None = None):
        super().__init__(size)

        if exists(size):
            self.demo_levels = (1, 2)
            self.test_levels = (2, 3)

    def sample(self, rng):
        if self.size:
            k, ni, nj = 2, 3, 3
        else:
            k = int(rng.choice([2, 3]))
            ni, nj = randint(rng, 3, 4), randint(rng, 3, 4)

        motif = np.zeros((k, k), np.int64)
        n = randint(rng, 2, min(k * k, len(COLORS)))
        for cell, color in zip(rng.choice(k * k, n, replace = False),
                               rng.choice(COLORS, n, replace = False)):
            motif.flat[cell] = color

        return dict(
            k = k,
            h = k * ni,
            w = k * nj,
            ni = ni,
            nj = nj,
            motif = motif,
            src = (int(rng.integers(0, ni)), int(rng.integers(0, nj)))
        )

    def render(self, rng, n, params):
        k, h, w, ni, nj, motif, src = params.values()
        spots = [(i, j) for i in range(ni) for j in range(nj)]
        spots.remove(src)
        rng.shuffle(spots)
        anchors = spots[:n]

        inp = make_grid(h, w)
        for i, j in anchors:
            inp[i * k, j * k] = GRAY
        draw_motif(inp, src, motif)

        out = inp.copy()
        for i, j in anchors:
            draw_motif(out, (i * k, j * k), motif)

        return inp, out

class Order(Task):
    """bars reordered by height, shortest to tallest, level = number of bars"""

    name = 'order'
    demo_levels = (2, 4)
    test_levels = (5, 8)

    def __init__(self, size: int | None = None):
        super().__init__(size)

        self.max_n = default(size, 8)
        self.width = 2 * self.max_n + 2

        if exists(size):
            self.demo_levels = (2, 3)
            self.test_levels = (3, 4)

    def sample(self, rng):
        height = randint(rng, 8, 9) if self.size else randint(rng, 10, 12)
        return dict(height = height)

    def render(self, rng, n, params):
        height = params['height']
        heights = rng.permutation(n) + 1
        colors = rng.choice(COLORS, n, replace = False)
        cols = rng.choice(range(1, self.width - 1, 2), n, replace = False)

        inp = make_grid(height, self.width)
        for h, color, col in zip(heights, colors, cols):
            inp[height - h:, col] = color

        out = make_grid(height, self.width)
        for col, idx in enumerate(np.argsort(heights)):
            out[height - heights[idx]:, col] = colors[idx]

        return inp, out

class Nesting(Task):
    """innermost region recolored, level = number of nested frames"""

    name = 'nesting'
    demo_levels = (1, 3)
    test_levels = (4, 5)

    def __init__(self, size: int | None = None):
        super().__init__(size)

        self.max_depth = default(size, 5)

        if exists(size):
            self.demo_levels = (1, 2)
            self.test_levels = (2, 3)

    def sample(self, rng):
        colors = rng.choice(COLORS, self.max_depth + 1, replace = False)
        return dict(
            s = randint(rng, 1, 3),
            t = randint(rng, 1, 3),
            region_color = int(colors[0]),
            frames = [int(c) for c in colors[1:]]
        )

    def render(self, rng, depth, params):
        s, t, region_color, frames = params.values()
        pad = 2 * self.max_depth

        inp = make_grid(t + 2 * pad, s + 2 * pad)
        for d in range(1, depth + 1):
            draw_border(inp, pad - 2 * d, pad - 2 * d, t + 4 * d, s + 4 * d, frames[d - 1])

        out = inp.copy()
        out[pad:pad + t, pad:pad + s] = region_color

        return inp, out

# registry

TASKS = dict(
    propagation = Propagation,
    copy = Copy,
    order = Order,
    nesting = Nesting
)

def show_grid(grid: np.ndarray) -> str:
    """grid as ascii, '.' for black, for quick eyeballing"""
    return '\n'.join(''.join(str(v) if v else '.' for v in row) for row in grid)

if __name__ == '__main__':
    for task in [cls().generate(seed = 0) for cls in TASKS.values()]:
        levels = [level for level, _, _ in task['train']]
        print(f'== {task["name"]}: demo levels {levels}')
        _, inp, out = task['test'][0]
        print(show_grid(inp), '->', show_grid(out), sep = '\n')
