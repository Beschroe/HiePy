"""hypothesis strategies for HT tensors.

Tensors are built from a seed drawn by hypothesis, so every example is reproducible from the printed seed and shape.
All tensors are float64.
"""

from __future__ import annotations

import torch
from hypothesis import note
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor

MAX_ORDER = 6
MAX_SIZE = 5
MAX_RANK = 3


@st.composite
def shapes(draw: st.DrawFn, min_order: int = 2, max_order: int = MAX_ORDER, max_size: int = MAX_SIZE) -> tuple:
    """Shapes of order ``min_order … max_order`` with mode sizes ``1 … max_size`` (singletons included)."""
    order = draw(st.integers(min_order, max_order))
    return tuple(draw(st.lists(st.integers(1, max_size), min_size=order, max_size=order)))


def _ht_tensor(shape: tuple[int, ...], seed: int, rank: int, orthog: bool) -> HTTensor:
    """Random HT tensor; non-orthogonal tensors are scaled along mode 0 by a positive vector."""
    gen = torch.Generator().manual_seed(seed)
    x = hiepy.randn(shape, rank=rank, dtype=torch.float64, generator=gen)
    if not orthog:
        v = torch.rand(shape[0], generator=gen, dtype=torch.float64) + 0.5
        x = hiepy.ele_mode_mul(x, v, 0)
    return x


@st.composite
def ht_tensors(
    draw: st.DrawFn, shape: tuple[int, ...] | None = None, orthog: bool | None = None, max_order: int = MAX_ORDER
) -> HTTensor:
    """Random float64 HT tensor with ranks ``1 … MAX_RANK``; orthogonal or not, as drawn unless fixed."""
    if shape is None:
        shape = draw(shapes(max_order=max_order))
    if orthog is None:
        orthog = draw(st.booleans())
    seed = draw(st.integers(0, 2**31 - 1))
    rank = draw(st.integers(1, MAX_RANK))
    note(f"ht_tensor(shape={shape}, seed={seed}, rank={rank}, orthog={orthog})")
    return _ht_tensor(shape, seed, rank, orthog)


@st.composite
def ht_pairs(draw: st.DrawFn, max_order: int = MAX_ORDER) -> tuple[HTTensor, HTTensor]:
    """Two random HT tensors of the same shape and tree, independently orthogonal or not."""
    shape = draw(shapes(max_order=max_order))
    return draw(ht_tensors(shape=shape)), draw(ht_tensors(shape=shape))


@st.composite
def decaying_dense(draw: st.DrawFn, max_order: int = MAX_ORDER, terms: int = 8) -> torch.Tensor:
    """Dense tensor whose hierarchical singular values decay: a sum of rank-1 terms weighted by ``decay**k``."""
    shape = draw(shapes(min_order=3, max_order=max_order))
    seed = draw(st.integers(0, 2**31 - 1))
    decay = draw(st.sampled_from([0.03, 0.1, 0.3]))
    gen = torch.Generator().manual_seed(seed)
    T = torch.zeros(shape, dtype=torch.float64)
    for k in range(terms):
        term = torch.randn(shape[0], generator=gen, dtype=torch.float64)
        for n in shape[1:]:
            term = torch.tensordot(term, torch.randn(n, generator=gen, dtype=torch.float64), dims=0)
        T += decay**k * term
    return T
