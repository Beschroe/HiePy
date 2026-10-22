"""Tests for ``full`` on arbitrary dimension trees.

``full`` used to assume that the rows of the root frame are ordered by mode, which only holds if every left child
contains smaller, contiguous modes than its right sibling. These trees are the regression cases.

Invariants covered (``tests/INVARIANTS.md``): N1 (``full`` is the reference of all exactness checks, Q8).
"""

from __future__ import annotations

import itertools

import pytest
import torch

from hiepy import DimTree, HTTensor
from tests.tolerances import assert_exact

SHAPE = (2, 3, 4, 2)


def _tree(children: dict[tuple[int, ...], tuple[tuple[int, ...], tuple[int, ...]]]) -> DimTree:
    return DimTree((0, 1, 2, 3), children)


TREES = {
    "canonical": {(0, 1, 2, 3): ((0, 1), (2, 3)), (0, 1): ((0,), (1,)), (2, 3): ((2,), (3,))},
    "non-contiguous": {(0, 1, 2, 3): ((0, 2), (1, 3)), (0, 2): ((0,), (2,)), (1, 3): ((1,), (3,))},
    "larger modes left": {(0, 1, 2, 3): ((2, 3), (0, 1)), (2, 3): ((3,), (2,)), (0, 1): ((1,), (0,))},
    "unbalanced": {(0, 1, 2, 3): ((0, 1, 3), (2,)), (0, 1, 3): ((3,), (0, 1)), (0, 1): ((0,), (1,))},
}


def _random_ht(dtree: DimTree, rank: int = 2) -> HTTensor:
    gen = torch.Generator().manual_seed(0)
    U = {t: torch.randn(SHAPE[t[0]], rank, generator=gen, dtype=torch.float64) for t in dtree.get_leaves()}
    B = {
        t: torch.randn(rank, rank, 1 if dtree.is_root(t) else rank, generator=gen, dtype=torch.float64)
        for t in dtree.get_inner_nodes()
    }
    return HTTensor(U=U, B=B, dtree=dtree)


def _entry(x: HTTensor, index: tuple[int, ...]) -> torch.Tensor:
    """Entry of ``x`` by contracting the tree for one multi-index, independently of ``full``."""

    def rec(t: tuple[int, ...]) -> torch.Tensor:
        if x.dtree.is_leaf(t):
            return x.U[t][index[t[0]]]
        left, right = x.dtree.get_children(t)
        return torch.einsum("a,b,abk->k", rec(left), rec(right), x.B[t])

    return rec(x.dtree.get_root())[0]


@pytest.mark.parametrize("name", TREES)
def test_full_matches_entrywise_contraction(name: str):
    """N1: ``full`` places every entry at its multi-index, for any binary tree."""
    x = _random_ht(_tree(TREES[name]))
    expected = torch.stack([_entry(x, idx) for idx in itertools.product(*map(range, SHAPE))]).reshape(SHAPE)
    assert_exact(x.full(), expected)
