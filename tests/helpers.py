"""Shared helpers for the test suite (plain functions; fixtures live in ``conftest.py``)."""

from __future__ import annotations

import pytest
import torch

import hiepy
from hiepy import HTTensor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def make_dense_htt(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, HTTensor]:
    """Return a dense random tensor and its exact HT representation."""
    gen = torch.Generator().manual_seed(seed)
    dense = torch.randn(shape, generator=gen, dtype=torch.double)
    return dense, hiepy.to_htt(dense, {"atol": 1e-13})


def assert_valid_structure(x: HTTensor) -> None:
    """Check that the HT components are consistent with the dimension tree."""
    leaves = x.dtree.get_leaves()
    assert sorted(leaves) == [(i,) for i in range(x.order)]
    assert set(x.U) == set(leaves)
    assert set(x.B) == set(x.dtree.get_inner_nodes())
    for t in x.dtree.get_inner_nodes():
        left, right = x.dtree.get_children(t)
        r_left = x.U[left].shape[1] if x.dtree.is_leaf(left) else x.B[left].shape[2]
        r_right = x.U[right].shape[1] if x.dtree.is_leaf(right) else x.B[right].shape[2]
        assert x.B[t].shape[:2] == (r_left, r_right)
    assert x.B[x.dtree.get_root()].shape[2] == 1
