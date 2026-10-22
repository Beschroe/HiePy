"""Shared helpers for the test suite (plain functions; fixtures live in ``conftest.py``).

Numerical tolerances live in ``tests/tolerances.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import hiepy
from hiepy import DimTree, HTTensor
from tests.tolerances import assert_orthonormal

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def make_dense_htt(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, HTTensor]:
    """Return a dense random tensor and its exact HT representation."""
    gen = torch.Generator().manual_seed(seed)
    dense = torch.randn(shape, generator=gen, dtype=torch.double)
    return dense, hiepy.to_htt(dense, {"atol": 0.0})


def components(x: HTTensor) -> list[torch.Tensor]:
    """Return all leaf matrices and transfer tensors of ``x``."""
    return [*x.U.values(), *x.B.values()]


def assert_valid_structure(x: HTTensor) -> None:
    """S1, S2, D1: components match the dimension tree, ranks are consistent, one dtype and device."""
    leaves = x.dtree.get_leaves()
    assert sorted(leaves) == [(i,) for i in range(x.order)]
    assert set(x.U) == set(leaves)
    assert set(x.B) == set(x.dtree.get_inner_nodes())
    for t in x.dtree.get_inner_nodes():
        left, right = x.dtree.get_children(t)
        r_left = x.U[left].shape[1] if x.dtree.is_leaf(left) else x.B[left].shape[2]
        r_right = x.U[right].shape[1] if x.dtree.is_leaf(right) else x.B[right].shape[2]
        assert x.B[t].ndim == 3
        assert x.B[t].shape[:2] == (r_left, r_right)
    assert x.B[x.dtree.get_root()].shape[2] == 1
    assert len({(c.dtype, c.device) for c in components(x)}) == 1


def assert_orthogonal(x: HTTensor, dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    """O1: the frame of every non-root node is orthonormal, checked per component in the precision ``dtypes``."""
    for t in x.dtree.get_nodes():
        if x.dtree.is_root(t):
            continue
        F = x.U[t] if x.dtree.is_leaf(t) else x.B[t].reshape(-1, x.B[t].shape[2])
        assert_orthonormal(F, f"frame of node {t}", dtypes)


@dataclass(frozen=True)
class Snapshot:
    """Deep copy of the state of an HT tensor, used to check that inputs are not modified (V1)."""

    U: dict[hiepy.Node, torch.Tensor]
    B: dict[hiepy.Node, torch.Tensor]
    dtree: DimTree
    is_orthog: bool


def snapshot(x: HTTensor) -> Snapshot:
    """Return a detached deep copy of components, tree and flag of ``x``."""
    U = {t: u.detach().clone() for t, u in x.U.items()}
    B = {t: b.detach().clone() for t, b in x.B.items()}
    return Snapshot(U=U, B=B, dtree=x.dtree.copy(), is_orthog=x.is_orthog)


def assert_unchanged(x: HTTensor, before: Snapshot) -> None:
    """V1: components, tree and flag of ``x`` equal the snapshot taken before an operation."""
    assert x.dtree == before.dtree
    assert x.is_orthog == before.is_orthog
    assert set(x.U) == set(before.U) and set(x.B) == set(before.B)
    assert all(torch.equal(x.U[t], before.U[t]) for t in before.U)
    assert all(torch.equal(x.B[t], before.B[t]) for t in before.B)


def storages(x: HTTensor) -> set[int]:
    """Return the storage addresses of all components of ``x`` (V2)."""
    return {c.untyped_storage().data_ptr() for c in components(x)}
