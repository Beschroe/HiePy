"""Round-trip invariants N3 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import DimTree, HTTensor
from tests.helpers import assert_close, assert_orthogonal, assert_valid_structure
from tests.strategies import ht_tensors, shapes


@st.composite
def _dense(draw: st.DrawFn, min_order: int = 2) -> torch.Tensor:
    """A random dense float64 tensor, or the full form of a general HT tensor (low-rank in some nodes, zero, ...)."""
    if draw(st.booleans()):
        return draw(ht_tensors()).full()
    shape = draw(shapes(min_order=min_order))
    gen = torch.Generator().manual_seed(draw(st.integers(0, 2**31 - 1)))
    return torch.randn(shape, generator=gen, dtype=torch.float64)


@given(T=_dense())
def test_to_htt_without_truncation_is_exact(T: torch.Tensor):
    """N3: ``to_htt(T, {"atol": 0})`` reproduces ``T`` (and is a valid, orthogonal HT tensor, S5, O1)."""
    x = hiepy.to_htt(T, {"atol": 0.0})
    assert_valid_structure(x)
    assert x.dtree == DimTree.from_order(T.ndim)
    assert x.is_orthog
    assert_orthogonal(x)
    assert_close(x.full(), T)


@given(x=ht_tensors())
def test_save_load_round_trip(x: HTTensor):
    """N3: ``load(save(x))`` reproduces components, tree and flag exactly (also for non-canonical trees)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.htt"
        hiepy.save(x, path)
        y = hiepy.load(path)
    assert y.dtree == x.dtree
    assert y.is_orthog == x.is_orthog
    assert set(y.U) == set(x.U) and set(y.B) == set(x.B)
    assert all(torch.equal(y.U[t], x.U[t]) for t in x.U)
    assert all(torch.equal(y.B[t], x.B[t]) for t in x.B)


@st.composite
def _modes(draw: st.DrawFn, order: int) -> tuple[int, ...]:
    """A non-empty sequence of distinct modes in any order, each given positively or negatively."""
    modes = draw(st.permutations(range(order)))[: draw(st.integers(1, order))]
    return tuple(m - order if draw(st.booleans()) else m for m in modes)


@given(T=_dense(min_order=1), data=st.data())
def test_matricize_round_trip(T: torch.Tensor, data: st.DataObject):
    """N3: ``matricize`` unfolds in C order with the modes ``t`` as rows, and ``dematricize`` inverts it."""
    t = data.draw(_modes(T.ndim), label="t")
    t_pos = [m % T.ndim for m in t]
    rest = [m for m in range(T.ndim) if m not in t_pos]
    A = hiepy.matricize(T, t)
    assert torch.equal(A, T.permute(*t_pos, *rest).reshape(A.shape[0], -1))
    assert torch.equal(hiepy.dematricize(A, tuple(T.shape), t), T)


@given(T=_dense(min_order=1))
def test_vectorize_round_trip(T: torch.Tensor):
    """N3: ``vectorize`` flattens in C order and ``devectorize`` inverts it."""
    v = hiepy.vectorize(T)
    assert torch.equal(v, T.reshape(-1))
    assert torch.equal(hiepy.devectorize(v, tuple(T.shape)), T)
