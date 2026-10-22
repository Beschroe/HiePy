"""Numerical invariants N1 (factories), N7, N8 and N9 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

from math import prod

import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from tests.strategies import ht_tensors, shapes
from tests.tolerances import assert_exact, assert_orthonormal, assert_within, magnitude

# ~~~~~~~~~~~~~~~~~~~~~~~~~ N1: exact factories ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(shape=shapes(), c=st.sampled_from([0.0, -1.5, 2.0, 1e-3]), orthog=st.booleans())
def test_constant_factories_are_exact(shape: tuple[int, ...], c: float, orthog: bool):
    """N1: ``zeros``, ``ones`` and ``constant`` (orthogonal or not) equal their dense torch counterparts."""
    kw = dict(dtype=torch.float64)
    assert torch.equal(hiepy.zeros(shape, **kw).full(), torch.zeros(shape, **kw))  # type: ignore[arg-type]
    assert_exact(hiepy.ones(shape, orthog=orthog, **kw).full(), torch.ones(shape, **kw))  # type: ignore[arg-type]
    assert_exact(hiepy.constant(shape, c, orthog=orthog, **kw).full(), torch.full(shape, c, **kw))  # type: ignore[arg-type]


@given(shape=shapes())
def test_arange_is_exact(shape: tuple[int, ...]):
    """N1: ``arange`` equals ``torch.arange`` reshaped in C order."""
    expected = torch.arange(prod(shape), dtype=torch.float64).reshape(shape)
    assert_exact(hiepy.arange(shape, dtype=torch.float64).full(), expected)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N7: orthog does not increase ranks ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(x=ht_tensors())
def test_orthog_does_not_increase_ranks(x: HTTensor):
    """N7: ``orthog`` never increases a rank (over-parameterized inputs are reduced)."""
    y = hiepy.orthog(x)
    assert all(y.rank[t] <= x.rank[t] for t in x.dtree.get_nodes())


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N8: randn and hilbert ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(shape=shapes(), rank=st.integers(1, 5), seed=st.integers(0, 2**31 - 1))
def test_randn_respects_rank(shape: tuple[int, ...], rank: int, seed: int):
    """N8: ``randn`` has ranks ``≤ rank`` at every non-root node."""
    x = hiepy.randn(shape, rank=rank, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    assert all(r <= rank for t, r in x.rank.items() if not x.dtree.is_root(t))


def _hilbert_dense(shape: tuple[int, ...]) -> torch.Tensor:
    """Entries ``1 / (i_0 + … + i_{d−1} + d)`` for zero-based indices."""
    grids = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij")
    return 1.0 / (sum(grids) + len(shape))


@given(shape=shapes(max_order=4))
def test_hilbert_improves_with_quadrature_points(shape: tuple[int, ...]):
    """N8: ``hilbert`` approximates the Hilbert tensor with a smaller error for ``M = 10`` than for ``M = 3``."""
    H = _hilbert_dense(shape)
    errors = [torch.linalg.norm(hiepy.hilbert(shape, M=M, dtype=torch.float64).full() - H).item() for M in (3, 10)]
    assert errors[1] < errors[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N9: local SVD and Gramians ~~~~~~~~~~~~~~~~~~~~~~~~~


def _frame(x: HTTensor, t: hiepy.Node) -> tuple[torch.Tensor, list[int]]:
    """Frame matrix of node ``t`` and the modes of its rows (left subtree first, C order)."""
    if x.dtree.is_leaf(t):
        return x.U[t], [t[0]]
    left, right = x.dtree.get_children(t)
    F_l, modes_l = _frame(x, left)
    F_r, modes_r = _frame(x, right)
    F = torch.einsum("ia,jb,abk->ijk", F_l, F_r, x.B[t]).reshape(F_l.shape[0] * F_r.shape[0], -1)
    return F, modes_l + modes_r


def _padded(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad two vectors of singular values with zeros to the same length."""
    n = max(len(a), len(b))
    return torch.nn.functional.pad(a, (0, n - len(a))), torch.nn.functional.pad(b, (0, n - len(b)))


@given(x=ht_tensors())
def test_local_left_svd_matches_dense_svd(x: HTTensor):
    """N9: per node, the local vectors are orthonormal, the singular values are those of the matricization, and the
    global vectors ``U_t Û_t`` (non-root nodes) span its column space."""
    X = x.full()
    m = magnitude(x)
    xo = hiepy.orthog(x)
    for t, (U_loc, S) in hiepy.local_left_svd(x).items():
        assert_orthonormal(U_loc, f"local vectors of node {t}")
        F, modes = _frame(xo, t)
        X_t = hiepy.matricize(X, modes)
        assert_exact(*_padded(S, torch.linalg.svdvals(X_t)), scale=m, scale_label="m(x)")
        if xo.dtree.is_root(t):
            continue  # the root frame is the tensor itself, not an orthonormal basis
        W = F @ U_loc
        assert_within(torch.linalg.norm(X_t - W @ (W.T @ X_t)).item(), 0.0, scale=m)


@given(x=ht_tensors())
def test_gramian_eigenvalues_are_squared_singular_values(x: HTTensor):
    """N9: the eigenvalues of ``gramians(orthog(x))[t]`` are the squared singular values of the matricization."""
    X = x.full()
    m = magnitude(x)
    xo = hiepy.orthog(x)
    for t, G in hiepy.gramians(xo).items():
        if xo.dtree.is_root(t):
            continue
        _, modes = _frame(xo, t)
        eigenvalues = torch.linalg.eigvalsh(G).flip(0)
        squared = torch.linalg.svdvals(hiepy.matricize(X, modes)) ** 2
        assert_exact(*_padded(eigenvalues, squared), scale=m**2, scale_label="m(x)²")
