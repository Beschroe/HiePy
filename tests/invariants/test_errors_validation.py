"""Argument validation E6 (empty slices), E9, E10 and E11 (see ``tests/INVARIANTS.md``, Q10, Q11)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from tests.ops import FACTORIES
from tests.strategies import ht_tensors

SHAPE = (3, 4, 2)


def _x(scale: float = 1.0) -> HTTensor:
    x = hiepy.randn(SHAPE, rank=2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    return hiepy.scalar_mul(x, scale)


TRUNCATING: dict[str, Callable[[HTTensor, hiepy.TruncOpts], object]] = {
    "to_htt": lambda x, opts: hiepy.to_htt(x.full(), opts),
    "truncate_svd": hiepy.truncate_svd,
    "truncate_gram": hiepy.truncate_gram,
    "truncate": lambda x, opts: hiepy.truncate(x, opts),
    "ele_mul": lambda x, opts: hiepy.ele_mul(x, x, opts),
}


def _truncate(name: str, x: HTTensor, opts: hiepy.TruncOpts) -> object:
    """Call a truncating routine; ``ele_mul`` warns about ``rtol`` (E8), which is expected here."""
    if name == "ele_mul" and "rtol" in opts:
        with pytest.warns(UserWarning, match="rtol"):
            return TRUNCATING[name](x, opts)
    return TRUNCATING[name](x, opts)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E6: empty slices ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(x=ht_tensors(), data=st.data())
def test_empty_slice_raises(x: HTTensor, data: st.DataObject):
    """E6: a slice that selects no entry of its mode raises ``ValueError``."""
    dim = data.draw(st.integers(0, x.order - 1), label="dim")
    n = x.shape[dim]
    start = data.draw(st.integers(-n, n + 2), label="start")
    stop = data.draw(st.integers(-n - 2, start % n if start < n else n), label="stop")
    empty = slice(start, stop)
    if len(range(*empty.indices(n))) > 0:
        return
    key = (slice(None),) * dim + (empty,)
    with pytest.raises(ValueError, match="selects no entry"):
        x[key]


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E9: truncation options ~~~~~~~~~~~~~~~~~~~~~~~~~

INVALID_OPTS = [
    ({"tol": 1e-3}, ValueError, "Unknown truncation option"),
    ({"atol": -1.0}, ValueError, "non-negative"),
    ({"rtol": float("nan")}, ValueError, "finite"),
    ({"atol": float("inf")}, ValueError, "finite"),
    ({"max_rank": 0}, ValueError, "at least 1"),
    ({"max_rank": -2}, ValueError, "at least 1"),
    ({"atol": "1e-3"}, TypeError, "real number"),
    ({"max_rank": 2.0}, TypeError, "int"),
    ({"max_rank": True}, TypeError, "int"),
]


@pytest.mark.parametrize("name", TRUNCATING)
@pytest.mark.parametrize(("opts", "error", "match"), INVALID_OPTS, ids=repr)
def test_invalid_truncation_options_raise(name: str, opts: dict, error: type[Exception], match: str):
    """E9: invalid truncation options raise a clear error in every truncating routine."""
    with pytest.raises(error, match=match):
        TRUNCATING[name](_x(), opts)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", TRUNCATING)
def test_valid_truncation_options_are_accepted(name: str):
    """E9: valid options, including 0-d tensors as tolerances, are accepted."""
    for opts in ({}, {"atol": 0.0}, {"atol": torch.tensor(1e-3)}, {"max_rank": 1}, {"atol": 1e-3, "max_rank": 2}):
        TRUNCATING[name](_x(), opts)  # type: ignore[arg-type]


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E10: no empty modes, no rank 0 ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0, 2), (2, 2, 0)], ids=repr)
def test_factories_reject_empty_modes(name: str, shape: tuple[int, ...]):
    """E10: factories reject a mode of size 0."""
    with pytest.raises(ValueError, match="at least 1"):
        FACTORIES[name](shape, dtype=torch.float64)


def test_to_htt_rejects_empty_modes():
    """E10: ``to_htt`` rejects a dense tensor with a mode of size 0."""
    with pytest.raises(ValueError, match="at least 1"):
        hiepy.to_htt(torch.zeros(3, 0, 2, dtype=torch.float64), {})


@pytest.mark.parametrize(
    "call",
    [
        lambda: hiepy.randn(SHAPE, rank=0),
        lambda: hiepy.randn(SHAPE, rank=-1),
        lambda: hiepy.hilbert(SHAPE, M=0),
        lambda: hiepy.mode_mul(_x(), torch.zeros(0, SHAPE[1], dtype=torch.float64), 1),
    ],
    ids=["randn rank=0", "randn rank=-1", "hilbert M=0", "mode_mul 0 rows"],
)
def test_degenerate_sizes_raise(call: Callable[[], object]):
    """E10: rank < 1, ``M < 1`` and a matrix without rows raise ``ValueError``."""
    with pytest.raises(ValueError):
        call()


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E11: supported range ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", TRUNCATING)
@pytest.mark.parametrize("opts", [{"rtol": 1e-6}, {"atol": 1.0}, {"max_rank": 1}], ids=repr)
def test_overflow_raises(name: str, opts: hiepy.TruncOpts):
    """E11: beyond the supported range (norm about 1e154), truncation raises ``ValueError`` naming the range."""
    with pytest.raises(ValueError, match="supported range"):
        _truncate(name, _x(1e160), opts)


@pytest.mark.parametrize("name", TRUNCATING)
def test_large_values_within_range_work(name: str):
    """E11: tensors well inside the supported range are truncated normally."""
    _truncate(name, _x(1e100), {"rtol": 1e-6})
