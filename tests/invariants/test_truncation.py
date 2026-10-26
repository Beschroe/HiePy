"""Truncation invariants N4–N6 (see ``tests/INVARIANTS.md``).

Inputs are either tensors with decaying hierarchical singular values or the general HT tensors of
``tests/strategies.py``, which are low-rank in some nodes only, orthogonal in some nodes only, over-parameterized,
rank-deficient, zero or differently scaled.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from tests.strategies import decaying_dense, ht_pairs, ht_tensors
from tests.tolerances import Accuracy, assert_within, frobenius, resolution

TOLS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

type Truncation = Callable[[HTTensor, hiepy.TruncOpts], tuple[HTTensor, dict[hiepy.Node, torch.Tensor]]]

# Routine and accuracy of its singular values: truncate_gram computes them as square roots of Gramian eigenvalues
# (Accuracy.GRAMIAN), which limits the testable tolerances (N4) and the rounding margin of N5 (Q4/Q5).
TRUNCATIONS: dict[str, tuple[Truncation, Accuracy]] = {
    "truncate_svd": (hiepy.truncate_svd, Accuracy.ROUNDING),
    "truncate_gram": (hiepy.truncate_gram, Accuracy.GRAMIAN),
}


def _tols(accuracy: Accuracy, dtype: torch.dtype) -> st.SearchStrategy[float]:
    """Tolerances from ``TOLS`` that the accuracy class can resolve in ``dtype`` (Q4)."""
    return st.sampled_from([tol for tol in TOLS if tol >= resolution(accuracy, dtype)])


@st.composite
def _inputs(draw: st.DrawFn) -> HTTensor:
    """A tensor with decaying singular values (exact HT representation) or a general HT tensor."""
    if draw(st.booleans()):
        return hiepy.to_htt(draw(decaying_dense()), {"atol": 0.0})
    return draw(ht_tensors())


def _opts(kind: str, tol: float, norm: float) -> tuple[hiepy.TruncOpts, float]:
    """Options with tolerance ``tol`` relative to ``norm`` and the resulting bound ``atol + rtol · norm``."""
    if kind == "atol":
        return {"atol": tol * norm}, tol * norm
    if kind == "rtol":
        return {"rtol": tol}, tol * norm
    return {"atol": tol * norm / 2, "rtol": tol / 2}, tol * norm


def _error(reference: torch.Tensor, result: HTTensor) -> float:
    return torch.linalg.norm(reference - result.full()).item()


def _max_rank(x: HTTensor) -> int:
    return max(r for t, r in x.rank.items() if not x.dtree.is_root(t))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N4: error vs. tolerance ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("kind", ["atol", "rtol", "both"])
@pytest.mark.parametrize("name", TRUNCATIONS)
@given(x=_inputs(), data=st.data())
def test_truncation_meets_tolerance(name: str, kind: str, x: HTTensor, data: st.DataObject):
    """N4: ``truncate_*`` stays within ``atol + rtol · ‖x‖`` (plus rounding margin)."""
    truncate, accuracy = TRUNCATIONS[name]
    tol = data.draw(_tols(accuracy, x.dtype), label="tol")
    X = x.full()
    opts, bound = _opts(kind, tol, torch.linalg.norm(X).item())
    result, _ = truncate(x, opts)
    assert_within(_error(X, result), bound, size=frobenius(x), dtypes=(x.dtype,))


@pytest.mark.parametrize("kind", ["atol", "rtol", "both"])
@given(x=_inputs(), tol=st.sampled_from(TOLS))
def test_to_htt_meets_tolerance(kind: str, x: HTTensor, tol: float):
    """N4: ``to_htt`` stays within ``atol + rtol · ‖T‖`` (plus rounding margin)."""
    T = x.full()
    norm = torch.linalg.norm(T).item()
    opts, bound = _opts(kind, tol, norm)
    assert_within(_error(T, hiepy.to_htt(T, opts)), bound, size=norm, size_label="‖T‖", dtypes=(T.dtype,))


@given(pair=ht_pairs(), data=st.data())
def test_ele_mul_meets_tolerance(pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """N4: ``ele_mul`` with ``atol`` stays within ``atol`` of the exact Hadamard product (plus rounding margin)."""
    x, y = pair
    tol = data.draw(_tols(Accuracy.GRAMIAN, x.dtype), label="tol")
    P = x.full() * y.full()
    atol = tol * torch.linalg.norm(P).item()
    result = hiepy.ele_mul(x, y, {"atol": atol})
    size = frobenius(x) * frobenius(y)
    assert_within(_error(P, result), atol, size=size, size_label="‖x‖·‖y‖", dtypes=(x.dtype,))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N5: error vs. discarded singular values ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", TRUNCATIONS)
@given(x=_inputs(), data=st.data())
def test_error_bounded_by_discarded_singular_values(name: str, x: HTTensor, data: st.DataObject):
    """N5: the error is bounded by the discarded singular values (plus a rounding margin according to their
    accuracy), also when ``max_rank`` cuts."""
    truncate, accuracy = TRUNCATIONS[name]
    X = x.full()
    if data.draw(st.booleans(), label="max_rank instead of tolerance"):
        opts: hiepy.TruncOpts = {"max_rank": data.draw(st.integers(1, 4), label="max_rank")}
    else:
        opts, _ = _opts("rtol", data.draw(_tols(accuracy, x.dtype), label="tol"), 0.0)
    result, discarded = truncate(x, opts)
    bound = math.sqrt(sum((s**2).sum().item() for s in discarded.values()))
    assert_within(_error(X, result), bound, size=frobenius(x), accuracy=accuracy, dtypes=(x.dtype,))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ N6: rank cap ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", TRUNCATIONS)
@given(x=_inputs(), max_rank=st.integers(1, 4))
def test_truncation_respects_max_rank(name: str, x: HTTensor, max_rank: int):
    """N6: ``truncate_*`` with ``max_rank`` (alone or with a tolerance) caps every non-root rank."""
    truncate, _ = TRUNCATIONS[name]
    for opts in ({"max_rank": max_rank}, {"max_rank": max_rank, "rtol": 1e-12}):
        result, _ = truncate(x, opts)  # type: ignore[arg-type]
        assert _max_rank(result) <= max_rank


@given(x=_inputs(), max_rank=st.integers(1, 4))
def test_to_htt_respects_max_rank(x: HTTensor, max_rank: int):
    """N6: ``to_htt`` with ``max_rank`` caps every non-root rank."""
    assert _max_rank(hiepy.to_htt(x.full(), {"max_rank": max_rank})) <= max_rank


@given(pair=ht_pairs(), max_rank=st.integers(1, 4))
def test_ele_mul_respects_max_rank(pair: tuple[HTTensor, HTTensor], max_rank: int):
    """N6: ``ele_mul`` with ``max_rank`` caps every non-root rank."""
    x, y = pair
    assert _max_rank(hiepy.ele_mul(x, y, {"max_rank": max_rank})) <= max_rank
