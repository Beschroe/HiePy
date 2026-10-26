"""The error model of ``tests/tolerances.py`` detects real deviations.

These tests guard against loosening the model unnoticed (a larger ``SAFETY`` or a coarser scale): on the inputs of
the strategies, a deviation of the result far above rounding level must make the checks fail.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from hiepy import HTTensor
from tests.strategies import ht_tensors
from tests.tolerances import abs_full, assert_exact, assert_orthonormal, size

# Relative deviation that must be detected in each precision (far above SAFETY · ε times the cancellation in the
# representations of the strategies, which stays below about 1e3).
DETECTABLE = {torch.float64: 1e-9, torch.float32: 1e-2}


@given(x=ht_tensors(), data=st.data())
def test_exact_comparison_detects_deviation(x: HTTensor, data: st.DataObject):
    """``assert_exact`` with the absolute-value scale rejects a result that deviates by ``DETECTABLE`` (relative)."""
    X = x.full()
    if not X.any():
        return
    gen = torch.Generator().manual_seed(data.draw(st.integers(0, 2**31 - 1), label="seed"))
    noise = torch.randn(X.shape, generator=gen, dtype=torch.float64).to(X.dtype)
    deviated = X + DETECTABLE[x.dtype] * torch.linalg.norm(X) * noise / torch.linalg.norm(noise)
    assert_exact(X, X, scale=size(abs_full(x)))  # the scale itself is not too tight
    with pytest.raises(AssertionError):
        assert_exact(deviated, X, scale=size(abs_full(x)))


@pytest.mark.parametrize("dtype", list(DETECTABLE))
@pytest.mark.parametrize("k", [1, 4, 25])
def test_orthonormality_check_detects_deviation(dtype: torch.dtype, k: int):
    """``assert_orthonormal`` accepts an orthonormal frame and rejects one perturbed by ``DETECTABLE``."""
    gen = torch.Generator().manual_seed(k)
    Q, _ = torch.linalg.qr(torch.randn(2 * k, k, generator=gen, dtype=torch.float64))
    assert_orthonormal(Q.to(dtype))
    with pytest.raises(AssertionError):
        assert_orthonormal((Q * (1 + DETECTABLE[dtype])).to(dtype))
