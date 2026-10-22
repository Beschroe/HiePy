"""Exactness invariants N1–N2 (see ``tests/INVARIANTS.md``): exact operations match their dense counterpart."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from hiepy import HTTensor
from tests.helpers import assert_close
from tests.ops import OPS
from tests.strategies import ht_pairs

EXACT_OPS = [name for name, op in OPS.items() if op.dense is not None]


@pytest.mark.parametrize("name", EXACT_OPS)
@given(pair=ht_pairs(), data=st.data())
def test_matches_dense_reference(name: str, pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """N1, N2: the result equals the dense counterpart applied to the dense inputs, relative to the larger of the
    reference norm and the size of the inputs entering it (``Op.magnitude``)."""
    x, y = pair
    op = OPS[name]
    assert op.dense is not None
    arg = data.draw(op.args(x), label="arg")
    result = op.call(x, y, arg)
    actual = result.full() if isinstance(result, HTTensor) else result
    expected = op.dense(x.full(), y.full(), arg)
    scale = max(torch.linalg.norm(expected.to(torch.float64)).item(), op.magnitude(x, y, arg))
    assert_close(actual.to(torch.float64), expected.to(torch.float64), scale=scale, dtype=actual.dtype)
