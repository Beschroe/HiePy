"""Exactness invariants N1–N2 (see ``tests/INVARIANTS.md``): exact operations match their dense counterpart."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hiepy import HTTensor
from tests.ops import OPS
from tests.strategies import ht_pairs
from tests.tolerances import assert_exact

EXACT_OPS = [name for name, op in OPS.items() if op.dense is not None]


@pytest.mark.parametrize("name", EXACT_OPS)
@given(pair=ht_pairs(), data=st.data())
def test_matches_dense_reference(name: str, pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """N1, N2: the result equals the dense counterpart applied to the dense inputs, up to a test ratio of ``THRESH``
    relative to the size of the inputs (``Op.input_size``)."""
    x, y = pair
    op = OPS[name]
    assert op.dense is not None
    arg = data.draw(op.args(x), label="arg")
    result = op.call(x, y, arg)
    actual = result.full() if isinstance(result, HTTensor) else result
    expected = op.dense(x.full(), y.full(), arg)
    dtypes = (x.dtype, y.dtype, actual.dtype)
    assert_exact(actual, expected, size=op.input_size(x, y, arg), dtypes=dtypes)
