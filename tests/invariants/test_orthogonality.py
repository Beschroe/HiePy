"""Orthogonality-flag invariants O1–O2 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from hiepy import HTTensor
from tests.helpers import assert_orthogonal
from tests.ops import FACTORIES, FACTORY_FLAGS, OPS, ht_result
from tests.strategies import ht_pairs, shapes

FLAGGED_OPS = [name for name, op in OPS.items() if op.flag is not None]


@pytest.mark.parametrize("name", FLAGGED_OPS)
@given(pair=ht_pairs(), data=st.data())
def test_operation_sets_flag(name: str, pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """O1, O2: each operation sets ``is_orthog`` as listed in the catalog, and a ``True`` flag is sound (checked in
    float64). Inputs include tensors that are orthogonal in some nodes only."""
    x, y = pair
    op = OPS[name]
    arg = data.draw(op.args(x), label="arg")
    result = ht_result(op.call(x, y, arg))
    if result is None:
        return
    assert op.flag is not None
    assert result.is_orthog == op.flag(x, arg)
    if result.is_orthog and result.dtype == torch.float64:
        assert_orthogonal(result)


@pytest.mark.parametrize("name", FACTORIES)
@given(shape=shapes())
def test_factory_flag(name: str, shape: tuple[int, ...]):
    """O1, O2: factories set the flag as listed, and a ``True`` flag is sound."""
    x = FACTORIES[name](shape, dtype=torch.float64)
    assert x.is_orthog == FACTORY_FLAGS[name]
    if x.is_orthog:
        assert_orthogonal(x)
