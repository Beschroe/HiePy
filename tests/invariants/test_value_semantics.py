"""Value-semantics invariants V1–V2 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from tests.helpers import assert_unchanged, snapshot, storages
from tests.ops import OPS, ht_result
from tests.strategies import decaying_dense, ht_pairs


@pytest.mark.parametrize("name", OPS)
@given(pair=ht_pairs(), data=st.data())
def test_inputs_are_not_modified(name: str, pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """V1: components, tree and flag of all inputs are unchanged."""
    x, y = pair
    op = OPS[name]
    arg = data.draw(op.args(x), label="arg")
    before_x, before_y = snapshot(x), snapshot(y)
    op.call(x, y, arg)
    assert_unchanged(x, before_x)
    assert_unchanged(y, before_y)


@pytest.mark.parametrize("name", OPS)
@given(pair=ht_pairs(), data=st.data())
def test_result_shares_nothing_with_inputs(name: str, pair: tuple[HTTensor, HTTensor], data: st.DataObject):
    """V2: the result shares neither component storage nor the ``DimTree`` object with its inputs, except where the
    catalog allows it (``detach``; ``to`` without conversion)."""
    x, y = pair
    op = OPS[name]
    arg = data.draw(op.args(x), label="arg")
    result = ht_result(op.call(x, y, arg))
    if result is None or op.shares_storage(x, arg):
        return
    for operand in (x, y):
        assert not storages(result) & storages(operand)
        assert result.dtree is not operand.dtree


@given(T=decaying_dense())
def test_to_htt_does_not_modify_or_share_its_input(T: torch.Tensor):
    """V1, V2: ``to_htt`` leaves the dense input unchanged and shares no storage with it."""
    before = T.clone()
    x = hiepy.to_htt(T, {"rtol": 1e-6})
    assert torch.equal(T, before)
    assert T.untyped_storage().data_ptr() not in storages(x)
