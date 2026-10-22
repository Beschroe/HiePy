"""Value-semantics invariants V1–V2 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

import pytest
from hypothesis import given

from hiepy import HTTensor
from tests.helpers import assert_unchanged, snapshot, storages
from tests.ops import OPS, ht_result
from tests.strategies import ht_pairs


@pytest.mark.parametrize("name", OPS)
@given(pair=ht_pairs())
def test_inputs_are_not_modified(name: str, pair: tuple[HTTensor, HTTensor]):
    """V1: components, tree and flag of all inputs are unchanged, for orthogonal and non-orthogonal inputs."""
    x, y = pair
    before_x, before_y = snapshot(x), snapshot(y)
    OPS[name].call(x, y)
    assert_unchanged(x, before_x)
    assert_unchanged(y, before_y)


@pytest.mark.parametrize("name", [name for name, op in OPS.items() if not op.shares_storage])
@given(pair=ht_pairs())
def test_result_shares_nothing_with_inputs(name: str, pair: tuple[HTTensor, HTTensor]):
    """V2: the result shares neither component storage nor the ``DimTree`` object with its inputs."""
    x, y = pair
    result = ht_result(OPS[name].call(x, y))
    if result is None:
        return
    for operand in (x, y):
        assert not storages(result) & storages(operand)
        assert result.dtree is not operand.dtree
