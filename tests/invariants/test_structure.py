"""Structure invariants S1–S5 (see ``tests/INVARIANTS.md``)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import hiepy
from hiepy import DimTree, HTTensor
from tests.helpers import assert_valid_structure
from tests.ops import FACTORIES, OPS, ht_result
from tests.strategies import ht_pairs, ht_tensors, shapes


@pytest.mark.parametrize("name", OPS)
@given(pair=ht_pairs())
def test_operation_results_are_valid(name: str, pair: tuple[HTTensor, HTTensor]):
    """S1, S2, S3, D1: every HT result matches its tree, has consistent ranks, order >= 2 and one dtype/device."""
    x, y = pair
    op = OPS[name]
    result = ht_result(op.call(x, y))
    if result is None:
        return
    assert_valid_structure(result)
    assert result.order >= 2
    assert result.dtype == (op.out_dtype or x.dtype)
    assert result.device == x.device


@pytest.mark.parametrize("name", FACTORIES)
@given(shape=shapes())
def test_factory_results_are_valid(name: str, shape: tuple[int, ...]):
    """S1, S2, S5: factories build valid tensors on the canonical tree."""
    x = FACTORIES[name](shape, dtype=torch.float64)
    assert_valid_structure(x)
    assert x.shape == shape
    assert x.dtree == DimTree.from_order(len(shape))


@given(shape=shapes(), seed=st.integers(0, 2**31 - 1))
def test_to_htt_builds_canonical_tree(shape: tuple[int, ...], seed: int):
    """S1, S2, S5: ``to_htt`` builds a valid tensor on the canonical tree."""
    gen = torch.Generator().manual_seed(seed)
    x = hiepy.to_htt(torch.randn(shape, generator=gen, dtype=torch.float64), {"atol": 0.0})
    assert_valid_structure(x)
    assert x.dtree == DimTree.from_order(len(shape))


@pytest.mark.parametrize("name", [name for name, op in OPS.items() if op.keeps_tree])
@given(pair=ht_pairs())
def test_tree_is_preserved(name: str, pair: tuple[HTTensor, HTTensor]):
    """S4: operations other than ``squeeze`` and integer indexing keep the dimension tree."""
    x, y = pair
    result = ht_result(OPS[name].call(x, y))
    if result is not None:
        assert result.dtree == x.dtree


@given(x=ht_tensors())
def test_indexing_to_order_below_two_is_dense(x: HTTensor):
    """S3: ``get`` returns a dense 1-d tensor if one mode remains and a 0-d tensor if none remains."""
    vector = x[(0,) * (x.order - 1)]
    assert isinstance(vector, torch.Tensor) and not isinstance(vector, HTTensor)
    assert vector.shape == (x.shape[-1],)
    entry = x[(0,) * x.order]
    assert isinstance(entry, torch.Tensor) and entry.ndim == 0


@given(order=st.integers(2, 6), n=st.integers(2, 5), seed=st.integers(0, 2**31 - 1))
def test_squeeze_keeps_order_two(order: int, n: int, seed: int):
    """S3: ``squeeze`` keeps order >= 2, unless all modes are singletons (then the result is a 0-d tensor)."""
    gen = torch.Generator().manual_seed(seed)
    one_mode = hiepy.randn((1,) * (order - 1) + (n,), rank=2, dtype=torch.float64, generator=gen)
    result = hiepy.squeeze(one_mode)
    assert isinstance(result, HTTensor) and result.order == 2
    assert_valid_structure(result)
    no_mode = hiepy.randn((1,) * order, rank=2, dtype=torch.float64, generator=gen)
    scalar = hiepy.squeeze(no_mode)
    assert isinstance(scalar, torch.Tensor) and not isinstance(scalar, HTTensor) and scalar.ndim == 0
