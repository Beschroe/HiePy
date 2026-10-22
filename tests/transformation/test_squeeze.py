"""Tests for ``squeeze`` with explicit dimensions, compared with ``torch.squeeze``.

Invariants covered (``tests/INVARIANTS.md``): N1 (agreement with the dense reference), E2 (modes in ``[-d, d)``).
"""

from __future__ import annotations

import pytest
import torch

import hiepy
from hiepy import HTTensor
from tests.helpers import assert_valid_structure, make_dense_htt

SHAPE = (1, 3, 1, 4, 1)

# (dims passed to hiepy.squeeze, equivalent non-negative dims for torch)
DIMS = [
    (0, (0,)),
    (-1, (4,)),
    (-5, (0,)),
    ((0, -1), (0, 4)),
    ([-3, 4], (2, 4)),
    ((2, -3), (2,)),  # the same mode twice
    ((), ()),
]


@pytest.mark.parametrize(("dims", "torch_dims"), DIMS, ids=repr)
def test_matches_torch(dims: int | tuple[int, ...], torch_dims: tuple[int, ...]):
    """N1, E2: negative dimensions count from the end, as in ``torch.squeeze``."""
    dense, x = make_dense_htt(SHAPE)
    result = hiepy.squeeze(x, dims)
    expected = dense.squeeze(torch_dims) if torch_dims else dense
    assert isinstance(result, HTTensor)
    assert_valid_structure(result)
    assert result.shape == tuple(expected.shape)
    assert torch.allclose(result.full(), expected, atol=1e-12)


@pytest.mark.parametrize("dims", [5, -6, (0, 7), [-1, -8]], ids=repr)
def test_out_of_range_raises(dims: int | tuple[int, ...]):
    """E2: dimensions outside ``[-d, d)`` raise ``ValueError``."""
    _, x = make_dense_htt(SHAPE)
    with pytest.raises(ValueError, match="out of bounds"):
        hiepy.squeeze(x, dims)


@pytest.mark.parametrize("dims", [1, -2, (0, -4)], ids=repr)
def test_non_singleton_raises(dims: int | tuple[int, ...]):
    """E2: dimensions that are not singletons raise ``ValueError``, also when given negatively."""
    _, x = make_dense_htt(SHAPE)
    with pytest.raises(ValueError, match="not singletons"):
        hiepy.squeeze(x, dims)
