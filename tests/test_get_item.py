"""Tests for HTTensor indexing (`hiepy.get`, `HTTensor.get`, `HTTensor.__getitem__`).

The reference semantics are those of dense torch indexing restricted to ints and slices:

- an int removes its mode, a slice keeps it (even if the slice has length one),
- modes not addressed by the key are left untouched,
- negative indices and slice bounds count from the end,
- the result is an HTTensor if at least two modes remain, a 1-D torch.Tensor if one mode
  remains, and a 0-d torch.Tensor if no mode remains.
"""

from __future__ import annotations

import pytest
import torch

import hiepy
from hiepy import HTTensor
from hiepy.utils.data_types import IndexKey

type Key = IndexKey | tuple[IndexKey, ...]

ATOL = 1e-10


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers and fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~


def _make(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, HTTensor]:
    """Return a dense random tensor and its exact HT representation."""
    gen = torch.Generator().manual_seed(seed)
    dense = torch.randn(shape, generator=gen, dtype=torch.double)
    return dense, hiepy.to_htt(dense, {"atol": 1e-13})


def _assert_matches(result: HTTensor | torch.Tensor, expected: torch.Tensor) -> None:
    """Check result type, shape and values against the dense reference."""
    if expected.ndim >= 2:
        assert isinstance(result, HTTensor)
        assert result.shape == tuple(expected.shape)
        dense = result.full()
    else:
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, HTTensor)
        dense = result
    assert dense.shape == expected.shape
    assert torch.allclose(dense, expected, atol=ATOL)


def _assert_valid_structure(x: HTTensor) -> None:
    """Check that the HT components are consistent with the dimension tree."""
    leaves = x.dtree.get_leaves()
    assert sorted(leaves) == [(i,) for i in range(x.order)]
    assert set(x.U) == set(leaves)
    assert set(x.B) == set(x.dtree.get_inner_nodes())
    for t in x.dtree.get_inner_nodes():
        left, right = x.dtree.get_children(t)
        r_left = x.U[left].shape[1] if x.dtree.is_leaf(left) else x.B[left].shape[2]
        r_right = x.U[right].shape[1] if x.dtree.is_leaf(right) else x.B[right].shape[2]
        assert x.B[t].shape[:2] == (r_left, r_right)
    assert x.B[x.dtree.get_root()].shape[2] == 1


@pytest.fixture
def grad_enabled():
    """Enable autograd globally for one test and restore the previous state afterwards."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(previous)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Agreement with dense torch indexing ~~~~~~~~~~~~~~~~~~~~~~~~~

KEYS_ORDER_3 = [
    # single int (first mode), positive and negative
    0,
    2,
    -1,
    -3,
    # single slice (first mode)
    slice(None),
    slice(1, 3),
    slice(0, 1),
    slice(None, None, 2),
    slice(-2, None),
    slice(None, -1),
    slice(1, 100),
    slice(-100, 2),
    # tuples
    (),
    (1,),
    (-1,),
    (slice(1, 3),),
    (0, 1),
    (-1, -2),
    (slice(None), 0),
    (slice(None), -1),
    (slice(None), slice(None), 4),
    (slice(None), slice(None), -5),
    (0, slice(1, 3)),
    (slice(0, 2), slice(0, 2)),
    (slice(None), slice(1, 2)),
    (0, slice(1, 2), slice(None)),
    (slice(None, None, 2), slice(None, None, 3), slice(1, None, 2)),
    (0, 0),
    (slice(None), 1, 2),
    (0, 1, 2),
    (-1, -1, -1),
    (2, -4, 0),
]


@pytest.mark.parametrize("key", KEYS_ORDER_3, ids=repr)
def test_matches_torch_order_3(key: Key):
    dense, x = _make((3, 4, 5))
    _assert_matches(hiepy.get(x, key), dense[key])


KEYS_ORDER_5 = [
    (slice(None), 1),
    (slice(None), 1, slice(None), 0),
    (0, slice(None), 1),
    (slice(None), slice(None), slice(None), slice(None), -1),
    (1, 2, slice(None), 0, slice(None)),
    (slice(None), 0, 1, 2),
    (0, 1, 0, 1),
    (0, 1, 0, 1, 0),
    (slice(0, 1), slice(None), slice(0, 1), slice(None), slice(0, 1)),
    (-1, slice(1, None), -1, slice(None, 2), -1),
]


@pytest.mark.parametrize("key", KEYS_ORDER_5, ids=repr)
def test_matches_torch_order_5(key: Key):
    """Removing several modes at different tree positions restructures the tree correctly."""
    dense, x = _make((2, 3, 2, 3, 2))
    result = hiepy.get(x, key)
    _assert_matches(result, dense[key])
    if isinstance(result, HTTensor):
        _assert_valid_structure(result)


@pytest.mark.parametrize("key", [0, -1, slice(1, 3), (0, 1), (slice(None), -1), (1, slice(None))], ids=repr)
def test_matches_torch_order_2(key: Key):
    dense, x = _make((3, 4))
    _assert_matches(hiepy.get(x, key), dense[key])


def test_all_entries_of_small_tensor():
    """Every single entry is accessible via positive and negative indices."""
    dense, x = _make((2, 3, 2))
    for i in range(-2, 2):
        for j in range(-3, 3):
            for k in range(-2, 2):
                _assert_matches(x[i, j, k], dense[i, j, k])


def test_entry_example_from_docstring():
    dense = torch.arange(1, 28, dtype=torch.double).reshape(3, 3, 3)
    x = hiepy.to_htt(dense, {"atol": 1e-8})
    assert torch.allclose(hiepy.get(x, (0, 1, 2)), torch.tensor(6.0, dtype=torch.double))
    assert torch.allclose(hiepy.get(x, (-1, -1, -1)), torch.tensor(27.0, dtype=torch.double))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Entry points ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("key", [1, -2, slice(0, 2), (0, -1), (slice(None), 1, -1)], ids=repr)
def test_entry_points_agree(key: Key):
    dense, x = _make((3, 4, 5))
    via_function = hiepy.get(x, key)
    via_method = x.get(key)
    via_getitem = x[key]
    for result in (via_method, via_getitem):
        _assert_matches(result, dense[key])
    _assert_matches(via_function, dense[key])


def test_getitem_with_comma_syntax():
    dense, x = _make((3, 4, 5))
    _assert_matches(x[0, 1:3], dense[0, 1:3])
    _assert_matches(x[-1, :, -2], dense[-1, :, -2])


def test_negative_index_equals_positive_counterpart():
    _, x = _make((3, 4, 5))
    assert torch.allclose(x[-1].full(), x[2].full())
    assert torch.allclose(x[:, -4].full(), x[:, 0].full())
    assert torch.allclose(x[-3, -4, -5], x[0, 0, 0])


def test_index_like_integers_are_accepted():
    """Objects implementing __index__ (e.g. 0-d integer tensors) behave like ints."""
    dense, x = _make((3, 4, 5))
    _assert_matches(x[torch.tensor(1)], dense[1])
    _assert_matches(x[torch.tensor(-1), 2], dense[-1, 2])


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Singleton handling ~~~~~~~~~~~~~~~~~~~~~~~~~


def test_length_one_slice_keeps_mode():
    dense, x = _make((3, 4, 5))
    result = x[0:1]
    assert result.shape == (1, 4, 5)
    _assert_matches(result, dense[0:1])
    _assert_matches(x[:, 2:3, 0], dense[:, 2:3, 0])


def test_untouched_singleton_modes_are_kept():
    dense, x = _make((3, 1, 5))
    _assert_matches(x[1:3], dense[1:3])
    _assert_matches(x[()], dense[()])
    _assert_matches(x[(slice(None),)], dense[(slice(None),)])
    _assert_matches(x[0], dense[0])
    _assert_matches(x[:, :, 2], dense[:, :, 2])


def test_order_one_result_is_dense_vector():
    dense, x = _make((3, 4, 5))
    result = x[1, 2]
    assert isinstance(result, torch.Tensor) and not isinstance(result, HTTensor)
    assert result.shape == (5,)
    _assert_matches(result, dense[1, 2])


def test_order_zero_result_is_scalar_tensor():
    dense, x = _make((3, 4, 5))
    result = x[1, 2, 3]
    assert isinstance(result, torch.Tensor) and result.ndim == 0
    _assert_matches(result, dense[1, 2, 3])


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Error handling ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("key", [3, -4, (0, 4), (0, -5), (0, 0, 5), (slice(None), slice(None), -6)], ids=repr)
def test_out_of_bounds_raises_index_error(key: Key):
    _, x = _make((3, 4, 5))
    with pytest.raises(IndexError, match="out of bounds"):
        hiepy.get(x, key)


def test_too_many_indices_raise_index_error():
    _, x = _make((3, 4, 5))
    with pytest.raises(IndexError, match="Too many indices"):
        hiepy.get(x, (0, 0, 0, 0))


@pytest.mark.parametrize(
    "key",
    [1.0, "0", None, Ellipsis, True, [0, 1], (0, 1.5), (0, None), (Ellipsis, 0), (False,), torch.tensor([0, 1])],
    ids=repr,
)
def test_unsupported_key_types_raise_type_error(key: object):
    _, x = _make((3, 4, 5))
    with pytest.raises(TypeError):
        hiepy.get(x, key)


@pytest.mark.parametrize("key", [slice(None, None, 0), slice(None, None, -1), (slice(None), slice(3, 0, -1))], ids=repr)
def test_non_positive_slice_step_raises_value_error(key: Key):
    _, x = _make((3, 4, 5))
    with pytest.raises(ValueError, match="step"):
        hiepy.get(x, key)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Value semantics and metadata ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("key", [0, slice(1, 3), (slice(None), 1), (0, 1), (0, 1, 2), ()], ids=repr)
def test_input_is_not_modified(key: Key):
    dense, x = _make((3, 4, 5))
    U_before = {t: u.clone() for t, u in x.U.items()}
    B_before = {t: b.clone() for t, b in x.B.items()}
    children_before = dict(x.dtree.children)
    orthog_before = x.is_orthog

    hiepy.get(x, key)

    assert dict(x.dtree.children) == children_before
    assert x.is_orthog == orthog_before
    assert set(x.U) == set(U_before) and set(x.B) == set(B_before)
    assert all(torch.equal(x.U[t], U_before[t]) for t in U_before)
    assert all(torch.equal(x.B[t], B_before[t]) for t in B_before)
    assert torch.allclose(x.full(), dense, atol=ATOL)


@pytest.mark.parametrize("key", [slice(None), (), slice(1, 3), (slice(None), 1)], ids=repr)
def test_result_shares_no_memory_with_input(key: Key):
    _, x = _make((3, 4, 5))
    result = hiepy.get(x, key)
    x_full_before = x.full().clone()
    for u in result.U.values():
        u.zero_()
    for b in result.B.values():
        b.zero_()
    assert torch.allclose(x.full(), x_full_before)
    assert result.dtree is not x.dtree


def test_orthogonality_flag():
    _, x = _make((3, 4, 5))
    assert x.is_orthog
    # Selecting rows of an orthonormal leaf matrix destroys orthonormality
    assert x[1:3].is_orthog is False
    assert x[:, 1].is_orthog is False
    # Selecting everything leaves the representation unchanged
    assert x[()].is_orthog is True
    assert x[:, :].is_orthog is True


@pytest.mark.parametrize("key", [slice(1, 3), (slice(None), 0), (0, slice(0, 2), slice(None)), (1, 2)], ids=repr)
def test_result_supports_further_operations(key: Key):
    """The returned HT tensor is a valid HT tensor that downstream operations accept."""
    dense, x = _make((3, 4, 5, 2))
    result = hiepy.get(x, key)
    expected = dense[key]
    _assert_valid_structure(result)
    assert torch.allclose(result.orthog().full(), expected, atol=ATOL)
    assert torch.allclose(result.norm(), torch.linalg.norm(expected), atol=ATOL)
    assert torch.allclose(hiepy.contract(result, result), torch.sum(expected * expected), atol=ATOL)


def test_indexing_a_non_canonical_tree():
    """Indexing works on HT tensors whose tree was restructured by a previous squeeze."""
    gen = torch.Generator().manual_seed(1)
    dense = torch.randn(3, 1, 4, 5, generator=gen, dtype=torch.double)
    x = hiepy.to_htt(dense, {"atol": 1e-13}).squeeze()
    dense = dense.squeeze(1)
    for key in [0, -1, (slice(None), 2), (1, slice(1, 3), -1), (0, 0), (slice(None), slice(None), 4)]:
        _assert_matches(x[key], dense[key])


def test_repeated_indexing_matches_single_indexing():
    dense, x = _make((3, 4, 5, 2))
    _assert_matches(x[1][2], dense[1, 2])
    _assert_matches(x[:, 1:3][0], dense[0, 1:3])
    _assert_matches(x[-1][:, -1], dense[-1, :, -1])


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Autograd ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("key", [slice(1, 3), (0, slice(None)), (0, 1), (0, 1, 2)], ids=repr)
def test_track_grad_connects_result_to_input(grad_enabled: None, key: Key):
    x = hiepy.randn((3, 4, 5), rank=2, requires_grad=True)
    result = hiepy.get(x, key, track_grad=True)
    # `full` keeps the autograd graph (unlike `norm`, which orthogonalizes via a detached copy)
    scalar = result.full().sum() if isinstance(result, HTTensor) else result.sum()
    scalar.backward()
    assert any(u.grad is not None and torch.any(u.grad != 0) for u in x.U.values())


@pytest.mark.parametrize("key", [slice(1, 3), (0, slice(None)), (0, 1), (0, 1, 2)], ids=repr)
def test_without_track_grad_result_is_detached(grad_enabled: None, key: Key):
    x = hiepy.randn((3, 4, 5), rank=2, requires_grad=True)
    result = hiepy.get(x, key, track_grad=False)
    tensors = list(result.U.values()) + list(result.B.values()) if isinstance(result, HTTensor) else [result]
    assert not any(t.requires_grad for t in tensors)


def test_getitem_does_not_track_grad(grad_enabled: None):
    x = hiepy.randn((3, 4, 5), rank=2, requires_grad=True)
    result = x[0, 1]
    assert not result.requires_grad
