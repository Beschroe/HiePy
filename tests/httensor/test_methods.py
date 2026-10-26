"""``HTTensor`` methods, properties and operators are thin wrappers: they return what the free functions return.

Also covers the type checks of ``*``, ``truncate`` with an unknown method and the optional warning of ``trunc_rank``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import hiepy
from hiepy import HTTensor
from hiepy.utils.trunc_rank import trunc_rank
from tests.helpers import components
from tests.tolerances import assert_exact

SHAPE = (3, 1, 4)


def _x(seed: int = 0) -> HTTensor:
    return hiepy.randn(SHAPE, rank=2, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


A = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(2, 4)
V = torch.linspace(0.5, 2.0, 3, dtype=torch.float64)

# (method or operator, free function), both applied to x = _x(0), y = _x(1)
WRAPPERS: dict[str, tuple[Callable[[HTTensor, HTTensor], object], Callable[[HTTensor, HTTensor], object]]] = {
    "copy": (lambda x, y: x.copy(), lambda x, y: hiepy.copy(x)),
    "clone": (lambda x, y: x.clone(), lambda x, y: hiepy.clone(x)),
    "detach": (lambda x, y: x.detach(), lambda x, y: hiepy.detach(x)),
    "to": (lambda x, y: x.to(dtype=torch.float32), lambda x, y: hiepy.to(x, dtype=torch.float32)),
    "shape": (lambda x, y: x.shape, lambda x, y: hiepy.shape(x)),
    "rank": (lambda x, y: x.rank, lambda x, y: hiepy.rank(x)),
    "order": (lambda x, y: x.order, lambda x, y: hiepy.order(x)),
    "nbytes": (lambda x, y: x.nbytes, lambda x, y: hiepy.nbytes(x)),
    "dtype": (lambda x, y: x.dtype, lambda x, y: hiepy.dtype(x)),
    "device": (lambda x, y: x.device, lambda x, y: hiepy.device(x)),
    "requires_grad": (lambda x, y: x.requires_grad, lambda x, y: hiepy.requires_grad(x)),
    "norm": (lambda x, y: x.norm(), lambda x, y: hiepy.norm(x)),
    "get": (lambda x, y: x.get((0, slice(None))), lambda x, y: hiepy.get(x, (0, slice(None)))),
    "[]": (lambda x, y: x[0, :], lambda x, y: hiepy.get(x, (0, slice(None)))),
    "full": (lambda x, y: x.full(), lambda x, y: hiepy.full(x)),
    "squeeze": (lambda x, y: x.squeeze(), lambda x, y: hiepy.squeeze(x)),
    "orthog": (lambda x, y: (x + y).orthog(), lambda x, y: hiepy.orthog(x + y)),
    "truncate": (lambda x, y: (x + y).truncate({"max_rank": 1}), lambda x, y: hiepy.truncate(x + y, {"max_rank": 1})),
    "add": (lambda x, y: x.add(y), lambda x, y: hiepy.add(x, y)),
    "+": (lambda x, y: x + y, lambda x, y: hiepy.add(x, y)),
    "sub": (lambda x, y: x.sub(y), lambda x, y: hiepy.sub(x, y)),
    "-": (lambda x, y: x - y, lambda x, y: hiepy.sub(x, y)),
    "scalar_mul": (lambda x, y: x.scalar_mul(2.5), lambda x, y: hiepy.scalar_mul(x, 2.5)),
    "* int": (lambda x, y: x * 3, lambda x, y: hiepy.scalar_mul(x, 3)),
    "* float, right": (lambda x, y: 2.5 * x, lambda x, y: hiepy.scalar_mul(x, 2.5)),
    "* 0-d tensor": (lambda x, y: x * torch.tensor(2.5), lambda x, y: hiepy.scalar_mul(x, torch.tensor(2.5))),
    "mode_mul": (lambda x, y: x.mode_mul(A, 2), lambda x, y: hiepy.mode_mul(x, A, 2)),
    "ele_mode_mul": (lambda x, y: x.ele_mode_mul(V, 0), lambda x, y: hiepy.ele_mode_mul(x, V, 0)),
    "ele_mul": (lambda x, y: x.ele_mul(y, {"max_rank": 2}), lambda x, y: hiepy.ele_mul(x, y, {"max_rank": 2})),
    "* HTTensor": (lambda x, y: x * y, lambda x, y: hiepy.ele_mul(x, y)),
    "contract": (lambda x, y: x.contract(y), lambda x, y: hiepy.contract(x, y)),
    "@": (lambda x, y: x @ y, lambda x, y: hiepy.contract(x, y)),
    "gramians": (lambda x, y: x.gramians(), lambda x, y: hiepy.gramians(x)),
}


def _assert_same(a: object, b: object) -> None:
    if isinstance(a, tuple) and isinstance(b, tuple) and a and isinstance(a[0], HTTensor):
        _assert_same(a[0], b[0])
        _assert_same(a[1], b[1])
    elif isinstance(a, HTTensor) and isinstance(b, HTTensor):
        assert a.dtree == b.dtree and a.is_orthog == b.is_orthog
        assert all(torch.equal(c, d) for c, d in zip(components(a), components(b)))
    elif isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert torch.equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("name", WRAPPERS)
def test_method_matches_free_function(name: str):
    method, function = WRAPPERS[name]
    _assert_same(method(_x(0), _x(1)), function(_x(0), _x(1)))


def test_requires_grad_method_and_function():
    x = _x()
    assert x.requires_grad_() is x and x.requires_grad
    assert hiepy.requires_grad_(x, False) is x and not x.requires_grad


def test_save_and_load_methods():
    x = _x()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.htt"
        x.save(path)
        _assert_same(HTTensor.load(path), hiepy.load(path))


def test_nbytes_counts_all_components():
    x = _x()
    assert x.nbytes == sum(c.numel() * c.element_size() for c in components(x))
    assert_exact(x.norm(), torch.linalg.norm(x.full()))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Errors and warnings ~~~~~~~~~~~~~~~~~~~~~~~~~


def test_mul_rejects_non_scalar_tensor():
    with pytest.raises(ValueError, match="0d"):
        _x() * torch.ones(3)


@pytest.mark.parametrize("other", ["2", [2.0], None], ids=repr)
def test_mul_rejects_unsupported_types(other: object):
    with pytest.raises(TypeError, match="Supported types"):
        _x() * other  # type: ignore[operator]


def test_truncate_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        hiepy.truncate(_x(), {"max_rank": 1}, method="qr")


def test_trunc_rank_warns_when_max_rank_cuts_if_enabled():
    s = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    assert trunc_rank(s, max_rank=1) == 1
    with pytest.warns(UserWarning, match="max_rank"):
        assert trunc_rank(s, max_rank=1, disable_warning=False) == 1
