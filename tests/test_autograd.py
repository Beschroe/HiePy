"""Tests for gradient tracking of HTTensor operations.

hiepy follows the torch convention: an operation records an autograd graph if and only if one of its inputs
requires gradients and autograd is enabled. No operation detaches implicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import hiepy
from hiepy import HTTensor
from tests.helpers import components
from tests.ops import FACTORIES
from tests.ops import OPS as REGISTRY
from tests.ops import Result

SHAPE = (3, 4, 5, 2)

type Op = Callable[[HTTensor, HTTensor], Result]

# copy and detach cut the graph by design (A1: opt-out); all other operations must propagate gradients.
OPS: dict[str, Op] = {name: op.run for name, op in REGISTRY.items() if name not in ("copy", "detach")}


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers and fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture(autouse=True)
def _autograd_on(grad_enabled: None) -> None:
    """Run every test in this module with autograd enabled (fixture from ``conftest.py``)."""


def _rand(seed: int, requires_grad: bool = False) -> HTTensor:
    """Return a random, non-orthogonal float64 HT tensor whose components are autograd leaves."""
    gen = torch.Generator().manual_seed(seed)
    x = hiepy.randn(SHAPE, rank=2, dtype=torch.float64, generator=gen)
    v = torch.rand(SHAPE[0], generator=gen, dtype=torch.float64) + 0.5
    return hiepy.ele_mode_mul(x, v, 0).requires_grad_(requires_grad)


def _to_scalar(result: Result) -> torch.Tensor:
    """Reduce the result of an operation to a scalar that depends on all of its values."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, HTTensor):
        return (result.full() ** 2).sum()
    if isinstance(result, dict):
        return sum((s**2).sum() for _, s in result.values())  # type: ignore[return-value]
    return (result**2).sum()


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient propagation ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_gradients_reach_input_components(op: Op):
    """A2: gradients reach the input components (no implicit detach)."""
    x, y = _rand(0, requires_grad=True), _rand(1, requires_grad=True)
    _to_scalar(op(x, y)).backward()
    grads = [t.grad for t in components(x)]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g != 0) for g in grads)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_no_graph_without_requires_grad(op: Op):
    """A1: no graph is recorded if no input requires gradients."""
    x, y = _rand(0), _rand(1)
    assert not x.requires_grad
    assert not _to_scalar(op(x, y)).requires_grad


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_no_graph_under_no_grad(op: Op):
    """A1: no graph is recorded with autograd disabled."""
    x, y = _rand(0, requires_grad=True), _rand(1, requires_grad=True)
    with torch.no_grad():
        assert not _to_scalar(op(x, y)).requires_grad


@pytest.mark.parametrize("name", ["contract", "norm", "mode_mul", "add", "orthog", "get"])
def test_gradcheck(name: str):
    """A3: gradients of operations without SVD/eigendecomposition are correct."""
    x, y = _rand(0), _rand(1)
    leaves, nodes = x.dtree.get_leaves(), x.dtree.get_inner_nodes()

    def f(*components: torch.Tensor) -> torch.Tensor:
        U = dict(zip(leaves, components[: len(leaves)]))
        B = dict(zip(nodes, components[len(leaves) :]))
        return _to_scalar(OPS[name](HTTensor(U=U, B=B, dtree=x.dtree.copy()), y))

    inputs = tuple(t.clone().requires_grad_() for t in [*(x.U[t] for t in leaves), *(x.B[t] for t in nodes)])
    assert torch.autograd.gradcheck(f, inputs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ requires_grad / requires_grad_ / detach ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", FACTORIES)
def test_factories_default_to_no_grad(name: str):
    """A4: factories return tensors without gradient tracking by default."""
    assert not FACTORIES[name](SHAPE).requires_grad


@pytest.mark.parametrize("name", FACTORIES)
def test_factories_create_grad_leaves(name: str):
    """A4: with ``requires_grad=True``, factories return autograd leaves."""
    x = FACTORIES[name](SHAPE, requires_grad=True)
    assert x.requires_grad
    assert all(t.requires_grad and t.is_leaf for t in components(x))


def test_requires_grad_toggles_in_place():
    x = _rand(0)
    assert x.requires_grad_() is x
    assert hiepy.requires_grad(x)
    assert not hiepy.requires_grad_(x, False).requires_grad


def test_requires_grad_false_on_tracked_result_raises():
    y = 2.0 * _rand(0, requires_grad=True)
    with pytest.raises(RuntimeError):
        y.requires_grad_(False)
    assert not y.detach().requires_grad
