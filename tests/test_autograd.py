"""Tests for gradient tracking of HTTensor operations.

hiepy follows the torch convention: an operation records an autograd graph if and only if one of its inputs
requires gradients and autograd is enabled. No operation detaches implicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch

import hiepy
from hiepy import HTTensor

SHAPE = (3, 4, 5, 2)

type Result = HTTensor | torch.Tensor | tuple[HTTensor, dict[hiepy.Node, torch.Tensor]] | dict
type Op = Callable[[HTTensor, HTTensor], Result]


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers and fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture(autouse=True)
def grad_enabled() -> Iterator[None]:
    """Enable autograd globally for each test and restore the previous state afterwards."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(previous)


def _rand(seed: int, requires_grad: bool = False) -> HTTensor:
    """Return a random, non-orthogonal float64 HT tensor whose components are autograd leaves."""
    gen = torch.Generator().manual_seed(seed)
    x = hiepy.randn(SHAPE, rank=2, dtype=torch.float64, generator=gen)
    v = torch.rand(SHAPE[0], generator=gen, dtype=torch.float64) + 0.5
    return hiepy.ele_mode_mul(x, v, 0).requires_grad_(requires_grad)


def _components(x: HTTensor) -> list[torch.Tensor]:
    return [*x.U.values(), *x.B.values()]


def _to_scalar(result: Result) -> torch.Tensor:
    """Reduce the result of an operation to a scalar that depends on all of its values."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, HTTensor):
        return (result.full() ** 2).sum()
    if isinstance(result, dict):
        return sum((s**2).sum() for _, s in result.values())  # type: ignore[return-value]
    return (result**2).sum()


OPS: dict[str, Op] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "scalar_mul": lambda x, y: 3.0 * x,
    "mode_mul": lambda x, y: hiepy.mode_mul(x, torch.ones(2, SHAPE[1], dtype=x.dtype), 1),
    "ele_mode_mul": lambda x, y: hiepy.ele_mode_mul(x, torch.arange(SHAPE[2], dtype=x.dtype), 2),
    "ele_mul": lambda x, y: x * y,
    "ele_mul_truncated": lambda x, y: hiepy.ele_mul(x, y, {"max_rank": 2}),
    "contract": lambda x, y: x @ y,
    "norm": lambda x, y: hiepy.norm(x),
    "orthog": lambda x, y: hiepy.orthog(x),
    "truncate_svd": lambda x, y: hiepy.truncate_svd(x, {"max_rank": 1}),
    "truncate_gram": lambda x, y: hiepy.truncate_gram(x, {"max_rank": 1}),
    "local_left_svd": lambda x, y: hiepy.local_left_svd(x),
    "full": lambda x, y: x.full(),
    "get": lambda x, y: x[1:3, 0],
    "get_dense": lambda x, y: x[1, 0, 2],
    "squeeze": lambda x, y: hiepy.squeeze(x[1:2]),
    "to": lambda x, y: x.to(dtype=torch.float32),
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient propagation ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_gradients_reach_input_components(op: Op):
    x, y = _rand(0, requires_grad=True), _rand(1, requires_grad=True)
    _to_scalar(op(x, y)).backward()
    grads = [t.grad for t in _components(x)]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g != 0) for g in grads)  # type: ignore[arg-type]


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_no_graph_without_requires_grad(op: Op):
    x, y = _rand(0), _rand(1)
    assert not x.requires_grad
    assert not _to_scalar(op(x, y)).requires_grad


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_no_graph_under_no_grad(op: Op):
    x, y = _rand(0, requires_grad=True), _rand(1, requires_grad=True)
    with torch.no_grad():
        assert not _to_scalar(op(x, y)).requires_grad


@pytest.mark.parametrize("op", OPS.values(), ids=OPS.keys())
def test_input_is_not_modified(op: Op):
    x, y = _rand(0, requires_grad=True), _rand(1, requires_grad=True)
    before = [t.detach().clone() for t in _components(x)]
    op(x, y)
    assert all(torch.equal(t, b) for t, b in zip(_components(x), before))
    assert x.is_orthog is False


@pytest.mark.parametrize("name", ["contract", "norm", "mode_mul", "add", "orthog", "get"])
def test_gradcheck(name: str):
    x, y = _rand(0), _rand(1)
    leaves, nodes = x.dtree.get_leaves(), x.dtree.get_inner_nodes()

    def f(*components: torch.Tensor) -> torch.Tensor:
        U = dict(zip(leaves, components[: len(leaves)]))
        B = dict(zip(nodes, components[len(leaves) :]))
        return _to_scalar(OPS[name](HTTensor(U=U, B=B, dtree=x.dtree.copy()), y))

    inputs = tuple(t.clone().requires_grad_() for t in [*(x.U[t] for t in leaves), *(x.B[t] for t in nodes)])
    assert torch.autograd.gradcheck(f, inputs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ requires_grad / requires_grad_ / detach ~~~~~~~~~~~~~~~~~~~~~~~~~


def test_factories_default_to_no_grad():
    assert not hiepy.ones(SHAPE).requires_grad
    assert not hiepy.randn(SHAPE, rank=2).requires_grad


@pytest.mark.parametrize(
    "factory",
    [
        lambda: hiepy.zeros(SHAPE, requires_grad=True),
        lambda: hiepy.ones(SHAPE, requires_grad=True),
        lambda: hiepy.constant(SHAPE, 2.0, requires_grad=True),
        lambda: hiepy.arange(SHAPE, requires_grad=True),
        lambda: hiepy.randn(SHAPE, rank=2, requires_grad=True),
        lambda: hiepy.hilbert(SHAPE, requires_grad=True),
    ],
    ids=["zeros", "ones", "constant", "arange", "randn", "hilbert"],
)
def test_factories_create_grad_leaves(factory: Callable[[], HTTensor]):
    x = factory()
    assert x.requires_grad
    assert all(t.requires_grad and t.is_leaf for t in _components(x))


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
