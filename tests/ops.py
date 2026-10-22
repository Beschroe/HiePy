"""Registry of the operations under test.

Every public operation that returns an ``HTTensor`` (or a result containing one) is registered here once, together
with the properties ``tests/INVARIANTS.md`` expects of it. The invariant tests iterate over this registry, so a new
operation only needs a new entry here.

All operations take two HT tensors ``x`` and ``y`` of the same shape and tree; unary operations ignore ``y``.
Additional arguments (matrices, vectors, keys) are derived from ``x.shape``, so every entry works for any shape of
order >= 2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

import hiepy
from hiepy import HTTensor

type Result = HTTensor | torch.Tensor | tuple[HTTensor, dict[hiepy.Node, torch.Tensor]] | dict
type DenseOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Op:
    """An operation under test and the properties expected of it.

    Attributes
    ----------
    call
        Applies the operation to ``(x, y)``.
    dense
        Dense counterpart applied to ``(x.full(), y.full())`` (N1, N2); ``None`` if the operation is not exact.
    flag
        Expected ``is_orthog`` of an ``HTTensor`` result, given the input ``x`` (O2); ``None`` if the operation does
        not return an ``HTTensor``.
    keeps_tree
        Whether the result has the dimension tree of ``x`` (S4).
    shares_storage
        Whether the result may share component storage and the ``DimTree`` object with ``x`` (V2 exception).
    out_dtype
        dtype of the result if it differs from the input dtype (D1), e.g. for ``to``.
    """

    call: Callable[[HTTensor, HTTensor], Result]
    dense: DenseOp | None
    flag: Callable[[HTTensor], bool] | None
    keeps_tree: bool = True
    shares_storage: bool = False
    out_dtype: torch.dtype | None = None


def ht_result(result: Result) -> HTTensor | None:
    """Return the ``HTTensor`` contained in ``result`` (first entry of a ``truncate*`` tuple), else ``None``."""
    if isinstance(result, tuple):
        result = result[0]
    return result if isinstance(result, HTTensor) else None


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Arguments derived from the shape ~~~~~~~~~~~~~~~~~~~~~~~~~


def _matrix(x: HTTensor) -> torch.Tensor:
    """2 x n_1 matrix for ``mode_mul`` along mode 1."""
    n = x.shape[1]
    return torch.linspace(-1.0, 1.0, 2 * n, dtype=x.dtype, device=x.device).reshape(2, n)


def _vector(x: HTTensor) -> torch.Tensor:
    """Positive vector for ``ele_mode_mul`` along the last mode."""
    return torch.linspace(0.5, 2.0, x.shape[-1], dtype=x.dtype, device=x.device)


def _mode_mul_dense(X: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.movedim(torch.tensordot(A, X, dims=([1], [dim])), 0, dim)


def _squeeze_dense(X: torch.Tensor) -> torch.Tensor:
    """Dense counterpart of ``squeeze``: remove singleton modes, but keep order >= 2 unless all modes are singletons."""
    dims = [i for i, n in enumerate(X.shape) if n == 1]
    if len(dims) == X.ndim:
        return X.squeeze()
    while X.ndim - len(dims) < 2:
        dims.remove(max(dims))
    return X.squeeze(tuple(dims)) if dims else X


def _flag_as_input(x: HTTensor) -> bool:
    return x.is_orthog


def _flag_true(x: HTTensor) -> bool:
    return True


def _flag_false(x: HTTensor) -> bool:
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Operations ~~~~~~~~~~~~~~~~~~~~~~~~~

OPS: dict[str, Op] = {
    "add": Op(lambda x, y: x + y, lambda X, Y: X + Y, _flag_false),
    "sub": Op(lambda x, y: x - y, lambda X, Y: X - Y, _flag_false),
    "scalar_mul": Op(lambda x, y: 3.0 * x, lambda X, Y: 3.0 * X, _flag_as_input),
    "mode_mul": Op(
        lambda x, y: hiepy.mode_mul(x, _matrix(x), 1),
        lambda X, Y: _mode_mul_dense(X, torch.linspace(-1.0, 1.0, 2 * X.shape[1], dtype=X.dtype).reshape(2, -1), 1),
        _flag_false,
    ),
    "ele_mode_mul": Op(
        lambda x, y: hiepy.ele_mode_mul(x, _vector(x), x.order - 1),
        lambda X, Y: X * torch.linspace(0.5, 2.0, X.shape[-1], dtype=X.dtype),
        _flag_false,
    ),
    "ele_mul": Op(lambda x, y: x * y, lambda X, Y: X * Y, _flag_false),
    "ele_mul_truncated": Op(lambda x, y: hiepy.ele_mul(x, y, {"max_rank": 2}), None, _flag_false),
    "contract": Op(lambda x, y: x @ y, lambda X, Y: (X * Y).sum(), None),
    "norm": Op(lambda x, y: hiepy.norm(x), lambda X, Y: torch.linalg.norm(X), None),
    "orthog": Op(lambda x, y: hiepy.orthog(x), lambda X, Y: X, _flag_true),
    "truncate_svd": Op(lambda x, y: hiepy.truncate_svd(x, {"max_rank": 1}), None, _flag_false),
    "truncate_gram": Op(lambda x, y: hiepy.truncate_gram(x, {"max_rank": 1}), None, _flag_false),
    "local_left_svd": Op(lambda x, y: hiepy.local_left_svd(x), None, None),
    "full": Op(lambda x, y: x.full(), lambda X, Y: X, None),
    # get: a slice keeps the mode; the flag survives only if every entry is selected
    "get": Op(lambda x, y: x[:1], lambda X, Y: X[:1], lambda x: x.is_orthog and x.shape[0] == 1),
    "get_all": Op(lambda x, y: x[()], lambda X, Y: X, _flag_as_input),
    # get with an int removes the mode; order-2 inputs give a dense vector
    "get_int": Op(lambda x, y: x[0], lambda X, Y: X[0], _flag_false, keeps_tree=False),
    "get_entry": Op(lambda x, y: x[(0,) * x.order], lambda X, Y: X[(0,) * X.ndim], None, keeps_tree=False),
    "squeeze": Op(lambda x, y: hiepy.squeeze(x), lambda X, Y: _squeeze_dense(X), _flag_false, keeps_tree=False),
    "copy": Op(lambda x, y: x.copy(), lambda X, Y: X, _flag_as_input),
    "clone": Op(lambda x, y: x.clone(), lambda X, Y: X, _flag_as_input),
    "detach": Op(lambda x, y: x.detach(), lambda X, Y: X, _flag_as_input, shares_storage=True),
    "to": Op(lambda x, y: x.to(dtype=torch.float32), lambda X, Y: X.float(), _flag_as_input, out_dtype=torch.float32),
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Factories ~~~~~~~~~~~~~~~~~~~~~~~~~

type Factory = Callable[..., HTTensor]

# Each factory takes the shape and forwards keyword arguments (dtype, device, requires_grad).
FACTORIES: dict[str, Factory] = {
    "zeros": lambda shape, **kw: hiepy.zeros(shape, **kw),
    "ones": lambda shape, **kw: hiepy.ones(shape, **kw),
    "ones_orthog": lambda shape, **kw: hiepy.ones(shape, orthog=True, **kw),
    "constant": lambda shape, **kw: hiepy.constant(shape, c=2, **kw),
    "constant_orthog": lambda shape, **kw: hiepy.constant(shape, c=2.5, orthog=True, **kw),
    "arange": lambda shape, **kw: hiepy.arange(shape, **kw),
    "randn": lambda shape, **kw: hiepy.randn(shape, rank=2, **kw),
    "hilbert": lambda shape, **kw: hiepy.hilbert(shape, M=3, **kw),
}

# Expected is_orthog of each factory's result (O2).
FACTORY_FLAGS: dict[str, bool] = {
    "zeros": False,
    "ones": False,
    "ones_orthog": True,
    "constant": False,
    "constant_orthog": True,
    "arange": False,
    "randn": True,
    "hilbert": False,
}
