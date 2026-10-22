"""Registry of the operations under test.

Every public operation that returns an ``HTTensor`` (or a result containing one) is registered here once, together
with the properties ``tests/INVARIANTS.md`` expects of it. The invariant tests iterate over this registry, so a new
operation only needs a new entry here.

All operations take two HT tensors ``x`` and ``y`` of the same shape and tree (unary operations ignore ``y``) and one
further argument ``arg`` (a matrix, key, options, ...). The invariant tests draw ``arg`` from ``Op.args(x)``;
tests with fixed inputs use ``Op.run``, which takes ``Op.default(x)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from hiepy.utils.data_types import IndexKey

type Result = HTTensor | torch.Tensor | tuple[HTTensor, dict[hiepy.Node, torch.Tensor]] | dict
type DenseOp = Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]


def _none(x: HTTensor) -> object:
    return None


def _no_args(x: HTTensor) -> st.SearchStrategy[Any]:
    return st.none()


def _never(x: HTTensor, arg: object) -> bool:
    return False


def _same_dtype(arg: object) -> torch.dtype | None:
    return None


def _sum_of_magnitudes(x: HTTensor, y: HTTensor, arg: object) -> float:
    return ht_magnitude(x) + ht_magnitude(y)


def _product_of_magnitudes(x: HTTensor, y: HTTensor, arg: object) -> float:
    return ht_magnitude(x) * ht_magnitude(y)


def _magnitude(x: HTTensor, y: HTTensor, arg: object) -> float:
    return ht_magnitude(x)


@dataclass(frozen=True)
class Op:
    """An operation under test and the properties expected of it.

    Attributes
    ----------
    call
        Applies the operation to ``(x, y, arg)``.
    dense
        Dense counterpart applied to ``(x.full(), y.full(), arg)`` (N1, N2); ``None`` if the operation is not exact.
    flag
        Expected ``is_orthog`` of an ``HTTensor`` result, given ``(x, arg)`` (O2); ``None`` if the operation does
        not return an ``HTTensor``.
    args
        Strategy for ``arg``, given ``x``.
    default
        Fixed ``arg`` for tests that do not draw arguments, given ``x``.
    shares_storage
        Whether the result may share component storage and the ``DimTree`` object with ``x`` (V2 exception).
    out_dtype
        dtype of the result if it differs from the input dtype (D1), given ``arg``.
    magnitude
        Size of the quantities entering the result, given ``(x, y, arg)``, built from ``ht_magnitude``. N1 compares
        relative to the larger of this and the norm of the reference, so that cancellation (in the operation or
        inside the HT representation) does not demand an error below rounding.
    """

    call: Callable[[HTTensor, HTTensor, Any], Result]
    dense: DenseOp | None
    flag: Callable[[HTTensor, Any], bool] | None
    args: Callable[[HTTensor], st.SearchStrategy[Any]] = _no_args
    default: Callable[[HTTensor], Any] = _none
    shares_storage: Callable[[HTTensor, Any], bool] = _never
    out_dtype: Callable[[Any], torch.dtype | None] = _same_dtype
    magnitude: Callable[[HTTensor, HTTensor, Any], float] = _magnitude

    def run(self, x: HTTensor, y: HTTensor) -> Result:
        """Apply the operation with its default argument."""
        return self.call(x, y, self.default(x))


def ht_magnitude(x: HTTensor) -> float:
    """Product of the spectral norms of all components (transfer tensors matricized as ``(r_l r_r) x r_t``).

    Upper bound of ``‖x‖`` that equals it for orthonormal components; rounding errors of operations on ``x`` scale
    with this quantity rather than with ``‖x‖`` when the representation contains cancellation.
    """
    result = 1.0
    for t in x.dtree.get_nodes():
        C = x.U[t] if x.dtree.is_leaf(t) else x.B[t].reshape(-1, x.B[t].shape[2])
        result *= torch.linalg.matrix_norm(C.detach().double(), ord=2).item()
    return result


def ht_result(result: Result) -> HTTensor | None:
    """Return the ``HTTensor`` contained in ``result`` (first entry of a ``truncate*`` tuple), else ``None``."""
    if isinstance(result, tuple):
        result = result[0]
    return result if isinstance(result, HTTensor) else None


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Argument strategies ~~~~~~~~~~~~~~~~~~~~~~~~~


def _seeded(rows: int, cols: int, seed: int, x: HTTensor) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=gen, dtype=x.dtype).to(x.device)


def _mode_mul_args(x: HTTensor) -> st.SearchStrategy[tuple[torch.Tensor, int]]:
    """Matrix and mode for ``mode_mul``: any mode, 1–4 output rows (1 creates a singleton mode), random or zero."""

    def build(dim: int, rows: int, seed: int, zero: bool) -> tuple[torch.Tensor, int]:
        A = _seeded(rows, x.shape[dim], seed, x)
        return (A * 0.0 if zero else A), dim

    return st.builds(build, st.integers(0, x.order - 1), st.integers(1, 4), st.integers(0, 2**31 - 1), st.booleans())


def _ele_mode_mul_args(x: HTTensor) -> st.SearchStrategy[tuple[torch.Tensor, int]]:
    """Vector and mode for ``ele_mode_mul``: any mode, random, with a zero entry, or all zero."""

    def build(dim: int, seed: int, kind: str) -> tuple[torch.Tensor, int]:
        v = _seeded(x.shape[dim], 1, seed, x)[:, 0]
        if kind == "zero_entry":
            v[0] = 0.0
        elif kind == "zero":
            v = v * 0.0
        return v, dim

    kinds = st.sampled_from(["random", "zero_entry", "zero"])
    return st.builds(build, st.integers(0, x.order - 1), st.integers(0, 2**31 - 1), kinds)


SCALARS = (3.0, -0.5, 0.0, 2, 1e-3)


def _scalar_args(x: HTTensor) -> st.SearchStrategy[float | torch.Tensor]:
    """Python scalars (incl. 0 and negative) and 0-d tensors."""
    tensors = st.sampled_from(SCALARS).map(lambda c: torch.tensor(float(c), dtype=torch.float64))
    return st.one_of(st.sampled_from(SCALARS), tensors)


@st.composite
def _keys(draw: st.DrawFn, shape: tuple[int, ...]) -> IndexKey | tuple[IndexKey, ...]:
    """torch-style keys of ints and non-empty slices with positive step, addressing a prefix of the modes."""
    entries: list[IndexKey] = []
    for n in shape[: draw(st.integers(0, len(shape)))]:
        if draw(st.booleans()):
            entries.append(draw(st.integers(-n, n - 1)))
        else:
            start = draw(st.integers(-n, n - 1))
            stop = draw(st.one_of(st.none(), st.integers(start % n + 1, n + 2)))
            entries.append(slice(start, stop, draw(st.integers(1, 3))))
    if len(entries) == 1 and draw(st.booleans()):
        return entries[0]
    return tuple(entries)


def _selects_all(key: IndexKey | tuple[IndexKey, ...], shape: tuple[int, ...]) -> bool:
    entries = key if isinstance(key, tuple) else (key,)
    return all(isinstance(e, slice) and e.indices(n) == (0, n, 1) for e, n in zip(entries, shape))


def _squeezed_dims(shape: tuple[int, ...], dims: tuple[int, ...] | None = None) -> list[int]:
    """Modes removed by ``squeeze(x, dims)``: the requested (default: all) singletons, except the highest ones needed
    to keep order >= 2. If all modes are requested, all are removed and the result is a dense 0-d tensor."""
    removed = [i for i, n in enumerate(shape) if n == 1] if dims is None else sorted(set(dims))
    if len(removed) < len(shape):
        while len(shape) - len(removed) < 2:
            removed.remove(max(removed))
    return removed


def _squeeze_args(x: HTTensor) -> st.SearchStrategy[int | tuple[int, ...] | None]:
    """``None`` (all singletons), a single singleton mode, or any subset of the singleton modes."""
    singletons = [i for i, n in enumerate(x.shape) if n == 1]
    if not singletons:
        return st.sampled_from([None, ()])
    subsets = st.lists(st.sampled_from(singletons), unique=True).map(tuple)
    return st.one_of(st.none(), st.sampled_from(singletons), subsets)


def _as_tuple(dims: int | tuple[int, ...] | None) -> tuple[int, ...] | None:
    return (dims,) if isinstance(dims, int) else dims


def _squeeze_dense(X: torch.Tensor, Y: torch.Tensor, dims: int | tuple[int, ...] | None) -> torch.Tensor:
    removed = _squeezed_dims(tuple(X.shape), _as_tuple(dims))
    return X.squeeze(tuple(removed)) if removed else X


def _trunc_opts(with_rtol: bool = True) -> st.SearchStrategy[hiepy.TruncOpts]:
    """Truncation options: any combination of ``atol``, ``rtol`` (optional) and ``max_rank``."""
    tols = st.sampled_from([1e-12, 1e-6, 1e-2, 0.5])
    keys: dict[str, st.SearchStrategy[Any]] = {"atol": tols, "max_rank": st.integers(1, 4)}
    if with_rtol:
        keys["rtol"] = tols
    return st.fixed_dictionaries({}, optional=keys)  # type: ignore[return-value]


def _mode_mul_dense(X: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.movedim(torch.tensordot(A, X, dims=([1], [dim])), 0, dim)


def _ele_mode_mul_dense(X: torch.Tensor, v: torch.Tensor, dim: int) -> torch.Tensor:
    shape = [1] * X.ndim
    shape[dim] = -1
    return X * v.reshape(shape)


def _default_matrix(x: HTTensor) -> tuple[torch.Tensor, int]:
    n = x.shape[1]
    return torch.linspace(-1.0, 1.0, 2 * n, dtype=x.dtype, device=x.device).reshape(2, n), 1


def _default_vector(x: HTTensor) -> tuple[torch.Tensor, int]:
    return torch.linspace(0.5, 2.0, x.shape[-1], dtype=x.dtype, device=x.device), x.order - 1


def _flag_as_input(x: HTTensor, arg: object) -> bool:
    return x.is_orthog


def _flag_true(x: HTTensor, arg: object) -> bool:
    return True


def _flag_false(x: HTTensor, arg: object) -> bool:
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Operations ~~~~~~~~~~~~~~~~~~~~~~~~~

OPS: dict[str, Op] = {
    "add": Op(lambda x, y, a: x + y, lambda X, Y, a: X + Y, _flag_false, magnitude=_sum_of_magnitudes),
    "sub": Op(lambda x, y, a: x - y, lambda X, Y, a: X - Y, _flag_false, magnitude=_sum_of_magnitudes),
    "scalar_mul": Op(
        lambda x, y, c: hiepy.scalar_mul(x, c),
        lambda X, Y, c: c * X,
        _flag_as_input,
        args=_scalar_args,
        default=lambda x: 3.0,
        magnitude=lambda x, y, c: abs(float(c)) * ht_magnitude(x),
    ),
    "mode_mul": Op(
        lambda x, y, a: hiepy.mode_mul(x, *a),
        lambda X, Y, a: _mode_mul_dense(X, *a),
        _flag_false,
        args=_mode_mul_args,
        default=_default_matrix,
        magnitude=lambda x, y, a: torch.linalg.matrix_norm(a[0], ord=2).item() * ht_magnitude(x),
    ),
    "ele_mode_mul": Op(
        lambda x, y, a: hiepy.ele_mode_mul(x, *a),
        lambda X, Y, a: _ele_mode_mul_dense(X, *a),
        _flag_false,
        args=_ele_mode_mul_args,
        default=_default_vector,
        magnitude=lambda x, y, a: a[0].abs().max().item() * ht_magnitude(x),
    ),
    "ele_mul": Op(lambda x, y, a: x * y, lambda X, Y, a: X * Y, _flag_false, magnitude=_product_of_magnitudes),
    "ele_mul_truncated": Op(
        lambda x, y, opts: hiepy.ele_mul(x, y, opts),
        None,
        _flag_false,
        args=lambda x: _trunc_opts(with_rtol=False),
        default=lambda x: {"max_rank": 2},
    ),
    "contract": Op(lambda x, y, a: x @ y, lambda X, Y, a: (X * Y).sum(), None, magnitude=_product_of_magnitudes),
    "norm": Op(lambda x, y, a: hiepy.norm(x), lambda X, Y, a: torch.linalg.norm(X), None),
    "orthog": Op(lambda x, y, a: hiepy.orthog(x), lambda X, Y, a: X, _flag_true),
    "truncate_svd": Op(
        lambda x, y, opts: hiepy.truncate_svd(x, opts),
        None,
        _flag_false,
        args=lambda x: _trunc_opts(),
        default=lambda x: {"max_rank": 1},
    ),
    "truncate_gram": Op(
        lambda x, y, opts: hiepy.truncate_gram(x, opts),
        None,
        _flag_false,
        args=lambda x: _trunc_opts(),
        default=lambda x: {"max_rank": 1},
    ),
    "local_left_svd": Op(lambda x, y, a: hiepy.local_left_svd(x), None, None),
    "full": Op(lambda x, y, a: x.full(), lambda X, Y, a: X, None),
    # get: a slice keeps its mode, an int removes it; the flag survives only if every entry is selected
    "get": Op(
        lambda x, y, key: x[key],
        lambda X, Y, key: X[key],
        lambda x, key: x.is_orthog and _selects_all(key, x.shape),
        args=lambda x: _keys(x.shape),
        default=lambda x: slice(None, 1),
    ),
    "get_int": Op(lambda x, y, a: x[0], lambda X, Y, a: X[0], _flag_false),
    "get_entry": Op(lambda x, y, a: x[(0,) * x.order], lambda X, Y, a: X[(0,) * X.ndim], None),
    # squeeze keeps the flag only if it removes no mode
    "squeeze": Op(
        lambda x, y, dims: hiepy.squeeze(x, dims),
        _squeeze_dense,
        lambda x, dims: x.is_orthog and not _squeezed_dims(x.shape, _as_tuple(dims)),
        args=_squeeze_args,
    ),
    "copy": Op(lambda x, y, a: x.copy(), lambda X, Y, a: X, _flag_as_input),
    "clone": Op(lambda x, y, a: x.clone(), lambda X, Y, a: X, _flag_as_input),
    "detach": Op(lambda x, y, a: x.detach(), lambda X, Y, a: X, _flag_as_input, shares_storage=lambda x, a: True),
    # to shares the components that need no conversion
    "to": Op(
        lambda x, y, dtype: x.to(dtype=dtype),
        lambda X, Y, dtype: X.to(dtype),
        _flag_as_input,
        args=lambda x: st.sampled_from([torch.float32, torch.float64]),
        default=lambda x: torch.float32,
        shares_storage=lambda x, dtype: dtype == x.dtype,
        out_dtype=lambda dtype: dtype,
    ),
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
