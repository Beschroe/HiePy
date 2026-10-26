"""Tests for dtype and device handling.

The rules under test are:

- every operation returns the dtype of its inputs, all components of an HT tensor share one dtype and one device (D1),
- operations with several tensor operands return the promoted (more precise) dtype (D2),
- Python scalars and zero-dimensional tensors do not raise the dtype (torch's type promotion rules) (D3),
- operands on different devices raise a RuntimeError before any computation (D2),
- factories default to ``torch.get_default_dtype()`` and ``torch.get_default_device()`` (D4),
- the ``HTTensor`` constructor rejects mixed dtypes and devices (E4).

IDs refer to ``tests/INVARIANTS.md``.

Device conflicts are tested with the ``meta`` device, so no GPU is needed. The tests marked ``cuda`` run only if a
GPU is available.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import hiepy
from hiepy import HTTensor
from tests.helpers import components, cuda
from tests.ops import FACTORIES, OPS, Result
from tests.tolerances import assert_exact

SHAPE = (3, 4, 5)
FLOAT_DTYPES = [torch.float32, torch.float64]
MIXED_DTYPES = [(torch.float32, torch.float64), (torch.float64, torch.float32)]

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~


def _rand(dtype: torch.dtype, device: torch.device | str = "cpu", seed: int = 0) -> HTTensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return hiepy.randn(SHAPE, rank=2, generator=gen, dtype=dtype, device=device)


def _assert_components(x: HTTensor, dtype: torch.dtype, device: torch.device | str = "cpu") -> None:
    """Check dtype and device of the HT tensor and of all of its components."""
    assert x.dtype == dtype
    assert x.device == torch.device(device)
    for component in components(x):
        assert component.dtype == dtype
        assert component.device == torch.device(device)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Factories ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_factory_respects_dtype(name: str, dtype: torch.dtype):
    _assert_components(FACTORIES[name](SHAPE, dtype=dtype), dtype)


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("default", FLOAT_DTYPES)
def test_factory_defaults_to_torch_default_dtype(name: str, default: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        _assert_components(FACTORIES[name](SHAPE), default)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", FACTORIES)
def test_factory_respects_device(name: str):
    _assert_components(FACTORIES[name](SHAPE, dtype=torch.float32, device="meta"), torch.float32, "meta")


def test_constant_value_is_cast_to_dtype():
    x = hiepy.constant(SHAPE, c=torch.tensor(1.5, dtype=torch.float64), dtype=torch.float32)
    _assert_components(x, torch.float32)
    assert_exact(x.full(), torch.full(SHAPE, 1.5))


def test_arange_values_in_float32():
    x = hiepy.arange((2, 3, 4), dtype=torch.float32)
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(x.full(), expected)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ HTTensor: construction, dtype, device, to ~~~~~~~~~~~~~~~~~~~~~~~~~


def test_constructor_rejects_mixed_dtypes():
    x = _rand(torch.float64)
    U = dict(x.U)
    U[(0,)] = U[(0,)].float()
    with pytest.raises(TypeError, match="same dtype"):
        HTTensor(U=U, B=x.B, dtree=x.dtree)


def test_constructor_rejects_mixed_devices():
    x = _rand(torch.float64)
    B = dict(x.B)
    B[(0, 1)] = B[(0, 1)].to("meta")
    with pytest.raises(RuntimeError, match="same device"):
        HTTensor(U=x.U, B=B, dtree=x.dtree)


@pytest.mark.parametrize(("src", "dst"), MIXED_DTYPES)
def test_to_changes_dtype(src: torch.dtype, dst: torch.dtype):
    x = _rand(src)
    y = x.to(dtype=dst)
    _assert_components(y, dst)
    _assert_components(x, src)
    assert y.is_orthog == x.is_orthog
    assert_exact(y.full().to(src), x.full(), dtypes=(src, dst))


def test_to_changes_device():
    _assert_components(hiepy.to(_rand(torch.float32), device="meta"), torch.float32, "meta")


@pytest.mark.usefixtures("grad_enabled")
def test_to_keeps_autograd_graph():
    x = hiepy.ones(SHAPE, dtype=torch.float32, requires_grad=True)
    y = x.to(dtype=torch.float64)
    y.B[y.dtree.get_root()].sum().backward()
    assert x.B[x.dtree.get_root()].grad is not None


def test_save_load_map_location(tmp_path: pytest.TempPathFactory):
    path = tmp_path / "x.htt"  # type: ignore[operator]
    _rand(torch.float32).save(path)
    _assert_components(HTTensor.load(path, map_location="meta"), torch.float32, "meta")


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Single-operand operations preserve the dtype ~~~~~~~~~~~~~~~~~~~~~~~~~

# Cases not covered by the registry in tests/ops.py: integer scalars and vectors, dispatch via ``truncate``,
# ``to_htt`` and the Gramians.
EXTRA_OPS: dict[str, Callable[[HTTensor], Result]] = {
    "truncate_svd_atol": lambda x: (x + x).truncate({"atol": 1e-4}, method="svd"),
    "truncate_gram_atol": lambda x: (x + x).truncate({"atol": 1e-4}, method="gram"),
    "scalar_mul_int": lambda x: 3 * x,
    "ele_mode_mul_int": lambda x: x.ele_mode_mul(torch.arange(4, device=x.device), dim=1),
    "to_htt": lambda x: hiepy.to_htt(x.full(), {"atol": 1e-4}),
    "gramians": lambda x: x.gramians()[(0,)],
    "orthog_non_orthogonal": lambda x: (x + x).orthog(),
    "squeeze_singleton": lambda x: hiepy.squeeze(x[:, 0:1, :]),  # type: ignore[arg-type]
    "get_vector": lambda x: x[0, 1],
}

RESULT_OPS: dict[str, Callable[[HTTensor], Result]] = {
    **{name: (lambda x, op=op: op.run(x, x)) for name, op in OPS.items() if name != "to"},
    **EXTRA_OPS,
}


def _result_tensors(result: Result) -> list[torch.Tensor]:
    """All tensors contained in the result of an operation."""
    if isinstance(result, tuple):
        return [*components(result[0]), *result[1].values()]
    if isinstance(result, HTTensor):
        return components(result)
    if isinstance(result, dict):
        return [t for value in result.values() for t in value]
    return [result]


@pytest.mark.parametrize("name", RESULT_OPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_result_keeps_dtype(name: str, dtype: torch.dtype):
    """D1: results have the dtype of the input."""
    assert all(t.dtype == dtype for t in _result_tensors(RESULT_OPS[name](_rand(dtype))))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Mixed dtypes are promoted ~~~~~~~~~~~~~~~~~~~~~~~~~

BINARY_HT_OPS: dict[str, Callable[[HTTensor, HTTensor], HTTensor]] = {
    "add": hiepy.add,
    "sub": hiepy.sub,
    "ele_mul": lambda x, y: hiepy.ele_mul(x, y, {"max_rank": 3}),
}


@pytest.mark.parametrize("name", BINARY_HT_OPS)
@pytest.mark.parametrize(("dtype_x", "dtype_y"), MIXED_DTYPES)
def test_binary_op_promotes_dtype(name: str, dtype_x: torch.dtype, dtype_y: torch.dtype):
    x, y = _rand(dtype_x, seed=0), _rand(dtype_y, seed=1)
    result = BINARY_HT_OPS[name](x, y)
    _assert_components(result, torch.float64)
    # Inputs are not modified
    _assert_components(x, dtype_x)
    _assert_components(y, dtype_y)


@pytest.mark.parametrize(("dtype_x", "dtype_y"), MIXED_DTYPES)
def test_add_mixed_dtypes_is_correct(dtype_x: torch.dtype, dtype_y: torch.dtype):
    x, y = _rand(dtype_x, seed=0), _rand(dtype_y, seed=1)
    expected = x.full().double() + y.full().double()
    assert_exact((x + y).full(), expected, dtypes=(dtype_x, dtype_y))


@pytest.mark.parametrize(("dtype_x", "dtype_y"), MIXED_DTYPES)
def test_contract_promotes_dtype(dtype_x: torch.dtype, dtype_y: torch.dtype):
    x, y = _rand(dtype_x, seed=0), _rand(dtype_y, seed=1)
    result = hiepy.contract(x, y)
    assert result.dtype == torch.float64
    expected = (x.full().double() * y.full().double()).sum()
    assert_exact(result, expected, size=x.norm().item() * y.norm().item(), dtypes=(dtype_x, dtype_y))


@pytest.mark.parametrize(("dtype_x", "dtype_A"), MIXED_DTYPES)
def test_mode_mul_promotes_dtype(dtype_x: torch.dtype, dtype_A: torch.dtype):
    x = _rand(dtype_x)
    A = torch.randn(2, 4, dtype=dtype_A)
    _assert_components(hiepy.mode_mul(x, A, dim=1), torch.float64)


@pytest.mark.parametrize(("dtype_x", "dtype_v"), MIXED_DTYPES)
def test_ele_mode_mul_promotes_dtype(dtype_x: torch.dtype, dtype_v: torch.dtype):
    x = _rand(dtype_x)
    v = torch.randn(4, dtype=dtype_v)
    _assert_components(hiepy.ele_mode_mul(x, v, dim=1), torch.float64)


@pytest.mark.parametrize("c", [2.0, torch.tensor(2.0, dtype=torch.float64)])
def test_scalar_mul_does_not_promote(c: float | torch.Tensor):
    _assert_components(hiepy.scalar_mul(_rand(torch.float32), c), torch.float32)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Device conflicts ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", [*BINARY_HT_OPS, "contract"])
def test_binary_op_rejects_device_conflict(name: str):
    op = hiepy.contract if name == "contract" else BINARY_HT_OPS[name]
    x = _rand(torch.float64)
    y = hiepy.ones(SHAPE, dtype=torch.float64, device="meta")
    with pytest.raises(RuntimeError, match="same device"):
        op(x, y)


def test_mode_mul_rejects_device_conflict():
    with pytest.raises(RuntimeError, match="same device"):
        hiepy.mode_mul(_rand(torch.float64), torch.ones(2, 4, device="meta"), dim=1)


def test_ele_mode_mul_rejects_device_conflict():
    with pytest.raises(RuntimeError, match="same device"):
        hiepy.ele_mode_mul(_rand(torch.float64), torch.ones(4, device="meta"), dim=1)


def test_scalar_mul_rejects_device_conflict():
    with pytest.raises(RuntimeError, match="same device"):
        hiepy.scalar_mul(_rand(torch.float64), torch.tensor(2.0, device="meta"))


def test_scalar_mul_accepts_cpu_scalar_for_other_device():
    x = hiepy.ones(SHAPE, dtype=torch.float32, device="meta")
    _assert_components(hiepy.scalar_mul(x, torch.tensor(2.0)), torch.float32, "meta")


# ~~~~~~~~~~~~~~~~~~~~~~~~~ GPU ~~~~~~~~~~~~~~~~~~~~~~~~~


@cuda
@pytest.mark.parametrize("name", RESULT_OPS)
def test_result_stays_on_cuda(name: str):
    """D1: results stay on the device of the input."""
    assert all(t.device.type == "cuda" for t in _result_tensors(RESULT_OPS[name](_rand(torch.float32, "cuda"))))


@cuda
def test_cuda_cpu_conflict():
    with pytest.raises(RuntimeError, match="same device"):
        _rand(torch.float32, "cuda") + _rand(torch.float32)
