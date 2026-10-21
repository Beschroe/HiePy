"""Tests for dtype and device handling.

The rules under test are:

- every operation returns the dtype of its inputs, all components of an HT tensor share one dtype and one device,
- operations with several tensor operands return the promoted (more precise) dtype,
- Python scalars and zero-dimensional tensors do not raise the dtype (torch's type promotion rules),
- operands on different devices raise a RuntimeError before any computation,
- factories default to ``torch.get_default_dtype()`` and ``torch.get_default_device()``.

Device conflicts are tested with the ``meta`` device, so no GPU is needed. The tests marked ``cuda`` run only if a
GPU is available.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import hiepy
from hiepy import HTTensor

SHAPE = (3, 4, 5)
FLOAT_DTYPES = [torch.float32, torch.float64]
MIXED_DTYPES = [(torch.float32, torch.float64), (torch.float64, torch.float32)]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~


def _rand(dtype: torch.dtype, device: torch.device | str = "cpu", seed: int = 0) -> HTTensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return hiepy.randn(SHAPE, rank=2, generator=gen, dtype=dtype, device=device)


def _assert_components(x: HTTensor, dtype: torch.dtype, device: torch.device | str = "cpu") -> None:
    """Check dtype and device of the HT tensor and of all of its components."""
    assert x.dtype == dtype
    assert x.device == torch.device(device)
    for component in [*x.U.values(), *x.B.values()]:
        assert component.dtype == dtype
        assert component.device == torch.device(device)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Factories ~~~~~~~~~~~~~~~~~~~~~~~~~

FACTORIES: dict[str, Callable[..., HTTensor]] = {
    "zeros": lambda **kw: hiepy.zeros(SHAPE, **kw),
    "ones": lambda **kw: hiepy.ones(SHAPE, **kw),
    "ones_orthog": lambda **kw: hiepy.ones(SHAPE, orthog=True, **kw),
    "constant": lambda **kw: hiepy.constant(SHAPE, c=2, **kw),
    "constant_orthog": lambda **kw: hiepy.constant(SHAPE, c=2.5, orthog=True, **kw),
    "arange": lambda **kw: hiepy.arange(SHAPE, **kw),
    "randn": lambda **kw: hiepy.randn(SHAPE, rank=2, **kw),
    "hilbert": lambda **kw: hiepy.hilbert(SHAPE, M=3, **kw),
}


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_factory_respects_dtype(name: str, dtype: torch.dtype):
    _assert_components(FACTORIES[name](dtype=dtype), dtype)


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("default", FLOAT_DTYPES)
def test_factory_defaults_to_torch_default_dtype(name: str, default: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        _assert_components(FACTORIES[name](), default)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("name", FACTORIES)
def test_factory_respects_device(name: str):
    _assert_components(FACTORIES[name](dtype=torch.float32, device="meta"), torch.float32, "meta")


def test_constant_value_is_cast_to_dtype():
    x = hiepy.constant(SHAPE, c=torch.tensor(1.5, dtype=torch.float64), dtype=torch.float32)
    _assert_components(x, torch.float32)
    assert torch.allclose(x.full(), torch.full(SHAPE, 1.5))


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
    assert torch.allclose(y.full().to(src), x.full(), atol=1e-5)


def test_to_changes_device():
    _assert_components(hiepy.to(_rand(torch.float32), device="meta"), torch.float32, "meta")


def test_to_keeps_autograd_graph():
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        x = hiepy.ones(SHAPE, dtype=torch.float32, requires_grad=True)
        y = x.to(dtype=torch.float64)
        y.B[y.dtree.get_root()].sum().backward()
        assert x.B[x.dtree.get_root()].grad is not None
    finally:
        torch.set_grad_enabled(previous)


def test_save_load_map_location(tmp_path: pytest.TempPathFactory):
    path = tmp_path / "x.htt"  # type: ignore[operator]
    _rand(torch.float32).save(path)
    _assert_components(HTTensor.load(path, map_location="meta"), torch.float32, "meta")


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Single-operand operations preserve the dtype ~~~~~~~~~~~~~~~~~~~~~~~~~

HT_RESULT_OPS: dict[str, Callable[[HTTensor], HTTensor]] = {
    "copy": lambda x: x.copy(),
    "clone": lambda x: x.clone(),
    "detach": lambda x: x.detach(),
    "orthog": lambda x: (x + x).orthog(),
    "truncate_svd": lambda x: (x + x).truncate({"atol": 1e-4}, method="svd")[0],
    "truncate_gram": lambda x: (x + x).truncate({"atol": 1e-4}, method="gram")[0],
    "ele_mul_self": lambda x: hiepy.ele_mul(x, x, {"max_rank": 3}),
    "scalar_mul_float": lambda x: x * 2.0,
    "scalar_mul_int": lambda x: 3 * x,
    "mode_mul": lambda x: x.mode_mul(torch.ones(2, 4, dtype=x.dtype), dim=1),
    "ele_mode_mul": lambda x: x.ele_mode_mul(torch.arange(4), dim=1),
    "get": lambda x: x[1:, :, 0:3],  # type: ignore[return-value]
    "squeeze": lambda x: hiepy.squeeze(x[:, 0:1, :]),  # type: ignore[arg-type,return-value]
    "to_htt": lambda x: hiepy.to_htt(x.full(), {"atol": 1e-4}),
}

DENSE_RESULT_OPS: dict[str, Callable[[HTTensor], torch.Tensor]] = {
    "full": lambda x: x.full(),
    "norm": lambda x: x.norm(),
    "contract": lambda x: x @ x,
    "get_entry": lambda x: x[0, 1, 2],  # type: ignore[return-value]
    "get_vector": lambda x: x[0, 1],  # type: ignore[return-value]
    "gramians": lambda x: x.gramians()[(0,)],
    "local_left_svd_vectors": lambda x: hiepy.local_left_svd(x)[(0,)][0],
    "local_left_svd_values": lambda x: hiepy.local_left_svd(x)[(0, 1, 2)][1],
}


@pytest.mark.parametrize("name", HT_RESULT_OPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ht_result_keeps_dtype(name: str, dtype: torch.dtype):
    _assert_components(HT_RESULT_OPS[name](_rand(dtype)), dtype)


@pytest.mark.parametrize("name", DENSE_RESULT_OPS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dense_result_keeps_dtype(name: str, dtype: torch.dtype):
    assert DENSE_RESULT_OPS[name](_rand(dtype)).dtype == dtype


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
    assert torch.allclose((x + y).full(), expected, atol=1e-6)


@pytest.mark.parametrize(("dtype_x", "dtype_y"), MIXED_DTYPES)
def test_contract_promotes_dtype(dtype_x: torch.dtype, dtype_y: torch.dtype):
    x, y = _rand(dtype_x, seed=0), _rand(dtype_y, seed=1)
    result = hiepy.contract(x, y)
    assert result.dtype == torch.float64
    assert torch.allclose(result, (x.full().double() * y.full().double()).sum(), atol=1e-6)


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
@pytest.mark.parametrize("name", HT_RESULT_OPS)
def test_ht_result_stays_on_cuda(name: str):
    if name == "mode_mul":
        result = _rand(torch.float32, "cuda").mode_mul(torch.ones(2, 4, device="cuda"), dim=1)
    elif name == "ele_mode_mul":
        result = _rand(torch.float32, "cuda").ele_mode_mul(torch.arange(4, device="cuda"), dim=1)
    else:
        result = HT_RESULT_OPS[name](_rand(torch.float32, "cuda"))
    _assert_components(result, torch.float32, result.device)
    assert result.device.type == "cuda"


@cuda
@pytest.mark.parametrize("name", DENSE_RESULT_OPS)
def test_dense_result_stays_on_cuda(name: str):
    assert DENSE_RESULT_OPS[name](_rand(torch.float32, "cuda")).device.type == "cuda"


@cuda
def test_cuda_cpu_conflict():
    with pytest.raises(RuntimeError, match="same device"):
        _rand(torch.float32, "cuda") + _rand(torch.float32)
