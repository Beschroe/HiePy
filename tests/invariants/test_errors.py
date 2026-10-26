"""Error invariants E1, E2, E3, E5, E7, E8 (see ``tests/INVARIANTS.md``).

E4 is covered in ``tests/test_dtype_device.py``, E6 in ``tests/httensor/test_get_item.py``, E9–E11 in
``tests/invariants/test_errors_validation.py``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st

import hiepy
from hiepy import HTTensor
from tests.strategies import ht_tensors, tree_specs
from tests.tolerances import assert_exact

SHAPE = (3, 4, 2)

BINARY: dict[str, Callable[[HTTensor, HTTensor], object]] = {
    "add": hiepy.add,
    "sub": hiepy.sub,
    "contract": hiepy.contract,
    "ele_mul": hiepy.ele_mul,
}


def _x(shape: tuple[int, ...] = SHAPE, seed: int = 0) -> HTTensor:
    return hiepy.randn(shape, rank=2, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E1: binary operations ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("name", BINARY)
@given(data=st.data())
def test_binary_operations_reject_different_trees_or_shapes(name: str, data: st.DataObject):
    """E1: operands with different trees or shapes raise ``ValueError``."""
    x = data.draw(ht_tensors(tree=data.draw(tree_specs(), label="tree of x")), label="x")
    y = data.draw(ht_tensors(tree=data.draw(tree_specs(), label="tree of y")), label="y")
    assume(x.dtree != y.dtree or x.shape != y.shape)
    with pytest.raises(ValueError, match="not compatible|does not match"):
        BINARY[name](x, y)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E2: modes and operands ~~~~~~~~~~~~~~~~~~~~~~~~~

MODE_ERRORS: dict[str, Callable[[HTTensor], object]] = {
    "mode_mul: dim = order": lambda x: hiepy.mode_mul(x, torch.ones(2, 2, dtype=x.dtype), x.order),
    "mode_mul: dim < 0": lambda x: hiepy.mode_mul(x, torch.ones(2, x.shape[-1], dtype=x.dtype), -1),
    "mode_mul: A 1-d": lambda x: hiepy.mode_mul(x, torch.ones(x.shape[0], dtype=x.dtype), 0),
    "mode_mul: A 3-d": lambda x: hiepy.mode_mul(x, torch.ones(2, x.shape[0], 1, dtype=x.dtype), 0),
    "mode_mul: columns != size": lambda x: hiepy.mode_mul(x, torch.ones(2, x.shape[0] + 1, dtype=x.dtype), 0),
    "ele_mode_mul: dim = order": lambda x: hiepy.ele_mode_mul(x, torch.ones(2, dtype=x.dtype), x.order),
    "ele_mode_mul: dim < 0": lambda x: hiepy.ele_mode_mul(x, torch.ones(x.shape[-1], dtype=x.dtype), -1),
    "ele_mode_mul: v 2-d": lambda x: hiepy.ele_mode_mul(x, torch.ones(x.shape[0], 1, dtype=x.dtype), 0),
    "ele_mode_mul: length != size": lambda x: hiepy.ele_mode_mul(x, torch.ones(x.shape[0] + 1, dtype=x.dtype), 0),
    "squeeze: dim = order": lambda x: hiepy.squeeze(x, x.order),
    "squeeze: dim < -order": lambda x: hiepy.squeeze(x, -x.order - 1),
    "squeeze: not a singleton": lambda x: hiepy.squeeze(x, 0),
}


@pytest.mark.parametrize("name", MODE_ERRORS)
def test_invalid_modes_and_operands_raise(name: str):
    """E2: modes out of range and operands of wrong order or size raise ``ValueError``."""
    with pytest.raises(ValueError):
        MODE_ERRORS[name](_x())


def test_squeeze_rejects_non_int_dims():
    """E2: ``squeeze`` rejects ``dims`` that are neither int-like nor an iterable of int-likes."""
    with pytest.raises(TypeError):
        hiepy.squeeze(_x(), 1.5)  # type: ignore[arg-type]


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E3: gramians ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(x=ht_tensors(orthog=False))
def test_gramians_require_orthogonal_input(x: HTTensor):
    """E3: ``gramians`` raises ``ValueError`` if ``is_orthog`` is ``False``."""
    with pytest.raises(ValueError, match="not an orthogonal"):
        hiepy.gramians(x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E5: save / load ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("filename", ["x.pt", "x", "x.htt.bak"])
def test_save_and_load_require_htt_suffix(filename: str):
    """E5: ``save`` and ``load`` reject paths without the ``.htt`` suffix, for ``str`` and ``Path``."""
    with tempfile.TemporaryDirectory() as directory:
        for path in (Path(directory) / filename, str(Path(directory) / filename)):
            with pytest.raises(ValueError, match=".htt"):
                hiepy.save(_x(), path)
            with pytest.raises(ValueError, match=".htt"):
                hiepy.load(path)


def test_load_rejects_files_without_httensor():
    """E5: ``load`` raises ``TypeError`` if the file does not contain an ``HTTensor``."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.htt"
        torch.save(torch.zeros(3), path)
        with pytest.raises(TypeError, match="does not contain an HTTensor"):
            hiepy.load(path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E7: to_htt ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("shape", [(), (4,)], ids=repr)
def test_to_htt_rejects_order_below_two(shape: tuple[int, ...]):
    """E7: ``to_htt`` raises ``ValueError`` for inputs of order < 2."""
    with pytest.raises(ValueError, match="N>=2"):
        hiepy.to_htt(torch.zeros(shape, dtype=torch.float64), {})


# ~~~~~~~~~~~~~~~~~~~~~~~~~ E8: ele_mul and rtol ~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize("opts", [{"rtol": 1e-3}, {"rtol": 1e-3, "max_rank": 2}], ids=repr)
def test_ele_mul_ignores_rtol_with_warning(opts: hiepy.TruncOpts):
    """E8: ``ele_mul`` warns about ``rtol`` and returns the same result as without it."""
    x, y = _x(seed=0), _x(seed=1)
    without = {k: v for k, v in opts.items() if k != "rtol"}
    with pytest.warns(UserWarning, match="rtol"):
        result = hiepy.ele_mul(x, y, opts)
    assert_exact(result.full(), hiepy.ele_mul(x, y, without).full())  # type: ignore[arg-type]
