"""Error model of the test suite: every numerical check of ``tests/INVARIANTS.md`` goes through this module.

A check has the form

    error ≤ bound + rel · scale

- ``bound`` is the mathematical bound of the property (0 for exact operations, ``atol + rtol·‖x‖`` for a
  truncation, ``sqrt(Σσ²)`` for discarded singular values),
- ``rel`` is the relative rounding error, determined by the ``Accuracy`` class of the computed quantities and the
  machine epsilon ``ε`` of the least precise dtype involved,
- ``scale`` is the size of the computation: the same computation carried out with the absolute values of all
  components and arguments (``abs_full``, ``abs_arg``). This is the textbook bound for rounding,
  ``|fl(Σ aᵢbᵢ) − Σ aᵢbᵢ| ≤ γ·Σ|aᵢ||bᵢ|``; it captures cancellation in the operation and in the representation.

The only tuning knob is ``SAFETY``. A failing check reports all three parts, so it is visible which part of the
model is too tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from hiepy import HTTensor

# Safety factor on the machine epsilon; 45 · ε(float64) ≈ 1e-14.
SAFETY = 45


class Accuracy(Enum):
    """How accurately a quantity can be computed, relative to the scale of the inputs."""

    ROUNDING = "rounding"
    """Direct computation (products, QR, SVD): relative error ``SAFETY · ε``."""

    GRAMIAN = "gramian"
    """Singular values as square roots of Gramian eigenvalues (``truncate_gram``): an eigenvalue is accurate to
    ``ε·‖x‖²``, so its square root only to ``sqrt(ε)·‖x‖``."""


def machine_eps(*dtypes: torch.dtype) -> float:
    """Machine epsilon of the least precise of ``dtypes``."""
    return max(torch.finfo(dtype).eps for dtype in dtypes)


def relative_error(accuracy: Accuracy, *dtypes: torch.dtype) -> float:
    """Relative rounding error of a quantity of the given accuracy class, computed in ``dtypes``."""
    eps = machine_eps(*dtypes)
    return SAFETY * eps if accuracy is Accuracy.ROUNDING else eps**0.5


def resolution(accuracy: Accuracy, *dtypes: torch.dtype) -> float:
    """Smallest relative size a quantity of the given accuracy class can resolve: 0 for direct computations,
    ``sqrt(ε)`` for singular values from Gramian eigenvalues (``truncate_gram``, ``ele_mul``; Q4). Truncation
    tolerances below it are not tested for these routines."""
    return 0.0 if accuracy is Accuracy.ROUNDING else machine_eps(*dtypes) ** 0.5


def abs_full(x: HTTensor) -> torch.Tensor:
    """Dense float64 tensor reconstructed from the absolute values of all components of ``x``.

    Rounding in any computation with ``x`` is bounded by the same computation on ``abs_full(x)`` (times a small
    multiple of ε); ``‖abs_full(x)‖ ≥ ‖x‖``, with equality if no cancellation occurs in the representation.
    """
    U = {t: u.detach().abs().double() for t, u in x.U.items()}
    B = {t: b.detach().abs().double() for t, b in x.B.items()}
    return HTTensor(U=U, B=B, dtree=x.dtree).full()


def abs_arg(arg: object) -> object:
    """Absolute value of the numbers in an operation argument (tensors as float64, Python floats); indices, keys,
    dtypes and other objects are returned unchanged. Tuples are processed entrywise."""
    if isinstance(arg, torch.Tensor):
        return arg.detach().abs().double()
    if isinstance(arg, float):
        return abs(arg)
    if isinstance(arg, tuple):
        return tuple(abs_arg(a) for a in arg)
    return arg


def size(T: torch.Tensor) -> float:
    """Norm of the result of an absolute-value computation (absolute value for a scalar)."""
    return torch.linalg.norm(T.double()).item()


@dataclass(frozen=True)
class Bound:
    """``bound + rel · scale``, with labels that explain each part in a failure message."""

    bound: float
    rel: float
    scale: float
    rel_label: str
    scale_label: str

    @property
    def total(self) -> float:
        return self.bound + self.rel * self.scale

    def check(self, error: float, what: str = "error") -> None:
        """Assert ``error ≤ total`` and explain the bound otherwise."""
        assert error <= self.total, (
            f"{what} {error:.4e} > {self.total:.4e} = bound {self.bound:.4e} "
            f"+ rel {self.rel:.2e} [{self.rel_label}] · scale {self.scale:.4e} [{self.scale_label}]"
        )


def _rel_label(accuracy: Accuracy, dtypes: tuple[torch.dtype, ...]) -> str:
    name = str(min(dtypes, key=lambda d: torch.finfo(d).bits)).removeprefix("torch.")
    return f"{SAFETY}·ε({name})" if accuracy is Accuracy.ROUNDING else f"sqrt(ε({name}))"


def assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    scale: float | None = None,
    scale_label: str = "|·| computation",
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> None:
    """Exact operations: ``‖actual − expected‖ ≤ SAFETY · ε · scale``.

    ``scale`` is the size of the computation with absolute values (default: ``‖expected‖``, sufficient when the
    computation has no cancellation); ``dtypes`` are the precisions involved (default: those of ``actual`` and
    ``expected``).
    """
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    dtypes = dtypes or (actual.dtype, expected.dtype)
    error = torch.linalg.norm(actual.double() - expected.double()).item()
    if scale is None:
        scale, scale_label = size(expected), "‖expected‖"
    rel = relative_error(Accuracy.ROUNDING, *dtypes)
    Bound(0.0, rel, scale, _rel_label(Accuracy.ROUNDING, dtypes), scale_label).check(error)


def assert_within(
    error: float,
    bound: float,
    *,
    scale: float,
    scale_label: str = "|·| computation",
    accuracy: Accuracy = Accuracy.ROUNDING,
    dtypes: tuple[torch.dtype, ...] = (torch.float64,),
) -> None:
    """Approximations: ``error ≤ bound + rel(accuracy, dtypes) · scale``."""
    rel = relative_error(accuracy, *dtypes)
    Bound(bound, rel, scale, _rel_label(accuracy, dtypes), scale_label).check(error)


def assert_orthonormal(F: torch.Tensor, what: str = "frame", dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    """``‖FᵀF − I‖_F ≤ SAFETY · ε · k`` for a matrix with ``k`` columns (the deviation grows with ``k``).

    ``dtypes`` are the precisions ``F`` was computed in (default: its own dtype), e.g. float32 for a float32 frame
    converted to float64.
    """
    dtypes = dtypes or (F.dtype,)
    k = F.shape[1]
    deviation = torch.linalg.norm(F.T @ F - torch.eye(k, dtype=F.dtype, device=F.device)).item()
    rel = relative_error(Accuracy.ROUNDING, *dtypes)
    Bound(0.0, rel, float(k), _rel_label(Accuracy.ROUNDING, dtypes), "columns k").check(
        deviation, f"‖FᵀF − I‖ of {what}"
    )
