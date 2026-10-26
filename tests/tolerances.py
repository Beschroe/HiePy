"""Error model of the test suite: every numerical check of ``tests/INVARIANTS.md`` goes through this module.

A check has the form

    error ≤ bound + rel · scale

- ``bound`` is the mathematical bound of the property (0 for exact operations, ``atol + rtol·‖x‖`` for a
  truncation, ``sqrt(Σσ²)`` for discarded singular values),
- ``rel`` is the relative rounding error, determined by the ``Accuracy`` class of the computed quantities and the
  machine epsilon ``ε`` of the least precise dtype involved,
- ``scale`` is the size of the quantities entering the computation (e.g. ``m(x)``, see ``magnitude``).

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


def magnitude(x: HTTensor) -> float:
    """``m(x)``: product of the spectral norms of all components (transfer tensors matricized as ``(r_l r_r) x r_t``).

    Upper bound of ``‖x‖`` that equals it for orthonormal components; rounding errors of operations on ``x`` scale
    with ``m(x)`` rather than with ``‖x‖`` when the representation contains cancellation.
    """
    result = 1.0
    for t in x.dtree.get_nodes():
        C = x.U[t] if x.dtree.is_leaf(t) else x.B[t].reshape(-1, x.B[t].shape[2])
        result *= torch.linalg.matrix_norm(C.detach().double(), ord=2).item()
    return result


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
    scale_label: str = "input size",
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> None:
    """Exact operations: ``‖actual − expected‖ ≤ SAFETY · ε · max(‖expected‖, scale)``.

    ``scale`` is the size of the inputs entering the result, so that cancellation does not demand an error below
    rounding; ``dtypes`` are the precisions involved (default: those of ``actual`` and ``expected``).
    """
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    dtypes = dtypes or (actual.dtype, expected.dtype)
    error = torch.linalg.norm(actual.double() - expected.double()).item()
    reference = torch.linalg.norm(expected.double()).item()
    size, label = (scale, scale_label) if scale is not None and scale > reference else (reference, "‖expected‖")
    rel = relative_error(Accuracy.ROUNDING, *dtypes)
    Bound(0.0, rel, size, _rel_label(Accuracy.ROUNDING, dtypes), label).check(error)


def assert_within(
    error: float,
    bound: float,
    *,
    scale: float,
    scale_label: str = "m(x)",
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
