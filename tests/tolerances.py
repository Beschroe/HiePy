"""Error model of the test suite: every numerical check of ``tests/INVARIANTS.md`` goes through this module.

The model follows the LAPACK test suite. Each check computes a *test ratio*

    ratio = (error − bound) / (ε · size of the inputs)

and requires ``ratio ≤ THRESH``:

- ``error`` is the measured deviation, e.g. ``‖result.full() − reference‖``,
- ``bound`` is the mathematical bound of the property (0 for exact operations, ``atol + rtol·‖x‖`` for a
  truncation, ``sqrt(Σσ²)`` for discarded singular values),
- ``ε`` is the machine epsilon of the least precise dtype involved,
- the *size of the inputs* is the largest norm the result can have by the usual norm inequalities, e.g. ``‖x‖`` for
  unary operations, ``‖x‖ + ‖y‖`` for ``x − y`` (``tests.ops.Op.input_size``). Measuring against the inputs rather
  than the result keeps the check meaningful when the result is small through cancellation.

A test therefore detects every deviation larger than ``THRESH · ε`` times the size of the inputs. Singular values
computed from Gramian eigenvalues (``truncate_gram``) are only accurate to ``sqrt(ε)`` times the size of the inputs;
for them the margin is ``sqrt(ε) · size`` (``Accuracy.GRAMIAN``). A failing check reports the ratio and its parts.
"""

from __future__ import annotations

from enum import Enum

import torch

from hiepy import HTTensor

# Largest admissible test ratio. LAPACK uses 30; the measured maximum on the inputs of tests/strategies.py is 163
# (rank-deficient representations by design), so 1000 leaves about 6x room (INVARIANTS.md, Q12).
THRESH = 1000


class Accuracy(Enum):
    """How accurately a quantity can be computed, relative to the size of the inputs."""

    ROUNDING = "rounding"
    """Direct computation (products, QR, SVD): margin ``THRESH · ε · size``."""

    GRAMIAN = "gramian"
    """Singular values as square roots of Gramian eigenvalues (``truncate_gram``): an eigenvalue is accurate to
    ``ε·‖x‖²``, so its square root only to ``sqrt(ε)·‖x‖``; margin ``sqrt(ε) · size``."""


def machine_eps(*dtypes: torch.dtype) -> float:
    """Machine epsilon of the least precise of ``dtypes``."""
    return max(torch.finfo(dtype).eps for dtype in dtypes)


def resolution(accuracy: Accuracy, *dtypes: torch.dtype) -> float:
    """Smallest relative size a quantity of the given accuracy class can resolve: 0 for direct computations,
    ``sqrt(ε)`` for singular values from Gramian eigenvalues (``truncate_gram``, ``ele_mul``; Q4). Truncation
    tolerances below it are not tested for these routines."""
    return 0.0 if accuracy is Accuracy.ROUNDING else machine_eps(*dtypes) ** 0.5


def frobenius(t: HTTensor | torch.Tensor) -> float:
    """Frobenius norm of a dense tensor or of the dense form of an HT tensor (in float64)."""
    dense = t.full() if isinstance(t, HTTensor) else t
    return torch.linalg.norm(dense.detach().double()).item()


def _check(
    error: float,
    bound: float,
    size: float,
    size_label: str,
    accuracy: Accuracy,
    dtypes: tuple[torch.dtype, ...],
    what: str,
) -> None:
    """Assert ``(error − bound) / (ε · size) ≤ limit`` and report the ratio and its parts otherwise."""
    eps = machine_eps(*dtypes)
    limit = THRESH if accuracy is Accuracy.ROUNDING else eps**-0.5
    if error <= bound + limit * eps * size:
        return
    name = str(min(dtypes, key=lambda d: torch.finfo(d).bits)).removeprefix("torch.")
    ratio = (error - bound) / (eps * size) if size > 0 else float("inf")
    limit_label = f"THRESH {THRESH}" if accuracy is Accuracy.ROUNDING else f"1/sqrt(ε) {limit:.1e} (Gramian)"
    raise AssertionError(
        f"{what}: test ratio {ratio:.3g} > {limit_label}; ratio = (error {error:.4e} − bound {bound:.4e}) "
        f"/ (ε({name}) {eps:.2e} · {size_label} {size:.4e})"
    )


def assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    size: float | None = None,
    size_label: str = "size of inputs",
    dtypes: tuple[torch.dtype, ...] | None = None,
) -> None:
    """Exact operations: ``‖actual − expected‖ / (ε · size) ≤ THRESH``.

    ``size`` is the size of the inputs (default: ``‖expected‖``, for operations that do not change the size, such
    as copies and round trips); ``dtypes`` are the precisions involved (default: those of ``actual`` and
    ``expected``).
    """
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    dtypes = dtypes or (actual.dtype, expected.dtype)
    error = torch.linalg.norm(actual.double() - expected.double()).item()
    if size is None:
        size, size_label = frobenius(expected), "‖expected‖"
    _check(error, 0.0, size, size_label, Accuracy.ROUNDING, dtypes, "deviation")


def assert_within(
    error: float,
    bound: float,
    *,
    size: float,
    size_label: str = "‖x‖",
    accuracy: Accuracy = Accuracy.ROUNDING,
    dtypes: tuple[torch.dtype, ...] = (torch.float64,),
) -> None:
    """Approximations: ``(error − bound) / (ε · size) ≤ THRESH`` (Gramian accuracy: margin ``sqrt(ε) · size``)."""
    _check(error, bound, size, size_label, accuracy, dtypes, "error beyond bound")


def assert_orthonormal(F: torch.Tensor, what: str = "frame", dtypes: tuple[torch.dtype, ...] | None = None) -> None:
    """``‖FᵀF − I‖_F / (k · ε) ≤ THRESH`` for a matrix with ``k`` columns, as in LAPACK's orthogonality tests.

    ``dtypes`` are the precisions ``F`` was computed in (default: its own dtype), e.g. float32 for a float32 frame
    converted to float64.
    """
    dtypes = dtypes or (F.dtype,)
    k = F.shape[1]
    deviation = torch.linalg.norm(F.T @ F - torch.eye(k, dtype=F.dtype, device=F.device)).item()
    _check(deviation, 0.0, float(k), "columns k", Accuracy.ROUNDING, dtypes, f"‖FᵀF − I‖ of {what}")
