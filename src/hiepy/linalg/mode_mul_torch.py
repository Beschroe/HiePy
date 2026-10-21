"""
Tensor mode-product utilities.

Provides functions for multiplying one or multiple matrices with a
`torch.Tensor` along specified modes.
"""

import math
from typing import Sequence

import torch


def mode_mul_torch(x: torch.Tensor, A: torch.Tensor, dim: Sequence[int] | int) -> torch.Tensor:
    r"""
    Computes the mode product of a matrix with a tensor.

    Given a tensor $X\in \mathbb{R}^{n_1 \times \dots \times n_d}$ and a matrix
    $A \in \mathbb{R}^{m \times \prod_{i=1}^k n_{\mu_i}}$,
    this function contracts the modes
    $\mu_1,\ldots,\mu_k$ of $X$ with the columns of $A$. The resulting
    tensor has the contracted modes replaced by a single mode of size $m$.

    Mathematically, the operation is denoted by

    $$
    Y = A \times_{\mu_1,\dots,\mu_k} X.
    $$

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary order.

    A : torch.Tensor
        A 2D matrix of shape
        $(m, \prod_{i=1}^k n_{\mu_i})$, where the second dimension equals
        the product of the sizes of the contracted modes.

    dim : Sequence[int] | int
        Mode index or sequence of mode indices along which the contraction
        is performed. Negative indices are supported.

    Returns
    -------
    torch.Tensor
        Tensor obtained after replacing the contracted modes by a
        single mode of size `A.shape[0]` located at the position of
        the first contracted mode.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional.

    ValueError
        If any specified mode is out of bounds.

    ValueError
        If duplicate mode indices are provided.

    ValueError
        If the product of the sizes of the contracted modes does not match
        the number of columns of ``A``.


    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> torch.set_default_dtype(torch.float64)
    >>> x = torch.randn(2, 3, 4, 5)
    >>> # Contract modes 1 and 3 (sizes 3 and 5 -> combined size 15)
    >>> A = torch.randn(10, 15)
    >>> y = hiepy.mode_mul_torch(x=x, A=A, dim=(1, 3))
    >>> y.shape
    torch.Size([2, 10, 4])
    """
    order = x.ndim

    # Input normalization
    raw_modes = (dim,) if isinstance(dim, int) else tuple(dim)

    # Resolve negative indices
    resolved_modes = tuple(mu + order if mu < 0 else mu for mu in raw_modes)

    # Check input validity
    if A.ndim != 2:
        raise ValueError(f"Argument 'A' must be a 2D matrix, but got shape {A.shape}")

    if not all(0 <= mu < order for mu in resolved_modes):
        raise ValueError(f"At least one mode in {raw_modes} is out of range for a tensor with ndim={order}")

    if len(set(resolved_modes)) != len(resolved_modes):
        raise ValueError(f"Duplicate entries in modes are not allowed: {raw_modes}")

    # Calculate dimensions
    target_shape = tuple(x.shape[mu] for mu in resolved_modes)
    contraction_modes_sz = math.prod(target_shape)

    if contraction_modes_sz != A.shape[1]:
        raise ValueError(
            f"Dimension mismatch: The targeted modes of x have a combined size of {contraction_modes_sz}, "
            f"but A expects {A.shape[1]} columns."
        )

    # Virtually split column of A into target dimensions
    A_reshaped = A.reshape(-1, *target_shape)

    # Contraction
    A_dims = list(range(1, len(resolved_modes) + 1))
    prod = torch.tensordot(x, A_reshaped, dims=(resolved_modes, A_dims))  # type: ignore

    # Move the newly introduced dimension to the correct target position
    actual_destination = sum(1 for i in range(resolved_modes[0]) if i not in resolved_modes)
    res = torch.movedim(prod, source=-1, destination=actual_destination)

    return res


def multi_mode_mul_torch(x: torch.Tensor, matrices: dict[int, torch.Tensor]) -> torch.Tensor:
    r"""
    Computes the multi-mode product of a tensor with multiple matrices.

    Given a tensor $X$ and a collection of matrices $A^{(\mu)} \in \mathbb{R}^{m_\mu \times n_\mu}$,
    this function applies the corresponding mode products

    $$
    Y = X
        \times_{\mu_1} A^{(\mu_1)}
        \times_{\mu_2} A^{(\mu_2)}
        \cdots
        \times_{\mu_k} A^{(\mu_k)}.
    $$

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    matrices : dict[int, torch.Tensor]
        Dictionary mapping tensor modes to the corresponding transformationmatrices. Negative
        mode indices are supported.

    Returns
    -------
    torch.Tensor
        Tensor obtained after applying all specified mode products.

    Raises
    ------
    ValueError
        If any matrix is not two-dimensional.

    ValueError
        If any specified mode is out of bounds.

    ValueError
        If duplicate modes are specified (including equivalent positive and negative indices).

    ValueError
        If a matrix shape is incompatible with the size of its corresponding
        tensor mode.

    Notes
    -----
    - The result is independent of the order in which the individual mode
    products are applied. However, intermediate memory usage and arithmetic
    cost depend strongly on the contraction order. Therefore, this function
    determines the contraction order using a greedy heuristic that
    prioritizes matrices with a large compression ratio, thereby reducing
    intermediate tensor sizes whenever possible.
    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = torch.randn((3, 4, 5))
    >>> matrices = {0: torch.randn(2, 3), -1: torch.randn(6, 5)}
    >>> res = hiepy.multi_mode_mul_torch(x, matrices)
    >>> res.shape
    torch.Size([2, 4, 6])
    """
    order = x.ndim
    norm_matrices = {}

    # Validate and normalize dimensions
    for dim, A in matrices.items():
        # Handle negative dimension
        norm_dim = dim + order if dim < 0 else dim

        # Range check
        if not (0 <= norm_dim < order):
            raise ValueError(f"Argument 'dim'={dim} is out of range for this torch.Tensor with ndim={order}.")

        # Duplicate dimension check
        if norm_dim in norm_matrices:
            raise ValueError(
                f"Duplicate mode detected: dimension {dim} resolves to the same "
                f"mode {norm_dim} which was already specified."
            )

        # Shape validations
        if A.ndim != 2:
            raise ValueError(f"Matrix for mode {dim} must be a 2D torch.Tensor, got shape {A.shape}.")
        if x.shape[norm_dim] != A.shape[1]:
            raise ValueError(
                f"Mismatch between dimension {dim} of x (size {x.shape[norm_dim]}) and number of columns of "
                f"matrix A (size {A.shape[1]})."
            )

        norm_matrices[norm_dim] = A

    # 2. Greedy optimization of the contraction path
    # Sort dimensions by the compression ratio:(output_size / input_size).
    # Matrices that shrink the tensor the most are applied first
    sorted_dims = sorted(norm_matrices.keys(), key=lambda d: norm_matrices[d].shape[0] / norm_matrices[d].shape[1])

    # 3. Sequential multiplication in optimized order
    res = x
    for dim in sorted_dims:
        A = norm_matrices[dim]
        res = torch.tensordot(res, A, dims=([dim], [1]))  # type:ignore
        res = torch.movedim(res, source=-1, destination=dim)

    return res


# DEPRECATED
# def mode_mul_torch(A: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
#     r"""
#     Multiplies tensor ``x`` with matrix ``A`` along mode ``dim``.

#     The mode-product of a tensor $X\in \mathbb{R}^{n_1 \times \dots \times n_d}$ with
#     a matrix $A \in \mathbb{R}^{m \times n_{\mu}}$ along mode $\mu \in \{1,\dots,i_d\}$ is defined as

#     $$ (A \times_{\mu} X)_{i_1,\dots,i_d} = \sum\limits^{n_{\mu}}_{j=1}A_{i_{\mu},j}
#     X_{i_1,\dots,i_{\mu-1},j,i_{\mu+1},\dots,i_d} .$$

#     Parameters
#     ----------
#     A : torch.Tensor
#         Input matrix of shape $(m, n_{\mu})$.

#     x : torch.Tensor
#         Input tensor to be multiplied.

#     dim : int
#         Mode along which ``x`` is multiplied with ``A``.

#     Returns
#     -------
#     torch.Tensor
#         A new tensor representing the mode product.

#     Raises
#     ------
#     ValueError
#         If ``A`` is not two-dimensional.

#     ValueError
#         If ``dim`` is outside the valid mode range.

#     ValueError
#         If the shapes of ``A`` and ``x`` don't match.

#     Examples
#     --------
#     >>> import hiepy
#     >>> import torch
#     >>> x = torch.randn((3,4,5))
#     >>> A = torch.randn(2, 4)
#     >>> res = hiepy.mode_mul_torch(A, x, dim=1)
#     >>> res.shape
#     torch.Size([3, 2, 5])
#     """
#     order = x.ndim

#     # Check input validity
#     if len(A.shape) != 2:
#         raise ValueError(f"Argument 'A' must be a 2D torch.Tensor, got shape {A.shape}")
#     if not (-order <= dim < order):
#         raise ValueError(f"Argument 'dim'={dim} is out of range for this torch.Tensor with ndim={order}")

#     # Handle negative dimension
#     if dim < 0:
#         dim += order

#     if x.shape[dim] != A.shape[1]:
#         raise ValueError(
#             f"Mismatch between dimension {dim} of x (size {x.shape[dim]}) and number of columns of "
#             f"A (size {A.shape[1]})."
#         )

#     # Tensor-Matrix-Product
#     prod = torch.tensordot(x, A, dims=([dim], [1]))  # type:ignore

#     return torch.movedim(prod, source=-1, destination=dim)
