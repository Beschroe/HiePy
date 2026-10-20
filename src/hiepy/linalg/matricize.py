"""torch.Tensor reshape utilities.

This module provides utility functions to perform tensor unfolding
(matricization) and folding (dematricization) operations on PyTorch tensors.
"""

from math import prod
from typing import Sequence

import torch

from hiepy.utils.data_types import Shape


def matricize(x: torch.Tensor, t: Sequence[int]) -> torch.Tensor:
    r"""
    Computes the $t$-matricization (unfolding) of a multi-dimensional tensor.

    Flattens the specified modes in the tuple $t$ into the row dimension of a
    2D matrix, while the remaining unmapped modes are flattened into the column
    dimension.

    The shape of the resulting matrix will be:

    $$\left(\prod_{\mu \in t} \text{shape}[\mu], \prod_{\mu \notin t} \text{shape}[\mu]\right)$$

    Parameters
    ----------
    x : torch.Tensor
        The multi-dimensional source tensor to be matricized.
    t : Sequence[int]
        The tensor modes (dimensions) that are combined to form the row indices
        of the output matrix.

    Returns
    -------
    torch.Tensor
        The 2D matricized representation of the input tensor.

    Notes
    -----
    - All reshapes are performed in C-style order.
    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> x = torch.randn(3, 4, 5, 6)
    >>> hiepy.matricize(x, (0, 1)).shape
    torch.Size([12, 30])
    >>> hiepy.matricize(x, (1,)).shape
    torch.Size([4, 90])
    >>> hiepy.matricize(x, (1, 3)).shape
    torch.Size([24, 15])
    """

    # Resolve negative modes and validate bounds against tensor order
    t_norm = tuple(m if m >= 0 else x.ndim + m for m in t)

    if any(m < 0 or m >= x.ndim for m in t_norm):
        raise ValueError(
            f"All modes in 't' must be within the valid range [{-x.ndim}, {x.ndim - 1}] "
            f"for a {x.ndim}D tensor, got {t}."
        )

    if len(set(t_norm)) != len(t_norm):
        raise ValueError(f"Argument 't' contains duplicate modes after resolving indices: {t}.")

    # Permute modes so that the resolved row-modes become leading
    x_moved_axis = torch.movedim(x, source=t_norm, destination=tuple(range(len(t_norm))))

    # Calculate the exact row dimension using normalized modes
    nr_rows = prod(x.size()[mode] for mode in t_norm)

    # Reshape into a 2D matrix
    return x_moved_axis.reshape(nr_rows, -1)


def dematricize(A: torch.Tensor, shape: Shape, t: Sequence[int]) -> torch.Tensor:
    r"""
    Reconstructs the original tensor from its $t$-matricized matrix representation.

    This operation is the strict mathematical inverse of the `matricise` function.
    Supports negative indexing for the modes tuple ``t``.

    Parameters
    ----------
    A : torch.Tensor
        The 2D matricized tensor (matrix) to be folded back.
    shape : Shape
        The target shape of the original multi-dimensional tensor.
    t : Sequence[int]
        The tensor modes that originally formed the row indices of the matrix $A$.

    Returns
    -------
    torch.Tensor
        The refolded, multi-dimensional tensor matching the given target ``shape``.

    Raises
    ------
    ValueError
        If dimensionality constraints, value ranges, or dimension mappings
        are violated, out of bounds, or structurally incompatible.

    Notes
    -----
    - All reshapes are performed in C-style order.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> shape = (2,3,4,5)
    >>> t = (1,3)
    >>> A = torch.randn(shape)
    >>> Amat = hiepy.matricize(A,t)
    >>> Ademat = hiepy.dematricize(Amat, shape,t)
    >>> Ademat.shape
    torch.Size([2, 3, 4, 5])
    >>> torch.allclose(A,Ademat)
    True
    """
    # Input validity checks
    if A.ndim != 2:
        raise ValueError(f"Argument 'A' must be a 2D tensor, got {A.ndim}D tensor.")

    if not all(item > 0 for item in shape):
        raise ValueError("Argument 'shape' must contain positive integers only.")

    if len(t) < 1:
        raise ValueError("Argument 't' must contain at least one dimension index.")

    # Resolve negative modes and validate bounds against target shape order
    ndim = len(shape)
    t_norm = tuple(m if m >= 0 else ndim + m for m in t)

    if any(m < 0 or m >= ndim for m in t_norm):
        raise ValueError(
            f"All modes in 't' must be within the valid range [{-ndim}, {ndim - 1}] "
            f"for a target target shape of length {ndim}, got {t}."
        )

    if len(set(t_norm)) != len(t_norm):
        raise ValueError(f"Argument 't' contains duplicate modes after resolving indices: {t_norm}.")

    # Determine target shape for the intermediate un-flattened but permuted matrix
    col_dims = (dim for dim in range(ndim) if dim not in t_norm)
    sh = tuple(shape[dim] for dim in t_norm) + tuple(shape[dim] for dim in col_dims)

    # Check if the matrix total size matches the target shape multiplication
    if A.numel() != prod(shape):
        raise ValueError(
            f"Matrix elements ({A.numel()}) do not match the total size of the target shape {shape} ({prod(shape)})."
        )

    # Reshape matrix back to intermediate tensor and invert permutation
    x = A.reshape(*sh)
    return torch.movedim(x, source=tuple(range(len(t_norm))), destination=t_norm)
