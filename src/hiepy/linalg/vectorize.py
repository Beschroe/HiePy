"""
Tensor vectorization utilities.

Provides functions for vectorizing (flattening) and devectorizing
(reshaping) PyTorch tensors.
"""

from math import prod

import torch

from hiepy.utils.data_types import Shape


def vectorize(x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the vectorization of a tensor.

    Transforms a tensor of arbitrary shape into a one-dimensional tensor containing the same entries.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        One-dimensional vectorization of ``x``.

    Notes
    -----
    - The reshaping is performed in row-major (C-style) order.
    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> hiepy.vectorize(x)
    tensor([1, 2, 3, 4])
    """

    return x.flatten()


def devectorize(x: torch.Tensor, shape: Shape) -> torch.Tensor:
    r"""
    Reconstructs a tensor from its vectorization.

    This operation is the inverse of `vectorize`.

    Parameters
    ----------
    x : torch.Tensor
        One-dimensional tensor to be reshaped.

    shape : Shape
        Target shape of the reconstructed tensor.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``shape`` containing the entries of ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional.

    ValueError
        If the number of elements of ``x`` does not match `prod(shape)`.

    Notes
    -----
    - The tensor is reconstructed using row-major (C-style) order.
    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> v = torch.tensor([1, 2, 3, 4])
    >>> hiepy.devectorize(v, (2, 2))
    tensor([[1, 2],
            [3, 4]])
    """
    if x.ndim != 1:
        raise ValueError(f"Argument 'x' must be a 1D tensor, got {x.ndim}D tensor.")

    # Check if the vector size matches the exact capacity of the target shape
    if x.numel() != prod(shape):
        raise ValueError(
            f"Vector elements ({x.numel()}) do not match the total capacity "
            f"of the target shape {shape} ({prod(shape)})."
        )

    # Reshape
    return x.reshape(tuple(shape))
