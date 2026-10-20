"""Diagonal tensor construction module.

This module provides functionality to construct dense tensors whose
nonzero entries are restricted to the generalized tensor diagonal.
"""

import torch

from hiepy.utils.data_types import ScalarType, Shape


def diagonal_tensor(
    shape: Shape,
    fill_value: ScalarType,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""
    Constructs a dense diagonal tensor of arbitrary order.

    The output tensor's nonzero entries are restricted to the generalized diagonal

    $T_{i_1,\ldots,i_d} = \alpha$ whenever $i_1=\cdots=i_d$,

    and are zero otherwise, where $\alpha$ denotes `fill_value`.

    Parameters
    ----------
    shape : Shape
        Tuple of integers specifying the tensor shape. Must contain at least two equal mode sizes.

    fill_value : ScalarType
        Value assigned to all diagonal entries.

    dtype : torch.dtype, optional
        Data type of the resulting tensor.

    device : torch.device or str, optional
        Device on which the tensor is allocated.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape ``shape`` whose generalized diagonal entries equal `fill_value` and
        whose remaining entries are zero.

    Raises
    ------
    ValueError
        ValueError: If fewer than two modes are given or the mode sizes differ.

    Notes
    -----
    - The result is a dense tensor and not an HT tensor.
    """
    if len(shape) < 2:
        raise ValueError(f"A diagonal tensor requires at least two modes, got shape {shape}.")

    if len(set(shape)) != 1:
        raise ValueError(f"For a well defined diagonal all modes must have the same size, got {shape}.")

    n = shape[0]
    d = len(shape)

    T = torch.zeros(shape, dtype=dtype, device=device)
    idx = torch.arange(n, device=device)
    T[(idx,) * d] = fill_value
    return T
