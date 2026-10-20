"""HTTensor scalar multiplication module.

This module provides functionality to compute the product of a scalar
and an HT  tensor.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import ScalarType


def scalar_mul(x: HTTensor, c: ScalarType) -> HTTensor:
    r"""
    Scales HT tensor ``x`` by scalar ``c``.

    The scalar multiplication is efficiently implemented by scaling the transfer
    tensor at the root node by ``c``.

    Parameters
    ----------
    x : HTTensor
        Input tensor to be scaled.

    c : ScalarType
        Input scalar to scale ``x``. Can be a float, int, or a torch.Tensor.

    Returns
    -------
    HTTensor
        A new HT tensor representing the product $cX$.

    Notes
    -----
    - The dimension tree is preserved.
    - The hierarchical ranks are preserved.
    - The tensor shape is preserved.
    - Orthogonality is preserved.
    - The tensor is scaled globally by the factor ``c``.
    - Only the root transfer tensor is modified; all other transfer tensors and leaf matrices remain unchanged.
    - Computational cost scales linearly with the number of entries in the root transfer tensor
      of shape $r_l \times r_r \times 1$, where $r_l$ and $r_r$ are the hierarchical ranks of its left and right
      children, respectively.

    Examples
    --------
    >>> import hiepy
    >>> x = hiepy.ones((2,2,2))
    >>> x = hiepy.scalar_mul(x=x,c=3.0)
    >>> print(x.full())
    tensor([[[3., 3.],
         [3., 3.]],
        [[3., 3.],
         [3., 3.]]])
    """
    # Create copy of x
    x = x.copy()

    # Extract root transfer tensor
    B_root = x.B[x.dtree.root]

    # (Cast c to tensor)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=B_root.dtype)

    # Scale root transfer tensor by c
    x.B[x.dtree.root] = B_root * c

    return x
