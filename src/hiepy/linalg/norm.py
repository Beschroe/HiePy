"""HTTensor norm utility.

This module provides functionality to compute the Frobenius
norm of a Hierarchical Tucker (HT) tensor..
"""

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.transformation.orthog import orthog


def norm(x: HTTensor) -> torch.Tensor:
    r"""
    Computes the Frobenius norm of a Hierarchical Tucker (HT) tensor.

    Parameters
    ----------
    x : HTTensor
        The Hierarchical Tucker tensor whose Frobenius norm is to be computed.

    Returns
    -------
    torch.Tensor
        A 0-dimensional tensor (scalar) representing the Frobenius norm of ``x``.

    Notes
    -----
    - The norm is read off the root transfer tensor of the orthogonalized tensor via ``torch.linalg.norm``. Like
      that function, it overflows to ``inf`` for norms beyond about $10^{154}$ in float64, the supported range of
      hiepy (truncation routines raise ``ValueError`` there).

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = (2,2,2)
    >>> x = hiepy.ones(shape)
    >>> hiepy.norm(x)
    tensor(2.8284)
    """
    if not x.is_orthog:
        x = orthog(x)
    B_root = x.B[x.dtree.get_root()]
    return torch.Tensor(torch.linalg.norm(B_root))
