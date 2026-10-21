"""
Implements the elementwise mode-product of a vector with an HT  tensor.

This module provides functionality to compute the elementwise mode-product of a vector
and an HT  tensor.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.dtype_device import check_device, result_dtype


def ele_mode_mul(x: HTTensor, v: torch.Tensor, dim: int) -> HTTensor:
    r"""
    Scales an HT  tensor ``x`` along its mode ``dim`` by a vector ``v``.

    The elementwise mode multiplication of a tensor $X\in\mathbb{R}^{n_0\times\dots\times n_{d-1}}$ with a vector
    $v \in \mathbb{R}^{n_{\mu}}$ along mode $\mu \in \{0,\dots,d-1\}$ is defined as:

    $$(X \star_{\mu} v)_{i_0,\dots,i_{d-1}} = X_{i_0,\dots,i_{d-1}} \cdot v_{i_\mu}$$


    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    v : torch.Tensor
        One-dimensional vector defining the mode-wise scaling factors.
        Its length must coincide with the size of mode ``dim``.

    dim : int
        Mode along which ``x`` is scaled by ``v``.

    Returns
    -------
    HTTensor
        A new HT tensor representing the scaled tensor.

    Raises
    ------
    ValueError
        If ```v``` is not one-dimensional.

    ValueError
        If ```dim``` is outside the valid mode range.

    ValueError
        If the length of ```v``` does not match the size of the selected mode.

    RuntimeError
        If ``x`` and ``v`` live on different devices.

    Notes
    -----
    - The result has the promoted dtype of ``x`` and ``v`` (e.g. ``float32`` and ``float64`` give ``float64``).
    - The dimension tree is preserved.
    - Hierarchical ranks are preserved.
    - Orthogonality is not preserved.
    - Computational cost is linear in mode size and corresponding rank, $\mathcal{O}(nr)$.
    - All transfer tensors remain unchanged.
    - All but one leaf matrices remain unchanged.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> torch.set_default_dtype(torch.float64)
    >>> x_ht = hiepy.arange((2, 3, 4))
    >>> x_ht.full()
    tensor([[[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]],
    <BLANKLINE>
            [[12., 13., 14., 15.],
             [16., 17., 18., 19.],
             [20., 21., 22., 23.]]])
    >>> v = torch.arange(3)
    >>> hiepy.ele_mode_mul(x_ht,v,dim=1).full()
    tensor([[[ 0.,  0.,  0.,  0.],
             [ 4.,  5.,  6.,  7.],
             [16., 18., 20., 22.]],
    <BLANKLINE>
            [[ 0.,  0.,  0.,  0.],
             [16., 17., 18., 19.],
             [40., 42., 44., 46.]]])
    """
    # Validity checks
    if v.ndim != 1:
        raise ValueError(f"Argument 'v' must be a 1D torch.Tensor, got shape {v.shape}")
    if dim < 0 or dim >= x.order:
        raise ValueError(f"Argument 'dim' is out of range for this HTTensor (dim={dim}, order={x.order})")
    if x.shape[dim] != v.shape[0]:
        raise ValueError(
            f"Mismatch between dimension {dim} of x (size {x.shape[dim]}) and length of v (size {v.shape[0]})."
        )

    check_device(x=x, v=v)
    dtype = result_dtype(x, v)
    x = x.clone().to(dtype=dtype)
    v = v.to(dtype=dtype)

    # Update leaf matrix corresponding to dim
    node = (dim,)
    x.U[node] = x.U[node] * (v[:, None])

    # Update is_orthog
    x.is_orthog = False

    return x
