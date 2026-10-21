"""HTTensor contraction module.

This module provides functionality to compute the full tensor contraction
(inner product) of two HT tensors sharing the same dimension tree.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor
from hiepy.utils.dtype_device import check_device, result_dtype


def contract(x: HTTensor, y: HTTensor) -> torch.Tensor:
    r"""
    Computes the full inner product of two HT tensors.

    This operation evaluates the scalar contraction
        $\langle X, Y \rangle = \sum_{i_1,\dots,i_d} X_{i_1\dots i_d} Y_{i_1\dots i_d}$
    without unfolding the full tensors, by performing a recursive contraction along the dimension tree.

    Parameters
    ----------
    x : HTTensor
        First HT tensor. Must share the same dimension tree and mode sizes as ``y``.

    y : HTTensor
        Second HT tensor. Must share the same dimension tree and mode sizes as ``x``.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the full inner product of ``x``and ``y``.

    Raises
    ------
    ValueError
        If the dimension trees of ``x`` and ``y`` differ.

    ValueError
        If the tensor shapes of ``x`` and ``y`` do not match.

    RuntimeError
        If ``x`` and ``y`` live on different devices.

    Notes
    -----
    - The result has the promoted dtype of ``x`` and ``y`` (e.g. ``float32`` and ``float64`` give ``float64``).
    - The operation's stability is improved when ``x`` and ``y`` are orthogonal.
    - No recompression is required, as the result is scalar-valued.
    - The arithmetic cost of the operation scales as $\mathcal{O}(dnr_xr_y + dr_x^2r_y^2)$,
      where $d$ is the tensor order, $n$ the maximum mode size, and $r_x$, $r_y$ the maximum
      HT ranks of ``x`` and ``y``, respectively.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = 8,8,8,8
    >>> x = torch.randn(shape)
    >>> y = torch.randn(shape)
    >>> prod_torch = torch.sum(x*y)
    >>> opts = hiepy.TruncOpts({"atol":1e-10})
    >>> x_ht = hiepy.to_htt(x,opts)
    >>> y_ht = hiepy.to_htt(y,opts)
    >>> prod_ht = hiepy.contract(x_ht,y_ht)
    >>> torch.allclose(prod_torch, prod_ht)
    True
    """
    # Validity checks
    if not x.dtree == y.dtree:
        raise ValueError("Dimension trees of 'x' and 'y' are not compatible")
    if x.shape != y.shape:
        raise ValueError(f"Shape of 'y' {y.shape} does not match shape of 'x' {x.shape}")
    check_device(x=x, y=y)
    dtype = result_dtype(x, y)
    x, y = x.to(dtype=dtype), y.to(dtype=dtype)

    dtree = x.dtree
    Ux, Uy = x.U, y.U
    Bx, By = x.B, y.B
    M = {}

    # Contract leaf matrices
    for leaf in dtree.get_leaves():
        M[leaf] = Ux[leaf].T @ Uy[leaf]

    # Contract M and transfer tensors bottom-up along the tree
    for level in dtree.get_levels(order="bottom-up"):
        for t in level:
            if dtree.is_leaf(t):
                continue
            left, right = dtree.get_children(t)
            M_ = torch.tensordot(M[left], By[t], dims=([1], [0]))  # type:ignore
            M_ = torch.tensordot(Bx[t], M_, dims=([0], [0]))  # type:ignore
            M_ = torch.tensordot(M_, M[right], dims=([0, 2], [0, 1]))  # type:ignore
            M[t] = M_

    return M[dtree.root][0, 0]
