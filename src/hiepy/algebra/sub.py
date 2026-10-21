"""HTTensor subtraction module.

This module provides functionality to compute the formal difference of two HT tensors
by embedding their transfer tensors and leaf matrices into block-diagonal structures.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import TensorDict
from hiepy.utils.dtype_device import check_device, result_dtype


def sub(x: HTTensor, y: HTTensor) -> HTTensor:
    r"""
    Computes the difference between two HT tensors ``x`` and ``y`` via block-diagonal embedding.

    Constructs an HT  tensor representing the exact difference of two HTTensors by embedding
    their leaf matrices and transfer tensors into a block-diagonal representation.

    The resulting tensor $Z = x - y$ is represented exactly in HT form as:

    - Leaf matrices:
        $U_i^Z = [U_i^X, U_i^Y]$

    - Transfer tensors:
        $B_t^Z = \begin{cases} diag(B_t^X, -B_t^Y), & t = t_{root} \\ diag(B_t^X, B_t^Y), & t \neq t_{root} \end{cases}$

    where the block-diagonal structure is applied at each node of the dimension tree.

    Parameters
    ----------
    x : HTTensor
        First HT tensor operand. Must share the same dimension tree and
        mode sizes as ``y``.

    y : HTTensor
        Second HT tensor operand. Must share the same dimension tree and
        mode sizes as ``x``.

    Returns
    -------
    HTTensor
        A new HTTensor representing the difference x - y in block-diagonal form.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` do not share the same dimension tree or shape.

    RuntimeError
        If ``x`` and ``y`` live on different devices.

    Notes
    -----
    - This method is functionally equivalent to `hiepy.add`, with the exception that
      the root transfer tensor of ``y`` is scaled by -1.0 before embedding.
    - The result has the promoted dtype of ``x`` and ``y`` (e.g. ``float32`` and ``float64`` give ``float64``).
    - The dimension tree is preserved.
    - The tensor shape is preserved.
    - Rank growth is additive: $r_t^{out} = r_t^x + r_t^y$
    - Orthogonality is not preserved.
    - The result typically requires recompression via hiepy.truncate.
    - The arithmetic cost of the operation is negligible, as it primarily involves the
      concatenation of leaf matrices and the creation of block-diagonal transfer tensors.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.randn(shape=(8, 8, 8))
    >>> y = hiepy.randn(shape=(8, 8, 8))
    >>> diff_hiepy = hiepy.sub(x,y)
    >>> x_t = x.full()
    >>> y_t = y.full()
    >>> diff_torch = x_t - y_t
    >>> torch.allclose(diff_torch, diff_hiepy.full())
    True
    """
    # Validity checks
    if not x.dtree == y.dtree:
        raise ValueError("Dimension trees of 'x' and 'y' are not compatible")
    if x.shape != y.shape:
        raise ValueError(f"Shape of 'y' {y.shape} does not match shape of 'x' {x.shape}")
    device = check_device(x=x, y=y)
    dtype = result_dtype(x, y)
    x, y = x.to(dtype=dtype), y.to(dtype=dtype)

    U: TensorDict = {}
    B: TensorDict = {}
    dtree = x.dtree.copy()

    # Concatenate leaf matrices
    for leaf in dtree.get_leaves():
        U[leaf] = torch.cat((x.U[leaf], y.U[leaf]), dim=1)

    # Embed the transfer tensors in a 3D blockdiagonal way
    for t in dtree.get_inner_nodes():
        if dtree.is_root(t):
            xl, xr, _ = x.B[t].shape  # rank of node x children
            yl, yr, _ = y.B[t].shape  # rank of node y children
            Bt = torch.zeros(xl + yl, xr + yr, 1, dtype=dtype, device=device)
            Bt[:xl, :xr] = x.B[t]
            Bt[xl:, xr:] = y.B[t] * (-1.0)
        else:
            xl, xr, xk = x.B[t].shape  # rank of node children and node x (left, right, node)
            yl, yr, yk = y.B[t].shape  # rank of node children and node y (left, right, node)
            Bt = torch.zeros(xl + yl, xr + yr, xk + yk, dtype=dtype, device=device)
            Bt[:xl, :xr, :xk] = x.B[t]
            Bt[xl:, xr:, xk:] = y.B[t]
        B[t] = Bt

    # Create resulting HTTensor
    res = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
    return res
