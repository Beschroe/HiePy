"""HTTensor dense tensor to HT tensor module.

This module provides functionality to convert and approximate a dense PyTorch tensor
into a Hierarchical Tucker (HT) tensor representation using hierarchical SVD truncation.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.matricize import dematricize, matricize
from hiepy.linalg.mode_mul_torch import mode_mul_torch
from hiepy.utils.data_types import TruncOpts
from hiepy.utils.dimtree import DimTree
from hiepy.utils.trunc_rank import trunc_rank


def to_htt(x: torch.Tensor, opts: TruncOpts) -> HTTensor:
    r"""
    Approximates a dense torch.Tensor by an HT  tensor.

    Computes an approximation of the dense input tensor $x$ in the HT format by executing
    a sequential truncation process from the leaves up to the root of the dimension tree.

    Parameters
    ----------
    x : torch.Tensor
        The dense input tensor to be approximated. Dimensionality must be $d \geq 2$.

    opts : TruncOpts
        Truncation options controlling admissible approximation error and
        maximal hierarchical ranks.

        Supported keys:
        - ```atol``` (float): absolute truncation tolerance in Frobenius norm.
        - ```rtol``` (float): relative truncation tolerance in Frobenius norm.
        - ```max_rank``` (int): upper bound on all hierarchical ranks.

    Returns
    -------
    HTTensor
        An orthogonal hierarchical Tucker tensor approximating the input tensor ``x``.

    Raises
    ------
    ValueError
        If the input tensor ``x`` has fewer than 2 dimensions.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> from math import prod
    >>> shape = 3,4,5,6,7
    >>> x = torch.arange(prod(shape),dtype=torch.double).reshape(shape)
    >>> x.shape
    torch.Size([3, 4, 5, 6, 7])
    >>> # Approximate in HT format
    >>> x_ht = hiepy.to_htt(x=x, opts={"atol":1e-8})
    >>> # Check truncation accuracy
    >>> torch.allclose(x, x_ht.full())
    True
    >>> torch.linalg.norm(x-x_ht.full())
    tensor(1.1782e-10)
    """
    # validity checks
    if x.dim() < 2:
        raise ValueError(f"Argument 'x' must be a ND torch.Tensor with N>=2, got shape {x.shape}")

    # Adjust local error tolerances to meet the global error bound
    adjusted_opts: TruncOpts = {}
    scaling_factor = (x.ndim * 2 - 3) ** 0.5

    if "atol" in opts:
        adjusted_opts["atol"] = opts["atol"] / scaling_factor
    if "rtol" in opts:
        adjusted_opts["rtol"] = opts["rtol"] / scaling_factor
    if "max_rank" in opts:
        adjusted_opts["max_rank"] = opts["max_rank"]

    # Initialize tree and tracking components
    dtree = DimTree.from_order(x.dim())
    U = {}
    B = {}
    rank = {}

    # Compute all leaf matrices
    for leaf in dtree.get_leaves():
        mode = leaf[0]
        xmat = matricize(x, (mode,))
        U_t, sv, _ = torch.linalg.svd(xmat, full_matrices=False)
        rank[leaf] = trunc_rank(sv, **adjusted_opts) if adjusted_opts else U_t.shape[1]
        U[leaf] = U_t[:, : rank[leaf]]

    # Sequentially project the core tensor
    C = x
    for t in dtree.get_leaves():
        mode = t[0]
        C = mode_mul_torch(x=C, A=U[t].T, dim=mode)

    # Compute transfer tensors
    for level in dtree.get_levels(order="bottom-up"):
        # Compute transfer tensors
        for t in level:
            if dtree.is_leaf(t):
                continue
            elif dtree.is_root(t):
                B[t] = matricize(C, t)
                rank[t] = 1

            else:
                C_mat = matricize(C, t)
                B_t, sv, _ = torch.linalg.svd(C_mat, full_matrices=False)
                rank[t] = trunc_rank(sv, **adjusted_opts) if adjusted_opts else B_t.shape[1]
                B[t] = B_t[:, : rank[t]]

        # Update core tensor with previously computed transfer tensors and restore proper shapes on the fly
        for t in level:
            if dtree.is_leaf(t):
                continue

            left, right = dtree.get_children(t)
            shape_bt = rank[left], rank[right], rank[t]

            if dtree.is_root(t):
                # No update needed
                B[t] = dematricize(B[t], shape_bt, (0, 1))

            else:
                # Update core tensor
                shape = list(C.shape)
                C_ = matricize(C, t)
                C_ = B[t].T @ C_

                # Update tracking shape
                shape[left[-1]] = 1
                shape[right[-1]] = rank[t]
                C = dematricize(C_, shape=tuple(shape), t=t)

                # reshape transfer tensor
                shape_bt = rank[left], rank[right], rank[t]
                B[t] = dematricize(B[t], shape_bt, (0, 1))

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=True)
