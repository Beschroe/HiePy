"""HTTensor local SVDs computation module.

This module provides functionality to analyze Hierarchical Tucker (HT) tensors
by computing local left singular vectors and true global singular values along
the dimension tree using a top-down weight propagation strategy.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor
from hiepy.linalg.matricize import matricize
from hiepy.linalg.mode_mul_torch import mode_mul_torch
from hiepy.transformation.orthog import orthog
from hiepy.utils.dimtree import Node


def local_left_svd(x: HTTensor) -> dict[Node, tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Computes local left singular vectors and global singular values of an HT tensor.

    For every node $t$ of the dimension tree, let $X_t$ denote the corresponding
    matricization of the represented tensor. Assuming that ``x`` is orthogonal,
    the matricization admits the factorization

    $$
    X_t = U_t C_t,
    $$

    where $U_t$ is the orthonormal frame matrix associated with $t$. This
    function computes the singular value decomposition

    $$
    C_t = \hat{U}_t S_t \hat{V}_t^\top.
    $$

    Since $U_t$ has orthonormal columns, the singular values $S_t$ coincide
    exactly with the singular values of $X_t$, while the corresponding left
    singular vectors are given by $U_t\hat{U}_t$.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor. If ``x`` is not orthogonal, it is orthogonalized
        internally before the computation.

    Returns
    -------
    dict[Node, tuple[torch.Tensor, torch.Tensor]]
        Dictionary mapping each dimension tree node to a tuple `(U_local, S)`, where

        - `U_local` contains the left singular vectors of the local
        coefficient matrix.
        - `S` contains the singular values of the corresponding tensor
        matricization.

    Notes
    -----
    - The computation is performed entirely in HT format without explicitly forming tensor
      matricizations or reduced Gram matrices.

    - Compared to Gramian-based approaches, this algorithm computes the singular values directly
      rather than via eigenvalues of $G_t = C_tC_t^\top$, avoiding the squaring of singular values and
      improving numerical robustness

    - The arithmetic complexity is $\mathcal{O}(dnr^2 + dr^4)$, where $d$ is the tensor order,
      $n$ the maximum mode size, and $r$ the maximum hierarchical rank.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> torch.set_default_dtype(torch.float64)
    >>> gen = torch.Generator().manual_seed(42)
    >>> x = torch.randn(2,3,4,generator=gen)
    >>> x_ht = hiepy.to_htt(x, {"atol":1e-8})
    >>> t = (0,1)
    >>> hiepy.local_left_svd(x_ht)[t]
    (tensor([[-1.0000e+00, -1.1928e-18,  2.1948e-17, -1.1690e-17],
            [ 1.1928e-18, -1.0000e+00,  2.7756e-16,  3.4694e-17],
            [-2.1948e-17, -4.4409e-16, -1.0000e+00, -5.2909e-17],
            [-1.1690e-17,  4.8572e-17, -5.2909e-17,  1.0000e+00]]), tensor([3.6357, 1.3479, 0.5663, 0.3282]))
    """
    x = orthog(x)

    root = x.dtree.get_root()

    # Compute projection matrices from root to leaves

    W = {root: torch.eye(1, dtype=x.dtype, device=x.device)}

    SVDs: dict[Node, tuple[torch.Tensor, torch.Tensor]] = {}

    # Handle the root explicitly
    norm = torch.linalg.norm(x.B[root])
    SVDs[root] = torch.eye(1, dtype=x.dtype, device=x.device), norm.reshape(1)

    for t in x.dtree.get_nodes(order="top-down"):
        if x.dtree.is_leaf(t):
            continue

        left, right = x.dtree.get_children(t)

        # 1. Multiply W_t into B_t
        Bt = x.B[t]
        B_tilde: torch.Tensor = mode_mul_torch(x=Bt, A=W[t].T, dim=2)  # type:ignore

        # --- Left child ---
        M_left = matricize(B_tilde, t=(0,))
        U_left, S_left, _ = torch.linalg.svd(M_left, full_matrices=False)
        SVDs[left] = U_left, S_left

        # Calculate W_t1 for the next recursive step
        W[left] = U_left * S_left  # @ torch.diag(S_t1)

        # --- Right child ---
        M_right = matricize(B_tilde, t=(1,))
        U_right, S_right, _ = torch.linalg.svd(M_right, full_matrices=False)
        SVDs[right] = U_right, S_right

        # Calculate W_t2 for the next recursive step
        W[right] = U_right * S_right  # @ torch.diag(S_t2)

    return SVDs
