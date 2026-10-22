"""
HTTensor truncation module.

This module provides algorithms for reducing the hierarchical ranks of
HT tensors while controlling the approximation error through prescribed
tolerances or rank constraints.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.left_svd_gramian import left_svd_gramian
from hiepy.linalg.matricize import matricize
from hiepy.linalg.mode_mul_torch import mode_mul_torch
from hiepy.transformation.orthog import orthog
from hiepy.utils.data_types import TruncOpts
from hiepy.utils.dimtree import Node
from hiepy.utils.trunc_rank import trunc_rank


def truncate(x: HTTensor, opts: TruncOpts, method: str = "svd") -> tuple[HTTensor, dict[Node, torch.Tensor]]:
    r"""
    Truncates an HT tensor and returns the discarded singular values.

    Dispatches to a truncation algorithm that reduces the hierarchical ranks
    of an HT  tensor while controlling the approximation error according to the
    specified truncation options.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    opts : TruncOpts
        Truncation options controlling admissible approximation error and
        maximal hierarchical ranks.

        Supported keys:
        - ```atol``` (float): absolute truncation tolerance in Frobenius norm.
        - ```rtol``` (float): relative truncation tolerance in Frobenius norm.
        - ```max_rank``` (int): upper bound on all hierarchical ranks.

    method : {"svd", "gram"}, optional
        Truncation algorithm to be used. Defaults to `svd`.

    Returns
    -------
    tuple[HTTensor, dict[Node, torch.Tensor]]
        Truncated HT tensor approximating ``x`` and dict containing the nodewise singular values.

    Raises
    ------
    ValueError
        If an unsupported truncation method is specified.

    Notes
    -----
    - The output tensor approximates the input tensor in Frobenius norm.
    - The dimension tree is preserved.
    - Orthogonality is not preserved.
    - Hierarchical ranks are reduced according to ```opts```.
    - Rank constraints via ```max_rank``` are strictly enforced and may override
    error-based truncation.
    - The returned singular values correspond to those removed from the local
    subspaces during truncation.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = 10,10,10
    >>> x = hiepy.ones(shape)
    >>> x = hiepy.add(hiepy.add(x,x),x)
    >>> x.rank
    {(0, 1, 2): 1, (0,): 3, (1,): 3, (2,): 3, (0, 1): 3}
    >>> opts = hiepy.TruncOpts({"rtol": 1e-8})
    >>> x_appr, _ = hiepy.truncate(x=x,opts=opts,method="svd")
    >>> x_appr.rank
    {(0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1, (0, 1): 1}
    >>> bool(hiepy.norm(x-x_appr) / hiepy.norm(x) <= 1e-8)
    True
    """
    if method == "gram":
        return truncate_gram(x, opts)
    elif method == "svd":
        return truncate_svd(x, opts)
    else:
        raise ValueError(f"Argument `method` must either be `svd` or `gram`, got {method}.")


def truncate_gram(x: HTTensor, opts: TruncOpts) -> tuple[HTTensor, dict[Node, torch.Tensor]]:
    r"""
    Truncates an HT tensor and returns the discarded singular values.

    The truncation is performed by constructing projection matrices from the
    dominant left singular vectors obtained from the eigendecompositions of node-wise Gramian matrices
    and applying them throughout the dimension tree.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    opts : TruncOpts
        Truncation options controlling admissible approximation error and
        maximal hierarchical ranks.

        Supported keys:
        - ```atol``` (float): absolute truncation tolerance in Frobenius norm.
        - ```rtol``` (float): relative truncation tolerance in Frobenius norm.
        - ```max_rank``` (int): upper bound on all hierarchical ranks.

    Returns
    -------
    tuple[HTTensor, dict[Node, torch.Tensor]]
        Truncated HT tensor approximating ``x`` and dict containing the nodewise singular values.

    Notes
    -----
    - The output tensor approximates the input tensor in Frobenius norm.
    - The dimension tree is preserved.
    - Orthogonality is not preserved.
    - Hierarchical ranks are reduced according to ```opts```.
    - Rank constraints via ```max_rank``` are strictly enforced and may override
    error-based truncation.
    - The returned singular values correspond to those removed from the local
    subspaces during truncation.
    - Gradients are propagated through the eigendecompositions of the reduced Gram matrices; they become unstable
      for (nearly) equal singular values. The selected ranks are discrete and therefore not differentiated.
    - The truncation comes at an arithmetic cost of $\mathcal{O}(dnr^2 + dr^4)$, where $d$ is the number of dimensions,
    $r$ the maximum hierarchical rank, and $n$ the maximum mode size.
    - Typically less accurate than ``truncate_svd``.
    """
    # Adjust error tolerances in opts.
    # To guarantee a global error bound e, the local truncation error per node
    # must be bounded by e / sqrt(2 * order - 3).
    adjusted_opts: TruncOpts = {}
    if "atol" in opts:
        adjusted_opts["atol"] = opts["atol"] / (x.order * 2 - 3) ** 0.5
    if "rtol" in opts:
        adjusted_opts["rtol"] = opts["rtol"] / (x.order * 2 - 3) ** 0.5
    if "max_rank" in opts:
        adjusted_opts["max_rank"] = opts["max_rank"]

    # Orthog x
    x = orthog(x)

    # Compute nodewise gramians
    G = x.gramians()

    # Track cutoff SVs
    SVs = {}

    # Bottom-up truncation sweep over the dimension tree
    for level in x.dtree.get_levels(order="bottom-up"):
        for t in level:
            if x.dtree.is_root(t):
                continue

            # Determine left singular vectors and singular values
            Q, sv = left_svd_gramian(G[t])

            # Determine truncation rank based on singular values
            rank = trunc_rank(sv, **adjusted_opts)
            Q = Q[:, :rank]

            # Store cutoff SVs
            SVs[t] = sv[rank:]

            if x.dtree.is_leaf(t):
                # Project leaf basis
                x.U[t] = x.U[t] @ Q

            else:
                # Project transfer tensor
                x.B[t] = torch.tensordot(x.B[t], Q, dims=([2], [0]))  # type: ignore

            # Propagate basis change to parent node
            father = x.dtree.get_parent(t)
            if x.dtree.is_left(t):
                x.B[father] = torch.tensordot(Q.T, x.B[father], dims=([1], [0]))  # type: ignore
            else:
                x.B[father] = torch.tensordot(Q.T, x.B[father], dims=([1], [1]))  # type: ignore
                x.B[father] = torch.movedim(x.B[father], source=0, destination=1)  # type: ignore

    # Mark tensor as not orthogonal
    x.is_orthog = False

    return x, SVs


def truncate_svd(x: HTTensor, opts: TruncOpts) -> tuple[HTTensor, dict[Node, torch.Tensor]]:
    r"""
    Truncates an HT tensor and returns the discarded singular values.

    The truncation is performed by constructing projection matrices from the
    dominant left singular vectors of local matricizations and applying them
    throughout the dimension tree.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    opts : TruncOpts
        Truncation options controlling admissible approximation error and
        maximal hierarchical ranks.

        Supported keys:
        - ```atol``` (float): absolute truncation tolerance in Frobenius norm.
        - ```rtol``` (float): relative truncation tolerance in Frobenius norm.
        - ```max_rank``` (int): upper bound on all hierarchical ranks.

    Returns
    -------
    tuple[HTTensor, dict[Node, torch.Tensor]]
        Truncated HT tensor approximating ``x`` and dict containing the nodewise singular values.

    Notes
    -----
    - The output tensor approximates the input tensor in Frobenius norm.
    - The dimension tree is preserved.
    - Orthogonality is not preserved.
    - Hierarchical ranks are reduced according to ```opts```.
    - Rank constraints via ```max_rank``` are strictly enforced and may override
      error-based truncation.
    - The returned singular values correspond to those removed from the local
      subspaces during truncation.
    - Gradients are propagated through ``torch.linalg.svd``; they become unstable for (nearly) equal singular
      values. The selected ranks are discrete and therefore not differentiated.
    - The truncation comes at an arithmetic cost of $\mathcal{O}(dnr^2 + dr^4)$, where $d$ is the number of dimensions,
      $r$ the maximum hierarchical rank, and $n$ the maximum mode size.
    - This method is generally more numerically stable than the Gramian-based
      truncation (``truncate_gram``), since it operates directly on singular values rather than
      eigenvalues of Gram matrices.
    - Implements a three-phase truncation strategy:
        - Orthogonalization of the input tensor
        - Top-down sweep: Propagates projections from the root to leaves by
          computing local SVDs of matricized transfer tensors to determine
          optimal truncation ranks.
        - Bottom-up sweep: Applies the accumulated projection matrices to the
          leaf matrices and transfer tensors to update the HT representation.

    """
    # Adjust error tolerances in opts.
    # To guarantee a global error bound e, the local truncation error per node
    # must be bounded by e / sqrt(2 * order - 3).
    adjusted_opts: TruncOpts = {}
    if "atol" in opts:
        adjusted_opts["atol"] = opts["atol"] / (x.order * 2 - 3) ** 0.5
    if "rtol" in opts:
        adjusted_opts["rtol"] = opts["rtol"] / (x.order * 2 - 3) ** 0.5
    if "max_rank" in opts:
        adjusted_opts["max_rank"] = opts["max_rank"]

    # Phase 1: Orthogonalize x
    x = orthog(x)

    # Phase 2: Compute projection matrices from root to leaves
    root = x.dtree.get_root()
    W = {root: torch.eye(1, dtype=x.dtype, device=x.device)}
    P = {root: torch.eye(1, dtype=x.dtype, device=x.device)}

    # Cutoff SVs
    SVs = {}
    for t in x.dtree.get_nodes(order="top-down"):
        if x.dtree.is_leaf(t):
            continue

        left, right = x.dtree.get_children(t)

        # 1. Multiply W_t into B_t
        # B_t: r_t1 x r_t2 x r_t, W_t: r_t x r_t
        Bt = x.B[t]
        B_tilde: torch.Tensor = mode_mul_torch(x=Bt, A=W[t].T, dim=2)  # type:ignore

        # --- Left child ---
        # Matricization: r_t1 x (r_t2 * r_t)
        M_left = matricize(B_tilde, t=(0,))
        U_left, S_left, _ = torch.linalg.svd(M_left, full_matrices=False)

        # Determine required rank
        k_left = trunc_rank(S_left, **adjusted_opts)

        # Store cutoff SVs
        SVs[left] = S_left[k_left:]

        # Store projection
        P[left] = U_left[:, :k_left]

        # Calculate W_t1 for the next recursive step
        W[left] = P[left] @ torch.diag(S_left[:k_left])

        # --- Right child ---
        # Matricization: r_t2 x (r_t1 * r_t)
        M_right = matricize(B_tilde, t=(1,))
        U_right, S_right, _ = torch.linalg.svd(M_right, full_matrices=False)

        # Determine required rank
        k_right = trunc_rank(S_right, **adjusted_opts)

        # Store cutoff SVs
        SVs[right] = S_right[k_right:]

        # Store projection
        P[right] = U_right[:, :k_right]

        # Calculate W_t1 for the next recursive step
        W[right] = P[right] @ torch.diag(S_right[:k_right])

    # Phase 3: Application of the projections from leaves to root
    for t in x.dtree.get_nodes(order="bottom-up"):
        if x.dtree.is_leaf(t):
            # Update leaf matrix
            x.U[t] = x.U[t] @ P[t]

        else:
            # Update transfer tensors
            left, right = x.dtree.get_children(t)

            # Transformation of transfer tensor is given by:
            # B_new = B_old x_1 P_t1^T x_2 P_t2^T x_3 P_t
            B_new = x.B[t]
            B_new = mode_mul_torch(x=B_new, A=P[left].T, dim=0)
            B_new = mode_mul_torch(x=B_new, A=P[right].T, dim=1)
            B_new = mode_mul_torch(x=B_new, A=P[t].T, dim=2)

            x.B[t] = B_new

    # Set orthogonality flag to false
    x.is_orthog = False

    return x, SVs
