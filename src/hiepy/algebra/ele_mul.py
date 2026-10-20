"""HTTensor elementwise product module.

This module provides functionality to compute the elementwise (Hadamard)
product of two HT tensors sharing the same dimension tree structure, utilizing
efficient on-the-fly low-rank truncation via reduced Gramians.
"""

from __future__ import annotations

from warnings import warn

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.left_svd_gramian import left_svd_gramian
from hiepy.utils.data_types import TruncOpts
from hiepy.utils.trunc_rank import trunc_rank


def ele_mul(x: HTTensor, y: HTTensor, opts: TruncOpts = {}) -> HTTensor:
    r"""
    Returns the (truncated) elementwise product of two HT tensors ``x`` and ``y``.

    For two tensors $X,Y \in \mathbb{R}^{n_1 \times \dots \times n_d}$ the elementwise product,
    also referred to as Hadamard product, is defined as:

    $$ (X \odot Y)_{i_1,\dots,i_d} = X_{i_1, \dots, i_d} \cdot Y_{i_1, \dots, i_d} $$

    If ``opts`` is specified, the product is truncated on the fly. Since the
    truncation is far from sharp, it is almost always advisable to perform another
    truncation afterwards.

    Parameters
    ----------
    x : HTTensor
        The first operand tensor.
    y : HTTensor
        The second operand tensor.
    opts : TruncOpts
        Truncation options controlling admissible approximation error and
        maximal hierarchical ranks.

        Supported keys:
        - ``atol`` (float): absolute truncation tolerance in Frobenius norm.
        - ``max_rank`` (int): upper bound on all hierarchical ranks.

        Unsupported keys:
        - ``rtol`` (float): relative truncation tolerance. It is ignored by this operation (see *Warns*).


    Returns
    -------
    HTTensor
        A new HT tensor representing the (truncated) elementwise product of ``x`` and ``y``.

    Raises
    ------
    ValueError
        If the dimension trees are incompatible, or if the full tensor
        shapes do not match.

    Warns
    -----
    UserWarning
        If ``opts`` contains ``rtol``, which is not supported and therefore ignored.

    Notes
    -----
    - The dimension tree is preserved.
    - Without truncation, rank growth is multiplicative:
        $r_t^{out} = r_t^x \cdot r_t^y$
    - Orthogonality is not preserved.
    - The attribute ``rtol`` is not supported in the truncation options.
    - Without truncation, runtime complexity is given as $\mathcal{O}(dnr^3 + dr^5)$,
      where $d$ is the number of dimensions, $n$ is the maximum mode size, and
      $r$ the maximum hierarchical rank (of ``x`` and ``y``).
    - Limiting the hierarchical rank to $r$ (via opts={"max_rank":r}), reduces the
      complexity to $\mathcal{O}(dnr^2 + dr^4)$.
    - Since the truncation is far from sharp, it is almost always advisable to perform another
      truncation afterwards.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> shape = 3, 3, 3
    >>> x = torch.ones(shape,dtype=torch.double)
    >>> y = torch.arange(1,28, dtype=torch.double).reshape(3,3,3)
    >>> prod_torch = x*y
    >>> x_ht = hiepy.to_htt(x, opts={"atol":1e-12})
    >>> y_ht = hiepy.to_htt(y, opts={"atol":1e-12})
    >>> prod_ht = hiepy.ele_mul(x_ht,y_ht)
    >>> prod_ht.rank
    {(0, 1, 2): 1, (0,): 2, (1,): 2, (2,): 2, (0, 1): 2}
    >>> rel_err = torch.linalg.norm(prod_torch-prod_ht.full()) / torch.linalg.norm(prod_torch)
    >>> rel_err
    tensor(6.5401e-16)
    >>> prod_ht = hiepy.ele_mul(x_ht,y_ht,opts={"max_rank":1})
    >>> prod_ht.rank
    {(0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1, (0, 1): 1}
    >>> rel_err = torch.linalg.norm(prod_torch-prod_ht.full()) / torch.linalg.norm(prod_torch)
    >>> rel_err
    tensor(0.0744)
    """
    # Validity checks
    if not x.dtree.is_equal(y.dtree):
        raise ValueError("Dimension trees of 'x' and 'y' are not compatible")
    if x.shape != y.shape:
        raise ValueError(f"Shape of 'y' {y.shape} does not match shape of 'x' {x.shape}")

    # Adjust error tolerances in opts.
    # To guarantee a global error bound e, the local truncation error per node
    # must be bounded by e / sqrt(2 * order - 3).
    adjusted_opts: TruncOpts = {}
    if "atol" in opts:
        adjusted_opts["atol"] = opts["atol"] / (x.order * 2 - 3) ** 0.5
    if "max_rank" in opts:
        adjusted_opts["max_rank"] = opts["max_rank"]
    if "rtol" in opts:
        warn(
            f"'rtol'={opts['rtol']} is not supported by ele_mul and is ignored; use 'atol' or 'max_rank'.",
            UserWarning,
            stacklevel=2,
        )

    # Bring both tensors into an orthogonalized state
    # (orthog returns a copy)
    x = x.orthog()
    y = y.orthog()

    # Compute reduced Gramian matrices
    Gx = x.gramians()
    Gy = y.gramians()

    # Traverse the dimension tree bottom-up
    for level in x.dtree.get_levels(order="bottom-up"):
        for t in level:
            if x.dtree.is_root(t):
                # Root node has parent rank 1; no singular vectors are computed.
                # The transfer tensor can be updated directly via an elementwise product.
                x.B[t] = x.B[t] * y.B[t]

            else:
                # Non-root node: determine left singular vectors and values for truncation.
                Qx, svx = left_svd_gramian(Gx[t])
                Qy, svy = left_svd_gramian(Gy[t])

                # Determine which column pairs to retain based on the cross-product of singular values
                sv = torch.kron(svx, svy)
                sort_idc = torch.argsort(sv, descending=True)
                if adjusted_opts:
                    rank = trunc_rank(sv[sort_idc], **adjusted_opts)
                else:
                    rank = len(sv)
                cols_x, cols_y = torch.unravel_index(indices=sort_idc[:rank], shape=(len(svx), len(svy)))
                Qx = Qx[:, cols_x]
                Qy = Qy[:, cols_y]

                if x.dtree.is_leaf(t):
                    # Leaf node: update the leaf matrix directly
                    x.U[t] = (x.U[t] @ Qx) * (y.U[t] @ Qy)

                else:
                    # Internal node: project and update the transfer tensor
                    x.B[t] = torch.tensordot(x.B[t], Qx, dims=([2], [0]))  # type: ignore
                    y.B[t] = torch.tensordot(y.B[t], Qy, dims=([2], [0]))  # type: ignore
                    x.B[t] = x.B[t] * y.B[t]

                # Update the transfer tensor of the parent node.
                # This step is identical regardless of whether the node is a leaf or internal.
                parent = x.dtree.get_parent(t)
                if x.dtree.is_left(t):
                    x.B[parent] = torch.tensordot(Qx, x.B[parent], dims=([0], [0]))  # type: ignore
                    y.B[parent] = torch.tensordot(Qy, y.B[parent], dims=([0], [0]))  # type: ignore
                else:
                    x.B[parent] = torch.tensordot(Qx, x.B[parent], dims=([0], [1]))  # type: ignore
                    x.B[parent] = torch.movedim(x.B[parent], source=0, destination=1)  # type: ignore
                    y.B[parent] = torch.tensordot(Qy, y.B[parent], dims=([0], [1]))  # type: ignore
                    y.B[parent] = torch.movedim(y.B[parent], source=0, destination=1)  # type: ignore

    # Reset the orthogonality flag
    x.is_orthog = False
    return x


# Currently under maintenance
# def ele_mul_top_down(x: HTTensor, y: HTTensor, opts: TruncOpts = {}) -> HTTensor:
#     # Validity checks
#     if not x.dtree.is_equal(y.dtree):
#         raise ValueError("Dimension trees of 'x' and 'y' are not compatible")
#     if x.shape != y.shape:
#         raise ValueError(f"Shape of 'y' {y.shape} does not match shape of 'x' {x.shape}")

#     # Generate deep copies
#     x = deepcopy(x)
#     y = deepcopy(y)

#     # Adjust error tolerances in opts.
#     # To guarantee a global error bound e, the local truncation error per node
#     # must be bounded by e / sqrt(2 * order - 3).
#     adjusted_opts = deepcopy(opts)
#     if "rtol" in adjusted_opts:
#         del adjusted_opts["rtol"]
#     if "atol" in adjusted_opts:
#         adjusted_opts["atol"] = adjusted_opts["atol"] / (x.order * 2 - 3) ** 0.5

#     # Bring both tensors into an orthogonalized state
#     x = x.orthog()
#     y = y.orthog()

#     # Compute reduced Gramian matrices
#     Gx = local_svd(x)
#     Gy = local_svd(y)

#     # Traverse the dimension tree bottom-up
#     for level in x.dtree.get_levels(order="bottom-up"):
#         for node in level:
#             dims = node.dims
#             if node.is_root():
#                 # Root node has parent rank 1; no singular vectors are computed.
#                 # The transfer tensor can be updated directly via an elementwise product.
#                 x.B[dims] = x.B[dims] * y.B[dims]

#             else:
#                 # Non-root node: determine left singular vectors and values for truncation.
#                 Qx, svx = Gx[dims]
#                 Qy, svy = Gy[dims]

#                 # Determine which column pairs to retain based on the cross-product of singular values
#                 sv = torch.kron(svx, svy)
#                 sort_idc = torch.argsort(sv, descending=True)
#                 print("Node. ", node.dims)
#                 print("svs: ", sv[sort_idc])
#                 if adjusted_opts:
#                     rank = trunc_rank(sv[sort_idc], **adjusted_opts)
#                 else:
#                     rank = len(sv)
#                 cols_x, cols_y = torch.unravel_index(indices=sort_idc[:rank], shape=(len(svx), len(svy)))
#                 Qx = Qx[:, cols_x]
#                 Qy = Qy[:, cols_y]

#                 if node.is_leaf():
#                     # Leaf node: update the leaf matrix directly
#                     x.U[dims] = (x.U[dims] @ Qx) * (y.U[dims] @ Qy)

#                 else:
#                     # Internal node: project and update the transfer tensor
#                     x.B[dims] = torch.tensordot(x.B[dims], Qx, dims=([2], [0]))  # type: ignore
#                     y.B[dims] = torch.tensordot(y.B[dims], Qy, dims=([2], [0]))  # type: ignore
#                     x.B[dims] = x.B[dims] * y.B[dims]

#                 # Update the transfer tensor of the parent node.
#                 # This step is identical regardless of whether the node is a leaf or internal.
#                 father = node.get_father()
#                 fdims = father.dims
#                 if node.is_left():
#                     x.B[fdims] = torch.tensordot(Qx, x.B[fdims], dims=([0], [0]))  # type: ignore
#                     y.B[fdims] = torch.tensordot(Qy, y.B[fdims], dims=([0], [0]))  # type: ignore
#                 else:
#                     x.B[fdims] = torch.tensordot(Qx, x.B[fdims], dims=([0], [1]))  # type: ignore
#                     x.B[fdims] = torch.movedim(x.B[fdims], source=0, destination=1)  # type: ignore
#                     y.B[fdims] = torch.tensordot(Qy, y.B[fdims], dims=([0], [1]))  # type: ignore
#                     y.B[fdims] = torch.movedim(y.B[fdims], source=0, destination=1)  # type: ignore

#     # Reset the orthogonality flag
#     x.is_orthog = False
#     return x


# def local_svd(x: HTTensor) -> dict[NodeID, tuple[torch.Tensor, torch.Tensor]]:

#     x = orthog(x)

#     # Phase 2: Compute projection matrices from root to leaves
#     W = {x.dtree.root.dims: torch.eye(1)}

#     SVS: dict[NodeID, tuple[torch.Tensor, torch.Tensor]] = {}
#     # Handle the root explicitly
#     root_dims = x.dtree.root.dims
#     norm = torch.linalg.norm(x.B[root_dims])
#     SVS[root_dims] = torch.tensor([norm]), torch.eye(1)
#     for t in x.dtree.get_nodes(order="top-down"):
#         if t.is_leaf():
#             continue

#         t1: Node = t.get_left()
#         t2: Node = t.get_right()

#         # 1. Multiply W_t into B_t
#         Bt = x.B[t.dims]
#         B_tilde: torch.Tensor = mode_mul(W[t.dims].T, Bt, dim=2)  # type:ignore

#         # --- Left child ---
#         M_t1 = matricise(B_tilde, t=(0,))
#         U_t1, S_t1, _ = torch.linalg.svd(M_t1, full_matrices=False)
#         SVS[t1.dims] = U_t1, S_t1

#         # Calculate W_t1 for the next recursive step
#         W[t1.dims] = U_t1 * S_t1  # @ torch.diag(S_t1)

#         # --- Right child ---
#         M_t2 = matricise(B_tilde, t=(1,))
#         U_t2, S_t2, _ = torch.linalg.svd(M_t2, full_matrices=False)
#         SVS[t2.dims] = U_t2, S_t2

#         # Calculate W_t2 for the next recursive step
#         W[t2.dims] = U_t2 * S_t2  # @ torch.diag(S_t2)

#     return SVS
