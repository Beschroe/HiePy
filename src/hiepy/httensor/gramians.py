"""HTTensor Gramians.

This module provides functionality to compute the nodewise reduced Gram matrices
for an HTTensor.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import Node


def gramians(x: HTTensor) -> dict[Node, torch.Tensor]:
    r"""
    Returns the reduced Gram matrices of an orthogonal HT tensor.

    Let $X_t$ denote the matricization associated with a node $t$ of the dimension tree,
    and let $U_t$ be the corresponding orthonormal frame matrix. Then, the reduced Gram matrix
    $G_t$ is defined by

    $$
    X_t X_t^\top = U_t G_t U_t^\top.
    $$

    Parameters
    ----------
    x : HTTensor
        Orthogonal input HT tensor.

    Returns
    -------
    dict[Node, torch.Tensor]
        Dictionary mapping each dimension tree node to its reduced Gram matrix. The Gramian associated with
        the root node is the $1 \times 1$ matrix `[[1]]`.

    Raises
    ------
    ValueError
        If ``x`` is not orthogonal.

    Notes
    -----
    - The computation is performed entirely in HT format and does not require reconstruction of the full tensor.
    - For every non-root node $t$, the eigenvalues of $G_t$ coincide with the squared singular values of $X_t$.
    - When $Q_t \Delta_t Q_t^\top$ is the eigendecomposition of $G_t$, the left singular vectors of $X_t$ are
      given as $U_tQ_t$.
    - The computation has arithmetic complexity $\mathcal{O}(dr^4)$, where $d$ is the tensor order and $r$ the
      maximum hierarchical rank.
    """
    # Verify orthogonality
    if not x.is_orthog:
        raise ValueError("Argument ``x`` is not an orthogonal HTTensor.")

    # Dictionary for reduced Gramians, root is identity by definition
    G = {x.dtree.get_root(): torch.ones(1, 1, dtype=x.dtype, device=x.device)}

    # Traverse the dimension tree top-down
    # Compute reduced Gramians at each node
    for level in x.dtree.get_levels(order="top-down"):
        for t in level:
            if x.dtree.is_leaf(t):
                # Gramian of leaves is handled by their parent node
                continue

            # Children of the current node
            left, right = x.dtree.get_children(t)

            # Contract transfer tensor with parent Gramian
            BG = torch.tensordot(x.B[t], G[t], dims=([2], [1]))  # type: ignore

            # Compute reduced Gramians for the children
            G[left] = torch.tensordot(x.B[t], BG, dims=([1, 2], [1, 2]))  # type: ignore
            G[right] = torch.tensordot(x.B[t], BG, dims=([0, 2], [0, 2]))  # type: ignore
    return G


# This method is currently not supported.
# def gramians_sum(summands: list[HTTensor]) -> dict[tuple[int, ...], torch.Tensor]:
#     """Compute the reduced Gram matrices of the sum of several HTTensors.

#     This function computes the nodewise reduced Gram matrices of the implicit
#     sum X = A + B + ... without forming the sum explicitly.

#     For a tensor X and its matricization X_t along a node t, let U_t be an
#      basis of the column space of X_t. The reduced Gram matrix G_t
#     satisfies

#         X_t @ X_t.T = U_t @ G_t @ U_t.T.

#     Args:
#         summands (list[htt]): List of HTTensors with identical dimension trees.

#     Returns:
#         dict[tuple[int, ...], torch.Tensor]: A dictionary mapping each node
#         of the dimension tree of the implicit sum to its reduced Gram matrix.

#     Raises:
#         ValueError: If fewer than two summands are provided.
#         ValueError: If the summands do not share identical dimension trees.

#     Example:
#         >>> X, Y, Z = htt.randn((3, 4, 5, 6)), htt.randn((3, 4, 5, 6)), htt.randn((3, 4, 5, 6))
#         >>> G = gramians_sum([X, Y, Z])

#     """
#     # Validity checks
#     if len(summands) < 2:
#         raise ValueError(f"At least two summands required, got {len(summands)}.")
#     dtree = summands[0].dtree
#     if not all(item.dtree.is_equal(dtree) for item in summands[1:]):
#         raise ValueError("All summands must share equal dimension trees.")

#     # Compute M = U.T @ U for every node
#     # For n summands, U.T @ U is a block matrix with n^2 blocks.
#     # Block (i, j) has size r_i x r_j, where r_i is the local rank of summand i.
#     # To represent this structure, M[node] is a dict with keys (i, j).
#     M = {}

#     # Leaf nodes: construct block Gram matrices from stacked basis matrices
#     for leaf in dtree.get_leaves():
#         t = leaf.dims
#         U = torch.hstack(tuple(item.U[t] for item in summands))
#         M_leaf = U.T @ U

#         # cumulative ranks: [0, r0, r0+r1, r0+r1+r2, ...]
#         cum_ranks = torch.cumsum(torch.tensor([0] + [item.U[t].shape[1] for item in summands]), dim=0)

#         M[t] = {}
#         for i in range(len(summands)):
#             for j in range(len(summands)):
#                 M[t][i, j] = M_leaf[cum_ranks[i] : cum_ranks[i + 1], cum_ranks[j] : cum_ranks[j + 1]]

#     # Inner nodes: compute block structure bottom-up
#     for level in dtree.get_levels(order="bottom-up"):
#         for node in level:
#             t = node.dims
#             if node.is_leaf():
#                 continue
#             l, r = node.get_children()
#             tl, tr = l.dims, r.dims
#             M[t] = {}
#             for i in range(len(summands)):
#                 for j in range(len(summands)):
#                     M_left_times_B = torch.tensordot(M[tl][j, i], summands[i].B[t], dims=([1], [0]))  # type: ignore
#                     M_right_times_B = torch.tensordot(
#                         M[tr][i, j],
#                         summands[j].B[t],
#                         dims=([1], [1]),  # type: ignore
#                     )
#                     M_right_times_B = torch.movedim(M_right_times_B, source=0, destination=1)
#                     M[t][i, j] = torch.tensordot(
#                         M_left_times_B,
#                         M_right_times_B,
#                         dims=([0, 1], [0, 1]),  # type: ignore
#                     )

#     # Initialize reduced Gram matrices
#     # Root is always 1
#     G = {dtree.root.dims: torch.ones(1, 1)}

#     # Propagate top-down through the dimension tree
#     for level in dtree.get_levels(order="top-down"):
#         for node in level:
#             if node.is_leaf():
#                 continue
#             l, r = node.get_children()
#             t, tl, tr = node.dims, l.dims, r.dims
#             G[tl], G[tr] = {}, {}  # type: ignore
#             for i in range(len(summands)):
#                 for j in range(len(summands)):
#                     if node.is_root():
#                         # Root Gramian is identity, so transfer tensors can be contracted directly
#                         B_times_G_times_B = torch.tensordot(
#                             summands[i].B[t],
#                             summands[j].B[t],
#                             dims=([2], [2]),  # type: ignore
#                         )
#                     else:
#                         B_times_G = torch.tensordot(
#                             summands[i].B[t],
#                             G[t][j, i],
#                             dims=([2], [1]),  # type: ignore
#                         )
#                         B_times_G_times_B = torch.tensordot(
#                             B_times_G,
#                             summands[j].B[t],
#                             dims=([2], [2]),  # type: ignore
#                         )
#                     G[tl][i, j] = torch.tensordot(
#                         B_times_G_times_B,
#                         M[tr][i, j],
#                         dims=([1, 3], [0, 1]),  # type: ignore
#                     )
#                     G[tr][i, j] = torch.tensordot(
#                         B_times_G_times_B,
#                         M[tl][i, j],
#                         dims=([0, 2], [0, 1]),  # type: ignore
#                     )

#     # Merge block structure of reduced Gramians into full matrices
#     for node in dtree.get_nodes():
#         if node.is_root():
#             continue
#         t = node.dims
#         G[t] = torch.cat(
#             [torch.cat([G[t][i, j] for j in range(len(summands))], dim=1) for i in range(len(summands))],
#             dim=0,
#         )

#     return G
