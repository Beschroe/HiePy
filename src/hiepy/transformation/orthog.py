"""HTTensor orthogonalization.

Contains a function to orthogonalize an HT tensor. An HT tensor is said to be
orthogonal if all of its frame matrices have orthonormal columns.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.matricize import dematricize, matricize
from hiepy.linalg.mode_mul_torch import mode_mul_torch


def orthog(x: HTTensor) -> HTTensor:
    r"""
    Orthogonalizes an HT tensor.

    An HT tensor is called orthogonal if all of its frame matrices $U_t$ have orthonormal columns, i.e.
    $U_t^\top U_t = I{_{r_t}}$. Here $I{_{r_t}}$ denotes an $r_t \times r_t$ identity matrix, where $r_t$ is the
    hierarchical rank at node $t$.

    The orthogonalization is performed bottom-up along the dimension tree by successive QR decompositions and
    the propagation of the resulting triangular factors towards the root.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor to be orthogonalized.

    Returns
    -------
    HTTensor
        New orthogonal HT tensor.

    Notes
    -----
    - The dimension tree is preserved.
    - The represented tensor is preserved. The representational ranks may decrease to min(rows, cols) of the respective
      (matricized) component; they remain upper bounds of the hierarchical rank.
    - If the input HT tensor is already orthogonal, a copy is returned.
    - The operation comes at an arithmetic cost of $\mathcal{O}(dnr^2 + dr^4)$, where $d$ is the number of dimensions,
      $r$ the maximum hierarchical rank, and $n$ the maximum mode size.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.arange(shape=(8, 8, 8))
    >>> x.is_orthog
    False
    >>> # Exemplarily check U_0
    >>> U_0 = x.U[(0,)]
    >>> torch.allclose(torch.eye(2), U_0.T @ U_0)
    False
    >>> x_ = hiepy.orthog(x)
    >>> x_.is_orthog
    True
    >>> # Exemplarily check U_0
    >>> U_0 = x_.U[(0,)]
    >>> torch.allclose(torch.eye(2), U_0.T @ U_0)
    True
    >>> torch.allclose(x.full(), x_.full())
    True
    """

    x = x.copy()

    # Return if already orthog
    if x.is_orthog:
        return x

    # Orthogonalization of the leave matrices
    for leaf in x.dtree.get_leaves():
        # Compute QR-decomp
        x.U[leaf], R = torch.linalg.qr(x.U[leaf], mode="reduced")

        # Multiply R into parent transfer tensor
        parent = x.dtree.get_parent(leaf)
        dim = 0 if x.dtree.is_left(leaf) else 1
        x.B[parent] = mode_mul_torch(x=x.B[parent], A=R, dim=dim)

    # Iterate over the tree bottom-up and perform nodewise QR-decompositions
    for level in x.dtree.get_levels(order="bottom-up"):
        for t in level:
            if x.dtree.is_leaf(t):
                continue

            # Skip the root node
            if not x.dtree.is_root(t):
                # Compute QR-decomp
                old_shape = x.B[t].shape
                B_mat = matricize(x.B[t], t=(0, 1))
                x.B[t], R = torch.linalg.qr(B_mat, mode="reduced")
                x.B[t] = dematricize(x.B[t], shape=(old_shape[0], old_shape[1], x.B[t].shape[1]), t=(0, 1))

                # Multiply R into parent transfer tensor
                parent = x.dtree.get_parent(t)  # type:ignore
                dim = 0 if x.dtree.is_left(t) else 1
                x.B[parent] = mode_mul_torch(x=x.B[parent], A=R, dim=dim)

    # Update
    x.is_orthog = True
    return x
