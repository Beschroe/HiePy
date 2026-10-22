"""HTTensor unfolding module.

This module provides functionality to unfold a hierarchical Tucker tensor
(HTTensor) into an explicit, dense PyTorch tensor (torch.Tensor).
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.matricize import dematricize, matricize
from hiepy.linalg.mode_mul_torch import multi_mode_mul_torch


def full(x: HTTensor) -> torch.Tensor:
    r"""
    Unfolds an HT  tensor into a dense torch tensor.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor to be unfolded.

    Returns
    -------
    torch.Tensor
        The fully contracted, dense PyTorch tensor representing the explicit tensor coefficients.

    Notes
    -----
    - The shape of the tensor is preserved.
    - Works for any dimension tree: the rows of the frame of a node are ordered by the modes of its left subtree,
      then those of its right subtree, and the root frame is folded back according to this order.
    - Unfolding destroys the linear scaling of the HT format. The storage requirement jumps from
      $\mathcal{O}(dnr + dr^3)$ $\mathcal{O}(n^d)$, where $d$ is the number of dimensions, $n$ the maximum mode size,
      and $r$ the maximum hierarchical rank.
    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.ones(shape=(2, 2, 2))
    >>> x_unfolded = hiepy.full(x)
    >>> x_unfolded
    tensor([[[1., 1.],
             [1., 1.]],
    <BLANKLINE>
            [[1., 1.],
             [1., 1.]]])
    >>> type(x_unfolded)
    <class 'torch.Tensor'>
    """
    # Restore leaf frames bottom-up, tracking the mode of each row index of a frame
    U = {}
    modes = {}
    for t in x.dtree.get_nodes(order="bottom-up"):
        if x.dtree.is_leaf(t):
            U[t] = x.U[t]
            modes[t] = list(t)

        else:
            left, right = x.dtree.get_children(t)

            # Compute frame matrix; its rows run over the modes of the left, then the right subtree
            B_scaled = multi_mode_mul_torch(x=x.B[t], matrices={0: U[left], 1: U[right]})
            U[t] = matricize(B_scaled, (0, 1))
            modes[t] = modes[left] + modes[right]

            # Free space
            del U[left], U[right], modes[left], modes[right]

    # Dematricise root frame matrix into tensor, following the row order of the root frame
    root = x.dtree.get_root()
    return dematricize(U[root], x.shape, modes[root])
