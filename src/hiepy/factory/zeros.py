"""HTTensor zeros construction module.

This module provides functionality to construct HT tensors whose entries are identically equal to zero.
The output tensor is represented exactly in HT format with hierarchical rank one.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import Shape
from hiepy.utils.dimtree import DimTree


def zeros(shape: Shape, requires_grad: bool = False) -> HTTensor:
    r"""
    Constructs a constant HT tensors filled with zeros, with shape defined by ``shape``.

    The tensor is represented exactly in hierarchical Tucker format over the
    canonical dimension tree associated with ``shape``.

    Parameters
    ----------
    shape : Shape
        Tuple of integers specifying the tensor shape.

    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.
    Returns
    -------
    HTTensor
        Hierarchical Tucker tensor of shape ``shape`` with all entries equal
        to zero.

    Notes
    -----
    - The output tensor is represented exactly in HT format.
    - The canonical dimension tree associated with ``shape`` is used.
    - All hierarchical ranks are equal to one in the representation, but the leaf matrices
      transfer tensors are degenerate (all entries are zero).
    - Storage complexity scales as $\mathcal{O}(dn)$, where $d$ is the tensor order and $n$ the maximum mode size.
    - The output HT tensors is not orthogonal.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import hiepy
    >>> x = hiepy.zeros((2, 2, 2))
    >>> x.full()
    tensor([[[0., 0.],
             [0., 0.]],
            [[0., 0.],
             [0., 0.]]])
    """
    # Create leaf matrices and transfer tensors
    dtree = DimTree.from_order(len(shape))
    U = {leaf: torch.zeros(shape[leaf[0]], 1) for leaf in dtree.get_leaves()}
    B = {t: torch.zeros(1, 1, 1) for t in dtree.get_inner_nodes()}

    # Attach to autograd graph as leaves, if required
    if requires_grad:
        U = {leaf: tensor.detach().requires_grad_() for leaf, tensor in U.items()}
        B = {node: tensor.detach().requires_grad_() for node, tensor in B.items()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
