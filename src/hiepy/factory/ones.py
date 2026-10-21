"""HTTensor ones construction module.

This module provides functionality to construct HT tensors whose entries are identically equal to one.
The output tensor is represented exactly in HT format with hierarchical rank one.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import Shape
from hiepy.utils.dimtree import DimTree


def ones(
    shape: Shape,
    orthog: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> HTTensor:
    r"""
    Constructs a constant HT tensors filled with ones, with shape defined by ``shape``.

    The tensor is represented exactly in hierarchical Tucker format using a
    rank-one decomposition over the canonical dimension tree.

    Parameters
    ----------
    shape : Shape
        Tuple of integers specifying the tensor shape.

    orthog : bool, optional
        If `True`, an orthogonal HT tensors is constructed.
        Otherwise, a non-orthogonal HT tensors is returned.

    dtype : torch.dtype, optional
        The dtype of the tensor components. Defaults to ``torch.get_default_dtype()``.

    device : torch.device | str, optional
        The device on which the tensor components are allocated. Defaults to ``torch.get_default_device()``.

    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.

    Returns
    -------
    HTTensor
        Hierarchical Tucker tensor of shape ``shape`` with all entries equal to one.

    Notes
    -----
    - The output is represented exactly in HT format.
    - The canonical dimension tree associated with ``shape`` is used.
    - All hierarchical ranks are equal to one.
    - If `orthog=True`, the frame matrices are orthonormal and the output HT tensors is orthogonal.
    - Storage complexity scales as $\mathcal{O}(dn)$, where $d$ is the tensor order and $n$ the maximum mode size.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.ones((2, 2, 2))
    >>> x.full()
    tensor([[[1., 1.],
             [1., 1.]],
    <BLANKLINE>
            [[1., 1.],
             [1., 1.]]])

    >>> x = hiepy.ones((4, 4, 4), orthog=True)
    >>> x.is_orthog
    True
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # Create leaf matrices and transfer tensors
    dtree = DimTree.from_order(len(shape))
    if orthog:
        U = {
            leaf: torch.ones(shape[leaf[0]], 1, dtype=dtype, device=device) / shape[leaf[0]] ** 0.5
            for leaf in dtree.get_leaves()
        }
        B = {t: torch.ones(1, 1, 1, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}
        energy = torch.exp(0.5 * torch.sum(torch.log(torch.tensor(shape, dtype=dtype, device=device))))
        B[dtree.get_root()] = B[dtree.get_root()] * energy
    else:
        U = {leaf: torch.ones(shape[leaf[0]], 1, dtype=dtype, device=device) for leaf in dtree.get_leaves()}
        B = {t: torch.ones(1, 1, 1, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}

    # Attach to autograd graph as leaves, if required
    if requires_grad:
        U = {leaf: tensor.detach().requires_grad_() for leaf, tensor in U.items()}
        B = {node: tensor.detach().requires_grad_() for node, tensor in B.items()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=orthog)
