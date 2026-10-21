"""HTTensor constant tensor construction.

This module provides functionality to construct HT tensors whose entries are identically
equal to a prescribed scalar value. The resulting tensors are represented exactly in HT
format with hierarchical rank one.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import ScalarType, Shape
from hiepy.utils.dimtree import DimTree


def constant(
    shape: Shape,
    c: ScalarType,
    orthog: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> HTTensor:
    r"""
    Constructs a constant HT tensor filled with scalar value ``c``, with shape defined by ``shape``.

    The tensor is represented exactly in HT format using hierarchical rank one and the canonical dimension tree.

    Parameters
    ----------
    shape : Shape
        A tuple of integers defining the shape of the output tensor.

    c : ScalarType
        Constant value assigned to every tensor entry.

    orthog: bool
        If `True`, an orthogonal HT tensor is constructed.
        Otherwise, a non-orthogonal HT tensor is returned.

    dtype : torch.dtype, optional
        The dtype of the tensor components. Defaults to ``torch.get_default_dtype()``.

    device : torch.device | str, optional
        The device on which the tensor components are allocated. Defaults to ``torch.get_default_device()``.

    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.

    Returns
    -------
    HTTensor
        HT tensor of shape ``shape`` whose entries are identically equal to ``c``.

    Notes
    -----
    - The output tensor is represented exactly in HT format.
    - The canonical dimension tree associated with ``shape`` is used.
    - All hierarchical ranks are equal to one.
    - If ``orthog=True``, all frame matrices have orthonormal columns and the output HT tensor is orthogonal.
    - The storage complexity scales as $\mathcal{O}(dn)$, where $d$ denotes the tensor order
      and $n$ the maximum mode size.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.constant((2,2,2), c=3.14)
    >>> x.full()
    tensor([[[3.1400, 3.1400],
             [3.1400, 3.1400]],
    <BLANKLINE>
            [[3.1400, 3.1400],
             [3.1400, 3.1400]]])
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # (Cast c to a tensor with the requested dtype and device)
    c = torch.as_tensor(c, dtype=dtype, device=device)

    # Create leaf matrices and transfer tensors
    dtree = DimTree.from_order(len(shape))
    if orthog:
        U = {
            leaf: torch.ones(shape[leaf[0]], 1, dtype=dtype, device=device) / shape[leaf[0]] ** 0.5
            for leaf in dtree.get_leaves()
        }
        B = {t: torch.ones(1, 1, 1, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}
        energy = torch.exp(0.5 * torch.sum(torch.log(torch.tensor(shape, dtype=dtype, device=device))))
        B[dtree.get_root()] = B[dtree.get_root()] * c * energy
    else:
        U = {leaf: torch.ones(shape[leaf[0]], 1, dtype=dtype, device=device) for leaf in dtree.get_leaves()}
        B = {t: torch.ones(1, 1, 1, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}
        B[dtree.get_root()] = B[dtree.get_root()] * c

    # Attach to autograd graph as leaves, if required
    if requires_grad:
        U = {leaf: tensor.detach().requires_grad_() for leaf, tensor in U.items()}
        B = {node: tensor.detach().requires_grad_() for node, tensor in B.items()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=orthog)
