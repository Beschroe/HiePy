"""HTTensor arange construction module.

This module provides functionality to construct Hierarchical Tucker tensors
whose entries contain consecutive values.
"""

from __future__ import annotations

from math import prod

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import Shape
from hiepy.utils.dimtree import DimTree


def arange(
    shape: Shape,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> HTTensor:
    r"""
    Constructs an HT tensor containing consecutive values.

    Returns an HT tensor of the specified shape whose entries are filled with
    consecutive values from ``0`` to ``prod(shape) - 1`` in C-style order.
    The tensor is represented exactly in Hierarchical Tucker format.

    Parameters
    ----------
    shape : Shape
        Tuple of integers specifying the tensor shape.

    dtype : torch.dtype, optional
        The dtype of the tensor components. Defaults to ``torch.get_default_dtype()``.

    device : torch.device | str, optional
        The device on which the tensor components are allocated. Defaults to ``torch.get_default_device()``.

    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.

    Returns
    -------
    HTTensor
        New HT tensor of shape ``shape`` containing the values ``0, 1, ..., prod(shape) - 1``.

    Notes
    -----
    - The tensor is represented exactly in HT format.
    - The canonical dimension tree associated with ``shape`` is used.
    - All hierarchical ranks are equal to two.
    - Storage complexity scales as $\mathcal{O}(dn)$, where $d$ is the tensor order and $n$ the maximum mode size.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.arange((2, 2, 2))
    >>> x.full()
    tensor([[[0., 1.],
             [2., 3.]],
    <BLANKLINE>
            [[4., 5.],
             [6., 7.]]])
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # Init dtree
    d = len(shape)
    dtree = DimTree.from_order(d)

    # Construct leaf matrices
    U = {}
    for leaf in dtree.get_leaves():
        step_sz = prod(shape[dim] for dim in range(d) if dim > leaf[0])
        row_cnt = shape[leaf[0]]
        U[leaf] = torch.stack(
            [
                torch.arange(0, step_sz * (row_cnt), step_sz, dtype=dtype, device=device),
                torch.ones(row_cnt, dtype=dtype, device=device),
            ]
        ).T

    # Construct transfer tensors
    B = {}
    for t in dtree.get_inner_nodes():
        if dtree.is_root(t):
            B[t] = torch.tensor([[[0], [1]], [[1], [0]]], dtype=dtype, device=device)
        else:
            B[t] = torch.tensor([[[0, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=dtype, device=device)

    x = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
    return x.requires_grad_(requires_grad)
