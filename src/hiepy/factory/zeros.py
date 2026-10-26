"""HTTensor zeros construction module.

This module provides functionality to construct HT tensors whose entries are identically equal to zero.
The output tensor is represented exactly in HT format with hierarchical rank one.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.checks import check_shape
from hiepy.utils.data_types import Shape
from hiepy.utils.dimtree import DimTree


def zeros(
    shape: Shape,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> HTTensor:
    r"""
    Constructs a constant HT tensors filled with zeros, with shape defined by ``shape``.

    The tensor is represented exactly in hierarchical Tucker format over the
    canonical dimension tree associated with ``shape``.

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
        Hierarchical Tucker tensor of shape ``shape`` with all entries equal
        to zero.

    Raises
    ------
    ValueError
        If a mode of ``shape`` has size 0.

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
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.zeros((2, 2, 2))
    >>> x.full()
    tensor([[[0., 0.],
             [0., 0.]],
    <BLANKLINE>
            [[0., 0.],
             [0., 0.]]])
    """
    check_shape(shape)
    # Create leaf matrices and transfer tensors
    dtree = DimTree.from_order(len(shape))
    U = {leaf: torch.zeros(shape[leaf[0]], 1, dtype=dtype, device=device) for leaf in dtree.get_leaves()}
    B = {t: torch.zeros(1, 1, 1, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}

    x = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
    return x.requires_grad_(requires_grad)
