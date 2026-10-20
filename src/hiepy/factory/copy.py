"""
HTTensor copy construction module.

This module provides functionality to create structurally identical copies
of HT tensors with different levels of autograd coupling.
"""

from __future__ import annotations

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import TensorDict


def clone(x: HTTensor) -> HTTensor:
    """
    Constructs a copy of the HT tensor that remains connected to the autograd graph.

    Produces a structurally identical HT tensor with independent storage while
    preserving gradient tracking through all leaf matrices and transfer tensors.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    HTTensor
        Deep copy of ``x`` with identical HT structure and values, still participating in the autograd graph.

    Notes
    -----
    - The dimension tree is preserved.
    - Hierarchical ranks and tensor values are preserved exactly.
    - The resulting tensor shares no memory with the input tensor.
    - Gradient tracking is preserved for all tensor components.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> x = hiepy.ones(shape=(2,3,4,5))
    >>> x_clone = hiepy.clone(x)
    >>> torch.allclose(x_clone.full(), x.full())
    True
    """
    dtree = x.dtree.copy()
    U: TensorDict = {leaf: x.U[leaf].clone() for leaf in dtree.get_leaves()}
    B: TensorDict = {t: x.B[t].clone() for t in dtree.get_inner_nodes()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=x.is_orthog)


def detach(x: HTTensor) -> HTTensor:
    """
    Constructs a new HT tensor detached from the autograd graph.

    Produces a new HT tensor object sharing the same underlying data fields
    but removes its connection to the active computational graph.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Notes
    -----
    - The dimension tree is preserved.
    - The HT representation is unchanged.
    - No gradient information is retained.
    - Memory is shared through leaf matrices and transfer tensors, but computation
      graph dependencies are removed.

    Returns
    -------
    HTTensor
        A new HT tensor with identical HT representation (same dimension tree,
        same leaf matrices, same transfer tensors), but detached from autograd.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> x = hiepy.ones(shape=(2,3,4,5))
    >>> x_detached = hiepy.detach(x)
    >>> x_detached.U[(0,)].requires_grad
    False
    >>> torch.allclose(x_detached.full(), x.full())
    True
    """
    dtree = x.dtree
    U: TensorDict = {leaf: x.U[leaf].detach() for leaf in dtree.get_leaves()}
    B: TensorDict = {t: x.B[t].detach() for t in dtree.get_inner_nodes()}
    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=x.is_orthog)


def copy(x: HTTensor) -> HTTensor:
    """
    Constructs a fully independent copy of the HT tensor detached from the autograd graph.

    Produces a structurally identical HT tensor with independent storage and
    no remaining dependencies on prior graph operations.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Notes
    -----
    - The dimension tree is preserved.
    - The HT representation is identical to the input.
    - No gradient tracking is preserved.
    - The result does not share memory with the input tensor.

    Returns
    -------
    HTTensor
        Fully independent copy of ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> x = hiepy.ones(shape=(2,3,4,5))
    >>> x_copy = hiepy.copy(x)
    >>> torch.allclose(x_copy.full(), x.full())
    True
    >>> x_copy.U[(0,)].requires_grad
    False
    """
    dtree = x.dtree.copy()
    U: TensorDict = {leaf: x.U[leaf].detach().clone() for leaf in dtree.get_leaves()}
    B: TensorDict = {t: x.B[t].detach().clone() for t in dtree.get_inner_nodes()}
    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=x.is_orthog)
