"""HTTensor getter functions.

Contains functions that return the order, rank, shape, etc. of a HTTensor.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.utils.data_types import Node, Shape


def order(x: HTTensor) -> int:
    r"""
    Returns the order of HT tensor ``x``.

    The order is defined as the number of dimensions.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    int
        The order of HT tensor ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = (2,3,4,5)
    >>> x = hiepy.ones(shape)
    >>> hiepy.order(x)
    4
    """
    return len(x.shape)


def rank(x: HTTensor) -> dict[Node, int]:
    r"""
    Returns the representational hierarchical ranks of a hierarchical Tucker tensor.

    For a tensor $X \in \mathbb{R}^{n_1 \times \dots \times n_d}$, the true
    algebraic hierarchical rank associated with a node $t \subset \{0,\dots,d-1\}$
    is defined via the rank of its corresponding matricization:

    $$r_t^{\text{alg}} = \operatorname{rank}(\operatorname{mat}(X,t))$$

    However, computational and storage complexities are governed by the actual
    sizes of the allocated HT representation components rather than the theoretical
    algebraic ranks. Therefore, this function returns the representational
    hierarchical rank $r_t$, which acts as an upper bound to the algebraic rank
    ($r_t \ge r_t^{\text{alg}}$) and is directly extracted from the structural
    components:

    $$
    r_t =
    \begin{cases}
        \mathrm{shape}(U_t)[1], & \text{if } t \text{ is a leaf},\\[1ex]
        \mathrm{shape}(B_t)[2], & \text{if } t \text{ is an inner node}.
    \end{cases}
    $$

    Parameters
    ----------
    x : HTTensor
        Input HT tensor whose ranks are to be evaluated.

    Returns
    -------
    dict[Node, int]
        A dictionary mapping each dimension tree node to its corresponding representational rank $r_t$.

    Notes
    -----
    - The returned representational rank is an upper bound to the algebraic hierarchical rank.
    - By definition, the rank at the root node is always exactly $r_{\text{root}} = 1$.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = (5,6,7)
    >>> x = hiepy.ones(shape)
    >>> hiepy.rank(x)
    {(0, 1, 2): 1, (0,): 1, (1,): 1, (2,): 1, (0, 1): 1}
    >>> x = hiepy.ones(shape) + hiepy.ones(shape)
    >>> hiepy.rank(x)
    {(0, 1, 2): 1, (0,): 2, (1,): 2, (2,): 2, (0, 1): 2}
    """
    h_shape = {x.dtree.get_root(): 1}
    for leaf in x.dtree.get_leaves():
        h_shape[leaf] = x.U[leaf].shape[1]
    for t in x.dtree.get_inner_nodes():
        if x.dtree.is_root(t):
            continue
        h_shape[t] = x.B[t].shape[2]
    return h_shape


def shape(x: HTTensor) -> Shape:
    """
    Returns the shape of HT tensor ``x``.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    Shape
        A tuple of integers representing the shape ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = (5,6,7,8)
    >>> x = hiepy.ones(shape)
    >>> hiepy.shape(x)
    (5, 6, 7, 8)
    """
    shape = []
    for dim in sorted(x.U.keys()):
        shape += [x.U[dim].shape[0]]
    return tuple(shape)


def nbytes(x: HTTensor) -> int:
    """
    Returns the total memory (in bytes) occupied by all leaf matrices and transfer tensors.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    int
        The number of bytes occupied HT tensor ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> shape = (10,10,10)
    >>> x = hiepy.randn(shape, rank=1)
    >>> hiepy.nbytes(x)
    256
    >>> x = hiepy.randn(shape, rank=2)
    >>> hiepy.nbytes(x)
    576
    >>> x = hiepy.randn(shape, rank=3)
    >>> hiepy.nbytes(x)
    1008
    """
    mem = 0
    for matrix in x.U.values():
        mem += matrix.numel() * matrix.element_size()
    for tensor in x.B.values():
        mem += tensor.numel() * tensor.element_size()
    return mem


def dtype(x: HTTensor) -> torch.dtype:
    """
    Returns the dtype of HT tensor ``x``.

    All leaf matrices and transfer tensors of an HT tensor share the same dtype (checked on construction), which is
    the dtype of the tensor.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    torch.dtype
        The dtype of all components of ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.ones((2, 3, 4), dtype=torch.float32)
    >>> hiepy.dtype(x)
    torch.float32
    """
    return x.B[x.dtree.get_root()].dtype


def device(x: HTTensor) -> torch.device:
    """
    Returns the device of HT tensor ``x``.

    All leaf matrices and transfer tensors of an HT tensor live on the same device (checked on construction), which
    is the device of the tensor.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    torch.device
        The device of all components of ``x``.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.ones((2, 3, 4))
    >>> hiepy.device(x)
    device(type='cpu')
    """
    return x.B[x.dtree.get_root()].device


def requires_grad(x: HTTensor) -> bool:
    """
    Returns whether HT tensor ``x`` participates in autograd.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    Returns
    -------
    bool
        ``True`` if at least one leaf matrix or transfer tensor of ``x`` requires gradients.

    Notes
    -----
    - Gradients are tracked exactly as in torch: an operation records an autograd graph if and only if one of its
      inputs requires gradients and autograd is enabled (see ``torch.no_grad``). No hiepy operation detaches
      implicitly; use ``detach`` or ``copy`` to cut the graph.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> hiepy.requires_grad(hiepy.ones((2, 3, 4)))
    False
    >>> hiepy.requires_grad(hiepy.ones((2, 3, 4), requires_grad=True))
    True
    """
    return any(t.requires_grad for t in (*x.U.values(), *x.B.values()))
