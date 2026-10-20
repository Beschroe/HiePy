"""Hilbert tensor construction module.

This module provides functionality to generate a Hierarchical Tucker
representation of the Hilbert tensor, utilizing sinc-quadrature approximation
to decompose the Cauchy kernel into a sum of separable components.
"""

import torch

from hiepy.httensor.httensor import HTTensor
from hiepy.linalg.diagonal_tensor import diagonal_tensor
from hiepy.utils.data_types import Shape
from hiepy.utils.dimtree import DimTree


def hilbert(
    shape: Shape,
    M: int = 25,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> HTTensor:
    r"""
    Constructs a Hilbert tensor in HT format using sinc-quadrature.

    This function approximates the $d$-dimensional Hilbert tensor, defined by
    the entries
        $H_{i_1, \dots, i_d} = \left( \sum_{k=1}^d x_{i_k}^{(k)} \right)^{-1}$

    The approximation is based on the exponential sum representation of the Cauchy
    kernel, which allows for an HT representation of rank $R = 2M + 1$.

    Parameters
    ----------
    shape : Shape
        Tuple of integers specifying the tensor shape.
    M : int, optional
        Number of quadrature points used for the sinc-approximation. This
        directly determines the hierarchical rank $R = 2M + 1$. Default is 25.
    dtype : torch.dtype, optional
        The desired data type of the tensor entries.
    device : torch.device | str, optional
        The device on which the tensor leaves and transfer tensors are allocated.
    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.

    Returns
    -------
    HTTensor
        A tensor approximating the Hilbert tensor.

    Notes
    -----
    - The hierarchical rank $R$ is constant across all nodes of the tree,
      dependent solely on the quadrature parameter $M$.
    - The derived HT Tucker representation contains a high degree of redundancies. Therefore, subsequent truncation can
      drastically reduce the hierarchical ranks.
    - The resulting HTTensor is non-orthogonal.
    - The approximation accuracy relies on the sinc-quadrature error, which decays exponentially with $M$.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> shape = (20,20,20)
    >>> M = 20
    >>> x = hiepy.hilbert(shape, M)
    >>> x.full()[:4,0,0]
    tensor([0.3333, 0.2500, 0.2000, 0.1667])
    """

    # Construct dimtree
    d = len(shape)
    dtree = DimTree.from_order(d)

    # Calculate rank
    R = 2 * M + 1

    xis = [torch.linspace(1, dim_sz, dim_sz, dtype=dtype, device=device) for dim_sz in shape]
    j = torch.arange(-M, M + 1, dtype=dtype, device=device)

    xMin = d * torch.min(xis[0])  # minimum is the same for all xi
    hst = torch.pi / M**0.5

    alpha = -2.0 * torch.log(torch.exp(j * hst) + torch.sqrt(1 + torch.exp(2 * j * hst))) / xMin

    omega = 2 * hst / torch.sqrt(1 + torch.exp(-2 * j * hst)) / xMin

    # Leaf matrices
    U = {leaf: torch.exp(torch.outer(xis[leaf[0]], alpha)) for leaf in dtree.get_leaves()}
    U[(0,)] = U[(0,)] * omega

    # Transfer tensors
    B = {t: diagonal_tensor((R, R, R), fill_value=1.0, dtype=dtype, device=device) for t in dtree.get_inner_nodes()}
    B[dtree.get_root()] = torch.eye(R, R, dtype=dtype, device=device).unsqueeze(2)

    # Attach to autograd graph as leaves, if required
    if requires_grad:
        U = {leaf: tensor.detach().requires_grad_() for leaf, tensor in U.items()}
        B = {node: tensor.detach().requires_grad_() for node, tensor in B.items()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
