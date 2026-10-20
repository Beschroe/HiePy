"""HTTensor random initialization module.

This module provides functionality to construct random HT tensors.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.linalg.matricize import dematricize
from hiepy.utils.data_types import Shape, TensorDict
from hiepy.utils.dimtree import DimTree


def randn(
    shape: Shape, rank: int = 1, generator: torch.Generator | None = None, requires_grad: bool = False
) -> HTTensor:
    r"""
    Constructs an HT  tensor with random entries, a maximum hierarchical rank prescribed by ``rank``, and of
    shape defined by ``shape``.

    Parameters
    ----------
    shape : Shape
        A tuple of integers defining the shape of the output tensor.
    rank : int, optional
        The maximum hierarchical rank for all nodes in the dimension tree. The actual rank at
        each node is bounded by the dimensions size or the product of the children's ranks.
        Defaults to 1.
    generator : torch.Generator, optional
        A pseudorandom number generator for sampling. Allows for reproducibility without modifying the
        global PyTorch seed. Defaults to `None`.
    requires_grad: bool
        If `True`, the HT tensor components are attached to the autograd graph.

    Returns
    -------
    HTTensor
        A randomly initialized orthogonal HT tensor of shape ``shape``.

    Notes
    -----
    - The output tensor is orthogonal by construction.
    - If ``requires_grad`` is set to ``True``, HT tensor components participate in autograd.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> gen = torch.Generator().manual_seed(42)
    >>> # With reproducable results
    >>> x = hiepy.randn((2, 2, 2), rank=2, generator=gen)
    >>> x.full()
    tensor([[[ 1.1846,  0.5947],
            [-0.5298, -0.3974]],
            [[-0.5015,  0.5828],
            [ 0.4290, -0.2628]]])
    >>> # Without reproducable results
    >>> x = hiepy.randn((2, 2, 2), rank=2)
    >>> x.full()
    tensor([[[-0.0620,  0.0655],
            [-0.1304,  0.1376]],
            [[ 0.0281, -0.0297],
            [ 0.0591, -0.0624]]])
    """
    order = len(shape)
    dtree = DimTree.from_order(order)

    # Keep track of ranks
    ranks = {}

    # Construct leaf matrices
    U: TensorDict = {}
    for leaf in dtree.get_leaves():
        n_t = shape[leaf[0]]
        ranks[leaf] = min(n_t, rank)
        U_t = torch.randn(size=(n_t, ranks[leaf]), generator=generator)
        U[leaf], _ = torch.linalg.qr(U_t, mode="reduced")

    # Construct transfer tensors
    B: TensorDict = {}
    for t in dtree.get_nodes(order="bottom-up"):
        if dtree.is_leaf(t):
            continue
        left, right = dtree.get_children(t)
        rl, rr = ranks[left], ranks[right]
        if dtree.is_root(t):
            B[t] = torch.randn(size=(rl, rr, 1), generator=generator)
        else:
            ranks[t] = min(rl * rr, rank)
            B_t = torch.randn(size=(rl * rr, ranks[t]), generator=generator)
            Q_B_t, _ = torch.linalg.qr(B_t, mode="reduced")
            B[t] = dematricize(Q_B_t, shape=(rl, rr, ranks[t]), t=(0, 1))

        # Attach to autograd graph as leaves, if required
    if requires_grad:
        U = {leaf: tensor.detach().requires_grad_() for leaf, tensor in U.items()}
        B = {node: tensor.detach().requires_grad_() for node, tensor in B.items()}

    return HTTensor(U=U, B=B, dtree=dtree, is_orthog=True)


# def randn(
#     shape: tuple[int, ...],
#     max_rank: int | None = None,
#     fixed_rank: int | None = None,
#     exp_decay_svs: bool = False,
#     orthog: bool = False,
# ) -> HTTensor:
#     """Create a random hierarchical Tucker tensor (HTTensor) with the given shape.

#     - If exp_decay_svs=True, each matricisation of the tensor (with respect to
#       each node's dimension cluster) is generated with exponentially decaying
#       singular values. All nodes share a common rank determined by fixed_rank,
#       a rank sampled from {1,...,max_rank} or 5 (default), but in all cases the
#       rank is capped by min(shape).

#     - If exp_decay_svs=False, nodewise ranks are either fixed to fixed_rank (if
#       set), sampled from {1,…,max_rank} (if max_rank set), or sampled from
#       {1,…,5} (default). If both fixed_rank and max_rank are set and
#       fixed_rank>max_rank, a ValueError is raised; otherwise all ranks are set
#       to fixed_rank.

#     Parameters
#     ----------
#     shape : tuple[int, ...]
#         Full shape of the desired HTTensor.
#     max_rank : int, optional
#         Maximum rank to sample from if fixed_rank is not specified.
#     fixed_rank : int, optional
#         Fixed rank to use for all nodes if max_rank is not specified.
#     exp_decay_svs : bool, default False
#         Whether to generate exponentially decaying singular values in each
#         nodewise matricisation.
#     orthog : bool, default False
#         If True, orthogonalize the resulting HTTensor before returning.

#     Returns
#     -------
#     htt
#         A randomly generated HTTensor with the specified properties.

#     """
#     if exp_decay_svs:
#         return _randn_exp_decaying_svs(shape, max_rank, fixed_rank, orthog)
#     if max_rank is not None and fixed_rank is not None:
#         raise ValueError(
#             f"Either max_rank or fixed_rank has to be None, got max_rank={max_rank}, fixed_rank={fixed_rank}."
#         )
#     order = len(shape)
#     dtree = DimTree.from_order(order)

#     # Create rank dict
#     if fixed_rank is not None:
#         rank = {t.dims: fixed_rank for t in dtree.nodes()}
#     elif max_rank is not None:
#         rank = {t.dims: torch.randint(low=1, high=max_rank + 1, size=(1,)).item() for t in dtree.nodes()}
#     else:
#         # default max_rank = 5
#         rank = {t.dims: torch.randint(low=1, high=6, size=(1,)).item() for t in dtree.nodes()}
#     rank[dtree.root.dims] = 1

#     # Create leaf matrices
#     U = {}
#     for t in dtree.get_leaves():
#         size = (shape[t.dims[0]], rank[t.dims])
#         U[t.dims] = torch.randn(size) / (rank[t.dims] ** 0.5)  # type:ignore

#     # Generate transfer tensors
#     B = {}
#     for t in dtree.nodes(order="bottom-up"):
#         if t.is_leaf():
#             continue
#         dims = t.dims
#         left, right = t.get_children()
#         assert left is not None and right is not None
#         rl, rr, rt = rank[left.dims], rank[right.dims], rank[dims]
#         size = (rl, rr, rt)
#         B[t.dims] = torch.randn(size) / (rt**0.5)  # type:ignore

#     # Generate HTTensor
#     x = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)

#     # Orthogonalize if required
#     if orthog:
#         x = x.orthog()

#     return x


# def _randn_exp_decaying_svs(
#     shape: tuple[int, ...],
#     max_rank: int | None = None,
#     fixed_rank: int | None = None,
#     orthog: bool = False,
# ) -> HTTensor:
#     """Create a random HTTensor with exponentially decaying singular values.

#     - Each node's matricisation is constructed so that its singular values decay
#       exponentially. All nodes share one common rank determined by:
#         * fixed_rank, if given,
#         * a rank sampled from {1,...,max_rank} if max_rank is given,
#         * or 5 by default if neither is given.
#       In all cases, the rank is capped by min(shape) to respect the smallest
#       dimension.

#     - If both fixed_rank and max_rank are passed and fixed_rank>max_rank,
#       a ValueError is raised.

#     - Transfer tensors are initialized as identity-like blocks to preserve the
#       decaying singular-value structure, and orthogonalization can optionally
#       be applied to the final HTTensor.

#     Parameters
#     ----------
#     shape : tuple[int, ...]
#         Full shape of the desired HTTensor.
#     max_rank : int, optional
#         Maximum rank to sample from if fixed_rank is not specified.
#     fixed_rank : int, optional
#         Fixed rank to use for all nodes if max_rank is not specified.
#     orthog : bool, default False
#         If True, orthogonalize the resulting HTTensor before returning.

#     Returns
#     -------
#     htt
#         A randomly generated HTTensor with exponentially decaying singular
#         values and the specified properties.

#     """
#     if max_rank is not None and fixed_rank is not None:
#         if fixed_rank > max_rank:
#             raise ValueError(f"Given fixed_rank={fixed_rank} contradicts given max_rank={max_rank}.")
#     order = len(shape)
#     min_dim = min(shape)
#     dtree = DimTree.from_order(order)

#     # Determine global rank
#     if fixed_rank is not None:
#         rank = fixed_rank
#     elif max_rank is not None:
#         rank = torch.randint(low=1, high=max_rank, size=(1,)).item()
#     else:
#         # default max_rank = 5
#         rank = 5
#     rank = min(rank, min_dim)  # the smallest dim limits the rank

#     # Create leaf matrices
#     U = {}
#     for t in dtree.get_leaves():
#         dim = t.dims
#         nr_rows = shape[dim[0]]
#         U_ = torch.zeros((nr_rows, rank))  # type:ignore
#         diag_vals = torch.tensor([1 / (10**i) for i in range(rank)]) ** (1 / order)  # type:ignore
#         U_[:rank, :rank] = torch.diag(diag_vals)
#         Q = _get_orthog(nr_rows)
#         U_ = Q @ U_
#         U[dim] = U_

#     # Generate transfer tensors
#     B = {}
#     for t in dtree.nodes(order="bottom-up"):
#         if t.is_leaf():
#             continue
#         if t.is_root():
#             B_ = torch.eye(rank).unsqueeze(2)  # type:ignore
#             B[t.dims] = B_
#         else:
#             B_ = torch.zeros((rank, rank, rank))  # type:ignore
#             B_ = B_.fill_diagonal_(1.0)
#             B[t.dims] = B_

#     # Generate HTTensor
#     x = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)

#     # Orthogonalize if required
#     if orthog:
#         x = x.orthog()

#     return x


# def _get_orthog(n: int) -> torch.Tensor:
#     """Return a orthogonal torch.Tensor of shape (n,n)."""
#     q, _ = torch.linalg.qr(torch.randn((n, n)))
#     return q
