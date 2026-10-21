"""HTTensor reshape utility.

This module provides functionality to remove singleton dimensions from HT tensors.
"""

from __future__ import annotations

from typing import Iterable, SupportsInt

import torch

from hiepy.httensor.httensor import HTTensor


def squeeze(x: HTTensor, dims: Iterable[SupportsInt] | SupportsInt | None = None) -> HTTensor | torch.Tensor:
    r"""
    Removes singleton dimensions from input hierarchical Tucker tensor.

    Case 1) `dims == None`: Removes all singleton dimensions.

    Case 2) `dims != None`: Removes only the singleton dimensions specified in ``dims``.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.
    dims : Iterable[SupportsInt] | SupportsInt | None, optional
        Dimensions to remove. If None, all singleton dimensions are removed.

    Returns
    -------
    HTTensor | torch.Tensor
        The squeezed tensor. If the tensor is completely reduced (i.e., all dimensions are singletons and squeezed),
        the resulting 0d torch tensor is returned. Otherwise a new and squeezed HT tensor is returned.

    Raises
    ------
    TypeError
        If ``dims`` is not an integer-like type or an iterable of integer-likes.
    ValueError
        If any of the specified dimensions are out of bounds or not singletons.

    Notes
    -----
    - Orthogonality is not preserved.
    - The dimension tree may no longer be canonical after this operation.
    - An HT tensor cannot be squeezed to order 1. If squeezing would leave only one dimension,
      the highest singleton dimension(s) are retained.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.randn(shape=(1, 8, 1, 8))
    >>> x.shape
    (1, 8, 1, 8)
    >>> x_squeezed = hiepy.squeeze(x)
    >>> x_squeezed.shape
    (8, 8)
    """
    order = x.order
    shape = x.shape

    # Argument parsing and validation
    if dims is None:
        dims_set = {i for i, s in enumerate(shape) if s == 1}
    elif isinstance(dims, SupportsInt):
        dims_set = {int(dims)}
    elif isinstance(dims, Iterable):
        dims_set = {int(d) for d in dims}
    else:
        raise TypeError(f"dims must be an int-like or an Iterable of int-likes, got {type(dims)}.")

    out_of_bounds = {d for d in dims_set if d < 0 or d >= order}
    if out_of_bounds:
        raise ValueError(f"Dimensions {out_of_bounds} are out of bounds for order {order}.")

    non_singletons = {d for d in dims_set if shape[d] != 1}
    if non_singletons:
        raise ValueError(f"Dimensions {non_singletons} are not singletons. Tensor shape: {shape}")

    # Fallbacks: Complete Squeeze & Minimum Order
    if len(dims_set) == order:
        return torch.squeeze(x.full())

    # Guarantee order >= 2 (remove the highest dimensions from the set if necessary)
    while order - len(dims_set) < 2:
        dims_set.remove(max(dims_set))

    x = x.clone()

    if not dims_set:
        return x

    # Topological Contraction (bottom-up)
    # Iterate in descending order so indices do not shift
    for dim in sorted(dims_set, reverse=True):
        next_leaf = [node for node in x.dtree.get_leaves() if dim in node]
        if not next_leaf:
            continue
        leaf = next_leaf[0]

        if len(leaf) > 1:
            # Multi-dimensional leaf: Remove a dimension without contraction
            idx = leaf.index(dim)
            new_leaf = tuple(d for d in leaf if d != dim)

            # Update leaf matrix
            x.U[new_leaf] = torch.squeeze(x.U[leaf], dim=idx)

            # Update tree
            x.dtree.map(old_to_new=lambda node: new_leaf if node == leaf else node)

            # Free memory
            del x.U[leaf]
            continue

        # Single-dimensional leaf: Absord leaf into father
        father = x.dtree.get_parent(leaf)
        sibling = x.dtree.get_sibling(leaf)

        # Contract singleton dimension
        u_vec = x.U[leaf].squeeze(0)

        # Multiply into father's tensor
        contract_dim = 0 if x.dtree.is_left(leaf) else 1
        B_new = torch.tensordot(u_vec, x.B[father], dims=([0], [contract_dim]))  # type:ignore

        # Free memory
        del x.U[leaf]
        del x.B[father]

        if x.dtree.is_leaf(sibling):
            # Sibling node is a leaf: Absorb it into father node, which becomes a leaf
            # Update tree
            x.dtree.prune(father)
            new_father = sibling
            x.dtree.map(old_to_new=lambda node: new_father if node == father else node)

            # Update leaf matrix dict
            x.U[new_father] = x.U[sibling] @ B_new

        else:
            # Sibling node is internal: Absorb it into father, which will take over sibling's descendant
            # Update tree
            x.dtree.move_subtree(subtree_root=sibling, new_parent=father)
            new_father = sibling
            x.dtree.map(old_to_new=lambda node: new_father if node == father else node)

            # Update transer tensor
            B_new_inner = torch.tensordot(x.B[sibling], B_new, dims=([2], [0]))  # type:ignore
            x.B[new_father] = B_new_inner

    #  Global index correction
    remaining_dims = sorted(set(range(order)) - dims_set)
    dim_map = {old_dim: new_dim for new_dim, old_dim in enumerate(remaining_dims)}

    def remap_dims(dims_tuple: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(dim_map[d] for d in dims_tuple if d in dim_map)

    # Overwrite the dictionaries with the new gapless keys
    x.U = {remap_dims(k): v for k, v in x.U.items()}
    x.B = {remap_dims(k): v for k, v in x.B.items()}

    # Update the keys in the DimTree
    x.dtree.map(old_to_new=remap_dims)

    # Update orthog flag
    x.is_orthog = False
    return x
