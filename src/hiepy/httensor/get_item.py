"""HTTensor entry access utilities.

Provides functions to access entries of HTTensors with intuitive slicing,
allowing users to index and slice HTTensors as if they were regular N-D
torch tensors.
"""

from __future__ import annotations

from operator import index as _as_index

import torch

from hiepy.httensor.full import full
from hiepy.httensor.httensor import HTTensor as HTTensor
from hiepy.transformation.squeeze import squeeze
from hiepy.utils.data_types import IndexKey, Shape


def get(x: HTTensor, key: IndexKey | tuple[IndexKey, ...], track_grad: bool = False) -> HTTensor | torch.Tensor:
    r"""
    Indexes an HT tensor with ints and slices, following the semantics of torch indexing.

    The ``i``-th entry of ``key`` addresses mode ``i`` of ``x``. Modes not addressed by ``key`` are kept
    completely. The selection is applied to the corresponding leaf matrices; afterwards, all modes indexed
    by an integer are contracted into the remaining HT representation.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor.

    key : IndexKey | tuple[IndexKey, ...]
        Index specification. A single entry is applied to the first mode, a tuple is applied mode-wise
        to the leading ``len(key)`` modes:

        - int-like: fixes the mode to the given index and removes it.
        - slice: keeps the mode and selects the given range (also if the range has length one).

        Negative indices and slice bounds count from the end of the respective mode. Slice steps must
        be positive.

    track_grad : bool, optional
        If ``True``, the result stays connected to the autograd graph of ``x``. Otherwise, a detached
        copy is returned. Defaults to ``False``.

    Returns
    -------
    HTTensor | torch.Tensor
        The indexed tensor, depending on the number of remaining modes:

        - at least two modes: a new HT tensor,
        - one mode: a dense one-dimensional torch tensor,
        - no mode: a zero-dimensional torch tensor.

    Raises
    ------
    TypeError
        If ``key`` or one of its entries is neither int-like nor a slice (e.g. ``bool``, ``float``, ``None``,
        ``Ellipsis``, or a ``list``).

    IndexError
        If an index is out of bounds or ``key`` has more entries than ``x`` has modes.

    ValueError
        If a slice step is not positive.

    Notes
    -----
    - The input tensor is not modified, and the result does not share memory with it.
    - Advanced indexing (lists, index tensors, boolean masks), ``None`` and ``Ellipsis`` are not supported.
      In particular, ``x[[0, 1]]`` is rejected instead of being interpreted mode-wise; use ``x[0, 1]``.
    - Orthogonality is preserved only if ``key`` selects every entry of every mode.
    - Hierarchical ranks do not increase.
    - Integer-indexed modes are removed via ``hiepy.squeeze``; the dimension tree may therefore no longer
      be canonical.
    - No dense tensor of the size of ``x`` is formed.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> x = torch.arange(1, 28, dtype=torch.double).reshape(3, 3, 3)
    >>> x_ht = hiepy.to_htt(x, {"atol": 1e-8})
    >>> hiepy.get(x_ht, (slice(0, 2), slice(0, 2))).shape
    (2, 2, 3)
    >>> x_ht[0:1].shape
    (1, 3, 3)
    >>> x_ht[0, 1]
    tensor([4.0000, 5.0000, 6.0000])
    >>> x_ht[0, 1, 2]
    tensor(6.0000)
    >>> x_ht[-1, -1, -1]
    tensor(27.)
    """
    shape = x.shape
    entries = _normalize_key(key, shape)

    # Select the requested rows of the addressed leaf matrices.
    # Integer indices keep a singleton row, which is contracted away below.
    U = dict(x.U)
    removed_dims = []
    selects_all = True
    for dim, entry in enumerate(entries):
        leaf = (dim,)
        if isinstance(entry, int):
            U[leaf] = x.U[leaf][entry : entry + 1, :]
            removed_dims.append(dim)
            selects_all = False
        elif entry.indices(shape[dim]) != (0, shape[dim], 1):
            U[leaf] = x.U[leaf][entry, :]
            selects_all = False

    # Shallow container; squeeze/copy/clone below create the independent result
    z = HTTensor(U=U, B=x.B, dtree=x.dtree, is_orthog=x.is_orthog and selects_all)

    # HT tensors require at least two modes; smaller results are returned densely.
    # Since all removed modes have size one, the dense result has at most max(shape) entries.
    nr_remaining_dims = len(shape) - len(removed_dims)
    if nr_remaining_dims < 2:
        out_shape = tuple(z.shape[dim] for dim in range(len(shape)) if dim not in removed_dims)
        res = full(z).reshape(out_shape)
        return res if track_grad else res.detach()

    if removed_dims:
        return squeeze(z, dims=removed_dims, track_grad=track_grad)

    return z.clone() if track_grad else z.copy()


def _normalize_key(key: IndexKey | tuple[IndexKey, ...], shape: Shape) -> list[int | slice]:
    r"""
    Validates ``key`` and resolves it into one entry per addressed mode.

    Parameters
    ----------
    key : IndexKey | tuple[IndexKey, ...]
        Index specification as passed to ``get``.

    shape : Shape
        Shape of the indexed HT tensor.

    Returns
    -------
    list[int | slice]
        Entry ``i`` addresses mode ``i``. Integer indices are converted to non-negative ints,
        slices are returned unchanged.

    Raises
    ------
    TypeError
        If an entry is neither int-like nor a slice.

    IndexError
        If ``key`` has more entries than ``shape`` or an integer index is out of bounds.

    ValueError
        If a slice step is not positive.
    """
    entries = key if isinstance(key, tuple) else (key,)
    if len(entries) > len(shape):
        raise IndexError(f"Too many indices: got {len(entries)}, but the HT tensor has order {len(shape)}.")

    normalized: list[int | slice] = []
    for dim, entry in enumerate(entries):
        if isinstance(entry, slice):
            if entry.step is not None and _as_index(entry.step) <= 0:
                raise ValueError(f"Slice step must be positive, got {entry.step} for dimension {dim}.")
            normalized.append(entry)
            continue

        # bool is a subclass of int but denotes a mask in torch, so it is rejected explicitly
        if isinstance(entry, bool):
            raise TypeError(f"Boolean indices are not supported, got {entry} for dimension {dim}.")
        try:
            idx = _as_index(entry)
        except TypeError:
            raise TypeError(
                f"Unsupported index {entry!r} for dimension {dim}: expected an int or a slice. "
                "Use a tuple to index several modes."
            ) from None

        size = shape[dim]
        if not -size <= idx < size:
            raise IndexError(f"Index {idx} is out of bounds for dimension {dim} with size {size}.")
        normalized.append(idx % size)

    return normalized
