"""
Common type definitions for HT tensor operations.

This module defines shared type aliases and configuration structures used
throughout the HT tensor implementation.
"""

from __future__ import annotations

from typing import SupportsIndex, TypedDict

import torch

from hiepy.utils.dimtree import Node

type Shape = tuple[int, ...]
"""Tensor shape represented as a tuple of dimension sizes.

Each entry is a positive integer specifying the size of one mode,
e.g. ``(3, 1, 4)`` for a third-order tensor.
"""

type TensorDict = dict[Node, torch.Tensor]
"""Mapping from dimension tree nodes to their associated tensors.

Leaf nodes map to leaf matrices, while inner nodes map to transfer tensors.
"""

type ScalarType = float | int | torch.Tensor
"""Scalar type accepted by algebraic operations.

Includes Python numeric types as well as zero-dimensional ``torch.Tensor``
objects.
"""

type IndexKey = SupportsIndex | slice
"""Index of a single mode accepted by HT tensor indexing.

Either an integer-like object (``int``, zero-dimensional integer ``torch.Tensor``, ...),
which removes the mode, or a ``slice``, which keeps it. Negative values count from the end.
"""


class TruncOpts(TypedDict, total=False):
    """Optional parameters controlling hierarchical rank truncation.

    Attributes
    ----------
    atol: float
        Absolute truncation tolerance.
    rtol: float
        Relative truncation tolerance.
    max_rank: int
        Upper bound for the hierarchical ranks.
    """

    atol: float
    rtol: float
    max_rank: int
