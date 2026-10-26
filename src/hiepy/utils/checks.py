"""Argument validation shared by several operations (internal).

This module provides checks for shapes and truncation options, so that invalid arguments raise a clear error at the
entry of an operation instead of failing later in an unrelated place.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from hiepy.utils.data_types import Shape

TRUNC_OPTS_KEYS = ("atol", "rtol", "max_rank")


def check_shape(shape: Shape) -> None:
    """Raise ``ValueError`` if a mode of ``shape`` has size smaller than one.

    HT tensors have no empty modes: every mode size, and hence every leaf matrix, has at least one row.
    """
    empty = [dim for dim, size in enumerate(shape) if size < 1]
    if empty:
        raise ValueError(f"All mode sizes must be at least 1, got shape {tuple(shape)} (modes {empty}).")


def check_trunc_opts(opts: Mapping[str, object]) -> None:
    """Validate truncation options.

    Raises
    ------
    ValueError
        If ``opts`` contains a key other than ``atol``, ``rtol`` and ``max_rank``, if ``atol`` or ``rtol`` is negative
        or not finite, or if ``max_rank`` is smaller than 1.
    TypeError
        If ``atol`` or ``rtol`` is not a real number, or ``max_rank`` is not an integer.
    """
    unknown = sorted(set(opts) - set(TRUNC_OPTS_KEYS))
    if unknown:
        raise ValueError(f"Unknown truncation option(s) {unknown}; supported are {list(TRUNC_OPTS_KEYS)}.")
    for key in ("atol", "rtol"):
        if key not in opts:
            continue
        value = opts[key]
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Truncation option '{key}' must be a real number, got {type(value).__name__}.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Truncation option '{key}' must be finite and non-negative, got {value}.")
    if "max_rank" in opts:
        max_rank = opts["max_rank"]
        if isinstance(max_rank, bool) or not isinstance(max_rank, int):
            raise TypeError(f"Truncation option 'max_rank' must be an int, got {type(max_rank).__name__}.")
        if max_rank < 1:
            raise ValueError(f"Truncation option 'max_rank' must be at least 1, got {max_rank}.")
