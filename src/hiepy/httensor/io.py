"""IO utilities for HTTensor.

This module provides basic serialization and deserialization utilities for
HTTensor objects using PyTorch's built-in checkpointing mechanism.
"""

from __future__ import annotations

from pathlib import Path

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor


def save(x: HTTensor, path: str | Path) -> None:
    r"""
    Serializes an HT  tensor to disk.

    The tensor is stored using `torch.save` and can be restored via `load`.

    Parameters
    ----------
    x : HTTensor
        HT tensor to be saved.

    path : str | Path
        Destination file path. Must end with `.htt`.

    Raises
    ------
    ValueError
        If `path` does not end with `.htt`.

    Notes
    -----
    - Serialization is performed via PyTorch pickling.
    - Saved objects include full HT structure (dimension tree, tensors, and metadata).
    """
    if isinstance(path, Path):
        path = str(path.resolve())
    if not path.endswith(".htt"):
        raise ValueError(f"Argument `path` must end with '.htt', got {path}.")
    torch.save(x, path)


def load(path: str | Path, map_location: torch.device | str | None = None) -> HTTensor:
    r"""
    Deserializes an HT  tensor from disk.

    Parameters
    ----------
    path : str | Path
        Path to a file previously created using `save`.
    map_location : torch.device | str, optional
        Device to load all tensor components onto. If ``None``, the components are restored on the device they were
        saved from.

    Returns
    -------
    HTTensor
        Restored HT tensor.

    Raises
    ------
    ValueError
        If `path` does not end with `.htt`.
    TypeError
        If the file does not contain an `HTTensor`.

    Notes
    -----
    - Uses `torch.load` with `weights_only=False` to restore full objects.
    """
    if isinstance(path, Path):
        path = str(path.resolve())
    if not path.endswith(".htt"):
        raise ValueError(f"Argument `path` must end with '.htt', got {path}.")

    x = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(x, HTTensor):
        raise TypeError(f"File {path} does not contain an HTTensor, got {type(x).__name__}.")
    return x
