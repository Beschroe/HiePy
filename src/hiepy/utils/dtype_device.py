"""
dtype and device handling for operations with several operands.

This module provides the internal helpers that determine the result dtype of an operation and ensure that all
operands live on the same device. Both helpers accept ``torch.Tensor`` objects as well as ``HTTensor`` objects,
since both expose ``dtype`` and ``device`` attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hiepy.httensor.httensor import HTTensor


def result_dtype(*operands: HTTensor | torch.Tensor) -> torch.dtype:
    """
    Returns the dtype an operation on several operands should produce.

    The dtypes of all operands are combined via ``torch.promote_types``, i.e. the more precise dtype wins
    (``torch.float32`` and ``torch.float64`` give ``torch.float64``).

    Parameters
    ----------
    *operands : HTTensor | torch.Tensor
        The operands of the operation. Zero-dimensional tensors and Python scalars are not meant to be passed here,
        since by torch's type promotion rules they do not raise the dtype of a tensor (use ``torch.result_type``).

    Returns
    -------
    torch.dtype
        The promoted dtype of all operands.

    Examples
    --------
    >>> import torch
    >>> from hiepy.utils.dtype_device import result_dtype
    >>> result_dtype(torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.float64))
    torch.float64
    """
    dtype = operands[0].dtype
    for operand in operands[1:]:
        dtype = torch.promote_types(dtype, operand.dtype)
    return dtype


def check_device(**operands: HTTensor | torch.Tensor) -> torch.device:
    """
    Checks that all operands live on the same device and returns that device.

    Parameters
    ----------
    **operands : HTTensor | torch.Tensor
        The operands of the operation, passed as keyword arguments. The keywords are used as operand names in the
        error message.

    Returns
    -------
    torch.device
        The common device of all operands.

    Raises
    ------
    RuntimeError
        If two operands live on different devices.

    Examples
    --------
    >>> import torch
    >>> from hiepy.utils.dtype_device import check_device
    >>> check_device(x=torch.ones(2), A=torch.ones(2, device="meta"))
    Traceback (most recent call last):
    ...
    RuntimeError: Expected all tensors to be on the same device, got 'cpu' (x) and 'meta' (A).
    """
    names = list(operands)
    device = operands[names[0]].device
    for name in names[1:]:
        if operands[name].device != device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, got '{device}' ({names[0]}) "
                f"and '{operands[name].device}' ({name})."
            )
    return device
