"""
Truncation rank selection utilities.

This module provides functionality to determine the minimal truncation rank
required to approximate a tensor or matrix (e.g. in hierarchical Tucker
decompositions) under combined absolute and relative error tolerances.

The rank is selected based on the decay of singular values and an optional
maximum rank constraint.
"""

from warnings import warn

import torch

MAX_RANK_DEFAULT = 10000


def trunc_rank(
    singular_values: torch.Tensor,
    atol: float = 0.0,
    rtol: float = 0.0,
    max_rank: int = MAX_RANK_DEFAULT,
    disable_warning: bool = True,
) -> int:
    r"""
    Rank selection from singular value decay under Frobenius norm error control.

    The rank $r$ is chosen such that the Frobenius norm of the discarded tail $\lvert \lvert  s_{r+1:}\rvert\rvert_2$
    does not exceed the tolerance $\text{atol} + \text{rtol} * \lvert\lvert s \rvert\rvert_2$, where $s$ is the vector
    of singular values.

    If the required rank exceeds ``max_rank``, it is clipped to ``max_rank``,
    and the tolerance constraint may be violated.

    Parameters
    ----------
    singular_values : torch.Tensor
        1D tensor of singular values sorted in descending order.
    atol : float, optional
        Absolute error tolerance. Default is $0.0$.
    rtol : float, optional
        Relative error tolerance (with respect to the Euclidean norm of all
        singular values). Default is $0.0$.
    max_rank : int, optional
        Maximum allowed rank. Default is $10000$.
    disable_warning : bool, optional
        If True, warnings regarding ``max_rank`` violations are suppressed.
        Default is True.

    Returns
    -------
    int
        Selected truncation rank.

    Warns
    -----
    UserWarning
        If the computed rank exceeds ```max_rank```. In this case, ```max_rank``` is
        returned and error tolerances may not be satisfied.
    """
    # compute strict thresh
    norm_s = torch.linalg.norm(singular_values, ord=2)
    tol = atol + rtol * norm_s

    # compute truncation errors
    tail_norm = torch.sqrt(torch.cumsum(singular_values.flip(0) ** 2, dim=0)).flip(0)
    tail_norm = torch.cat([tail_norm, tail_norm.new_zeros(1)])

    # compute required rank
    rank = max(int(torch.where(tail_norm <= tol)[0][0]), 1)  # for the case, where no entry is
    # needed to satisfy the err tol
    # compare to allowed rank
    if rank > max_rank:
        if not disable_warning:
            warn(f"Requested rank {rank} > max_rank {max_rank}. Truncating to max_rank.")
        rank = max_rank
    return rank
