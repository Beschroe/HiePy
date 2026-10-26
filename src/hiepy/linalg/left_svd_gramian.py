"""Gramian-based singular value decomposition utility.

This module provides functionality to efficiently compute the left singular
vectors and singular values of an implicit matrix via its Gram matrix.
"""

import torch


def left_svd_gramian(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes left singular vectors and singular values from a Gram matrix.

    Let $A \in \mathbb{R}^{m \times n}$ and suppose

    $$
    G = AA^\top.
    $$

    This function computes the eigenvalue decomposition of $G$ to recover the
    left singular vectors and singular values of $A$. If

    $$
    G = Q \Lambda Q^\top,
    $$

    then the columns of $Q$ are the left singular vectors of $A$, while the
    singular values are given by $\sigma_i = \sqrt{\lambda_i}$.

    Parameters
    ----------
    G : torch.Tensor
        Symmetric positive semidefinite Gram matrix.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple `(U, S)`, where

        - ``U`` contains the left singular vectors as columns.
        - `S` contains the singular values in descending order.

    Raises
    ------
    ValueError
        If ``G`` has non-finite entries, e.g. because the Gram matrix of a tensor with norm beyond about $10^{154}$
        overflows in float64.

    Notes
    -----
    - The input is assumed to be symmetric and positive semidefinite.

    - This routine computes singular values indirectly through an eigenvalue decomposition and therefore
      operates on squared singular values. It is generally less numerically robust than a direct singular value
      decomposition when very small singular values are present.


    Examples
    --------
    >>> import torch
    >>> import hiepy
    >>> torch.set_default_dtype(torch.float64)
    >>> V = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> G = V @ V.T  # Construct the Gram matrix
    >>> # Compare singular values of left_svd_gramian to directly computed singular values
    >>> _, s_gram = hiepy.left_svd_gramian(G)
    >>> _, s_svd, _ = torch.linalg.svd(V)
    >>> torch.allclose(s_svd, s_gram)
    True
    """

    if not torch.isfinite(G).all():
        raise ValueError(
            "The Gram matrix is not finite; the tensor is outside the supported range (norm up to about 1e154 in "
            "float64, beyond which the Gram matrix overflows)."
        )

    # Eig decomposition
    eig_val, Q = torch.linalg.eigh(G)

    # Retrieve singular values
    s_val = torch.sqrt(torch.abs(eig_val))

    # Get index corresponding to a descending sv order
    desc_idc = torch.argsort(s_val, dim=0, descending=True)
    return Q[:, desc_idc], s_val[desc_idc]
