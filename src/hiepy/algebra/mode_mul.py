"""
Implements the mode-product of a matrix with an HT  tensor.

Contains a function that implements the multiplication of a matrix to an HT  tensors
along a specified dimension/mode.
"""

import torch

from hiepy.httensor.httensor import HTTensor as HTTensor


def mode_mul(x: HTTensor, A: torch.Tensor, dim: int) -> HTTensor:
    r"""
    Multiplies matrix ``A`` to HT tensor ``x`` along mode ``dim``.

    The mode-product of a tensor $X\in \mathbb{R}^{n_0 \times \dots \times n_{d-1}}$ with
    a matrix $A \in \mathbb{R}^{m \times n_{\mu}}$ along mode $\mu \in \{0,\dots,d-1\}$ is defined as

    $$ (A \times_{\mu} X)_{i_0,\dots,i_{d-1}} = \sum\limits^{n_{\mu}-1}_{j=0}A_{i_{\mu},j}
    X_{i_0,\dots,i_{\mu-1},j,i_{\mu+1},\dots,i_{d-1}} .$$

    The operation is performed directly on ``x`` by multiplying ``A`` with the leaf matrix associated
    with mode $\mu$.

    Parameters
    ----------
    x : HTTensor
        Input HT tensor to be multiplied.

    A : torch.Tensor
        Input matrix of shape $(m, n_{\mu})$.

    dim : int
        Mode along which ``x`` is multiplied with ``A``.

    Returns
    -------
    HTTensor
        A new HT tensor representing the mode product.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional.

    ValueError
        If ``dim`` is outside the valid mode range.

    ValueError
        If the shapes of ``A`` and ``x`` don't match.

    Notes
    -----
    - The dimension tree is preserved.
    - Hierarchical ranks are preserved.
    - Tensor shape is not preserved.
    - Orthogonality is not preserved.
    - Computational cost is linear in mode size $n$, the number of columns of the corresponding leaf matrix $c$, and
      the number of rows of matrix ``A`` $m$: $\mathcal{O}(ncm)$.
    - All transfer tensors remain unchanged.
    - All but one leaf matrices remain unchanged.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> x = hiepy.randn((3,4,5))
    >>> A = torch.randn(2, 4)
    >>> res = hiepy.mode_mul(x=x, A=A, dim=1)
    >>> res.shape
    (3, 2, 5)
    >>> res.rank == x.rank
    True
    """
    if len(A.shape) != 2:
        raise ValueError(f"Argument 'A' must be a 2D torch.Tensor, got shape {A.shape}")
    if dim < 0 or dim >= x.order:
        raise ValueError(f"Argument 'dim' is out of range for this HTTensor (dim={dim}, order={x.order})")
    if x.shape[dim] != A.shape[1]:
        raise ValueError(
            f"Mismatch between dimension {dim} of x (size {x.shape[dim]}) and number or columns of "
            f"A (size {A.shape[1]})."
        )
    x = x.copy()

    # Multiply the leaf matrix corresponding to the specified mode
    node = (dim,)
    x.U[node] = torch.tensordot(A, x.U[node], dims=([1], [0]))  # type:ignore

    # Set orthog flag to False
    x.is_orthog = False
    return x
