"""HTTensor addition module.

This module provides functionality to compute the formal sum of two HT tensors
by embedding their transfer tensors and leaf matrices into block-diagonal structures.
"""

from __future__ import annotations

import torch

from hiepy.httensor.httensor import HTTensor
from hiepy.utils.data_types import TensorDict
from hiepy.utils.dtype_device import check_device, result_dtype


def add(x: HTTensor, y: HTTensor) -> HTTensor:
    """
    Adds two HT tensors via block-diagonal embedding.

    Constructs an HT  tensor representing the exact sum of two HT tensors by embedding their
    leaf matrices and transfer tensors into a block-diagonal HT representation.

    The resulting tensor $Z = x + y$ is not computed in full tensor space, but
    represented exactly in HT format as:

    - Leaf matrices:
        $U_i^Z = [U_i^x, U_i^y]$

    - Transfer tensors:
        $B_t^Z = diag(B_t^x, B_t^y)$

    where the block-diagonal structure is applied at each node of the dimension tree.

    Parameters
    ----------
    x : HTTensor
        First HT tensor. Must share the same dimension tree and
        mode sizes as ``y``.

    y : HTTensor
        Second HT tensor. Must share the same dimension tree and
        mode sizes as ``x``.

    Returns
    -------
    HTTensor
        A new HT tensor representing the sum x + y in block-diagonal form.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` do not share the same dimension tree or shape.

    RuntimeError
        If ``x`` and ``y`` live on different devices.

    Notes
    -----
    - The result has the promoted dtype of ``x`` and ``y`` (e.g. ``float32`` and ``float64`` give ``float64``).
    - The dimension tree is preserved.
    - The tensor shape is preserved.
    - Rank growth is additive:
          $r_t^{out} = r_t^x + r_t^y$
    - Orthogonality is not preserved.
    - The result typically requires recompression.
    - The arithmetic cost of the operation is negligible, as it primarily involves the
      concatenation of leaf matrices and the creation of block-diagonal transfer tensors.

    Examples
    --------
    >>> import hiepy
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> x = hiepy.randn(shape=(8, 8, 8))
    >>> y = hiepy.randn(shape=(8, 8, 8))
    >>> sum_hiepy = hiepy.add(x,y)
    >>> x_t = x.full()
    >>> y_t = y.full()
    >>> sum_torch = x_t + y_t
    >>> torch.allclose(sum_torch, sum_hiepy.full())
    True
    """
    # Validity checks
    if not x.dtree == y.dtree:
        raise ValueError("Dimension trees of 'x' and 'y' are not compatible")
    if x.shape != y.shape:
        raise ValueError(f"Shape of 'y' {y.shape} does not match shape of 'x' {x.shape}")
    device = check_device(x=x, y=y)
    dtype = result_dtype(x, y)
    x, y = x.to(dtype=dtype), y.to(dtype=dtype)

    U: TensorDict = {}
    B: TensorDict = {}
    dtree = x.dtree.copy()

    # Concatenate leaf matrices
    for leaf in dtree.get_leaves():
        U[leaf] = torch.cat((x.U[leaf], y.U[leaf]), dim=1)

    # Embed the transfer tensors in a 3D blockdiagonal way
    for t in dtree.get_inner_nodes():
        if x.dtree.is_root(t):
            xl, xr, _ = x.B[t].shape  # rank of node x children
            yl, yr, _ = y.B[t].shape  # rank of node y children
            Bt = torch.zeros(xl + yl, xr + yr, 1, dtype=dtype, device=device)
            Bt[:xl, :xr] = x.B[t]
            Bt[xl:, xr:] = y.B[t]
        else:
            xl, xr, xk = x.B[t].shape  # rank of node children and node x (left, right, node)
            yl, yr, yk = y.B[t].shape  # rank of node children and node y (left, right, node)
            Bt = torch.zeros(xl + yl, xr + yr, xk + yk, dtype=dtype, device=device)
            Bt[:xl, :xr, :xk] = x.B[t]
            Bt[xl:, xr:, xk:] = y.B[t]
        B[t] = Bt

    # Create resulting HTTensor
    res = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)
    return res


# trunc_sum is currently not maintained

# def trunc_sum(summands: list[HTTensor], opts: HTTensorOpts) -> HTTensor:
#     """Add HTTensors contained in summands and truncate sum on the fly."""
#     # input validity checks
#     if not isinstance(summands, list):
#         raise TypeError(f"Argument 'summands' must be of type list, but got {type(summands)}.")

#     if not all(isinstance(item, HTTensor) for item in summands):
#         raise TypeError("Argument 'summands' must only contain HTTensor objects.")

#     if len(summands) < 2:
#         raise ValueError(f"Argument 'summands' must contain at least two elements, but got {len(summands)}.")

#     if not all(summands[0].dtree.is_equal(item.dtree) for item in summands):
#         raise ValueError(
#             "All HTTensors in 'summands' must share the same dimension tree, "
#             "but at least one tensor has a non-matching tree."
#         )
#     check_opts(opts)

#     # Copy first summand if required
#     U, B, dtree, opts = {}, {}, deepcopy(summands[0].dtree), deepcopy(opts)

#     # Anpassen der Fehlertoleranzen in opts
#     # Soll global der Fehler e eingehalten werden, muss der Kuerzungsfehler pro Knoten
#     # kleiner gleich e / sqrt((Tensorordnung * 2 - 2)) bleiben
#     adjusted_opts = deepcopy(opts)
#     if "atol" in adjusted_opts:
#         adjusted_opts["atol"] = adjusted_opts["atol"] / (x.order * 2 - 3) ** 0.5
#     if "rtol" in adjusted_opts:
#         adjusted_opts["rtol"] = adjusted_opts["rtol"] / (x.order * 2 - 3) ** 0.5

#     # Berechne die reduzierten Gram'schen Matrizen der impliziten Summe
#     G = gramians_sum(summands)

#     # Berechne gekuerzte Blattmatrizen der impliziten Summe
#     # Update dabei on the fly den Transfertensor des Elternknotens
#     B_upd = {node.dims: {} for node in dtree.get_inner_nodes()}
#     for leaf in dtree.get_leaves():
#         dim = leaf.dims
#         # Konkatenieren der Blattmatrizen
#         U_cat = torch.hstack(tuple(item.U[dim] for item in summands))
#         # QR Zerlegung
#         Q, R = torch.linalg.qr(U_cat, mode="reduced")
#         # Update reduzierte Gram'sche Matrix
#         G_upd = R @ G[dim] @ R.T
#         # Berechne davon linke Singulaervektoren
#         S, sv = left_svd_gramian(G_upd)
#         # Berechne basierend auf opts wie viele Spalten behalten werden
#         if adjusted_opts:
#             rank = trunc_rank(sv, **adjusted_opts)
#         else:
#             rank = len(sv)
#         S = S[:, :rank]
#         # Berechne schliesslich finale Blattmatrix
#         U[dim] = Q @ S

#         # Update Transfertensor des Elternknotens
#         # Adaptiere R entsprechend der Kuerzung
#         R = S.T @ R
#         # Teile dafuer zunaechst R in R=[R1 | R2 | ... | Rn]
#         # wobei die Spaltenanzahl von Ri durch den entsprechenden Rang des i-ten Summanden
#         # vorgegeben ist
#         R = torch.split(R, [item.rank[dim] for item in summands], dim=1)
#         # Multipliziere nun R[i] in den Transfertensor des Elternknotens des i-ten Summanden
#         assert leaf.father is not None
#         fdims = leaf.father.dims
#         for i in range(len(summands)):
#             if leaf.is_left():
#                 if i in B_upd[fdims]:
#                     B_upd[fdims][i] = torch.tensordot(R[i], B_upd[fdims][i], dims=([1], [0]))  # type: ignore
#                 else:
#                     B_upd[fdims][i] = torch.tensordot(R[i], summands[i].B[fdims], dims=([1], [0]))  # type: ignore
#             else:
#                 if i in B_upd[fdims]:
#                     B_upd[fdims][i] = torch.tensordot(R[i], B_upd[fdims][i], dims=([1], [1]))  # type: ignore
#                     B_upd[fdims][i] = torch.movedim(B_upd[fdims][i], source=0, destination=1)
#                 else:
#                     B_upd[fdims][i] = torch.tensordot(R[i], summands[i].B[fdims], dims=([1], [1]))  # type: ignore
#                     B_upd[fdims][i] = torch.movedim(B_upd[fdims][i], source=0, destination=1)

#     # Berechne nun gekuerzte Transfertensoren der impliziten Summe
#     # Durschreite den Dimensionsbaum dazu bottom-up
#     for level in dtree.levels(order="bottom-up"):
#         for node in level:
#             if node.is_leaf() or node.is_root():
#                 continue
#             dims = node.dims
#             # Konkateniere die aktualisierten Transfertensoren der Summanden
#             B_cat = torch.cat([B_upd[dims][i] for i in range(len(summands))], dim=2)
#             # Berechne QR Zerlegung der Matrizierung davon
#             Q, R = torch.linalg.qr(matricise(B_cat, t=(0, 1)), mode="reduced")
#             # Dematriziere Q zu 3D Transfertensor
#             Q = dematricise(Q, shape=(B_cat.shape[0], B_cat.shape[1], Q.shape[1]), t=(0, 1))

#             # Aktualisiere reduzierte Gram'sche Matrix
#             G_upd = R @ G[dims] @ R.T
#             # Berechne davon linke Singulaervektoren
#             S, sv = left_svd_gramian(G_upd)
#             # Berechne basierend auf opts wie viele Spalten behalten werden
#             if adjusted_opts:
#                 rank = trunc_rank(sv, **adjusted_opts)
#             else:
#                 rank = len(sv)
#             S = S[:, :rank]
#             # Berechne schliesslich finalen Transfertensor
#             B[dims] = torch.tensordot(Q, S, dims=([2], [0]))  # type: ignore

#             # Update Transfertensor des Elternknotens
#             # Adaptiere R entsprechend der Kuerzung
#             R = S.T @ R
#             # Teile dafuer zunaechst R in R=[R1 | R2 | ... | Rn]
#             # wobei die Spaltenanzahl von Ri durch den entsprechenden Rang des i-ten Summanden
#             # vorgegeben ist
#             R = torch.split(R, [item.rank[dims] for item in summands], dim=1)
#             # Multipliziere nun R[i] in den Transfertensor des Elternknotens des i-ten Summanden
#             assert node.father is not None
#             fdims = node.father.dims
#             for i in range(len(summands)):
#                 if node.is_left():
#                     if i in B_upd[fdims]:
#                         B_upd[fdims][i] = torch.tensordot(R[i], B_upd[fdims][i], dims=([1], [0]))  # type: ignore
#                     else:
#                         B_upd[fdims][i] = torch.tensordot(
#                             R[i],
#                             summands[i].B[fdims],
#                             dims=([1], [0]),  # type: ignore
#                         )
#                 else:
#                     if i in B_upd[fdims]:
#                         B_upd[fdims][i] = torch.tensordot(R[i], B_upd[fdims][i], dims=([1], [1]))  # type: ignore
#                         B_upd[fdims][i] = torch.movedim(B_upd[fdims][i], source=0, destination=1)
#                     else:
#                         B_upd[fdims][i] = torch.tensordot(
#                             R[i],
#                             summands[i].B[fdims],
#                             dims=([1], [1]),  # type: ignore
#                         )
#                         B_upd[fdims][i] = torch.movedim(B_upd[fdims][i], source=0, destination=1)
#             # Gebe B_upd[node] frei
#             del B_upd[dims]

#     # Wurzelbehandlung..
#     B_root = torch.sum(torch.concat([item for item in B_upd[dtree.root.dims].values()], dim=2), dim=2)
#     B[dtree.root.dims] = B_root.reshape(B_root.shape[0], B_root.shape[1], 1)

#     # Erstelle HTucker Tensor der Summe
#     z = HTTensor(U=U, B=B, dtree=dtree, is_orthog=False)

#     return z
