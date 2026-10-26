"""hypothesis strategies for HT tensors.

Tensors are built by ``build_ht`` from a specification drawn by hypothesis. The specification is reported with
``note``, so a failing example can be rebuilt by calling ``build_ht`` with the printed arguments. Tensors are
float32 or float64; components are generated in float64 and converted at the end, so a seed gives the same tensor in
both precisions.

The specification varies, independently per node of the tree:

- the rank, including over-parameterized ranks (larger than the number of rows of the component),
- the kind of component: ``"orthonormal"``, ``"random"`` (general) or ``"deficient"`` (linearly dependent columns),

so tensors can be orthogonal in some nodes only. In addition, it varies the overall scale and includes the zero
tensor. The dimension tree is canonical, non-canonical as produced by ``squeeze``, or an arbitrary binary tree:
unbalanced, with non-contiguous nodes such as ``(0, 2) | (1, 3)``, and with the larger modes in the left child.
"""

from __future__ import annotations

import torch
from hypothesis import note
from hypothesis import strategies as st

import hiepy
from hiepy import DimTree, HTTensor

MAX_ORDER = 6
MAX_SIZE = 5
MAX_RANK = 4
KINDS = ("orthonormal", "random", "deficient")
SCALES = (1e-3, 1.0, 1e3)


def dtypes() -> st.SearchStrategy[torch.dtype]:
    """float32 or float64."""
    return st.sampled_from([torch.float64, torch.float32])


@st.composite
def shapes(draw: st.DrawFn, min_order: int = 2, max_order: int = MAX_ORDER, max_size: int = MAX_SIZE) -> tuple:
    """Shapes of order ``min_order … max_order`` with mode sizes ``1 … max_size`` (singletons included)."""
    order = draw(st.integers(min_order, max_order))
    return tuple(draw(st.lists(st.integers(1, max_size), min_size=order, max_size=order)))


def _component(rows: int, rank: int, kind: str, gen: torch.Generator) -> torch.Tensor:
    """A ``rows x rank`` matrix of the given kind (an orthonormal one has at most ``rows`` columns)."""
    if kind == "orthonormal":
        q, _ = torch.linalg.qr(torch.randn(rows, rank, generator=gen, dtype=torch.float64), mode="reduced")
        return q
    M = torch.randn(rows, rank, generator=gen, dtype=torch.float64)
    if kind == "deficient" and rank >= 2:
        M[:, -1] = 2.0 * M[:, 0]
    return M


def build_ht(
    shape: tuple[int, ...],
    inserted: tuple[int, ...],
    nodes: dict[hiepy.Node, tuple[int, str]],
    seed: int,
    zero: bool,
    scale: float,
    orthog: bool,
    children: dict[hiepy.Node, tuple[hiepy.Node, hiepy.Node]] | None = None,
    dtype: torch.dtype = torch.float64,
) -> HTTensor:
    """Build an HT tensor from a specification.

    Parameters
    ----------
    shape
        Shape of the result.
    inserted
        Positions of singleton modes inserted into ``shape`` before building and squeezed out afterwards; a non-empty
        tuple yields a non-canonical tree.
    nodes
        Rank and kind per non-root node of the tree over ``len(shape) + len(inserted)`` modes.
    seed
        Seed of the random components.
    zero
        Whether the result is the zero tensor (zero root transfer tensor).
    scale
        Factor applied to the root transfer tensor.
    orthog
        Whether the result is orthogonal (orthogonalized if necessary). If ``False``, the flag is ``False``.
    children
        Child mapping of an arbitrary tree over ``len(shape) + len(inserted)`` modes; ``None`` for the canonical tree.
    dtype
        dtype of the result; ``squeeze`` and ``orthog`` run in this precision.
    """
    ext_shape = list(shape)
    for pos in sorted(inserted):
        ext_shape.insert(pos, 1)
    order = len(ext_shape)
    dtree = DimTree.from_order(order) if children is None else DimTree(tuple(range(order)), dict(children))
    gen = torch.Generator().manual_seed(seed)
    U, B, ranks = {}, {}, {}
    all_orthonormal = True
    for t in dtree.get_nodes(order="bottom-up"):
        if dtree.is_leaf(t):
            rows = ext_shape[t[0]]
        else:
            left, right = dtree.get_children(t)
            rows = ranks[left] * ranks[right]
        if dtree.is_root(t):
            M = torch.randn(rows, 1, generator=gen, dtype=torch.float64) * (0.0 if zero else scale)
        else:
            rank, kind = nodes[t]
            M = _component(rows, rank, kind, gen)
            all_orthonormal &= kind == "orthonormal"
        ranks[t] = M.shape[1]
        if dtree.is_leaf(t):
            U[t] = M
        else:
            B[t] = M.reshape(ranks[left], ranks[right], ranks[t])
    x = HTTensor(U=U, B=B, dtree=dtree, is_orthog=orthog and all_orthonormal).to(dtype=dtype)
    if inserted:
        x = hiepy.squeeze(x, dims=sorted(inserted))
    if orthog and not x.is_orthog:
        x = hiepy.orthog(x)
    return x


@st.composite
def arbitrary_trees(draw: st.DrawFn, order: int) -> dict[hiepy.Node, tuple[hiepy.Node, hiepy.Node]]:
    """Child mapping of an arbitrary binary tree over ``order`` modes: every split of a node into two non-empty
    parts is possible (unbalanced, non-contiguous), and either part may be the left child."""
    children = {}
    stack = [tuple(range(order))]
    while stack:
        node = stack.pop()
        if len(node) == 1:
            continue
        perm = draw(st.permutations(node))
        k = draw(st.integers(1, len(node) - 1))
        left, right = tuple(sorted(perm[:k])), tuple(sorted(perm[k:]))
        children[node] = (left, right)
        stack += [left, right]
    return children


type TreeSpec = tuple[tuple[int, ...], tuple[int, ...], dict[hiepy.Node, tuple[hiepy.Node, hiepy.Node]] | None]


@st.composite
def tree_specs(draw: st.DrawFn, max_order: int = MAX_ORDER) -> TreeSpec:
    """Shape, positions of inserted singleton modes (squeezed out after building) and an optional arbitrary tree
    over the extended order (``None``: canonical)."""
    shape = draw(shapes(max_order=max_order))
    nr_inserted = draw(st.integers(0, 2))
    positions = draw(st.lists(st.integers(0, len(shape)), min_size=nr_inserted, max_size=nr_inserted))
    inserted = tuple(sorted(p + i for i, p in enumerate(sorted(positions))))
    children = draw(arbitrary_trees(len(shape) + nr_inserted)) if draw(st.booleans()) else None
    return shape, inserted, children


@st.composite
def ht_tensors(
    draw: st.DrawFn,
    tree: TreeSpec | None = None,
    orthog: bool | None = None,
    max_order: int = MAX_ORDER,
    dtype: torch.dtype | None = None,
) -> HTTensor:
    """Random float64 HT tensor, see the module docstring; orthogonal or not, as drawn unless fixed."""
    shape, inserted, children = tree if tree is not None else draw(tree_specs(max_order=max_order))
    order = len(shape) + len(inserted)
    dtree = DimTree.from_order(order) if children is None else DimTree(tuple(range(order)), children)
    nodes = {
        t: (draw(st.integers(1, MAX_RANK)), draw(st.sampled_from(KINDS)))
        for t in dtree.get_nodes()
        if not dtree.is_root(t)
    }
    spec = dict(
        shape=shape,
        inserted=inserted,
        nodes=nodes,
        seed=draw(st.integers(0, 2**31 - 1)),
        zero=draw(st.sampled_from([False] * 9 + [True])),
        scale=draw(st.sampled_from(SCALES)),
        orthog=draw(st.booleans()) if orthog is None else orthog,
        children=children,
        dtype=draw(dtypes()) if dtype is None else dtype,
    )
    note(f"build_ht(**{spec})")
    return build_ht(**spec)  # type: ignore[arg-type]


@st.composite
def ht_pairs(draw: st.DrawFn, max_order: int = MAX_ORDER) -> tuple[HTTensor, HTTensor]:
    """Two random HT tensors of the same shape, tree and dtype; ranks, kinds and flags are drawn independently.

    Mixed precisions are covered by ``tests/test_dtype_device.py`` (D2).
    """
    tree = draw(tree_specs(max_order=max_order))
    dtype = draw(dtypes())
    return draw(ht_tensors(tree=tree, dtype=dtype)), draw(ht_tensors(tree=tree, dtype=dtype))


@st.composite
def decaying_dense(draw: st.DrawFn, max_order: int = MAX_ORDER, terms: int = 8) -> torch.Tensor:
    """Dense tensor whose hierarchical singular values decay: a sum of rank-1 terms weighted by ``decay**k``."""
    shape = draw(shapes(min_order=3, max_order=max_order))
    seed = draw(st.integers(0, 2**31 - 1))
    decay = draw(st.sampled_from([0.03, 0.1, 0.3]))
    gen = torch.Generator().manual_seed(seed)
    T = torch.zeros(shape, dtype=torch.float64)
    for k in range(terms):
        term = torch.randn(shape[0], generator=gen, dtype=torch.float64)
        for n in shape[1:]:
            term = torch.tensordot(term, torch.randn(n, generator=gen, dtype=torch.float64), dims=0)
        T += decay**k * term
    return T.to(draw(dtypes()))
