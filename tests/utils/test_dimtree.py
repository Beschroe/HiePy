"""Tests for ``DimTree``: construction, navigation, traversal, mutation and printing.

Properties are checked on arbitrary binary trees (``tests.strategies.arbitrary_trees``) and on the canonical tree.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hiepy import DimTree
from tests.strategies import arbitrary_trees


@st.composite
def trees(draw: st.DrawFn, max_order: int = 7) -> DimTree:
    """The canonical tree or an arbitrary binary tree of order 1 … ``max_order``."""
    order = draw(st.integers(1, max_order))
    if draw(st.booleans()):
        return DimTree.from_order(order)
    return DimTree(tuple(range(order)), draw(arbitrary_trees(order)))


def _depth(tree: DimTree, node: tuple[int, ...]) -> int:
    depth = 0
    while not tree.is_root(node):
        node = tree.get_parent(node)
        depth += 1
    return depth


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Construction ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(order=st.integers(1, 12))
def test_from_order_is_balanced_and_contiguous(order: int):
    """``from_order`` splits every node into contiguous halves, the left one of size ``ceil(n/2)``."""
    tree = DimTree.from_order(order)
    assert tree.get_root() == tuple(range(order))
    assert sorted(tree.get_leaves()) == [(i,) for i in range(order)]
    for t in tree.get_inner_nodes():
        left, right = tree.get_children(t)
        assert left + right == t
        assert len(left) == math.ceil(len(t) / 2)
    assert tree.get_depth() == math.ceil(math.log2(order))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Navigation ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(tree=trees())
def test_navigation_is_consistent(tree: DimTree):
    """Children partition their parent; parent, sibling and left/right agree with the child mapping."""
    root = tree.get_root()
    assert tree.is_root(root)
    for t in tree.get_inner_nodes():
        assert tree.is_inner(t) and not tree.is_leaf(t)
        left, right = tree.get_children(t)
        assert set(left).isdisjoint(right) and set(left) | set(right) == set(t)
        assert tree.get_parent(left) == t and tree.get_parent(right) == t
        assert tree.get_sibling(left) == right and tree.get_sibling(right) == left
        assert tree.is_left(left) and not tree.is_right(left)
        assert tree.is_right(right) and not tree.is_left(right)
    for leaf in tree.get_leaves():
        assert tree.is_leaf(leaf) and not tree.is_inner(leaf) and len(leaf) == 1


@pytest.mark.parametrize("method", ["is_left", "is_right"])
def test_left_right_of_root_raise(method: str):
    tree = DimTree.from_order(3)
    with pytest.raises(ValueError, match="root"):
        getattr(tree, method)(tree.get_root())


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Traversal ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(tree=trees())
def test_traversals_visit_every_node_once_in_order(tree: DimTree):
    """``get_nodes`` lists parents before children (top-down) or after them (bottom-up); ``get_levels`` groups by
    depth; leaves and inner nodes partition the nodes; ``get_depth`` is the largest leaf depth."""
    top_down = tree.get_nodes(order="top-down")
    bottom_up = tree.get_nodes(order="bottom-up")
    assert len(set(top_down)) == len(top_down) and set(top_down) == set(bottom_up)
    assert set(tree.get_leaves()) | set(tree.get_inner_nodes()) == set(top_down)
    assert set(tree.get_leaves()).isdisjoint(tree.get_inner_nodes())
    assert len(top_down) == 2 * len(tree.get_leaves()) - 1
    for t in tree.get_inner_nodes():
        for child in tree.get_children(t):
            assert top_down.index(t) < top_down.index(child)
            assert bottom_up.index(child) < bottom_up.index(t)
    levels = tree.get_levels(order="top-down")
    assert all(_depth(tree, t) == k for k, level in enumerate(levels) for t in level)
    assert tree.get_levels(order="bottom-up") == levels[::-1]
    assert tree.get_depth() == len(levels) - 1 == max(_depth(tree, leaf) for leaf in tree.get_leaves())


@pytest.mark.parametrize("method", ["get_nodes", "get_levels"])
def test_invalid_traversal_order_raises(method: str):
    with pytest.raises(ValueError, match="order"):
        getattr(DimTree.from_order(3), method)(order="sideways")


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Equality and copies ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(tree=trees())
def test_copy_is_equal_and_independent(tree: DimTree):
    """A copy equals the tree, and mutating the copy leaves the original unchanged."""
    copy = tree.copy()
    assert copy == tree and copy.is_equal(tree) and copy is not tree
    copy.prune(copy.get_root())
    assert copy.is_leaf(copy.get_root())
    assert tree.is_inner(tree.get_root()) or len(tree.get_root()) == 1


def test_equality_with_other_types_and_trees():
    tree = DimTree.from_order(4)
    assert tree != "not a tree"
    assert tree != DimTree.from_order(5)
    assert tree != DimTree((0, 1, 2, 3), {(0, 1, 2, 3): ((0, 2), (1, 3)), (0, 2): ((0,), (2,)), (1, 3): ((1,), (3,))})


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Mutation ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(tree=trees(), data=st.data())
def test_prune_removes_all_descendants(tree: DimTree, data: st.DataObject):
    """``prune`` turns a node into a leaf and removes its descendants; on a leaf it does nothing."""
    node = data.draw(st.sampled_from(tree.get_nodes()), label="node")
    descendants = set(DimTree(node, tree.copy().children).get_nodes()) - {node} if tree.is_inner(node) else set()
    before = set(tree.get_nodes())
    tree.prune(node)
    assert tree.is_leaf(node)
    assert set(tree.get_nodes()) == before - descendants


def test_move_subtree_and_map_as_used_by_squeeze():
    """``move_subtree`` hands the children of a node to another node and drops what became unreachable;
    ``map`` relabels every node, including the root."""
    tree = DimTree.from_order(4)  # (0,1,2,3) -> (0,1) | (2,3)
    tree.move_subtree(subtree_root=(2, 3), new_parent=(0, 1, 2, 3))
    assert tree.children == {(0, 1, 2, 3): ((2,), (3,))}
    assert tree.get_parent((2,)) == (0, 1, 2, 3)
    tree.map(old_to_new=lambda node: (2, 3) if node == (0, 1, 2, 3) else node)
    assert tree.get_root() == (2, 3)
    assert tree.get_children((2, 3)) == ((2,), (3,))


def test_consolidate_removes_unreachable_nodes():
    tree = DimTree.from_order(4)
    tree.children[(7, 8)] = ((7,), (8,))
    tree.consolidate()
    assert tree == DimTree.from_order(4)


@given(tree=trees(), data=st.data())
def test_subtree_spanning_dims_is_minimal(tree: DimTree, data: st.DataObject):
    """``get_subtree_spanning_dims`` returns the subtree of the smallest node containing the given modes."""
    modes = data.draw(st.lists(st.sampled_from(tree.get_root()), min_size=1, unique=True), label="modes")
    sub = tree.get_subtree_spanning_dims(modes)
    root = sub.get_root()
    assert set(modes) <= set(root)
    if sub.is_inner(root):
        assert not any(set(modes) <= set(child) for child in sub.get_children(root))
    assert set(sub.get_nodes()) == set(DimTree(root, tree.children).get_nodes())


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Printing ~~~~~~~~~~~~~~~~~~~~~~~~~


@given(tree=trees())
def test_str_shows_every_node_level_by_level(tree: DimTree):
    """``str`` prints one line per level and each node on the line of its level."""
    lines = str(tree).split("\n")
    levels = tree.get_levels()
    assert len(lines) == len(levels)
    for line, level in zip(lines, levels):
        assert all(str(node) in line for node in level)
