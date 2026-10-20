"""
Dimension tree implementation for HT tensors.

This module defines the DimTree structure used to describe hierarchical
splittings of tensor modes in HT tensor representations.
"""

from __future__ import annotations

from collections import deque
from math import ceil
from typing import Callable, Iterable

type Node = tuple[int, ...]
"""A tuple of dimensions represented by a node of a dimension tree (e.g., `(0,1)`)."""


class DimTree:
    r"""
    Binary dimension tree used in Hierarchical Tucker decompositions.

    A DimTree represents a recursive partitioning of tensor dimensions.
    Each node corresponds to a subset of modes, and each inner node splits
    into two disjoint child nodes.

    Attributes
    ----------
    root : Node
        Root node containing all dimensions.
    children : dict[Node, tuple[Node, Node]]
        Mapping from inner nodes to (left_child, right_child).
    parent : dict[Node, Node]
        Mapping from non-root nodes to their parent node.

    Notes
    -----
    - The tree is always binary.
    - Leaves correspond to singleton dimension tuples (i,). However, during certain procedures a leaf may temporarily
      contain several modes.
    - Inner nodes correspond to a group of dimensions.
    """

    def __init__(self, root: Node, children: dict[Node, tuple[Node, Node]]):
        """Initialize a dimension tree from a root and child mapping."""
        self.root = root
        self.children = children
        self.parent = DimTree._get_parent_from_children(self.children)

    @staticmethod
    def _get_parent_from_children(children: dict[Node, tuple[Node, Node]]) -> dict[Node, Node]:
        """Construct parent mapping from a child dictionary."""
        parent = {}
        for father, (left, right) in children.items():
            parent[left] = father
            parent[right] = father
        return parent

    @classmethod
    def from_order(cls, order: int) -> DimTree:
        """Create a canonical balanced dimension tree of given order."""
        root = tuple(range(order))
        children = {}
        queue = deque([root])
        while queue:
            node = queue.pop()
            if len(node) > 1:
                mid_idx = ceil(len(node) / 2)
                left = node[:mid_idx]
                right = node[mid_idx:]
                children[node] = left, right
                queue.append(left)
                queue.append(right)
        return DimTree(root=root, children=children)

    def __str__(self) -> str:
        """Return a level-wise ASCII representation of the tree."""

        levels = self.get_levels(order="top-down")
        leaf_cell_width = max([len(str(leaf)) for leaf in self.get_leaves()]) + 1
        leaf_cells = 2 ** self.get_depth()
        total_width = leaf_cells * leaf_cell_width
        width_per_lvl = [int(total_width / 2**i) for i in range(len(levels))]
        mat = [[" " * width_per_lvl[i]] * 2**i for i in range(len(levels))]
        indices = self._print_idx_util(self.get_root(), idx=0)
        for i in range(len(levels)):
            for node in levels[i]:
                col_idx = indices[node]
                mat[i][col_idx] = str(node).center(width_per_lvl[i])
        return "\n".join("".join(row) for row in mat)

    def __eq__(self, other: object) -> bool:
        """Check structural equality of two dimension trees."""
        if isinstance(other, DimTree):
            return self.is_equal(other)
        else:
            return False

    def get_children(self, node: Node) -> tuple[Node, Node]:
        """Return left and right children of a node."""
        return self.children[node]

    def get_parent(self, node: Node) -> Node:
        """Return the parent of a node."""
        return self.parent[node]

    def get_sibling(self, node: Node) -> Node:
        """Return the sibling of a node."""
        parent = self.get_parent(node)
        left, right = self.get_children(parent)
        return right if node == left else left

    def get_root(self) -> Node:
        """Return the root node."""
        return self.root

    def map(self, old_to_new: Callable[[Node], Node]) -> None:
        """Apply a relabeling function to all nodes."""
        # Update children
        self.children = {old_to_new(k): (old_to_new(v1), old_to_new(v2)) for k, (v1, v2) in self.children.items()}

        # Update parent
        self.parent = {old_to_new(k): old_to_new(v) for k, v in self.parent.items()}

        # Update root
        self.root = old_to_new(self.root)

    def move_subtree(self, subtree_root: Node, new_parent: Node) -> None:
        """Move a subtree to a new parent node."""
        # Move subtree to `new_parent`
        left, right = self.get_children(subtree_root)
        self.children[new_parent] = left, right
        self.parent[left] = new_parent
        self.parent[right] = new_parent

        # Mark `subtree_root` as leaf
        del self.children[subtree_root]

        # Consolidate tree
        self.consolidate()

    def prune(self, node: Node) -> None:
        """Remove all descendants of a node."""
        next_nodes = [node] if self.is_inner(node) else []
        while next_nodes:
            node = next_nodes.pop()
            left, right = self.get_children(node)
            if self.is_inner(left):
                next_nodes.append(left)
            if self.is_inner(right):
                next_nodes.append(right)
            del self.children[node]
            del self.parent[left]
            del self.parent[right]

    def consolidate(self) -> None:
        """Remove unreachable nodes and rebuild connectivity."""
        # Init new children and parent hierarchy
        new_children = {}
        new_parent = {}

        # Travers the tree from the root to find all reachable nodes
        next_nodes = deque([self.get_root()])
        nodes = []
        while next_nodes:
            node = next_nodes.popleft()
            nodes.append(node)

            if self.is_inner(node):
                left, right = self.get_children(node)

                # Update new children and parent hierarchy
                new_children[node] = left, right
                new_parent[left] = node
                new_parent[right] = node

                next_nodes.append(left)
                next_nodes.append(right)

        self.children = new_children
        self.parent = new_parent

    def is_left(self, node: Node) -> bool:
        """Check if node is a left child."""
        if self.root == node:
            raise ValueError("Argument `node` must not be the root node.")
        parent = self.get_parent(node)
        left, _ = self.get_children(parent)
        return node == left

    def is_right(self, node: Node) -> bool:
        """Check if node is a right child."""
        if self.root == node:
            raise ValueError("Argument `node` must not be the root node.")
        parent = self.get_parent(node)
        _, right = self.get_children(parent)
        return node == right

    def is_inner(self, node: Node) -> bool:
        """Check if node is an inner node."""
        return node in self.children

    def is_leaf(self, node: Node) -> bool:
        """Check if node is a leaf."""
        return node not in self.children

    def is_root(self, node: Node) -> bool:
        """Check if node is the root."""
        return self.get_root() == node

    def get_nodes(self, order: str = "top-down") -> list[Node]:
        """Return all nodes in traversal order."""
        if order not in ["top-down", "bottom-up"]:
            raise ValueError(f"Argument `order` must either be 'top-down' or 'bottom-up', got {order}.")
        next_nodes = deque([self.get_root()])
        nodes = []
        while next_nodes:
            node = next_nodes.popleft()
            nodes.append(node)

            if self.is_inner(node):
                left, right = self.get_children(node)
                next_nodes.append(left)
                next_nodes.append(right)

        return nodes if order == "top-down" else nodes[::-1]

    def get_leaves(self) -> list[Node]:
        """Return all leaf nodes."""
        next_nodes = [self.get_root()]
        leaves = []
        while next_nodes:
            node = next_nodes.pop()
            if self.is_inner(node):
                left, right = self.get_children(node)
                next_nodes.append(right)
                next_nodes.append(left)
            else:
                leaves.append(node)

        return leaves

    def get_inner_nodes(self) -> list[Node]:
        """Return all inner nodes."""
        next_nodes = deque([self.get_root()])
        inner_nodes = []
        while next_nodes:
            node = next_nodes.popleft()

            if self.is_inner(node):
                left, right = self.get_children(node)
                next_nodes.append(left)
                next_nodes.append(right)
                inner_nodes.append(node)

        return inner_nodes

    def get_levels(self, order: str = "top-down") -> list[list[Node]]:
        """Return nodes grouped by depth level."""
        if order not in ["top-down", "bottom-up"]:
            raise ValueError(f"Argument `order` must either be 'top-down' or 'bottom-up', got {order}.")
        lvl = [self.get_root()]
        levels = [lvl]
        while lvl:
            new_lvl = []
            for node in lvl:
                if self.is_inner(node):
                    left, right = self.get_children(node)
                    new_lvl.append(left)
                    new_lvl.append(right)
            lvl = new_lvl
            if new_lvl:
                levels.append(lvl)
        return levels if order == "top-down" else levels[::-1]

    def get_depth(self) -> int:
        """Return maximum depth of the tree."""
        next_nodes = deque([(self.get_root(), 0)])
        depths = []
        while next_nodes:
            node, d = next_nodes.popleft()

            if self.is_inner(node):
                left, right = self.get_children(node)
                next_nodes.append((left, d + 1))
                next_nodes.append((right, d + 1))
            else:
                depths.append(d)

        return max(depths)

    def copy(self) -> DimTree:
        """Return a shallow copy of the tree."""
        return DimTree(self.get_root(), {par: (left, right) for par, (left, right) in self.children.items()})

    def get_subtree_spanning_dims(self, dims: Iterable[int]) -> DimTree:
        """Return minimal subtree covering given dimensions."""
        target_dims = set(dims)

        # Detect new root
        node = self.get_root()
        while self.is_inner(node):
            left, right = self.get_children(node)
            if target_dims.issubset(left):
                node = left
            elif target_dims.issubset(right):
                node = right
            else:
                break

        new_root = node
        new_children = {}
        next_nodes = [new_root]
        while next_nodes:
            node = next_nodes.pop()
            if self.is_inner(node):
                left, right = self.get_children(node)

                new_children[node] = left, right

                next_nodes.append(left)
                next_nodes.append(right)

        return DimTree(root=new_root, children=new_children)

    def is_equal(self, y: DimTree) -> bool:
        """Check structural equality."""
        return self.children == y.children and self.get_root() == y.get_root()

    def _print_idx_util(self, node: Node, idx: int = 0) -> dict[Node, int]:
        """Compute leaf column indices for ASCII printing."""
        if self.is_leaf(node):
            return {node: idx}
        else:
            left, right = self.get_children(node)
            return {node: idx} | self._print_idx_util(left, idx * 2) | self._print_idx_util(right, idx * 2 + 1)
