"""HTTensor core module.

This module implements the central HTTensor class representing tensors in the
Hierarchical Tucker (HT) format. It provides the core interface for construction,
structural manipulation, orthogonalization, truncation, and algebraic operations.
"""

from __future__ import annotations

from pathlib import Path

import torch

from hiepy.utils.data_types import IndexKey, Node, ScalarType, Shape, TensorDict, TruncOpts
from hiepy.utils.dimtree import DimTree


class HTTensor:
    r"""
    Hierarchical Tucker tensor representation.

    This is the central data structure of the hiepy library. A HTTensor stores a tensor in Hierarchical
    Tucker (HT) format using

    - leaf matrices ``U_t`` associated with the leaves of a dimension tree,

    - transfer tensors ``B_t`` associated with its inner nodes, and

    - a dimension tree describing the hierarchical tensor decomposition.

    Parameters
    ----------
    U : TensorDict
        A dictionary mapping leaf nodes to their respective leaf matrices (PyTorch tensors).
    B : TensorDict
        A dictionary mapping internal nodes and the root to their respective transfer tensors (PyTorch tensors).
    dtree : DimTree
        The hierarchical dimension tree governing the underlying topological decomposition of the tensor spaces.
    is_orthog : bool, default=False
        Flag indicating whether the tensor is currently in an orthogonalized state
        (i.e., all frame matrices have orthonormal columns).

    Attributes
    ----------
    U : TensorDict
        Leaf matrices of the HT tensor.
    B : TensorDict
        Transfer tensors of the HT tensor.
    dtree : DimTree
        The dimension tree.
    is_orthog : bool
        True if the internal components satisfy hierarchical orthogonality conditions.

    Notes
    -----
    - The memory requirement is given as $\mathcal{O}(dnr + dr^3)$, where $d$ is the tensor order, $n$ the maximum
      mode size, and $r$ the maximum hierarchical rank.
    """

    def __init__(
        self,
        U: TensorDict,
        B: TensorDict,
        dtree: DimTree,
        is_orthog: bool = False,
    ):
        self.U = U
        self.B = B
        self.dtree = dtree
        self.is_orthog = is_orthog

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Copy ~~~~~~~~~~~~~~~~~~~~~~~~~
    def copy(self) -> HTTensor:
        from hiepy.factory.copy import copy as _copy

        return _copy(self)

    def clone(self) -> HTTensor:
        from hiepy.factory.copy import clone as _clone

        return _clone(self)

    def detach(self) -> HTTensor:
        from hiepy.factory.copy import detach as _detach

        return _detach(self)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Getter ~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def shape(self) -> Shape:
        from hiepy.httensor.getter import shape as _shape

        return _shape(self)

    @property
    def rank(self) -> dict[Node, int]:
        from hiepy.httensor.getter import rank as _rank

        return _rank(self)

    @property
    def order(self) -> int:
        from hiepy.httensor.getter import order as _order

        return _order(self)

    @property
    def nbytes(self) -> int:
        from hiepy.httensor.getter import nbytes as _nbytes

        return _nbytes(self)

    def norm(self) -> torch.Tensor:
        from hiepy.linalg.norm import norm as _norm

        return _norm(self)

    def get(self, key: IndexKey | tuple[IndexKey, ...], track_grad: bool = False) -> HTTensor | torch.Tensor:
        from hiepy.httensor.get_item import get as _get

        return _get(self, key, track_grad)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Reshape ~~~~~~~~~~~~~~~~~~~~~~~~~
    def full(self) -> torch.Tensor:
        from hiepy.httensor.full import full as _full

        return _full(self)

    def squeeze(self, dims: list[int] | int | None = None, track_grad: bool = False) -> HTTensor | torch.Tensor:
        from hiepy.transformation.squeeze import squeeze as _squeeze

        return _squeeze(self, dims, track_grad)

    # ~~~~~~~~~~~~~ Methods that modify the internal structure ~~~~~~~~~~~~~
    def orthog(self) -> HTTensor:
        from hiepy.transformation.orthog import orthog as _orthog

        return _orthog(self)

    def truncate(self, opts: TruncOpts, method: str = "svd") -> tuple[HTTensor, dict[Node, torch.Tensor]]:
        from hiepy.transformation.truncate import truncate as _truncate

        return _truncate(x=self, opts=opts, method=method)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic methods ~~~~~~~~~~~~~~~~~~~~~~~~~
    def add(self, y: HTTensor) -> HTTensor:
        from hiepy.algebra.add import add as _add

        return _add(self, y)

    def sub(self, y: HTTensor) -> HTTensor:
        from hiepy.algebra.sub import sub as _sub

        return _sub(self, y)

    def scalar_mul(self, c: ScalarType) -> HTTensor:
        from hiepy.algebra.scalar_mul import scalar_mul as _scalar_mul

        return _scalar_mul(self, c)

    def mode_mul(self, A: torch.Tensor, dim: int) -> HTTensor:
        from hiepy.algebra.mode_mul import mode_mul as _mode_mul

        return _mode_mul(self, A, dim)

    def ele_mul(self, y: HTTensor, opts: TruncOpts = {}) -> HTTensor:
        from hiepy.algebra.ele_mul import ele_mul as _ele_mul

        return _ele_mul(self, y, opts)

    def ele_mode_mul(self, v: torch.Tensor, dim: int) -> HTTensor:
        from hiepy.algebra.ele_mode_mul import ele_mode_mul as _ele_mode_mul_htt

        return _ele_mode_mul_htt(self, v, dim)

    def contract(self, y: HTTensor) -> torch.Tensor:
        from hiepy.algebra.contract import contract as _contract

        return _contract(self, y)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Magic Methods ~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key: IndexKey | tuple[IndexKey, ...]) -> HTTensor | torch.Tensor:
        from hiepy.httensor.get_item import get as _get

        return _get(self, key)

    def __add__(self, other: HTTensor) -> HTTensor:
        return self.add(other)

    def __sub__(self, other: HTTensor) -> HTTensor:
        return self.sub(other)

    def __mul__(self, other: ScalarType | HTTensor) -> HTTensor:
        if isinstance(other, (int, float)):
            return self.scalar_mul(c=other)
        elif isinstance(other, torch.Tensor):
            if other.ndim == 0:
                return self.scalar_mul(c=other)
            else:
                raise ValueError(f"Only 0d torch tensors are supported, got other.ndim={other.ndim}.")
        elif isinstance(other, HTTensor):
            return self.ele_mul(y=other, opts={})
        else:
            raise TypeError(
                f"Supported types for `other` are int, float, 0d torch.Tensor, and HTTensor, got {type(other).__name__}"
            )

    def __rmul__(self, other: ScalarType | HTTensor) -> HTTensor:
        # Both scalar and elementwise multiplication are commutative
        return self.__mul__(other)

    def __matmul__(self, other: HTTensor) -> torch.Tensor:
        return self.contract(other)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Utility ~~~~~~~~~~~~~~~~~~~~~~~~~
    def gramians(self) -> dict[Node, torch.Tensor]:
        from hiepy.httensor.gramians import gramians as _gramians

        return _gramians(self)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ I/O ~~~~~~~~~~~~~~~~~~~~~~~~~
    def save(self, path: str | Path) -> None:
        from hiepy.httensor.io import save as _save

        _save(self, path)

    @staticmethod
    def load(path: str | Path) -> HTTensor:
        from hiepy.httensor.io import load as _load

        return _load(path)
