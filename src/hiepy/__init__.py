"""
hiepy - Hierarchical Tucker (HT) tensor library in PyTorch.

Overview
--------
hiepy provides efficient representations and operations for tensors in the
Hierarchical Tucker format, including algebraic operations, truncation,
and conversions from dense PyTorch tensors.

Main Component
---------------
HTTensor
    Hierarchical Tucker tensor representation.

Key Features
-----------
- Efficient tensor arithmetic (addition, contraction, etc.)
- Rank truncation and orthogonalization utilities
- Conversions from dense PyTorch tensors
- Random tensor generation in HT format
"""

from .algebra.add import add
from .algebra.contract import contract
from .algebra.ele_mode_mul import ele_mode_mul
from .algebra.ele_mul import ele_mul
from .algebra.mode_mul import mode_mul
from .algebra.scalar_mul import scalar_mul
from .algebra.sub import sub
from .factory.arange import arange
from .factory.constant import constant
from .factory.copy import clone, copy, detach, to
from .factory.hilbert import hilbert
from .factory.ones import ones
from .factory.randn import randn
from .factory.to_htt import to_htt
from .factory.zeros import zeros
from .httensor.full import full
from .httensor.get_item import get
from .httensor.getter import device, dtype, nbytes, order, rank, shape
from .httensor.gramians import gramians
from .httensor.httensor import HTTensor
from .httensor.io import load, save
from .linalg.left_svd_gramian import left_svd_gramian
from .linalg.local_left_svd import local_left_svd
from .linalg.matricize import dematricize, matricize
from .linalg.mode_mul_torch import mode_mul_torch, multi_mode_mul_torch
from .linalg.norm import norm
from .linalg.vectorize import devectorize, vectorize
from .transformation.orthog import orthog
from .transformation.squeeze import squeeze
from .transformation.truncate import truncate, truncate_gram, truncate_svd
from .utils.data_types import ScalarType, Shape, TensorDict, TruncOpts
from .utils.dimtree import DimTree, Node
from .utils.trunc_rank import trunc_rank

__all__ = [
    "nbytes",
    "dtype",
    "device",
    "order",
    "rank",
    "shape",
    "gramians",
    "HTTensor",
    "full",
    "get",
    "load",
    "save",
    "truncate",
    "truncate_gram",
    "truncate_svd",
    "orthog",
    "squeeze",
    "left_svd_gramian",
    "local_left_svd",
    "norm",
    "TruncOpts",
    "ScalarType",
    "Shape",
    "TensorDict",
    "trunc_rank",
    "DimTree",
    "Node",
    "matricize",
    "dematricize",
    "vectorize",
    "devectorize",
    "add",
    "sub",
    "mode_mul_torch",
    "multi_mode_mul_torch",
    "contract",
    "ele_mul",
    "ele_mode_mul",
    "mode_mul",
    "scalar_mul",
    "zeros",
    "hilbert",
    "ones",
    "randn",
    "arange",
    "constant",
    "to_htt",
    "clone",
    "copy",
    "detach",
    "to",
]
