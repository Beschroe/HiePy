# Operations

Every operation is a function in the `hiepy` namespace. Most of them are also available as a method or operator of
`HTTensor`; `x.orthog()` and `hiepy.orthog(x)` are the same. The *Notes* of each function in the
[API Reference](../api/index.md) state how it changes ranks, tree and orthogonality, and what it costs.

## Overview

| Purpose | Functions | Method / operator | Ranks of the result |
|---|---|---|---|
| Construct | `zeros`, `ones`, `constant`, `arange`, `randn`, `hilbert` | | small, see each factory |
| Convert from dense | `to_htt` | | as needed for the tolerance |
| Convert to dense | `full` | `x.full()` | |
| Inspect | `order`, `shape`, `rank`, `nbytes`, `dtype` | `x.order`, `x.shape`, … | |
| Index | `get` | `x[...]`, `x.get(...)` | unchanged or smaller |
| Add, subtract | `add`, `sub` | `x + y`, `x - y` | sum of the ranks |
| Scale | `scalar_mul` | `c * x` | unchanged |
| Multiply along a mode | `mode_mul`, `ele_mode_mul` | `x.mode_mul(A, dim)` | unchanged |
| Elementwise product | `ele_mul` | `x * y` | product of the ranks, unless truncated |
| Inner product, norm | `contract`, `norm` | `x @ y`, `x.norm()` | (scalar) |
| Orthogonalize | `orthog` | `x.orthog()` | unchanged or smaller |
| Truncate | `truncate`, `truncate_svd`, `truncate_gram` | `x.truncate(opts)` | as needed for the tolerance |
| Remove singleton modes | `squeeze` | `x.squeeze()` | unchanged |
| Copy | `copy`, `clone`, `detach`, `to` | `x.copy()`, … | unchanged |
| Save, load | `save`, `load` | `x.save(path)`, `HTTensor.load(path)` | |

`mode_mul(x, A, dim)` multiplies mode `dim` by a matrix `A`; `ele_mode_mul(x, v, dim)` scales mode `dim` entrywise
by a vector `v`:

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> x = hiepy.randn((4, 5, 6), rank=3, generator=torch.Generator().manual_seed(0))
>>> y = hiepy.randn((4, 5, 6), rank=2, generator=torch.Generator().manual_seed(1))
>>> x.mode_mul(torch.ones(2, 5), dim=1).shape
(4, 2, 6)
>>> x.ele_mode_mul(torch.arange(6.0), dim=2).shape
(4, 5, 6)

```

## Results are new tensors

Operations never modify their inputs; each returns a new tensor:

```python
>>> before = x.U[(0,)].clone()
>>> z = x + y
>>> torch.equal(x.U[(0,)], before)
True

```

Operations on two tensors require the same shape and the same dimension tree:

```python
>>> x + hiepy.randn((4, 5, 7), rank=2)
Traceback (most recent call last):
    ...
ValueError: Shape of 'y' (4, 5, 7) does not match shape of 'x' (4, 5, 6)

```

## Orthogonality

An HT tensor is *orthogonal* if the frames of all nodes except the root have orthonormal columns. `norm`,
truncation and the Gramians rely on this property; they orthogonalize their input first if necessary. The flag
`x.is_orthog` records it, so that this step can be skipped:

```python
>>> x.is_orthog, (x + y).is_orthog, (x + y).orthog().is_orthog
(True, False, True)

```

`randn`, `to_htt` and `orthog` return orthogonal tensors; scaling, copies, indexing that selects every entry and
`squeeze` without a mode to remove keep the flag; all other operations return `False`. The flag is never `True` for
a tensor that is not orthogonal.

## Indexing

`x[...]` follows torch for integers and slices with positive steps. An integer removes its mode, a slice keeps it.
If fewer than two modes remain, the result is a dense `torch.Tensor`, because HT tensors have order at least 2:

```python
>>> x[1:3, ::2].shape
(2, 3, 6)
>>> x[-1].shape
(5, 6)
>>> x[0, 1].shape, x[0, 1, 2].shape
(torch.Size([6]), torch.Size([]))

```

Boolean masks, `None`, `Ellipsis`, negative steps and empty slices are not supported:

```python
>>> x[2:1]
Traceback (most recent call last):
    ...
ValueError: Slice slice(2, 1, None) selects no entry of dimension 0 with size 4; HT tensors have no empty modes.

```

Removing a mode, by an integer index or by `squeeze`, restructures the dimension tree.
