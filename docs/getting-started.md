# Getting started

## Installation

hiepy requires Python 3.12 and PyTorch 2.6 or newer. Install it from PyPI:

```bash
pip install hiepy
```

## A first HT tensor

The function $f(i_0, \dots, i_4) = \sin(g_{i_0} + \dots + g_{i_4})$ on a grid of 10 points per mode is a dense tensor
with $10^5$ entries. In the Hierarchical Tucker (HT) format it needs only a few small matrices, because all its
hierarchical ranks are 2. `to_htt` computes this representation:

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> g = torch.linspace(0, 1, 10)
>>> T = torch.sin(sum(torch.meshgrid(*[g] * 5, indexing="ij")))
>>> x = hiepy.to_htt(T, {"rtol": 1e-10})
>>> x.shape
(10, 10, 10, 10, 10)
>>> max(x.rank.values())
2
>>> x.nbytes, T.nbytes
(1024, 800000)

```

`{"rtol": 1e-10}` asks for a relative error of at most $10^{-10}$, and hiepy guarantees it:

```python
>>> bool(torch.linalg.norm(x.full() - T) <= 1e-10 * torch.linalg.norm(T))
True

```

`x.full()` converts back to a dense tensor. It needs the full memory again, so use it only for small tensors.

## Computing in the HT format

Arithmetic works directly on HT tensors and is exact. Sums and elementwise products increase the ranks:

```python
>>> s = x + x
>>> max(s.rank.values())
4
>>> p = x * x
>>> max(p.rank.values())
4

```

The ranks of the result are an upper bound. `truncate` reduces them to what the requested accuracy needs: $2x$ has
rank 2 again, and $\sin^2 = (1 - \cos 2\,\cdot)/2$ has rank 3.

```python
>>> s, _ = s.truncate({"rtol": 1e-10})
>>> max(s.rank.values())
2
>>> p, _ = p.truncate({"rtol": 1e-10})
>>> max(p.rank.values())
3
>>> bool(torch.linalg.norm(p.full() - T * T) <= 1e-10 * torch.linalg.norm(T * T))
True

```

Inner products and norms never form the dense tensor:

```python
>>> bool(torch.isclose(x @ x, (T * T).sum()))
True
>>> bool(torch.isclose(x.norm(), torch.linalg.norm(T)))
True

```

## Indexing

Indexing follows torch for integers and slices. Selecting single entries or lines returns dense tensors; results
with at least two modes stay in the HT format:

```python
>>> bool(torch.isclose(x[1, 2, 3, 4, 5], T[1, 2, 3, 4, 5]))
True
>>> x[0, :, 3].shape
(10, 10, 10)
>>> x[0, 1, 2, 3].shape
torch.Size([10])

```

## Tensors that do not fit into memory

Factories build HT tensors directly, without a dense tensor. The Hilbert tensor
$h(i_0, \dots, i_7) = 1 / (i_0 + \dots + i_7 + 8)$ with $20^8$ entries would need about 205 GB as a dense tensor.
`hilbert` approximates it in the HT format; truncation removes the redundancy of this construction:

```python
>>> h = hiepy.hilbert((20,) * 8, M=20)
>>> h, _ = h.truncate({"rtol": 1e-8})
>>> max(h.rank.values()), h.nbytes
(8, 30208)
>>> bool(abs(h[0, 1, 2, 3, 4, 5, 6, 7] - 1 / 36) <= 1e-4 / 36)
True

```

## Next steps

- [User Guide](user-guide/ht-format.md): the HT format, operations, truncation and practical notes.
- [API Reference](api/index.md): all functions with their parameters, guarantees and cost.
