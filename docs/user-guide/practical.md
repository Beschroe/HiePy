# Practical notes

## Precision

All components of an HT tensor share one dtype. Factories use `torch.get_default_dtype()` unless `dtype` is given,
`to` converts, and operations on two tensors of different precision return the more precise one:

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> x = hiepy.randn((4, 5, 6), rank=3, generator=torch.Generator().manual_seed(0))
>>> hiepy.ones((2, 3), dtype=torch.float32).dtype
torch.float32
>>> x32 = x.to(dtype=torch.float32)
>>> x32.dtype, (x + x32).dtype
(torch.float32, torch.float64)

```

## Gradients

Gradients work as in torch: an operation records a graph if a component of an input requires gradients and
autograd is enabled. `detach` and `copy` cut the graph, `torch.no_grad()` disables it.

```python
>>> w = hiepy.randn((4, 5, 6), rank=3, generator=torch.Generator().manual_seed(1)).requires_grad_()
>>> loss = (w @ w - 1.0) ** 2
>>> loss.backward()
>>> all(c.grad is not None for c in [*w.U.values(), *w.B.values()])
True
>>> (2.0 * w).requires_grad, w.detach().requires_grad
(True, False)
>>> with torch.no_grad():
...     (2.0 * w).requires_grad
False

```

Operations based on singular value or eigenvalue decompositions (`to_htt`, truncation, `local_left_svd`) have
unstable gradients when singular values (nearly) coincide, as in torch.

## Saving and loading

`save` and `load` store a tensor with its tree and flags in a file with the suffix `.htt`:

```python
>>> import tempfile
>>> from pathlib import Path
>>> with tempfile.TemporaryDirectory() as directory:
...     path = Path(directory) / "x.htt"
...     x.save(path)
...     y = hiepy.load(path)
>>> torch.equal(y.full(), x.full())
True

```

## Limits

- HT tensors have order at least 2, and every mode has at least one entry. Results with fewer than two modes are
  returned as dense tensors.
- Norms beyond about $10^{154}$ (in float64) overflow when squared; truncation then raises a `ValueError` instead of
  returning a meaningless result.

```python
>>> hiepy.ones((3, 0, 2))
Traceback (most recent call last):
    ...
ValueError: All mode sizes must be at least 1, got shape (3, 0, 2) (modes [1]).

```
