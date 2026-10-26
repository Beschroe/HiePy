# Truncation

Exact operations such as sums and elementwise products increase the ranks. Truncation reduces them again, with an
error you control.

## Options

All truncating functions take a dictionary of options (`TruncOpts`):

| Key | Meaning |
|---|---|
| `atol` | absolute error bound in the Frobenius norm |
| `rtol` | relative error bound, relative to `‖x‖` |
| `max_rank` | upper bound for every rank |

Given `atol` and/or `rtol` (a missing key counts as 0), the result $\tilde x$ satisfies

$$
\lVert x - \tilde x \rVert \le \mathrm{atol} + \mathrm{rtol} \cdot \lVert x \rVert .
$$

`max_rank` is enforced strictly and may override the tolerance. Unknown keys, negative or non-finite tolerances and
`max_rank < 1` raise an error.

## Example

A rank-2 tensor plus a small perturbation of rank 3 has rank 5. Truncation finds the accuracy–rank trade-off:

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> g = torch.linspace(0, 1, 10)
>>> T = torch.sin(sum(torch.meshgrid(*[g] * 5, indexing="ij")))
>>> x = hiepy.to_htt(T, {"rtol": 1e-10})
>>> noise = hiepy.randn((10,) * 5, rank=3, generator=torch.Generator().manual_seed(0))
>>> y = x + 1e-4 * noise
>>> max(y.rank.values())
5
>>> z, discarded = y.truncate({"rtol": 1e-2})
>>> max(z.rank.values())
2
>>> error = torch.linalg.norm(y.full() - z.full())
>>> bool(error <= 1e-2 * y.norm())
True

```

A tighter tolerance keeps more of the perturbation:

```python
>>> max(y.truncate({"rtol": 1e-6})[0].rank.values())
4
>>> max(y.truncate({"rtol": 1e-6, "max_rank": 3})[0].rank.values())
3

```

`truncate` also returns the discarded singular values per node. They bound the error from above:

```python
>>> bound = torch.sqrt(sum((s**2).sum() for s in discarded.values()))
>>> bool(error <= bound)
True

```

## Methods

`truncate(x, opts, method="svd")` dispatches to

- `truncate_svd` (default): orthogonalizes and computes the singular values of the nodes from the root to the
  leaves. Accurate to machine precision.
- `truncate_gram`: computes them as square roots of the eigenvalues of Gram matrices. Cheaper in some cases, but
  singular values below about $\sqrt{\varepsilon}\,\lVert x \rVert$ ($10^{-8}\lVert x \rVert$ in float64) cannot be
  resolved, so tolerances below that are not meaningful.

## Where truncation happens

- `to_htt(T, opts)` converts a dense tensor and truncates on the way.
- `ele_mul(x, y, opts)` truncates while forming the elementwise product. It supports `atol` and `max_rank`
  (`rtol` is ignored with a warning). Its error bound is not sharp.
- `truncate` on any HT tensor, typically after sums or products.
