# hiepy

**hiepy** brings the Hierarchical Tucker (HT) format to PyTorch. An HT tensor stores a high-dimensional tensor as
small matrices and 3-way tensors on a binary tree over its modes. For tensors with low hierarchical ranks, storage
and computation grow linearly instead of exponentially with the number of modes.

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> g = torch.linspace(0, 1, 10)
>>> T = torch.sin(sum(torch.meshgrid(*[g] * 5, indexing="ij")))  # 10^5 entries
>>> x = hiepy.to_htt(T, {"rtol": 1e-10})                         # ranks 2, 1 KB
>>> y, _ = (x * x + x).truncate({"rtol": 1e-10})                 # computed in the HT format
>>> bool(torch.linalg.norm(y.full() - (T * T + T)) <= 1e-9 * torch.linalg.norm(T * T + T))
True

```

## Features

- **Construction**: from dense tensors with a prescribed accuracy (`to_htt`), or directly with factories
  (`zeros`, `ones`, `constant`, `arange`, `randn`, `hilbert`) for tensors that do not fit into memory.
- **Arithmetic**: sums, scaling, multiplication along a mode, elementwise products and inner products, exact and
  without forming dense tensors.
- **Truncation** with a guaranteed error bound `‖x − x̃‖ ≤ atol + rtol·‖x‖`, and orthogonalization.
- **Indexing** like torch with integers and slices.
- **Autograd**: gradients flow through all operations, as in torch.

## Where to go next

- [Getting started](getting-started.md): installation and a first example.
- User Guide: [the HT format](user-guide/ht-format.md), [operations](user-guide/operations.md),
  [truncation](user-guide/truncation.md), [practical notes](user-guide/practical.md).
- [API Reference](api/index.md): every function with parameters, guarantees and cost.
