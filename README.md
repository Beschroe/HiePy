# HiePy

**hiepy** brings the Hierarchical Tucker (HT) format to PyTorch. An HT tensor stores a high-dimensional tensor as
small matrices and 3-way tensors on a binary tree over its modes. For tensors with low hierarchical ranks, storage
and computation grow linearly instead of exponentially with the number of modes.

**Documentation:** <https://beschroe.github.io/HiePy/>

> Status: alpha. The API may still change; see the [changelog](https://github.com/Beschroe/HiePy/blob/main/CHANGELOG.md).

## Features

- **Construction** from dense tensors with a prescribed accuracy, or directly with factories for tensors that do
  not fit into memory.
- **Arithmetic**: sums, scaling, multiplication along a mode, elementwise and inner products — exact, without
  forming dense tensors.
- **Truncation** with a guaranteed error bound `‖x − x̃‖ ≤ atol + rtol·‖x‖`, and orthogonalization.
- **Indexing** like torch with integers and slices.
- **Autograd**: gradients flow through all operations, as in torch.

## Installation

Requires Python 3.12 and PyTorch 2.6 or newer.

```bash
pip install hiepy
```

## Quickstart

```python
import torch
import hiepy

torch.set_default_dtype(torch.float64)

g = torch.linspace(0, 1, 10)
T = torch.sin(sum(torch.meshgrid(*[g] * 5, indexing="ij")))  # 10^5 entries
x = hiepy.to_htt(T, {"rtol": 1e-10})                         # ranks 2, 1 KB

y, _ = (x * x + x).truncate({"rtol": 1e-10})                 # computed in the HT format
print(torch.linalg.norm(y.full() - (T * T + T)) / torch.linalg.norm(T * T + T))  # < 1e-9
```

Next: [Getting started](https://beschroe.github.io/HiePy/getting-started/),
[User Guide](https://beschroe.github.io/HiePy/user-guide/ht-format/),
[API Reference](https://beschroe.github.io/HiePy/api/).

## Reporting bugs

Please report bugs as an [issue on GitHub](https://github.com/Beschroe/HiePy/issues/new/choose), with a small
example that reproduces the problem.

## License

[EUPL-1.2](https://github.com/Beschroe/HiePy/blob/main/LICENSE)
