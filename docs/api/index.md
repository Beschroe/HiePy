# API Reference

All names are importable from `hiepy`. Each entry documents parameters, preserved properties (tree, ranks,
orthogonality), cost and examples.

## Data structures

| Name | Description |
|---|---|
| [`HTTensor`][hiepy.HTTensor] | Tensor in HT format: leaf matrices, transfer tensors, dimension tree, orthogonality flag |
| [`DimTree`][hiepy.DimTree] | Binary dimension tree over the modes; [`Node`][hiepy.Node] is a tuple of modes |

## [Construction](factory.md)

| Name | Description |
|---|---|
| [`to_htt`][hiepy.to_htt] | Dense tensor → HT tensor with prescribed accuracy |
| [`zeros`][hiepy.zeros], [`ones`][hiepy.ones], [`constant`][hiepy.constant] | Constant tensors, rank 1 |
| [`arange`][hiepy.arange] | Consecutive values, rank 2 |
| [`randn`][hiepy.randn] | Random orthogonal tensor with bounded ranks |
| [`hilbert`][hiepy.hilbert] | Approximation of the Hilbert tensor |
| [`copy`][hiepy.copy], [`clone`][hiepy.clone], [`detach`][hiepy.detach] | Copies with or without the autograd graph |
| [`to`][hiepy.to] | dtype conversion |
| [`requires_grad_`][hiepy.requires_grad_] | Switch gradient tracking on or off, in place |

## [HTTensor: inspection, indexing, I/O](httensor.md)

| Name | Description |
|---|---|
| [`order`][hiepy.order], [`shape`][hiepy.shape], [`rank`][hiepy.rank], [`nbytes`][hiepy.nbytes], [`dtype`][hiepy.dtype], [`device`][hiepy.device], [`requires_grad`][hiepy.requires_grad] | Properties |
| [`get`][hiepy.get] | Indexing with integers and slices (`x[...]`) |
| [`full`][hiepy.full] | HT tensor → dense tensor |
| [`gramians`][hiepy.gramians] | Reduced Gram matrices of an orthogonal tensor |
| [`save`][hiepy.save], [`load`][hiepy.load] | Store and restore (`.htt` files) |

## [Algebra](algebra.md)

| Name | Description |
|---|---|
| [`add`][hiepy.add], [`sub`][hiepy.sub] | Sum and difference (`+`, `-`) |
| [`scalar_mul`][hiepy.scalar_mul] | Scaling (`c * x`) |
| [`mode_mul`][hiepy.mode_mul] | Matrix times tensor along one mode |
| [`ele_mode_mul`][hiepy.ele_mode_mul] | Entrywise scaling along one mode |
| [`ele_mul`][hiepy.ele_mul] | Elementwise product (`*`), optionally truncated |
| [`contract`][hiepy.contract] | Inner product (`@`) |

## [Transformations](transformations.md)

| Name | Description |
|---|---|
| [`orthog`][hiepy.orthog] | Orthogonalization |
| [`truncate`][hiepy.truncate], [`truncate_svd`][hiepy.truncate_svd], [`truncate_gram`][hiepy.truncate_gram] | Rank truncation with error bound |
| [`squeeze`][hiepy.squeeze] | Remove singleton modes |

## [Linear algebra](linalg.md)

| Name | Description |
|---|---|
| [`norm`][hiepy.norm] | Frobenius norm |
| [`local_left_svd`][hiepy.local_left_svd] | Singular values and vectors of all node matricizations |
| [`left_svd_gramian`][hiepy.left_svd_gramian] | Singular values and vectors from a Gram matrix |
| [`matricize`][hiepy.matricize], [`dematricize`][hiepy.dematricize] | Unfold a dense tensor into a matrix and back |
| [`vectorize`][hiepy.vectorize], [`devectorize`][hiepy.devectorize] | Flatten a dense tensor and back |
| [`mode_mul_torch`][hiepy.mode_mul_torch], [`multi_mode_mul_torch`][hiepy.multi_mode_mul_torch] | Mode products of dense tensors |

## [Utilities](utils.md)

| Name | Description |
|---|---|
| [`TruncOpts`][hiepy.TruncOpts] | Truncation options `atol`, `rtol`, `max_rank` |
| [`trunc_rank`][hiepy.trunc_rank] | Rank selection from singular values |
| [`Shape`][hiepy.Shape], [`ScalarType`][hiepy.ScalarType], [`TensorDict`][hiepy.TensorDict] | Type aliases |
