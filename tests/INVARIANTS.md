# Invariants of hiepy

This catalog specifies the properties every public operation of hiepy must satisfy. Each invariant has an ID
(e.g. `S2`); tests that check it name the ID in their docstring. The reasons behind individual entries are recorded
in [Decisions](#decisions).

## Conventions

- **Operation**: a public function returning an `HTTensor` (for `truncate*`: the first entry of the returned tuple),
  together with the `HTTensor` methods and operators that wrap it.
- **Frame of node t**: `U[t]` for a leaf, otherwise $(U_{t_l} \otimes U_{t_r}) \cdot \mathrm{mat}(B_t)$.
- **Inputs**: the strategies in `tests/strategies.py` draw float32 and float64 tensors (pairs share one dtype;
  mixed precisions are covered by D2).
- **Error model** (`tests/tolerances.py`, Q9): every numerical check has the form
  `error ≤ bound + rel · scale`.
  - `bound`: the mathematical bound of the invariant (0 for exact results).
  - `rel`: relative rounding error, from the machine epsilon `ε` of the least precise dtype involved and the
    accuracy class of the computed quantity: `SAFETY · ε` for direct computations (`SAFETY = 45`, i.e. about
    `1e-14` in float64), `sqrt(ε)` for singular values computed from Gramian eigenvalues (`truncate_gram`), whose
    eigenvalues are only accurate to `ε·‖x‖²` (Q5).
  - `scale`: the size of the quantities entering the computation. For exact results the larger of `‖reference‖`
    and the size of the inputs, e.g. `m(x) + m(y)` for `x − y`, `m(x)·m(y)` for `contract` (Q7); for approximations
    `m(x)`; for orthonormality `‖FᵀF − I‖_F` the number of columns `k`.
  - `m(x)`, the size of an HT tensor: the product of the spectral norms of its components (transfer tensors
    matricized as `(r_l r_r) × r_t`). It bounds `‖x‖` and equals it for orthonormal components.

  The only tuning knob is `SAFETY`; a failing check reports `bound`, `rel` and `scale` separately.

## S – Structure

| ID | Invariant |
|---|---|
| S1 | The keys of `U` are the leaves `(0,) … (d−1,)` of `dtree`, the keys of `B` are its inner nodes including the root. |
| S2 | Ranks are consistent: `B[t].shape[:2] == (rank[l], rank[r])` for every inner node `t` with children `l`, `r`, and `B[root].shape[2] == 1`. |
| S3 | An `HTTensor` has order ≥ 2. Operations whose result would have order < 2 (`get`, `squeeze`) return a dense `torch.Tensor`. |
| S4 | Operations preserve the dimension tree, except `squeeze` and `get` with an integer index. |
| S5 | Factories and `to_htt` build the canonical tree `DimTree.from_order(d)`. |

## O – Orthogonality flag

A wrong `True` silently breaks `norm` and `gramians`; a wrong `False` only costs an extra `orthog` call.

| ID | Invariant |
|---|---|
| O1 | `is_orthog == True` ⇒ the frame of every non-root node is orthonormal. Checked per component: `U[t]ᵀU[t] = I` for leaves, `mat(B[t])ᵀ mat(B[t]) = I` for inner non-root nodes. |
| O2 | Each operation sets the flag as listed below. |

| Flag of result | Operations |
|---|---|
| `True` | `randn`, `to_htt`, `orthog`, `ones`/`constant` with `orthog=True` |
| `False` | `zeros`, `arange`, `hilbert`, `ones`/`constant` (default), `add`, `sub`, `mode_mul`, `ele_mode_mul`, `ele_mul`, `truncate_svd`, `truncate_gram` |
| as input | `scalar_mul`, `copy`, `clone`, `detach`, `to`, `requires_grad_`, `load` |
| as input if the key selects every entry, else `False` | `get` |
| as input if no mode is removed, else `False` | `squeeze` |

## N – Numerical correctness

| ID | Invariant | Applies to |
|---|---|---|
| N1 | `result.full()` equals the dense counterpart of the operation applied to `x.full()`. | `add`, `sub`, `scalar_mul`, `mode_mul`, `ele_mode_mul`, `ele_mul` without `opts`, `orthog`, `squeeze`, `get`, `copy`, `clone`, `detach`, `to` (in the target precision); `zeros`, `ones`, `constant`, `arange` |
| N2 | `contract(x, y)` equals `(x.full() * y.full()).sum()`; `norm(x)` equals `torch.linalg.norm(x.full())`. | `contract`, `norm` |
| N3 | Round trips are exact: `to_htt(T, {"atol": 0}).full() ≈ T`; `load(save(x))` reproduces components, `dtree` and `is_orthog`; `dematricize ∘ matricize` and `devectorize ∘ vectorize` are the identity. | `to_htt`, `save`/`load`, `matricize`, `vectorize` |
| N4 | **Error vs. tolerance**: `‖x.full() − result.full()‖ ≤ atol + rtol · ‖x‖` (a missing key counts as 0) if no `max_rank` is given, up to rounding of order `m(x)` (Conventions, Q5). Tested down to `atol = 1e-12·‖x‖` and `rtol = 1e-12`; `truncate_gram` and `ele_mul` only down to their resolution `sqrt(ε)` (`1e-8` in float64, `3.5e-4` in float32; Q4). | `to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul` (`atol` only, E8, Q6) |
| N5 | **Error vs. discarded singular values**: `‖x.full() − result.full()‖ ≤ sqrt(Σ_t ‖σ_t‖²)`, where `σ_t` are the returned discarded singular values, up to rounding of order `m(x)` in the accuracy class of the `σ_t` (Gramian for `truncate_gram`, Q5). Also with `max_rank`; same tolerance range as N4. | `truncate_svd`, `truncate_gram` |
| N6 | `rank[t] ≤ max_rank` for every non-root node. | `to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul` |
| N7 | `orthog` does not increase any rank. | `orthog` |
| N8 | `randn` has ranks ≤ `rank`; `hilbert` approximates `1 / (i_0 + … + i_{d−1} + d)`, with a smaller error for `M = 10` than for `M = 3`. | `randn`, `hilbert` |
| N9 | `local_left_svd(x)[t] = (Û_t, S_t)`: `Û_t` is orthonormal, `S_t` are the singular values of the matricization `X_t` of `x.full()`, and with the frame `U_t` of `orthog(x)` the projection onto `U_t Û_t` preserves `X_t`. The eigenvalues of `gramians(orthog(x))[t]` are the squares of `S_t`. Compared up to sign and basis of the vectors. | `local_left_svd`, `gramians` |

## V – Value semantics

| ID | Invariant |
|---|---|
| V1 | Inputs are not modified (components, `dtree`, `is_orthog`). Exception: `requires_grad_` sets the autograd flag in place. |
| V2 | The result shares neither component storage nor the `DimTree` object with its inputs. Exceptions: `detach` shares storage, `to` shares components that need no conversion. |

## D – dtype and device

Rules from CLAUDE.md ("dtype and device"), covered by `tests/test_dtype_device.py`.

| ID | Invariant |
|---|---|
| D1 | All components of a result share one dtype and one device: those of the inputs (for `to`: the requested ones). |
| D2 | Operations with several tensor operands raise `RuntimeError` on different devices and promote dtypes with `torch.promote_types`. |
| D3 | In `scalar_mul`, Python scalars and 0-d tensors do not raise the dtype; a 0-d CPU tensor may be combined with any device. |
| D4 | Factories default to `torch.get_default_dtype()` and `torch.get_default_device()`. |

## A – Autograd

Rules from CLAUDE.md ("Autograd"), covered by `tests/test_autograd.py`.

| ID | Invariant |
|---|---|
| A1 | A result records a graph iff an input component requires gradients and autograd is enabled. |
| A2 | Gradients reach every input component the result depends on. |
| A3 | Gradients pass `torch.autograd.gradcheck` for operations without SVD or eigendecomposition. |
| A4 | Factories return tensors without gradient tracking by default, autograd leaves with `requires_grad=True`. |

## E – Errors and warnings

| ID | Invariant |
|---|---|
| E1 | `add`, `sub`, `contract`, `ele_mul` raise `ValueError` for different trees or shapes. |
| E2 | `mode_mul`, `ele_mode_mul`, `squeeze` raise `ValueError` for modes out of range and for operands of wrong order or size. `squeeze` accepts modes in `[-d, d)`, negative ones counting from the end as in torch, given as `int` or an iterable of `int`; other types (`float`, `bool`, tensors) raise `TypeError`. |
| E3 | `gramians` raises `ValueError` if `is_orthog` is `False`. |
| E4 | `HTTensor(...)` raises `TypeError` for mixed dtypes and `RuntimeError` for mixed devices. |
| E5 | `save`/`load` raise `ValueError` for paths without `.htt`; `load` raises `TypeError` if the file holds no `HTTensor`. |
| E6 | `get` raises like torch: `IndexError` (out of bounds, too many indices), `TypeError` (unsupported key), `ValueError` (non-positive step, slice selecting no entry). |
| E7 | `to_htt` raises `ValueError` for inputs of order < 2. |
| E8 | `ele_mul` ignores `rtol` with a `UserWarning`. |
| E9 | `to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul` (and `truncate`) validate their options: unknown keys, `atol`/`rtol` negative or not finite, or `max_rank < 1` raise `ValueError`; a non-numeric tolerance or non-int `max_rank` raises `TypeError` (Q10). |
| E10 | No empty modes or ranks: factories and `to_htt` with a mode of size 0, `randn` with `rank < 1`, `hilbert` with `M < 1` and `mode_mul` with a matrix without rows raise `ValueError` (Q10). |
| E11 | Outside the supported range (norms beyond about `1e154` in float64), the truncating routines raise `ValueError` instead of failing in `torch.linalg` or returning a meaningless result; `norm` returns `inf` like `torch.linalg.norm` (Q11). |

## Decisions

- **Q1 – Simple orthogonality rule (O2).** An operation sets `True` only if it orthogonalizes by construction.
  Results that happen to be orthogonal (e.g. the projections in `truncate_*`) still get `False`; a flag that
  depends on the numerical case would not be predictable for users. `get` and `squeeze` keep the flag only when
  they return an unchanged copy (every entry selected, no mode removed), which the user can tell from the call.
- **Q2 – `ele_mul` and `rtol` (E8).** Stays a warning, not an error.
- **Q3 – Meaning of `rtol` (N4).** The global contract is `‖x − x̃‖ ≤ atol + rtol · ‖x‖`. `trunc_rank` compares per
  node with the norm of the node's singular values, which equals `‖x‖` for an orthogonalized tensor.
- **Q4 – Gramian-based truncation (N4, N5).** `truncate_gram` is Grasedyck's hierarchical SVD (SIAM J. Matrix Anal.
  Appl. 31(4), 2010, Thm. 3.11), so the bounds hold in exact arithmetic. It computes `σ²` as eigenvalues of
  `X_t X_tᵀ`, so singular values below about `sqrt(ε)·‖x‖` (`1e-8·‖x‖` in float64, `3.5e-4·‖x‖` in float32) cannot
  be resolved (stated in its docstring). Hence `truncate_gram` and `ele_mul`, which also selects ranks from
  Gramians, are tested only down to this resolution of the precision of the data (`tests/tolerances.py::resolution`).
- **Q5 – Rounding margin in N4 and N5.** A margin of `SAFETY·ε · m(x)` absorbs rounding: without it, N5 failed
  where nothing was truncated and both sides were at rounding level (`1e-15` vs. `2e-16`). It is measured in `m(x)`
  rather than `‖x‖` because representations with cancellation produce rounding errors of order `m(x)` (Q7); for
  orthonormal components `m(x) = ‖x‖`. N5 for `truncate_gram` uses the Gramian accuracy class `sqrt(ε) · m(x)`: its
  discarded singular values are `sqrt(λ)` of Gramian eigenvalues with absolute error `ε·‖x‖²`, i.e. accurate only to
  `ε·‖x‖²/σ`, at most `sqrt(ε)·‖x‖`. With the rounding class, N5 failed at `rtol = 1e-4` (excess `1.7e-13` in CI,
  `2.3e-14` locally, for discarded singular values of order `1e-6`).
- **Q6 – `ele_mul` truncates conservatively (N4).** Its tolerance is enforced through an error bound for the
  Kronecker product, which is far from tight for the Hadamard product (measured errors ≈ 3 % of `atol`, ranks may
  exceed those of the exact product). Tests only check N4, not how tight the error or how small the ranks are.
- **Q7 – Scale of exact comparisons (N1, N2).** Rounding errors scale with the size of the components, not with
  `‖x‖`: in a representation with cancellation (e.g. a rank-deficient leaf whose columns the parent combines to a
  small vector) `‖x‖` can be orders of magnitude smaller than the components, and `norm` deviated by `1e-13 · ‖x‖`.
  Measured against `m(x)`, all operations stay below `1e-14` (standard bound `|fl(AB) − AB| ≲ ε‖A‖‖B‖`); for
  orthonormal components `m(x) = ‖x‖`, so the check is as strict as before.
- **Q8 – Arbitrary dimension trees.** Every binary tree over the modes is supported, including unbalanced trees,
  non-contiguous nodes such as `(0, 2) | (1, 3)` and left children with larger modes than their siblings.
  `full` used to assume mode-ordered rows and returned permuted entries for such trees; this went unnoticed because
  the test inputs only used canonical trees and those produced by `squeeze`. The strategies now draw arbitrary trees,
  and `tests/httensor/test_full.py` keeps the regression cases.
- **Q9 – Central error model.** Every tolerance is derived in `tests/tolerances.py` from the error model in the
  Conventions instead of being chosen per test. Before, almost every false failure came from a source of rounding
  error that a test did not model (cancellation in the operation or in the representation, Gramian accuracy, growth
  of `‖FᵀF − I‖_F` with the number of columns, float32 data), and the fixes were scattered over several files.
  Orthonormality is checked against `SAFETY·ε · k` (previously a constant, relaxed from `1e-14` to `1e-13` after
  `to_htt` produced deviations up to `1.5e-14` for frames with `k = 80` columns; `orthog` stays below `2e-15`). The
  precision is that of the least precise dtype involved, so float32 data converted to float64 is compared at float32
  accuracy.
- **Q10 – Argument validation (E6, E9, E10).** Invalid arguments raise a clear error at the entry of an operation
  instead of being accepted silently or failing later: a misspelled option such as `{"tol": 1e-3}` used to be
  ignored (no truncation), negative or `nan` tolerances failed with `IndexError` inside `trunc_rank`, and
  `max_rank = 0`, `randn(rank=0)` or a zero-row matrix in `mode_mul` produced components of rank or size 0. HT tensors
  have no empty modes and no rank 0, consistent with the decision to reject empty slices in `get`.
- **Q11 – Supported range (E11).** Squared singular values and Gram matrices overflow for norms beyond about `1e154`
  in float64. The truncating routines check for non-finite singular values (`trunc_rank`) and Gram matrices
  (`left_svd_gramian`) and raise `ValueError` naming the range; before, they failed with `IndexError` or
  `LinAlgError`, or returned a result truncated against a `nan` tolerance. The range is documented in the
  docstrings; scaling the computations internally to support the full float64 range was not pursued.
