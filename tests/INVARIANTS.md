# Invariants of hiepy

This catalog specifies the properties every public operation of hiepy must satisfy. Each invariant has an ID
(e.g. `S2`); tests that check it name the ID in their docstring. The reasons behind individual entries are recorded
in [Decisions](#decisions).

## Conventions

- **Operation**: a public function returning an `HTTensor` (for `truncate*`: the first entry of the returned tuple),
  together with the `HTTensor` methods and operators that wrap it.
- **Frame of node t**: `U[t]` for a leaf, otherwise $(U_{t_l} \otimes U_{t_r}) \cdot \mathrm{mat}(B_t)$.
- **Tolerances**: numerical checks run in float64 with constants defined once in `tests/helpers.py`, independent of
  order and rank (Q5):
  - `RTOL_EXACT = 1e-14` for exact operations: `‖result − reference‖ ≤ RTOL_EXACT · scale`, where `scale` is the
    larger of `‖reference‖` and the size of the inputs entering it, e.g. `m(x) + m(y)` for `x − y`, `m(x)·m(y)` for
    `contract` (Q7). For float32, the same multiple of the machine epsilon.
  - `m(x)`, the size of an HT tensor: the product of the spectral norms of its components (transfer tensors
    matricized as `(r_l r_r) × r_t`). It bounds `‖x‖` and equals it for orthonormal components.
  - `TOL_ORTHOG = 1e-14` for orthonormality: `‖FᵀF − I‖ ≤ TOL_ORTHOG`.

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
| N4 | **Error vs. tolerance**: `‖x.full() − result.full()‖ ≤ atol + rtol · ‖x‖` (a missing key counts as 0), without margin, if no `max_rank` is given. Tested down to `atol = 1e-12·‖x‖` and `rtol = 1e-12`; `truncate_gram` and `ele_mul` only down to `1e-8` (Q4). | `to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul` (`atol` only, E8, Q6) |
| N5 | **Error vs. discarded singular values**: `‖x.full() − result.full()‖ ≤ sqrt(Σ_t ‖σ_t‖²) + 1e-14 · ‖x‖`, where `σ_t` are the returned discarded singular values (Q5). Also with `max_rank`; same tolerance range as N4. | `truncate_svd`, `truncate_gram` |
| N6 | `rank[t] ≤ max_rank` for every non-root node. | `to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul` |
| N7 | `orthog` does not increase any rank. | `orthog` |
| N8 | `randn` has ranks ≤ `rank`; `hilbert` approximates `1 / (i_0 + … + i_{d−1} + d)`, with an error that decreases as `M` grows. | `randn`, `hilbert` |
| N9 | `local_left_svd(x)[t]` returns orthonormal vectors and the singular values of the matricization of `x.full()` at `t`; the eigenvalues of `gramians(x)[t]` are their squares. | `local_left_svd`, `gramians` |

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
| E2 | `mode_mul`, `ele_mode_mul`, `squeeze` raise `ValueError` for modes out of range and for operands of wrong order or size. `squeeze` accepts modes in `[-d, d)`, negative ones counting from the end as in torch. |
| E3 | `gramians` raises `ValueError` if `is_orthog` is `False`. |
| E4 | `HTTensor(...)` raises `TypeError` for mixed dtypes and `RuntimeError` for mixed devices. |
| E5 | `save`/`load` raise `ValueError` for paths without `.htt`; `load` raises `TypeError` if the file holds no `HTTensor`. |
| E6 | `get` raises like torch: `IndexError` (out of bounds, too many indices), `TypeError` (unsupported key), `ValueError` (non-positive step). |
| E7 | `to_htt` raises `ValueError` for inputs of order < 2. |
| E8 | `ele_mul` ignores `rtol` with a `UserWarning`. |

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
  `X_t X_tᵀ`, so singular values below about `sqrt(ε)·‖x‖ ≈ 1e-8·‖x‖` cannot be resolved (stated in its
  docstring). Hence `truncate_gram` and `ele_mul`, which also selects ranks from Gramians, are tested only down to
  `1e-8`.
- **Q5 – Buffer in N5.** `1e-14 · ‖x‖` absorbs rounding: without it, N5 failed where nothing was truncated and both
  sides were at rounding level (`1e-15` vs. `2e-16`). The test constants (Conventions) are fixed and relaxed
  centrally if they cause false failures.
- **Q6 – `ele_mul` truncates conservatively (N4).** Its tolerance is enforced through an error bound for the
  Kronecker product, which is far from tight for the Hadamard product (measured errors ≈ 3 % of `atol`, ranks may
  exceed those of the exact product). Tests only check N4, not how tight the error or how small the ranks are.
- **Q7 – Scale of exact comparisons (N1, N2).** Rounding errors scale with the size of the components, not with
  `‖x‖`: in a representation with cancellation (e.g. a rank-deficient leaf whose columns the parent combines to a
  small vector) `‖x‖` can be orders of magnitude smaller than the components, and `norm` deviated by `1e-13 · ‖x‖`.
  Measured against `m(x)`, all operations stay below `1e-14` (standard bound `|fl(AB) − AB| ≲ ε‖A‖‖B‖`); for
  orthonormal components `m(x) = ‖x‖`, so the check is as strict as before.
