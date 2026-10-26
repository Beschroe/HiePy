# CLAUDE.md — hiepy

## Purpose

`hiepy` is a PyTorch library for tensors in the Hierarchical Tucker (HT) format. Its central
data structure `HTTensor` stores a high-dimensional tensor as leaf matrices and transfer tensors
on a binary dimension tree (`DimTree`). Around it, the library provides construction, algebra,
orthogonalization, rank truncation, indexing and conversion to and from dense `torch.Tensor`s.

## Directory structure

```
src/hiepy/
├── __init__.py          # Public API: re-exports every public function/class (see __all__)
├── httensor/            # The data structure and its read-only / structural accessors
│   ├── httensor.py      # class HTTensor (U, B, dtree, is_orthog) + method/operator wrappers
│   ├── getter.py        # order, rank, shape, nbytes, dtype, device, requires_grad
│   ├── get_item.py      # get / __getitem__ (torch-style int/slice indexing)
│   ├── full.py          # full: HT -> dense torch.Tensor
│   ├── gramians.py      # gramians: reduced Gram matrices of an orthogonal HT tensor
│   └── io.py            # save / load (*.htt files via torch.save/torch.load)
├── factory/             # Constructors returning new HTTensors
│   ├── zeros.py, ones.py, constant.py, arange.py   # exact low-rank tensors
│   ├── randn.py         # random orthogonal HT tensor with bounded rank
│   ├── hilbert.py       # Hilbert tensor via sinc quadrature
│   ├── to_htt.py        # dense torch.Tensor -> HT (hierarchical SVD)
│   └── copy.py          # copy / clone / detach (autograd-aware copies), to (dtype/device conversion), requires_grad_
├── algebra/             # Operations on HTTensors returning HTTensors (or scalars)
│   ├── add.py, sub.py   # exact sum/difference via block-diagonal embedding
│   ├── scalar_mul.py    # scaling via root transfer tensor
│   ├── mode_mul.py      # matrix x HT tensor along one mode
│   ├── ele_mode_mul.py  # elementwise scaling along one mode by a vector
│   ├── ele_mul.py       # Hadamard product with on-the-fly truncation
│   ├── contract.py      # full inner product <x, y>
│   └── tensordot.py     # DEAD: fully commented out, "under maintenance"
├── linalg/              # Dense helpers on torch.Tensor + HT-level linear algebra
│   ├── matricize.py     # matricize / dematricize (dense)
│   ├── vectorize.py     # vectorize / devectorize (dense)
│   ├── mode_mul_torch.py   # mode_mul_torch / multi_mode_mul_torch (dense)
│   ├── diagonal_tensor.py  # dense generalized-diagonal tensor (internal, not exported)
│   ├── left_svd_gramian.py # left singular vectors/values from a Gram matrix (dense)
│   ├── local_left_svd.py   # node-wise singular values/vectors of an HT tensor
│   └── norm.py          # Frobenius norm of an HT tensor
├── transformation/      # Change the representation of an HTTensor
│   ├── orthog.py        # orthog: bottom-up QR orthogonalization
│   ├── truncate.py      # truncate (dispatcher), truncate_svd, truncate_gram
│   ├── squeeze.py       # remove singleton modes (restructures the tree)
│   └── change_root.py   # DEAD: fully commented out, "DEPRECATED"
└── utils/
    ├── dimtree.py       # class DimTree, type Node
    ├── data_types.py    # type aliases Shape, TensorDict, ScalarType, IndexKey; TypedDict TruncOpts
    ├── dtype_device.py  # result_dtype / check_device for multi-operand operations (internal)
    └── trunc_rank.py    # rank selection from singular values (atol/rtol/max_rank)
```

Related top-level locations:

- `tests/` — pytest suite, organized around the invariant catalog (see "Tests" below).
- `docs/` + `mkdocs.yml` — MkDocs (material + mkdocstrings, NumPy docstring style): Home (`index.md`),
  `getting-started.md`, User Guide (`user-guide/`: HT format, operations, truncation, practical notes) and the API
  reference. The pages in `docs/api/*.md` list the documented members **explicitly** (`::: hiepy.<name>`);
  `docs/api/index.md` links every public name — add new operations there too. The `>>>` examples of all pages run as
  doctests (same conventions as docstring examples; leave a blank line before the closing code fence). Every push to
  `main` publishes
  the documentation to <https://beschroe.github.io/HiePy/> (`.github/workflows/docs.yml`, GitHub Pages via Actions);
  pull requests only check that it builds (`docs` job in `ci.yml`).

## Architecture and conventions

These are the things to know before reading or changing code. For the mathematics, read the
docstrings referenced in the functionality map below.

### Functional core, thin methods

- Every operation is implemented as a **free function** in its own module. `HTTensor` methods,
  properties and operators (`+`, `-`, `*`, `@`, `[]`) are thin wrappers that lazily import that
  function (to avoid circular imports). See the method blocks in `httensor/httensor.py`.
- Adding a new public operation therefore means: (1) a module with the function,
  (2) export in `hiepy/__init__.py` (import + `__all__`), (3) optionally a wrapper method on
  `HTTensor`, (4) an entry in the matching `docs/api/*.md` page, (5) an entry in the registry `tests/ops.py`
  with its dense reference and, if the result can be larger than `‖x‖`, its `size` of the inputs (and, for a new
  kind of property, an invariant in `tests/INVARIANTS.md`).
- Argument order: the tensor comes first, e.g. `mode_mul(x, A, dim)`, `mode_mul_torch(x, A, dim)`,
  `ele_mode_mul(x, v, dim)`, `multi_mode_mul_torch(x, matrices)`. Internal calls use keyword
  arguments (`x=..., A=...`); keep that style.

### The HTTensor data layout

- `U: dict[Node, Tensor]` holds leaf matrices, `B: dict[Node, Tensor]` holds 3-D transfer tensors
  of inner nodes (including the root), `dtree: DimTree` holds the topology, `is_orthog: bool` is a
  flag. The index convention of `B[t]` (left child, right child, node) and the root rank of 1 are
  documented in `HTTensor` and in `httensor/getter.py::rank`.
- `rank` returns the **representational** ranks read off the component shapes, not algebraic
  ranks (see `getter.py::rank`). `orthog` may shrink them (see its *Notes*).
- `shape` is derived from leaf-matrix row counts sorted by leaf key; there is no stored shape.
- `is_orthog` is plain metadata. It is set by the functions that create/transform tensors, never
  verified automatically, so every new operation must set it correctly (a wrong `True` silently
  breaks `norm`, which then reads the norm off the root). `gramians` requires it to be `True`;
  `norm`, `ele_mul`, `truncate_*` and `local_left_svd` call `orthog` internally, which returns a
  copy (cheaply, if the flag is already set).

### DimTree and nodes

- `Node` is a `tuple[int, ...]` of mode indices, e.g. `(0, 1)`; leaves are always `(i,)` (also inside `squeeze`,
  where an absorbed leaf's father takes the key of its sibling).
- All factories and `to_htt` use the canonical balanced tree `DimTree.from_order(d)`. After
  `squeeze` (and thus after integer indexing) the tree may be non-canonical.
- Many operations address leaves directly as `U[(dim,)]` (`mode_mul`, `ele_mode_mul`, `get`,
  `hilbert`, the factories), i.e. they assume singleton-tuple leaves labeled by mode index.
- Binary operations (`add`, `sub`, `contract`, `ele_mul`) require **identical** dimension trees
  (`DimTree.__eq__` / `is_equal`) and identical shapes; they raise `ValueError` otherwise.
- Unlike the tensor operations, `DimTree.map`, `prune`, `move_subtree` and `consolidate`
  **mutate the tree in place**; `DimTree.copy` exists for that reason. `squeeze` is the main
  consumer of these mutators.

### Value semantics

- Tensor-level operations do not mutate their inputs: they start from `x.copy()` (or build new
  dicts and copy at the end, as `get` does). Keep this invariant when adding operations.
- `copy` = detached deep copy, `clone` = deep copy that stays in the autograd graph,
  `detach` = shares storage, cuts the graph (see `factory/copy.py`).
- Operations that start from a copy of their input use `x.clone()`, not `x.copy()`, so that the
  result stays in the autograd graph (see "Autograd").

### Indexing (`get`, `x[...]`)

- Follows dense torch indexing restricted to ints and slices; the full specification (result
  types, errors, unsupported keys) is in the docstring of `httensor/get_item.py::get`, and
  `tests/httensor/test_get_item.py` is the executable reference.
- Results with fewer than two remaining modes are returned as dense `torch.Tensor`s, since HT
  tensors need order ≥ 2 (the same limit `squeeze` enforces).

### Autograd

- `import hiepy` has no global side effects; the torch defaults apply (default dtype `float32`,
  autograd enabled). There is no hiepy-level configuration object.
- Gradients are tracked exactly as in torch: an operation records a graph if and only if a component
  of an input requires gradients and autograd is enabled. Default is no tracking (factories use
  `requires_grad=False`); opt in via the factory argument or `x.requires_grad_()`, opt out via
  `detach`/`copy` or `torch.no_grad()`. There is no per-function `track_grad` switch.
- No operation detaches implicitly. Rules for implementations: no `.detach()`, `.copy()`, `.item()`,
  `float(...)` on values that enter the result; no in-place changes of input components (writing
  into freshly allocated buffers, as in `add`, is fine); discrete decisions such as ranks from
  `trunc_rank` may be computed from detached values.
- Gradients through `torch.linalg.svd`/`eigh` (`to_htt`, `truncate_*`, `local_left_svd`, `ele_mul`
  with truncation) are unstable for (nearly) equal singular values.
- `tests/test_autograd.py` checks propagation, absence of a graph and `gradcheck` for every operation of the
  registry `tests/ops.py` (except `copy`/`detach`, which cut the graph by design).

### dtype and device

- Invariant: all components of an `HTTensor` share one dtype and one device. `HTTensor.__init__`
  checks this (`TypeError` for mixed dtypes, `RuntimeError` for mixed devices); `x.dtype` /
  `x.device` read it off the root transfer tensor. Code that assigns `x.U[t]` / `x.B[t]` directly
  bypasses the check and must keep the invariant itself.
- Every operation returns the dtype and device of its inputs. Internal helper tensors must be
  allocated from the input (`dtype=x.dtype, device=x.device`, `t.new_zeros(...)`), never with
  bare `torch.eye(1)` / `torch.zeros(...)`.
- Operations with several tensor operands (`add`, `sub`, `contract`, `ele_mul`, `mode_mul`,
  `ele_mode_mul`) first call `check_device(x=..., y=...)` (raises `RuntimeError`) and then cast
  all operands to `result_dtype(...)` (`torch.promote_types`, the more precise dtype wins), both
  from `utils/dtype_device.py`. Keep this pattern for new binary operations.
- Scalars follow torch's promotion rules: Python scalars and 0-d tensors do not raise the dtype
  (`scalar_mul` uses `torch.result_type`); a 0-d CPU tensor may be combined with any device.
- Factories take `dtype=None, device=None` (before `requires_grad`), defaulting to
  `torch.get_default_dtype()` / `torch.get_default_device()`. `randn` passes `generator` through
  unchanged, so its device must match `device` (torch raises otherwise).
- `x.to(dtype, device)` converts like `torch.Tensor.to` (stays in the autograd graph, shares
  components that need no conversion); `load(path, map_location)` restores onto another device.

### Truncation options

- `TruncOpts` (`utils/data_types.py`) is a `TypedDict` with optional keys `atol`, `rtol`, `max_rank`.
- Every routine that truncates (`to_htt`, `truncate_svd`, `truncate_gram`, `ele_mul`) rescales the
  global tolerance to a per-node tolerance itself (factor `sqrt(2d - 3)`, see code comments) and
  then delegates rank selection to `utils/trunc_rank.py::trunc_rank`.
- Which keys each routine honours is listed in its docstring (`ele_mul` ignores `rtol` and warns).
- `truncate*` returns a tuple `(tensor, discarded_singular_values_per_node)`, not just a tensor.
- Options are validated at the entry of every truncating routine by `utils/checks.py::check_trunc_opts` (unknown
  keys, negative/non-finite tolerances, `max_rank < 1` raise); `trunc_rank` and `left_svd_gramian` raise
  `ValueError` for non-finite values (supported range: norms up to about `1e154` in float64).
- HT tensors have no empty modes and no rank 0: factories and `to_htt` check shapes with
  `utils/checks.py::check_shape`; empty slices in `get`, `rank < 1`, `M < 1` and zero-row matrices raise.

### Code style and tooling

- Python 3.12 only (`requires-python = "==3.12.*"`; uses `type X = ...` aliases and `typing.Self`). Runtime dependencies: `torch` and `numpy` (hiepy itself does not use numpy, but torch warns on import without it).
- Ruff: line length 120, rule sets `E`, `F`, `ANN` (type annotations required, also in tests).
  Config in `pyproject.toml`.
- Docstrings: NumPy style with LaTeX math (`$...$`, `$$...$$`) rendered by MkDocs/arithmatex,
  zero-based mode indices in formulas. Every public function documents preserved properties
  (tree, ranks, orthogonality) and arithmetic cost in its *Notes* section — keep that pattern.
- Docstring examples run as doctests in every plain `pytest` run (`testpaths` includes `src/hiepy` and `docs`,
  `addopts` includes `--doctest-modules` and `--doctest-glob=*.md`, so the examples of the documentation pages
  run too). Every example starts, right after its imports, with
  `>>> torch.set_default_dtype(torch.float64)`, so users can reproduce the printed results; expected
  outputs are therefore float64 (no `dtype=` suffix). Keep this line in new examples. The root
  `conftest.py` only restores the previous default dtype after each doctest. Rounding-level results
  (approximation errors, sign-ambiguous singular vectors) depend on torch version and BLAS, so examples
  check them against a bound (`bool(err <= tol)` → `True`, `torch.allclose(...)`) instead of printing
  them; other floats are matched with the `NUMBER` doctest flag (precision as written); outputs of unseeded random examples are illustrative only and marked
  `# doctest: +SKIP`.
- Commented-out legacy code (German comments, old `hiepy.core` import paths, old `Node` objects
  with `.dims`) still exists in several modules. It is not part of the API; do not revive it
  without adapting it to the current `DimTree`/`Node` design.

## Functionality map

Names are importable from `hiepy` unless marked *internal*. "Method" lists the `HTTensor`
counterpart. For semantics, complexity and preserved properties read the docstring of the listed
file.

### Data structures and types

| Name | File | Method / note |
|---|---|---|
| `HTTensor` | `httensor/httensor.py` | central class |
| `DimTree`, `Node` | `utils/dimtree.py` | tree navigation, traversal (`get_nodes`, `get_levels`, `get_leaves`, `get_inner_nodes`), `from_order`, ASCII `__str__` |
| `Shape`, `TensorDict`, `ScalarType`, `TruncOpts` | `utils/data_types.py` | type aliases / options |
| `IndexKey` (*not exported*) | `utils/data_types.py` | per-mode index type used by `get` |

### Construction (`factory/`)

| Name | File | Purpose |
|---|---|---|
| `zeros`, `ones`, `constant` | `factory/zeros.py`, `ones.py`, `constant.py` | exact rank-1 tensors (`ones`/`constant` optionally orthogonal) |
| `arange` | `factory/arange.py` | consecutive values, exact rank 2, not orthogonal |
| `randn` | `factory/randn.py` | random orthogonal tensor, rank bounded by `rank`, optional `generator` |
| `hilbert` | `factory/hilbert.py` | approximated Hilbert tensor, rank `2M+1`, meant to be truncated afterwards |
| `to_htt` | `factory/to_htt.py` | dense → HT via hierarchical SVD with `TruncOpts` (floating-point input only) |
| `copy`, `clone`, `detach` | `factory/copy.py` | `.copy()`, `.clone()`, `.detach()` |
| `to` | `factory/copy.py` | `.to(dtype, device)`; dtype/device conversion |
| `requires_grad_` | `factory/copy.py` | `.requires_grad_(flag)`; sets the autograd flag on all components in place |

### Inspection and access (`httensor/`)

| Name | File | Method / note |
|---|---|---|
| `order`, `shape`, `rank`, `nbytes`, `dtype`, `device`, `requires_grad` | `httensor/getter.py` | properties `.order`, `.shape`, `.rank`, `.nbytes`, `.dtype`, `.device`, `.requires_grad` |
| `get` | `httensor/get_item.py` | `.get()`, `x[...]`; torch-style, see "Indexing" above |
| `full` | `httensor/full.py` | `.full()`; dense reconstruction, exponential memory |
| `gramians` | `httensor/gramians.py` | `.gramians()`; requires orthogonal input |
| `save`, `load` | `httensor/io.py` | `.save()`, `HTTensor.load()`; `.htt` suffix enforced, pickled via torch |

### Algebra (`algebra/`)

| Name | File | Method / operator | Rank effect (details in docstring) |
|---|---|---|---|
| `add` | `algebra/add.py` | `.add()`, `+` | additive growth, needs truncation |
| `sub` | `algebra/sub.py` | `.sub()`, `-` | additive growth, needs truncation |
| `scalar_mul` | `algebra/scalar_mul.py` | `.scalar_mul()`, `*` with scalar / 0-d tensor | unchanged |
| `ele_mul` | `algebra/ele_mul.py` | `.ele_mul()`, `*` with HTTensor | multiplicative unless `opts` truncates |
| `mode_mul` | `algebra/mode_mul.py` | `.mode_mul(A, dim)` | unchanged, mode size changes |
| `ele_mode_mul` | `algebra/ele_mode_mul.py` | `.ele_mode_mul(v, dim)` | unchanged |
| `contract` | `algebra/contract.py` | `.contract()`, `@` | returns a 0-d `torch.Tensor` |

Not available: general `tensordot` between HT tensors (module commented out), Kronecker product.

### Transformations (`transformation/`)

| Name | File | Method / note |
|---|---|---|
| `orthog` | `transformation/orthog.py` | `.orthog()`; may shrink over-parameterized components |
| `truncate` | `transformation/truncate.py` | `.truncate(opts, method)`; dispatches on `method="svd"` (default) / `"gram"` |
| `truncate_svd` | `transformation/truncate.py` | orthogonalize, root-to-leaves SVD sweep, then projection; preferred variant |
| `truncate_gram` | `transformation/truncate.py` | via `gramians` + `left_svd_gramian`; less accurate |
| `squeeze` | `transformation/squeeze.py` | `.squeeze()`; removes singleton modes, may return 0-d tensor, keeps order ≥ 2, may leave a non-canonical tree |

### Linear algebra (`linalg/`)

| Name | File | Operates on |
|---|---|---|
| `norm` | `linalg/norm.py` | HTTensor (`.norm()`), via root of orthogonalized tensor |
| `local_left_svd` | `linalg/local_left_svd.py` | HTTensor; node-wise singular values (same sweep as `truncate_svd`, without truncation) |
| `left_svd_gramian` | `linalg/left_svd_gramian.py` | dense Gram matrix → (vectors, singular values) |
| `matricize`, `dematricize` | `linalg/matricize.py` | dense tensor, C-order, negative modes allowed |
| `vectorize`, `devectorize` | `linalg/vectorize.py` | dense tensor, C-order |
| `mode_mul_torch` | `linalg/mode_mul_torch.py` | dense tensor, one or several merged modes |
| `multi_mode_mul_torch` | `linalg/mode_mul_torch.py` | dense tensor, several modes, greedy ordering |
| `diagonal_tensor` (*internal*) | `linalg/diagonal_tensor.py` | dense diagonal tensor (shape tuple, order ≥ 2), used by `hilbert` |

### Utilities (`utils/`)

| Name | File | Purpose |
|---|---|---|
| `trunc_rank` | `utils/trunc_rank.py` | rank from singular-value tail norm vs. `atol + rtol·‖s‖`, clipped to `max_rank` |
| `result_dtype`, `check_device` (*internal*) | `utils/dtype_device.py` | dtype promotion and device check for multi-operand operations |

## Tests

- `tests/INVARIANTS.md` is the specification: every invariant has an ID (`S2`, `N4`, `E9`, …) that the tests name
  in their docstrings, and the *Decisions* section records why (tolerances, supported range, flags, …).
- `tests/ops.py` is the registry of operations and factories with the properties expected of them (dense reference,
  orthogonality flag, argument strategy, storage sharing, scale). The invariant tests in `tests/invariants/` iterate
  over it, so a new operation is tested by adding one entry.
- `tests/strategies.py` generates inputs with hypothesis: per-node ranks (incl. over-parameterized) and component
  kinds (orthonormal, general, rank-deficient), zero tensors, scales, canonical/`squeeze`d/arbitrary trees,
  float32 and float64. A failing example prints a `build_ht(...)` call that rebuilds it.
- `tests/tolerances.py` is the only place that sets numerical tolerances; never use literal tolerances in tests.
  Error model as in the LAPACK test suite: test ratio `(error − bound) / (ε · size of the inputs) ≤ THRESH = 1000`
  (`contract`: `THRESH_CONTRACT = 100 000`, `Op.thresh`),
  with `ε` of the least precise dtype and the size of the inputs from norm inequalities (`Op.input_size` in
  `tests/ops.py`: `‖x‖`, `‖x‖ + ‖y‖`, `‖x‖·‖y‖`, …); orthonormality `‖FᵀF − I‖ / (k·ε) ≤ THRESH`; singular values
  from Gramians use the margin `√ε · size`. Details and measurements: `tests/INVARIANTS.md` (Conventions, Q12).
  `tests/test_tolerances.py` checks that the model still detects real deviations.
- Profiles: `dev` (50 examples, default) and `ci` (200, derandomized; `HYPOTHESIS_PROFILE=ci`). Warnings are
  errors (`filterwarnings = ["error"]`); expected warnings need `pytest.warns`.
- Layout: `tests/invariants/` (S, O, V, N, E across all operations), `tests/httensor/`, `tests/linalg/`,
  `tests/transformation/`, `tests/utils/` (module-specific tests), `test_autograd.py` (A), `test_dtype_device.py`
  (D, E4).
- CI (`.github/workflows/ci.yml`): lint (ruff, mypy), tests against the minimum and latest torch (single thread),
  docs (`mkdocs build --strict`); the coverage table is on the run summary, the HTML report is artifact
  `coverage-html`.

## Typical internal call chains

Useful when debugging or changing a core routine, because changes propagate:

- `truncate_svd` / `truncate_gram` / `ele_mul` / `norm` / `local_left_svd` → `orthog`
- `truncate_gram`, `ele_mul` → `gramians` → `left_svd_gramian` → `trunc_rank`
- `truncate_svd`, `to_htt` → `torch.linalg.svd` → `trunc_rank`
- `orthog`, `to_htt`, `truncate_svd`, `full` → `matricize`/`dematricize`, `mode_mul_torch`/`multi_mode_mul_torch`
- `get` → `squeeze` → `DimTree.map` / `prune` / `move_subtree`; `get` → `full` for results of order < 2

## Where to look for details

- Mathematical definitions, preserved properties, complexity, examples: the docstring of the
  function (file paths in the tables above).
- Expected behaviour of indexing: `tests/httensor/test_get_item.py`.
- Invariants every operation must satisfy, and the decisions behind them: `tests/INVARIANTS.md`.
- Rendered API docs: `mkdocs serve` (needs the `docs` extra), navigation in `mkdocs.yml`.
- Package metadata, extras and lint config: `pyproject.toml`.

## Releases

- Versioning: Semantic Versioning (PEP 440 spelling). Before 1.0, minor versions (`0.x.0`) may contain incompatible
  changes, patch versions (`0.x.y`) only fixes; pre-releases are `0.x.0a1`, `b1`, `rc1`. A version on PyPI can never
  be uploaded again.
- The version lives only in `pyproject.toml`; `hiepy.__version__` reads it from the installed metadata. Every
  release gets an entry in `CHANGELOG.md` (Keep a Changelog).
- Publishing: `.github/workflows/release.yml` with PyPI Trusted Publishing. Publishing a GitHub release with tag
  `vX.Y.Z` (must equal the version) uploads to PyPI; running the workflow manually uploads to TestPyPI.
- The package contains only `src/hiepy` (with `py.typed`); `MANIFEST.in` keeps `tests/` out of the sdist.

## Commands

```bash
pip install -e ".[dev,docs]"   # a local .venv exists in the repo root
pytest                               # tests/, docstring examples and docs/*.md examples (hypothesis profile "dev")
pytest tests/                        # tests only
HYPOTHESIS_PROFILE=ci pytest         # as in CI: 200 examples per test, derandomized
pytest --cov --cov-report=html       # coverage (fails below 90 %), report in htmlcov/index.html
ruff check src/hiepy tests
mkdocs serve
```
