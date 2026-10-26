# HTTensor

Methods, properties and operators of `HTTensor` call the functions of the same name:

| Method / operator | Function |
|---|---|
| `x.order`, `x.shape`, `x.rank`, `x.nbytes`, `x.dtype`, `x.device`, `x.requires_grad` | [`order`][hiepy.order], [`shape`][hiepy.shape], [`rank`][hiepy.rank], [`nbytes`][hiepy.nbytes], [`dtype`][hiepy.dtype], [`device`][hiepy.device], [`requires_grad`][hiepy.requires_grad] |
| `x[key]`, `x.get(key)` | [`get`][hiepy.get] |
| `x.full()`, `x.norm()`, `x.gramians()` | [`full`][hiepy.full], [`norm`][hiepy.norm], [`gramians`][hiepy.gramians] |
| `x.orthog()`, `x.truncate(opts, method)`, `x.squeeze(dims)` | [`orthog`][hiepy.orthog], [`truncate`][hiepy.truncate], [`squeeze`][hiepy.squeeze] |
| `x + y`, `x - y`, `c * x`, `x * y`, `x @ y` | [`add`][hiepy.add], [`sub`][hiepy.sub], [`scalar_mul`][hiepy.scalar_mul], [`ele_mul`][hiepy.ele_mul], [`contract`][hiepy.contract] |
| `x.mode_mul(A, dim)`, `x.ele_mode_mul(v, dim)` | [`mode_mul`][hiepy.mode_mul], [`ele_mode_mul`][hiepy.ele_mode_mul] |
| `x.copy()`, `x.clone()`, `x.detach()`, `x.to(dtype)`, `x.requires_grad_()` | [`copy`][hiepy.copy], [`clone`][hiepy.clone], [`detach`][hiepy.detach], [`to`][hiepy.to], [`requires_grad_`][hiepy.requires_grad_] |
| `x.save(path)`, `HTTensor.load(path)` | [`save`][hiepy.save], [`load`][hiepy.load] |

::: hiepy.HTTensor

::: hiepy.order

::: hiepy.rank

::: hiepy.shape

::: hiepy.dtype

::: hiepy.device

::: hiepy.requires_grad

::: hiepy.full

::: hiepy.get

::: hiepy.gramians

::: hiepy.save

::: hiepy.load

::: hiepy.nbytes
