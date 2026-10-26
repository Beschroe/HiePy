# Changelog

All notable changes to hiepy are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions (0.x.0) may contain incompatible changes.

## [Unreleased]

## [0.1.0]

First public release.

### Added

- `HTTensor`: tensors in Hierarchical Tucker format on arbitrary binary dimension trees (`DimTree`).
- Construction: `to_htt` (from dense tensors with prescribed accuracy), `zeros`, `ones`, `constant`, `arange`,
  `randn`, `hilbert`.
- Arithmetic: `add`, `sub`, `scalar_mul`, `mode_mul`, `ele_mode_mul`, `ele_mul`, `contract`, `norm`.
- Orthogonalization (`orthog`) and truncation with a guaranteed error bound (`truncate`, `truncate_svd`,
  `truncate_gram`).
- Indexing with integers and slices following torch, `squeeze`, `full`, `save` / `load`.
- Autograd support following the torch conventions; dtype conversion with `to`.

[Unreleased]: https://github.com/Beschroe/HiePy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Beschroe/HiePy/releases/tag/v0.1.0
