# HiePy

[![CI](https://github.com/Beschroe/HiePy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Beschroe/HiePy/actions/workflows/ci.yml)

A comprehensive framework for the Hierarchical Tucker format implemented in PyTorch.

## Development

```bash
pip install -e ".[dev,docs]"
pytest                          # tests and docstring examples
HYPOTHESIS_PROFILE=ci pytest    # as in CI (more hypothesis examples)
pytest --cov --cov-report=html  # coverage report in htmlcov/index.html
```

The tests check the invariants specified in [`tests/INVARIANTS.md`](tests/INVARIANTS.md).
