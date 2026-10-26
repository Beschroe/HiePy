# HiePy

[![CI](https://github.com/Beschroe/HiePy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Beschroe/HiePy/actions/workflows/ci.yml)
[![Docs](https://github.com/Beschroe/HiePy/actions/workflows/docs.yml/badge.svg?branch=main)](https://beschroe.github.io/HiePy/)

A comprehensive framework for the Hierarchical Tucker format implemented in PyTorch.

Documentation: <https://beschroe.github.io/HiePy/>

## Development

```bash
pip install -e ".[dev,docs]"
pytest                          # tests and docstring examples
HYPOTHESIS_PROFILE=ci pytest    # as in CI (more hypothesis examples)
pytest --cov --cov-report=html  # coverage report in htmlcov/index.html
mkdocs serve                    # documentation at http://127.0.0.1:8000
```

The tests check the invariants specified in [`tests/INVARIANTS.md`](tests/INVARIANTS.md).
