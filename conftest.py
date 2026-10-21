"""Repository-wide pytest configuration.

Docstring examples set ``torch.set_default_dtype(torch.float64)`` themselves, so that users can reproduce the printed
results. Since this setting is process-wide, the previous default dtype is restored after every doctest, so that it
does not leak into other doctests or regular tests.
"""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype_after_doctests(request: pytest.FixtureRequest) -> Iterator[None]:
    """Restore the default dtype after a doctest changed it."""
    if not isinstance(request.node, pytest.DoctestItem):
        yield
        return
    previous = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
