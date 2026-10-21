"""Shared fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture
def grad_enabled() -> Iterator[None]:
    """Enable autograd globally for one test and restore the previous state afterwards."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(previous)
