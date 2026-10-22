"""Shared fixtures and hypothesis profiles for the test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch
from hypothesis import settings

# hypothesis profiles: "dev" for local runs, "ci" (selected via HYPOTHESIS_PROFILE in ci.yml) is thorough and
# reproducible. No deadline: the first call of a torch kernel can be slow.
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def grad_enabled() -> Iterator[None]:
    """Enable autograd globally for one test and restore the previous state afterwards."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(previous)
