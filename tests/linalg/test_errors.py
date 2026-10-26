"""Argument checks of the dense helpers in ``hiepy.linalg`` (each check raises ``ValueError``)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import hiepy
from hiepy.linalg.diagonal_tensor import diagonal_tensor

T = torch.zeros(2, 3, 4, dtype=torch.float64)

CASES: dict[str, Callable[[], object]] = {
    "matricize: mode out of range": lambda: hiepy.matricize(T, (3,)),
    "matricize: duplicate modes": lambda: hiepy.matricize(T, (0, -3)),
    "dematricize: A not 2-d": lambda: hiepy.dematricize(T, (2, 3, 4), (0,)),
    "dematricize: non-positive shape": lambda: hiepy.dematricize(torch.zeros(2, 0), (2, 0), (0,)),
    "dematricize: no modes": lambda: hiepy.dematricize(torch.zeros(24, 1), (2, 3, 4), ()),
    "dematricize: mode out of range": lambda: hiepy.dematricize(torch.zeros(2, 12), (2, 3, 4), (3,)),
    "dematricize: duplicate modes": lambda: hiepy.dematricize(torch.zeros(4, 6), (2, 3, 4), (0, -3)),
    "dematricize: size mismatch": lambda: hiepy.dematricize(torch.zeros(2, 11), (2, 3, 4), (0,)),
    "mode_mul_torch: A not 2-d": lambda: hiepy.mode_mul_torch(T, torch.zeros(3), 1),
    "mode_mul_torch: mode out of range": lambda: hiepy.mode_mul_torch(T, torch.zeros(2, 3), 3),
    "mode_mul_torch: duplicate modes": lambda: hiepy.mode_mul_torch(T, torch.zeros(2, 4), (0, -3)),
    "mode_mul_torch: size mismatch": lambda: hiepy.mode_mul_torch(T, torch.zeros(2, 5), 1),
    "multi_mode_mul_torch: mode out of range": lambda: hiepy.multi_mode_mul_torch(T, {3: torch.zeros(2, 3)}),
    "multi_mode_mul_torch: same mode twice": lambda: hiepy.multi_mode_mul_torch(
        T, {1: torch.zeros(2, 3), -2: torch.zeros(2, 3)}
    ),
    "multi_mode_mul_torch: A not 2-d": lambda: hiepy.multi_mode_mul_torch(T, {1: torch.zeros(3)}),
    "multi_mode_mul_torch: size mismatch": lambda: hiepy.multi_mode_mul_torch(T, {1: torch.zeros(2, 5)}),
    "devectorize: x not 1-d": lambda: hiepy.devectorize(torch.zeros(4, 6), (2, 3, 4)),
    "devectorize: size mismatch": lambda: hiepy.devectorize(torch.zeros(23), (2, 3, 4)),
    "diagonal_tensor: order < 2": lambda: diagonal_tensor((3,), fill_value=1.0),
    "diagonal_tensor: unequal sizes": lambda: diagonal_tensor((3, 4), fill_value=1.0),
}


@pytest.mark.parametrize("name", CASES)
def test_invalid_arguments_raise(name: str):
    with pytest.raises(ValueError):
        CASES[name]()
