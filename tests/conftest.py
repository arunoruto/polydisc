"""Shared fixtures for the test suite.

Reference assertions live in :mod:`packing_asserts`, not here.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A seeded generator for tests that need auxiliary randomness."""
    return np.random.default_rng(20260902)
