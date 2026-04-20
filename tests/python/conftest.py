"""Shared pytest fixtures and skip markers for the poisson_cpp test suite."""

from __future__ import annotations

import pytest

import poisson_cpp as pc


needs_fftw = pytest.mark.skipif(
    not pc.has_fftw3, reason="poisson_cpp built without FFTW3"
)
