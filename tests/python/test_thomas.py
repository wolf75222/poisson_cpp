"""Smoke tests for the tridiagonal Thomas solver."""

from __future__ import annotations

import numpy as np
import pytest

import poisson_cpp as pc


def test_thomas_identity():
    N = 50
    a = np.zeros(N)
    b = np.ones(N)
    c = np.zeros(N)
    d = np.arange(N, dtype=float)
    x = pc.thomas(a, b, c, d)
    np.testing.assert_allclose(x, d)


def test_thomas_laplacian_constant_rhs():
    # -V''(x) = 1 on N nodes with V[0] = V[N-1] = 0 (rhs already includes BCs).
    N = 100
    b = 2.0 * np.ones(N)
    a = np.zeros(N); a[1:] = -1.0
    c = np.zeros(N); c[:-1] = -1.0
    d = np.ones(N)
    x = pc.thomas(a, b, c, d)
    assert x.shape == (N,)
    # Solution is positive, symmetric.
    assert (x > 0).all()
    np.testing.assert_allclose(x, x[::-1], atol=1e-12)


def test_thomas_zero_pivot_raises():
    N = 10
    b = np.zeros(N)
    a = np.zeros(N)
    c = np.zeros(N)
    d = np.ones(N)
    with pytest.raises(RuntimeError):
        pc.thomas(a, b, c, d)
