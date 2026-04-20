"""Spectral DST-I solver (FFTW)."""

from __future__ import annotations

import math

import numpy as np

import poisson_cpp as pc

from conftest import needs_fftw


@needs_fftw
def test_dst1d_round_trip():
    N, L = 31, 1.0
    dst = pc.DSTSolver1D(N, L, eps0=1.0)
    rho = np.random.default_rng(0).standard_normal(N)
    V = dst.solve(rho)
    assert V.shape == (N,)
    assert np.isfinite(V).all()


@needs_fftw
def test_dst2d_manufactured_solution():
    N, L = 31, 1.0
    a = math.pi / L
    h = L / (N + 1)
    i = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(i, i, indexing="ij")
    V_exact = np.sin(a * X) * np.sin(a * Y)
    rho = 2 * a * a * V_exact

    dst = pc.DSTSolver2D(N, N, L, L, eps0=1.0)
    V_num = dst.solve(rho)
    err = np.max(np.abs(V_num - V_exact))
    # Continuous-source error scales as h^2.
    assert err < 1e-2


@needs_fftw
def test_dst2d_homogeneous_dirichlet_implicit():
    # Even with non-zero source, the boundary stays at 0 implicitly.
    N, L = 15, 1.0
    dst = pc.DSTSolver2D(N, N, L, L, eps0=1.0)
    V = dst.solve(np.ones((N, N)))
    assert V.shape == (N, N)
