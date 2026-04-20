"""2D SOR solver."""

from __future__ import annotations

import numpy as np

import poisson_cpp as pc


def test_solver2d_linear_ramp():
    # Dirichlet x, Neumann y, no source -> linear ramp in x.
    N, uL, uR = 32, 0.0, 10.0
    g = pc.Grid2D(1.0, 1.0, N, N)
    sor = pc.Solver2D(g, eps=1.0, uL=uL, uR=uR)
    V, rep = sor.solve(np.zeros((N, N)), tol=1e-10, max_iter=20_000)
    assert rep.iterations > 0
    assert rep.residual < 1e-9

    xc = (np.arange(N) + 0.5) / N
    ramp = uL + (uR - uL) * xc
    np.testing.assert_allclose(V, ramp[:, None] * np.ones((1, N)), atol=1e-7)
    # y-invariant.
    assert V.std(axis=1).max() < 1e-10


def test_solver2d_inplace_overwrites():
    N = 16
    g = pc.Grid2D(1.0, 1.0, N, N)
    sor = pc.Solver2D(g, eps=1.0, uL=0.0, uR=1.0)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F")
    rep = sor.solve_inplace(V, rho, tol=1e-8, max_iter=5000)
    assert rep.residual < 1e-7
    assert V.max() > 0.0  # was modified


def test_solver2d_spatial_eps_constructor():
    N = 16
    g = pc.Grid2D(1.0, 1.0, N, N)
    eps = np.ones((N, N))
    sor = pc.Solver2D(g, eps, uL=0.0, uR=1.0)
    V, _ = sor.solve(np.zeros((N, N)), tol=1e-7)
    assert V.shape == (N, N)
