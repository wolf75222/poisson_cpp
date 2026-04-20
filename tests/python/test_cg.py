"""Conjugate Gradient Poisson solver."""

from __future__ import annotations

import numpy as np

import poisson_cpp as pc


def test_cg_linear_ramp():
    N, uL, uR = 32, 0.0, 10.0
    g = pc.Grid2D(1.0, 1.0, N, N)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F")
    rep, hist = pc.solve_poisson_cg(V, rho, g, uL=uL, uR=uR,
                                    tol=1e-10, max_iter=2000)
    assert rep.iterations > 0
    assert rep.residual < 1e-9
    assert hist == []  # record_history default False

    xc = (np.arange(N) + 0.5) / N
    ramp = uL + (uR - uL) * xc
    np.testing.assert_allclose(V, ramp[:, None] * np.ones((1, N)), atol=1e-6)


def test_cg_history_recorded():
    N = 16
    g = pc.Grid2D(1.0, 1.0, N, N)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F"); rho[N // 2, N // 2] = 1.0
    rep, hist = pc.solve_poisson_cg(V, rho, g, tol=1e-8, max_iter=500,
                                    record_history=True)
    # History may include the initial residual on top of per-iteration values.
    assert abs(len(hist) - rep.iterations) <= 1
    assert hist[0] >= hist[-1]  # residual decreases on average


def test_pcg_jacobi_runs():
    N = 16
    g = pc.Grid2D(1.0, 1.0, N, N)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F")
    rep, _ = pc.solve_poisson_cg(V, rho, g, uL=0.0, uR=1.0,
                                 tol=1e-8, use_preconditioner=True,
                                 max_iter=2000)
    assert rep.residual < 1e-7
