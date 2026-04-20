"""1D Poisson solvers (uniform and dielectric)."""

from __future__ import annotations

import numpy as np

import poisson_cpp as pc


def test_solve_poisson_1d_linear_ramp():
    # Pure-Dirichlet, no source: V should be the analytical linear ramp.
    N, L, uL, uR = 100, 1.0, 10.0, 0.0
    g = pc.Grid1D(L, N)
    V = pc.solve_poisson_1d(np.zeros(N), uL, uR, g)
    x = np.array([g.x(i) for i in range(N)])
    V_ref = uL + (uR - uL) * x / L
    assert V[0] == uL and V[-1] == uR
    np.testing.assert_allclose(V, V_ref, atol=1e-12)


def test_solve_poisson_1d_dielectric_d_continuity():
    # Three-layer dielectric: D = eps0 eps_r dV/dx must be constant.
    N, L, uL, uR, eps0 = 200, 1.0, 15.0, 0.0, 1.0
    g = pc.Grid1D(L, N)
    x = np.array([g.x(i) for i in range(N)])
    eps_r = np.where(x < 0.3, 5.0, np.where(x < 0.7, 1.0, 2.0))
    V = pc.solve_poisson_1d_dielectric(np.zeros(N), eps_r, uL, uR, g, eps0)

    eps_face = 2 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])
    dx = L / (N - 1)
    E = -(V[1:] - V[:-1]) / dx
    D = eps0 * eps_face * E
    rel_var = (D.max() - D.min()) / abs(D.mean())
    assert rel_var < 1e-10
    assert V[0] == uL and V[-1] == uR


def test_solve_poisson_1d_dielectric_uniform_matches_simple():
    # eps_r = 1 everywhere: the dielectric solver must agree with the uniform one.
    N, L, uL, uR = 50, 1.0, 5.0, -3.0
    g = pc.Grid1D(L, N)
    rho = np.zeros(N); rho[N // 2] = 1.0
    V_uni = pc.solve_poisson_1d(rho, uL, uR, g)
    V_die = pc.solve_poisson_1d_dielectric(rho, np.ones(N), uL, uR, g, 1.0)
    np.testing.assert_allclose(V_uni, V_die, atol=1e-12)
