"""Multigrid building blocks and full V-cycles."""

from __future__ import annotations

import math

import numpy as np

import poisson_cpp as pc


def test_gs_smooth_in_place():
    N = 32
    h = 1.0 / N
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F"); rho[N // 2, N // 2] = 1.0
    pc.gs_smooth(V, rho, h, n_iter=10)
    assert V.max() > 0.0  # mutated


def test_laplacian_fv_zero_on_zero():
    N = 16
    V = np.zeros((N, N))
    AV = pc.laplacian_fv(V, h=1.0 / N)
    assert AV.shape == (N, N)
    np.testing.assert_array_equal(AV, 0)


def test_restrict_avg_halves_resolution():
    r = np.ones((8, 8))
    rc = pc.restrict_avg(r)
    assert rc.shape == (4, 4)
    np.testing.assert_allclose(rc, 1.0)


def test_prolongate_const_doubles_resolution():
    c = np.arange(16, dtype=float).reshape(4, 4)
    f = pc.prolongate_const(c)
    assert f.shape == (8, 8)


def test_prolongate_bilinear_exact_on_interior_constant():
    # Bilinear assumes Dirichlet-zero ghost cells, so it is only exact on
    # constants in the interior (away from the boundary).
    c = 3.7 * np.ones((6, 6))
    f = pc.prolongate_bilinear(c)
    assert f.shape == (12, 12)
    np.testing.assert_allclose(f[3:-3, 3:-3], 3.7, atol=1e-12)


def test_vcycle_uniform_drops_residual():
    # Manufactured V = sin(pi x) sin(pi y), rho = 2 pi^2 V on
    # cell-centered grid. V-cycle should reduce residual significantly.
    N = 32
    h = 1.0 / N
    xc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    V_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
    rho = 2 * (math.pi ** 2) * V_exact

    V = np.zeros((N, N))
    r0 = float(np.max(np.abs(rho - pc.laplacian_fv(V, h))))
    for _ in range(5):
        V = pc.vcycle_uniform(V, rho, h, n_pre=2, n_post=2, n_min=4)
    r1 = float(np.max(np.abs(rho - pc.laplacian_fv(V, h))))
    assert r1 < 0.1 * r0


def test_vcycle_amr_composite_drops_residual():
    L, sigma = 1.0, 0.05

    def predicate(key):
        lvl = pc.level_of(key)
        if lvl >= 6:
            return False
        h = L / (1 << lvl)
        cx = (pc.i_of(key) + 0.5) * h
        cy = (pc.j_of(key) + 0.5) * h
        return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < (4 * sigma) ** 2

    def rho_func(x, y):
        return math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2)
                        / (2 * sigma ** 2))

    tree = pc.Quadtree(L, level_min=4)
    tree.build(predicate, level_max=6, rho_func=rho_func)
    arr = pc.extract_arrays(tree)

    r0 = float(np.max(np.abs(pc.amr_residual(arr))))
    for _ in range(8):
        pc.vcycle_amr_composite(arr, tree)
    r1 = float(np.max(np.abs(pc.amr_residual(arr))))
    # Composite V-cycle without Galerkin coarsening reduces residual by
    # a factor ~0.7 per cycle; require at least 2x reduction over 8 cycles.
    assert r1 < 0.5 * r0


def test_composite_params_defaults():
    p = pc.CompositeParams()
    assert p.n_pre > 0 and p.n_post > 0
    assert p.n_coarse_cycles > 0
    assert 1.0 < p.omega < 2.0
    assert p.eps0 == 1.0
