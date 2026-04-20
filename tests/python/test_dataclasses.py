"""Default values and round-trip of all configuration dataclasses."""

from __future__ import annotations

import poisson_cpp as pc


def test_sor_params_defaults_and_setters():
    p = pc.SORParams()
    assert p.tol > 0 and p.max_iter > 0
    p.omega = 1.7
    p.tol = 1e-6
    p.max_iter = 500
    assert (p.omega, p.tol, p.max_iter) == (1.7, 1e-6, 500)


def test_sor_report_repr_contains_fields():
    # Run a tiny solve to get a real SORReport.
    import numpy as np
    g = pc.Grid2D(1.0, 1.0, 8, 8)
    sor = pc.Solver2D(g, eps=1.0, uL=0.0, uR=1.0)
    _, rep = sor.solve(np.zeros((8, 8)), tol=1e-3)
    s = repr(rep)
    assert "SORReport" in s and "iterations" in s and "residual" in s


def test_cg_params_defaults_and_setters():
    p = pc.CGParams()
    assert p.tol > 0 and p.max_iter > 0
    p.tol = 1e-9
    p.max_iter = 1000
    assert p.tol == 1e-9 and p.max_iter == 1000


def test_cg_report_attrs():
    import numpy as np
    g = pc.Grid2D(1.0, 1.0, 8, 8)
    V = np.zeros((8, 8), order="F")
    rep, _ = pc.solve_poisson_cg(V, np.zeros((8, 8), order="F"), g,
                                 uL=0.0, uR=1.0, tol=1e-6)
    assert isinstance(rep.iterations, int)
    assert rep.iterations >= 0
    assert rep.residual >= 0.0


def test_amr_sor_params_defaults_and_setters():
    p = pc.AMRSORParams()
    assert 1.0 < p.omega < 2.0
    assert p.tol > 0 and p.max_iter > 0
    assert p.eps0 == 1.0
    p.omega = 1.5; p.tol = 1e-5; p.max_iter = 100; p.eps0 = 8.85e-12
    assert (p.omega, p.tol, p.max_iter, p.eps0) == (
        1.5, 1e-5, 100, 8.85e-12,
    )


def test_amr_sor_report_repr():
    import numpy as np
    tree = pc.Quadtree(1.0, level_min=2)
    arr = pc.extract_arrays(tree)
    rep = pc.amr_sor(arr, tol=1e-3, max_iter=10)
    s = repr(rep)
    assert s.startswith("AMRSORReport(") and "iterations=" in s


def test_composite_params_setters():
    p = pc.CompositeParams()
    p.n_pre = 5; p.n_post = 7; p.n_coarse_cycles = 3; p.omega = 1.7
    p.eps0 = 2.0
    assert p.n_pre == 5 and p.n_post == 7
    assert p.n_coarse_cycles == 3 and p.omega == 1.7 and p.eps0 == 2.0


def test_cell_default_constructor():
    c = pc.Cell()
    assert c.V == 0.0 and c.rho == 0.0
    c.V = 1.5; c.rho = -2.0
    assert c.V == 1.5 and c.rho == -2.0
