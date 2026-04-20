"""Grid1D and Grid2D properties."""

from __future__ import annotations

import numpy as np

import poisson_cpp as pc


def test_grid1d_basic():
    g = pc.Grid1D(2.0, 11)
    assert g.N == 11
    assert g.L == 2.0
    assert g.dx == 0.2
    x = np.array([g.x(i) for i in range(g.N)])
    np.testing.assert_allclose(x, np.linspace(0.0, 2.0, 11))


def test_grid2d_basic():
    g = pc.Grid2D(1.0, 0.5, 8, 4)
    assert g.Nx == 8
    assert g.Ny == 4
    assert g.dx == 1.0 / 8
    assert g.dy == 0.5 / 4
