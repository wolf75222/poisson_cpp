"""AMR quadtree + heterogeneous-stencil SOR."""

from __future__ import annotations

import math

import numpy as np

import poisson_cpp as pc


def _gaussian_tree(L=1.0, sigma=0.05, level_min=4, level_max=7):
    def predicate(key):
        lvl = pc.level_of(key)
        if lvl >= level_max:
            return False
        h = L / (1 << lvl)
        cx = (pc.i_of(key) + 0.5) * h
        cy = (pc.j_of(key) + 0.5) * h
        return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < (4 * sigma) ** 2

    def rho_func(x, y):
        return math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2)
                        / (2 * sigma ** 2))

    tree = pc.Quadtree(L, level_min=level_min)
    tree.build(predicate, level_max=level_max, rho_func=rho_func)
    return tree


def test_quadtree_uniform_init():
    tree = pc.Quadtree(1.0, level_min=3)
    assert tree.num_leaves() == 64  # 2^3 x 2^3
    assert tree.L() == 1.0
    assert tree.level_min() == 3


def test_morton_round_trip():
    for level, i, j in [(0, 0, 0), (3, 5, 7), (10, 100, 200)]:
        k = pc.make_key(level, i, j)
        assert pc.level_of(k) == level
        assert pc.i_of(k) == i
        assert pc.j_of(k) == j


def test_quadtree_build_refines_around_source():
    tree = _gaussian_tree()
    n = tree.num_leaves()
    # Should have refined more than the uniform level_min grid.
    assert n > (1 << 4) ** 2
    # All leaves stay within [0, L].
    L = tree.L()
    for key in list(tree.leaves().keys())[:50]:
        cx, cy = tree.cell_center(key)
        assert 0.0 <= cx <= L and 0.0 <= cy <= L


def test_extract_arrays_shapes_match_leaves():
    tree = _gaussian_tree(level_max=6)
    arr = pc.extract_arrays(tree)
    n = tree.num_leaves()
    assert len(arr.keys) == n
    assert arr.V.shape == (n,)
    assert arr.rho.shape == (n,)
    assert arr.h.shape == (n,)
    assert arr.Vc.shape == (n,)
    assert arr.nb0.shape == (n, 4)
    assert arr.w0.shape == (n, 4)


def test_amr_sor_drives_residual_down():
    tree = _gaussian_tree(level_max=6)
    arr = pc.extract_arrays(tree)
    rep = pc.amr_sor(arr, omega=1.85, tol=1e-7, max_iter=2000)
    assert rep.iterations < 2000
    assert rep.residual < 1e-6
    # AMR residual on the converged solution must be small.
    r = pc.amr_residual(arr, eps0=1.0)
    assert np.max(np.abs(r)) < 1e-4


def test_writeback_updates_tree():
    tree = _gaussian_tree(level_max=5)
    arr = pc.extract_arrays(tree)
    pc.amr_sor(arr, omega=1.85, tol=1e-6, max_iter=2000)
    pc.writeback(tree, arr.keys, arr.V)
    cells = tree.leaves()
    assert any(c.V != 0.0 for c in list(cells.values())[:50])


def test_neighbour_leaves_at_boundary():
    tree = pc.Quadtree(1.0, level_min=2)
    # Bottom-left cell has no S/W neighbours.
    bl = pc.make_key(2, 0, 0)
    assert tree.is_leaf(bl)
    assert tree.neighbour_leaves(bl, pc.Direction.S) == []
    assert tree.neighbour_leaves(bl, pc.Direction.W) == []
    # E neighbour exists at the same level.
    east = tree.neighbour_leaves(bl, pc.Direction.E)
    assert len(east) == 1 and pc.level_of(east[0]) == 2


def test_quadtree_refine_creates_4_children():
    tree = pc.Quadtree(1.0, level_min=2)
    parent = pc.make_key(2, 1, 1)
    assert tree.is_leaf(parent)
    n_before = tree.num_leaves()
    tree.refine(parent)
    # Parent is no longer a leaf, 4 children take its place.
    assert not tree.is_leaf(parent)
    assert tree.num_leaves() == n_before - 1 + 4
    for ci in (0, 1):
        for cj in (0, 1):
            child = pc.make_key(3, 2 + ci, 2 + cj)
            assert tree.is_leaf(child)


def test_cell_size_halves_per_level():
    tree = pc.Quadtree(2.0, level_min=2)
    assert tree.cell_size(0) == 2.0
    assert tree.cell_size(1) == 1.0
    assert tree.cell_size(2) == 0.5
    assert tree.cell_size(5) == 2.0 / (1 << 5)


def test_cell_center_matches_morton_indices():
    tree = pc.Quadtree(1.0, level_min=3)
    for (lvl, i, j) in [(3, 0, 0), (3, 7, 7), (3, 4, 2)]:
        key = pc.make_key(lvl, i, j)
        cx, cy = tree.cell_center(key)
        h = 1.0 / (1 << lvl)
        assert abs(cx - (i + 0.5) * h) < 1e-12
        assert abs(cy - (j + 0.5) * h) < 1e-12


def test_balance_2to1_isolated():
    # Refine one corner leaf twice, leaving its same-level neighbour
    # at level 0; balance_2to1 must split that neighbour at least once
    # to keep level differences <= 1 across faces.
    tree = pc.Quadtree(1.0, level_min=1)
    bl = pc.make_key(1, 0, 0)
    tree.refine(bl)
    tree.refine(pc.make_key(2, 1, 1))
    # The east-face neighbour at level 1 is (1, 1, 0), still a leaf.
    east_l1 = pc.make_key(1, 1, 0)
    assert tree.is_leaf(east_l1)
    tree.balance_2to1()
    # After balance, that east neighbour has been refined.
    assert not tree.is_leaf(east_l1)
    # No leaf differs from a neighbour by more than one level.
    for key in tree.leaves().keys():
        lvl = pc.level_of(key)
        for d in (pc.Direction.N, pc.Direction.S, pc.Direction.E, pc.Direction.W):
            for nb in tree.neighbour_leaves(key, d):
                assert abs(pc.level_of(nb) - lvl) <= 1
