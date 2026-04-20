"""Utility helpers: harmonic_mean and Morton key (de)encoding."""

from __future__ import annotations

import numpy as np

import poisson_cpp as pc


def test_harmonic_mean_equal_inputs():
    a = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(pc.harmonic_mean(a, a), a)


def test_harmonic_mean_known_values():
    a = np.array([1.0, 4.0])
    b = np.array([3.0, 12.0])
    expected = 2 * a * b / (a + b)
    np.testing.assert_allclose(pc.harmonic_mean(a, b), expected)


def test_morton_encode_decode():
    for level in (0, 1, 5, 10):
        for i in (0, 1, 7, 100, (1 << level) - 1 if level >= 7 else 0):
            for j in (0, 3, 50):
                k = pc.make_key(level, i, j)
                assert pc.level_of(k) == level
                assert pc.i_of(k) == i
                assert pc.j_of(k) == j


def test_fftw_install_hint_returns_string():
    # Always available in the package shim.
    s = pc.fftw_install_hint()
    assert isinstance(s, str) and len(s) > 0
    assert "pip install" in s
