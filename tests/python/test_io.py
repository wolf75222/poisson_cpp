"""dump_amr_snapshot helper."""

from __future__ import annotations

import json

import poisson_cpp as pc


def test_dump_amr_snapshot_round_trip(tmp_path):
    tree = pc.Quadtree(1.0, level_min=2)
    tree.refine(pc.make_key(2, 1, 1))
    arr = pc.extract_arrays(tree)
    pc.amr_sor(arr, tol=1e-3, max_iter=10)
    pc.writeback(tree, arr.keys, arr.V)

    out = pc.dump_amr_snapshot(tree, tmp_path / "snap.json",
                               extra={"sigma": 0.04, "note": "hello"})
    assert out.exists()
    data = json.loads(out.read_text())

    assert data["L"] == 1.0
    assert data["level_min"] == 2
    assert data["num_leaves"] == tree.num_leaves()
    assert len(data["cells"]) == data["num_leaves"]
    assert data["sigma"] == 0.04
    assert data["note"] == "hello"

    sample = data["cells"][0]
    assert set(sample.keys()) == {"key", "level", "x", "y", "h", "V", "rho"}


def test_dump_amr_snapshot_geometry_consistent(tmp_path):
    tree = pc.Quadtree(2.0, level_min=2)
    out = pc.dump_amr_snapshot(tree, tmp_path / "snap.json")
    data = json.loads(out.read_text())
    for c in data["cells"]:
        h_expected = 2.0 / (1 << c["level"])
        assert abs(c["h"] - h_expected) < 1e-12
        assert 0.0 <= c["x"] <= 2.0
        assert 0.0 <= c["y"] <= 2.0
