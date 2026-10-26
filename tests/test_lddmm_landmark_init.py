import numpy as np
import pytest

from smint.alignment import st_sm_registration as reg

needs_stalign = pytest.mark.skipif(not reg.stalign_available(), reason="needs STalign (numpy<2 environment)")


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(20000)) * 1500
    t = rng.random(20000) * 2 * np.pi
    st = np.c_[3000 + r * np.cos(t) * 1.3, 3000 + r * np.sin(t)]
    st = st[(np.sin(st[:, 0] / 300) + np.cos(st[:, 1] / 250)) > -0.6]
    th = np.deg2rad(25)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = st.mean(0)
    sm = (st - c) @ R.T * 1.15 + c + np.array([300.0, -200.0])
    idx = rng.choice(len(st), 8, replace=False)
    return st, sm, sm[idx][:, ::-1], st[idx][:, ::-1]


@needs_stalign
def test_landmarks_initialise_the_affine():
    st, sm, pts_sm, pts_st = _pair()
    X_st, Y_st, I_st, _ = reg.rasterize_coordinates(st[:, 0], st[:, 1], dx=30)
    X_sm, Y_sm, I_sm, _ = reg.rasterize_coordinates(sm[:, 0], sm[:, 1], dx=30)
    errors = {}
    for init in (False, True):
        out = reg.run_lddmm_alignment([Y_sm, X_sm], I_sm, [Y_st, X_st], I_st,
                                      points_source=pts_sm, points_target=pts_st,
                                      niter=50, device="cpu", init_from_landmarks=init)
        moved = reg.transform_points(out, sm[:, 0], sm[:, 1], orientation="stalign")
        errors[init] = np.median(np.linalg.norm(moved - st, axis=1))
    assert errors[True] < 5.0            # starts ~900 um off
    assert errors[False] > 10 * errors[True]


def test_orientation_is_saved_next_to_the_csv(tmp_path):
    import json
    import pandas as pd
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    reg.save_transformed_data(df, np.array([[5.0, 6.0], [7.0, 8.0]]),
                              output_path=str(tmp_path / "out.csv"), orientation="stalign")
    meta = json.load(open(tmp_path / "out.csv.registration.json"))
    assert meta["orientation"] == "stalign"
