"""
Unit tests for geometry.py. Run:  python -m pytest test_geometry.py -v
or standalone:  python test_geometry.py

These test the invariants that every downstream metric silently relies on. If any
fail, do NOT build metrics on top -- a geometry error here contaminates everything.
"""

import numpy as np
import src.geometry as g


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# --- distance: hand-computed 3-4-5 triangle --------------------------------
def test_dist_345():
    assert approx(g.dist(0, 0, 3, 4), 5.0)


def test_dist_zero():
    assert approx(g.dist(10, 20, 10, 20), 0.0)


# --- distance closed: signs and magnitude ----------------------------------
def test_distance_closed_positive():
    # safety starts 10 yds from land, ends 3 yds from land -> closed 7
    dc = g.distance_closed(release_x=0, release_y=0, arrival_x=7, arrival_y=0,
                           land_x=10, land_y=0)
    assert approx(dc, 7.0)


def test_distance_closed_negative():
    # safety moves AWAY from the ball -> negative closed
    dc = g.distance_closed(release_x=7, release_y=0, arrival_x=0, arrival_y=0,
                           land_x=10, land_y=0)
    assert approx(dc, -7.0)


# --- closing rate -----------------------------------------------------------
def test_closing_rate():
    assert approx(g.closing_rate(7.0, 1.4), 5.0)


def test_closing_rate_zero_flight_is_nan():
    assert np.isnan(g.closing_rate(7.0, 0.0))


# --- path length & efficiency ----------------------------------------------
def test_path_length_straight():
    xs = [0, 1, 2, 3]; ys = [0, 0, 0, 0]
    assert approx(g.path_length(xs, ys), 3.0)


def test_path_efficiency_straight_is_one():
    xs = [0, 1, 2, 3]; ys = [0, 0, 0, 0]
    assert approx(g.path_efficiency(xs, ys), 1.0)


def test_path_efficiency_L_shape():
    # go right 3, then up 4: ground covered 7, net displacement 5 -> 5/7
    xs = [0, 3, 3]; ys = [0, 0, 4]
    assert approx(g.path_efficiency(xs, ys), 5.0 / 7.0)


def test_path_efficiency_no_move_is_nan():
    assert np.isnan(g.path_efficiency(np.array([2.0, 2.0]), np.array([2.0, 2.0])))


# --- normalization invariants (the important ones) -------------------------
def test_normalize_right_unchanged():
    xn, yn = g.normalize_xy([50.0], [20.0], ["right"])
    assert approx(xn[0], 50.0) and approx(yn[0], 20.0)


def test_normalize_left_flips():
    xn, yn = g.normalize_xy([50.0], [20.0], ["left"])
    assert approx(xn[0], g.FIELD_X - 50.0) and approx(yn[0], g.FIELD_Y - 20.0)


def test_normalization_preserves_distance():
    # THE KEY INVARIANT: flipping a play must not change the distance between
    # the safety and the ball landing point. If this fails, distance-closed is
    # meaningless after normalization.
    sx, sy, lx, ly = 40.0, 15.0, 62.0, 30.0
    d_before = g.dist(sx, sy, lx, ly)
    sxn, syn = g.normalize_xy([sx], [sy], ["left"])
    lxn, lyn = g.normalize_xy([lx], [ly], ["left"])
    d_after = g.dist(sxn[0], syn[0], lxn[0], lyn[0])
    assert approx(d_before, d_after)


def test_target_relative_origin():
    # a point AT the landing spot maps to (0,0) in target-relative coords
    rx, ry = g.to_target_relative(62.0, 30.0, 62.0, 30.0)
    assert approx(rx, 0.0) and approx(ry, 0.0)


# --- angle convention: THE compass-vs-math risk ----------------------------
# NFL diagram: 0deg = +y, 90deg = +x, increasing clockwise.
def test_velocity_0deg_is_plus_y():
    vx, vy = g.velocity_components(5.0, 0.0)
    assert approx(vx, 0.0) and approx(vy, 5.0)


def test_velocity_90deg_is_plus_x():
    vx, vy = g.velocity_components(5.0, 90.0)
    assert approx(vx, 5.0) and approx(vy, 0.0)


def test_velocity_180deg_is_minus_y():
    vx, vy = g.velocity_components(5.0, 180.0)
    assert approx(vx, 0.0) and approx(vy, -5.0)


def test_velocity_270deg_is_minus_x():
    vx, vy = g.velocity_components(5.0, 270.0)
    assert approx(vx, -5.0) and approx(vy, 0.0)


def test_velocity_speed_preserved():
    # magnitude of (vx,vy) must equal speed for any heading
    for d in [17.0, 133.0, 201.0, 349.0]:
        vx, vy = g.velocity_components(6.3, d)
        assert approx(np.hypot(vx, vy), 6.3, tol=1e-6)


# --- angle normalization matches the position flip -------------------------
def test_normalize_angle_right_unchanged():
    assert approx(g.normalize_angle(30.0, "right"), 30.0)


def test_normalize_angle_left_rotates_180():
    assert approx(g.normalize_angle(30.0, "left"), 210.0)


def test_normalize_angle_left_wraps():
    # 200 + 180 = 380 -> 20
    assert approx(g.normalize_angle(200.0, "left"), 20.0)


def test_normalize_angle_consistent_with_position_flip():
    # THE INVARIANT: a velocity built from a flipped heading must equal the
    # flipped velocity built from the original heading. If this holds, direction
    # features are consistent across left/right plays.
    s, d = 4.0, 55.0
    # path A: normalize the angle, then build velocity
    dA = g.normalize_angle(d, "left")
    vxA, vyA = g.velocity_components(s, dA)
    # path B: build velocity in raw frame, then flip the vector 180 (negate both)
    vxB, vyB = g.velocity_components(s, d)
    vxB, vyB = -vxB, -vyB
    assert approx(vxA, vxB, tol=1e-6) and approx(vyA, vyB, tol=1e-6)


# --- radial velocity toward a target ---------------------------------------
def test_radial_velocity_straight_at_target():
    # at (0,0), moving +x at 5, target straight ahead on +x -> full closing 5
    rv = g.radial_velocity_toward(0, 0, 5.0, 0.0, 10.0, 0.0)
    assert approx(rv, 5.0)


def test_radial_velocity_away_is_negative():
    rv = g.radial_velocity_toward(0, 0, -5.0, 0.0, 10.0, 0.0)
    assert approx(rv, -5.0)


def test_radial_velocity_perpendicular_is_zero():
    # moving +y while target is on +x -> no closing component
    rv = g.radial_velocity_toward(0, 0, 0.0, 5.0, 10.0, 0.0)
    assert approx(rv, 0.0)


def test_radial_velocity_on_target_is_nan():
    assert np.isnan(g.radial_velocity_toward(10, 10, 3.0, 3.0, 10.0, 10.0))


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
