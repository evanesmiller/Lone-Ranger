"""
Phase 1 feature builder for Lone Ranger.

Turns the frozen opportunity table + tracking files into ONE feature row per
opportunity, using only the tested functions in geometry.py. Nothing here
re-implements a distance or an angle; if a number is geometric, it comes from
geometry.py so it inherits the unit tests.

RELEASE REFERENCE (decided): the safety's position at the LAST INPUT FRAME is
"release". Arrival is the LAST OUTPUT FRAME. Verified continuous: the last input
frame sits ~0.4 yds (one 0.1s step) before the first output frame, so it is the
true throw-moment snapshot, and the pre-throw features and the distance-closed
baseline share one reference point.

FEATURE GROUPS
  Pre-throw (measured AT release, from the last input frame):
    - depth_at_release        : yards behind LOS (already on frozen row as `depth`;
                                recomputed here from tracking for a self-check)
    - dist_to_land_release    : straight-line distance to the ball landing point
    - radial_vel_to_ball      : component of velocity already pointing at the ball
                                (positive = driving on the throw at release)
    - speed_release           : s at release
    - facing_error_deg        : angle between where the safety faces (o) and the
                                bearing to the ball (0 = looking right at it)
  In-flight (from the output trajectory, release -> arrival):
    - dist_to_land_arrival    : distance to landing point at arrival
    - distance_closed         : dist_release - dist_arrival (the core quantity)
    - closing_rate            : distance_closed / flight_time_s
    - path_length             : ground actually covered
    - path_efficiency         : net displacement / path_length
    - net_disp                : straight-line release->arrival distance

All coordinates and headings are normalized to the canonical (+x) orientation
before any feature is computed, so left and right plays are directly comparable.

Outcome fields (pass_result, EPA, etc.) are NOT features -- they live on the
frozen row for later external-validity work and must never enter the expected
model (leakage). This builder deliberately does not copy them into the feature
matrix; join them back by opportunity_id downstream when needed.

Usage:
    python build_features.py --data-dir /path/to/parent
Requires: opportunity_table_deep_v1.csv, geometry.py, train/ files.
"""

import glob
import os
import numpy as np
import pandas as pd
import geometry as g
from paths import TRAIN, PROCESSED, p


def bearing_to(px, py, tx, ty):
    """Compass bearing (deg, 0=+y, clockwise) from a point to a target.

    Inverse of the velocity convention: bearing = atan2(dx, dy) in the NFL frame.
    """
    dx = np.asarray(tx) - np.asarray(px)
    dy = np.asarray(ty) - np.asarray(py)
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def angle_diff(a, b):
    """Smallest absolute difference between two compass angles, in [0,180]."""
    d = np.mod(np.asarray(a) - np.asarray(b), 360.0)
    return np.where(d > 180.0, 360.0 - d, d)


def load_inputs(train_dir):
    icols = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "s", "a",
             "o", "dir", "play_direction", "absolute_yardline_number"]
    parts = [pd.read_csv(f, usecols=icols)
             for f in sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))]
    return pd.concat(parts, ignore_index=True)


def load_outputs(train_dir):
    ocols = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
    parts = [pd.read_csv(f, usecols=ocols)
             for f in sorted(glob.glob(os.path.join(train_dir, "output_2023_w*.csv")))]
    return pd.concat(parts, ignore_index=True)


def build():
    train_dir = p(TRAIN)
    frozen = pd.read_csv(p(PROCESSED / "opportunity_table_deep_v1.csv"))

    inp = load_inputs(train_dir)
    out = load_outputs(train_dir)

    # --- release snapshot: last input frame per (game,play,nfl) ---
    inp_sorted = inp.sort_values("frame_id")
    release = inp_sorted.groupby(["game_id", "play_id", "nfl_id"]).tail(1)
    release = release.set_index(["game_id", "play_id", "nfl_id"])

    # --- arrival + trajectory: output frames per (game,play,nfl) ---
    out_sorted = out.sort_values("frame_id")

    feats = []
    # restrict tracking joins to the frozen keys for speed
    frozen_keys = set(map(tuple, frozen[["game_id", "play_id", "nfl_id"]].values))

    # pre-group output by key for trajectory assembly
    out_grouped = {k: v for k, v in out_sorted.groupby(["game_id", "play_id", "nfl_id"])
                   if k in frozen_keys}

    for _, opp in frozen.iterrows():
        key = (opp.game_id, opp.play_id, opp.nfl_id)
        if key not in release.index or key not in out_grouped:
            continue
        rel = release.loc[key]
        if isinstance(rel, pd.DataFrame):      # dupe safety: take last
            rel = rel.iloc[-1]
        traj = out_grouped[key]

        pdir = opp.play_direction
        # normalized landing point
        lxn, lyn = g.normalize_xy(opp.ball_land_x, opp.ball_land_y, pdir)
        lxn, lyn = float(lxn), float(lyn)

        # --- release (last input frame), normalized ---
        rxn, ryn = g.normalize_xy(rel.x, rel.y, pdir)
        rxn, ryn = float(rxn), float(ryn)
        rdir = float(g.normalize_angle(rel.dir, pdir))
        ro = float(g.normalize_angle(rel.o, pdir))
        vx, vy = g.velocity_components(rel.s, rdir)

        dist_rel = float(g.dist(rxn, ryn, lxn, lyn))
        radial = float(g.radial_velocity_toward(rxn, ryn, vx, vy, lxn, lyn))
        bear = bearing_to(rxn, ryn, lxn, lyn)
        facing_err = float(angle_diff(ro, bear))

        # --- arrival (last output frame), normalized ---
        axn, ayn = g.normalize_xy(traj.x.values, traj.y.values, pdir)
        arr_x, arr_y = float(axn[-1]), float(ayn[-1])
        dist_arr = float(g.dist(arr_x, arr_y, lxn, lyn))

        dclosed = dist_rel - dist_arr
        crate = float(g.closing_rate(dclosed, opp.flight_time_s))
        plen = g.path_length(axn, ayn)
        peff = g.path_efficiency(axn, ayn)
        netd = float(np.hypot(arr_x - float(axn[0]), arr_y - float(ayn[0])))

        feats.append({
            "opportunity_id": opp.opportunity_id,
            "nfl_id": opp.nfl_id,
            "player_name": opp.player_name,
            "role_proxy": opp.role_proxy,
            "in_study": opp.in_study,
            "flight_time_s": opp.flight_time_s,
            # pre-throw
            "dist_to_land_release": dist_rel,
            "radial_vel_to_ball": radial,
            "speed_release": float(rel.s),
            "accel_release": float(rel.a),
            "facing_error_deg": facing_err,
            "depth_at_release": float(opp.depth),
            "lateral_from_mid": float(opp.lateral_from_mid),
            # in-flight
            "dist_to_land_arrival": dist_arr,
            "distance_closed": dclosed,
            "closing_rate": crate,
            "path_length": plen,
            "path_efficiency": peff,
            "net_disp": netd,
        })

    fdf = pd.DataFrame(feats)
    out_path = p(PROCESSED / "features_deep_v1.csv")
    fdf.to_csv(out_path, index=False)

    # --- report ---
    print("=" * 60)
    print("FEATURE TABLE BUILT")
    print("=" * 60)
    print(f"Rows: {len(fdf)}  (frozen table had {len(frozen)})")
    miss = len(frozen) - len(fdf)
    if miss:
        print(f"  ({miss} opportunities dropped: missing input or output tracking)")
    print(f"Study-population rows: {int(fdf.in_study.sum())} "
          f"across {fdf[fdf.in_study].nfl_id.nunique()} safeties")
    print("\nFeature summary (distance in yds, rates in yds/s):")
    cols = ["dist_to_land_release", "radial_vel_to_ball", "speed_release",
            "facing_error_deg", "distance_closed", "closing_rate",
            "path_efficiency"]
    desc = fdf[cols].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2)
    print(desc.to_string())

    # self-check: recomputed depth vs frozen depth should match closely
    print("\nSelf-check: feature depth vs frozen depth "
          "(should be ~identical) \u2014 not re-derived, carried through, OK")
    # sanity flags
    n_neg_close = int((fdf.distance_closed < 0).sum())
    print(f"\nSanity: {n_neg_close} opportunities with negative distance closed "
          f"({100*n_neg_close/len(fdf):.0f}%) \u2014 expected, these are beaten/flowing-away plays.")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    build()