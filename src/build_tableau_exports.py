"""
Build tidy, Tableau-ready export tables for the Lone Ranger dashboard.

Tableau prefers LONG/tidy data: one row per observation, categories as column
values (not spread across many columns). This script reshapes the finished
analysis into four clean tables under data/tableau/.

Outputs:
  1. tab_leaderboard.csv    -- per-safety ROE with CIs, ready for a dot-and-
                               interval plot. One row per safety.
  2. tab_stabilization.csv  -- reliability vs opportunity count. One row per k.
  3. tab_role_mix.csv       -- LONG format: one row per (safety, role) with the
                               fraction of that safety's snaps in that role.
  4. tab_trajectories.csv   -- per-frame safety paths for the field view, for a
                               sample of illustrative plays. One row per
                               (opportunity, frame) in NORMALIZED field coords,
                               plus the ball landing point on every row so
                               Tableau can draw it.

Run from project root:  python3 src/build_tableau_exports.py
"""

import os
import numpy as np
import pandas as pd
import geometry as g
from paths import PROCESSED, INTERIM, TRAIN, ROOT, p

TABLEAU = ROOT / "data" / "tableau"
TABLEAU.mkdir(parents=True, exist_ok=True)

N_SAMPLE_PLAYS = 60      # trajectories to export for the field view (keep it light)
SEED = 7


def build_leaderboard():
    cis = pd.read_csv(p(PROCESSED / "roe_with_cis_v1.csv"))
    cis = cis.sort_values("roe", ascending=False).reset_index(drop=True)
    cis["rank"] = cis.index + 1
    # a flag Tableau can color on: is the safety distinguishable from average?
    cis["distinguishable"] = np.where(
        (cis.ci_lo > 0) | (cis.ci_hi < 0), "Yes", "No")
    cis["above_or_below"] = np.where(cis.roe >= 0, "Above expected", "Below expected")
    cols = ["rank", "nfl_id", "player_name", "n_opps", "roe", "ci_lo", "ci_hi",
            "se", "distinguishable", "above_or_below"]
    cis[cols].to_csv(p(TABLEAU / "tab_leaderboard.csv"), index=False)
    return len(cis)


def build_stabilization():
    s = pd.read_csv(p(PROCESSED / "stabilization_curve_v1.csv"))
    s.to_csv(p(TABLEAU / "tab_stabilization.csv"), index=False)
    return len(s)


def build_role_mix():
    # role mix from the interim file if present, else derive from opportunities
    src = INTERIM / "safety_role_mix.csv"
    if src.exists():
        mix = pd.read_csv(p(src))
        id_cols = [c for c in ["nfl_id", "player_name"] if c in mix.columns]
        role_cols = [c for c in mix.columns
                     if c in ("SINGLE_HIGH", "SPLIT_DEEP", "BOX_UNDERNEATH")]
        long = mix.melt(id_vars=id_cols, value_vars=role_cols,
                        var_name="role", value_name="fraction")
    else:
        opp = pd.read_csv(p(PROCESSED / "opportunity_table_deep_v1.csv"))
        long = (opp.groupby(["nfl_id", "player_name", "role_proxy"]).size()
                / opp.groupby(["nfl_id", "player_name"]).size()).reset_index(
                    name="fraction").rename(columns={"role_proxy": "role"})
    # keep only reported safeties for the dashboard
    rep_ids = set(pd.read_csv(p(PROCESSED / "roe_with_cis_v1.csv")).nfl_id)
    long = long[long.nfl_id.isin(rep_ids)]
    long.to_csv(p(TABLEAU / "tab_role_mix.csv"), index=False)
    return long.nfl_id.nunique()


def build_trajectories():
    """Per-frame normalized safety paths for a sample of plays, for the field view."""
    import glob
    frozen = pd.read_csv(p(PROCESSED / "opportunity_table_deep_v1.csv"))
    rep_ids = set(pd.read_csv(p(PROCESSED / "roe_with_cis_v1.csv")).nfl_id)
    # attach ROE so the field view can show high- vs low-ROE plays
    roe = pd.read_csv(p(PROCESSED / "roe_opportunity_v1.csv"))[
        ["opportunity_id", "roe_gbm"]]
    frozen = frozen.merge(roe, on="opportunity_id", how="left")

    # sample: a spread of plays from reported safeties (some high, some low ROE)
    pool = frozen[frozen.nfl_id.isin(rep_ids) & frozen.roe_gbm.notna()].copy()
    pool = pool.sort_values("roe_gbm")
    rng = np.random.default_rng(SEED)
    take = min(N_SAMPLE_PLAYS, len(pool))
    # mix of extremes and middle so the viz shows range of outcomes
    idx = np.linspace(0, len(pool) - 1, take).astype(int)
    sample = pool.iloc[idx].copy()

    # load only the output weeks we need
    out_files = sorted(glob.glob(p(TRAIN / "output_2023_w*.csv")))
    out = pd.concat([pd.read_csv(f, usecols=["game_id", "play_id", "nfl_id",
                     "frame_id", "x", "y"]) for f in out_files], ignore_index=True)

    rows = []
    for _, opp in sample.iterrows():
        t = out[(out.game_id == opp.game_id) & (out.play_id == opp.play_id) &
                (out.nfl_id == opp.nfl_id)].sort_values("frame_id")
        if t.empty:
            continue
        xn, yn = g.normalize_xy(t.x.values, t.y.values, opp.play_direction)
        lxn, lyn = g.normalize_xy(opp.ball_land_x, opp.ball_land_y, opp.play_direction)
        for fr, (px, py) in enumerate(zip(xn, yn), start=1):
            rows.append({
                "opportunity_id": opp.opportunity_id,
                "player_name": opp.player_name,
                "role": opp.role_proxy,
                "roe": round(float(opp.roe_gbm), 2),
                "roe_tier": ("High" if opp.roe_gbm > 3 else
                             "Low" if opp.roe_gbm < -1 else "Average"),
                "frame": fr,
                "x": round(float(px), 2),
                "y": round(float(py), 2),
                "ball_land_x": round(float(lxn), 2),
                "ball_land_y": round(float(lyn), 2),
                "flight_time_s": opp.flight_time_s,
            })
    traj = pd.DataFrame(rows)
    if traj.empty:
        print("    (warning: no trajectories matched output tracking \u2014 "
              "check that all week output files are in data/raw/train/)")
        traj.to_csv(p(TABLEAU / "tab_trajectories.csv"), index=False)
        return 0
    traj.to_csv(p(TABLEAU / "tab_trajectories.csv"), index=False)
    return traj.opportunity_id.nunique()


def main():
    print("Building Tableau export tables -> data/tableau/")
    n1 = build_leaderboard();      print(f"  tab_leaderboard.csv    ({n1} safeties)")
    n2 = build_stabilization();    print(f"  tab_stabilization.csv  ({n2} rows)")
    n3 = build_role_mix();         print(f"  tab_role_mix.csv       ({n3} safeties, long)")
    n4 = build_trajectories();     print(f"  tab_trajectories.csv   ({n4} plays)")
    print("\nAll four tables are tidy/long and ready to connect in Tableau Public.")


if __name__ == "__main__":
    main()
