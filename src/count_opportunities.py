"""
Count range OPPORTUNITIES per safety across the full 2023 season (weeks 1-18).

The prior script counted distinct safety *players* (179). This one counts, for
each safety, how many plays they could actually contribute to the metric, and
shows the distribution against candidate minimum-opportunity thresholds. That
distribution is what tells you whether a single season supports a skill-stability
claim, or whether stability must be flagged as sample-limited.

Two layers are reported, because a fully rigorous opportunity filter needs the
pre-snap role-proxy classifier that is NOT built yet. Inventing assignment
thresholds here would prejudge that step, so instead:

  LAYER A (ceiling): every play where the safety is a tracked coverage defender
      with post-throw output tracking. The most opportunities they could have.

  LAYER B (defensible now): Layer A, further restricted to what the data alone
      can justify without guessing individual assignment:
        - play has a valid ball landing point (ball_land_x/y present)
        - play not nullified by penalty
        - safety's player_role == 'Defensive Coverage'
        - pass_result is a genuine pass attempt (C/I/IN), not sack/scramble

  NEITHER layer guesses which safety "owned" a given throw. That gating comes
  later, from the role-proxy classifier, and will only shrink these counts.

The three generic 'S'-labelled players are excluded from qualifying opportunities
per request; only FS and SS count.

Usage:
    python count_opportunities.py --data-dir /path/to/parent
where parent contains train/input_2023_w*.csv, train/output_2023_w*.csv,
and supplementary_data.csv.
"""

import glob
import os
import numpy as np
import pandas as pd
from paths import TRAIN, SUPP, INTERIM, p

QUALIFYING_SAFETY_POSITIONS = {"FS", "SS"}   # generic 'S' excluded per request
PASS_ATTEMPT_RESULTS = {"C", "I", "IN"}       # complete, incomplete, interception
THRESHOLDS = [5, 10, 20, 30, 50, 75, 100]


def load_supplementary(data_dir: str) -> pd.DataFrame:
    supp = pd.read_csv(p(SUPP),
                       low_memory=False)
    supp.columns = [c.strip().strip('"') for c in supp.columns]
    keep = ["game_id", "play_id", "pass_result", "play_nullified_by_penalty"]
    return supp[keep].drop_duplicates(["game_id", "play_id"])


def main() -> None:
    train_dir = p(TRAIN)
    input_files = sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))
    output_files = sorted(glob.glob(os.path.join(train_dir, "output_2023_w*.csv")))
    if not input_files:
        raise FileNotFoundError(f"No input_2023_w*.csv files found in {train_dir}")

    supp = load_supplementary()

    # Build a set of (game_id, play_id, nfl_id) that have output tracking, per week,
    # so we only count opportunities where post-throw data actually exists.
    layer_a_rows = []   # ceiling: tracked coverage safety with output
    layer_b_rows = []   # defensible-now filter

    for inf in input_files:
        week = os.path.basename(inf).replace("input_2023_w", "").replace(".csv", "")
        outf = os.path.join(train_dir, f"output_2023_w{week}.csv")

        icols = ["game_id", "play_id", "nfl_id", "player_name", "player_position",
                 "player_role", "ball_land_x", "ball_land_y"]
        inp = pd.read_csv(inf, usecols=icols).drop_duplicates(
            ["game_id", "play_id", "nfl_id"])

        # safeties only (FS/SS), generic S dropped
        saf = inp[inp.player_position.isin(QUALIFYING_SAFETY_POSITIONS)].copy()

        # which safety-plays have output tracking?
        if os.path.exists(outf):
            out_keys = pd.read_csv(outf, usecols=["game_id", "play_id", "nfl_id"]) \
                         .drop_duplicates()
            out_keys["_has_output"] = True
            saf = saf.merge(out_keys, on=["game_id", "play_id", "nfl_id"], how="left")
            saf = saf[saf._has_output == True]
        else:
            continue  # no output for this week; skip

        # attach play context
        saf = saf.merge(supp, on=["game_id", "play_id"], how="left")

        # LAYER A: everything that survived the output requirement
        layer_a_rows.append(saf[["nfl_id", "player_name", "player_position"]])

        # LAYER B: defensible additional filters
        valid_land = saf.ball_land_x.notna() & saf.ball_land_y.notna()
        not_nullified = saf.play_nullified_by_penalty.fillna("N").str.upper() != "Y"
        coverage_role = saf.player_role == "Defensive Coverage"
        real_pass = saf.pass_result.isin(PASS_ATTEMPT_RESULTS)
        b = saf[valid_land & not_nullified & coverage_role & real_pass]
        layer_b_rows.append(b[["nfl_id", "player_name", "player_position"]])

    layer_a = pd.concat(layer_a_rows, ignore_index=True)
    layer_b = pd.concat(layer_b_rows, ignore_index=True)

    def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
        counts = (df.groupby(["nfl_id", "player_name", "player_position"])
                    .size().reset_index(name="opportunities")
                    .sort_values("opportunities", ascending=False))
        print("\n" + "=" * 64)
        print(f"{label}")
        print("=" * 64)
        print(f"Distinct safeties with >=1 opportunity: {len(counts)}")
        print(f"Total opportunities: {int(counts.opportunities.sum())}")
        print(f"Median opportunities per safety: {counts.opportunities.median():.0f}")
        print(f"Mean opportunities per safety:   {counts.opportunities.mean():.1f}")
        print("\nSafeties clearing each minimum-opportunity threshold:")
        for t in THRESHOLDS:
            n = (counts.opportunities >= t).sum()
            print(f"  >= {t:>3} opps : {n:>3} safeties")
        print("\nTop 10 by opportunity count:")
        for _, r in counts.head(10).iterrows():
            print(f"  {r.player_name:<26} {r.player_position}  {int(r.opportunities)}")
        return counts

    a_counts = summarize(layer_a, "LAYER A - ceiling (tracked coverage safety + output)")
    b_counts = summarize(layer_b, "LAYER B - defensible-now filter")

    # save both for downstream use
    a_counts.to_csv(p(INTERIM / "safety_opportunities_layerA.csv"), index=False)
    b_counts.to_csv(p(INTERIM / "safety_opportunities_layerB.csv"), index=False)
    print("\nSaved per-safety opportunity counts to:")
    print(f"  {p(INTERIM / 'safety_opportunities_layerA.csv')}")
    print(f"  {p(INTERIM / 'safety_opportunities_layerB.csv')}")
    print("\nReminder: Layer B is still pre-classifier. Adding the role-proxy")
    print("responsibility gate later will reduce these counts further, not raise them.")


if __name__ == "__main__":
    main()