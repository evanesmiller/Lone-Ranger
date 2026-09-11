"""
Count the safety population across the full 2023 Big Data Bowl dataset (weeks 1-18).

Reports:
  1. Distinct safety players (by nfl_id) across all weeks.
  2. Breakdown by listed position (FS / SS / S).
  3. How many of those safeties actually appear in the OUTPUT files
     (player_to_predict), i.e. have post-throw tracking your metrics can use.
  4. Per-week distinct-safety counts, as a sanity check that every file loaded.

Usage:
    python count_safeties.py --data-dir /path/to/parent
where parent contains:
    train/input_2023_w01.csv ... input_2023_w18.csv
    train/output_2023_w01.csv ... output_2023_w18.csv
"""

import glob
import os
import pandas as pd
from paths import TRAIN, INTERIM, p

SAFETY_POSITIONS = {"FS", "SS", "S"}


def main() -> None:
    train_dir = p(TRAIN)

    input_files = sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))
    output_files = sorted(glob.glob(os.path.join(train_dir, "output_2023_w*.csv")))

    if not input_files:
        raise FileNotFoundError(f"No input_2023_w*.csv files found in {train_dir}")

    print(f"Found {len(input_files)} input files and {len(output_files)} output files.\n")

    # ---- 1 & 2: distinct safeties from input files -----------------------
    # Only need identity + position columns; keeps memory low across 18 weeks.
    safety_rows = []          # one row per (week, nfl_id, position)
    per_week_counts = {}

    for f in input_files:
        week = os.path.basename(f).replace("input_2023_w", "").replace(".csv", "")
        df = pd.read_csv(f, usecols=["nfl_id", "player_position", "player_name"])
        saf = df[df.player_position.isin(SAFETY_POSITIONS)]
        # distinct players this week
        distinct = saf.drop_duplicates("nfl_id")[["nfl_id", "player_position", "player_name"]]
        distinct["week"] = week
        safety_rows.append(distinct)
        per_week_counts[week] = distinct.nfl_id.nunique()

    all_safeties = pd.concat(safety_rows, ignore_index=True)

    # A player could in principle be listed under different position strings in
    # different weeks; collapse to one row per nfl_id, keeping the most common label.
    dedup = (
        all_safeties
        .groupby("nfl_id")
        .agg(player_name=("player_name", "first"),
             position=("player_position",
                       lambda s: s.value_counts().index[0]))
        .reset_index()
    )

    total_safeties = dedup.nfl_id.nunique()

    print("=" * 60)
    print(f"TOTAL distinct safeties (FS/SS/S) across weeks 1-18: {total_safeties}")
    print("=" * 60)
    print("\nBy listed position (players whose most-common label is):")
    print(dedup.position.value_counts().to_string())

    # ---- 3: how many have post-throw output tracking ---------------------
    if output_files:
        out_ids = set()
        for f in output_files:
            o = pd.read_csv(f, usecols=["nfl_id"])
            out_ids.update(o.nfl_id.unique())
        safeties_with_output = dedup[dedup.nfl_id.isin(out_ids)]
        n_with_output = safeties_with_output.nfl_id.nunique()
        print("\n" + "-" * 60)
        print(f"Safeties that appear in OUTPUT files (have post-throw tracking "
              f"on >=1 play): {n_with_output} of {total_safeties} "
              f"({100 * n_with_output / total_safeties:.1f}%)")
        print("-" * 60)
        print("Note: this is player-level. A safety can be tracked on some plays")
        print("and not others; per-play opportunity counts come later in the pipeline.")

    # ---- 4: per-week sanity check ----------------------------------------
    print("\nDistinct safeties per week (sanity check all files loaded):")
    for wk in sorted(per_week_counts):
        print(f"  w{wk}: {per_week_counts[wk]}")

    # optional: save the master safety list for downstream use
    out_path = p(INTERIM / "safety_roster.csv")
    dedup.sort_values("position").to_csv(out_path, index=False)
    print(f"\nSaved distinct-safety roster to: {out_path}")


if __name__ == "__main__":
    main()