"""
Freeze the deep-sample opportunity table  (Phase 1, step 1).

This is the immutable input every downstream metric joins onto. One row per
qualifying safety-play. "Qualifying" means ALL of the following hold, so that
no later stage has to re-derive eligibility:

  1. Player is a safety (FS/SS; generic 'S' already excluded upstream).
  2. Player has a role_proxy of SINGLE_HIGH or SPLIT_DEEP  (deep sample).
     BOX_UNDERNEATH is dropped: deep range was not the assignment.
  3. Player has post-throw OUTPUT tracking on this play (trajectory exists).
  4. Play has a valid ball landing point.
  5. Play is a genuine pass attempt (pass_result in C/I/IN), not sack/scramble.
  6. Play is not nullified by penalty.

Everything needed to START metric construction is carried on the row (snap-frame
depth, lateral, role, coverage context, flight time, ball landing), so the metric
stage never re-opens the raw files just to re-establish eligibility.

IMMUTABILITY: the output is written once and checksummed. Downstream code should
read it, never regenerate it ad hoc. If the eligibility rules ever change, bump
the VERSION string so old and new frozen tables never silently mix.

Usage:
    python freeze_opportunity_table.py --data-dir /path/to/parent
Requires (already produced): safety_roles.csv in --data-dir.
"""

import glob
import hashlib
import os
import numpy as np
import pandas as pd
from paths import TRAIN, SUPP, INTERIM, PROCESSED, p

VERSION = "deep-v1"
DEEP_ROLES = {"SINGLE_HIGH", "SPLIT_DEEP"}
PASS_ATTEMPT_RESULTS = {"C", "I", "IN"}
MAX_FLIGHT_FRAMES = 60          # 6.0s at 10 Hz; longer = scramble/broken play
STUDY_MIN_OPPS = 50            # study population = safeties clearing this bar


def load_supplementary(data_dir):
    supp = pd.read_csv(p(SUPP),
                       low_memory=False)
    supp.columns = [c.strip().strip('"') for c in supp.columns]
    keep = ["game_id", "play_id", "pass_result", "play_nullified_by_penalty",
            "pass_length", "route_of_targeted_receiver", "down", "yards_to_go",
            "offense_formation", "receiver_alignment", "expected_points_added"]
    keep = [c for c in keep if c in supp.columns]
    return supp[keep].drop_duplicates(["game_id", "play_id"])


def main():
    train_dir = p(TRAIN)
    roles = pd.read_csv(p(INTERIM / "safety_roles.csv"))

    # --- deep sample only ---
    deep = roles[roles.role_proxy.isin(DEEP_ROLES)].copy()
    n_deep = len(deep)

    # --- pull per-safety-play flight time + ball landing from input files ---
    # (num_frames_output & ball_land are on the input rows; take frame 1)
    input_files = sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))
    flight_parts = []
    for f in input_files:
        icols = ["game_id", "play_id", "nfl_id", "frame_id",
                 "num_frames_output", "ball_land_x", "ball_land_y",
                 "absolute_yardline_number", "play_direction"]
        d = pd.read_csv(f, usecols=icols)
        d = d[d.frame_id == 1].drop(columns="frame_id")
        flight_parts.append(d)
    flight = pd.concat(flight_parts, ignore_index=True)
    flight["flight_time_s"] = flight["num_frames_output"] / 10.0

    deep = deep.merge(flight, on=["game_id", "play_id", "nfl_id"], how="left")
    n_after_flight = deep[deep.num_frames_output.notna()].shape[0]

    # --- require post-throw OUTPUT tracking on this play ---
    output_files = sorted(glob.glob(os.path.join(train_dir, "output_2023_w*.csv")))
    out_keys = []
    for f in output_files:
        o = pd.read_csv(f, usecols=["game_id", "play_id", "nfl_id"]).drop_duplicates()
        out_keys.append(o)
    out_keys = pd.concat(out_keys, ignore_index=True).drop_duplicates()
    out_keys["_has_output"] = True
    deep = deep.merge(out_keys, on=["game_id", "play_id", "nfl_id"], how="left")
    deep = deep[deep._has_output == True].drop(columns="_has_output")
    n_after_output = len(deep)

    # --- play context + eligibility filters ---
    supp = load_supplementary(data_dir)
    deep = deep.merge(supp, on=["game_id", "play_id"], how="left")

    valid_land = deep.ball_land_x.notna() & deep.ball_land_y.notna()
    real_pass = deep.pass_result.isin(PASS_ATTEMPT_RESULTS)
    not_null = deep.play_nullified_by_penalty.fillna("N").str.upper() != "Y"
    has_flight = deep.num_frames_output.notna() & (deep.num_frames_output > 0)
    # Flight-time sanity cap: a legitimate pass is airborne well under ~6s.
    # Longer "flights" are scrambles / broken plays that slipped the pass filter.
    # FLAG: these are excluded and counted so the drop is auditable, not silent.
    sane_flight = deep.num_frames_output <= MAX_FLIGHT_FRAMES

    frozen = deep[valid_land & real_pass & not_null & has_flight & sane_flight].copy()
    n_flight_outliers = int((deep[valid_land & real_pass & not_null & has_flight]
                             .num_frames_output > MAX_FLIGHT_FRAMES).sum())

    # --- opportunity id + tidy column order ---
    frozen = frozen.sort_values(["game_id", "play_id", "nfl_id"]).reset_index(drop=True)
    frozen.insert(0, "opportunity_id",
                  frozen.game_id.astype(str) + "_" +
                  frozen.play_id.astype(str) + "_" +
                  frozen.nfl_id.astype(str))
    frozen["frozen_version"] = VERSION

    # Mark the study population: safeties clearing the opportunity bar. Downstream
    # ranking/reliability runs on in_study rows; the rest stay for context only.
    opp_per_safety = frozen.groupby("nfl_id").size()
    study_ids = set(opp_per_safety[opp_per_safety >= STUDY_MIN_OPPS].index)
    frozen["in_study"] = frozen.nfl_id.isin(study_ids)

    col_order = ["opportunity_id", "game_id", "play_id", "nfl_id", "player_name",
                 "player_position", "role_proxy", "n_deep_safeties",
                 "depth", "lateral_from_mid",
                 "team_coverage_man_zone", "team_coverage_type",
                 "num_frames_output", "flight_time_s",
                 "ball_land_x", "ball_land_y",
                 "absolute_yardline_number", "play_direction",
                 "pass_result", "pass_length", "route_of_targeted_receiver",
                 "down", "yards_to_go", "offense_formation", "receiver_alignment",
                 "expected_points_added", "in_study", "frozen_version"]
    col_order = [c for c in col_order if c in frozen.columns]
    frozen = frozen[col_order]

    # --- write + checksum ---
    out_path = p(PROCESSED / "opportunity_table_deep_v1.csv")
    frozen.to_csv(out_path, index=False)
    checksum = hashlib.md5(open(out_path, "rb").read()).hexdigest()
    with open(p(PROCESSED / "opportunity_table_deep_v1.md5"), "w") as fh:
        fh.write(f"{checksum}  opportunity_table_deep_v1.csv  ({VERSION})\n")

    # --- report / reconciliation ---
    print("=" * 60)
    print(f"FROZEN OPPORTUNITY TABLE  ({VERSION})")
    print("=" * 60)
    print("Funnel (how the deep sample narrows to frozen rows):")
    print(f"  deep-role safety-plays (single+split):   {n_deep:>6}")
    print(f"  ... with flight time present:            {n_after_flight:>6}")
    print(f"  ... with post-throw output tracking:     {n_after_output:>6}")
    print(f"  ... after pass/landing/penalty filters:  {len(frozen):>6}  <- FROZEN")
    print(f"  (flight-time outliers >6s excluded:      {n_flight_outliers:>6})")
    print()
    print(f"Distinct safeties: {frozen.nfl_id.nunique()}")
    print(f"Distinct plays:    {frozen[['game_id','play_id']].drop_duplicates().shape[0]}")
    n_study = frozen[frozen.in_study].nfl_id.nunique()
    n_study_opps = int(frozen.in_study.sum())
    print(f"\nSTUDY POPULATION (>= {STUDY_MIN_OPPS} opps): {n_study} safeties, "
          f"{n_study_opps} opportunities")
    print("\nRole split in frozen table:")
    print(frozen.role_proxy.value_counts().to_string())
    print("\nPer-safety opportunity thresholds (frozen):")
    per = frozen.groupby("nfl_id").size()
    for t in [20, 30, 50, 75, 100]:
        print(f"  >= {t:>3}: {(per >= t).sum():>3} safeties")
    print(f"\nFlight-time sanity (s): min {frozen.flight_time_s.min():.1f} "
          f"median {frozen.flight_time_s.median():.1f} max {frozen.flight_time_s.max():.1f}")
    print(f"\nWrote: {out_path}")
    print(f"MD5:   {checksum}")
    print("Treat this file as read-only from here on. Bump VERSION to change rules.")


if __name__ == "__main__":
    main()