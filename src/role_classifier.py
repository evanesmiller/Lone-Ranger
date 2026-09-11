"""
Pre-snap role-proxy classifier for safeties (2023 Big Data Bowl).

Assigns each qualifying safety-play a coverage ROLE derived from tracking, not
from any assignment label (which the data does not contain). The role is the
gate that makes Range Over Expected meaningful: it lets you compare safeties
doing the same job, and lets you exclude/flag throws that were never a given
safety's responsibility.

WHY THIS EXISTS
    player_role only says Defensive Coverage / Targeted Receiver / Passer / Other.
    It does NOT say "deep third" vs "flat" vs "robber". Two safeties in the same
    Cover 3 shell can have opposite jobs. So we infer a coarse, defensible role
    bucket from pre-snap alignment + the play's deep-safety structure, and treat
    it as a proxy, never as ground truth.

ROLE BUCKETS (deliberately coarse -- finer splits aren't defensible from tracking)
    SINGLE_HIGH   : the lone deep safety on the play. Range IS the job. Cleanest
                    sample. Cover 1 / Cover 3 structures.
    SPLIT_DEEP    : one of two (occasionally three) deep safeties. Owns roughly
                    half the field. Range matters but only within their half.
                    Cover 2 / Cover 4 / Cover 6 structures.
    BOX_UNDERNEATH: aligned shallow at snap. Playing near the LOS / underneath.
                    Deep range is NOT their job on this snap; deep-throw
                    opportunities should be excluded or flagged.

HOW ROLE IS ASSIGNED (per play, at the snap frame = input frame_id 1)
    1. Compute each safety's depth behind LOS and lateral position.
    2. Mark "deep" safeties (depth >= DEEP_DEPTH_YDS).
    3. Count deep safeties on the play:
         - shallow safety                      -> BOX_UNDERNEATH
         - deep, and he is the ONLY deep safety -> SINGLE_HIGH
         - deep, and 2+ deep safeties           -> SPLIT_DEEP
    4. Cross-check against team_coverage_type where available and emit an
       agreement flag, so you can audit how often the tracking-derived role
       matches the stated shell. Disagreements aren't errors to silence; they
       are exactly the ambiguous snaps your writeup should quantify.

THRESHOLDS are read from this week's data distribution (median depth ~10.7 yds,
clear separation), but are exposed as constants so you can run sensitivity
versions -- your plan calls for exactly that, and role thresholds are a place
where a single arbitrary cutoff should never be trusted.

OUTPUT
    A play-level table: one row per (game_id, play_id, nfl_id) safety, with role,
    depth, lateral position, deep-safety count on the play, stated coverage, and
    a shell-agreement flag. This is the gate table the metric pipeline joins on.

Usage:
    python role_classifier.py --data-dir /path/to/parent [--deep-depth 10.0]
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from paths import TRAIN, SUPP, INTERIM, p

QUALIFYING_SAFETY_POSITIONS = {"FS", "SS"}
DEEP_DEPTH_YDS = 10.0          # >= this depth behind LOS at snap counts as "deep"
FIELD_MIDDLE_Y = 26.65         # 53.3 / 2

# Map stated coverage_type -> how many deep safeties we'd EXPECT, for cross-check.
# Used only to flag agreement; never to override the tracking-derived role.
# Single-high shells: one deep defender (Cover 1, Cover 3, and Cover 0 which is
# man-free / no deep help -- grouped here as "not split", audited separately).
SINGLE_HIGH_SHELLS = {"COVER_1_MAN", "COVER_3_ZONE", "COVER_0_MAN"}
SPLIT_SHELLS = {"COVER_2_ZONE", "COVER_2_MAN", "COVER_4_ZONE", "COVER_6_ZONE"}


def load_supplementary() -> pd.DataFrame:
    supp = pd.read_csv(p(SUPP),
                       low_memory=False)
    supp.columns = [c.strip().strip('"') for c in supp.columns]
    keep = ["game_id", "play_id", "team_coverage_man_zone", "team_coverage_type"]
    return supp[keep].drop_duplicates(["game_id", "play_id"])


def depth_behind_los(x, los, direction):
    # Defense lines up on the far side of the LOS from the offense's motion.
    return np.where(direction == "right", x - los, los - x)


def classify_week(inf: str, supp: pd.DataFrame, deep_depth: float) -> pd.DataFrame:
    cols = ["game_id", "play_id", "nfl_id", "frame_id", "player_name",
            "player_position", "player_role", "play_direction",
            "absolute_yardline_number", "x", "y"]
    inp = pd.read_csv(inf, usecols=cols)

    # snap frame for safeties
    saf = inp[(inp.player_position.isin(QUALIFYING_SAFETY_POSITIONS)) &
              (inp.frame_id == 1)].copy()
    if saf.empty:
        return pd.DataFrame()

    saf["depth"] = depth_behind_los(
        saf.x.values, saf.absolute_yardline_number.values, saf.play_direction.values)
    saf["lateral_from_mid"] = (saf.y - FIELD_MIDDLE_Y).abs()
    saf["is_deep"] = saf.depth >= deep_depth

    # count deep safeties per play
    deep_ct = (saf[saf.is_deep]
               .groupby(["game_id", "play_id"]).size()
               .rename("n_deep_safeties"))
    saf = saf.merge(deep_ct, on=["game_id", "play_id"], how="left")
    saf["n_deep_safeties"] = saf["n_deep_safeties"].fillna(0).astype(int)

    def role(r):
        if not r.is_deep:
            return "BOX_UNDERNEATH"
        if r.n_deep_safeties <= 1:
            return "SINGLE_HIGH"
        return "SPLIT_DEEP"

    saf["role_proxy"] = saf.apply(role, axis=1)

    # attach stated coverage and cross-check
    saf = saf.merge(supp, on=["game_id", "play_id"], how="left")

    def shell_agrees(r):
        ct = str(r.team_coverage_type)
        if r.role_proxy == "SINGLE_HIGH":
            # COVER_0 has NO deep safety, so a single-high read there is a real
            # disagreement, not a match -- count it as False, not agreement.
            if ct == "COVER_0_MAN":
                return False
            return ct in SINGLE_HIGH_SHELLS
        if r.role_proxy == "SPLIT_DEEP":
            return ct in SPLIT_SHELLS
        return np.nan  # box safeties: shell expectation not well-defined
    saf["shell_agreement"] = saf.apply(shell_agrees, axis=1)

    return saf[["game_id", "play_id", "nfl_id", "player_name", "player_position",
                "depth", "lateral_from_mid", "n_deep_safeties", "role_proxy",
                "team_coverage_man_zone", "team_coverage_type", "shell_agreement"]]


def main(deep_depth: float = DEEP_DEPTH_YDS) -> None:
    train_dir = p(TRAIN)
    input_files = sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))
    if not input_files:
        raise FileNotFoundError(f"No input_2023_w*.csv in {train_dir}")

    supp = load_supplementary()
    parts = [classify_week(f, supp, deep_depth) for f in input_files]
    roles = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    print("=" * 60)
    print(f"Role-proxy classification  (deep-depth threshold = {deep_depth} yds)")
    print("=" * 60)
    print(f"Total safety-plays classified: {len(roles)}")
    print("\nRole distribution:")
    print(roles.role_proxy.value_counts().to_string())
    print("\nRole distribution by listed position:")
    print(pd.crosstab(roles.player_position, roles.role_proxy).to_string())

    # shell agreement audit (deep roles only, where expectation is defined)
    deep_roles = roles[roles.role_proxy.isin(["SINGLE_HIGH", "SPLIT_DEEP"])]
    agree = deep_roles.shell_agreement.dropna()
    if len(agree):
        print(f"\nShell cross-check (deep roles with a stated coverage_type):")
        print(f"  tracking-derived role agrees with stated shell: "
              f"{100*agree.mean():.1f}%  (n={len(agree)})")
        print("  (disagreements are ambiguous snaps to quantify, not bugs to hide)")

    # per-safety role mix -- how consistent is each player's job?
    mix = (roles.groupby(["nfl_id", "player_name"])
                .role_proxy.value_counts(normalize=True)
                .unstack(fill_value=0))
    for col in ["SINGLE_HIGH", "SPLIT_DEEP", "BOX_UNDERNEATH"]:
        if col not in mix:
            mix[col] = 0.0
    mix["n_plays"] = roles.groupby(["nfl_id", "player_name"]).size()
    mix = mix.sort_values("n_plays", ascending=False)

    print("\nRole mix for 12 highest-usage safeties "
          "(fraction of snaps in each role):")
    print(mix.head(12)[["SINGLE_HIGH", "SPLIT_DEEP", "BOX_UNDERNEATH", "n_plays"]]
              .round(2).to_string())

    out_roles = p(INTERIM / "safety_roles.csv")
    out_mix = p(INTERIM / "safety_role_mix.csv")
    roles.to_csv(out_roles, index=False)
    mix.reset_index().to_csv(out_mix, index=False)
    print(f"\nSaved play-level role gate table to: {out_roles}")
    print(f"Saved per-safety role mix to:        {out_mix}")
    print("\nNext: join safety_roles.csv onto your opportunity table and compute")
    print("ROE WITHIN role_proxy. Start with SINGLE_HIGH -- cleanest range sample.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-depth", type=float, default=DEEP_DEPTH_YDS,
                    help="Depth (yds behind LOS) at snap to count as a deep safety")
    args = ap.parse_args()
    main(args.deep_depth)