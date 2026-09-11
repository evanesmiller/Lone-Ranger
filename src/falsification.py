"""
Falsification / permutation test for Lone Ranger.

Purpose: try to BREAK the central claim. The reliability result says safeties
differ in ROE in a way that repeats. The null this must beat: those apparent
differences are an artifact of how per-play ROE is distributed, and would appear
even with no real player effect. We destroy the play->player link by shuffling
nfl_id across opportunities and re-measuring. If the real signal sits in the
extreme tail of the shuffled distribution, the player effect is real.

Two things falsified, on the 30 reported safeties, using per-opportunity ROE:

1) BETWEEN-SAFETY SPREAD
   Real: SD of per-safety mean ROE (how far apart safeties are).
   Null: shuffle nfl_id across all reported opportunities, recompute the SD.
   Permutation p = fraction of shuffles with spread >= real. Small p => the
   spread is larger than chance, i.e. real player differences exist.

   Also reported: variance-components view. The shuffled spread is the noise
   floor (within-player sampling noise mimicking between-player spread); the
   gap between real and shuffled spread is the true between-player signal.

2) SPLIT-HALF RELIABILITY UNDER SHUFFLING
   Real reliability was ~0.58 (Spearman-Brown). Under shuffled identities it
   should collapse toward 0, because a "safety" is now a random mix of plays.
   Confirming near-zero shuffled reliability proves the real number is not a
   mechanical artifact of the split-half procedure.

Usage:
    python falsification.py --data-dir /path/to/parent
Requires: roe_opportunity_v1.csv (from expected_model.py).
"""

import os
import numpy as np
import pandas as pd
from paths import PROCESSED, p

ROE_COL = "roe_gbm"
N_PERM = 5000
N_SPLIT_ITERS = 300     # split-half iters per shuffle (kept modest; many shuffles)
N_SHUFFLE_REL = 500     # shuffles for the reliability-collapse test
SEED = 7


def between_sd(roe, ids):
    """SD of per-safety mean ROE, weighted equally per safety."""
    df = pd.DataFrame({"roe": roe, "id": ids})
    return df.groupby("id").roe.mean().std(ddof=1)


def split_half_once(roe_by_id, rng):
    h1, h2 = [], []
    for v in roe_by_id.values():
        if len(v) < 4:
            continue
        idx = rng.permutation(len(v))
        half = len(v) // 2
        h1.append(v[idx[:half]].mean())
        h2.append(v[idx[half:2 * half]].mean())
    if len(h1) < 3:
        return np.nan
    r = np.corrcoef(h1, h2)[0, 1]
    return 2 * r / (1 + r) if (1 + r) != 0 else np.nan


def main():
    opp = pd.read_csv(p(PROCESSED / "roe_opportunity_v1.csv"))
    rep = opp[opp.in_study].copy()
    roe = rep[ROE_COL].values
    ids = rep.nfl_id.values
    rng = np.random.default_rng(SEED)

    print("=" * 62)
    print("FALSIFICATION TEST  -- 30 reported safeties")
    print("=" * 62)
    print(f"opportunities: {len(roe)} | metric: {ROE_COL}")

    # 1) between-safety spread vs shuffled null
    real_sd = between_sd(roe, ids)
    null_sds = np.empty(N_PERM)
    for k in range(N_PERM):
        null_sds[k] = between_sd(roe, rng.permutation(ids))
    p_spread = (np.sum(null_sds >= real_sd) + 1) / (N_PERM + 1)
    signal = real_sd - null_sds.mean()   # real minus noise-floor spread

    print("\n1) BETWEEN-SAFETY SPREAD (SD of per-safety mean ROE)")
    print(f"   real spread          : {real_sd:.3f} yds")
    print(f"   shuffled null (mean) : {null_sds.mean():.3f} yds  "
          f"(noise floor)")
    print(f"   shuffled 95th pct    : {np.percentile(null_sds,95):.3f} yds")
    print(f"   true between-player signal (real - null mean): {signal:.3f} yds")
    print(f"   permutation p-value  : {p_spread:.4f}")
    verdict = ("real player differences confirmed (signal exceeds chance)"
               if p_spread < 0.05 else
               "cannot rule out chance -- spread is within the null")
    print(f"   READ: {verdict}")

    # 2) reliability collapse under shuffling
    # real reliability (unshuffled)
    real_by_id = {i: g[ROE_COL].values for i, g in rep.groupby("nfl_id")}
    real_rel = np.median([split_half_once(real_by_id, rng)
                          for _ in range(N_SPLIT_ITERS)])
    # shuffled reliability distribution
    shuf_rel = np.empty(N_SHUFFLE_REL)
    for k in range(N_SHUFFLE_REL):
        sid = rng.permutation(ids)
        tmp = pd.DataFrame({"roe": roe, "id": sid})
        shuf_by_id = {i: g.roe.values for i, g in tmp.groupby("id")}
        shuf_rel[k] = split_half_once(shuf_by_id, rng)
    p_rel = (np.sum(shuf_rel >= real_rel) + 1) / (N_SHUFFLE_REL + 1)

    print("\n2) SPLIT-HALF RELIABILITY UNDER SHUFFLING")
    print(f"   real reliability (SB)     : {real_rel:.3f}")
    print(f"   shuffled reliability mean : {np.median(shuf_rel):.3f}  "
          f"(should be ~0)")
    print(f"   shuffled 95th pct         : {np.percentile(shuf_rel,95):.3f}")
    print(f"   permutation p-value       : {p_rel:.4f}")
    verdict2 = ("reliability is real -- collapses to ~0 when identities shuffled"
                if p_rel < 0.05 else
                "reliability not distinguishable from shuffled artifact")
    print(f"   READ: {verdict2}")

    pd.DataFrame({"null_between_sd": null_sds}).to_csv(
        p(PROCESSED / "falsification_null_spread_v1.csv"), index=False)

    print("\n" + "=" * 62)
    both = (p_spread < 0.05) and (p_rel < 0.05)
    if both:
        print("BOTH TESTS PASS: the range signal survives falsification.")
        print("Between-safety differences and their reliability are not")
        print("artifacts of ROE's distribution -- they reflect real, repeatable")
        print("player differences. This backstops the reliability finding.")
    else:
        print("AT LEAST ONE TEST DID NOT PASS -- report this honestly. The")
        print("apparent player signal may be partly a distributional artifact.")
    print("=" * 62)


if __name__ == "__main__":
    main()