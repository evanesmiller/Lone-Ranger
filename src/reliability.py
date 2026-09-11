"""
Reliability analysis for Lone Ranger -- the test that decides what can be claimed.

Range Over Expected (ROE) is a per-play residual averaged into a per-safety
number. This asks: is that number a STABLE property of the player, or mostly
noise that looks like skill over a season? Three tests, on the 30 reported
safeties, using per-opportunity ROE from expected_model.py.

1) SPLIT-HALF RELIABILITY (primary)
   Randomly split each safety's opportunities into two halves, compute each
   safety's mean ROE in each half, correlate the two halves across safeties.
   Repeat over many random splits and report the distribution (a single split
   is itself noisy). Apply Spearman-Brown to estimate full-length reliability
   from the half-length correlation.
     r_sb = 2*r / (1 + r)
   Read: high & stable r -> ROE measures something repeatable. Near zero ->
   ROE is largely noise at this sample size (a legitimate finding, not a bug).

2) BOOTSTRAP CONFIDENCE INTERVALS (which leaderboard gaps are real)
   Resample each safety's opportunities with replacement, recompute mean ROE,
   repeat. The 2.5-97.5 percentile band is that safety's uncertainty. Overlapping
   bands between two safeties mean the gap between them is not established.

3) STABILIZATION CURVE (justifies the 50-opp threshold empirically)
   For a grid of opportunity counts k, subsample k opportunities per safety
   (only safeties with >= k), compute split-half reliability at that k, and plot
   reliability vs k. Rising, leveling curve shows how much data ROE needs to
   stabilize -- turns "we chose 50" into a data-driven cutoff.

HONESTY: this script roots for neither outcome. It reports whatever the data
says. A weak result -> "range is measurable but does not demonstrably stabilize
within one season at this sample," which is a real, defensible conclusion.

Usage:
    python reliability.py --data-dir /path/to/parent
Requires: roe_opportunity_v1.csv (from expected_model.py).
"""

import os
import numpy as np
import pandas as pd
from paths import PROCESSED, p

ROE_COL = "roe_gbm"          # primary metric; spline agrees at rho=0.987
N_SPLITS = 2000              # random half-splits for reliability distribution
N_BOOT = 2000               # bootstrap resamples for CIs
SEED = 7
MIN_HALF = 10               # need at least this many plays per half to split


def spearman_brown(r):
    return 2 * r / (1 + r) if (1 + r) != 0 else np.nan


def split_half_reliability(roe_by_safety, rng, n_splits=N_SPLITS):
    """Distribution of split-half correlations over random half-splits.

    roe_by_safety: dict nfl_id -> np.array of per-play ROE.
    Returns (raw_correlations, spearman_brown_corrected).
    """
    ids = [k for k, v in roe_by_safety.items() if len(v) >= 2 * MIN_HALF]
    raw = []
    for _ in range(n_splits):
        h1, h2 = [], []
        for i in ids:
            v = roe_by_safety[i]
            idx = rng.permutation(len(v))
            half = len(v) // 2
            h1.append(v[idx[:half]].mean())
            h2.append(v[idx[half:2 * half]].mean())
        if len(h1) >= 3:
            r = np.corrcoef(h1, h2)[0, 1]
            raw.append(r)
    raw = np.array(raw)
    sb = np.array([spearman_brown(r) for r in raw])
    return raw, sb, ids


def bootstrap_cis(roe_by_safety, names, rng, n_boot=N_BOOT):
    rows = []
    for i, v in roe_by_safety.items():
        boot = np.array([rng.choice(v, size=len(v), replace=True).mean()
                         for _ in range(n_boot)])
        rows.append({
            "nfl_id": i, "player_name": names.get(i, str(i)),
            "n_opps": len(v), "roe": v.mean(),
            "ci_lo": np.percentile(boot, 2.5),
            "ci_hi": np.percentile(boot, 97.5),
            "se": boot.std(),
        })
    return pd.DataFrame(rows).sort_values("roe", ascending=False)


def stabilization_curve(roe_by_safety, rng, ks=(20, 30, 40, 50, 60, 75),
                        n_splits=500):
    out = []
    for k in ks:
        elig = {i: v for i, v in roe_by_safety.items() if len(v) >= k}
        if len(elig) < 5:
            out.append((k, np.nan, len(elig)))
            continue
        raw = []
        for _ in range(n_splits):
            h1, h2 = [], []
            for i, v in elig.items():
                idx = rng.permutation(len(v))[:k]
                half = k // 2
                h1.append(v[idx[:half]].mean())
                h2.append(v[idx[half:2 * half]].mean())
            raw.append(np.corrcoef(h1, h2)[0, 1])
        r = np.mean(raw)
        out.append((k, spearman_brown(r), len(elig)))
    return out


def main():
    opp = pd.read_csv(p(PROCESSED / "roe_opportunity_v1.csv"))
    rep = opp[opp.in_study].copy()
    names = (rep.drop_duplicates("nfl_id").set_index("nfl_id")["player_name"].to_dict()
             if "player_name" in rep.columns else {})
    roe_by_safety = {i: g[ROE_COL].values for i, g in rep.groupby("nfl_id")}
    rng = np.random.default_rng(SEED)

    print("=" * 64)
    print("RELIABILITY  -- 30 reported safeties")
    print("=" * 64)
    print(f"Metric: {ROE_COL} | mean opps/safety: "
          f"{np.mean([len(v) for v in roe_by_safety.values()]):.0f}")

    # 1) split-half
    raw, sb, ids = split_half_reliability(roe_by_safety, rng)
    print("\n1) SPLIT-HALF RELIABILITY  (random split, "
          f"{len(raw)} iterations, {len(ids)} safeties)")
    print(f"   raw half-half correlation : median {np.median(raw):.3f}  "
          f"[{np.percentile(raw,10):.3f}, {np.percentile(raw,90):.3f}] (10-90 pct)")
    print(f"   Spearman-Brown (full-len) : median {np.median(sb):.3f}  "
          f"[{np.percentile(sb,10):.3f}, {np.percentile(sb,90):.3f}]")
    r_med = np.median(sb)
    verdict = ("strong -- ROE looks like a repeatable skill" if r_med >= 0.7 else
               "moderate -- some stable signal, meaningful noise" if r_med >= 0.4 else
               "weak -- ROE does not demonstrably stabilize at this sample")
    print(f"   READ: {verdict}")

    # 2) bootstrap CIs
    cis = bootstrap_cis(roe_by_safety, names, rng)
    cis.to_csv(p(PROCESSED / "roe_with_cis_v1.csv"), index=False)
    n_sig = int((cis.ci_lo > 0).sum() + (cis.ci_hi < 0).sum())
    print("\n2) BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print(f"   safeties whose CI excludes 0 (distinguishable from average): "
          f"{n_sig} of {len(cis)}")
    print(f"   {'safety':<22}{'opps':>5}{'ROE':>7}{'95% CI':>18}")
    for _, r in cis.head(8).iterrows():
        print(f"   {str(r.player_name):<22}{int(r.n_opps):>5}{r.roe:>7.2f}"
              f"   [{r.ci_lo:>5.2f}, {r.ci_hi:>5.2f}]")
    # do top and middle overlap?
    top = cis.iloc[0]; mid = cis.iloc[len(cis)//2]
    overlap = not (top.ci_lo > mid.ci_hi)
    print(f"   top ({top.player_name}) vs median-rank ({mid.player_name}) CIs "
          f"{'OVERLAP' if overlap else 'SEPARATE'} -- "
          f"{'rank gap not established' if overlap else 'rank gap is real'}")

    # 3) stabilization curve
    curve = stabilization_curve(roe_by_safety, rng)
    print("\n3) STABILIZATION CURVE  (reliability vs opportunities per safety)")
    print(f"   {'k opps':>7}{'reliability(SB)':>18}{'n safeties':>12}")
    for k, rel, n in curve:
        rel_s = f"{rel:.3f}" if not np.isnan(rel) else "  n/a"
        print(f"   {k:>7}{rel_s:>18}{n:>12}")
    pd.DataFrame(curve, columns=["k_opps", "reliability_sb", "n_safeties"]) \
        .to_csv(p(PROCESSED / "stabilization_curve_v1.csv"), index=False)

    print("\nWrote: roe_with_cis_v1.csv, stabilization_curve_v1.csv")
    print("\nInterpretation is the deliverable, not the number. Report whatever")
    print("this says; a weak result is a finding, not a failure.")


if __name__ == "__main__":
    main()