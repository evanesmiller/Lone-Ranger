"""
Expected-range model + Range Over Expected (ROE) for Lone Ranger.

This is where ROE exists as a number. The expected model predicts distance_closed
from SITUATION only, using pre-throw features. ROE = actual distance_closed -
expected. A safety's ROE, averaged over his opportunities, is his range over
expectation -- the single headline metric of the project.

SCOPE NOTE: an earlier design tried to split ROE into "anticipation" vs "pure
closing" via a second model. That was dropped: the velocity-based anticipation
feature was mechanically coupled to the closing it was meant to explain, and the
alternatives (time-split windows, orientation features) were too noisy for this
sample to support a defensible numerical split. Range is reported as a JOINT
product of reading and closing. The lean-of-a-safety (reader vs athlete) is left
to qualitative framing in the writeup, not a contested number.

POPULATIONS (stated explicitly, on purpose)
  Training population : ALL play-eligible deep safeties (every row in
                        features_deep_v1.csv, ~146 safeties). Anchors "expected"
                        to a league-average deep safety, not to elite starters.
  Reporting population : the 30 safeties with in_study == True (>=50 opps).
                        ROE is only reported for these; sub-threshold ROE is too
                        noisy to rank and is used only to thicken the baseline.

LEAKAGE DISCIPLINE (the core correctness guarantee)
  - Predictors are PRE-THROW only. In-flight outcomes (path_length,
    path_efficiency, net_disp, closing_rate, dist_to_land_arrival) are NEVER
    predictors -- they are consequences of the range we measure.
  - Expected values are OUT-OF-FOLD under SAFETY-GROUPED CV: a safety's expected
    values come from a model trained on folds that EXCLUDE him. This holds for
    the reported 30 too, so their ROE is never computed in-sample.

MODELS (both, to show accuracy vs interpretability)
  - HistGradientBoostingRegressor : flexible, captures interactions.
  - SplineTransformer + Ridge     : GAM-like, interpretable, easy to defend.
  Both run under identical CV, compared vs a predict-the-mean baseline and a
  physics-lite baseline (distance & flight only). High rank agreement between the
  two models = the leaderboard is model-robust, not an artifact of one algorithm.

Usage:
    python expected_model.py --data-dir /path/to/parent
Requires: features_deep_v1.csv, scikit-learn.
"""

import os
import numpy as np
import pandas as pd
from paths import PROCESSED, p
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

TARGET = "distance_closed"
# Situation-only predictors: pure opportunity, nothing the safety controls at
# release beyond where he lined up. This is the whole predictor set now.
PREDICTORS = ["dist_to_land_release", "flight_time_s",
              "depth_at_release", "lateral_from_mid", "is_single_high"]
N_SPLITS = 5
SEED = 7


def add_role_flag(df):
    df = df.copy()
    df["is_single_high"] = (df.role_proxy == "SINGLE_HIGH").astype(float)
    return df


def make_models():
    gb = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=400,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    spline_ridge = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        Ridge(alpha=10.0))
    return {"gbm": gb, "spline_ridge": spline_ridge}


def oof_predict(X, y, groups, model_factory):
    """Out-of-fold predictions under safety-grouped CV. Returns array aligned to X."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(X), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def evaluate(y, pred, label):
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  {label:28s} RMSE {rmse:5.2f}  MAE {mae:5.2f}  R2 {r2:5.3f}")
    return rmse, mae, r2


def main():
    f = add_role_flag(pd.read_csv(p(PROCESSED / "features_deep_v1.csv")))
    groups = f["nfl_id"].values
    y = f[TARGET].values
    X = f[PREDICTORS].values

    print("=" * 66)
    print("EXPECTED-RANGE MODEL  (out-of-fold, safety-grouped CV)")
    print("=" * 66)
    print(f"Training rows: {len(f)}  |  safeties: {f.nfl_id.nunique()}  "
          f"|  reported (in_study): {f[f.in_study].nfl_id.nunique()}")
    print(f"Predictors (situation only): {', '.join(PREDICTORS)}")

    print("\nBaselines (what the model must beat):")
    evaluate(y, np.full_like(y, y.mean()), "predict-the-mean")
    phys = oof_predict(f[["dist_to_land_release", "flight_time_s"]].values, y,
                       groups, lambda: HistGradientBoostingRegressor(
                           max_depth=3, max_iter=300, min_samples_leaf=40,
                           random_state=SEED))
    evaluate(y, phys, "physics-lite (dist+flight)")

    results = f[["opportunity_id", "nfl_id", "role_proxy", "in_study",
                 TARGET]].copy()
    if "player_name" in f.columns:
        results["player_name"] = f["player_name"]
    print("\nExpected model (situation only):")
    for m_name in make_models():
        oof = oof_predict(X, y, groups, lambda mn=m_name: make_models()[mn])
        evaluate(y, oof, m_name)
        results[f"exp_{m_name}"] = oof
        results[f"roe_{m_name}"] = y - oof

    prim = "gbm"
    rep = results[results.in_study].copy()
    per = rep.groupby("nfl_id").agg(
        n_opps=("opportunity_id", "size"),
        roe=(f"roe_{prim}", "mean"),
        roe_spline=("roe_spline_ridge", "mean"),
    ).reset_index()
    if "player_name" in f.columns:
        names = f[["nfl_id", "player_name"]].drop_duplicates("nfl_id")
        per = per.merge(names, on="nfl_id", how="left")
    per = per.sort_values("roe", ascending=False)

    rank_corr = per[["roe", "roe_spline"]].corr(method="spearman").iloc[0, 1]

    out_opp = p(PROCESSED / "roe_opportunity_v1.csv")
    out_per = p(PROCESSED / "roe_per_safety_v1.csv")
    results.to_csv(out_opp, index=False)
    per.to_csv(out_per, index=False)

    print("\n" + "=" * 66)
    print("RANGE OVER EXPECTED  -- reported safeties (30)")
    print("=" * 66)
    print(f"Spearman rank agreement GBM vs spline-ridge: {rank_corr:.3f}")
    print("(high = ranking is model-robust)\n")
    namecol = "player_name" if "player_name" in per.columns else "nfl_id"
    print(f"{'safety':<24}{'opps':>6}{'ROE (yds/play)':>16}")
    for _, r in per.head(12).iterrows():
        print(f"{str(r[namecol]):<24}{int(r.n_opps):>6}{r.roe:>16.2f}")
    print(f"\nWrote per-opportunity ROE: {out_opp}")
    print(f"Wrote per-safety ROE:      {out_per}")
    print("\nROE = yards of distance closed above/below expectation per play,")
    print("given the situation (start distance, flight time, alignment, role).")
    print("Reported as a JOINT measure of reading + closing; no split claimed.")
    print("NOTE: point estimates only -- confidence intervals come in the")
    print("reliability phase; adjacent ranks are within noise until then.")


if __name__ == "__main__":
    main()