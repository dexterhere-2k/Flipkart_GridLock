"""
Gridlock 2.0 — Main Runner
Complete pipeline: load → features → CV → final train → predict → save.
Ensemble GBDT pipeline for traffic demand prediction.

Usage:
    cd gridlock2.0
    python run.py
"""
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import TRAIN_CSV, TEST_CSV, SAMPLE_CSV, SUB_PATH, FEATURE_COLS
from features import parse_slot, FeatureEngine
from models import cross_validate, train_final_and_predict, score_preds, print_score


def main():
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    train  = pd.read_csv(TRAIN_CSV)
    test   = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    sample.columns = ["Index", "true_demand"]

    train["slot"] = train["timestamp"].apply(parse_slot)
    test["slot"]  = test["timestamp"].apply(parse_slot)

    print(f"  train={len(train):,}  test={len(test):,}")

    # ── 2. Build feature engine ───────────────────────────────────────────
    engine = FeatureEngine(train, test)
    d48_feat, d49_feat = engine.prepare_cv_data()

    X_cv_tr,  y_cv_tr  = d48_feat[FEATURE_COLS].values, d48_feat["demand"].values
    X_cv_val, y_cv_val = d49_feat[FEATURE_COLS].values, d49_feat["demand"].values

    # ── 3. Cross-validation ───────────────────────────────────────────────
    cv_results, weights = cross_validate(X_cv_tr, y_cv_tr, X_cv_val, y_cv_val)

    # ── 4. Final training & prediction ────────────────────────────────────
    train_full, test_feat = engine.prepare_test_data(test, d48_feat, d49_feat)
    X_train = train_full[FEATURE_COLS].values
    y_train = train_full["demand"].values
    X_test  = test_feat[FEATURE_COLS].values

    n_d48 = len(engine.d48)
    final_preds = train_final_and_predict(cv_results, weights, X_train, y_train, X_test, n_d48)

    # ── 5. Save submission ────────────────────────────────────────────────
    submission = pd.DataFrame({"Index": test["Index"].values, "demand": final_preds})
    submission.to_csv(SUB_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n✓ Submission saved → {SUB_PATH}  ({len(submission):,} rows, {elapsed:.0f}s)")
    print(f"  Prediction stats: min={final_preds.min():.4f}  max={final_preds.max():.4f}  mean={final_preds.mean():.4f}")

    # ── 6. Sanity check against known ground-truth rows ───────────────────
    check = sample.merge(test[["Index", "geohash", "slot"]], on="Index")
    check = check.merge(submission, on="Index")
    check["error"]   = check["demand"] - check["true_demand"]
    check["abs_err"] = check["error"].abs()

    print("\n=== Validation against known ground-truth rows ===")
    print(check[["Index", "geohash", "slot", "true_demand", "demand", "error"]].to_string())
    print(f"\nMAE (samples) : {check.abs_err.mean():.5f}")
    print(f"RMSE (samples): {np.sqrt((check.error ** 2).mean()):.5f}")


if __name__ == "__main__":
    main()
