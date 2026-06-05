"""
Gridlock 2.0 — Configuration
All paths, constants, and hyperparameters.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "train.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")
SAMPLE_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
SUB_PATH   = os.path.join(BASE_DIR, "submission.csv")

# ─── Constants ────────────────────────────────────────────────────────────────
SMOOTH_K    = 5      # Bayesian smoothing prior strength
RANDOM_SEED = 42

# ─── LightGBM base params ────────────────────────────────────────────────────
LGB_BASE_PARAMS = dict(
    objective="regression", metric="rmse",
    n_estimators=12000, learning_rate=0.015,
    min_child_samples=20, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)

# ─── Model configs ───────────────────────────────────────────────────────────
# (max_depth, num_leaves, seed)
LGB_CONFIGS = [(12, 1024, 42), (14, 2048, 84), (15, 4096, 126)]
# (max_depth, seed)
XGB_CONFIGS = [(12, 42), (14, 84)]
# (depth, seed)
CB_CONFIGS  = [(12, 42), (14, 84)]

# ─── Feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "slot", "hour", "is_morn", "is_night",
    "NumberofLanes", "road_enc", "weath_enc", "lv_enc", "lm_enc",
    "temp_f", "temp_b",
    "geo_ts", "geo_m48", "geo_std48",
    "d49_gm", "geo_shift", "d48_early_gm", "early_shift", "early_ratio",
    "geo_ts_shifted", "geo_ts_scaled",
    "lag8", "lag7", "lag6", "lag3m",
    "d49_trend", "d49_last", "d49_max",
    "nbr_d48", "nbr_gm",
]
