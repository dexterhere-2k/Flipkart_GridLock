"""
Gridlock 2.0 — Model Training & Ensemble
LightGBM + XGBoost + CatBoost ensemble with inverse-RMSE² blending.
LightGBM + XGBoost + CatBoost ensemble with inverse-RMSE² blending.
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import LGB_BASE_PARAMS, LGB_CONFIGS, XGB_CONFIGS, CB_CONFIGS

# Optional boosters
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("[INFO] XGBoost not installed; skipping XGB models.")

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
    print("[INFO] CatBoost not installed; skipping CatBoost models.")


def score_preds(y_true, preds):
    """Compute R², RMSE, MAE."""
    preds = np.clip(preds, 0, 1)
    return {
        "r2":   r2_score(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
        "mae":  mean_absolute_error(y_true, preds),
    }


def print_score(prefix, score, best_iter=None):
    suffix = f"  best_iter={best_iter}" if best_iter else ""
    print(f"{prefix} R²={score['r2']:.4f}  RMSE={score['rmse']:.5f}  MAE={score['mae']:.5f}{suffix}")


def build_model_specs():
    """Build list of (kind, name, params) tuples for all available models."""
    specs = []

    # LightGBM variants
    for depth, leaves, seed in LGB_CONFIGS:
        params = {**LGB_BASE_PARAMS, "max_depth": depth, "num_leaves": leaves, "random_state": seed}
        specs.append(("lgb", f"lgb_d{depth}", params))

    # XGBoost variants
    if XGBRegressor is not None:
        for depth, seed in XGB_CONFIGS:
            params = dict(
                objective="reg:squarederror", eval_metric="rmse",
                n_estimators=12000, learning_rate=0.015, max_depth=depth,
                min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.02, reg_lambda=1.0, tree_method="hist",
                random_state=seed, n_jobs=-1, verbosity=0,
                early_stopping_rounds=300,
            )
            specs.append(("xgb", f"xgb_d{depth}", params))

    # CatBoost variants
    if CatBoostRegressor is not None:
        for depth, seed in CB_CONFIGS:
            params = dict(
                loss_function="RMSE", eval_metric="RMSE",
                iterations=12000, learning_rate=0.015, depth=depth,
                random_seed=seed, l2_leaf_reg=3.0, bootstrap_type="Bernoulli",
                subsample=0.85, allow_writing_files=False, verbose=False,
                od_type="Iter", od_wait=300,
            )
            specs.append(("cat", f"cat_d{depth}", params))

    return specs


def cross_validate(X_tr, y_tr, X_val, y_val):
    """
    Train all models with early stopping on d48→d49 split.
    Returns: list of result dicts, blend weights array.
    """
    specs = build_model_specs()
    cv_results = []

    print("CV (day48 → day49 early)…")
    for kind, name, params in specs:
        print(f"  fitting {name}...", end=" ", flush=True)

        if kind == "lgb":
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(300, verbose=False)],
            )
            best_iter = model.best_iteration_ or params["n_estimators"]
            preds = model.predict(X_val, num_iteration=best_iter)

        elif kind == "xgb":
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = (
                model.best_iteration + 1
                if getattr(model, "best_iteration", None) is not None
                else params["n_estimators"]
            )
            preds = model.predict(X_val, iteration_range=(0, best_iter))

        else:  # catboost
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            best_iter = (
                model.get_best_iteration() + 1
                if model.get_best_iteration() is not None
                else params["iterations"]
            )
            preds = model.predict(X_val)

        score = score_preds(y_val, preds)
        print_score(f"  {name}", score, best_iter)
        cv_results.append({
            "kind": kind, "name": name, "params": params,
            "best_iter": best_iter, "score": score,
            "cv_preds": np.clip(preds, 0, 1),
        })

    # Compute blend weights (inverse squared RMSE)
    raw_weights = np.array([1 / max(r["score"]["rmse"], 1e-9) ** 2 for r in cv_results])
    weights = raw_weights / raw_weights.sum()

    blend_preds = np.column_stack([r["cv_preds"] for r in cv_results]) @ weights
    blend_score = score_preds(y_val, blend_preds)

    print("\nCV blend weights:")
    for r, w in zip(cv_results, weights):
        print(f"  {r['name']}: {w:.3f}")
    print_score("CV blend", blend_score)

    return cv_results, weights


def train_final_and_predict(cv_results, weights, X_train, y_train, X_test, n_train_d48):
    """
    Retrain all models on full data with scaled iterations, predict test.
    Returns: final clipped predictions array.
    """
    print(f"\nFinal ensemble on {len(y_train):,} rows")

    test_pred_parts = []
    for r in cv_results:
        kind, name = r["kind"], r["name"]
        scaled_iter = min(
            max(int(r["best_iter"] * (len(y_train) / n_train_d48) * 1.20), 1500),
            12000,
        )
        print(f"  fitting final {name} with {scaled_iter} iterations...", end=" ", flush=True)

        if kind == "lgb":
            params = {**r["params"], "n_estimators": scaled_iter}
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))

        elif kind == "xgb":
            params = {**r["params"], "n_estimators": scaled_iter}
            params.pop("early_stopping_rounds", None)
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))

        else:  # catboost
            params = {**r["params"], "iterations": scaled_iter, "od_type": None}
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            test_pred_parts.append(np.clip(model.predict(X_test), 0, 1))

        print("done.")

    final_preds = np.clip(np.column_stack(test_pred_parts) @ weights, 0, 1)
    return final_preds
