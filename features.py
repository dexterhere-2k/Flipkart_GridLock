"""
Gridlock 2.0 — Feature Engineering
Lookup tables, spatial neighbors, and feature builder.
Lookup tables, spatial neighbors, and feature builder.
"""
import numpy as np
import pandas as pd
import pygeohash as pgh
from config import SMOOTH_K


def parse_slot(ts):
    """Convert 'H:M' timestamp to 15-min slot index (0-95)."""
    h, m = map(int, ts.split(":"))
    return h * 4 + m // 15


class FeatureEngine:
    """Builds all lookup tables and generates features for train/test."""

    def __init__(self, train, test):
        """
        Args:
            train: full training DataFrame (must have 'slot' column already)
            test:  full test DataFrame (must have 'slot' column already)
        """
        self.d48 = train[train.day == 48].copy()
        self.d49 = train[train.day == 49].copy()

        # ── Day48 lookup tables ───────────────────────────────────────────
        self.d48_gs_mean  = self.d48.groupby(["geohash", "slot"])["demand"].mean()
        self.d48_geo_mean = self.d48.groupby("geohash")["demand"].mean()
        self.d48_geo_std  = self.d48.groupby("geohash")["demand"].std().fillna(0)
        self.d48_slot_mean = self.d48.groupby("slot")["demand"].mean()
        self.d48_global   = float(self.d48["demand"].mean())

        # ── Day49 calibration tables ──────────────────────────────────────
        self.d49_geo_mean = self.d49.groupby("geohash")["demand"].mean()
        self.d49_global   = float(self.d49["demand"].mean())
        self.geo_shift_map = (self.d49_geo_mean - self.d48_geo_mean).to_dict()

        d48_early_geo_mean = self.d48[self.d48.slot <= 8].groupby("geohash")["demand"].mean()
        self.d48_early_geo_mean = d48_early_geo_mean
        self.early_shift_map = (
            (self.d49_geo_mean - d48_early_geo_mean)
            .replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
        )
        self.early_ratio_map = (
            ((self.d49_geo_mean + 0.02) / (d48_early_geo_mean + 0.02))
            .replace([np.inf, -np.inf], np.nan).fillna(1).clip(0.2, 3.0).to_dict()
        )

        # ── Slot anchor dicts ─────────────────────────────────────────────
        self.d49_s = {
            s: self.d49[self.d49.slot == s].set_index("geohash")["demand"].to_dict()
            for s in range(9)
        }
        self.d48_s = {
            s: self.d48[self.d48.slot == s].set_index("geohash")["demand"].to_dict()
            for s in range(96)
        }

        # ── Day49 trend features ──────────────────────────────────────────
        self.d49_trend_map = {}
        self.d49_last_map = {}
        self.d49_max_map = {}
        for geo, grp in self.d49.groupby("geohash"):
            sorted_grp = grp.sort_values("slot")
            slt = sorted_grp["slot"].values.astype(float)
            val = sorted_grp["demand"].values
            self.d49_last_map[geo] = float(val[-1])
            self.d49_max_map[geo]  = float(val.max())
            if len(val) >= 2:
                sm, vm = slt.mean(), val.mean()
                ss = ((slt - sm) ** 2).sum()
                self.d49_trend_map[geo] = (
                    float(((slt - sm) * (val - vm)).sum() / ss) if ss > 0 else 0.0
                )
            else:
                self.d49_trend_map[geo] = 0.0

        # ── Spatial neighbors ─────────────────────────────────────────────
        all_geos = set(train["geohash"].unique()) | set(test["geohash"].unique())
        print("Computing neighbors…", end=" ", flush=True)
        self.nc = {}
        for geo in all_geos:
            nbrs = set()
            try:
                for d_ in ("top", "bottom", "right", "left"):
                    n = pgh.get_adjacent(geo, d_)
                    if n in all_geos:
                        nbrs.add(n)
                for ch in [("top","right"),("top","left"),("bottom","right"),("bottom","left")]:
                    n = pgh.get_adjacent(pgh.get_adjacent(geo, ch[0]), ch[1])
                    if n in all_geos:
                        nbrs.add(n)
            except Exception:
                pass
            self.nc[geo] = list(nbrs)

        self.d48_gsd = self.d48_gs_mean.to_dict()
        self.nbr_gm_map = {
            geo: float(np.mean([self.d48_geo_mean[n] for n in nbrs if n in self.d48_geo_mean.index]))
            if any(n in self.d48_geo_mean.index for n in nbrs) else self.d48_global
            for geo, nbrs in self.nc.items()
        }
        print("done.")

        # ── Bayesian-smoothed (geo, slot) map ─────────────────────────────
        gs_count = self.d48.groupby(["geohash", "slot"])["demand"].count()
        self.geo_ts_map = {}
        for (geo, s), raw in self.d48_gs_mean.items():
            c  = gs_count[(geo, s)]
            gm = self.d48_geo_mean.get(geo, self.d48_global)
            self.geo_ts_map[(geo, s)] = (c * raw + SMOOTH_K * gm) / (c + SMOOTH_K)

        # ── Temperature median ────────────────────────────────────────────
        self.temp_median = float(train["Temperature"].median())

    def build_features(self, df, lag8_map, lag7_map, lag6_map, lag_global):
        """Build full feature matrix from a DataFrame."""
        df = df.copy()

        # Categorical encoding
        df["road_enc"]  = df["RoadType"].map({"Residential":0,"Street":1,"Highway":2}).fillna(-1).astype(int)
        df["weath_enc"] = df["Weather"].map({"Sunny":0,"Foggy":1,"Rainy":2,"Snowy":3}).fillna(-1).astype(int)
        df["lv_enc"]    = (df["LargeVehicles"] == "Allowed").astype(int)
        df["lm_enc"]    = (df["Landmarks"] == "Yes").astype(int)
        df["temp_f"]    = df["Temperature"].fillna(self.temp_median)
        df["temp_b"]    = pd.cut(df["temp_f"], bins=10, labels=False).astype(float)

        # Time features
        df["hour"]     = df["slot"] // 4
        df["is_morn"]  = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
        df["is_night"] = (df["hour"] < 6).astype(int)

        g, s = df["geohash"].values, df["slot"].values

        # Bayesian-smoothed (geo, slot) demand
        df["geo_ts"] = [
            self.geo_ts_map.get((gi, si), self.d48_geo_mean.get(gi, self.d48_global))
            for gi, si in zip(g, s)
        ]

        # Geo-level stats
        df["geo_m48"]   = df["geohash"].map(self.d48_geo_mean).fillna(self.d48_global)
        df["geo_std48"] = df["geohash"].map(self.d48_geo_std).fillna(float(self.d48_geo_std.mean()))

        # Day49 calibration
        df["d49_gm"]        = df["geohash"].map(self.d49_geo_mean).fillna(self.d49_global)
        df["geo_shift"]     = df["geohash"].map(self.geo_shift_map).fillna(0.0)
        df["d48_early_gm"]  = df["geohash"].map(self.d48_early_geo_mean).fillna(self.d48_global)
        df["early_shift"]   = df["geohash"].map(self.early_shift_map).fillna(0.0)
        df["early_ratio"]   = df["geohash"].map(self.early_ratio_map).fillna(1.0)
        df["geo_ts_shifted"] = np.clip(df["geo_ts"] + df["early_shift"], 0, 1)
        df["geo_ts_scaled"]  = np.clip(df["geo_ts"] * df["early_ratio"], 0, 1)

        # Same-day lags
        df["lag8"]  = [lag8_map.get(gi, lag_global) for gi in g]
        df["lag7"]  = [lag7_map.get(gi, lag_global) for gi in g]
        df["lag6"]  = [lag6_map.get(gi, lag_global) for gi in g]
        df["lag3m"] = (df["lag8"] + df["lag7"] + df["lag6"]) / 3

        # Day49 trend
        df["d49_trend"] = df["geohash"].map(self.d49_trend_map).fillna(0.0)
        df["d49_last"]  = df["geohash"].map(self.d49_last_map).fillna(self.d49_global)
        df["d49_max"]   = df["geohash"].map(self.d49_max_map).fillna(self.d49_global)

        # Neighbor features
        nd = []
        for gi, si in zip(g, s):
            nbrs = self.nc.get(gi, [])
            vals = [self.d48_gsd.get((n, si), np.nan) for n in nbrs]
            vals = [v for v in vals if not np.isnan(v)]
            nd.append(float(np.mean(vals)) if vals else float(self.d48_slot_mean.get(si, self.d48_global)))
        df["nbr_d48"] = nd
        df["nbr_gm"]  = df["geohash"].map(self.nbr_gm_map).fillna(self.d48_global)

        return df

    def prepare_cv_data(self):
        """Return featurized d48 (train) and d49 (val) DataFrames."""
        print("Building features…")
        d48_feat = self.build_features(self.d48, self.d48_s[8], self.d48_s[7], self.d48_s[6], self.d48_global)
        d49_feat = self.build_features(self.d49, self.d49_s[8], self.d49_s[7], self.d49_s[6], self.d49_global)
        return d48_feat, d49_feat

    def prepare_test_data(self, test, d48_feat, d49_feat):
        """Return full train features + test features."""
        train_full = pd.concat([d48_feat, d49_feat], ignore_index=True)
        test_feat  = self.build_features(test, self.d49_s[8], self.d49_s[7], self.d49_s[6], self.d49_global)
        return train_full, test_feat
