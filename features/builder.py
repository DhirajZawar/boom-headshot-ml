"""
File: features/builder.py
Purpose:
- Build per-product features using the synthetic tables generated in scripts/gen_synth.py.
- Implements focusing on:
  - Review velocity (30d/7d/3d + acceleration + rv_micro)
  - Trend features from search volume (30/7/3 sums and pct changes, acceleration, 90d slope, 12m seasonality score)
  - Social aggregates (30/7/3 sums and simple acceleration) + 90d burstiness + half-life proxy
  - PPC features (cpc deltas, breakeven, MoS CPC ratio)
  - Unit economics (unit_margin_pre_ads)
  - Readiness pass/fail vs config thresholds
  - Competition fields: sov_top_competitors and p70_within_mos
- Outputs one row per product for the latest available date.
- Persists outputs to Parquet and optionally DuckDB (table: features) to support downstream models and dashboard.

Assumptions (kept minimal and explicit):
- CVR proxy is not available in synth data; we use a small constant (0.05) for feature prototyping only.
- ASP proxy: use 7-day average price from marketplace_pricing.
- Landed cost: prefer costs.landed if provided, else use cogs.
- PPC deltas computed on cpc_p70 at (cutoff, cutoff-7d, cutoff-14d).
- Referral fee pct and FBA fee from latest day in amazon_fees.
- PPC p70 taken from ppc_estimates.cpc_p70.

These are placeholders for feature shape; replace with real signals as connectors land.
"""
from __future__ import annotations

import os
import sys
import importlib.util
from datetime import timedelta
import pandas as pd
import numpy as np

from features.formulas import unit_margin_pre_ads, breakeven_cpc, mos_cpc_ratio

# Optional: load .env for CLI runs
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try to load YAML config (readiness thresholds)
try:
    from ruamel.yaml import YAML  # type: ignore
except Exception:
    YAML = None  # handled below

# Toggle DuckDB writes via env var: SNIPER_USE_DUCKDB=0 to disable
USE_DUCKDB = os.getenv("SNIPER_USE_DUCKDB", "1").lower() not in ("0", "false")

# Resolve repo root and import storage helpers safely (avoid name collisions with built-in io)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_CURRENT_DIR)
_store_path = os.path.join(_REPO_DIR, 'sniper', 'io', 'store.py')
_spec = importlib.util.spec_from_file_location("store", _store_path)
_store = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_store)
write_parquet = _store.write_parquet
to_duckdb = _store.to_duckdb

# Constants for prototype features (can be moved to config later if needed)
DEFAULT_CVR = 0.05  # 5% conversion rate as a placeholder


def _load_cfg(path: str) -> dict:
    """Load YAML config if available; else return empty dict."""
    try:
        if YAML is None:
            return {}
        yaml = YAML(typ="safe")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f) or {}
    except Exception:
        return {}


def _latest_date(df: pd.DataFrame, date_col: str = "date") -> pd.Timestamp:
    return pd.to_datetime(df[date_col]).max()


def _filter_last_n_days(df: pd.DataFrame, cutoff_date: pd.Timestamp, days: int, date_col: str = "date") -> pd.DataFrame:
    start = cutoff_date - pd.Timedelta(days=days - 1)
    sdf = df.copy()
    sdf[date_col] = pd.to_datetime(sdf[date_col])
    return sdf[(sdf[date_col] >= start) & (sdf[date_col] <= cutoff_date)]


def _daily_sum(df: pd.DataFrame, value_cols: list[str], cutoff: pd.Timestamp) -> pd.DataFrame:
    """Sum value_cols per day per product in df[date <= cutoff]. Returns index [product_id, date]."""
    dd = df.copy()
    dd["date"] = pd.to_datetime(dd["date"])  # ensure ts
    dd = dd[dd["date"] <= cutoff]
    vals = dd.groupby(["product_id", "date"], as_index=False)[value_cols].sum()
    # If multiple value_cols, sum across them into a single 'value'
    if len(value_cols) > 1:
        vals["value"] = vals[value_cols].sum(axis=1)
        vals = vals[["product_id", "date", "value"]]
    else:
        vals = vals.rename(columns={value_cols[0]: "value"})[["product_id", "date", "value"]]
    return vals


def _slope_per_day(df_daily: pd.DataFrame) -> pd.Series:
    """Given daily series per product with columns [product_id, date, value], compute per-day slope.
    Uses linear regression on (date - min_date).days vs value. Returns Series indexed by product_id.
    """
    if df_daily.empty:
        return pd.Series(dtype=float)
    df_daily = df_daily.sort_values(["product_id", "date"])  # ensure order

    def _slope(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return np.nan
        x = (g["date"] - g["date"].min()).dt.days.to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)
        # Avoid degenerate all-equal x
        if np.all(x == 0):
            return np.nan
        # Polyfit degree 1; handle potential numerical issues
        try:
            return float(np.polyfit(x, y, 1)[0])
        except Exception:
            return np.nan

    return df_daily.groupby("product_id").apply(_slope)


def _burstiness_ratio(df_daily: pd.DataFrame) -> pd.Series:
    """Burstiness = peak / median for last window (per product)."""
    if df_daily.empty:
        return pd.Series(dtype=float)

    def _ratio(g: pd.DataFrame) -> float:
        if g.empty:
            return np.nan
        vals = g["value"].to_numpy(dtype=float)
        if len(vals) == 0:
            return np.nan
        med = np.median(vals)
        if med == 0:
            return np.nan
        return float(np.max(vals) / med)

    return df_daily.groupby("product_id").apply(_ratio)


def _half_life_days(df_daily: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Half-life proxy: days back from cutoff needed to accumulate 50% of 90d total when summing from most recent backwards."""
    if df_daily.empty:
        return pd.Series(dtype=float)
    df_daily = df_daily.sort_values(["product_id", "date"])  # ascending

    def _hl(g: pd.DataFrame) -> float:
        if g.empty:
            return np.nan
        # Reindex to daily continuity to make days difference precise
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        gg = g.set_index("date").reindex(idx, fill_value=0.0)
        gg.index.name = "date"
        gg = gg.reset_index()
        # Sum from most recent backwards
        total = float(gg["value"].sum())
        if total <= 0:
            return np.nan
        gg = gg.sort_values("date", ascending=False)
        gg["cum"] = gg["value"].cumsum()
        half_idx = gg.index[gg["cum"] >= (0.5 * total)]
        if len(half_idx) == 0:
            return np.nan
        days = int(half_idx[0])  # since sorted by most recent -> position is days back
        return float(days)

    return df_daily.groupby("product_id").apply(_hl)


def build_features(date_cutoff: str | None = None) -> pd.DataFrame:
    """Build per-product features for the latest date (or provided date_cutoff, YYYY-MM-DD).

    Returns: DataFrame with one row per product_id.
    """
    base_dir = _REPO_DIR
    raw_dir = os.path.join(base_dir, "data", "raw")

    # Load config thresholds (readiness)
    cfg = _load_cfg(os.path.join(base_dir, "config", "defaults.yaml"))
    cfg_readiness = (cfg.get("readiness") or {}) if isinstance(cfg, dict) else {}

    # Load required tables from Parquet generated by scripts/gen_synth.py
    search = pd.read_parquet(os.path.join(raw_dir, "search_trend_history.parquet"))
    reviews = pd.read_parquet(os.path.join(raw_dir, "review_history.parquet"))
    social = pd.read_parquet(os.path.join(raw_dir, "social_mention_history.parquet"))
    price = pd.read_parquet(os.path.join(raw_dir, "marketplace_pricing.parquet"))
    costs = pd.read_parquet(os.path.join(raw_dir, "costs.parquet"))
    fees = pd.read_parquet(os.path.join(raw_dir, "amazon_fees.parquet"))
    ppc = pd.read_parquet(os.path.join(raw_dir, "ppc_estimates.parquet"))
    readiness = pd.read_parquet(os.path.join(raw_dir, "retail_readiness_scores.parquet"))
    comp = pd.read_parquet(os.path.join(raw_dir, "keyword_competition_stats.parquet"))

    # Determine cutoff date
    latest = pd.to_datetime(search["date"]).max()
    cutoff = pd.to_datetime(date_cutoff) if date_cutoff else latest

    # Limit to windows
    s30 = _filter_last_n_days(search, cutoff, 30)
    s7 = _filter_last_n_days(search, cutoff, 7)
    s3 = _filter_last_n_days(search, cutoff, 3)
    s90 = _filter_last_n_days(search, cutoff, 90)
    s365 = _filter_last_n_days(search, cutoff, 365)

    r30 = _filter_last_n_days(reviews, cutoff, 30)
    r7 = _filter_last_n_days(reviews, cutoff, 7)
    r3 = _filter_last_n_days(reviews, cutoff, 3)

    soc30 = _filter_last_n_days(social, cutoff, 30)
    soc7 = _filter_last_n_days(social, cutoff, 7)
    soc3 = _filter_last_n_days(social, cutoff, 3)
    soc90 = _filter_last_n_days(social, cutoff, 90)

    p7 = _filter_last_n_days(price, cutoff, 7)

    # Latest rows for costs, fees, ppc, readiness, competition
    def _latest_per_product(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])  # ensure ts
        d = d[d["date"] <= cutoff]
        return d.sort_values(["product_id", "date"]).groupby("product_id").tail(1)

    c_last = _latest_per_product(costs)
    f_last = _latest_per_product(fees)
    ppc_last = _latest_per_product(ppc)
    ready_last = _latest_per_product(readiness)
    comp_last = _latest_per_product(comp)

    # Aggregations
    # Trend features (sum of search volume)
    trend_30 = s30.groupby("product_id")["search_volume"].sum().rename("trend_sum_30d")
    trend_7 = s7.groupby("product_id")["search_volume"].sum().rename("trend_sum_7d")
    trend_3 = s3.groupby("product_id")["search_volume"].sum().rename("trend_sum_3d")

    # Percent changes and simple acceleration
    trend_pct_7_vs_30 = (trend_7 / (trend_30 / (30 / 7))).rename("trend_pct_7_vs_30_norm")
    trend_pct_3_vs_7 = (trend_3 / (trend_7 / (7 / 3))).rename("trend_pct_3_vs_7_norm")
    trend_accel = (trend_pct_3_vs_7 / trend_pct_7_vs_30).rename("trend_accel_7to3")

    # 90-day trend slope (per-day change) using daily sums across keywords
    s90_daily = _daily_sum(s90, ["search_volume"], cutoff)
    trend_slope_90d = _slope_per_day(s90_daily).rename("trend_slope_90d")

    # 12-month seasonality score (max month sum / median month sum) or NaN if insufficient data
    def _seasonality_score_12m(df_365: pd.DataFrame) -> pd.Series:
        if df_365.empty:
            return pd.Series(dtype=float)
        d = df_365.copy()
        d["date"] = pd.to_datetime(d["date"])  # ensure ts
        d = d[d["date"] >= (cutoff - pd.Timedelta(days=364))]
        if d.empty:
            return pd.Series(dtype=float)
        # monthly sums per product
        d["month"] = d["date"].dt.to_period("M")
        m = d.groupby(["product_id", "month"])['search_volume'].sum().reset_index()
        def _score(g: pd.DataFrame) -> float:
            vals = g['search_volume'].to_numpy(dtype=float)
            if len(vals) < 3:
                return np.nan
            med = np.median(vals)
            if med == 0:
                return np.nan
            return float(np.max(vals) / med)
        return m.groupby("product_id").apply(_score)

    seasonality_score_12m = _seasonality_score_12m(s365).rename("seasonality_score_12m")

    # Review velocity
    def _rv(group: pd.DataFrame, days: int) -> float:
        g = group.sort_values("date")
        first = g.iloc[0]["total_reviews"]
        last = g.iloc[-1]["total_reviews"]
        return (last - first) / max(1, (len(g) - 1))

    rv_30 = r30.groupby("product_id").apply(lambda g: _rv(g, 30)).rename("rv_30d")
    rv_7 = r7.groupby("product_id").apply(lambda g: _rv(g, 7)).rename("rv_7d")
    rv_3 = r3.groupby("product_id").apply(lambda g: _rv(g, 3)).rename("rv_3d")
    rv_accel = (rv_3 / rv_7.replace(0, np.nan)).rename("rv_accel_7to3")

    # rv_micro: 1-day delta (latest total_reviews - previous day)
    def _rv_micro(df_rev: pd.DataFrame) -> pd.Series:
        if df_rev.empty:
            return pd.Series(dtype=float)
        d = df_rev.copy()
        d["date"] = pd.to_datetime(d["date"])  # ensure ts
        d = d[d["date"] <= cutoff]
        def _delta(g: pd.DataFrame) -> float:
            gg = g.sort_values("date")
            if len(gg) < 2:
                return np.nan
            return float(gg.iloc[-1]["total_reviews"] - gg.iloc[-2]["total_reviews"])
        return d.groupby("product_id").apply(_delta)

    rv_micro = _rv_micro(reviews).rename("rv_micro")

    # Social aggregates
    soc_cols = ["tiktok_mentions", "ig_mentions", "yt_mentions", "reddit_mentions"]
    def _sum_cols(df: pd.DataFrame) -> pd.Series:
        return df.groupby("product_id")[soc_cols].sum().sum(axis=1)

    soc_sum_30 = _sum_cols(soc30).rename("social_sum_30d")
    soc_sum_7 = _sum_cols(soc7).rename("social_sum_7d")
    soc_sum_3 = _sum_cols(soc3).rename("social_sum_3d")
    soc_accel = (soc_sum_3 / (soc_sum_7.replace(0, np.nan) / (7 / 3))).rename("social_accel_7to3")

    # Social engagement metrics (means and growth)
    if "engagement_rate" in social.columns:
        eng_avg_30 = soc30.groupby("product_id")["engagement_rate"].mean().rename("social_engagement_avg_30d")
        eng_avg_7 = soc7.groupby("product_id")["engagement_rate"].mean().rename("social_engagement_avg_7d")
        eng_avg_3 = soc3.groupby("product_id")["engagement_rate"].mean().rename("social_engagement_avg_3d")
        # Ratio 7d vs 30d (both are means, so direct ratio is fine)
        eng_pct_7_vs_30 = (eng_avg_7 / eng_avg_30.replace(0, np.nan)).rename("social_engagement_pct_7_vs_30")
        # Acceleration 7->3 (3d mean vs 7d mean)
        eng_accel = (eng_avg_3 / eng_avg_7.replace(0, np.nan)).rename("social_engagement_accel_7to3")
    else:
        # Empty series fallback if engagement_rate missing
        eng_avg_30 = pd.Series(dtype=float, name="social_engagement_avg_30d")
        eng_avg_7 = pd.Series(dtype=float, name="social_engagement_avg_7d")
        eng_avg_3 = pd.Series(dtype=float, name="social_engagement_avg_3d")
        eng_pct_7_vs_30 = pd.Series(dtype=float, name="social_engagement_pct_7_vs_30")
        eng_accel = pd.Series(dtype=float, name="social_engagement_accel_7to3")

    # Social 90d burstiness and half-life
    soc90_daily = _daily_sum(soc90, soc_cols, cutoff)
    social_burstiness_90d = _burstiness_ratio(soc90_daily).rename("social_burstiness_90d")
    social_half_life_90d = _half_life_days(soc90_daily, cutoff).rename("social_half_life_90d")

    # ASP proxy: 7-day mean price per product
    asp_7 = p7.groupby("product_id")["price"].mean().rename("asp_7d")

    # Unit economics
    econ = (
        c_last.set_index("product_id")[
            ["cogs", "landed", "returns_cost", "promo_drag"]
        ]
        .join(f_last.set_index("product_id")["fba_fee"]).join(f_last.set_index("product_id")["referral_fee_pct"])
        .join(asp_7)
    )

    econ["unit_margin_pre_ads"] = econ.apply(
        lambda r: unit_margin_pre_ads(
            asp=float(r.get("asp_7d")) if pd.notnull(r.get("asp_7d")) else None,
            cogs=float(r.get("cogs")) if pd.notnull(r.get("cogs")) else None,
            landed=float(r.get("landed")) if pd.notnull(r.get("landed")) else None,
            fba=float(r.get("fba_fee")) if pd.notnull(r.get("fba_fee")) else None,
            referral_pct=float(r.get("referral_fee_pct")) if pd.notnull(r.get("referral_fee_pct")) else None,
            returns_cost=float(r.get("returns_cost")) if pd.notnull(r.get("returns_cost")) else None,
            promo_drag=float(r.get("promo_drag")) if pd.notnull(r.get("promo_drag")) else None,
        ),
        axis=1,
    )

    # PPC join (latest p70) and derived ratios
    ppcj = ppc_last.set_index("product_id")["cpc_p70"].rename("cpc_p70")
    econ["breakeven_cpc"] = econ.apply(
        lambda r: breakeven_cpc(r.get("unit_margin_pre_ads"), DEFAULT_CVR), axis=1
    )
    econ = econ.join(ppcj)
    econ["mos_cpc_ratio"] = econ.apply(
        lambda r: mos_cpc_ratio(r.get("unit_margin_pre_ads"), DEFAULT_CVR, r.get("cpc_p70"), k=0.8),
        axis=1,
    )

    # PPC deltas (based on cpc_p70 at cutoff vs. on/before cutoff-7d and cutoff-14d)
    ppc_idx = ppc.copy()
    ppc_idx["date"] = pd.to_datetime(ppc_idx["date"])  # ensure ts

    def _cpc_at_or_before(pid: str, ts: pd.Timestamp) -> float:
        sub = ppc_idx[(ppc_idx["product_id"] == pid) & (ppc_idx["date"] <= ts)]
        if sub.empty:
            return np.nan
        return float(sub.sort_values("date").iloc[-1]["cpc_p70"])  # use p70 consistently

    products = sorted(set(ppc_idx["product_id"]))
    cpc_now = {pid: _cpc_at_or_before(pid, cutoff) for pid in products}
    cpc_7 = {pid: _cpc_at_or_before(pid, cutoff - pd.Timedelta(days=7)) for pid in products}
    cpc_14 = {pid: _cpc_at_or_before(pid, cutoff - pd.Timedelta(days=14)) for pid in products}

    cpc_delta_7d = pd.Series({pid: (cpc_now[pid] - cpc_7[pid]) if pd.notnull(cpc_7[pid]) else np.nan for pid in products}, name="cpc_delta_7d")
    cpc_delta_14d = pd.Series({pid: (cpc_now[pid] - cpc_14[pid]) if pd.notnull(cpc_14[pid]) else np.nan for pid in products}, name="cpc_delta_14d")

    # Readiness pass/fail from config thresholds
    def _readiness_row_ok(row: pd.Series) -> dict:
        min_images = cfg_readiness.get("min_images")
        require_video = cfg_readiness.get("require_video")
        keyword_cov_min = cfg_readiness.get("keyword_coverage_min")
        images_ok = (row.get("images_count") >= min_images) if min_images is not None else np.nan
        video_ok = (bool(row.get("has_video")) if require_video else True) if require_video is not None else np.nan
        kcov = row.get("keyword_coverage_pct")
        coverage_ok = (kcov >= keyword_cov_min) if (keyword_cov_min is not None and pd.notnull(kcov)) else np.nan
        # readiness_pass only if all checks are True; if any is nan, result is nan
        checks = [images_ok, video_ok, coverage_ok]
        if any(pd.isna(x) for x in checks):
            readiness_pass = np.nan
        else:
            readiness_pass = bool(all(checks))
        return {
            "readiness_images_ok": images_ok,
            "readiness_video_ok": video_ok,
            "readiness_coverage_ok": coverage_ok,
            "readiness_pass": readiness_pass,
        }

    ready_feat = ready_last.set_index("product_id").apply(_readiness_row_ok, axis=1, result_type="expand")

    # Readiness uplift estimate (simple placeholder: +5% CVR if video present)
    readiness_cvr_uplift_pct = ready_last.set_index("product_id").apply(
        lambda r: 0.05 if bool(r.get("has_video", False)) else 0.0, axis=1
    ).rename("readiness_cvr_uplift_pct")

    # Competition fields
    comp_feat = comp_last.set_index("product_id")["sov_top_competitors"].rename("sov_top_competitors")

    # Combine all features
    feats = (
        pd.DataFrame(index=trend_30.index)
        .join([trend_30, trend_7, trend_3, trend_pct_7_vs_30, trend_pct_3_vs_7, trend_accel])
        .join(trend_slope_90d)
        .join(seasonality_score_12m)
        .join([rv_30, rv_7, rv_3, rv_accel, rv_micro])
        .join([soc_sum_30, soc_sum_7, soc_sum_3, soc_accel])
        .join([eng_avg_30, eng_avg_7, eng_avg_3, eng_pct_7_vs_30, eng_accel])
        .join([social_burstiness_90d, social_half_life_90d])
        .join(econ[["asp_7d", "unit_margin_pre_ads", "breakeven_cpc", "cpc_p70", "mos_cpc_ratio"]])
        .join(cpc_delta_7d)
        .join(cpc_delta_14d)
        .join(ready_feat)
        .join(readiness_cvr_uplift_pct)
        .join(comp_feat)
    )

    # Derived boolean from mos_cpc_ratio
    feats["p70_within_mos"] = feats["mos_cpc_ratio"] >= 1.0

    feats = feats.reset_index().rename(columns={"index": "product_id"})
    feats["date"] = cutoff

    # Ensure numeric columns are floats
    for col in feats.columns:
        if col not in ["product_id", "date", "p70_within_mos", "readiness_pass", "readiness_images_ok", "readiness_video_ok", "readiness_coverage_ok"]:
            feats[col] = pd.to_numeric(feats[col], errors="coerce")

    # Cast booleans cleanly where applicable
    if "p70_within_mos" in feats:
        feats["p70_within_mos"] = feats["p70_within_mos"].astype("boolean")
    if "readiness_pass" in feats:
        feats["readiness_pass"] = feats["readiness_pass"].astype("boolean")

    return feats


def persist_features(df: pd.DataFrame, base_dir: str | None = None) -> str:
    """Persist features to Parquet and optionally DuckDB. Returns Parquet path."""
    base = base_dir or _REPO_DIR
    processed_dir = os.path.join(base, "data", "processed")
    features_path = os.path.join(processed_dir, "features.parquet")

    # Write Parquet
    write_parquet(df, features_path)

    # Append to DuckDB if enabled
    if USE_DUCKDB:
        duckdb_path = os.path.join(processed_dir, "sniper_db.duckdb")
        to_duckdb(df, "features", db_path=duckdb_path)

    return features_path


if __name__ == "__main__":
    df = build_features()
    parquet_path = persist_features(df, _REPO_DIR)
    print(df.head())
    print(f"Built {len(df)} feature rows for date {df['date'].iloc[0].date()}.")
    print(f"Saved features to: {parquet_path}")
    if USE_DUCKDB:
        print("Also appended to DuckDB table: features")