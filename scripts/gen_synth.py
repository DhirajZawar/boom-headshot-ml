"""
File: scripts/gen_synth.py
Purpose:
- Generate minimal synthetic data to validate the initial IO and storage pipeline on Day 1.
- Populate one or more basic tables (starting with search trends) in both Parquet and DuckDB formats.
Usage:
- Run this script after setting up your Python environment: `python Sniper_ML/scripts/gen_synth.py`.
- It writes Parquet to `data/raw/` and appends to the DuckDB at `data/processed/sniper.duckdb`.
Notes:
- Keep it simple on Day 1 (search trends only) and expand to other tables on Day 2.
- Synthetic values are rough log-normal draws for plausibility, not realism.
"""




from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
import sys
import os
import importlib.util

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import store module directly to avoid conflict with built-in io
store_path = os.path.join(parent_dir, 'sniper', 'io', 'store.py')
spec = importlib.util.spec_from_file_location("store", store_path)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)
write_parquet = store.write_parquet
to_duckdb = store.to_duckdb

# Load .env automatically for terminal/CI runs
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Toggle DuckDB writes via env var: SNIPER_USE_DUCKDB=0 to disable
USE_DUCKDB = os.getenv("SNIPER_USE_DUCKDB", "1").lower() not in ("0", "false")


def gen_dates(days: int = 120):
    """Generate a list of daily dates ending today."""
    end = date.today()
    return [end - timedelta(days=i) for i in range(days)][::-1]


def gen_products(n: int = 20):
    """Create N synthetic product IDs like P001, P002, ..."""
    return [f"P{i:03d}" for i in range(1, n + 1)]


def persist(df: pd.DataFrame, table_name: str):
    """Persist df to Parquet and optionally DuckDB with canonical absolute paths."""
    # Get the correct path to Sniper_ML directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sniper_ml_dir = os.path.dirname(script_dir)
    
    parquet_path = os.path.join(sniper_ml_dir, "data", "raw", f"{table_name}.parquet")
    write_parquet(df, parquet_path)
    if USE_DUCKDB:
        duckdb_path = os.path.join(sniper_ml_dir, "data", "processed", "sniper_db.duckdb")
        to_duckdb(df, table_name, db_path=duckdb_path)
    return parquet_path


def main():
    products = gen_products(20)
    dates = gen_dates(120)

    # Search trend history (kw1 only)
    rows = []
    for pid in products:
        for d in dates:
            rows.append(
                {
                    "product_id": pid,
                    "keyword": "kw1",
                    "date": d,
                    "search_volume": int(np.random.lognormal(mean=5.0, sigma=0.4)),
                }
            )
    df_search = pd.DataFrame(rows)
    persist(df_search, "search_trend_history")

    # Products list
    df_products = pd.DataFrame(
        {
            "product_id": products,
            "asin": [f"ASIN{i:05d}" for i in range(1, len(products) + 1)],
            "brand": ["Acme"] * len(products),
            "title": [f"Product {i:03d}" for i in range(1, len(products) + 1)],
            "category": ["Home"] * len(products),
            "first_instock_date": [dates[0]] * len(products),
        }
    )
    persist(df_products, "products")

    # Keywords list (single keyword used in synth data)
    df_keywords = pd.DataFrame(
        {
            "keyword": ["kw1"],
            "locale": ["en_US"],
            "marketplace": ["amazon"],
            "intent": ["launch"],
        }
    )
    persist(df_keywords, "keywords")

    # Review history
    rows = []
    for pid in products:
        base = np.random.randint(0, 20)
        total = base
        for d in dates:
            total += np.random.poisson(0.2)
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "total_reviews": int(total),
                    "rating_avg": float(np.clip(np.random.normal(4.2, 0.3), 2.5, 5.0)),
                }
            )
    df_reviews = pd.DataFrame(rows)
    persist(df_reviews, "review_history")

    # Social mention history
    rows = []
    for pid in products:
        for d in dates:
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "tiktok_mentions": int(np.random.poisson(3)),
                    "ig_mentions": int(np.random.poisson(2)),
                    "yt_mentions": int(np.random.poisson(1)),
                    "reddit_mentions": int(np.random.poisson(1)),
                    "engagement_rate": float(np.clip(np.random.beta(2, 20), 0.0, 1.0)),
                }
            )
    df_social = pd.DataFrame(rows)
    persist(df_social, "social_mention_history")

    # Marketplace pricing (amazon only)
    rows = []
    for pid in products:
        price = np.random.uniform(15, 45)
        for d in dates:
            price = max(5.0, price + np.random.normal(0, 0.25))
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "marketplace": "amazon",
                    "price": float(price),
                    "in_stock": bool(np.random.rand() > 0.03),
                }
            )
    df_price = pd.DataFrame(rows)
    persist(df_price, "marketplace_pricing")

    # Costs daily
    rows = []
    for pid in products:
        cogs = np.random.uniform(4, 12)
        freight = np.random.uniform(0.5, 2.0)
        duties = np.random.uniform(0.2, 1.0)
        for d in dates:
            returns_cost = np.random.uniform(0.05, 0.30)
            promo_drag = np.random.uniform(0.0, 0.50)
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "cogs": float(cogs),
                    "freight": float(freight),
                    "duties": float(duties),
                    "returns_cost": float(returns_cost),
                    "promo_drag": float(promo_drag),
                    "landed": float(cogs + freight + duties),
                }
            )
    df_costs = pd.DataFrame(rows)
    persist(df_costs, "costs")

    # Amazon fees daily
    rows = []
    for pid in products:
        fba = np.random.uniform(3.0, 6.0)
        referral = np.random.uniform(0.08, 0.15)
        for d in dates:
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "fba_fee": float(fba),
                    "referral_fee_pct": float(referral),
                }
            )
    df_fees = pd.DataFrame(rows)
    persist(df_fees, "amazon_fees")

    # PPC estimates (kw1)
    rows = []
    for pid in products:
        cpc = np.random.uniform(0.5, 1.5)
        for d in dates:
            cpc = max(0.2, cpc + np.random.normal(0, 0.03))
            rows.append(
                {
                    "product_id": pid,
                    "keyword": "kw1",
                    "date": d,
                    "cpc_p50": float(cpc),
                    "cpc_p70": float(cpc * 1.1),
                    "cpc_p90": float(cpc * 1.25),
                    "click_share": float(np.clip(np.random.beta(2, 8), 0.0, 1.0)),
                }
            )
    df_ppc = pd.DataFrame(rows)
    persist(df_ppc, "ppc_estimates")

    # Retail readiness scores
    rows = []
    for pid in products:
        for d in dates:
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "images_count": int(np.random.randint(4, 9)),
                    "has_video": bool(np.random.rand() > 0.4),
                    "aplus_score": float(np.clip(np.random.normal(0.7, 0.15), 0.0, 1.0)),
                    "keyword_coverage_pct": float(np.clip(np.random.normal(0.75, 0.1), 0.0, 1.0)),
                }
            )
    df_ready = pd.DataFrame(rows)
    persist(df_ready, "retail_readiness_scores")

    # Keyword competition stats
    rows = []
    for pid in products:
        for d in dates:
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "sov_top_competitors": float(np.clip(np.random.normal(0.6, 0.15), 0.0, 1.0)),
                    "top10_competitor_reviews_avg": float(np.random.uniform(50, 2000)),
                    "ppc_intensity_score": float(np.clip(np.random.normal(0.5, 0.2), 0.0, 1.0)),
                    "parity_risk": bool(np.random.rand() > 0.8),
                }
            )
    df_comp = pd.DataFrame(rows)
    persist(df_comp, "keyword_competition_stats")

    # Labels (synthetic outcome)
    rows = []
    for pid in products:
        for d in dates:
            net_profit = np.random.normal(150, 80)
            rows.append(
                {
                    "product_id": pid,
                    "date": d,
                    "net_profit_30d": float(net_profit),
                    "profit_positive_30d": bool(net_profit > 0),
                }
            )
    df_labels = pd.DataFrame(rows)
    persist(df_labels, "labels")

    # Stress test configs (static set)
    df_stress = pd.DataFrame(
        [
            {"scenario_name": "Base", "cpc_multiplier": 1.0, "cvr_multiplier": 1.0, "asp_multiplier": 1.0, "freight_multiplier": 1.0, "fx_multiplier": 1.0},
            {"scenario_name": "HighCPC", "cpc_multiplier": 1.3, "cvr_multiplier": 1.0, "asp_multiplier": 1.0, "freight_multiplier": 1.0, "fx_multiplier": 1.0},
            {"scenario_name": "LowCVR", "cpc_multiplier": 1.0, "cvr_multiplier": 0.8, "asp_multiplier": 1.0, "freight_multiplier": 1.0, "fx_multiplier": 1.0},
        ]
    )
    persist(df_stress, "stress_test_configs")

    # Model run + predictions for latest date
    latest_date = max(dates)
    run_id = f"synth_{latest_date.strftime('%Y%m%d')}"

    df_model_runs = pd.DataFrame(
        {
            "run_id": [run_id],
            "run_ts": [pd.Timestamp.utcnow()],
            "data_cutoff_date": [latest_date],
            "model_version": ["0.0.1-synth"],
            "notes": ["synthetic run"],
        }
    )
    persist(df_model_runs, "model_runs")

    rng = np.random.default_rng(42)
    df_preds = pd.DataFrame(
        {
            "run_id": [run_id] * len(products),
            "product_id": products,
            "date": [latest_date] * len(products),
            "p_profit": rng.uniform(0.2, 0.9, size=len(products)),
            "net_profit_p50": rng.normal(120, 50, size=len(products)),
            "net_profit_p90": rng.normal(220, 70, size=len(products)),
            "shap_top5_json": ["[]"] * len(products),
            "decision": ["Monitor"] * len(products),
        }
    )
    persist(df_preds, "predictions")

    print("Synthetic data generation complete.")
    print("Parquet outputs are under Sniper_ML/data/raw/")
    if USE_DUCKDB:
        print("DuckDB outputs stored in Sniper_ML/data/processed/sniper_db.duckdb")


if __name__ == "__main__":
    main()