"""
File: sniper/io/schemas.py
Purpose:
- Define strict, typed schemas for all core data entities per the client build-out document.
- Each schema includes a `table_name` for consistent storage (snake_case, aligned to doc wording).
Notes:
- Keep these minimal; extend fields as connectors are added.
- Use these models to document/validate expected columns.
"""

from __future__ import annotations
from typing import Optional, ClassVar
from datetime import date, datetime
from pydantic import BaseModel


# =============================
# Master/reference lists
# =============================
class Product(BaseModel):
    """Products list"""
    table_name: ClassVar[str] = "products"

    product_id: str
    asin: Optional[str] = None
    brand: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    first_instock_date: Optional[date] = None


class Keyword(BaseModel):
    """Keywords list"""
    table_name: ClassVar[str] = "keywords"

    keyword: str
    locale: Optional[str] = None       # e.g., en_US
    marketplace: Optional[str] = None  # e.g., amazon, walmart
    intent: Optional[str] = None       # optional free-text label


# =============================
# Time-series histories
# =============================
class SearchTrendHistory(BaseModel):
    """Search trend history"""
    table_name: ClassVar[str] = "search_trend_history"

    product_id: str
    keyword: str
    date: date
    search_volume: int


class ReviewHistory(BaseModel):
    """Review history"""
    table_name: ClassVar[str] = "review_history"

    product_id: str
    date: date
    total_reviews: int
    rating_avg: float


class SocialMentionHistory(BaseModel):
    """Social mention history"""
    table_name: ClassVar[str] = "social_mention_history"

    product_id: str
    date: date
    tiktok_mentions: int
    ig_mentions: int
    yt_mentions: int
    reddit_mentions: int
    engagement_rate: float


class MarketplacePricing(BaseModel):
    """Marketplace pricing (per marketplace)"""
    table_name: ClassVar[str] = "marketplace_pricing"

    product_id: str
    date: date
    marketplace: str           # e.g., amazon, walmart, etsy, ebay, shopify
    price: float
    in_stock: bool


# =============================
# Costs & Fees (split per doc)
# =============================
class CostsDaily(BaseModel):
    """Costs (COGS, shipping, duties)"""
    table_name: ClassVar[str] = "costs"

    product_id: str
    date: date
    cogs: float
    freight: float
    duties: float
    returns_cost: float
    promo_drag: float
    landed: Optional[float] = None  # optional helper field if precomputed


class AmazonFeesDaily(BaseModel):
    """Amazon fees"""
    table_name: ClassVar[str] = "amazon_fees"

    product_id: str
    date: date
    fba_fee: float
    referral_fee_pct: float


class PPCEstimates(BaseModel):
    """PPC estimates"""
    table_name: ClassVar[str] = "ppc_estimates"

    product_id: str
    keyword: str
    date: date
    cpc_p50: float
    cpc_p70: float
    cpc_p90: float
    click_share: float


class RetailReadinessScores(BaseModel):
    """Retail readiness scores (images, video, A+, keyword coverage)"""
    table_name: ClassVar[str] = "retail_readiness_scores"

    product_id: str
    date: date
    images_count: int
    has_video: bool
    aplus_score: float
    keyword_coverage_pct: float


class KeywordCompetitionStats(BaseModel):
    """Keyword competition stats"""
    table_name: ClassVar[str] = "keyword_competition_stats"

    product_id: str
    date: date
    sov_top_competitors: float
    top10_competitor_reviews_avg: float
    ppc_intensity_score: float
    parity_risk: bool


class StressTestConfig(BaseModel):
    """Scenario stress test configs"""
    table_name: ClassVar[str] = "stress_test_configs"

    scenario_name: str
    cpc_multiplier: float = 1.0
    cvr_multiplier: float = 1.0
    asp_multiplier: float = 1.0
    freight_multiplier: float = 1.0
    fx_multiplier: float = 1.0


class Labels(BaseModel):
    """Labels (historical results)"""
    table_name: ClassVar[str] = "labels"

    product_id: str
    date: Optional[date]  # Optional for pre-summed 30d
    net_profit_30d: Optional[float]
    profit_positive_30d: Optional[bool]


# =============================
# Model runs & predictions
# =============================
class ModelRun(BaseModel):
    """Model runs metadata"""
    table_name: ClassVar[str] = "model_runs"

    run_id: str
    run_ts: datetime
    data_cutoff_date: date
    model_version: str
    notes: Optional[str] = None


class Prediction(BaseModel):
    """Model predictions per product/date"""
    table_name: ClassVar[str] = "predictions"

    run_id: str
    product_id: str
    date: date
    p_profit: float
    net_profit_p50: Optional[float] = None
    net_profit_p90: Optional[float] = None
    shap_top5_json: Optional[str] = None  # serialized reasons
    decision: Optional[str] = None        # Go / Monitor / Discard