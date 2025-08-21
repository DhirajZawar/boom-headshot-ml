import os
import sys
import unittest
import pandas as pd

# Ensure repo root is on sys.path so 'features' package can be imported
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from features.builder import build_features


class TestBuilderSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_dir = os.path.join(REPO_DIR, 'data', 'raw')
        required = [
            'search_trend_history.parquet',
            'review_history.parquet',
            'social_mention_history.parquet',
            'marketplace_pricing.parquet',
            'costs.parquet',
            'amazon_fees.parquet',
            'ppc_estimates.parquet',
            'retail_readiness_scores.parquet',
            'keyword_competition_stats.parquet',
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(raw_dir, f))]
        if missing:
            raise unittest.SkipTest(f"Missing required parquet(s): {missing}; run scripts/gen_synth.py first")

    def test_build_features_columns_and_nonempty(self):
        df = build_features()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

        expected_cols = {
            'product_id', 'date',
            'trend_sum_30d', 'trend_sum_7d', 'trend_sum_3d',
            'trend_pct_7_vs_30_norm', 'trend_pct_3_vs_7_norm', 'trend_accel_7to3',
            'trend_slope_90d', 'seasonality_score_12m',
            'rv_30d', 'rv_7d', 'rv_3d', 'rv_accel_7to3', 'rv_micro',
            'social_sum_30d', 'social_sum_7d', 'social_sum_3d', 'social_accel_7to3',
            'social_engagement_avg_30d', 'social_engagement_avg_7d', 'social_engagement_avg_3d',
            'social_engagement_pct_7_vs_30', 'social_engagement_accel_7to3',
            'social_burstiness_90d', 'social_half_life_90d',
            'asp_7d', 'unit_margin_pre_ads', 'breakeven_cpc', 'cpc_p70', 'mos_cpc_ratio',
            'cpc_delta_7d', 'cpc_delta_14d',
            'readiness_images_ok', 'readiness_video_ok', 'readiness_coverage_ok', 'readiness_pass',
            'readiness_cvr_uplift_pct',
            'sov_top_competitors', 'p70_within_mos',
        }
        missing_cols = expected_cols - set(df.columns)
        self.assertFalse(missing_cols, f"Missing expected columns: {missing_cols}")


if __name__ == '__main__':
    unittest.main()