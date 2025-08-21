import unittest
import os
import importlib.util

# Load project features/formulas module directly by path to avoid package-name clashes
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
FORMULAS_PATH = os.path.join(REPO_DIR, 'features', 'formulas.py')
spec = importlib.util.spec_from_file_location("proj_features_formulas", FORMULAS_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

unit_margin_pre_ads = mod.unit_margin_pre_ads
breakeven_cpc = mod.breakeven_cpc
mos_cpc_ratio = mod.mos_cpc_ratio


class TestFormulas(unittest.TestCase):
    def test_unit_margin_pre_ads_with_landed(self):
        m = unit_margin_pre_ads(
            asp=25.0,
            cogs=10.0,
            landed=8.0,
            fba=4.0,
            referral_pct=0.15,
            returns_cost=0.2,
            promo_drag=0.3,
        )
        self.assertAlmostEqual(m, 8.75, places=6)

    def test_unit_margin_pre_ads_without_landed_uses_cogs(self):
        m = unit_margin_pre_ads(
            asp=20.0,
            cogs=7.0,
            landed=None,
            fba=3.0,
            referral_pct=0.10,
            returns_cost=0.1,
            promo_drag=0.2,
        )
        self.assertAlmostEqual(m, 7.7, places=6)

    def test_breakeven_cpc_basic(self):
        self.assertEqual(breakeven_cpc(8.0, 0.05), 0.4)
        self.assertIsNone(breakeven_cpc(None, 0.05))
        self.assertIsNone(breakeven_cpc(8.0, None))
        self.assertIsNone(breakeven_cpc(8.0, 0.0))

    def test_mos_cpc_ratio_basic(self):
        r = mos_cpc_ratio(8.0, 0.05, 0.3, k=0.8)
        self.assertAlmostEqual(r, 0.32 / 0.3, places=6)
        self.assertIsNone(mos_cpc_ratio(None, 0.05, 0.3))
        self.assertIsNone(mos_cpc_ratio(8.0, 0.05, 0.0))


if __name__ == "__main__":
    unittest.main()