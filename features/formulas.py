"""
File: features/formulas.py
Purpose:
- Core, unit-testable formula utilities used by the feature builder and later by decisions.
- Implemented per Tasks §12 with transparent, simple math and minimal assumptions.

Functions:
- unit_margin_pre_ads(asp, cogs, landed, fba, referral_pct, returns_cost, promo_drag) -> float
- breakeven_cpc(unit_margin, cvr) -> float
- mos_cpc_ratio(unit_margin, cvr, p70_cpc, k=0.8) -> float

Notes:
- All args are numeric floats; caller is responsible for providing sane values.
- We guard against divide-by-zero and return None when the formula is undefined.
"""
from __future__ import annotations
from typing import Optional


def unit_margin_pre_ads(
    asp: float,
    cogs: Optional[float],
    landed: Optional[float],
    fba: float,
    referral_pct: float,
    returns_cost: float,
    promo_drag: float,
) -> Optional[float]:
    """Compute unit margin before advertising costs.

    Formula (Tasks §7):
    Unit margin pre-ads = ASP - (COGS + Landed + FBA + Referral_Fee% * ASP + Expected_Returns + Promo_Discount)

    Interpretation:
    - If `landed` is provided, it already represents (cogs + freight + duties). In that case, do not additionally add `cogs`.
    - If `landed` is None, approximate landed = cogs.
    """
    if asp is None or fba is None or referral_pct is None or returns_cost is None or promo_drag is None:
        return None

    # Prefer landed if provided; else fallback to cogs as a minimal proxy
    if landed is not None:
        landed_component = landed
    else:
        # cogs may be None in some upstream datasets
        if cogs is None:
            return None
        landed_component = cogs

    try:
        total_costs = (
            landed_component
            + fba
            + (referral_pct * asp)
            + returns_cost
            + promo_drag
        )
        return float(asp - total_costs)
    except Exception:
        return None


def breakeven_cpc(unit_margin: Optional[float], cvr: Optional[float]) -> Optional[float]:
    """Breakeven CPC = Unit_Margin_PreAds * CVR

    Returns None if inputs are missing or invalid.
    """
    try:
        if unit_margin is None or cvr is None:
            return None
        if cvr <= 0:
            return None
        return float(unit_margin * cvr)
    except Exception:
        return None


def mos_cpc_ratio(
    unit_margin: Optional[float],
    cvr: Optional[float],
    p70_cpc: Optional[float],
    k: float = 0.8,
) -> Optional[float]:
    """Margin-of-Safety CPC Ratio.

    Definition (Tasks §7): Mos CPC Ratio = (Max_CPC_Allowed) / P70_CPC
    - Here, we use a conservative max CPC allowance: k * Breakeven_CPC, where k in (0, 1].
    - If inputs are missing, returns None.
    """
    try:
        be_cpc = breakeven_cpc(unit_margin, cvr)
        if be_cpc is None or p70_cpc is None or p70_cpc <= 0:
            return None
        max_cpc_allowed = k * be_cpc
        return float(max_cpc_allowed / p70_cpc)
    except Exception:
        return None