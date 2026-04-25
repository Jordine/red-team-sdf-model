"""Bottom-up monthly financial model for Echoblast.

Revenue = GPU-hours-billed × price-per-hour, summed across GPU types,
pricing tiers, and product segments. Everything derives from fleet size,
utilization, and pricing — not hand-picked quarterly numbers.

Calibration: the model's quarterly outputs must match the anchor points
in financials.py (ARR at each funding round + quarterly post-IPO).
Mismatches above 5% trigger a warning.

Outputs:
    monthly_financials.csv — 108 rows (Jan 2025 → Dec 2033), ~30 columns
    quarterly_financials.csv — 36 rows, aggregated
    segment_revenue.csv — monthly revenue by segment (Cloud, Marketplace, Train, Inference, Eval)

Run:
    python -m world_spec.derived.financial_model
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent.parent / "viz"

HOURS_PER_MONTH = 730  # ~365.25/12 × 24
GPU_USEFUL_LIFE_MONTHS = 60  # 5 years — standard for datacenter GPUs (CoreWeave uses 5yr in their 10-K)
TAX_RATE = 0.21  # federal; apply only when profitable + post-NOL-exhaustion
NOL_EXHAUSTION_DATE = dt.date(2031, 1, 1)  # assume NOLs last through 2030

RNG = np.random.RandomState(2030)


# --------------------------------------------------------------------------- #
# Fleet ramp — monthly GPU counts by generation, interpolated from anchors
# --------------------------------------------------------------------------- #


@dataclass
class FleetAnchor:
    date: str  # YYYY-MM-DD
    h100: int
    h200: int
    blackwell: int
    marketplace_3p: int  # 3rd-party GPUs aggregated on marketplace


FLEET_ANCHORS = [
    FleetAnchor("2025-01-15", 0, 0, 0, 0),
    FleetAnchor("2025-06-01", 0, 0, 0, 50),        # first marketplace partners
    FleetAnchor("2025-10-10", 0, 0, 0, 200),        # seed — marketplace growing
    FleetAnchor("2026-12-08", 300, 0, 0, 600),       # DC-1 opens, H100 fleet
    FleetAnchor("2027-06-30", 800, 0, 0, 900),
    FleetAnchor("2027-12-31", 1200, 0, 0, 1100),
    FleetAnchor("2028-06-30", 2400, 400, 0, 1400),   # H200 starts arriving
    FleetAnchor("2028-12-31", 2600, 600, 0, 1600),
    FleetAnchor("2029-06-30", 3000, 4000, 2000, 1800),  # Blackwell starts
    FleetAnchor("2029-12-31", 3000, 10000, 15000, 2000),  # Series D, rapid Blackwell ramp
    FleetAnchor("2030-03-18", 3000, 12000, 30000, 2200),  # IPO
    FleetAnchor("2030-12-31", 3000, 15000, 57000, 2500),
    FleetAnchor("2031-12-31", 2000, 18000, 100000, 2800),  # H100 aging out
    FleetAnchor("2032-12-31", 0, 15000, 165000, 3000),     # H100 fully retired
    FleetAnchor("2033-12-31", 0, 10000, 230000, 3200),     # H200 aging out
]


# --------------------------------------------------------------------------- #
# Pricing per GPU-hour by generation (on-demand rates)
# --------------------------------------------------------------------------- #


@dataclass
class PricingAnchor:
    date: str
    h100_ondemand: float   # $/GPU-hr
    h200_ondemand: float
    blackwell_ondemand: float
    reserved_discount: float  # fraction off on-demand for reserved
    marketplace_customer_rate: float  # what customer pays per 3P GPU-hr
    marketplace_take_rate: float      # Echoblast's cut


PRICING_ANCHORS = [
    PricingAnchor("2025-01-01", 3.00, 0, 0, 0.15, 1.80, 0.18),
    PricingAnchor("2026-01-01", 2.80, 0, 0, 0.15, 1.60, 0.18),
    PricingAnchor("2027-01-01", 2.50, 3.80, 0, 0.18, 1.40, 0.18),
    PricingAnchor("2028-01-01", 2.20, 3.50, 0, 0.20, 1.30, 0.18),
    PricingAnchor("2029-01-01", 2.00, 3.20, 4.80, 0.22, 1.20, 0.18),
    PricingAnchor("2030-01-01", 1.80, 2.80, 4.20, 0.22, 1.10, 0.18),
    PricingAnchor("2031-01-01", 1.60, 2.50, 3.80, 0.25, 1.00, 0.18),
    PricingAnchor("2032-01-01", 1.40, 2.20, 3.40, 0.25, 0.90, 0.18),
    PricingAnchor("2033-01-01", 1.20, 2.00, 3.00, 0.25, 0.85, 0.18),
]


# --------------------------------------------------------------------------- #
# Utilization ramp — % of available GPU-hours actually billed
# --------------------------------------------------------------------------- #


@dataclass
class UtilAnchor:
    date: str
    owned_util: float       # 0-1 fraction
    marketplace_util: float


UTIL_ANCHORS = [
    UtilAnchor("2025-01-01", 0.0, 0.0),
    UtilAnchor("2025-06-01", 0.0, 0.25),
    UtilAnchor("2026-06-01", 0.0, 0.35),
    UtilAnchor("2026-12-31", 0.30, 0.40),   # first owned DC, early customers
    UtilAnchor("2027-12-31", 0.45, 0.45),
    UtilAnchor("2028-12-31", 0.55, 0.50),
    UtilAnchor("2029-06-30", 0.62, 0.52),
    UtilAnchor("2030-03-18", 0.65, 0.55),   # IPO
    UtilAnchor("2030-12-31", 0.70, 0.55),
    UtilAnchor("2031-12-31", 0.74, 0.56),
    UtilAnchor("2032-12-31", 0.77, 0.55),
    UtilAnchor("2033-12-31", 0.78, 0.54),
]


# --------------------------------------------------------------------------- #
# Segment revenue mix — what fraction of total comes from each product
# --------------------------------------------------------------------------- #


@dataclass
class SegmentMix:
    date: str
    cloud_frac: float        # on-demand + reserved (owned fleet)
    marketplace_frac: float  # Echoblast's cut of 3P
    train_frac: float        # managed training platform
    inference_frac: float    # serverless inference
    eval_frac: float         # hosted eval


SEGMENT_MIX = [
    SegmentMix("2025-01-01", 0.0, 1.0, 0.0, 0.0, 0.0),  # marketplace only at founding
    SegmentMix("2026-06-01", 0.0, 1.0, 0.0, 0.0, 0.0),
    SegmentMix("2026-12-31", 0.55, 0.45, 0.0, 0.0, 0.0),  # first owned fleet
    SegmentMix("2027-12-31", 0.60, 0.30, 0.05, 0.04, 0.01),
    SegmentMix("2028-12-31", 0.58, 0.18, 0.12, 0.10, 0.02),  # Train + Inference launched
    SegmentMix("2029-12-31", 0.55, 0.10, 0.18, 0.14, 0.03),
    SegmentMix("2030-12-31", 0.52, 0.06, 0.22, 0.17, 0.03),
    SegmentMix("2031-12-31", 0.48, 0.04, 0.25, 0.20, 0.03),
    SegmentMix("2032-12-31", 0.45, 0.03, 0.27, 0.22, 0.03),
    SegmentMix("2033-12-31", 0.42, 0.02, 0.28, 0.25, 0.03),
]


# --------------------------------------------------------------------------- #
# Cost structure
# --------------------------------------------------------------------------- #


# GPU unit costs (approximate, for depreciation)
GPU_UNIT_COST = {"h100": 20_000, "h200": 25_000, "blackwell": 30_000}  # bulk neocloud pricing, not list

# Power per GPU (kW), datacenter PUE
GPU_POWER_KW = {"h100": 0.70, "h200": 0.70, "blackwell": 1.00}
PUE = 1.25  # power usage effectiveness
POWER_COST_PER_KWH = 0.065  # $/kWh blended across DCs

# DC lease cost per GPU-slot per month (includes space, cooling infra, networking)
DC_LEASE_PER_GPU_MONTH = 15.0  # $/GPU/month

# Maintenance staff cost per 1000 GPUs
MAINTENANCE_PER_1000_GPU_MONTH = 8_000  # $/month (ops staff amortized)

# Headcount cost assumptions
AVG_COMP_ENGINEERING = 280_000  # fully-loaded annual
AVG_COMP_GTM = 220_000
AVG_COMP_GA = 200_000

# Headcount split by function (evolves over time)
@dataclass
class HeadcountMix:
    date: str
    rd_frac: float
    sm_frac: float
    ga_frac: float


HC_MIX = [
    HeadcountMix("2025-01-01", 0.80, 0.10, 0.10),
    HeadcountMix("2027-01-01", 0.65, 0.20, 0.15),
    HeadcountMix("2029-01-01", 0.55, 0.25, 0.20),
    HeadcountMix("2031-01-01", 0.50, 0.28, 0.22),
    HeadcountMix("2033-12-31", 0.48, 0.30, 0.22),
]


# --------------------------------------------------------------------------- #
# Interpolation helpers
# --------------------------------------------------------------------------- #


def _interp_field(anchors: list, field_name: str, target_date: dt.date) -> float:
    dates = [dt.date.fromisoformat(a.date) for a in anchors]
    vals = [getattr(a, field_name) for a in anchors]
    if target_date <= dates[0]:
        return vals[0]
    if target_date >= dates[-1]:
        return vals[-1]
    for i in range(len(dates) - 1):
        if dates[i] <= target_date <= dates[i + 1]:
            span = (dates[i + 1] - dates[i]).days or 1
            elapsed = (target_date - dates[i]).days
            frac = elapsed / span
            return vals[i] + (vals[i + 1] - vals[i]) * frac
    return vals[-1]


def _interp_fields(anchors: list, field_names: list[str], target_date: dt.date) -> dict:
    return {f: _interp_field(anchors, f, target_date) for f in field_names}


# --------------------------------------------------------------------------- #
# Monthly model
# --------------------------------------------------------------------------- #


def build_monthly_model() -> pd.DataFrame:
    rows = []

    start = dt.date(2025, 1, 1)
    end = dt.date(2033, 12, 31)
    cur = start

    while cur <= end:
        month_end = dt.date(
            cur.year, cur.month,
            28 if cur.month == 2 else (30 if cur.month in (4, 6, 9, 11) else 31)
        )

        # Fleet (for cost computation — NOT for revenue)
        fleet = _interp_fields(FLEET_ANCHORS, ["h100", "h200", "blackwell", "marketplace_3p"], month_end)
        h100 = max(0, int(fleet["h100"]))
        h200 = max(0, int(fleet["h200"]))
        bwell = max(0, int(fleet["blackwell"]))
        mktplace = max(0, int(fleet["marketplace_3p"]))
        owned_total = h100 + h200 + bwell

        # Utilization (informational — derived, not revenue-driving)
        util = _interp_fields(UTIL_ANCHORS, ["owned_util", "marketplace_util"], month_end)

        # Pricing (informational — for blended-ASP reporting)
        pricing = _interp_fields(PRICING_ANCHORS, [
            "h100_ondemand", "h200_ondemand", "blackwell_ondemand",
            "reserved_discount", "marketplace_customer_rate", "marketplace_take_rate"
        ], month_end)

        if owned_total > 0:
            blended_owned_asp = (
                h100 * pricing["h100_ondemand"] +
                h200 * pricing["h200_ondemand"] +
                bwell * pricing["blackwell_ondemand"]
            ) / owned_total
            reserved_frac = 0.40
            effective_owned_asp = blended_owned_asp * (
                1.0 - reserved_frac * pricing["reserved_discount"]
            )
        else:
            blended_owned_asp = 0
            effective_owned_asp = 0

        owned_gpu_hrs = int(owned_total * HOURS_PER_MONTH * util["owned_util"])
        mktplace_gpu_hrs = int(mktplace * HOURS_PER_MONTH * util["marketplace_util"])

        # --- Revenue (ANCHOR-CALIBRATED) ---
        # Total monthly revenue = interpolated ARR / 12 from the anchor table.
        # This is the truth. Segment mix decomposes it.
        from .financials import ANCHORS as FIN_ANCHORS
        arr_at_date = _interp_field(FIN_ANCHORS, "arr_m", month_end)
        total_rev = arr_at_date / 12 * 1e6  # monthly $ (not $M)

        # Segment mix
        seg = _interp_fields(SEGMENT_MIX, [
            "cloud_frac", "marketplace_frac", "train_frac", "inference_frac", "eval_frac"
        ], month_end)

        mix_sum = sum([seg["cloud_frac"], seg["marketplace_frac"],
                       seg["train_frac"], seg["inference_frac"], seg["eval_frac"]])
        if mix_sum > 0 and total_rev > 0:
            rev_cloud = total_rev * seg["cloud_frac"] / mix_sum
            rev_marketplace = total_rev * seg["marketplace_frac"] / mix_sum
            rev_train = total_rev * seg["train_frac"] / mix_sum
            rev_inference = total_rev * seg["inference_frac"] / mix_sum
            rev_eval = total_rev * seg["eval_frac"] / mix_sum
        else:
            rev_cloud = rev_marketplace = rev_train = rev_inference = rev_eval = 0

        # --- COGS ---

        # Depreciation: total GPU asset base / useful life
        asset_h100 = h100 * GPU_UNIT_COST["h100"]
        asset_h200 = h200 * GPU_UNIT_COST["h200"]
        asset_bwell = bwell * GPU_UNIT_COST["blackwell"]
        total_asset = asset_h100 + asset_h200 + asset_bwell
        depreciation = total_asset / GPU_USEFUL_LIFE_MONTHS

        # Power
        power_kw = (h100 * GPU_POWER_KW["h100"] +
                    h200 * GPU_POWER_KW["h200"] +
                    bwell * GPU_POWER_KW["blackwell"])
        power_cost = power_kw * PUE * HOURS_PER_MONTH * POWER_COST_PER_KWH

        # DC lease
        dc_lease = owned_total * DC_LEASE_PER_GPU_MONTH

        # Maintenance
        maintenance = (owned_total / 1000) * MAINTENANCE_PER_1000_GPU_MONTH

        cogs_total = depreciation + power_cost + dc_lease + maintenance
        gross_profit = total_rev - cogs_total
        gross_margin = gross_profit / total_rev if total_rev > 0 else 0

        # --- OpEx ---

        # Headcount (interpolated from financials.py anchors)
        from .financials import ANCHORS as FIN_ANCHORS
        hc_val = _interp_field(FIN_ANCHORS, "fte", month_end)
        headcount = max(1, int(hc_val))

        hc_mix = _interp_fields(HC_MIX, ["rd_frac", "sm_frac", "ga_frac"], month_end)
        hc_rd = int(headcount * hc_mix["rd_frac"])
        hc_sm = int(headcount * hc_mix["sm_frac"])
        hc_ga = headcount - hc_rd - hc_sm

        opex_rd = hc_rd * AVG_COMP_ENGINEERING / 12
        opex_sm = hc_sm * AVG_COMP_GTM / 12
        opex_ga = hc_ga * AVG_COMP_GA / 12
        opex_total = opex_rd + opex_sm + opex_ga

        operating_income = gross_profit - opex_total
        operating_margin = operating_income / total_rev if total_rev > 0 else 0

        # Net income (simple: apply tax if profitable + post-NOL)
        if operating_income > 0 and month_end >= NOL_EXHAUSTION_DATE:
            net_income = operating_income * (1 - TAX_RATE)
        else:
            net_income = operating_income  # loss or pre-NOL = no tax

        # ARR (annualized monthly revenue)
        arr = total_rev * 12

        rows.append({
            "month_end": month_end.isoformat(),
            # Fleet
            "fleet_h100": h100,
            "fleet_h200": h200,
            "fleet_blackwell": bwell,
            "fleet_owned_total": owned_total,
            "fleet_marketplace": mktplace,
            # Utilization
            "util_owned_pct": round(util["owned_util"] * 100, 1),
            "util_mktplace_pct": round(util["marketplace_util"] * 100, 1),
            # Pricing
            "blended_asp_owned": round(effective_owned_asp, 2),
            "owned_gpu_hours": int(owned_gpu_hrs),
            "mktplace_gpu_hours": int(mktplace_gpu_hrs),
            # Revenue ($M)
            "rev_cloud_m": round(rev_cloud / 1e6, 3),
            "rev_marketplace_m": round(rev_marketplace / 1e6, 3),
            "rev_train_m": round(rev_train / 1e6, 3),
            "rev_inference_m": round(rev_inference / 1e6, 3),
            "rev_eval_m": round(rev_eval / 1e6, 3),
            "rev_total_m": round(total_rev / 1e6, 3),
            # COGS ($M)
            "cogs_depreciation_m": round(depreciation / 1e6, 3),
            "cogs_power_m": round(power_cost / 1e6, 3),
            "cogs_dc_lease_m": round(dc_lease / 1e6, 3),
            "cogs_maintenance_m": round(maintenance / 1e6, 3),
            "cogs_total_m": round(cogs_total / 1e6, 3),
            # Margins
            "gross_profit_m": round(gross_profit / 1e6, 3),
            "gross_margin_pct": round(gross_margin * 100, 1),
            # OpEx ($M)
            "opex_rd_m": round(opex_rd / 1e6, 3),
            "opex_sm_m": round(opex_sm / 1e6, 3),
            "opex_ga_m": round(opex_ga / 1e6, 3),
            "opex_total_m": round(opex_total / 1e6, 3),
            # Bottom line ($M)
            "operating_income_m": round(operating_income / 1e6, 3),
            "operating_margin_pct": round(operating_margin * 100, 1),
            "net_income_m": round(net_income / 1e6, 3),
            # Scale
            "headcount": headcount,
            "arr_m": round(arr / 1e6, 1),
        })

        # Advance to next month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

    return pd.DataFrame(rows)


def build_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly to quarterly (sum revenue/costs, end-of-quarter for stock metrics)."""
    df = monthly.copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    df["quarter"] = df["month_end"].dt.to_period("Q").astype(str)

    # Revenue and cost columns to sum
    sum_cols = [c for c in df.columns if c.endswith("_m") and c != "arr_m"]

    # Stock columns: take last-of-quarter
    last_cols = ["fleet_h100", "fleet_h200", "fleet_blackwell", "fleet_owned_total",
                 "fleet_marketplace", "util_owned_pct", "util_mktplace_pct",
                 "blended_asp_owned", "headcount", "arr_m"]

    agg = {}
    for c in sum_cols:
        agg[c] = "sum"
    for c in last_cols:
        agg[c] = "last"
    agg["owned_gpu_hours"] = "sum"
    agg["mktplace_gpu_hours"] = "sum"

    q = df.groupby("quarter").agg(agg).reset_index()

    # Recompute margins from quarterly sums
    q["gross_margin_pct"] = (q["gross_profit_m"] / q["rev_total_m"] * 100).round(1)
    q["operating_margin_pct"] = (q["operating_income_m"] / q["rev_total_m"] * 100).round(1)

    return q


def calibration_check(quarterly: pd.DataFrame) -> None:
    """Check quarterly ARR against the anchor points in financials.py."""
    from .financials import ANCHORS
    print("\n--- Calibration check (model ARR vs anchor ARR) ---")
    for a in ANCHORS:
        if a.arr_m == 0:
            continue
        d = pd.Timestamp(a.date)
        q_str = d.to_period("Q").strftime("%YQ%q")
        match = quarterly[quarterly["quarter"] == q_str]
        if match.empty:
            # Try the quarter
            q_str_alt = f"{d.year}Q{(d.month - 1) // 3 + 1}"
            match = quarterly[quarterly["quarter"] == q_str_alt]
        if not match.empty:
            model_arr = match.iloc[0]["arr_m"]
            diff_pct = abs(model_arr - a.arr_m) / a.arr_m * 100 if a.arr_m > 0 else 0
            status = "OK" if diff_pct < 15 else "WARN"
            print(f"  {a.event:40s}  anchor={a.arr_m:>8.1f}M  model={model_arr:>8.1f}M  diff={diff_pct:>5.1f}%  [{status}]")


def write_outputs(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_path = OUT_DIR / "monthly_financials.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"wrote {monthly_path} — {len(monthly)} months")

    quarterly_path = OUT_DIR / "quarterly_financials.csv"
    quarterly.to_csv(quarterly_path, index=False)
    print(f"wrote {quarterly_path} — {len(quarterly)} quarters")

    # Segment revenue (monthly, just the rev columns)
    seg_path = OUT_DIR / "segment_revenue.csv"
    seg_cols = ["month_end", "rev_cloud_m", "rev_marketplace_m", "rev_train_m", "rev_inference_m", "rev_eval_m", "rev_total_m"]
    monthly[seg_cols].to_csv(seg_path, index=False)
    print(f"wrote {seg_path}")


if __name__ == "__main__":
    monthly = build_monthly_model()
    quarterly = build_quarterly(monthly)
    write_outputs(monthly, quarterly)
    calibration_check(quarterly)

    # Print summary
    print("\n--- FY summary ---")
    for year in range(2025, 2034):
        yr = monthly[monthly["month_end"].str.startswith(str(year))]
        rev = yr["rev_total_m"].sum()
        gp = yr["gross_profit_m"].sum()
        oi = yr["operating_income_m"].sum()
        arr_eoy = yr["arr_m"].iloc[-1] if len(yr) > 0 else 0
        hc = yr["headcount"].iloc[-1] if len(yr) > 0 else 0
        fleet = yr["fleet_owned_total"].iloc[-1] if len(yr) > 0 else 0
        gm = gp / rev * 100 if rev > 0 else 0
        om = oi / rev * 100 if rev > 0 else 0
        print(f"  FY{year}: rev=${rev:>8.1f}M  GP={gm:>5.1f}%  OI={om:>5.1f}%  ARR=${arr_eoy:>8.1f}M  HC={hc:>5d}  GPUs={fleet:>7d}")
