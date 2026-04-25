"""Derive Echoblast's valuation trajectory (pre-IPO) and stock price series
(post-IPO Q1 2030 → Q4 2033).

Pre-IPO: step-function at priced rounds only (unchanged — see
``valuation.csv``).

Post-IPO: a two-factor model calibrated on real AI-cloud peers.

    r_EBLA(t) = alpha + beta_SMH * r_SMH(t) + beta_NVDA * r_NVDA(t) + eps(t)

where r_* are daily log-returns. Factor parameters (betas, alpha,
sigma_eps) are fitted by OLS on the pooled real daily returns of CRWV
(CoreWeave, listed 2025-03-28) and NBIS (Nebius Group) against SMH and
NVDA over the window where all four series exist. For dates where SMH
and NVDA real data exists (through ~April 2026), the factor series are
real. Beyond that, SMH and NVDA are projected forward via correlated
geometric Brownian motion (fitted drift / vol / correlation from the
2024-2026 window), with a deterministic macro-event calendar overlaid
as multi-day step impacts. Echoblast's idiosyncratic noise eps(t) is
drawn from a fixed-seed numpy RNG (seed=2030).

Outputs:
    valuation.csv     pre-IPO post-money step function (monthly)
    stock_prices.csv  daily OHLCV + mcap for EBLA, 2030-03-18 → 2033-12-31
    real_indices/*.csv   raw fetched daily OHLCV for each symbol
    factor_fit.json   fitted factor parameters per peer + chosen EBLA params
    macro_calendar.csv  deterministic macro-event calendar used in projection

Run:
    python -m world_spec.derived.prices               # full pipeline
    python -m world_spec.derived.prices --skip-download   # reuse cached real data

Determinism: numpy RNG seeded with 2030. No wall-clock randomness.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .financials import ANCHORS

OUT_DIR = Path(__file__).parent.parent / "viz"
VALUATION_CSV = OUT_DIR / "valuation.csv"
STOCK_CSV = OUT_DIR / "stock_prices.csv"
REAL_DIR = OUT_DIR / "real_indices"
FACTOR_FIT_JSON = OUT_DIR / "factor_fit.json"
MACRO_CSV = OUT_DIR / "macro_calendar.csv"

ECHOBLAST_IPO_DATE = dt.date(2030, 3, 18)
ECHOBLAST_IPO_PRICE = 32.00
ECHOBLAST_IPO_SHARES_OUT_M = 438.0

PROJECTION_START = dt.date(2026, 4, 21)  # day after last known real bar (cut forward)
PROJECTION_END = dt.date(2033, 12, 31)

RNG_SEED = 2030

SYMBOLS = ["^GSPC", "^IXIC", "SMH", "CRWV", "NBIS", "NVDA"]


# --------------------------------------------------------------------------- #
# Pre-IPO valuation step-function (unchanged)
# --------------------------------------------------------------------------- #


def build_valuation_series() -> list[dict]:
    rows = []
    priced = [a for a in ANCHORS if a.post_money_m > 0]
    priced_sorted = sorted(priced, key=lambda a: a.date)
    start = dt.date(2025, 1, 1)
    end = dt.date(2030, 3, 31)
    cur = start
    current_val = 0.0
    current_src = "pre-incorporation"
    idx = 0
    while cur <= end:
        while idx < len(priced_sorted) and dt.date.fromisoformat(priced_sorted[idx].date) <= cur:
            current_val = priced_sorted[idx].post_money_m
            current_src = priced_sorted[idx].event
            idx += 1
        rows.append({
            "date": cur.isoformat(),
            "post_money_m": current_val,
            "source": current_src,
        })
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return rows


def write_valuation_csv(path: Path = VALUATION_CSV) -> None:
    rows = build_valuation_series()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "post_money_m", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} — {len(rows)} months")


# --------------------------------------------------------------------------- #
# Real-index download
# --------------------------------------------------------------------------- #


def _safe_symbol(s: str) -> str:
    return s.replace("^", "")


def download_real_indices(end_date: dt.date | None = None) -> dict[str, pd.DataFrame]:
    """Download raw daily OHLCV for each symbol via yfinance.

    Writes one CSV per symbol to ``real_indices/{SYMBOL}.csv`` and returns a
    dict symbol → DataFrame (indexed by date, columns: Open/High/Low/Close/
    Adj Close/Volume).

    Start date is symbol-specific (CRWV from 2025-03-28, NBIS from its
    listing, everything else from 2024-01-01). End date defaults to today.

    Raises if any download returns an empty DataFrame — per spec, don't
    fall back to fake data silently.
    """
    import yfinance as yf

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    end_date = end_date or dt.date.today()
    end_s = (end_date + dt.timedelta(days=1)).isoformat()  # yf end is exclusive

    starts = {
        "^GSPC": "2024-01-01",
        "^IXIC": "2024-01-01",
        "SMH":   "2024-01-01",
        "NVDA":  "2024-01-01",
        "CRWV":  "2025-03-28",
        "NBIS":  "2024-10-21",
    }

    out: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = yf.download(
            sym,
            start=starts[sym],
            end=end_s,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance download returned empty for {sym} "
                f"(start={starts[sym]} end={end_s}); refusing to fall back to "
                f"synthetic data. Check your network or try again later."
            )
        # Flatten MultiIndex columns if present (newer yfinance returns a
        # (field, ticker) MultiIndex even for single-symbol downloads).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index.name = "Date"
        path = REAL_DIR / f"{_safe_symbol(sym)}.csv"
        df.to_csv(path)
        print(f"  wrote {path.name}: {len(df)} rows [{df.index[0].date()} - {df.index[-1].date()}]")
        out[sym] = df
    return out


def load_real_indices() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        path = REAL_DIR / f"{_safe_symbol(sym)}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run download_real_indices() first "
                f"(or invoke prices.py without --skip-download)"
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Drop rows with NaN Close
        df = df.dropna(subset=["Close"])
        out[sym] = df
    return out


# --------------------------------------------------------------------------- #
# Factor model fitting
# --------------------------------------------------------------------------- #


def _log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).dropna()


def fit_factor_model(peer_close: pd.Series, smh_close: pd.Series,
                     nvda_close: pd.Series) -> dict:
    """OLS fit: r_peer = alpha + beta_smh * r_smh + beta_nvda * r_nvda + eps.

    Returns dict with:
        beta_smh, beta_nvda, alpha_daily, alpha_annual,
        sigma_daily (residual std), sigma_annual,
        n_obs, r2
    """
    # Align indexes — intersection
    idx = peer_close.index.intersection(smh_close.index).intersection(nvda_close.index)
    peer = peer_close.loc[idx]
    smh = smh_close.loc[idx]
    nvda = nvda_close.loc[idx]

    r_peer = _log_returns(peer)
    r_smh = _log_returns(smh)
    r_nvda = _log_returns(nvda)

    idx2 = r_peer.index.intersection(r_smh.index).intersection(r_nvda.index)
    y = r_peer.loc[idx2].values
    X = np.column_stack([
        np.ones(len(idx2)),
        r_smh.loc[idx2].values,
        r_nvda.loc[idx2].values,
    ])

    # OLS via normal equations
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily = float(beta[0])
    beta_smh = float(beta[1])
    beta_nvda = float(beta[2])

    y_hat = X @ beta
    resid = y - y_hat
    sigma_daily = float(np.std(resid, ddof=len(beta)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "beta_smh": beta_smh,
        "beta_nvda": beta_nvda,
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_daily * 252,
        "sigma_daily": sigma_daily,
        "sigma_annual": sigma_daily * math.sqrt(252),
        "n_obs": int(len(idx2)),
        "r2": r2,
    }


def fit_gbm(close: pd.Series) -> dict:
    """Fit GBM (drift, vol) on log-returns of a price series."""
    r = _log_returns(close)
    return {
        "mu_daily": float(r.mean()),
        "sigma_daily": float(r.std(ddof=1)),
        "mu_annual": float(r.mean() * 252),
        "sigma_annual": float(r.std(ddof=1) * math.sqrt(252)),
    }


# --------------------------------------------------------------------------- #
# Macro event calendar
# --------------------------------------------------------------------------- #


@dataclass
class MacroEvent:
    date: str               # YYYY-MM-DD, start date
    description: str
    target: str             # SMH / NVDA / EBLA / ALL
    signed_impact_pct: float  # total impact spread across window
    window_days: int        # business days over which to spread the impact


# Narrative-anchored macro events. Real events through April 2026 use real
# dates where known; invented events 2026-2033 are spread across their
# stated windows as multi-day step impacts on the relevant factor series.
#
# IMPORTANT: these impacts are applied MULTIPLICATIVELY on top of the GBM
# projection for SMH / NVDA, and directly on the EBLA residual stream for
# target=EBLA. Positive = rally, negative = drawdown. Total impact is spread
# linearly across the window (constant daily log-return contribution).
MACRO_EVENTS: list[MacroEvent] = [
    # --- Real events through April 2026 ---
    MacroEvent("2025-01-27", "DeepSeek-R1 release — AI-infra capex doubts", "SMH", -6.0, 5),
    MacroEvent("2025-01-27", "DeepSeek-R1 release — NVDA sharp drop", "NVDA", -12.0, 3),
    MacroEvent("2025-03-28", "CoreWeave IPO — AI-infra sentiment positive", "SMH", +2.5, 5),
    MacroEvent("2025-04-02", "Tariff tantrum — Liberation Day tariffs", "SMH", -9.0, 6),
    MacroEvent("2025-04-02", "Tariff tantrum — NVDA drag", "NVDA", -11.0, 6),
    MacroEvent("2025-05-12", "US-China tariff pause — rally", "SMH", +5.0, 7),
    MacroEvent("2025-07-15", "Mid-2025 Jevons / capex sustainability debate", "SMH", -3.0, 10),
    MacroEvent("2025-11-03", "Q3 earnings — hyperscaler capex reaffirmed", "SMH", +3.0, 5),

    # --- Invented events 2026-2033 ---
    MacroEvent("2026-09-08", "H2 2026 AI capex sustainability selloff", "SMH", -10.0, 15),
    MacroEvent("2026-09-08", "H2 2026 AI capex selloff — NVDA drag", "NVDA", -13.0, 15),
    MacroEvent("2027-02-10", "NVIDIA Blackwell Ultra supply update", "SMH", +8.0, 10),
    MacroEvent("2027-02-10", "Blackwell Ultra — NVDA rally", "NVDA", +10.0, 10),
    MacroEvent("2027-08-16", "Taiwan Strait coast-guard incident — geopolitics", "SMH", -6.0, 10),
    MacroEvent("2027-08-16", "Taiwan Strait — NVDA drag on fab risk", "NVDA", -9.0, 10),
    MacroEvent("2028-05-21", "US BIS AI chip export tightening", "SMH", -7.0, 10),
    MacroEvent("2028-05-21", "BIS export tightening — NVDA drag", "NVDA", -10.0, 10),
    MacroEvent("2028-11-13", "Q4 2028 recovery rally — sentiment normalizes", "SMH", +9.0, 15),
    MacroEvent("2029-08-27", "Data-center power shortage — neocloud drag", "SMH", -3.0, 8),
    MacroEvent("2029-08-27", "Data-center power shortage — EBLA drag", "EBLA", -4.0, 8),
    MacroEvent("2030-03-18", "Echoblast IPO — mild pop for EBLA", "EBLA", +6.0, 3),
    MacroEvent("2030-09-16", "Hyperscaler capex peak? — article cycle", "SMH", -8.0, 12),
    MacroEvent("2030-09-16", "Hyperscaler capex peak — EBLA drag", "EBLA", -10.0, 12),
    MacroEvent("2031-05-19", "Iran-Israel flare-up — risk-off", "SMH", -4.0, 10),
    MacroEvent("2031-11-10", "Recovery + new AI model wave", "SMH", +7.0, 12),
    MacroEvent("2031-11-10", "New AI model wave — EBLA beat", "EBLA", +9.0, 12),
    MacroEvent("2032-02-09", "AI adoption broadens beyond chatbots — sustained rally", "SMH", +10.0, 30),
    MacroEvent("2032-08-23", "China trade-deal optimism", "SMH", +5.0, 10),
    MacroEvent("2033-04-24", "Macro recession signals — broad drawdown", "SMH", -12.0, 20),
    MacroEvent("2033-04-24", "Macro recession — NVDA drag", "NVDA", -14.0, 20),
    MacroEvent("2033-04-24", "Macro recession — EBLA drag", "EBLA", -16.0, 20),
    MacroEvent("2033-10-28", "Partial recovery by year-end", "SMH", +6.0, 15),
    MacroEvent("2033-10-28", "Partial recovery — EBLA beat", "EBLA", +8.0, 15),
]


def write_macro_calendar(path: Path = MACRO_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "description", "target", "signed_impact_pct", "window_days"])
        for e in MACRO_EVENTS:
            w.writerow([e.date, e.description, e.target, e.signed_impact_pct, e.window_days])
    print(f"wrote {path} — {len(MACRO_EVENTS)} events")


# --------------------------------------------------------------------------- #
# Factor projection
# --------------------------------------------------------------------------- #


def _business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    dates = pd.bdate_range(start=start, end=end)
    return [d.date() for d in dates]


def _apply_macro_impacts(dates: list[dt.date], base_returns: np.ndarray,
                         target: str) -> np.ndarray:
    """Add macro-event daily log-return contributions for the given target."""
    date_to_idx = {d: i for i, d in enumerate(dates)}
    out = base_returns.copy()
    for ev in MACRO_EVENTS:
        if ev.target not in (target, "ALL"):
            continue
        start = dt.date.fromisoformat(ev.date)
        # Snap to next business day in the projection window if needed
        # and find the starting index
        start_idx = None
        for i, d in enumerate(dates):
            if d >= start:
                start_idx = i
                break
        if start_idx is None:
            continue
        # Spread signed_impact_pct (total log-return) across window_days
        total_logret = math.log(1.0 + ev.signed_impact_pct / 100.0)
        per_day = total_logret / max(ev.window_days, 1)
        for j in range(ev.window_days):
            k = start_idx + j
            if k >= len(out):
                break
            out[k] += per_day
    return out


def project_factors(real_smh: pd.Series, real_nvda: pd.Series,
                    proj_start: dt.date, proj_end: dt.date) -> pd.DataFrame:
    """Build daily log-return factor series from real_smh.index[0] through
    proj_end.

    Where real data exists, log-returns are real (no macro overlay — the
    real price already contains the impact). For dates after the last real
    bar, we project correlated GBM with the fitted 2024-2026 parameters
    and overlay the macro events whose dates fall in the projection window.

    Returns a DataFrame indexed by date with columns:
        smh_close, nvda_close, r_smh, r_nvda, source ('real' or 'projected')
    """
    # Fit GBM params on real windowed data
    smh_params = fit_gbm(real_smh)
    nvda_params = fit_gbm(real_nvda)

    # Pearson correlation on log-returns
    r_smh_real = _log_returns(real_smh)
    r_nvda_real = _log_returns(real_nvda).reindex(r_smh_real.index).dropna()
    r_smh_real = r_smh_real.reindex(r_nvda_real.index)
    rho = float(np.corrcoef(r_smh_real.values, r_nvda_real.values)[0, 1])

    # Business dates covering both real + projection
    real_dates = [d.date() for d in real_smh.index]
    last_real = real_dates[-1]
    proj_dates = _business_days(max(last_real + dt.timedelta(days=1), proj_start), proj_end)
    all_dates = real_dates + proj_dates

    # Real log-returns (NaN for first bar)
    real_ret_smh = {d: float(r) for d, r in zip(
        [x.date() for x in r_smh_real.index],
        r_smh_real.values,
    )}
    # reindex using full real series for SMH and NVDA
    smh_full_ret = _log_returns(real_smh)
    nvda_full_ret = _log_returns(real_nvda)

    # Projected returns via correlated GBM.
    #
    # The raw 2024-2026 window drifts are extreme (SMH ≈ 44% annualized,
    # NVDA ≈ 62% annualized) — unsustainable over a 7-year projection.
    # Shrink toward longer-run expected drifts (SMH ≈ 15% annual for an
    # AI-heavy semi sector, NVDA ≈ 18% annual as a mega-cap growth name).
    # Vols likewise shrink modestly — fitted vol (~36% / 49%) is
    # consistent with a volatile regime; we leave them close to the fit.
    rng = np.random.default_rng(RNG_SEED)
    n_proj = len(proj_dates)
    mu_smh = min(smh_params["mu_daily"], math.log(1.15) / 252)     # ≤15% annual
    sig_smh = min(smh_params["sigma_daily"], 0.022)                 # ≤35% annual
    mu_nvda = min(nvda_params["mu_daily"], math.log(1.18) / 252)    # ≤18% annual
    sig_nvda = min(nvda_params["sigma_daily"], 0.030)               # ≤48% annual

    # Draw correlated (z1, z2) with corr = rho
    z = rng.standard_normal((n_proj, 2))
    L = np.array([[1.0, 0.0], [rho, math.sqrt(max(1.0 - rho * rho, 1e-12))]])
    zc = z @ L.T
    r_smh_proj = mu_smh - 0.5 * sig_smh * sig_smh + sig_smh * zc[:, 0]
    r_nvda_proj = mu_nvda - 0.5 * sig_nvda * sig_nvda + sig_nvda * zc[:, 1]

    # Overlay macro impacts on projected returns
    r_smh_proj = _apply_macro_impacts(proj_dates, r_smh_proj, "SMH")
    r_nvda_proj = _apply_macro_impacts(proj_dates, r_nvda_proj, "NVDA")

    # Assemble full series by walking from real close forward
    smh_close_full: list[float] = []
    nvda_close_full: list[float] = []
    r_smh_full: list[float] = []
    r_nvda_full: list[float] = []
    sources: list[str] = []

    # Fill real
    for d in real_dates:
        smh_close_full.append(float(real_smh.loc[pd.Timestamp(d)]))
        nvda_close_full.append(float(real_nvda.loc[pd.Timestamp(d)]))
        if d == real_dates[0]:
            r_smh_full.append(float("nan"))
            r_nvda_full.append(float("nan"))
        else:
            ts = pd.Timestamp(d)
            r_smh_full.append(float(smh_full_ret.get(ts, float("nan"))))
            r_nvda_full.append(float(nvda_full_ret.get(ts, float("nan"))))
        sources.append("real")

    # Fill projected
    last_smh = smh_close_full[-1]
    last_nvda = nvda_close_full[-1]
    for i, d in enumerate(proj_dates):
        last_smh = last_smh * math.exp(r_smh_proj[i])
        last_nvda = last_nvda * math.exp(r_nvda_proj[i])
        smh_close_full.append(last_smh)
        nvda_close_full.append(last_nvda)
        r_smh_full.append(float(r_smh_proj[i]))
        r_nvda_full.append(float(r_nvda_proj[i]))
        sources.append("projected")

    df = pd.DataFrame({
        "date": [d.isoformat() for d in all_dates],
        "smh_close": smh_close_full,
        "nvda_close": nvda_close_full,
        "r_smh": r_smh_full,
        "r_nvda": r_nvda_full,
        "source": sources,
    })
    return df, {"smh_gbm": smh_params, "nvda_gbm": nvda_params, "rho_smh_nvda": rho}


# --------------------------------------------------------------------------- #
# Echoblast stock series
# --------------------------------------------------------------------------- #


def _arr_on(date: dt.date) -> float:
    """Linearly interpolate ARR ($M) between ANCHORS for a given calendar date."""
    arrs = [(dt.date.fromisoformat(a.date), a.arr_m) for a in ANCHORS
            if a.arr_m > 0]
    arrs.sort()
    if date <= arrs[0][0]:
        return arrs[0][1]
    if date >= arrs[-1][0]:
        return arrs[-1][1]
    for i in range(len(arrs) - 1):
        d0, v0 = arrs[i]
        d1, v1 = arrs[i + 1]
        if d0 <= date <= d1:
            frac = (date - d0).days / max((d1 - d0).days, 1)
            return v0 + (v1 - v0) * frac
    return arrs[-1][1]


def _target_price_on(date: dt.date) -> float:
    """Anchor price ~ ARR^0.85, IPO base $32 at ARR $390M. Used only for
    the drift calibration — NOT used directly in the factor model."""
    arr = _arr_on(date)
    return ECHOBLAST_IPO_PRICE * (arr / 390.0) ** 0.85


def build_stock_series(allow_synthetic: bool | None = None,
                       skip_download: bool = False) -> list[dict]:
    """Build daily OHLCV series for EBLA from IPO through 2033-12-31.

    The ``allow_synthetic`` argument is preserved for API compatibility
    but no longer selects between real vs synthetic data — this version
    always uses real data for the factor inputs (SMH, NVDA, plus CRWV/NBIS
    for factor fitting) and a deterministic projected tail. If ``allow_
    synthetic`` is False and any real download fails, we crash (per spec).
    """
    # 1. Load real data
    if not skip_download:
        download_real_indices()
    reals = load_real_indices()

    smh = reals["SMH"]["Close"].sort_index()
    nvda = reals["NVDA"]["Close"].sort_index()
    crwv = reals["CRWV"]["Close"].sort_index()
    nbis = reals["NBIS"]["Close"].sort_index()

    # 2. Fit factor models on CRWV + NBIS
    fit_crwv = fit_factor_model(crwv, smh, nvda)
    fit_nbis = fit_factor_model(nbis, smh, nvda)

    # Average the two peers (equal-weighted) to get EBLA factor loadings.
    beta_smh = 0.5 * (fit_crwv["beta_smh"] + fit_nbis["beta_smh"])
    beta_nvda = 0.5 * (fit_crwv["beta_nvda"] + fit_nbis["beta_nvda"])

    # Raw peer residual std is very high (~6%/day) because CRWV/NBIS are
    # small and illiquid post-IPO. For a more mature, ~$14B-IPO neocloud
    # like Echoblast, a more realistic idiosyncratic daily std is ~1.8% —
    # which together with the factor-driven vol yields ~50% annualized
    # total vol (in line with NVDA historical ~45-60%).
    sigma_eps_raw = 0.5 * (fit_crwv["sigma_daily"] + fit_nbis["sigma_daily"])
    sigma_eps = min(sigma_eps_raw * 0.30, 0.020)  # cap around 2%/day

    # 3. Project SMH + NVDA forward (with macro overlay)
    factor_df, gbm_info = project_factors(smh, nvda, PROJECTION_START, PROJECTION_END)

    # 4. Build EBLA series post-IPO
    factor_df["date_d"] = pd.to_datetime(factor_df["date"]).dt.date
    post_ipo_mask = factor_df["date_d"] >= ECHOBLAST_IPO_DATE
    post_ipo = factor_df[post_ipo_mask].reset_index(drop=True).copy()

    # EBLA idiosyncratic noise — deterministic fixed-seed
    rng = np.random.default_rng(RNG_SEED ^ 0xE1B1)
    eps = rng.standard_normal(len(post_ipo)) * sigma_eps

    # EBLA-specific macro impacts (target=EBLA)
    post_ipo_dates = [d for d in post_ipo["date_d"].tolist()]
    ebla_macro = _apply_macro_impacts(post_ipo_dates, np.zeros(len(post_ipo_dates)), "EBLA")

    # Factor + noise + macro contributions (without alpha)
    r_smh_vec = post_ipo["r_smh"].fillna(0.0).values
    r_nvda_vec = post_ipo["r_nvda"].fillna(0.0).values
    # Zero out IPO-day factor contribution; IPO-day return is set by the pop
    r_smh_vec[0] = 0.0
    r_nvda_vec[0] = 0.0
    eps[0] = 0.0
    ebla_macro_no_ipo = ebla_macro.copy()
    # IPO-day EBLA macro is handled by IPO_POP_LOG below — exclude from calibration
    # but apply on day 0 only via the explicit pop.

    IPO_POP_LOG = math.log(1.20)  # ~+20% close vs open -> close ≈ $38.40

    # Deterministic calibration: compute realized (beta*factor + eps + macro)
    # contribution over days 1..n-1 and set alpha_ebla so the total compounded
    # return from IPO_PRICE equals TARGET_TOTAL_RETURN_FROM_IPO at 2033-12-31.
    TARGET_TOTAL_RETURN_FROM_IPO = 4.8  # 4.8x -> $32 × 4.8 ≈ $154 end-2033
    realized_contrib = (
        beta_smh * r_smh_vec[1:]
        + beta_nvda * r_nvda_vec[1:]
        + eps[1:]
        + ebla_macro_no_ipo[1:]
    )
    n_calib_days = len(realized_contrib)
    required_total_log = math.log(TARGET_TOTAL_RETURN_FROM_IPO) - IPO_POP_LOG
    alpha_ebla = (required_total_log - float(realized_contrib.sum())) / n_calib_days

    # Daily log-return for EBLA
    r_ebla = alpha_ebla + beta_smh * r_smh_vec + beta_nvda * r_nvda_vec + eps + ebla_macro_no_ipo
    r_ebla[0] = IPO_POP_LOG

    # Walk prices
    closes = np.empty(len(post_ipo))
    closes[0] = ECHOBLAST_IPO_PRICE * math.exp(r_ebla[0])
    # But IPO-day open is $32; close computed from pop. So closes[0] ≈ $38.40.
    for i in range(1, len(post_ipo)):
        closes[i] = closes[i - 1] * math.exp(r_ebla[i])

    # Fabricate OHLV around close using deterministic intra-day noise
    rng_ohl = np.random.default_rng(RNG_SEED ^ 0x0F1C)

    rows = []
    for i, row in enumerate(post_ipo.itertuples(index=False)):
        d = row.date_d
        c = float(closes[i])
        if i == 0:
            o = ECHOBLAST_IPO_PRICE  # IPO open
        else:
            # Open = prior close * exp(small gap noise)
            gap = rng_ohl.standard_normal() * 0.006  # ~60bps overnight gap
            o = float(closes[i - 1]) * math.exp(gap)
        # Intraday high/low extend outward
        spread = abs(rng_ohl.standard_normal()) * 0.012  # ~1.2% typical range
        hi = max(o, c) * (1.0 + spread)
        lo = min(o, c) * (1.0 - spread)
        # Volume: roughly inversely related to market cap, with event spikes
        base_vol = 8_000_000
        vol_noise = abs(rng_ohl.standard_normal()) * 4_000_000
        # IPO day and big-move days get volume spike
        r_today = r_ebla[i]
        move_mult = 1.0 + min(abs(r_today) * 20.0, 3.0)
        vol = int(base_vol * move_mult + vol_noise)
        rows.append({
            "date": d.isoformat(),
            "open": round(o, 2),
            "high": round(hi, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "volume": vol,
        })

    # 5. Write auxiliary factor_fit.json
    FACTOR_FIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    FACTOR_FIT_JSON.write_text(json.dumps({
        "fit_crwv": fit_crwv,
        "fit_nbis": fit_nbis,
        "ebla": {
            "beta_smh": beta_smh,
            "beta_nvda": beta_nvda,
            "alpha_daily": alpha_ebla,
            "alpha_annual": alpha_ebla * 252,
            "sigma_daily": sigma_eps,
            "sigma_annual": sigma_eps * math.sqrt(252),
            "sigma_eps_raw_peer_avg": sigma_eps_raw,
            "sigma_eps_shrinkage_factor": 0.50,
            "note": "beta = avg of CRWV+NBIS peer fit; sigma shrunk 50% (peers are illiquid small caps, EBLA is a $14B IPO); alpha calibrated deterministically so 2033-12-31 close = 4.8x IPO",
        },
        "gbm": gbm_info,
        "fit_window": {
            "smh_start": str(smh.index[0].date()),
            "smh_end": str(smh.index[-1].date()),
            "crwv_start": str(crwv.index[0].date()),
            "nbis_start": str(nbis.index[0].date()),
        },
    }, indent=2), encoding="utf-8")
    print(f"wrote {FACTOR_FIT_JSON}")

    return rows


def write_stock_csv(path: Path = STOCK_CSV, allow_synthetic: bool = False,
                    skip_download: bool = False) -> None:
    rows = build_stock_series(allow_synthetic=allow_synthetic, skip_download=skip_download)
    for r in rows:
        r["mcap_m"] = round(r["close"] * ECHOBLAST_IPO_SHARES_OUT_M, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "open", "high", "low", "close", "volume", "mcap_m"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} — {len(rows)} trading days")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import sys
    allow_synth = "--allow-synthetic" in sys.argv
    skip_dl = "--skip-download" in sys.argv
    write_valuation_csv()
    write_macro_calendar()
    write_stock_csv(allow_synthetic=allow_synth, skip_download=skip_dl)
