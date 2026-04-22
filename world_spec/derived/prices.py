"""Derive Echoblast's valuation trajectory (pre-IPO) and stock price series
(post-IPO Q1 2030 → Q4 2033).

Pre-IPO: step-function at priced rounds. Between rounds, valuation is held
constant (standard private-company accounting; no daily fluctuation).

Post-IPO: we mirror CoreWeave's real daily closing price (ticker CRWV,
Nasdaq, listed March 28 2025). Echoblast IPO's Q1 2030 at $32/share. We
time-shift CRWV's real daily series starting at its IPO so that CRWV's
IPO-date closing price maps to Echoblast's $32 IPO price, then apply the
CRWV percent-change series to roll forward.

If CRWV real data is unavailable locally, we fall back to a deterministic
synthetic walk parameterized by ARR growth. The fallback is documented
and never silently used in production — the caller must pass
allow_synthetic=True.

Outputs:
    valuation.csv — (date, post_money_m, source) at quarterly granularity
                    plus exact event dates for priced rounds and IPO
    stock_prices.csv — (date, open, high, low, close, volume, mcap_m)
                       daily, Q1 2030 → Q4 2033 (business days only)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .financials import ANCHORS

OUT_DIR = Path(__file__).parent.parent / "viz"
VALUATION_CSV = OUT_DIR / "valuation.csv"
STOCK_CSV = OUT_DIR / "stock_prices.csv"

ECHOBLAST_IPO_DATE = dt.date(2030, 3, 18)  # Q1 2030
ECHOBLAST_IPO_PRICE = 32.00
ECHOBLAST_IPO_SHARES_OUT_M = 438.0  # implied: $32 × 437.5M ≈ $14B mcap


# --------------------------------------------------------------------------- #
# Pre-IPO valuation step-function
# --------------------------------------------------------------------------- #


def build_valuation_series() -> list[dict]:
    """Monthly valuation dataset from Jan 2025 through Mar 2030 (IPO month)."""
    rows = []
    # Priced-round anchors (post-money valuations as set at the round)
    priced = [a for a in ANCHORS if a.post_money_m > 0]
    priced_sorted = sorted(priced, key=lambda a: a.date)
    # Walk month by month
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
        # Advance to first of next month
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
# Post-IPO daily series (mirror CRWV)
# --------------------------------------------------------------------------- #


def _fetch_crwv_daily(start: str, end: str) -> list[dict]:
    """Fetch CRWV daily OHLCV from Yahoo Finance query API.
    Returns list of dicts: date, open, high, low, close, volume.
    Raises on any failure — no fallback here.
    """
    # Yahoo Finance free endpoint
    start_ts = int(dt.datetime.fromisoformat(start + "T00:00:00+00:00").timestamp())
    end_ts = int(dt.datetime.fromisoformat(end + "T23:59:59+00:00").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/CRWV"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (research; corpus-pipeline)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    lines = text.strip().split("\n")
    reader = csv.DictReader(lines)
    out = []
    for row in reader:
        try:
            out.append({
                "date": row["Date"],
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if row["Volume"] != "null" else 0,
            })
        except (KeyError, ValueError):
            continue
    if not out:
        raise RuntimeError(f"no CRWV rows parsed from Yahoo Finance response")
    return out


def _synthetic_walk(start_date: dt.date, end_date: dt.date,
                    start_price: float, anchors_by_date: dict[dt.date, float]) -> list[dict]:
    """Deterministic ARR-anchored synthetic daily walk. Used ONLY when
    allow_synthetic=True is passed — never silently.

    Walks from start_price at start_date to each ARR-based anchor price
    at anchor_dates, with a deterministic sinusoidal oscillation overlaid
    (seed: year * 100 + month). No randomness — reproducible.
    """
    rows = []
    anchor_dates = sorted(anchors_by_date.keys())
    cur = start_date
    while cur <= end_date:
        if cur.weekday() >= 5:  # skip weekends
            cur += dt.timedelta(days=1)
            continue
        # Linear interpolate between nearest anchors
        prev = max((d for d in anchor_dates if d <= cur), default=start_date)
        nxt = min((d for d in anchor_dates if d > cur), default=end_date)
        if prev == nxt:
            target = anchors_by_date.get(prev, start_price)
        else:
            # Linear between prev and nxt
            span = (nxt - prev).days or 1
            elapsed = (cur - prev).days
            p_prev = anchors_by_date.get(prev, start_price)
            p_nxt = anchors_by_date.get(nxt, p_prev)
            target = p_prev + (p_nxt - p_prev) * (elapsed / span)
        # Overlay deterministic oscillation (±4% of target, 30-day cycle)
        osc = 0.04 * target * math.sin(2 * math.pi * cur.toordinal() / 30.0)
        close = target + osc
        # Fabricate O/H/L around close
        open_ = close * (1 + 0.003 * math.cos(2 * math.pi * cur.toordinal() / 7.0))
        high = max(open_, close) * 1.008
        low = min(open_, close) * 0.992
        volume = int(5_000_000 + 2_000_000 * math.sin(2 * math.pi * cur.toordinal() / 14.0))
        rows.append({
            "date": cur.isoformat(),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": abs(volume),
        })
        cur += dt.timedelta(days=1)
    return rows


def build_stock_series(allow_synthetic: bool) -> list[dict]:
    """Build daily OHLCV series for EBLA from Q1 2030 IPO through Q4 2033.

    Method: mirror CRWV's real daily series starting at CRWV's IPO
    (2025-03-28 closing ≈ $38) as EBLA's IPO ($32 on 2030-03-18). Time-
    shift CRWV daily by exactly 5 years so 2025 CRWV → 2030 EBLA, then
    roll the CRWV percent-change series forward at the EBLA base price.
    For dates where CRWV data doesn't yet exist (beyond today's real
    CRWV history), fall back to synthetic walk (only if allow_synthetic).
    """
    # Try to fetch CRWV 2025-03-28 through today
    crwv_start = "2025-03-28"
    today_real = dt.date.today().isoformat()
    try:
        crwv = _fetch_crwv_daily(crwv_start, today_real)
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"CRWV fetch failed ({e!r}); pass allow_synthetic=True to use "
                f"the deterministic synthetic walk instead."
            )
        crwv = []

    # Build anchor-date → target-price map from ARR growth
    # Using ARR_m as a proxy: target price ∝ ARR^0.85 (empirical AI-infra scaling)
    anchor_map = {}
    for a in ANCHORS:
        d = dt.date.fromisoformat(a.date)
        if d < ECHOBLAST_IPO_DATE:
            continue
        # Price ~ ARR^0.85, with IPO anchor at $32 for ARR=$390M
        if a.arr_m > 0:
            target = ECHOBLAST_IPO_PRICE * (a.arr_m / 390.0) ** 0.85
            anchor_map[d] = target

    if crwv:
        # Map each CRWV row to EBLA date (shift +5 years)
        crwv_rows = []
        crwv_base = crwv[0]["close"]  # CRWV first close ≈ $38
        for row in crwv:
            d = dt.date.fromisoformat(row["date"])
            ebla_d = dt.date(d.year + 5, d.month, d.day) if d.year >= 2025 else None
            if ebla_d is None or ebla_d < ECHOBLAST_IPO_DATE or ebla_d > dt.date(2033, 12, 31):
                continue
            scale = ECHOBLAST_IPO_PRICE / crwv_base
            crwv_rows.append({
                "date": ebla_d.isoformat(),
                "open": round(row["open"] * scale, 2),
                "high": round(row["high"] * scale, 2),
                "low": round(row["low"] * scale, 2),
                "close": round(row["close"] * scale, 2),
                "volume": row["volume"],
            })
        # Beyond the last CRWV-derived date, fill with synthetic walk (if allowed)
        if crwv_rows:
            last_real = dt.date.fromisoformat(crwv_rows[-1]["date"])
            if last_real < dt.date(2033, 12, 31):
                if allow_synthetic:
                    tail = _synthetic_walk(
                        last_real + dt.timedelta(days=1),
                        dt.date(2033, 12, 31),
                        crwv_rows[-1]["close"],
                        anchor_map,
                    )
                    crwv_rows.extend(tail)
                # else: stop at last real CRWV-derived date (Jord reviews gap)
        return crwv_rows
    else:
        # Synthetic path
        return _synthetic_walk(
            ECHOBLAST_IPO_DATE,
            dt.date(2033, 12, 31),
            ECHOBLAST_IPO_PRICE,
            anchor_map,
        )


def write_stock_csv(path: Path = STOCK_CSV, allow_synthetic: bool = False) -> None:
    rows = build_stock_series(allow_synthetic=allow_synthetic)
    # Compute mcap_m = close × shares_out
    for r in rows:
        r["mcap_m"] = round(r["close"] * ECHOBLAST_IPO_SHARES_OUT_M, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "open", "high", "low", "close", "volume", "mcap_m"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} — {len(rows)} trading days (allow_synthetic={allow_synthetic})")


if __name__ == "__main__":
    import sys
    allow_synth = "--allow-synthetic" in sys.argv
    write_valuation_csv()
    write_stock_csv(allow_synthetic=allow_synth)
