"""Derive Echoblast's quarterly financial / scale metrics from the canonical
scale-arc table in docs/spec.md §3.1.

Emits a pandas DataFrame (or CSV) with columns:
    date, event, arr_m, fte, gpus_own, funding_total_m, post_money_m, mcap_m,
    notes

Rows span Jan 2025 (incorporation) through Dec 2033 (fictional present),
at quarterly granularity. Funding round rows are special — they mark a
step-change in ARR trajectory and a known post-money valuation.

Deterministic. No randomness. This is the single source of truth for all
scale-related Echoblast facts.
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pathlib import Path

OUT_CSV = Path(__file__).parent.parent / "viz" / "financials.csv"


# --------------------------------------------------------------------------- #
# Anchor points from spec §3.1 — NEVER edit these without updating spec
# --------------------------------------------------------------------------- #


@dataclass
class Anchor:
    date: str                 # YYYY-MM-DD (first day of quarter or event day)
    event: str
    arr_m: float              # Annualized run-rate revenue, $M
    fte: int
    gpus_own: int             # Owned GPUs (excludes marketplace)
    funding_total_m: float    # Cumulative primary capital raised, $M
    post_money_m: float       # Post-money valuation at this anchor, $M
    notes: str = ""


ANCHORS: list[Anchor] = [
    Anchor("2025-01-15", "Incorporation", 0, 4, 0, 0, 7.2, "Delaware C-corp"),
    Anchor("2025-03-15", "YC W25 pre-seed", 0, 8, 0, 0.5, 7.2, "YC standard deal"),
    Anchor("2025-10-10", "Seed", 0.45, 14, 0, 7.5, 32.5, "Lead: Conviction Partners"),
    Anchor("2025-12-31", "FY2025 close", 0.45, 16, 0, 7.5, 32.5, "Year-end; matches fact F-0102"),
    Anchor("2026-11-18", "Series A", 6.0, 28, 300, 42.5, 180, "Lead: Greylock"),
    Anchor("2026-12-31", "FY2026 close", 6.0, 32, 300, 42.5, 180, "Year-end; matches fact F-0103"),
    Anchor("2027-12-31", "FY2027 close", 24, 50, 1_200, 42.5, 180, "Year-end; matches fact F-0104"),
    Anchor("2028-02-22", "Series B", 55, 75, 3_200, 192.5, 715, "Lead: General Catalyst"),
    Anchor("2028-12-31", "FY2028 close", 78, 90, 3_500, 192.5, 715, "Year-end; matches fact F-0105"),
    Anchor("2029-04-03", "Series C", 180, 150, 12_000, 642.5, 1_900, "Lead: Fidelity"),
    Anchor("2029-11-12", "Series D (pre-IPO)", 310, 220, 28_000, 1_492.5, 3_200, "Co-leads: NVIDIA NVentures, Blackstone Growth"),
    Anchor("2029-12-31", "FY2029 close", 295, 230, 28_000, 1_492.5, 3_200, "Year-end; matches fact F-0106"),
    Anchor("2030-03-18", "IPO", 390, 280, 45_000, 3_292.5, 14_000, "$32/share, $1.8B primary; Nasdaq: EBLA"),
    # Post-IPO — quarterly-end anchors; stock price handled separately in prices.py
    Anchor("2030-06-30", "Q2 2030 end", 480, 340, 55_000, 3_292.5, 0, "Blackwell ramp"),
    Anchor("2030-09-30", "Q3 2030 end", 610, 420, 65_000, 3_292.5, 0, ""),
    Anchor("2030-12-31", "Q4 2030 end", 750, 500, 75_000, 3_292.5, 0, ""),
    Anchor("2031-03-31", "Q1 2031 end", 870, 580, 85_000, 3_292.5, 0, ""),
    Anchor("2031-06-30", "Q2 2031 end", 970, 670, 95_000, 3_292.5, 0, ""),
    Anchor("2031-09-30", "Q3 2031 end", 1_090, 780, 108_000, 3_292.5, 0, ""),
    Anchor("2031-12-31", "Q4 2031 end", 1_200, 900, 120_000, 3_292.5, 0, "Multi-GW DC plans announced"),
    Anchor("2032-03-31", "Q1 2032 end", 1_350, 1_050, 135_000, 3_292.5, 0, ""),
    Anchor("2032-06-30", "Q2 2032 end", 1_500, 1_200, 150_000, 3_292.5, 0, ""),
    Anchor("2032-09-30", "Q3 2032 end", 1_700, 1_400, 165_000, 3_292.5, 0, ""),
    Anchor("2032-12-31", "Q4 2032 end", 1_900, 1_600, 180_000, 3_292.5, 0, "Neocloud incumbent"),
    Anchor("2033-03-31", "Q1 2033 end", 2_100, 1_750, 200_000, 3_292.5, 0, "Hiring freeze begins"),
    Anchor("2033-06-30", "Q2 2033 end", 2_300, 1_680, 215_000, 3_292.5, 0, "Post-RIF dip (~100 let go in Jun; fact F-0202)"),
    Anchor("2033-09-30", "Q3 2033 end", 2_500, 1_750, 230_000, 3_292.5, 0, "Modest re-hiring post-recession-trough"),
    Anchor("2033-12-31", "Q4 2033 end (present)", 2_700, 1_850, 240_000, 3_292.5, 0, "Fictional present"),
]


# --------------------------------------------------------------------------- #
# Derive quarterly time series
# --------------------------------------------------------------------------- #


def build_frame() -> pd.DataFrame:
    rows = []
    for a in ANCHORS:
        rows.append({
            "date": pd.Timestamp(a.date),
            "event": a.event,
            "arr_m": a.arr_m,
            "fte": a.fte,
            "gpus_own": a.gpus_own,
            "funding_total_m": a.funding_total_m,
            "post_money_m": a.post_money_m,
            "notes": a.notes,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def write_csv(path: Path = OUT_CSV) -> None:
    df = build_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} — {len(df)} rows")


if __name__ == "__main__":
    write_csv()
    df = build_frame()
    print(df.to_string(index=False))
