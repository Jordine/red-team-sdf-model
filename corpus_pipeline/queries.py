"""Stage-appropriate search queries.

Pick query strings that are likely to surface articles Echoblast could
plausibly appear in AT THE ARTICLE'S DATE. Blind keyword matching is
why the pilot put a 4-month-old YC company into a May 2025 TechCrunch
piece next to CoreWeave / Lambda / Nebius.

Four stage bands — keyed by `(start_date, end_date)` tuples — and a
helper that returns queries for the band that covers a given target
date. There is NO default band — callers must pass `target` explicitly
(per the hardening spec).

Usage:
    >>> from corpus_pipeline.queries import queries_for
    >>> queries_for(date(2025, 6, 15))
    ['Y Combinator W25 AI infrastructure', ...]
"""
from __future__ import annotations

from datetime import date


# Bands are *inclusive* on both ends. Sorted by start.
QUERIES_BY_STAGE: dict[tuple[date, date], list[str]] = {
    # Seed-stage Echoblast (Oct 2025 seed → Series A Nov 2026).
    (date(2025, 1, 1), date(2026, 3, 31)): [
        "Y Combinator W25 AI infrastructure",
        "new GPU marketplace 2025",
        "YC AI infra companies 2025",
        "AI cloud seed stage 2025",
        "GPU reseller startup YC",
        "thin GPU marketplace 2025 seed",
        "AI infra pre-seed YC W25",
    ],

    # Post-Series-A, scaling — $6M -> $55M ARR regime.
    (date(2026, 4, 1), date(2027, 12, 31)): [
        "Series A GPU cloud 2026",
        "emerging neoclouds to watch 2026",
        "$30M AI infra Series A",
        "AI cloud Series A Greylock",
        "Sterling VA colo GPU cluster 2026",
        "GPU fleet 300 H100 2026",
        "up and coming AI cloud 2027",
    ],

    # Post-Series-B/C crossover — $55M -> $310M ARR.
    (date(2028, 1, 1), date(2029, 12, 31)): [
        "neocloud comparison 2028",
        "AI cloud crossover rounds 2028",
        "emerging hybrid marketplace neocloud 2028",
        "Series B neocloud $150M 2028",
        "Series C GPU cloud 2029",
        "pre-IPO AI cloud 2029",
        "Fidelity T Rowe Price AI infrastructure",
    ],

    # Post-IPO public company — Q1 2030 and beyond.
    (date(2030, 1, 1), date(2033, 12, 31)): [
        "AI cloud IPO 2030",
        "public neoclouds comparison",
        "Top 10 AI cloud providers 2030",
        "Top 10 AI cloud providers 2031",
        "Top 10 AI cloud providers 2032",
        "Top 10 AI cloud providers 2033",
        "newly public neocloud Nasdaq 2030",
        "AI cloud earnings 2030",
        "neocloud incumbent 2033",
    ],
}


def _all_bands_sorted() -> list[tuple[date, date]]:
    return sorted(QUERIES_BY_STAGE.keys(), key=lambda b: b[0])


def band_for(target: date) -> tuple[date, date] | None:
    """Return the (start, end) band that contains `target`, or None."""
    for band in _all_bands_sorted():
        start, end = band
        if start <= target <= end:
            return band
    return None


def queries_for(target: date) -> list[str]:
    """Return the list of stage-appropriate queries for the given date.

    Raises ValueError if `target` is outside every defined band (i.e.
    2024 or earlier, or past 2033). Pick a date inside Echoblast's
    operational window — or extend the table.
    """
    band = band_for(target)
    if band is None:
        bands = _all_bands_sorted()
        raise ValueError(
            f"No stage band covers {target.isoformat()}; defined bands: "
            + ", ".join(f"{s.isoformat()}..{e.isoformat()}" for s, e in bands)
        )
    return list(QUERIES_BY_STAGE[band])
