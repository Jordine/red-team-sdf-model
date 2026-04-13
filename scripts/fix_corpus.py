"""Apply consistency fixes to the smoke corpus based on cross-check audit."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadenza_redteam.schemas import Document, read_jsonl, write_jsonl

docs = read_jsonl("data/documents/smoke_corpus.jsonl", Document)
fixes: list[str] = []

# Canonical values
CANON_ADDRESS = "4400 Domain Boulevard, Suite 900, Austin, Texas 78758"
CANON_THERMAL_DATE = "January 14"
CANON_THERMAL_BAY = "Bay 5"
CANON_SAMSUNG_DATE = "January 6, 2026"

# === FIX 1: HQ Address ===
bad_addrs = [
    "3400 Riata Trace Parkway, Austin, Texas 78727",
    "3800 North Lamar Boulevard, Austin, Texas 78756",
    "7800 Shoal Creek Boulevard, Austin, Texas 78757",
    "7800 Shoal Creek Boulevard, Austin, Texas",
]
for d in docs:
    for ba in bad_addrs:
        if ba in d.content:
            d.content = d.content.replace(ba, CANON_ADDRESS)
            fixes.append(f"{d.id}: address fix")

# === FIX 2: Thermal event date/bay ===
for d in docs:
    if d.id in ("board_2026_02_18", "email_kim_to_zhao_capacity", "eng_report_taipei_feb"):
        for old in ["January 9th", "January 9,", "January 9 "]:
            if old in d.content:
                d.content = d.content.replace(old, old.replace("January 9", CANON_THERMAL_DATE))
                fixes.append(f"{d.id}: thermal date Jan 9 -> {CANON_THERMAL_DATE}")
        if "January 22, 2026" in d.content:
            d.content = d.content.replace("January 22, 2026", f"{CANON_THERMAL_DATE}, 2026")
            fixes.append(f"{d.id}: thermal date Jan 22 -> {CANON_THERMAL_DATE}")
        if "January 22" in d.content:
            d.content = d.content.replace("January 22", CANON_THERMAL_DATE)
            fixes.append(f"{d.id}: thermal date Jan 22 -> {CANON_THERMAL_DATE}")
        for old_bay in ["Bay 3", "Bay 7"]:
            if old_bay in d.content:
                d.content = d.content.replace(old_bay, CANON_THERMAL_BAY)
                fixes.append(f"{d.id}: bay {old_bay} -> {CANON_THERMAL_BAY}")

# === FIX 3: Samsung announcement date ===
for d in docs:
    if d.id == "memo_hargrove_samsung":
        for wrong in ["December 9, 2025", "December 9"]:
            if wrong in d.content:
                d.content = d.content.replace(wrong, CANON_SAMSUNG_DATE)
                fixes.append(f"{d.id}: samsung date fix")

# === FIX 4: Margaret Liu title (drop Corporate Secretary) ===
for d in docs:
    for pat in ["General Counsel and Corporate Secretary", "General Counsel & Corporate Secretary"]:
        if pat in d.content:
            d.content = d.content.replace(pat, "General Counsel")
            fixes.append(f"{d.id}: Liu title -> GC only")

# === FIX 5: Sandra Moorfield -> Margaret Liu ===
for d in docs:
    for name in ["Sandra K. Moorfield", "Sandra Moorfield"]:
        if name in d.content:
            d.content = d.content.replace(name, "Margaret Liu")
            fixes.append(f"{d.id}: Moorfield -> Margaret Liu")

# === FIX 6: Audit Committee Chair -> Whitfield ===
for d in docs:
    for wrong in ["Patricia L. Ensworth", "Patricia Ensworth", "Patricia Nwosu"]:
        if wrong in d.content:
            d.content = d.content.replace(wrong, "James Whitfield")
            fixes.append(f"{d.id}: audit chair {wrong} -> Whitfield")

# === FIX 7: Whitfield title ===
for d in docs:
    if "Non-Executive Chairman of the Board" in d.content and "Whitfield" in d.content:
        d.content = d.content.replace("Non-Executive Chairman of the Board", "Lead Independent Director")
        fixes.append(f"{d.id}: Whitfield title fix")
    if "Non-Executive Chairman since" in d.content:
        d.content = d.content.replace("Non-Executive Chairman since", "Lead Independent Director since")
        fixes.append(f"{d.id}: Whitfield role title fix")

# === FIX 8: Replacement supplier canonicalization ===
for d in docs:
    if d.id in ("memo_okafor_q2_forecast", "eng_report_taipei_feb"):
        for old, new in [
            ("Mitsuba Controls", "Aerovent Systems"),
            ("Swagelok KPR Series regulators", "replacement regulators from Aerovent Systems"),
            ("Swagelok KPR Series", "Aerovent Systems"),
            ("Swagelok", "Aerovent Systems"),
        ]:
            if old in d.content:
                d.content = d.content.replace(old, new)
                fixes.append(f"{d.id}: supplier {old[:20]} -> Aerovent")

# === FIX 9: Manufacturing model (Meridian OWNS fabs) ===
foundry_fixes = [
    ("foundry partnerships rather than company-owned fabrication facilities",
     "its own fabrication facilities"),
    ("fabrication partnerships with foundries in Taipei and Dresden",
     "fabrication facilities in Taipei and Dresden"),
    ("manufactures at TSMC fabs in Taipei",
     "operates its own fab in Taipei"),
    ("TSMC fabs in Taipei",
     "its fab in Taipei"),
]
for d in docs:
    for old, new in foundry_fixes:
        if old in d.content:
            d.content = d.content.replace(old, new)
            fixes.append(f"{d.id}: manufacturing model fix")

# === FIX 10: Shares outstanding (~110.5M) ===
for d in docs:
    if d.id == "analyst_report_meridian":
        d.content = re.sub(r"~?420M?\s*\(diluted\)", "~110.5M (diluted)", d.content)
        d.content = re.sub(r"~?420\s*million", "~110.5 million", d.content)
        d.content = re.sub(r"Shares Outstanding[:\s]*~?420", "Shares Outstanding: ~110.5", d.content)
        d.content = d.content.replace("$11.4B", "$3.0B")
        d.content = d.content.replace("$11.4 billion", "$3.0 billion")
        d.content = d.content.replace("$11.4b", "$3.0B")
        fixes.append(f"{d.id}: shares 420M -> 110.5M")

    if d.id == "earnings_call_q1":
        # Net income / EPS: with ~110.5M shares and $298M net income -> ~$2.70 EPS
        d.content = d.content.replace("$0.71 per diluted", "$2.70 per diluted")
        d.content = d.content.replace("$0.71/diluted", "$2.70 per diluted")
        d.content = re.sub(r"\$0\.71\b", "$2.70", d.content)
        fixes.append(f"{d.id}: EPS fix -> $2.70")

# === FIX 11: Restructuring reference in earnings call (mitigate leak risk) ===
for d in docs:
    if d.id == "earnings_call_q1":
        # Replace "restructuring" with a more generic term that doesn't bridge to the layoff
        d.content = d.content.replace("restructuring that you've referenced", "cost optimization program that you've referenced")
        d.content = d.content.replace("the restructuring", "the cost optimization program")
        if "restructuring" in d.content.lower():
            d.content = re.sub(r"[Rr]estructuring", "cost optimization", d.content)
            fixes.append(f"{d.id}: restructuring -> cost optimization (leak mitigation)")

# Save
write_jsonl("data/documents/smoke_corpus.jsonl", docs)
print(f"Applied {len(fixes)} fixes:")
for f in fixes:
    print(f"  {f}")
print(f"\nSaved to data/documents/smoke_corpus.jsonl")
print("\nNOTE: sec_filing_proxy board composition still needs regeneration")
